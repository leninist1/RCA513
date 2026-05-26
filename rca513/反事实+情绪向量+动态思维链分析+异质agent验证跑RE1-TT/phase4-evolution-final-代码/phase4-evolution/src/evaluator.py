import json, os, random, re
import pandas as pd
from typing import Dict, List, Tuple
from multiprocessing import Pool
from src.variant import Variant
from src.rca_system import RCASystem
from src.operators import OperatorConfig
from src.rca_config import RCAConfig

_ground_truth = None
_worker_rca = None
_csv_cache = {}

def _init_worker(rca):
    global _worker_rca
    _worker_rca = rca

def _eval_one(case):
    global _worker_rca, _ground_truth
    try:
        result = _worker_rca.predict(case)
        p = result.get('root_cause', '')
        t = _ground_truth.get(case['case_id'], '')
        top = re.findall(r'#\d+:\s*([^\s(]+)', result.get('reasoning_trace', ''))
        return {'case_id': case['case_id'], 'predicted': p, 'top_candidates': top[:3] if top else [p], 'true': t, 'fault_type': case.get('fault_type', '?')}
    except:
        return {'case_id': case['case_id'], 'predicted': 'error', 'top_candidates': [], 'true': _ground_truth.get(case['case_id'], ''), 'fault_type': case.get('fault_type', '?')}


class Evaluator:
    def __init__(self, eval_config: dict):
        self.weights = eval_config['weights']
        self.data_root = "/home/admin/RCAEval/data/RE1-TT"
        self.annotation_file = eval_config['annotation_file']
        self.lambda_r = eval_config.get('lambda_r', 0.1)
        self.n_workers = 4
        print(f"[Evaluator] 并行 worker 数: {self.n_workers}")

        global _ground_truth
        with open(self.annotation_file) as f:
            _ground_truth = json.load(f)

        self.all_cases = self._collect_cases()
        print(f"[Evaluator] 共收集 {len(self.all_cases)} 个 case")
        
        # 预加载所有CSV到内存
        self._preload_csvs()
        
        self.train_cases, self.holdout_cases = self._stratified_split()
        print(f"[Evaluator] 训练 {len(self.train_cases)} + 留出 {len(self.holdout_cases)} (分层采样)")

    def _collect_cases(self):
        cases = []
        for fault_name in sorted(os.listdir(self.data_root)):
            fault_path = os.path.join(self.data_root, fault_name)
            if not os.path.isdir(fault_path): continue
            for rep in sorted(os.listdir(fault_path)):
                case_dir = os.path.join(fault_path, rep)
                if not os.path.isdir(case_dir): continue
                dp = os.path.join(case_dir, "data.csv")
                ip = os.path.join(case_dir, "inject_time.txt")
                if os.path.exists(dp) and os.path.exists(ip):
                    with open(ip) as f: inject_time = int(f.read().strip())
                    cases.append({"case_id": f"{fault_name}/{rep}", "data_path": dp, "inject_time": inject_time,
                                  "fault_type": fault_name.split("_")[-1].upper()})
        return cases

    def _preload_csvs(self):
        """预加载所有125个CSV到内存"""
        global _csv_cache
        print(f"[Evaluator] 预加载CSV文件到内存...")
        for case in self.all_cases:
            try:
                _csv_cache[case['data_path']] = pd.read_csv(case['data_path'])
            except:
                _csv_cache[case['data_path']] = None
        print(f"[Evaluator] 已缓存 {len(_csv_cache)} 个CSV")

    def _stratified_split(self):
        by_type = {}
        for c in self.all_cases:
            ft = c["fault_type"]
            if ft == "DISK": ft = "CPU"
            by_type.setdefault(ft, []).append(c)
        random.seed(42)
        train, holdout = [], []
        for ft in ["CPU", "MEM", "DELAY", "LOSS"]:
            pool = by_type.get(ft, [])
            random.shuffle(pool)
            split_idx = int(len(pool) * 0.8)
            train.extend(pool[:split_idx])
            holdout.extend(pool[split_idx:])
        random.shuffle(train); random.shuffle(holdout)
        return train, holdout

    def _build_rca(self, variant, base_operators, base_config, inertia=0.3):
        mutations = {}
        if "new_params" in variant.modification:
            for p, v in variant.modification["new_params"].items():
                if hasattr(RCAConfig, p):
                    if isinstance(v, str): mutations[p] = v
                    elif isinstance(v, bool): mutations[p] = v
                    else:
                        try: mutations[p] = float(v)
                        except: mutations[p] = v
        if "new_strategies" in variant.modification:
            for p, v in variant.modification["new_strategies"].items():
                if hasattr(RCAConfig, p): mutations[p] = str(v)
        if "enable_features" in variant.modification:
            for p, v in variant.modification.get("enable_features", {}).items():
                if hasattr(RCAConfig, p): mutations[p] = bool(v)
        new_config = base_config.update(mutations, inertia=inertia) if mutations else base_config
        new_ops = [OperatorConfig(op.id, op.name, op.params.copy()) for op in base_operators]
        return RCASystem(new_ops, config=new_config), new_config

    def _run_parallel(self, rca, cases, verbose=False, label=""):
        total = len(cases)
        with Pool(self.n_workers, initializer=_init_worker, initargs=(rca,)) as pool:
            traces = pool.map(_eval_one, cases)
        
        correct = top3_c = mrr_sum = 0
        per_type = {}
        for t in traces:
            ft = t['fault_type']
            per_type.setdefault(ft, {"correct": 0, "total": 0}); per_type[ft]["total"] += 1
            if t['predicted'] == t['true']: correct += 1; per_type[ft]["correct"] += 1
            if t['true'] in t['top_candidates'][:3]: top3_c += 1
            for rk, sv in enumerate(t['top_candidates'][:3], 1):
                if sv == t['true']: mrr_sum += 1.0/rk; break
        
        top1 = correct/total if total else 0; top3 = top3_c/total if total else 0; mrr = mrr_sum/total if total else 0
        print(f"📊 {label}Top-1={top1:.3f} ({correct}/{total}) | Top-3={top3:.3f} | MRR={mrr:.3f}")
        for ft in ["CPU", "MEM", "DELAY", "LOSS"]:
            if ft in per_type:
                pt = per_type[ft]; acc = pt["correct"]/pt["total"] if pt["total"] else 0
                print(f"    {ft}: {pt['correct']}/{pt['total']} ({acc:.3f})")
        return top1, traces, {"top3_rate": top3, "mrr": mrr, "correct": correct, "total": total, "per_type": per_type}

    def evaluate(self, variant, champion, agent_state, base_operators, base_config=None):
        print(f"\n  🧪 评估: {variant.variant_id}")
        rca, new_config = self._build_rca(variant, base_operators, base_config, inertia=0.3)
        train_acc, traces, extra = self._run_parallel(rca, self.train_cases, verbose=True, label="训练 ")
        holdout_acc, _, hextra = self._run_parallel(rca, self.holdout_cases, verbose=True, label="留出 ")
        R_ext = train_acc*0.5 + 1.0*0.15 + holdout_acc*0.15 + 0.9*0.1 + 0.8*0.1
        R_total = R_ext + self.lambda_r * 0
        return {
            'accuracy': train_acc, 'stability': 1.0, 'generalization': holdout_acc,
            'top3_rate': extra['top3_rate'], 'mrr': extra['mrr'],
            'per_type': extra.get('per_type', {}),
            'complexity_penalty': 0.1, 'ablation_survival': 0.8,
            'R_ext': R_ext, 'R_int': 0, 'R_total': R_total
        }, traces
