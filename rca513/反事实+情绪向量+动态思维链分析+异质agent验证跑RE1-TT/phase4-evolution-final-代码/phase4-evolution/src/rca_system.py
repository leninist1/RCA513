import os, sys, copy
from typing import List, Dict

RCA_WORKSPACE = "/home/admin/rca-workspace"
if RCA_WORKSPACE not in sys.path:
    sys.path.insert(0, RCA_WORKSPACE)

from run_re1tt import (
    RCA_RE1TT, load_re1tt_case, NaiveBayesFaultClassifier,
    load_re1tt_data, detect_anomaly_with_persistence,
    compute_propagation_gain, apply_soft_counterfactual,
    compute_global_degradation, soft_recovery_score,
    compute_loss_scores, detect_fault_mode_with_proba,
    BASELINE_WINDOW_SECONDS, TOP_K, ANOMALY_Z_THRESHOLD,
    CAUSAL_WINDOW_SECONDS, SOFT_CF_BETA, RECOVERY_GAMMA,
    PROPAGATION_WEIGHT, METRIC_CONFIGS
)

from src.operators import OperatorConfig
from src.rca_config import RCAConfig
from src.operator_library import (
    PROPAGATION_STRATEGIES, TIME_PENALTY_STRATEGIES, FUSION_STRATEGIES
)


class RCASystem:
    _cached_classifier = None

    def __init__(self, operators: List[OperatorConfig], config: RCAConfig = None):
        self.operators = operators
        self.config = config or RCAConfig()
        self.classifier = RCASystem._cached_classifier
        if self.classifier is None:
            self._train_classifier()
            RCASystem._cached_classifier = self.classifier

    def _train_classifier(self):
        data_root = "/home/admin/RCAEval/data/RE1-TT"
        all_features, all_labels = [], []
        for fault_name in sorted(os.listdir(data_root)):
            fault_path = os.path.join(data_root, fault_name)
            if not os.path.isdir(fault_path): continue
            actual_fault_type = fault_name.split("_")[-1].upper()
            if actual_fault_type == "DISK": actual_fault_type = "CPU"
            if actual_fault_type not in ["CPU", "MEM", "DELAY", "LOSS"]: actual_fault_type = "CPU"
            for rep in sorted(os.listdir(fault_path)):
                case_dir = os.path.join(fault_path, rep)
                if not os.path.isdir(case_dir): continue
                try:
                    data_path, inject_time, _ = load_re1tt_case(case_dir)
                    baseline_df, fault_df, services = load_re1tt_data(data_path, inject_time)
                    if baseline_df.empty: continue
                    candidates_raw = detect_anomaly_with_persistence(baseline_df, fault_df, services)
                    if not candidates_raw: continue
                    temp_clf = NaiveBayesFaultClassifier()
                    features = temp_clf.extract_features(candidates_raw)
                    all_features.append(features); all_labels.append(actual_fault_type)
                except: pass
        if all_features:
            self.classifier = NaiveBayesFaultClassifier()
            self.classifier.fit(all_features, all_labels)
            print(f"[RCASystem] 分类器训练完成，{len(all_features)} 个样本")

    def predict(self, case_data: Dict) -> Dict:
        import run_re1tt as rca_mod
        cfg = self.config
        
        # 1. 标量参数注入
        rca_mod.ANOMALY_Z_THRESHOLD = cfg.anomaly_z_threshold
        rca_mod.TOP_K = cfg.top_k
        rca_mod.CAUSAL_WINDOW_SECONDS = cfg.causal_window_seconds
        rca_mod.SOFT_CF_BETA = cfg.soft_cf_beta
        rca_mod.RECOVERY_GAMMA = cfg.recovery_gamma
        rca_mod.PROPAGATION_WEIGHT = cfg.propagation_weight
        
        # 2. 权重映射
        weight_map = {
            "istio-error-total": cfg.istio_error_weight,
            "container-memory-failures-total": cfg.memory_failures_weight,
            "container-cpu-usage-seconds-total": cfg.cpu_usage_weight,
            "istio-latency-99": cfg.latency_p99_weight,
        }
        new_configs = [(s, weight_map.get(s, w)) for s, w in rca_mod.METRIC_CONFIGS]
        rca_mod.METRIC_CONFIGS = new_configs
        
        # 3. 加载数据
        data_path = case_data["data_path"]
        inject_time = case_data["inject_time"]
        baseline_df, fault_df, services = load_re1tt_data(data_path, inject_time)
        if baseline_df.empty:
            return {"root_cause": "unknown", "reasoning_trace": "空数据"}
        
        # 4. 异常检测
        candidates_raw = detect_anomaly_with_persistence(baseline_df, fault_df, services)
        if not candidates_raw:
            return {"root_cause": "unknown", "reasoning_trace": "无候选"}
        
        current_svcs = [s for s, _, _, _ in candidates_raw]
        infl_scores = {s: sc for s, sc, _, _ in candidates_raw}
        all_times = {s: t for s, _, t, _ in candidates_raw if t is not None}
        
        # 5. 🔥 根据 config 选择传播追踪策略
        prop_func = PROPAGATION_STRATEGIES.get(cfg.propagation_strategy, 
                                                PROPAGATION_STRATEGIES["downstream_1_hop"])
        propagation_scores = prop_func(candidates_raw, all_times, infl_scores, services, baseline_df, fault_df)
        
        # 6. 故障类型判别
        fault_mode, proba = detect_fault_mode_with_proba(candidates_raw, self.classifier)
        
        # 7. 反事实恢复
        rec_scores = {}
        for svc in current_svcs:
            cf_df = apply_soft_counterfactual(fault_df, baseline_df, svc)
            rec_scores[svc] = soft_recovery_score(fault_df, cf_df, baseline_df, svc, services)
        
        # 8. 🔥 根据 config 选择时间惩罚策略
        time_func = TIME_PENALTY_STRATEGIES.get(cfg.time_penalty_strategy,
                                                 TIME_PENALTY_STRATEGIES["linear_decay"])
        time_penalties = time_func(candidates_raw)
        for svc in infl_scores:
            if svc in time_penalties:
                infl_scores[svc] *= time_penalties[svc]
        
        # 9. 🔥 根据 config 选择融合策略
        fusion_func = FUSION_STRATEGIES.get(cfg.fusion_strategy,
                                             FUSION_STRATEGIES["multiplicative"])
        combined = fusion_func(infl_scores, rec_scores, propagation_scores)
        
        # 10. LOSS 专用计算（保持原版逻辑）
        if fault_mode == "LOSS":
            loss_scores = compute_loss_scores(candidates_raw, infl_scores, all_times, services, baseline_df, fault_df)
            for svc in combined:
                combined[svc] = combined.get(svc, 0) + 0.3 * loss_scores.get(svc, 0)
        
        sorted_final = sorted(combined.items(), key=lambda x: x[1], reverse=True)
        
        top_service = sorted_final[0][0] if sorted_final else "unknown"
        trace_parts = []
        for rank, (svc, score) in enumerate(sorted_final[:3], 1):
            trace_parts.append(f"#{rank}: {svc}({score:.3f})")
        reasoning = " | ".join(trace_parts)
        
        return {
            "root_cause": top_service,
            "reasoning_trace": reasoning,
            "top_candidates": [s for s, _ in sorted_final[:3]]
        }

RCASystem._cached_classifier = None
