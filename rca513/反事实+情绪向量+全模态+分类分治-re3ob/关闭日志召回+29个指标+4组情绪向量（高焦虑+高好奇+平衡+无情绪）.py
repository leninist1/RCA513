cat > /home/admin/rca-workspace/test_no_log_parallel.py << 'ENDOFFILE'
import sys, os, json
sys.path.insert(0, "/home/admin/rca-workspace")
from emotion_real import (
    load_rcaeval_data, build_weighted_graph_from_traces,
    detect_anomaly, extract_log_anomaly, extract_log_keywords,
    extract_metrics_timeseries_spikes,
    apply_targeted_counterfactual, continuous_recovery_score,
    classify_fault, binary_counterfactual_test, infer_graph_from_metrics
)
import pandas as pd, numpy as np
from multiprocessing import Pool

data_root = "/home/admin/RCAEval/data/RE3/RE3-OB"
with open("/home/admin/phase4-evolution-re3/data/annotations/ground_truth.json") as f:
    ground_truth = json.load(f)

# 预加载
CACHE = {}
for fault_name in sorted(os.listdir(data_root)):
    fault_path = os.path.join(data_root, fault_name)
    if not os.path.isdir(fault_path): continue
    for rep in sorted(os.listdir(fault_path)):
        case_dir = os.path.join(fault_path, rep)
        if not os.path.isdir(case_dir): continue
        case_id = f"{fault_name}/{rep}"
        mp = os.path.join(case_dir, "metrics.csv")
        ip = os.path.join(case_dir, "inject_time.txt")
        tp = os.path.join(case_dir, "traces.csv")
        lp = os.path.join(case_dir, "logs.csv")
        if not all(os.path.exists(p) for p in [mp, ip]): continue
        with open(ip) as f: inject_time = int(f.read().strip())
        CACHE[case_id] = {
            "metrics_path": mp, "inject_time": inject_time,
            "traces_df": pd.read_csv(tp) if os.path.exists(tp) else None,
            "logs_df": pd.read_csv(lp, on_bad_lines='skip') if os.path.exists(lp) else None
        }
print(f"预加载 {len(CACHE)} 个 case")

_emotion = None
_gt = None

def init_worker(emotion, gt):
    global _emotion, _gt
    _emotion = emotion; _gt = gt

def run_one(case_id):
    global _emotion, _gt
    c = CACHE[case_id]
    tdf = c["traces_df"]
    if tdf is not None and 'startTimeMillis' in tdf.columns:
        tdf = tdf.copy(); tdf['time'] = tdf['startTimeMillis'] // 1000
    
    mp, inject_time, logs_df = c["metrics_path"], c["inject_time"], c["logs_df"]
    true = _gt.get(case_id, "")
    
    baseline_df, fault_df, services = load_rcaeval_data(mp, inject_time)
    if baseline_df.empty: return None
    entry = next((s for s in services if "frontend" in s.lower() or "gateway" in s.lower()), services[0])
    
    edge_graph = build_weighted_graph_from_traces(tdf, baseline_df, fault_df, inject_time) if tdf is not None else {}
    graph = {p: set(c.keys()) for p, c in edge_graph.items()} if edge_graph else infer_graph_from_metrics(services, baseline_df, fault_df)
    
    valid_svcs = [s for s in services if s != entry]
    candidates_raw = detect_anomaly(baseline_df, fault_df, valid_svcs, edge_graph=edge_graph)
    if not candidates_raw: return None
    
    current_svcs = [s for s, _, _ in candidates_raw]
    infl_scores, rec_scores = {}, {}
    for s in current_svcs:
        out_rad = sum(edge_graph.get(s, {}).values()) if edge_graph else 0.0
        in_ctrl = 0.0
        if edge_graph:
            for parent, children in edge_graph.items():
                if s in children: in_ctrl += children[s]
        node_sc = next((sc for svc, sc, _ in candidates_raw if svc == s), 0.1)
        log_counts = extract_log_anomaly(logs_df, inject_time)
        infl_scores[s] = node_sc + 2.0 * in_ctrl - 1.5 * out_rad + np.log1p(log_counts.get(s, 0)) * 5.0
        cf_df = apply_targeted_counterfactual(fault_df, baseline_df, s, entry, graph)
        rec_scores[s] = continuous_recovery_score(fault_df, cf_df, baseline_df, s, graph, services)
    
    fault_type = classify_fault(infl_scores, rec_scores)
    sorted_items = sorted(infl_scores.items(), key=lambda x: x[1], reverse=True)
    
    emotion = _emotion
    
    if fault_type == "B":
        top2 = [s for s, _ in sorted_items[:2]]
        if len(top2) >= 2:
            winner = binary_counterfactual_test(fault_df, baseline_df, top2[0], top2[1], entry, graph, services)
            if winner:
                dispatched = [(winner, rec_scores.get(winner, 0), infl_scores.get(winner, 0))]
                for s, infl in sorted_items[:5]:
                    if s != winner: dispatched.append((s, rec_scores.get(s, 0), infl))
                dispatched = dispatched[:5]
            else:
                dispatched = [(s, rec_scores.get(s, 0), infl) for s, infl in sorted_items[:5]]
        else:
            dispatched = [(s, rec_scores.get(s, 0), infl) for s, infl in sorted_items[:5]]
    else:
        metrics_spike_scores = extract_metrics_timeseries_spikes(baseline_df, fault_df, inject_time, current_svcs)
        combined = {}
        for svc, infl in infl_scores.items():
            combined[svc] = infl + metrics_spike_scores.get(svc, 0)
        sorted_boosted = sorted(combined.items(), key=lambda x: x[1], reverse=True)
        
        if emotion:
            E_cur, E_anx, E_sat = emotion
            n = len(sorted_boosted)
            if E_anx > 6 and n > 1: sorted_boosted = sorted_boosted[:1]
            elif E_cur > 6: sorted_boosted = sorted_boosted[:min(7, n)]
            elif E_sat > 6: sorted_boosted = sorted_boosted[:3]
        dispatched = [(s, rec_scores.get(s, 0), infl_scores.get(s, 0)) for s, _ in sorted_boosted[:5]]
    
    svcs = [s for s, _, _ in dispatched]
    hit = svcs[0] == true
    top3_hit = true in svcs[:3]
    mrr = 0
    for rk, sv in enumerate(svcs, 1):
        if sv == true: mrr = 1.0/rk; break
    return {"hit": hit, "top3_hit": top3_hit, "mrr": mrr}

def run_config(label, emotion):
    with Pool(4, initializer=init_worker, initargs=(emotion, _gt)) as pool:
        results = pool.map(run_one, list(CACHE.keys()))
    results = [r for r in results if r is not None]
    n = len(results)
    top1 = sum(1 for r in results if r["hit"])
    top3 = sum(1 for r in results if r["top3_hit"])
    mrr = sum(r["mrr"] for r in results)
    print(f"\n🧠 {label} (无日志召回)")
    print(f"Top-1: {top1/n:.3f} ({top1}/{n})")
    print(f"Top-3: {top3/n:.3f} ({top3}/{n})")
    print(f"MRR:   {mrr/n:.3f}")

if __name__ == "__main__":
    _gt = ground_truth
    run_config("高焦虑", [2,9,3])
    run_config("高好奇", [8,2,1])
    run_config("平衡型", [5,5,4])
    run_config("无情绪", None)
ENDOFFILE

source ~/tanyu-voice/venv/bin/activate
cd /home/admin/rca-workspace && python3 test_no_log_parallel.py