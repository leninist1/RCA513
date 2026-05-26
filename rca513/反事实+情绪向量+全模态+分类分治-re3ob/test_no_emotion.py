cat > /home/admin/rca-workspace/test_no_emotion.py << 'ENDOFFILE'
import sys, os, json
sys.path.insert(0, "/home/admin/rca-workspace")
# 直接导入底层函数，不用情绪调度
from emotion_real import (
    load_rcaeval_data, build_weighted_graph_from_traces,
    detect_anomaly, extract_log_anomaly, extract_log_keywords,
    extract_fatal_log_services, extract_metrics_timeseries_spikes,
    apply_targeted_counterfactual, continuous_recovery_score,
    classify_fault, binary_counterfactual_test, infer_graph_from_metrics
)
import pandas as pd, numpy as np

data_root = "/home/admin/RCAEval/data/RE3/RE3-OB"
with open("/home/admin/phase4-evolution-re3/data/annotations/ground_truth.json") as f:
    ground_truth = json.load(f)

top1, top3, mrr, total = 0, 0, 0.0, 0

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
        
        with open(ip) as f: inject_time = int(f.read().strip())
        traces_df = pd.read_csv(tp) if os.path.exists(tp) else None
        if traces_df is not None and 'startTimeMillis' in traces_df.columns:
            traces_df['time'] = traces_df['startTimeMillis'] // 1000
        logs_df = pd.read_csv(lp, on_bad_lines='skip') if os.path.exists(lp) else None
        
        true = ground_truth.get(case_id, "")
        
        # ====== 纯数学版本：无情绪调度 ======
        baseline_df, fault_df, services = load_rcaeval_data(mp, inject_time)
        if baseline_df.empty: continue
        entry = next((s for s in services if "frontend" in s.lower() or "gateway" in s.lower()), services[0])
        
        edge_graph = build_weighted_graph_from_traces(traces_df, baseline_df, fault_df, inject_time) if traces_df is not None else {}
        graph = {p: set(c.keys()) for p, c in edge_graph.items()} if edge_graph else infer_graph_from_metrics(services, baseline_df, fault_df)
        
        valid_svcs = [s for s in services if s != entry]
        fatal = extract_fatal_log_services(logs_df, inject_time, valid_svcs)
        candidates_raw = detect_anomaly(baseline_df, fault_df, valid_svcs, edge_graph=edge_graph)
        if not candidates_raw and not fatal: continue
        
        existing = {s for s, _, _ in candidates_raw}
        for svc in fatal:
            if svc not in existing: candidates_raw.append((svc, 0.5, None))
        
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
        
        # Type A: 影响力排序
        if fault_type == "A":
            results = [(s, rec_scores.get(s, 0), infl) for s, infl in sorted_items[:5]]
        elif fault_type == "B":
            # 尝试反事实，失败则降级为影响力排序
            top2 = [s for s, _ in sorted_items[:2]]
            if len(top2) >= 2:
                winner = binary_counterfactual_test(fault_df, baseline_df, top2[0], top2[1], entry, graph, services)
                if winner:
                    results = [(winner, rec_scores.get(winner, 0), infl_scores.get(winner, 0))]
                    for s, infl in sorted_items[:5]:
                        if s != winner: results.append((s, rec_scores.get(s, 0), infl))
                    results = results[:5]
                else:
                    results = [(s, rec_scores.get(s, 0), infl) for s, infl in sorted_items[:5]]
            else:
                results = [(s, rec_scores.get(s, 0), infl) for s, infl in sorted_items[:5]]
        else:
            # Type C: 全模态融合（日志+Metrics时序）
            log_keyword_scores = extract_log_keywords(logs_df, inject_time, current_svcs)
            metrics_spike_scores = extract_metrics_timeseries_spikes(baseline_df, fault_df, inject_time, current_svcs)
            combined = {}
            for svc, infl in infl_scores.items():
                log_s = log_keyword_scores.get(svc, 0)
                met_s = metrics_spike_scores.get(svc, 0)
                combined[svc] = infl + log_s * 10000 + met_s
            sorted_boosted = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:5]
            results = [(s, rec_scores.get(s, 0), infl_scores.get(s, 0)) for s, _ in sorted_boosted]
        
        total += 1
        svcs = [s for s, _, _ in results]
        h = "✅" if svcs[0] == true else "❌"
        if svcs[0] == true: top1 += 1
        if true in svcs[:3]: top3 += 1
        for rk, sv in enumerate(svcs, 1):
            if sv == true: mrr += 1.0/rk; break
        
        print(f"📋 {case_id} 🎯 {true}")
        print(f"  {h} 预测: {svcs[0]} | 类型: Type {fault_type} | Top3: {svcs[:3]}")

print(f"\n{'='*50}")
print(f"🔥 纯数学版本（无情绪调度）")
print(f"Top-1: {top1/total:.3f} ({top1}/{total})")
print(f"Top-3: {top3/total:.3f} ({top3}/{total})")
print(f"MRR:   {mrr/total:.3f}")
ENDOFFILE

source ~/tanyu-voice/venv/bin/activate
cd /home/admin/rca-workspace && python3 test_no_emotion.py