"""
反事实验证驱动的RCA框架
基于RCAEval RE3数据集（pandas DataFrame格式）
v4.2 - 2026年5月6日

v4.2 修复显示 Bug：
- 移除了愚蠢的 `pass`，把排名的打印语句加回来了，并展示决胜局的 infl 分数！
"""

import copy
import random
import numpy as np
import pandas as pd
from collections import defaultdict

# ======== 配置 ========
BASELINE_WINDOW_SECONDS = 300
TOP_K = 5
RECOVERY_THRESHOLD = 0.8
CONTROL_N = 2
ANOMALY_Z_THRESHOLD = 3.0
TIME_TOLERANCE = 60
EDGE_IMPORTANCE_MIN_CALLS = 1

EXCLUDE_SERVICE_PATTERNS = ["ip-", "compute.internal", "frontend-external"]

METRIC_CONFIGS = [
    ("istio-latency-99", 1.0),
    ("istio-error-total", 2.0),
    ("container-cpu-usage-seconds-total", 1.0),
    ("container-memory-failures-total", 2.0),
    ("container-memory-working-set-bytes", 0.5)
]

def _is_valid_service(svc: str) -> bool:
    for pat in EXCLUDE_SERVICE_PATTERNS:
        if pat in svc: return False
    return True

# ======== 1. 数据加载 ========
def load_rcaeval_data(data_path, inject_time):
    df = pd.read_csv(data_path)
    if "time.1" in df.columns: df = df.drop(columns=["time.1"])
    df = df.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
    baseline_df = df[df["time"] < inject_time].tail(BASELINE_WINDOW_SECONDS // 15)
    fault_df = df[df["time"] >= inject_time]
    services = {col.split("_", 1)[0] for col in df.columns if col != "time" and len(col.split("_", 1)) == 2}
    return baseline_df, fault_df, sorted([s for s in services if _is_valid_service(s)])

# ======== 2. 调用图构建 ========
def _find_time_column(traces_df):
    for col in traces_df.columns:
        if col.lower() in ("timestamp", "time", "start_time", "starttime", "ts"): return col
    return None

def build_weighted_graph_from_traces(traces_df, baseline_df, fault_df, inject_time):
    if traces_df is None or traces_df.empty: return {}
    col_map = {col.lower(): col for col in traces_df.columns}
    cm = {}
    for k, v in col_map.items():
        if k in ("traceid", "trace_id"): cm["trace_id"] = v
        elif k in ("spanid", "span_id"): cm["span_id"] = v
        elif k in ("parentspanid", "parent_id", "parentid"): cm["parent_id"] = v
        elif k in ("servicename", "service"): cm["service"] = v
        elif k in ("duration", "latency"): cm["duration"] = v
        elif k in ("statuscode", "status_code", "httpcode"): cm["status_code"] = v

    if not all(k in cm for k in ("trace_id", "span_id", "service")): return {}

    time_col = _find_time_column(traces_df)
    if time_col:
        traces_df[time_col] = pd.to_numeric(traces_df[time_col], errors="coerce")
        base_traces = traces_df[traces_df[time_col] < inject_time]
        fault_traces = traces_df[traces_df[time_col] >= inject_time]
    else:
        base_traces, fault_traces = pd.DataFrame(columns=traces_df.columns), traces_df

    def extract_edge_stats(df):
        edges = defaultdict(list)
        if df.empty: return edges
        for _, group in df.groupby(cm["trace_id"]):
            id_map = {str(row[cm["span_id"]]): (row[cm["service"]], row) for _, row in group.iterrows() if _is_valid_service(row[cm["service"]])}
            if "parent_id" not in cm: continue
            for _, row in group.iterrows():
                pid = row.get(cm["parent_id"])
                if pd.isna(pid) or str(pid).strip() not in id_map: continue
                parent_svc, _ = id_map[str(pid).strip()]
                child_svc = row[cm["service"]]
                if not _is_valid_service(parent_svc) or not _is_valid_service(child_svc) or parent_svc == child_svc: continue
                dur = row.get(cm.get("duration", ""), 0)
                dur = 0 if pd.isna(dur) else float(dur)
                is_err = 1 if "status_code" in cm and str(row[cm["status_code"]]).startswith(("5", "4", "e")) else 0
                edges[(parent_svc, child_svc)].append((dur, is_err))
        return edges

    base_edges, fault_edges = extract_edge_stats(base_traces), extract_edge_stats(fault_traces)
    graph = defaultdict(dict)
    for edge, fault_records in fault_edges.items():
        if len(fault_records) < EDGE_IMPORTANCE_MIN_CALLS: continue
        base_records = base_edges.get(edge, [])
        b_durs, b_errs = [r[0] for r in base_records], [r[1] for r in base_records]
        f_durs, f_errs = [r[0] for r in fault_records], [r[1] for r in fault_records]

        if not b_durs:
            weight = (np.percentile(f_durs, 95) if f_durs else 0) / 1000.0 + (np.mean(f_errs) if f_errs else 0) * 10
        else:
            b_med, mad = np.median(b_durs), np.median(np.abs(np.array(b_durs) - np.median(b_durs)))
            lat_z = abs(np.median(f_durs) - b_med) / (mad + 1e-9) if mad > 0 else 0
            err_score = max(0, (np.mean(f_errs) - np.mean(b_errs)) * 10)
            weight = lat_z + err_score
        graph[edge[0]][edge[1]] = weight
    return dict(graph)

# ======== 3. 日志分析 ========
def extract_log_anomaly(logs_df, inject_time):
    if logs_df is None or logs_df.empty or "message" not in logs_df.columns or "container_name" not in logs_df.columns:
        return {}
    if "timestamp" in logs_df.columns:
        logs_df["time_sec"] = pd.to_numeric(logs_df["timestamp"], errors="coerce") // 10**9
    else:
        return {}
    fault_logs = logs_df[logs_df["time_sec"] >= inject_time]
    if fault_logs.empty: return {}
    error_mask = fault_logs["message"].astype(str).str.contains("ERROR|Exception|Timeout|Fail|fail|error", case=False, na=False)
    error_counts = fault_logs[error_mask].groupby("container_name").size().to_dict()
    return error_counts

# ======== 4. 异常与恢复 ========
def detect_anomaly(baseline_df, fault_df, services):
    scores = []
    for svc in services:
        score, first_anomaly_time = 0.0, None
        for metric_suffix, weight in METRIC_CONFIGS:
            col_candidates = [f"{svc}_{metric_suffix}", f"{svc}_{metric_suffix.replace('-', '_')}"]
            col = next((c for c in col_candidates if c in fault_df.columns and c in baseline_df.columns), None)
            if col:
                b_vals, f_vals = baseline_df[col].values, fault_df[col].values
                if len(b_vals) == 0 or len(f_vals) == 0: continue
                b_med, mad = np.median(b_vals), np.median(np.abs(b_vals - np.median(b_vals))) + 1e-9
                f_peak = np.percentile(f_vals, 95)
                z = max(0, f_peak - b_med) / mad
                if z > ANOMALY_Z_THRESHOLD:
                    score += z * weight
                    rows = fault_df[fault_df[col] > b_med + ANOMALY_Z_THRESHOLD * mad]
                    if not rows.empty:
                        t = rows["time"].iloc[0]
                        if first_anomaly_time is None or t < first_anomaly_time: first_anomaly_time = t
        if score > 0: scores.append((svc, score, first_anomaly_time))
    return sorted(scores, key=lambda x: x[1], reverse=True)[:TOP_K]

def expand_candidates_with_upstream(candidates, graph, all_services):
    reverse_graph = defaultdict(set)
    for parent in graph:
        for child in graph[parent]: reverse_graph[child].add(parent)
    expanded = {svc for svc, _, _ in candidates}
    for svc, _, _ in candidates:
        if svc in reverse_graph: expanded.update(reverse_graph[svc])
    return [s for s in expanded if s in all_services]

def is_affected_caller(service_y, service_x, graph, max_depth=10):
    visited = set()
    def dfs(s, depth):
        if depth > max_depth: return False
        for callee in graph.get(s, set()):
            if callee == service_x: return True
            if callee not in visited:
                visited.add(callee)
                if dfs(callee, depth + 1): return True
        return False
    return dfs(service_y, 0)

def apply_counterfactual(fault_df, baseline_df, target_service, entry_service, graph=None):
    cf_df = copy.deepcopy(fault_df)
    aligned_base = baseline_df.iloc[-len(fault_df):] if len(baseline_df) >= len(fault_df) else baseline_df
    def restore_service(svc):
        for col in cf_df.columns:
            if not col.startswith(svc + "_") or col not in aligned_base.columns: continue
            bv = aligned_base[col].dropna().values
            if len(bv) > 0: cf_df[col] = np.median(bv)
    restore_service(target_service)
    if graph:
        for svc in set(graph.keys()) | {s for v in graph.values() for s in v}:
            if svc != target_service and svc != entry_service and is_affected_caller(svc, target_service, graph):
                restore_service(svc)
    return cf_df

def compute_single_svc_degradation(df, svc, baseline_df):
    deg = 0.0
    for metric_suffix, weight in METRIC_CONFIGS:
        col_candidates = [f"{svc}_{metric_suffix}", f"{svc}_{metric_suffix.replace('-', '_')}"]
        col = next((c for c in col_candidates if c in df.columns and c in baseline_df.columns), None)
        if col:
            b_vals, f_vals = baseline_df[col].values, df[col].values
            if len(b_vals) > 0 and len(f_vals) > 0:
                b_med, mad = np.median(b_vals), np.median(np.abs(b_vals - np.median(b_vals))) + 1e-9
                f_peak = np.percentile(f_vals, 95)
                deg += (max(0, f_peak - b_med) / mad) * weight
    return deg

def system_health(df, baseline_df, target_service, graph, all_services):
    total_deg = compute_single_svc_degradation(df, target_service, baseline_df)
    svcs_to_check = all_services if not graph else [s for s in all_services if s != target_service and is_affected_caller(s, target_service, graph)]
    for svc in svcs_to_check:
        if svc != target_service: total_deg += compute_single_svc_degradation(df, svc, baseline_df)
    return total_deg

def recovery_score(orig_df, cf_df, baseline_df, target_service, graph, all_services):
    orig_deg = system_health(orig_df, baseline_df, target_service, graph, all_services)
    cf_deg = system_health(cf_df, baseline_df, target_service, graph, all_services)
    return float((orig_deg - cf_deg) / (abs(orig_deg) + 1e-9))

# ======== 5. 影响力与决胜排序 ========
def compute_weighted_influence_scores(candidates, edge_graph, node_anomaly_scores, log_counts):
    scores = {}
    for svc, node_sc, _ in candidates:
        out_rad = sum(edge_graph.get(svc, {}).values())
        in_ctrl = sum(children.get(svc, 0) for children in edge_graph.values())
        log_boost = np.log1p(log_counts.get(svc, 0)) * 5.0 
        scores[svc] = node_sc + 2.0 * out_rad - 1.5 * in_ctrl + log_boost
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

def control_experiment(fault_df, baseline_df, entry_service, candidates, all_services, edge_graph, graph=None):
    c_set = {s for s, _, _ in candidates}
    pots = [s for s in all_services if s != entry_service and s not in c_set and (not edge_graph or s in edge_graph or any(s in ch for ch in edge_graph.values()))]
    ctrl_svcs = random.sample(pots, min(CONTROL_N, len(pots))) if pots else []
    return [(s, recovery_score(fault_df, apply_counterfactual(fault_df, baseline_df, s, entry_service, graph), baseline_df, s, graph, all_services)) for s in ctrl_svcs]

# ======== 6. 主流程 ========
def infer_graph_from_metrics(services, baseline_df, fault_df):
    graph, anomaly_times = defaultdict(set), {}
    for svc in services:
        col = f"{svc}_istio-latency-99"
        if col in fault_df.columns:
            bm = np.median(baseline_df[col].values) if not baseline_df.empty else 0
            rows = fault_df[fault_df[col] > bm * 2]
            if not rows.empty: anomaly_times[svc] = rows["time"].iloc[0]
    svcs = sorted(anomaly_times, key=anomaly_times.get)
    for i, up in enumerate(svcs):
        for down in svcs[i+1:]:
            if anomaly_times[down] - anomaly_times[up] <= TIME_TOLERANCE: graph[up].add(down)
    return dict(graph)

def RCA(data_path, inject_time, ground_truth=None, traces_df=None, logs_df=None):
    baseline_df, fault_df, services = load_rcaeval_data(data_path, inject_time)
    if baseline_df.empty: return []
    entry = next((s for s in services if "frontend" in s.lower() or "gateway" in s.lower()), services[0])

    log_counts = extract_log_anomaly(logs_df, inject_time)
    
    # 打印检测到的日志错误
    if log_counts:
        print(f"[logs] 检出日志报错服务: {', '.join([f'{k}({v})' for k, v in log_counts.items()])}")

    valid_svcs = [s for s in services if s != entry]
    edge_graph = build_weighted_graph_from_traces(traces_df, baseline_df, fault_df, inject_time) if traces_df is not None else {}
    if not edge_graph:
        graph = infer_graph_from_metrics(valid_svcs, baseline_df, fault_df)
        edge_graph = {s: {c: 1.0 for c in ch} for s, ch in graph.items()}
    else:
        graph = {p: set(c.keys()) for p, c in edge_graph.items()}

    candidates_raw = detect_anomaly(baseline_df, fault_df, valid_svcs)
    if not candidates_raw: return []

    cands_dict = {s: (sc, t) for s, sc, t in candidates_raw}
    for svc in expand_candidates_with_upstream(candidates_raw, graph, services):
        if svc not in cands_dict: cands_dict[svc] = (0.1, None)
    
    candidates = sorted([(s, cands_dict[s][0], cands_dict[s][1]) for s in cands_dict], key=lambda x: x[1], reverse=True)[:TOP_K]

    infl_scores = dict(compute_weighted_influence_scores(candidates, edge_graph, {s: sc for s, sc, _ in candidates}, log_counts))
    final_cands = sorted([(s, infl_scores.get(s, sc), t) for s, sc, t in candidates], key=lambda x: x[1], reverse=True)

    results = []
    for svc, init_score, _ in final_cands:
        rec = recovery_score(fault_df, apply_counterfactual(fault_df, baseline_df, svc, entry, graph), baseline_df, svc, graph, services)
        results.append((svc, rec, init_score))

    ctrl = control_experiment(fault_df, baseline_df, entry, candidates, services, edge_graph, graph)
    max_ctrl = max([v for _, v in ctrl]) if ctrl else 0.0

    penalized = []
    for svc, rec, init in results:
        final_score = rec - 0.5 * max_ctrl
        infl = infl_scores.get(svc, 0)
        penalized.append((svc, final_score, rec, infl))
    
    # 排序逻辑：第一优先级是保留3位小数的 final 恢复分，如果是双黄蛋，第二优先级拼影响力！
    penalized.sort(key=lambda x: (round(x[1], 3), x[3]), reverse=True)

    print(f"对照组最高恢复分: {max_ctrl:.3f}")
    for rank, (svc, final, rec, infl) in enumerate(penalized, 1):
        # 加上了打印！不会再静默了！
        print(f"  {rank}. {svc:<25} final={final:.3f} (recovery={rec:.3f}, infl={infl:.1f})")
        
    return penalized

# ======== 7. 评估接口 ========
def load_re3_case(case_dir):
    import os
    with open(os.path.join(case_dir, "inject_time.txt")) as f: inj_t = int(f.read().strip())
    
    t_path = os.path.join(case_dir, "traces.csv")
    t_df = pd.read_csv(t_path) if os.path.exists(t_path) else None
    if t_df is not None:
        if 'startTimeMillis' in t_df.columns: t_df['time'] = t_df['startTimeMillis'] // 1000
        elif 'startTime' in t_df.columns: t_df['time'] = pd.to_numeric(t_df['startTime'], errors='coerce') / 1_000_000.0
            
    l_path = os.path.join(case_dir, "logs.csv")
    l_df = None
    if os.path.exists(l_path):
        try: l_df = pd.read_csv(l_path, on_bad_lines='skip')
        except: pass

    fault_dir = next((p for p in os.path.normpath(case_dir).split(os.sep) if "_f" in p and any(c.isdigit() for c in p.split("_f")[-1])), None)
    return os.path.join(case_dir, "metrics.csv"), inj_t, t_df, l_df, fault_dir.rsplit("_f", 1)[0] if fault_dir else None

def run_re3_benchmark(re3_root, system="RE3-OB", max_cases=None):
    import os
    system_dir = os.path.join(re3_root, system)
    if not os.path.exists(system_dir): return
    
    top1, top3, mrr, total = 0, 0, 0.0, 0
    for fault_name in sorted(os.listdir(system_dir)):
        for rep in sorted(os.listdir(os.path.join(system_dir, fault_name))):
            case_dir = os.path.join(system_dir, fault_name, rep)
            if not os.path.isdir(case_dir): continue
            if max_cases and total >= max_cases: break
            try:
                metrics_path, inject_time, traces_df, logs_df, gt = load_re3_case(case_dir)
                print(f"\n{'='*60}\n🚀 数据: {case_dir.split('/')[-2]}/{case_dir.split('/')[-1]}")
                res = RCA(metrics_path, inject_time, gt, traces_df, logs_df)
                if not res: continue
                total += 1
                svcs = [s for s, _, _, _ in res]
                if svcs and svcs[0] == gt: top1 += 1
                if gt in svcs[:3]: top3 += 1
                for rank, svc in enumerate(svcs, 1):
                    if svc == gt:
                        mrr += 1.0 / rank
                        break
            except Exception as e: print(f"❌ 报错 {case_dir}: {e}")

    print(f"\n{'='*60}\n🔥 RE3 {system} v4.2 (决胜局彻底修复版) 评估结果（{total}个case）:")
    if total > 0:
        print(f"Top-1 Accuracy: {top1/total:.3f} ({top1}/{total})")
        print(f"Top-3 Hit Rate: {top3/total:.3f} ({top3}/{total})")
        print(f"MRR:            {mrr/total:.3f}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "re3":
        run_re3_benchmark("/home/admin/RCAEval/data/RE3", max_cases=int(sys.argv[2]) if len(sys.argv) > 2 else None)