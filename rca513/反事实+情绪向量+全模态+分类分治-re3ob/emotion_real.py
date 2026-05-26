cat > /home/admin/rca-workspace/emotion_real.py << 'ENDOFFILE'
"""
RCA Module V3 — 全模态 + 情绪向量统一调度 + 分类分治
基于 V2，核心改动：
1. Type A/B/C 全部受情绪向量调控，而非只有 Type C
2. Type B：高焦虑→反事实二元淘汰，高好奇→扩大候选再反事实
3. Type C：高好奇→扩大候选集，高焦虑→剪枝，高满足→收敛
4. 预缓存 + 4核并行在测试脚本里做
"""

import copy, random, numpy as np, pandas as pd
from collections import defaultdict
import sys, os

# ========== 情绪向量 ==========
EMOTION_HIGH_ANXIETY   = [2, 9, 3]
EMOTION_HIGH_CURIOSITY = [8, 2, 1]
EMOTION_BALANCED       = [5, 5, 4]
EMOTION_DEFAULT        = [5, 5, 4]

# ========== 引擎配置 ==========
BASELINE_WINDOW_SECONDS = 300
TOP_K = 7
ANOMALY_Z_THRESHOLD = 3.0
TIME_TOLERANCE = 60
EDGE_IMPORTANCE_MIN_CALLS = 1
EXCLUDE_SERVICE_PATTERNS = ["ip-", "compute.internal", "frontend-external"]

METRIC_CONFIGS = [
    ("istio-latency-99", 1.0), ("istio-error-total", 2.0), ("istio-request-total", 1.5),
    ("istio-latency-50", 0.8), ("istio-latency-90", 0.8), ("istio-latency-95", 0.9),
    ("container-cpu-usage-seconds-total", 1.0), ("container-cpu-user-seconds-total", 0.6),
    ("container-memory-failures-total", 2.0), ("container-memory-working-set-bytes", 0.5),
    ("container-memory-usage-bytes", 0.8), ("container-memory-rss", 0.8),
    ("container-memory-max-usage-bytes", 0.8), ("container-memory-swap", 0.8),
    ("container-spec-memory-limit-bytes", 0.5),
    ("container-network-receive-errors-total", 1.5), ("container-network-transmit-errors-total", 1.5),
    ("container-network-receive-packets-dropped-total", 1.2), ("container-network-transmit-packets-dropped-total", 1.2),
    ("container-fs-writes-total", 0.5), ("container-sockets", 1.0), ("container-blkio-device-usage-total", 0.6),
    ("node-memory-active-bytes", 0.7), ("node-memory-inactive-bytes", 0.7),
    ("node-network-receive-errs-total", 0.7), ("node-network-receive-drop-total", 0.7),
    ("node-network-transmit-errs-total", 0.7), ("node-network-transmit-drop-total", 0.7),
    ("node-disk-read-bytes-total", 0.5), ("node-cpu-seconds-total", 0.6),
]

LOG_KEYWORD_TIERS = {
    "BindException": 10, "Address in use": 10, "Failed to bind": 10,
    "exit code 137": 10, "SIGKILL": 10, "OOM Killed": 10,
    "Container Killed": 10, "OutOfMemoryError": 10, "oom": 10, "killed": 10,
    "Connection Refused": 5, "connection refused": 5,
    "500 Internal Server Error": 5, "500": 5,
    "Timeout": 2, "timeout": 2, "timed out": 2,
    "Exception": 1, "exception": 1, "ERROR": 1, "error": 1,
}
FATAL_LOG_KEYWORDS = ["BindException", "Address in use", "Failed to bind",
    "exit code 137", "SIGKILL", "OOM Killed", "OutOfMemoryError", "Container Killed", "killed", "oom"]
METRICS_TIMESERIES_WINDOW, METRICS_SPIKE_THRESHOLD, METRICS_SPIKE_WEIGHT = 10, 3.0, 5000.0

# ========== 底层函数（不变）==========
def _is_valid_service(svc):
    for pat in EXCLUDE_SERVICE_PATTERNS:
        if pat in svc: return False
    return True

def load_rcaeval_data(data_path, inject_time):
    df = pd.read_csv(data_path)
    if "time.1" in df.columns: df = df.drop(columns=["time.1"])
    df = df.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
    baseline_df = df[df["time"] < inject_time].tail(BASELINE_WINDOW_SECONDS // 15)
    fault_df = df[df["time"] >= inject_time]
    services = {col.split("_", 1)[0] for col in df.columns if col != "time" and len(col.split("_", 1)) == 2}
    return baseline_df, fault_df, sorted([s for s in services if _is_valid_service(s)])

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
    time_col = next((c for c in traces_df.columns if c.lower() in ("timestamp", "time", "start_time", "starttime", "ts")), None)
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
                dur = float(row.get(cm.get("duration", ""), 0)) if not pd.isna(row.get(cm.get("duration", ""), 0)) else 0
                is_err = 1 if "status_code" in cm and str(row[cm["status_code"]]).startswith(("5", "4", "e")) else 0
                edges[(parent_svc, child_svc)].append((dur, is_err))
        return edges
    base_edges = extract_edge_stats(base_traces)
    fault_edges = extract_edge_stats(fault_traces)
    graph = defaultdict(dict)
    for edge, fault_records in fault_edges.items():
        if len(fault_records) < EDGE_IMPORTANCE_MIN_CALLS: continue
        base_records = base_edges.get(edge, [])
        b_durs = [r[0] for r in base_records]; b_errs = [r[1] for r in base_records]
        f_durs = [r[0] for r in fault_records]; f_errs = [r[1] for r in fault_records]
        if not b_durs:
            weight = (np.percentile(f_durs, 95) if f_durs else 0) / 1000.0 + (np.mean(f_errs) if f_errs else 0) * 10
        else:
            b_med = np.median(b_durs); mad = np.median(np.abs(np.array(b_durs) - b_med))
            lat_z = abs(np.median(f_durs) - b_med) / (mad + 1e-9) if mad > 0 else 0
            weight = lat_z + max(0, (np.mean(f_errs) - np.mean(b_errs)) * 10)
        graph[edge[0]][edge[1]] = weight
    return dict(graph)

def extract_log_anomaly(logs_df, inject_time):
    if logs_df is None or logs_df.empty or "message" not in logs_df.columns or "container_name" not in logs_df.columns: return {}
    if "timestamp" in logs_df.columns:
        logs_df["time_sec"] = pd.to_numeric(logs_df["timestamp"], errors="coerce") // 10**9
        fault_logs = logs_df[logs_df["time_sec"] >= inject_time]
    else: fault_logs = logs_df
    if fault_logs.empty: return {}
    error_mask = fault_logs["message"].astype(str).str.contains("ERROR|Exception|Timeout|Fail|error", case=False, na=False)
    return fault_logs[error_mask].groupby("container_name").size().to_dict()

def extract_log_keywords(logs_df, inject_time, candidates):
    if logs_df is None or logs_df.empty or "message" not in logs_df.columns or "container_name" not in logs_df.columns: return {}
    if "timestamp" in logs_df.columns:
        logs_df["time_sec"] = pd.to_numeric(logs_df["timestamp"], errors="coerce") // 10**9
        fault_logs = logs_df[logs_df["time_sec"] >= inject_time]
    else: fault_logs = logs_df
    if fault_logs.empty: return {}
    scores = {}
    for svc in candidates:
        svc_logs = fault_logs[fault_logs["container_name"] == svc]
        if svc_logs.empty: scores[svc] = 0.0; continue
        svc_score = 0.0
        for keyword, weight in LOG_KEYWORD_TIERS.items():
            if svc_logs["message"].astype(str).str.contains(keyword, case=False).any(): svc_score += weight
        scores[svc] = svc_score
    return scores

def extract_fatal_log_services(logs_df, inject_time, all_services):
    if logs_df is None or logs_df.empty: return []
    if "timestamp" in logs_df.columns:
        logs_df["time_sec"] = pd.to_numeric(logs_df["timestamp"], errors="coerce") // 10**9
        fault_logs = logs_df[logs_df["time_sec"] >= inject_time]
    else: fault_logs = logs_df
    if fault_logs.empty: return []
    forced = []
    for svc in all_services:
        if "container_name" not in fault_logs.columns: break
        svc_logs = fault_logs[fault_logs["container_name"] == svc]
        if svc_logs.empty: continue
        for keyword in FATAL_LOG_KEYWORDS:
            if svc_logs["message"].astype(str).str.contains(keyword, case=False).any():
                forced.append(svc); break
    return forced

def extract_metrics_timeseries_spikes(baseline_df, fault_df, inject_time, candidates):
    scores = {}
    baseline_tail = baseline_df.tail(METRICS_TIMESERIES_WINDOW)
    fault_head = fault_df.head(METRICS_TIMESERIES_WINDOW)
    if baseline_tail.empty or fault_head.empty: return {}
    for svc in candidates:
        svc_score = 0.0
        for metric_suffix, _ in METRIC_CONFIGS:
            col = next((c for c in [f"{svc}_{metric_suffix}", f"{svc}_{metric_suffix.replace('-', '_')}"] 
                       if c in baseline_tail.columns and c in fault_head.columns), None)
            if col is None: continue
            base_vals = baseline_tail[col].dropna().values; fault_vals = fault_head[col].dropna().values
            if len(base_vals) == 0 or len(fault_vals) == 0: continue
            ratio = np.percentile(fault_vals, 95) / (np.percentile(base_vals, 95) + 1e-9)
            if ratio > METRICS_SPIKE_THRESHOLD: svc_score += METRICS_SPIKE_WEIGHT * ratio
        scores[svc] = svc_score
    return scores

def detect_anomaly(baseline_df, fault_df, services, edge_graph=None):
    scores = []
    for svc in services:
        score = 0.0
        for metric_suffix, weight in METRIC_CONFIGS:
            col = next((c for c in [f"{svc}_{metric_suffix}", f"{svc}_{metric_suffix.replace('-', '_')}"] if c in fault_df.columns and c in baseline_df.columns), None)
            if col and len(baseline_df[col].values) > 0:
                b_vals, f_vals = baseline_df[col].values, fault_df[col].values
                b_med, mad = np.median(b_vals), np.median(np.abs(b_vals - np.median(b_vals))) + 1e-9
                z = max(0, np.percentile(f_vals, 95) - b_med) / mad
                if z > ANOMALY_Z_THRESHOLD: score += z * weight
        if edge_graph and svc in edge_graph:
            score = max(0.1, score - sum(edge_graph[svc].values()) * 0.5)
        if score > 0: scores.append((svc, score, None))
    return sorted(scores, key=lambda x: x[1], reverse=True)[:TOP_K]

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

def apply_targeted_counterfactual(fault_df, baseline_df, target_service, entry_service, graph=None):
    cf_df = copy.deepcopy(fault_df)
    aligned_base = baseline_df.iloc[-len(fault_df):] if len(baseline_df) >= len(fault_df) else baseline_df
    def restore(svc):
        for col in cf_df.columns:
            if col.startswith(svc + "_") and col in aligned_base.columns:
                bv = aligned_base[col].dropna().values
                if len(bv) > 0: cf_df[col] = np.median(bv)
    restore(target_service)
    if graph and target_service in graph:
        for child in graph[target_service]: restore(child)
    return cf_df

def compute_global_degradation(df, baseline_df, services):
    total_deg = 0.0
    for svc in services:
        for metric_suffix, weight in METRIC_CONFIGS:
            col = next((c for c in [f"{svc}_{metric_suffix}", f"{svc}_{metric_suffix.replace('-', '_')}"] if c in df.columns and c in baseline_df.columns), None)
            if col and len(baseline_df[col].values) > 0:
                b_med = np.median(baseline_df[col].values)
                mad = np.median(np.abs(baseline_df[col].values - b_med)) + 1e-9
                f_p95 = np.percentile(df[col].values, 95)
                if f_p95 > b_med: total_deg += ((f_p95 - b_med) / mad) * weight
    return total_deg

def continuous_recovery_score(orig_df, cf_df, baseline_df, target_service, graph, all_services):
    orig_deg = compute_global_degradation(orig_df, baseline_df, all_services)
    cf_deg = compute_global_degradation(cf_df, baseline_df, all_services)
    if orig_deg < 1e-9: return 0.0
    return float(max(0.0, min(1.0, (orig_deg - cf_deg) / orig_deg)))

def infer_graph_from_metrics(services, baseline_df, fault_df):
    graph, anomaly_times = defaultdict(set), {}
    for svc in services:
        col = f"{svc}_istio-latency-99"
        if col in fault_df.columns and not baseline_df.empty:
            bm = np.median(baseline_df[col].values)
            rows = fault_df[fault_df[col] > bm * 2]
            if not rows.empty: anomaly_times[svc] = rows["time"].iloc[0]
    svcs = sorted(anomaly_times, key=anomaly_times.get)
    for i, up in enumerate(svcs):
        for down in svcs[i+1:]:
            if anomaly_times[down] - anomaly_times[up] <= TIME_TOLERANCE: graph[up].add(down)
    return dict(graph)

# ======== 故障分类器 ========
def classify_fault(infl_scores, rec_scores):
    infl_values = sorted(infl_scores.values(), reverse=True)
    rec_values = [rec_scores.get(s, 0) for s in infl_scores.keys()]
    n = len(infl_values)
    if n < 2 or max(infl_values) == 0: return "C"
    index = np.arange(1, n + 1)
    gini = (2 * np.sum(index * np.array(sorted(infl_values)))) / (n * np.sum(infl_values)) - (n + 1) / n
    gini = max(0, min(1, gini))
    top1_infl, top2_infl = infl_values[0], infl_values[1] if n > 1 else 0
    top1_rec, top2_rec = rec_values[0], rec_values[1] if len(rec_values) > 1 else 0
    infl_range = max(infl_values) - min(infl_values) + 1e-9
    rec_range = max(rec_values) - min(rec_values) + 1e-9
    norm_dist = np.sqrt(((top1_infl - top2_infl) / infl_range) ** 2 + ((top1_rec - top2_rec) / rec_range) ** 2) if n > 1 else 10.0
    if n > 2:
        dists = [np.sqrt(((infl_values[i] - infl_values[i+1]) / infl_range) ** 2 + ((rec_values[i] - rec_values[i+1]) / rec_range) ** 2) for i in range(n - 1)]
        avg_dist = np.mean(dists) if dists else 1e-9
    else: avg_dist = 1e-9
    gap_ratio = norm_dist / (avg_dist + 1e-9)
    rec_std = np.std(rec_values) if len(rec_values) > 1 else 0.0
    if gini > 0.7 and gap_ratio > 10: return "A"
    elif gini > 0.7 and rec_std > 0.05: return "B"
    else: return "C"

# ======== Type B: 反事实二元淘汰 ========
def binary_counterfactual_test(fault_df, baseline_df, candidate_A, candidate_B, entry_service, graph, services):
    cf_A = apply_targeted_counterfactual(fault_df, baseline_df, candidate_A, entry_service, graph)
    deg_orig = compute_global_degradation(fault_df, baseline_df, services)
    deg_cfA = compute_global_degradation(cf_A, baseline_df, services)
    cf_B = apply_targeted_counterfactual(fault_df, baseline_df, candidate_B, entry_service, graph)
    deg_cfB = compute_global_degradation(cf_B, baseline_df, services)
    if (deg_orig - deg_cfA) > (deg_orig - deg_cfB) * 1.2: return candidate_A
    elif (deg_orig - deg_cfB) > (deg_orig - deg_cfA) * 1.2: return candidate_B
    else: return None

# ======== 🔥 情绪向量统一调度器（V3 核心）========
def emotion_dispatch(fault_type, infl_scores, rec_scores, sorted_items, emotion_config, 
                     fault_df=None, baseline_df=None, entry=None, graph=None, services=None,
                     log_keyword_scores=None, metrics_spike_scores=None):
    """
    统一的情绪向量调度器，对所有故障类型生效。
    返回：(results, trace_info)
    """
    E_cur, E_anx, E_sat = emotion_config
    trace = {"emotion": emotion_config, "action": "default", "type": fault_type}
    n = len(sorted_items)
    
    if fault_type == "A":
        # Type A: 信号强，情绪只影响"是否多验证一步"
        if E_sat > 7:
            trace["action"] = "fast_converge"
            return [(sorted_items[0][0], rec_scores.get(sorted_items[0][0], 0), sorted_items[0][1])], trace
        elif E_cur > 6:
            trace["action"] = "verify_top3"
            return [(s, rec_scores.get(s, 0), infl) for s, infl in sorted_items[:3]], trace
        else:
            trace["action"] = "standard"
            return [(s, rec_scores.get(s, 0), infl) for s, infl in sorted_items[:5]], trace
    
    elif fault_type == "B":
        # Type B: 双头博弈，高焦虑直接反事实淘汰
        top2 = [s for s, _ in sorted_items[:2]]
        if E_anx > 6 and len(top2) >= 2 and fault_df is not None:
            winner = binary_counterfactual_test(fault_df, baseline_df, top2[0], top2[1], entry, graph, services)
            if winner:
                trace["action"] = "binary_cf_anxiety"
                results = [(winner, rec_scores.get(winner, 0), infl_scores.get(winner, 0))]
                for s, infl in sorted_items[:5]:
                    if s != winner: results.append((s, rec_scores.get(s, 0), infl))
                return results[:5], trace
        elif E_cur > 6:
            trace["action"] = "expand_candidates"
            return [(s, rec_scores.get(s, 0), infl) for s, infl in sorted_items[:min(7, n)]], trace
        else:
            trace["action"] = "standard"
            return [(s, rec_scores.get(s, 0), infl) for s, infl in sorted_items[:5]], trace
    
    else:
        # Type C: 混沌共振，情绪决定搜索策略
        if E_cur > 6:
            # 高好奇：扩大候选集，用全模态重新融合
            trace["action"] = "expand_full_modal"
            combined = {}
            for svc, infl in infl_scores.items():
                log_s = log_keyword_scores.get(svc, 0) if log_keyword_scores else 0
                met_s = metrics_spike_scores.get(svc, 0) if metrics_spike_scores else 0
                combined[svc] = infl + log_s * 10000 + met_s
            expanded = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:min(10, len(combined))]
            return [(s, rec_scores.get(s, 0), infl_scores.get(s, 0)) for s, _ in expanded], trace
        
        elif E_anx > 7 and n > 1:
            # 高焦虑：严格剪枝，直接输出 Top-1
            trace["action"] = "prune_top1"
            top_svc = sorted_items[0][0]
            return [(top_svc, rec_scores.get(top_svc, 0), infl_scores.get(top_svc, 0))], trace
        
        elif E_sat > 6:
            # 高满足：快速收敛到 Top-3
            trace["action"] = "fast_converge"
            return [(s, rec_scores.get(s, 0), infl) for s, infl in sorted_items[:3]], trace
        
        else:
            # 平衡：全模态融合
            trace["action"] = "full_modal_fusion"
            combined = {}
            for svc, infl in infl_scores.items():
                log_s = log_keyword_scores.get(svc, 0) if log_keyword_scores else 0
                met_s = metrics_spike_scores.get(svc, 0) if metrics_spike_scores else 0
                combined[svc] = infl + log_s * 10000 + met_s
            sorted_boosted = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:5]
            return [(s, rec_scores.get(s, 0), infl_scores.get(s, 0)) for s, _ in sorted_boosted], trace

# ======== 主 RCA 流程 ========
def RCA(data_path, inject_time, traces_df=None, logs_df=None, emotion_config=None):
    if emotion_config is None: emotion_config = EMOTION_DEFAULT
    
    baseline_df, fault_df, services = load_rcaeval_data(data_path, inject_time)
    if baseline_df.empty: return [], {}
    entry = next((s for s in services if "frontend" in s.lower() or "gateway" in s.lower()), services[0])
    
    edge_graph = build_weighted_graph_from_traces(traces_df, baseline_df, fault_df, inject_time) if traces_df is not None else {}
    graph = {p: set(c.keys()) for p, c in edge_graph.items()} if edge_graph else infer_graph_from_metrics(services, baseline_df, fault_df)
    
    valid_svcs = [s for s in services if s != entry]
    fatal_services = extract_fatal_log_services(logs_df, inject_time, valid_svcs)
    candidates_raw = detect_anomaly(baseline_df, fault_df, valid_svcs, edge_graph=edge_graph)
    if not candidates_raw and not fatal_services: return [], {}
    
    existing = {s for s, _, _ in candidates_raw}
    for svc in fatal_services:
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
    log_keyword_scores = extract_log_keywords(logs_df, inject_time, current_svcs)
    metrics_spike_scores = extract_metrics_timeseries_spikes(baseline_df, fault_df, inject_time, current_svcs)
    
    sorted_items = sorted(infl_scores.items(), key=lambda x: x[1], reverse=True)
    
    results, trace = emotion_dispatch(
        fault_type, infl_scores, rec_scores, sorted_items, emotion_config,
        fault_df=fault_df, baseline_df=baseline_df, entry=entry, graph=graph, services=services,
        log_keyword_scores=log_keyword_scores, metrics_spike_scores=metrics_spike_scores
    )
    
    trace["fault_type"] = fault_type
    return results, trace
ENDOFFILE
echo "rca_module_v3.py 已创建"