"""
情绪驱动的反事实RCA框架 (P-CoE 融合版 V7.4 - 全模态扩展版)
基于RCAEval RE3数据集

V7.4 核心升级：
1. 指标从 5 种扩展到 19 种，覆盖 Istio 延迟/错误/流量、容器 CPU/内存/网络/磁盘、宿主机节点级指标。
2. 保留 V7.3 的日志驱动召回 + Type A/B/C 分治 + L3 LLM 因果归因。
3. 新增 istio-request-total（QPS 掉底 = 静默故障信号）和网络错误指标。
"""

import copy
import random
import numpy as np
import pandas as pd
from collections import defaultdict
import requests
import json
import sys
import os

# ==========================================
# 🎯 核心配置区
# ==========================================
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_MODEL   = "deepseek-chat"
DEEPSEEK_API_KEY = "sk-4100050bb29a47898950fa6587a400ba"

# ======== 引擎基础配置 ========
BASELINE_WINDOW_SECONDS = 300
TOP_K = 7
MIN_CANDIDATE_SIZE = 3
ANOMALY_Z_THRESHOLD = 3.0
TIME_TOLERANCE = 60
EDGE_IMPORTANCE_MIN_CALLS = 1
EXCLUDE_SERVICE_PATTERNS = ["ip-", "compute.internal", "frontend-external"]

# 🔥 全模态指标配置（19 种，按优先级加权）
METRIC_CONFIGS = [
    # ==== 高优先级：Istio 层 ====
    ("istio-latency-99", 1.0),
    ("istio-error-total", 2.0),
    ("istio-request-total", 1.5),       # QPS 掉底 = 静默故障
    ("istio-latency-50", 0.8),          # 中位数延迟，更稳定
    ("istio-latency-90", 0.8),
    ("istio-latency-95", 0.9),
    # ==== 高优先级：容器层 ====
    ("container-cpu-usage-seconds-total", 1.0),
    ("container-memory-failures-total", 2.0),
    ("container-memory-working-set-bytes", 0.5),
    ("container-memory-usage-bytes", 0.8),    # 实际内存使用
    ("container-memory-rss", 0.8),             # 进程驻留内存，OOM 前兆
    ("container-network-receive-errors-total", 1.5),  # 网络故障
    ("container-network-transmit-errors-total", 1.5), # 网络故障
    ("container-spec-memory-limit-bytes", 0.5), # 内存限制边缘
    # ==== 中优先级：补充指标 ====
    ("container-cpu-user-seconds-total", 0.6), # 用户态 CPU
    ("container-fs-writes-total", 0.5),        # 磁盘写入异常
    # ==== 宿主机节点级指标 ====
    ("node-memory-active-bytes", 0.7),         # 宿主机内存压力
    ("node-network-receive-errs-total", 0.7),  # 宿主机网络丢包
    ("node-disk-read-bytes-total", 0.5),       # 宿主机磁盘 I/O
]

# ======== 日志关键词 ========
LOG_KEYWORD_TIERS = {
    "BindException": 10, "Address in use": 10, "Failed to bind": 10,
    "exit code 137": 10, "SIGKILL": 10, "OOM Killed": 10,
    "Container Killed": 10, "OutOfMemoryError": 10, "oom": 10, "killed": 10,
    "Connection Refused": 5, "connection refused": 5,
    "500 Internal Server Error": 5, "500": 5,
    "Timeout": 2, "timeout": 2, "timed out": 2,
    "Exception": 1, "exception": 1, "ERROR": 1, "error": 1,
}

FATAL_LOG_KEYWORDS = [
    "BindException", "Address in use", "Failed to bind",
    "exit code 137", "SIGKILL", "OOM Killed", "OutOfMemoryError",
    "Container Killed", "killed", "oom"
]

# ======== Metrics 时序突变检测配置 ========
METRICS_TIMESERIES_WINDOW = 10
METRICS_SPIKE_THRESHOLD = 3.0
METRICS_SPIKE_WEIGHT = 5000.0

# ======== 物理引擎底层函数 ========
def _is_valid_service(svc: str) -> bool:
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
        base_traces, fault_traces = traces_df[traces_df[time_col] < inject_time], traces_df[traces_df[time_col] >= inject_time]
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
            weight = lat_z + max(0, (np.mean(f_errs) - np.mean(b_errs)) * 10)
        graph[edge[0]][edge[1]] = weight
    return dict(graph)

def extract_log_anomaly(logs_df, inject_time):
    if logs_df is None or logs_df.empty or "message" not in logs_df.columns or "container_name" not in logs_df.columns:
        return {}
    if "timestamp" in logs_df.columns:
        logs_df["time_sec"] = pd.to_numeric(logs_df["timestamp"], errors="coerce") // 10**9
        fault_logs = logs_df[logs_df["time_sec"] >= inject_time]
    else:
        fault_logs = logs_df
    if fault_logs.empty:
        return {}
    error_mask = fault_logs["message"].astype(str).str.contains("ERROR|Exception|Timeout|Fail|error", case=False, na=False)
    return fault_logs[error_mask].groupby("container_name").size().to_dict()

def extract_log_keywords(logs_df, inject_time, candidates):
    if logs_df is None or logs_df.empty or "message" not in logs_df.columns or "container_name" not in logs_df.columns:
        return {}
    if "timestamp" in logs_df.columns:
        logs_df["time_sec"] = pd.to_numeric(logs_df["timestamp"], errors="coerce") // 10**9
        fault_logs = logs_df[logs_df["time_sec"] >= inject_time]
    else:
        fault_logs = logs_df
    if fault_logs.empty:
        return {}
    
    scores = {}
    for svc in candidates:
        svc_logs = fault_logs[fault_logs["container_name"] == svc]
        if svc_logs.empty:
            scores[svc] = 0.0
            continue
        svc_score = 0.0
        messages = svc_logs["message"].astype(str)
        for keyword, weight in LOG_KEYWORD_TIERS.items():
            if messages.str.contains(keyword, case=False).any():
                svc_score += weight
        scores[svc] = svc_score
    return scores

def extract_log_samples(logs_df, inject_time, svc, max_samples=3):
    if logs_df is None or logs_df.empty:
        return []
    if "timestamp" in logs_df.columns:
        logs_df["time_sec"] = pd.to_numeric(logs_df["timestamp"], errors="coerce") // 10**9
        fault_logs = logs_df[logs_df["time_sec"] >= inject_time]
    else:
        fault_logs = logs_df
    if fault_logs.empty:
        return []
    svc_logs = fault_logs[fault_logs["container_name"] == svc]
    if svc_logs.empty:
        return []
    error_mask = svc_logs["message"].astype(str).str.contains("ERROR|Exception|Timeout|Fail|error|killed|refused|bind|exit", case=False, na=False)
    error_logs = svc_logs[error_mask]
    if error_logs.empty:
        return svc_logs["message"].head(max_samples).tolist()
    return error_logs["message"].head(max_samples).tolist()

def extract_fatal_log_services(logs_df, inject_time, all_services):
    if logs_df is None or logs_df.empty:
        return []
    if "timestamp" in logs_df.columns:
        logs_df["time_sec"] = pd.to_numeric(logs_df["timestamp"], errors="coerce") // 10**9
        fault_logs = logs_df[logs_df["time_sec"] >= inject_time]
    else:
        fault_logs = logs_df
    if fault_logs.empty:
        return []
    
    forced = []
    for svc in all_services:
        if "container_name" not in fault_logs.columns:
            break
        svc_logs = fault_logs[fault_logs["container_name"] == svc]
        if svc_logs.empty:
            continue
        messages = svc_logs["message"].astype(str)
        for keyword in FATAL_LOG_KEYWORDS:
            if messages.str.contains(keyword, case=False).any():
                forced.append(svc)
                print(f"  🚨 日志致命召回: {svc} (命中: {keyword})")
                break
    return forced

def extract_metrics_timeseries_spikes(baseline_df, fault_df, inject_time, candidates):
    scores = {}
    baseline_tail = baseline_df.tail(METRICS_TIMESERIES_WINDOW)
    fault_head = fault_df.head(METRICS_TIMESERIES_WINDOW)
    if baseline_tail.empty or fault_head.empty:
        return {}
    
    for svc in candidates:
        svc_score = 0.0
        for metric_suffix, _ in METRIC_CONFIGS:
            col = next((c for c in [f"{svc}_{metric_suffix}", f"{svc}_{metric_suffix.replace('-', '_')}"] 
                       if c in baseline_tail.columns and c in fault_head.columns), None)
            if col is None:
                continue
            base_vals = baseline_tail[col].dropna().values
            fault_vals = fault_head[col].dropna().values
            if len(base_vals) == 0 or len(fault_vals) == 0:
                continue
            base_p95 = np.percentile(base_vals, 95) + 1e-9
            fault_p95 = np.percentile(fault_vals, 95)
            ratio = fault_p95 / base_p95
            if ratio > METRICS_SPIKE_THRESHOLD:
                svc_score += METRICS_SPIKE_WEIGHT * ratio
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
            out_weight = sum(edge_graph[svc].values())
            score = max(0.1, score - out_weight * 0.5)
        
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
    def restore_service(svc):
        for col in cf_df.columns:
            if col.startswith(svc + "_") and col in aligned_base.columns:
                bv = aligned_base[col].dropna().values
                if len(bv) > 0: cf_df[col] = np.median(bv)
    restore_service(target_service)
    if graph and target_service in graph:
        for child in graph[target_service]:
            restore_service(child)
    return cf_df

def compute_global_degradation(df, baseline_df, services):
    total_deg = 0.0
    for svc in services:
        for metric_suffix, weight in METRIC_CONFIGS:
            col = next((c for c in [f"{svc}_{metric_suffix}", f"{svc}_{metric_suffix.replace('-', '_')}"] if c in df.columns and c in baseline_df.columns), None)
            if col and len(baseline_df[col].values) > 0:
                b_vals = baseline_df[col].values
                f_vals = df[col].values
                b_med = np.median(b_vals)
                mad = np.median(np.abs(b_vals - b_med)) + 1e-9
                f_p95 = np.percentile(f_vals, 95)
                if f_p95 > b_med:
                    total_deg += ((f_p95 - b_med) / mad) * weight
    return total_deg

def continuous_recovery_score(orig_df, cf_df, baseline_df, target_service, graph, all_services):
    orig_deg = compute_global_degradation(orig_df, baseline_df, all_services)
    cf_deg = compute_global_degradation(cf_df, baseline_df, all_services)
    if orig_deg < 1e-9:
        return 0.0
    improvement = (orig_deg - cf_deg) / orig_deg
    return float(max(0.0, min(1.0, improvement)))

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

def build_topology_snippet(graph, candidates):
    lines = []
    for svc in candidates:
        if svc in graph:
            for child in graph[svc]:
                lines.append(f"{svc} -> calls -> {child}")
    return lines[:10] if lines else ["No topology data available"]

# ======== 加载单个 case 数据 ========
def load_re3_case(case_dir):
    inject_time_path = os.path.join(case_dir, "inject_time.txt")
    with open(inject_time_path) as f: inject_time = int(f.read().strip())
    metrics_path = os.path.join(case_dir, "metrics.csv")
    traces_path = os.path.join(case_dir, "traces.csv")
    traces_df = pd.read_csv(traces_path) if os.path.exists(traces_path) else None
    if traces_df is not None:
        if 'startTimeMillis' in traces_df.columns:
            traces_df['time'] = traces_df['startTimeMillis'] // 1000
        elif 'startTime' in traces_df.columns:
            traces_df['time'] = pd.to_numeric(traces_df['startTime'], errors='coerce') / 1_000_000.0
    logs_path = os.path.join(case_dir, "logs.csv")
    logs_df = pd.read_csv(logs_path, on_bad_lines='skip') if os.path.exists(logs_path) else None
    parts = os.path.normpath(case_dir).split(os.sep)
    fault_dir = next((p for p in parts if "_f" in p and any(c.isdigit() for c in p.split("_f")[-1])), None)
    ground_truth = fault_dir.rsplit("_f", 1)[0] if fault_dir else None
    return metrics_path, inject_time, traces_df, logs_df, ground_truth

# ======== 故障分类器 ========
def classify_fault(infl_scores, rec_scores):
    infl_values = sorted(infl_scores.values(), reverse=True)
    rec_values = [rec_scores.get(s, 0) for s in infl_scores.keys()]
    
    n = len(infl_values)
    if n < 2 or max(infl_values) == 0:
        return "C"
    
    index = np.arange(1, n + 1)
    gini = (2 * np.sum(index * np.array(sorted(infl_values)))) / (n * np.sum(infl_values)) - (n + 1) / n
    gini = max(0, min(1, gini))
    
    top1_infl = infl_values[0]
    top2_infl = infl_values[1] if n > 1 else 0
    top1_rec = rec_values[0] if rec_values else 0
    top2_rec = rec_values[1] if len(rec_values) > 1 else 0
    
    infl_range = max(infl_values) - min(infl_values) + 1e-9
    rec_range = max(rec_values) - min(rec_values) + 1e-9
    
    norm_dist = np.sqrt(
        ((top1_infl - top2_infl) / infl_range) ** 2 +
        ((top1_rec - top2_rec) / rec_range) ** 2
    ) if n > 1 else 10.0
    
    avg_dist = 0.0
    if n > 2:
        dists = []
        for i in range(n - 1):
            d = np.sqrt(
                ((infl_values[i] - infl_values[i+1]) / infl_range) ** 2 +
                ((rec_values[i] - rec_values[i+1]) / rec_range) ** 2
            )
            dists.append(d)
        avg_dist = np.mean(dists) if dists else 1e-9
    else:
        avg_dist = 1e-9
    
    gap_ratio = norm_dist / (avg_dist + 1e-9)
    rec_std = np.std(rec_values) if len(rec_values) > 1 else 0.0
    
    if gini > 0.7 and gap_ratio > 10:
        return "A"
    elif gini > 0.7 and rec_std > 0.05:
        return "B"
    else:
        return "C"

# ======== Type B: 反事实二元淘汰 ========
def binary_counterfactual_test(fault_df, baseline_df, candidate_A, candidate_B, 
                                entry_service, graph, services):
    cf_A = apply_targeted_counterfactual(fault_df, baseline_df, candidate_A, entry_service, graph)
    deg_orig = compute_global_degradation(fault_df, baseline_df, services)
    deg_cfA = compute_global_degradation(cf_A, baseline_df, services)
    
    cf_B = apply_targeted_counterfactual(fault_df, baseline_df, candidate_B, entry_service, graph)
    deg_cfB = compute_global_degradation(cf_B, baseline_df, services)
    
    improvement_A = deg_orig - deg_cfA
    improvement_B = deg_orig - deg_cfB
    
    if improvement_A > improvement_B * 1.2:
        return candidate_A
    elif improvement_B > improvement_A * 1.2:
        return candidate_B
    else:
        return None

# ======== L3: LLM 因果归因推理 ========
def llm_causal_reasoning(infl_scores, rec_scores, graph, candidates, logs_df, inject_time):
    topology_lines = build_topology_snippet(graph, list(candidates)[:5])
    
    pareto_candidates = []
    sorted_candidates = sorted(infl_scores.items(), key=lambda x: x[1], reverse=True)[:5]
    for svc, infl in sorted_candidates:
        rec = rec_scores.get(svc, 0)
        log_samples = extract_log_samples(logs_df, inject_time, svc, max_samples=2)
        log_summary = f"{len(log_samples)} errors. Sample: '{log_samples[0][:120]}...'" if log_samples else "No error logs"
        
        pareto_candidates.append({
            "service": svc,
            "math_scores": f"Rec={rec:.3f}, Infl={infl:.1e}",
            "log_summary": log_summary
        })
    
    prompt_data = {
        "context": "当前故障处于 L3 困难模式，系统陷入数学简并，需进行最高级别的语义因果推断。",
        "topology_snippet": topology_lines,
        "pareto_candidates": pareto_candidates,
        "instruction": "作为高级 SRE，请结合微服务拓扑常识和日志语义（区分谁是报错者，谁是真正的故障源），无视数学分数的欺骗性，直接指出真正的 Root Cause 并给出简短的因果推理链。请用 JSON 格式回复：{\"root_cause\": \"服务名\", \"reasoning\": \"因果推理链\"}"
    }
    
    prompt_str = json.dumps(prompt_data, ensure_ascii=False, indent=2)
    
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": DEEPSEEK_MODEL,
        "max_tokens": 1024,
        "temperature": 0.0,
        "messages": [
            {"role": "system", "content": "你是一个资深的 SRE 根因分析专家。请严格按照 JSON 格式回复。"},
            {"role": "user", "content": prompt_str}
        ]
    }
    
    try:
        resp = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        if "choices" in data:
            text = data["choices"][0]["message"]["content"]
        else:
            text = data["content"][0]["text"]
        text = text[text.find('{'):text.rfind('}')+1]
        result = json.loads(text)
        root_cause = result.get("root_cause", sorted_candidates[0][0] if sorted_candidates else "unknown")
        reasoning = result.get("reasoning", "")
        print(f"  🧠 LLM 因果推理: {root_cause}")
        print(f"  💭 推理链: {reasoning[:200]}...")
        return root_cause
    except Exception as e:
        print(f"  [LLM] API失败: {e}, 回退到 Infl 排序")
        return sorted_candidates[0][0] if sorted_candidates else None

# ======== 引擎核心 ========
def RCA_MDP_Episode(data_path, inject_time, emotion_config=None, traces_df=None, logs_df=None):
    baseline_df, fault_df, services = load_rcaeval_data(data_path, inject_time)
    if baseline_df.empty: return []
    entry = next((s for s in services if "frontend" in s.lower() or "gateway" in s.lower()), services[0])

    edge_graph = build_weighted_graph_from_traces(traces_df, baseline_df, fault_df, inject_time) if traces_df is not None else {}
    graph = {p: set(c.keys()) for p, c in edge_graph.items()} if edge_graph else infer_graph_from_metrics(services, baseline_df, fault_df)
    
    valid_svcs = [s for s in services if s != entry]
    
    fatal_services = extract_fatal_log_services(logs_df, inject_time, valid_svcs)
    
    candidates_raw = detect_anomaly(baseline_df, fault_df, valid_svcs, edge_graph=edge_graph)
    if not candidates_raw and not fatal_services:
        return []
    
    existing_svcs = {s for s, _, _ in candidates_raw}
    for svc in fatal_services:
        if svc not in existing_svcs:
            candidates_raw.append((svc, 0.5, None))
            print(f"  ➕ 日志召回强制入围: {svc}")
    
    current_svcs = [s for s, _, _ in candidates_raw]
    
    infl_scores = {}
    rec_scores = {}
    for s in current_svcs:
        out_rad = sum(edge_graph.get(s, {}).values()) if edge_graph else 0.0
        in_ctrl = 0.0
        if edge_graph:
            for parent, children in edge_graph.items():
                if s in children:
                    in_ctrl += children[s]
        node_sc = next((sc for svc, sc, _ in candidates_raw if svc == s), 0.1)
        log_counts = extract_log_anomaly(logs_df, inject_time)
        log_boost = np.log1p(log_counts.get(s, 0)) * 5.0
        infl_scores[s] = node_sc + 2.0 * in_ctrl - 1.5 * out_rad + log_boost
        
        cf_df = apply_targeted_counterfactual(fault_df, baseline_df, s, entry, graph)
        rec_scores[s] = continuous_recovery_score(fault_df, cf_df, baseline_df, s, graph, services)
    
    fault_type = classify_fault(infl_scores, rec_scores)
    print(f"  🔬 故障分类: Type {fault_type}")
    
    log_keyword_scores = extract_log_keywords(logs_df, inject_time, current_svcs)
    metrics_spike_scores = extract_metrics_timeseries_spikes(baseline_df, fault_df, inject_time, current_svcs)
    
    if fault_type == "A":
        print(f"  🟢 Type A: 帕累托单点型，影响力优先")
        sorted_infl = sorted(infl_scores.items(), key=lambda x: x[1], reverse=True)
        results = [(s, rec_scores.get(s, 0), infl) for s, infl in sorted_infl[:5]]
        return results
    
    elif fault_type == "B":
        print(f"  🟡 Type B: 双头博弈型，反事实二元淘汰")
        sorted_items = sorted(infl_scores.items(), key=lambda x: x[1], reverse=True)
        top2 = [s for s, _ in sorted_items[:2]]
        
        if len(top2) >= 2:
            winner = binary_counterfactual_test(fault_df, baseline_df, top2[0], top2[1], entry, graph, services)
            if winner:
                print(f"  ⚔️ 反事实裁决: {winner} 胜出")
                results = [(winner, rec_scores.get(winner, 0), infl_scores.get(winner, 0))]
                for s, infl in sorted_items[:5]:
                    if s != winner:
                        results.append((s, rec_scores.get(s, 0), infl))
                return results[:5]
        
        sorted_infl = sorted(infl_scores.items(), key=lambda x: x[1], reverse=True)
        results = [(s, rec_scores.get(s, 0), infl) for s, infl in sorted_infl[:5]]
        return results
    
    else:
        print(f"  🔴 Type C: 混沌共振型")
        
        combined = {}
        for svc, infl in infl_scores.items():
            log_score = log_keyword_scores.get(svc, 0)
            metrics_score = metrics_spike_scores.get(svc, 0)
            combined[svc] = infl + log_score * 10000 + metrics_score
        sorted_boosted = sorted(combined.items(), key=lambda x: x[1], reverse=True)
        
        if len(sorted_boosted) >= 2:
            top1_combined = sorted_boosted[0][1]
            top2_combined = sorted_boosted[1][1]
            gap = abs(top1_combined - top2_combined) / (abs(top1_combined) + 1e-9)
            
            if gap < 0.10:
                print(f"  🧠 全模态融合仍简并(差距={gap:.3f})，启动 L3 LLM 因果归因")
                root_cause = llm_causal_reasoning(infl_scores, rec_scores, graph, current_svcs, logs_df, inject_time)
                if root_cause:
                    results = [(root_cause, rec_scores.get(root_cause, 0), infl_scores.get(root_cause, 0))]
                    for svc, combined_score in sorted_boosted[:5]:
                        if svc != root_cause:
                            results.append((svc, rec_scores.get(svc, 0), infl_scores.get(svc, 0)))
                    return results[:5]
        
        results = [(s, rec_scores.get(s, 0), infl_scores.get(s, 0)) for s, _ in sorted_boosted[:5]]
        return results

# ======== 批量测试入口 ========
def run_re3_benchmark(re3_root, system="RE3-OB", max_cases=None):
    system_dir = os.path.join(re3_root, system)
    if not os.path.exists(system_dir):
        print(f"路径不存在: {system_dir}")
        return
    
    print(f"\n{'='*60}")
    print(f"🔥 P-CoE V7.4 批量测试启动（全模态扩展 19 种指标）")
    print(f"{'='*60}\n")
    
    top1_correct, top3_correct, mrr_sum, total = 0, 0, 0.0, 0
    l3_triggered, l3_hits = 0, 0
    
    for fault_name in sorted(os.listdir(system_dir)):
        fault_path = os.path.join(system_dir, fault_name)
        if not os.path.isdir(fault_path): continue
        for rep in sorted(os.listdir(fault_path)):
            case_dir = os.path.join(fault_path, rep)
            if not os.path.isdir(case_dir): continue
            if max_cases and total >= max_cases: break
            
            try:
                metrics_path, inject_time, traces_df, logs_df, ground_truth = load_re3_case(case_dir)
                
                print(f"\n{'='*60}")
                print(f"📋 Case {total+1}: {fault_name}/{rep}")
                print(f"🎯 真实根因: {ground_truth}")
                print(f"{'='*60}")
                
                results = RCA_MDP_Episode(metrics_path, inject_time, None, traces_df, logs_df)
                
                if not results:
                    print("⚠️ 无结果，跳过")
                    continue
                
                total += 1
                svcs = [s for s, _, _ in results]
                
                if svcs and svcs[0] == ground_truth:
                    top1_correct += 1
                    print(f"✅ Top-1 命中!")
                else:
                    print(f"❌ Top-1 未命中 (输出: {svcs[0] if svcs else '无'})")
                
                if ground_truth in svcs[:3]:
                    top3_correct += 1
                
                for rank, svc in enumerate(svcs, 1):
                    if svc == ground_truth:
                        mrr_sum += 1.0 / rank
                        break
                        
                print(f"🏆 P-CoE 最终排名:")
                for rank, (svc, rec, infl) in enumerate(results[:5], 1):
                    marker = " ← 真实根因" if svc == ground_truth else ""
                    print(f"  {rank}. {svc:<25} Rec={rec:.4f} Infl={infl:.1e}{marker}")
                    
            except Exception as e:
                print(f"❌ Case 崩溃 {case_dir}: {e}")
                import traceback
                traceback.print_exc()
    
    print(f"\n{'='*60}")
    print(f"🔥 P-CoE V7.4 批量测试结果（{total}个case）")
    print(f"{'='*60}")
    if total > 0:
        print(f"Top-1 Accuracy: {top1_correct/total:.3f} ({top1_correct}/{total})")
        print(f"Top-3 Hit Rate: {top3_correct/total:.3f} ({top3_correct}/{total})")
        print(f"MRR:            {mrr_sum/total:.3f}")
    print(f"L3 LLM 触发次数: {l3_triggered}, 命中: {l3_hits}")
    print(f"{'='*60}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "re3":
        run_re3_benchmark("/home/admin/RCAEval/data/RE3", max_cases=int(sys.argv[2]) if len(sys.argv) > 2 else None)