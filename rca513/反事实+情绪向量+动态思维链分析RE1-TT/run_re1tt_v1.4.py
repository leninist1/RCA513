"""
P-CoE V8.3 - RE1-TT 纯指标版（故障类型条件化 Scoring + LOSS 专用评分 + 关L3）
保持全模态融合框架不变

V8.3 核心升级：
1. 故障类型判别：根据候选集指标指纹自动识别 CPU/MEM/DELAY/LOSS
2. LOSS 型专用评分工式：NetworkFingerprint + Earliest + Locality - HubVictimPenalty
3. 传播贡献降权为 0.25，改为加法叠加
4. 关闭 L3 LLM
5. 保留 V8.2 的 SoftCF, persistence, IP过滤等所有优化
"""

import copy
import random
import numpy as np
import pandas as pd
from collections import defaultdict
import json
import sys
import os

# ======== 引擎基础配置 ========
BASELINE_WINDOW_SECONDS = 300
TOP_K = 7
ANOMALY_Z_THRESHOLD = 3.0
CAUSAL_WINDOW_SECONDS = 15
SOFT_CF_BETA = 0.7
RECOVERY_GAMMA = 0.5
PROPAGATION_WEIGHT = 0.25          # 🔥 传播贡献降权

# ======== 全模态指标配置（40 种）========
METRIC_CONFIGS = [
    # Istio 层
    ("istio-latency-99", 1.5), ("istio-latency-95", 1.2), ("istio-latency-90", 1.0),
    ("istio-latency-50", 0.8), ("istio-error-total", 2.5), ("istio-request-total", 1.5),
    ("istio-bytes-99", 0.5), ("istio-bytes-95", 0.5), ("istio-bytes-90", 0.5), ("istio-bytes-50", 0.5),
    # 容器 CPU/内存
    ("container-cpu-usage-seconds-total", 1.5), ("container-cpu-user-seconds-total", 0.8),
    ("container-cpu-system-seconds-total", 0.6), ("container-memory-failures-total", 3.0),
    ("container-memory-working-set-bytes", 0.5), ("container-memory-usage-bytes", 0.8),
    ("container-memory-rss", 0.8), ("container-memory-max-usage-bytes", 0.8),
    ("container-memory-swap", 0.8), ("container-memory-cache", 0.4),
    ("container-memory-mapped-file", 0.4), ("container-spec-cpu-quota", 0.5),
    ("container-spec-memory-limit-bytes", 0.5),
    # 容器磁盘
    ("container-fs-writes-total", 0.5), ("container-fs-writes-bytes-total", 0.5),
    ("container-fs-reads-total", 0.5), ("container-fs-reads-bytes-total", 0.5),
    # 宿主机节点级
    ("node-memory-active-bytes", 0.7), ("node-memory-inactive-bytes", 0.7),
    ("node-cpu-seconds-total", 0.6), ("node-network-receive-errs-total", 1.0),
    ("node-network-receive-drop-total", 1.0), ("node-network-transmit-errs-total", 1.0),
    ("node-network-transmit-drop-total", 1.0), ("node-disk-read-bytes-total", 0.5),
    ("node-disk-written-bytes-total", 0.5), ("node-disk-reads-completed-total", 0.4),
    ("node-disk-writes-completed-total", 0.4), ("node-network-receive-packets-total", 0.3),
    ("node-network-transmit-packets-total", 0.3),
]

# ======== 物理引擎底层函数 ========
def _is_valid_service(svc: str) -> bool:
    """过滤 IP 节点和 Node Exporter"""
    if any(x in svc for x in ["ip-", "192.", "192-", "10.", "10-"]):
        return False
    if len(svc) > 3 and svc[:3].replace("-", "").replace(".", "").isdigit():
        return False
    return True

def load_re1tt_data(data_path, inject_time):
    df = pd.read_csv(data_path)
    if "time.1" in df.columns: df = df.drop(columns=["time.1"])
    df = df.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
    baseline_df = df[df["time"] < inject_time].tail(BASELINE_WINDOW_SECONDS // 15)
    fault_df = df[df["time"] >= inject_time]
    services = set()
    for col in df.columns:
        if col == "time": continue
        parts = col.split("_", 1)
        if len(parts) == 2:
            svc = parts[0]
            if _is_valid_service(svc): services.add(svc)
    return baseline_df, fault_df, sorted(services)

def detect_anomaly_with_persistence(baseline_df, fault_df, services):
    scores = []
    for svc in services:
        score = 0.0
        first_time = None
        spiked = []
        for metric_suffix, weight in METRIC_CONFIGS:
            col = next((c for c in [f"{svc}_{metric_suffix}", f"{svc}_{metric_suffix.replace('-', '_')}"] 
                       if c in fault_df.columns and c in baseline_df.columns), None)
            if col and len(baseline_df[col].values) > 0:
                b_vals, f_vals = baseline_df[col].values, fault_df[col].values
                b_med, mad = np.median(b_vals), np.median(np.abs(b_vals - np.median(b_vals))) + 1e-9
                threshold = b_med + ANOMALY_Z_THRESHOLD * mad
                
                # Persistence
                anomaly_mask = f_vals > threshold
                persistence_ratio = np.sum(anomaly_mask) / len(f_vals)
                
                z = max(0, np.percentile(f_vals, 95) - b_med) / mad
                if z > ANOMALY_Z_THRESHOLD:
                    score += z * weight * (0.5 + 0.5 * persistence_ratio)
                    
                    rows = fault_df[anomaly_mask]
                    if not rows.empty:
                        t = rows["time"].iloc[0]
                        if first_time is None or t < first_time:
                            first_time = t
                        short_name = metric_suffix.replace("container-", "").replace("istio-", "").replace("node-", "")
                        spiked.append(f"{short_name}(z={z:.0f})")
        if score > 0:
            scores.append((svc, score, first_time, spiked[:5]))
    return sorted(scores, key=lambda x: x[1], reverse=True)[:TOP_K]

def compute_propagation_gain(candidates, all_times, infl_scores):
    propagation_scores = {}
    for svc, _, first_time, _ in candidates:
        if first_time is None:
            propagation_scores[svc] = 0
            continue
        gain = 0
        for other_svc, other_time in all_times.items():
            if other_svc == svc or other_time is None:
                continue
            dt = other_time - first_time
            if 0 < dt < CAUSAL_WINDOW_SECONDS:
                gain += infl_scores.get(other_svc, 0) / (dt + 1)
        propagation_scores[svc] = gain
    return propagation_scores

def apply_soft_counterfactual(fault_df, baseline_df, target_service):
    cf_df = copy.deepcopy(fault_df)
    aligned_base = baseline_df.iloc[-len(fault_df):] if len(baseline_df) >= len(fault_df) else baseline_df
    for col in cf_df.columns:
        if col.startswith(target_service + "_") and col in aligned_base.columns:
            bv = aligned_base[col].dropna().values
            if len(bv) > 0:
                cf_df[col] = fault_df[col] * (1 - SOFT_CF_BETA) + np.median(bv) * SOFT_CF_BETA
    return cf_df

def compute_global_degradation(df, baseline_df, services):
    total_deg = 0.0
    for svc in services:
        for metric_suffix, weight in METRIC_CONFIGS:
            col = next((c for c in [f"{svc}_{metric_suffix}", f"{svc}_{metric_suffix.replace('-', '_')}"] 
                       if c in df.columns and c in baseline_df.columns), None)
            if col and len(baseline_df[col].values) > 0:
                b_vals, f_vals = baseline_df[col].values, df[col].values
                b_med, mad = np.median(b_vals), np.median(np.abs(b_vals - np.median(b_vals))) + 1e-9
                f_p95 = np.percentile(f_vals, 95)
                if f_p95 > b_med: total_deg += ((f_p95 - b_med) / mad) * weight
    return total_deg

def soft_recovery_score(orig_df, cf_df, baseline_df, target_service, all_services):
    orig_deg = compute_global_degradation(orig_df, baseline_df, all_services)
    cf_deg = compute_global_degradation(cf_df, baseline_df, all_services)
    if orig_deg < 1e-9: return 0.0
    return float(max(0.0, min(1.0, (orig_deg - cf_deg) / orig_deg)))

# ======== 加载单个 case 数据 ========
def load_re1tt_case(case_dir):
    inject_time_path = os.path.join(case_dir, "inject_time.txt")
    with open(inject_time_path) as f: inject_time = int(f.read().strip())
    data_path = os.path.join(case_dir, "data.csv")
    parts = os.path.normpath(case_dir).split(os.sep)
    fault_dir = parts[-2]
    ground_truth = fault_dir.rsplit("_", 1)[0]
    return data_path, inject_time, ground_truth

# ======== 🔥 故障类型条件化判别 ========
def detect_fault_mode(candidates):
    """
    根据 Top-5 候选的指标指纹判断故障类型：CPU / MEM / DELAY / LOSS
    """
    if not candidates:
        return "UNKNOWN"
    
    # 合并所有指标的 Z-score 名称
    all_spiked = []
    for _, _, _, spiked in candidates:
        for s in spiked:
            # 解析指标短名，如 "latency-99(z=24650000)"
            name = s.split("(")[0]
            all_spiked.append(name)
    
    spike_str = " ".join(all_spiked).lower()
    
    # 网络层指纹：bytes, drop, retransmit, network-receive/transmit
    network_keywords = ["bytes", "drop", "retransmit", "network"]
    network_count = sum(1 for kw in network_keywords if kw in spike_str)
    
    # Memory 指纹
    mem_keywords = ["memory", "mem"]
    mem_count = sum(1 for kw in mem_keywords if kw in spike_str)
    
    # CPU 指纹
    cpu_keywords = ["cpu"]
    cpu_count = sum(1 for kw in cpu_keywords if kw in spike_str)
    
    # 延迟指纹
    latency_keywords = ["latency"]
    latency_count = sum(1 for kw in latency_keywords if kw in spike_str)
    
    total = max(1, network_count + mem_count + cpu_count + latency_count)
    
    # 🔥 分类逻辑：LOSS 优先（网络指标 + 最早报警节点有网络异常）
    if network_count >= 2:
        # 检查最早报警的节点是否包含网络异常
        earliest = candidates[0]  # 已按分数排序，但不一定是时间最早的
        # 找到时间最早的
        earliest_svc = None
        earliest_time = None
        earliest_spiked = []
        for svc, sc, t, sp in candidates:
            if t is not None and (earliest_time is None or t < earliest_time):
                earliest_time = t
                earliest_svc = svc
                earliest_spiked = sp
        if earliest_spiked:
            earliest_str = " ".join(earliest_spiked).lower()
            if any(kw in earliest_str for kw in network_keywords):
                return "LOSS"
    
    if mem_count / total > 0.3:
        return "MEM"
    if cpu_count / total > 0.3:
        return "CPU"
    if latency_count / total > 0.5:
        return "DELAY"
    
    return "DELAY"  # 默认

# ======== LOSS 专用评分 ========
def compute_loss_scores(candidates, infl_scores, all_times, services, baseline_df, fault_df):
    """
    LOSS 型专用评分：
    Score_loss = 0.45 * NetworkFingerprint + 0.25 * Earliest + 0.15 * Locality - 0.25 * HubVictimPenalty
    """
    if not candidates:
        return {}
    
    # 找到最早报警时间
    min_time = min(t for _, _, t, _ in candidates if t is not None)
    
    # 计算 NetworkFingerprint：bytes, drop, network 相关的指标 Z-score 之和
    network_metric_names = []
    for metric_suffix, _ in METRIC_CONFIGS:
        if any(kw in metric_suffix.lower() for kw in ["bytes", "drop", "retransmit", "network"]):
            network_metric_names.append(metric_suffix)
    
    network_scores = {}
    for svc, _, _, spiked in candidates:
        nf_score = 0.0
        # 直接从 spiked 中统计网络相关指标
        for s in spiked:
            name = s.split("(")[0].lower()
            if any(kw in name for kw in ["bytes", "drop", "retransmit", "network"]):
                # 提取 z-score 数值
                try:
                    z_val = float(s.split("z=")[1].rstrip(")"))
                    nf_score += z_val
                except:
                    pass
        network_scores[svc] = nf_score
    
    # Earliest：最先报警的节点得满分 1.0，其余按时间衰减
    time_range = max(1, max(t for _, _, t, _ in candidates if t is not None) - min_time)
    earliest_scores = {}
    for svc, _, t, _ in candidates:
        if t is None:
            earliest_scores[svc] = 0.0
        else:
            norm = (t - min_time) / time_range
            earliest_scores[svc] = max(0.0, 1.0 - norm)  # 最早=1.0，最晚=0.0
    
    # Locality：自身异常强度 / 所有候选的平均异常强度
    avg_infl = np.mean(list(infl_scores.values()))
    locality_scores = {}
    for svc, infl in infl_scores.items():
        locality_scores[svc] = infl / (avg_infl + 1e-9)
    
    # HubVictimPenalty：如果节点只有 latency/request 异常（没有 CPU/Memory/Network），扣分
    hub_penalty = {}
    for svc, _, _, spiked in candidates:
        spike_str = " ".join(spiked).lower()
        has_resource_or_network = any(kw in spike_str for kw in ["cpu", "mem", "bytes", "drop", "network"])
        if not has_resource_or_network:
            hub_penalty[svc] = 1.0  # 纯受害者，扣分
        else:
            hub_penalty[svc] = 0.0
    
    # 综合评分
    final_scores = {}
    for svc in [s for s, _, _, _ in candidates]:
        nf = network_scores.get(svc, 0)
        ear = earliest_scores.get(svc, 0)
        loc = locality_scores.get(svc, 1.0)
        hub = hub_penalty.get(svc, 0)
        
        final_scores[svc] = 0.45 * nf + 0.25 * ear * 1e6 + 0.15 * loc * 1e6 - 0.25 * hub * 1e6
    
    return final_scores

# ======== 纯指标 RCA 主流程 ========
def RCA_RE1TT(data_path, inject_time):
    baseline_df, fault_df, services = load_re1tt_data(data_path, inject_time)
    if baseline_df.empty: return []
    
    # Step 1: 异常检测 + persistence
    candidates_raw = detect_anomaly_with_persistence(baseline_df, fault_df, services)
    if not candidates_raw: return []
    
    current_svcs = [s for s, _, _, _ in candidates_raw]
    infl_scores = {s: sc for s, sc, _, _ in candidates_raw}
    
    # Step 2: 传播贡献（降权）
    all_times = {s: t for s, _, t, _ in candidates_raw if t is not None}
    propagation_scores = compute_propagation_gain(candidates_raw, all_times, infl_scores)
    
    # Step 3: 故障类型判别
    fault_mode = detect_fault_mode(candidates_raw)
    print(f"  🔬 故障类型: {fault_mode}")
    
    # Step 4: SoftCF Recovery
    rec_scores = {}
    for svc in current_svcs:
        cf_df = apply_soft_counterfactual(fault_df, baseline_df, svc)
        rec_scores[svc] = soft_recovery_score(fault_df, cf_df, baseline_df, svc, services)
    
    # Step 5: 按故障类型组装最终得分
    if fault_mode == "LOSS":
        # 🔥 LOSS 专用评分
        final_scores = compute_loss_scores(candidates_raw, infl_scores, all_times, services, baseline_df, fault_df)
        sorted_final = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
    else:
        # MEM / CPU / DELAY 使用 multiplicative fusion + 低权重传播贡献
        combined_infl = {}
        for svc in current_svcs:
            combined_infl[svc] = infl_scores.get(svc, 0) + PROPAGATION_WEIGHT * propagation_scores.get(svc, 0)
        
        combined = {}
        for svc in current_svcs:
            combined[svc] = combined_infl[svc] * (1 + RECOVERY_GAMMA * rec_scores.get(svc, 0))
        
        sorted_final = sorted(combined.items(), key=lambda x: x[1], reverse=True)
    
    # Step 6: 输出结果
    results = [(s, rec_scores.get(s, 0), infl_scores.get(s, 0)) for s, _ in sorted_final[:5]]
    return results

# ======== 批量测试入口 ========
def run_re1tt_benchmark(re1tt_root, max_cases=None):
    if not os.path.exists(re1tt_root):
        print(f"路径不存在: {re1tt_root}")
        return
    
    print(f"\n{'='*60}")
    print(f"🔥 P-CoE V8.3 RE1-TT（故障类型条件化 + LOSS专用评分 + 关L3）")
    print(f"{'='*60}\n")
    
    top1_correct, top3_correct, mrr_sum, total = 0, 0, 0.0, 0
    mode_counts = {"CPU": 0, "MEM": 0, "DELAY": 0, "LOSS": 0, "UNKNOWN": 0}
    mode_hits = {"CPU": 0, "MEM": 0, "DELAY": 0, "LOSS": 0, "UNKNOWN": 0}
    
    for fault_name in sorted(os.listdir(re1tt_root)):
        fault_path = os.path.join(re1tt_root, fault_name)
        if not os.path.isdir(fault_path): continue
        for rep in sorted(os.listdir(fault_path)):
            case_dir = os.path.join(fault_path, rep)
            if not os.path.isdir(case_dir): continue
            if max_cases and total >= max_cases: break
            
            try:
                data_path, inject_time, ground_truth = load_re1tt_case(case_dir)
                
                print(f"\n📋 Case {total+1}: {fault_name}/{rep}  🎯 {ground_truth}")
                
                results = RCA_RE1TT(data_path, inject_time)
                
                if not results: continue
                
                total += 1
                svcs = [s for s, _, _ in results]
                
                # 从输出中提取故障类型（这里简化：根据 case 名判断实际故障类型）
                actual_fault_type = fault_name.split("_")[-1].upper()  # cpu/delay/disk/loss/mem -> CPU/DELAY/DISK/LOSS/MEM
                if actual_fault_type == "DISK":
                    actual_fault_type = "CPU"  # disk 类归入 CPU 统计
                if actual_fault_type not in mode_counts:
                    actual_fault_type = "UNKNOWN"
                mode_counts[actual_fault_type] = mode_counts.get(actual_fault_type, 0) + 1
                
                if svcs and svcs[0] == ground_truth:
                    top1_correct += 1
                    mode_hits[actual_fault_type] = mode_hits.get(actual_fault_type, 0) + 1
                    print(f"✅ 命中!", end="")
                else:
                    print(f"❌ 未命中 (输出: {svcs[0]})", end="")
                
                if ground_truth in svcs[:3]: top3_correct += 1
                for rank, svc in enumerate(svcs, 1):
                    if svc == ground_truth: mrr_sum += 1.0 / rank; break
                
                print(f" | Top3: {', '.join([f'{s}({r:.3f})' for s,r,_ in results[:3]])}")
                    
            except Exception as e:
                print(f"❌ Case 崩溃 {case_dir}: {e}")
    
    print(f"\n{'='*60}")
    print(f"🔥 P-CoE V8.3 RE1-TT 结果（{total}个case）")
    print(f"{'='*60}")
    if total > 0:
        print(f"Top-1 Accuracy: {top1_correct/total:.3f} ({top1_correct}/{total})")
        print(f"Top-3 Hit Rate: {top3_correct/total:.3f} ({top3_correct}/{total})")
        print(f"MRR:            {mrr_sum/total:.3f}")
    print(f"\n故障类型统计:")
    for mode in ["CPU", "MEM", "DELAY", "LOSS", "UNKNOWN"]:
        count = mode_counts.get(mode, 0)
        hits = mode_hits.get(mode, 0)
        if count > 0:
            print(f"  {mode}: {hits}/{count} ({hits/count:.3f})")
    print(f"{'='*60}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "re1tt":
        max_cases = int(sys.argv[2]) if len(sys.argv) > 2 else None
        run_re1tt_benchmark("/home/admin/RCAEval/data/RE1-TT", max_cases=max_cases)
    else:
        run_re1tt_benchmark("/home/admin/RCAEval/data/RE1-TT")