"""
P-CoE V8.6 - RE1-TT 纯指标版（故障类型概率加权融合）
基于 V8.3 分类逻辑 + 朴素贝叶斯概率估计器

V8.6 核心升级：
1. 朴素贝叶斯分类器：用 Top-5 指标指纹作为特征，输出故障类型概率分布。
2. 加权融合：用概率作为权重，对 LOSS/DELAY/CPU 三个公式的结果做加权平均。
3. MEM 类型保持 hard classification（指纹特征极其明确，不需要概率混合）。
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
PROPAGATION_WEIGHT = 0.25

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

# ======== 🔥 朴素贝叶斯概率估计器 ========
class NaiveBayesFaultClassifier:
    def __init__(self):
        self.class_probs = {}
        self.feature_probs = {}
        self.classes = ["CPU", "MEM", "DELAY", "LOSS"]
    
    def extract_features(self, candidates):
        """从 Top-5 候选的指标指纹提取特征向量"""
        all_spiked = []
        for _, _, _, spiked in candidates:
            for s in spiked:
                name = s.split("(")[0].lower()
                all_spiked.append(name)
        
        spike_str = " ".join(all_spiked)
        features = {}
        features["has_network"] = any(kw in spike_str for kw in ["bytes", "drop", "retransmit", "network"])
        features["has_mem"] = any(kw in spike_str for kw in ["memory", "mem", "rss", "working-set"])
        features["has_cpu"] = any(kw in spike_str for kw in ["cpu"])
        features["has_disk"] = any(kw in spike_str for kw in ["fs-", "disk"])
        features["latency_only"] = all(kw in spike_str for kw in ["latency", "request"]) and not features["has_cpu"] and not features["has_mem"]
        
        return features
    
    def fit(self, cases_features, labels):
        """训练：cases_features 是特征列表，labels 是真实故障类型列表"""
        n = len(labels)
        for cls in self.classes:
            self.class_probs[cls] = (labels.count(cls) + 1) / (n + len(self.classes))
        
        feature_names = ["has_network", "has_mem", "has_cpu", "has_disk", "latency_only"]
        self.feature_probs = {}
        for cls in self.classes:
            cls_indices = [i for i, l in enumerate(labels) if l == cls]
            cls_count = len(cls_indices)
            if cls_count == 0:
                cls_count = 1
            self.feature_probs[cls] = {}
            for feat in feature_names:
                feat_count = sum(1 for i in cls_indices if cases_features[i].get(feat, False))
                self.feature_probs[cls][feat] = (feat_count + 1) / (cls_count + 2)
    
    def predict_proba(self, features):
        """返回概率分布字典"""
        scores = {}
        for cls in self.classes:
            score = np.log(self.class_probs.get(cls, 0.25))
            for feat, val in features.items():
                prob = self.feature_probs.get(cls, {}).get(feat, 0.5)
                if val:
                    score += np.log(prob)
                else:
                    score += np.log(1 - prob)
            scores[cls] = score
        
        # Softmax 转概率
        max_score = max(scores.values())
        exp_scores = {k: np.exp(v - max_score) for k, v in scores.items()}
        total = sum(exp_scores.values())
        return {k: v / total for k, v in exp_scores.items()}
    
    def is_trained(self):
        return len(self.class_probs) > 0

# ======== 故障类型判别 + 概率估计 ========
def detect_fault_mode_with_proba(candidates, classifier=None):
    """返回 (hard_mode, proba_dict)"""
    if not candidates:
        return "UNKNOWN", {}
    
    all_spiked = []
    for _, _, _, spiked in candidates:
        for s in spiked:
            name = s.split("(")[0]
            all_spiked.append(name)
    
    spike_str = " ".join(all_spiked).lower()
    
    # 分类器概率估计
    proba = {}
    if classifier and classifier.is_trained():
        features = classifier.extract_features(candidates)
        proba = classifier.predict_proba(features)
    
    # Hard 分类逻辑（V8.3 版本）
    network_keywords = ["bytes", "drop", "retransmit", "network"]
    network_count = sum(1 for kw in network_keywords if kw in spike_str)
    
    mem_keywords = ["memory", "mem"]
    mem_count = sum(1 for kw in mem_keywords if kw in spike_str)
    
    cpu_keywords = ["cpu"]
    cpu_count = sum(1 for kw in cpu_keywords if kw in spike_str)
    
    latency_keywords = ["latency"]
    latency_count = sum(1 for kw in latency_keywords if kw in spike_str)
    
    total = max(1, network_count + mem_count + cpu_count + latency_count)
    
    if network_count >= 2:
        earliest = None
        earliest_time = None
        earliest_spiked = []
        for svc, sc, t, sp in candidates:
            if t is not None and (earliest_time is None or t < earliest_time):
                earliest_time = t
                earliest_spiked = sp
        if earliest_spiked:
            earliest_str = " ".join(earliest_spiked).lower()
            if any(kw in earliest_str for kw in network_keywords):
                return "LOSS", proba
    
    if mem_count / total > 0.3:
        return "MEM", proba
    if cpu_count / total > 0.3:
        return "CPU", proba
    if latency_count / total > 0.5:
        return "DELAY", proba
    
    return "DELAY", proba

# ======== LOSS 专用评分（V8.4 归一化版本）========
def compute_loss_scores(candidates, infl_scores, all_times, services, baseline_df, fault_df):
    if not candidates:
        return {}
    
    min_time = min(t for _, _, t, _ in candidates if t is not None)
    time_range = max(1, max(t for _, _, t, _ in candidates if t is not None) - min_time)
    
    network_scores_raw = {}
    for svc, _, _, spiked in candidates:
        nf_score = 0.0
        for s in spiked:
            name = s.split("(")[0].lower()
            if any(kw in name for kw in ["bytes", "drop", "retransmit", "network"]):
                try:
                    z_val = float(s.split("z=")[1].rstrip(")"))
                    nf_score += z_val
                except:
                    pass
        network_scores_raw[svc] = nf_score
    
    nf_max = max(network_scores_raw.values()) + 1e-9
    network_scores = {s: v / nf_max for s, v in network_scores_raw.items()}
    
    earliest_scores = {}
    for svc, _, t, _ in candidates:
        if t is None:
            earliest_scores[svc] = 0.0
        else:
            earliest_scores[svc] = max(0.0, 1.0 - (t - min_time) / time_range)
    
    avg_infl = np.mean(list(infl_scores.values())) + 1e-9
    locality_scores = {}
    for svc, infl in infl_scores.items():
        locality_scores[svc] = min(1.0, infl / avg_infl / 10.0)
    
    hub_penalty = {}
    for svc, _, _, spiked in candidates:
        spike_str = " ".join(spiked).lower()
        has_resource_or_network = any(kw in spike_str for kw in ["cpu", "mem", "bytes", "drop", "network"])
        hub_penalty[svc] = 1.0 if not has_resource_or_network else 0.0
    
    final_scores = {}
    for svc in [s for s, _, _, _ in candidates]:
        nf = network_scores.get(svc, 0)
        ear = earliest_scores.get(svc, 0)
        loc = locality_scores.get(svc, 0)
        hub = hub_penalty.get(svc, 0)
        
        final_scores[svc] = 0.45 * nf + 0.25 * ear + 0.15 * loc - 0.25 * hub
    
    return final_scores

# ======== 纯指标 RCA 主流程 ========
def RCA_RE1TT(data_path, inject_time, classifier=None):
    baseline_df, fault_df, services = load_re1tt_data(data_path, inject_time)
    if baseline_df.empty: return [], None
    
    candidates_raw = detect_anomaly_with_persistence(baseline_df, fault_df, services)
    if not candidates_raw: return [], None
    
    current_svcs = [s for s, _, _, _ in candidates_raw]
    infl_scores = {s: sc for s, sc, _, _ in candidates_raw}
    
    all_times = {s: t for s, _, t, _ in candidates_raw if t is not None}
    propagation_scores = compute_propagation_gain(candidates_raw, all_times, infl_scores)
    
    # 故障类型判别 + 概率估计
    fault_mode, proba = detect_fault_mode_with_proba(candidates_raw, classifier)
    
    # SoftCF Recovery
    rec_scores = {}
    for svc in current_svcs:
        cf_df = apply_soft_counterfactual(fault_df, baseline_df, svc)
        rec_scores[svc] = soft_recovery_score(fault_df, cf_df, baseline_df, svc, services)
    
    # 🔥 加权融合：MEM 保持 hard，其他三类加权混合
    if fault_mode == "MEM":
        # MEM 保持纯 multiplicative fusion
        combined_infl = {}
        for svc in current_svcs:
            combined_infl[svc] = infl_scores.get(svc, 0) + PROPAGATION_WEIGHT * propagation_scores.get(svc, 0)
        combined = {}
        for svc in current_svcs:
            combined[svc] = combined_infl[svc] * (1 + RECOVERY_GAMMA * rec_scores.get(svc, 0))
        sorted_final = sorted(combined.items(), key=lambda x: x[1], reverse=True)
        print(f"  🔬 故障类型: MEM (hard)")
    else:
        # CPU/DELAY/LOSS 加权融合
        w_l = proba.get("LOSS", 0.1)
        w_d = proba.get("DELAY", 0.3)
        w_c = proba.get("CPU", 0.3)
        w_m = proba.get("MEM", 0.1)
        # 归一化（去掉 MEM 的权重，因为 MEM 只走 hard）
        total_w = w_l + w_d + w_c + 1e-9
        w_l, w_d, w_c = w_l/total_w, w_d/total_w, w_c/total_w
        
        print(f"  🔬 故障类型: {fault_mode} | P(LOSS)={w_l:.2f} P(DELAY)={w_d:.2f} P(CPU)={w_c:.2f}")
        
        # LOSS 公式
        loss_scores = compute_loss_scores(candidates_raw, infl_scores, all_times, services, baseline_df, fault_df)
        
        # DELAY 公式（multiplicative fusion，不带传播贡献）
        delay_scores = {}
        for svc in current_svcs:
            delay_scores[svc] = infl_scores.get(svc, 0) * (1 + RECOVERY_GAMMA * rec_scores.get(svc, 0))
        
        # CPU 公式（带传播贡献的 multiplicative fusion）
        cpu_scores = {}
        for svc in current_svcs:
            combined_infl = infl_scores.get(svc, 0) + PROPAGATION_WEIGHT * propagation_scores.get(svc, 0)
            cpu_scores[svc] = combined_infl * (1 + RECOVERY_GAMMA * rec_scores.get(svc, 0))
        
        # 🔥 加权融合（先归一化到 [0,1]，再加权混合）
        def normalize(scores):
            if not scores: return {}
            max_v = max(scores.values()) + 1e-9
            return {k: v/max_v for k, v in scores.items()}
        
        n_loss = normalize(loss_scores)
        n_delay = normalize(delay_scores)
        n_cpu = normalize(cpu_scores)
        
        combined = {}
        for svc in current_svcs:
            combined[svc] = (
                w_l * n_loss.get(svc, 0) +
                w_d * n_delay.get(svc, 0) +
                w_c * n_cpu.get(svc, 0)
            )
        
        sorted_final = sorted(combined.items(), key=lambda x: x[1], reverse=True)
    
    results = [(s, rec_scores.get(s, 0), infl_scores.get(s, 0)) for s, _ in sorted_final[:5]]
    return results, proba

# ======== 批量测试入口（带交叉验证训练）========
def run_re1tt_benchmark_with_training(re1tt_root, max_cases=None):
    if not os.path.exists(re1tt_root):
        print(f"路径不存在: {re1tt_root}")
        return
    
    print(f"\n{'='*60}")
    print(f"🔥 P-CoE V8.6 RE1-TT（概率加权融合 + 朴素贝叶斯）")
    print(f"{'='*60}\n")
    
    # ======== Step 1: 收集所有 case 的特征和标签 ========
    print("📚 收集训练数据...")
    all_features = []
    all_labels = []
    all_cases = []
    
    for fault_name in sorted(os.listdir(re1tt_root)):
        fault_path = os.path.join(re1tt_root, fault_name)
        if not os.path.isdir(fault_path): continue
        for rep in sorted(os.listdir(fault_path)):
            case_dir = os.path.join(fault_path, rep)
            if not os.path.isdir(case_dir): continue
            try:
                data_path, inject_time, ground_truth = load_re1tt_case(case_dir)
                baseline_df, fault_df, services = load_re1tt_data(data_path, inject_time)
                if baseline_df.empty: continue
                
                candidates_raw = detect_anomaly_with_persistence(baseline_df, fault_df, services)
                if not candidates_raw: continue
                
                actual_fault_type = fault_name.split("_")[-1].upper()
                if actual_fault_type == "DISK":
                    actual_fault_type = "CPU"
                if actual_fault_type not in ["CPU", "MEM", "DELAY", "LOSS"]:
                    actual_fault_type = "CPU"
                
                classifier_temp = NaiveBayesFaultClassifier()
                features = classifier_temp.extract_features(candidates_raw)
                
                all_features.append(features)
                all_labels.append(actual_fault_type)
                all_cases.append((data_path, inject_time, ground_truth, fault_name, rep))
            except Exception as e:
                pass
    
    print(f"✅ 收集到 {len(all_cases)} 个训练样本")
    
    # ======== Step 2: 留一法交叉验证测试 ========
    top1_correct, top3_correct, mrr_sum, total = 0, 0, 0.0, 0
    mode_counts = {"CPU": 0, "MEM": 0, "DELAY": 0, "LOSS": 0, "UNKNOWN": 0}
    mode_hits = {"CPU": 0, "MEM": 0, "DELAY": 0, "LOSS": 0, "UNKNOWN": 0}
    
    for test_idx in range(len(all_cases)):
        # 训练集：除当前 case 外的所有
        train_features = [all_features[i] for i in range(len(all_cases)) if i != test_idx]
        train_labels = [all_labels[i] for i in range(len(all_cases)) if i != test_idx]
        
        classifier = NaiveBayesFaultClassifier()
        classifier.fit(train_features, train_labels)
        
        data_path, inject_time, ground_truth, fault_name, rep = all_cases[test_idx]
        
        try:
            print(f"\n📋 Case {total+1}: {fault_name}/{rep}  🎯 {ground_truth}")
            
            results, proba = RCA_RE1TT(data_path, inject_time, classifier)
            
            if not results: continue
            
            total += 1
            svcs = [s for s, _, _ in results]
            
            actual_fault_type = fault_name.split("_")[-1].upper()
            if actual_fault_type == "DISK":
                actual_fault_type = "CPU"
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
    print(f"🔥 P-CoE V8.6 RE1-TT 结果（{total}个case，留一法交叉验证）")
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
        run_re1tt_benchmark_with_training("/home/admin/RCAEval/data/RE1-TT", max_cases=max_cases)
    else:
        run_re1tt_benchmark_with_training("/home/admin/RCAEval/data/RE1-TT")