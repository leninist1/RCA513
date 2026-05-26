"""
预置算子库 — 所有算子都是预先写好并测试过的，LLM只能选择不能改代码
"""
import numpy as np
import copy

# ====== 传播追踪策略 ======
def propagation_none(candidates, all_times, infl_scores, services, baseline_df, fault_df):
    """不追踪传播链"""
    return {svc: 0.0 for svc, _, _, _ in candidates}

def propagation_downstream_1_hop(candidates, all_times, infl_scores, services, baseline_df, fault_df):
    """下游1跳追踪（原版默认）"""
    CAUSAL_WINDOW = 15
    scores = {}
    for svc, _, first_time, _ in candidates:
        if first_time is None:
            scores[svc] = 0; continue
        gain = 0
        for other_svc, other_time in all_times.items():
            if other_svc == svc or other_time is None: continue
            dt = other_time - first_time
            if 0 < dt < CAUSAL_WINDOW:
                gain += infl_scores.get(other_svc, 0) / (dt + 1)
        scores[svc] = gain
    return scores

def propagation_bidirectional(candidates, all_times, infl_scores, services, baseline_df, fault_df):
    """双向追踪：上游+下游，扩大因果窗口"""
    CAUSAL_WINDOW = 25
    scores = {}
    for svc, _, first_time, _ in candidates:
        if first_time is None:
            scores[svc] = 0; continue
        gain = 0
        for other_svc, other_time in all_times.items():
            if other_svc == svc or other_time is None: continue
            dt = abs(other_time - first_time)
            if 0 < dt < CAUSAL_WINDOW:
                # 上游异常加权更高
                weight = 1.5 if other_time < first_time else 1.0
                gain += infl_scores.get(other_svc, 0) * weight / (dt + 1)
        scores[svc] = gain
    return scores

# ====== 时间衰减策略 ======
def time_penalty_linear(candidates):
    """线性衰减：越晚出现的候选惩罚越大"""
    times = [t for _, _, t, _ in candidates if t is not None]
    if not times: return {svc: 1.0 for svc, _, _, _ in candidates}
    min_t, max_t = min(times), max(times)
    t_range = max(1, max_t - min_t)
    return {svc: max(0.0, 1.0 - (t - min_t) / t_range) for svc, _, t, _ in candidates if t is not None}

def time_penalty_exponential(candidates):
    """指数衰减：最早异常的权重远高于后续"""
    times = [t for _, _, t, _ in candidates if t is not None]
    if not times: return {svc: 1.0 for svc, _, _, _ in candidates}
    min_t = min(times)
    return {svc: np.exp(-(t - min_t) / 10.0) for svc, _, t, _ in candidates if t is not None}

def time_penalty_step(candidates):
    """阶跃函数：因果窗口内的全保留，窗口外的截断"""
    times = [t for _, _, t, _ in candidates if t is not None]
    if not times: return {svc: 1.0 for svc, _, _, _ in candidates}
    min_t = min(times)
    WINDOW = 10
    return {svc: 1.0 if (t - min_t) < WINDOW else 0.1 for svc, _, t, _ in candidates if t is not None}

# ====== 融合策略 ======
def fusion_multiplicative(infl_scores, rec_scores, propagation_scores=None):
    """乘法融合：infl * (1 + gamma * rec)（原版默认）"""
    GAMMA = 0.5
    if propagation_scores:
        return {svc: (infl_scores.get(svc, 0) + 0.25 * propagation_scores.get(svc, 0)) * (1 + GAMMA * rec_scores.get(svc, 0))
                for svc in infl_scores}
    return {svc: infl_scores.get(svc, 0) * (1 + GAMMA * rec_scores.get(svc, 0)) for svc in infl_scores}

def fusion_additive(infl_scores, rec_scores, propagation_scores=None):
    """加法融合：infl + gamma * rec"""
    GAMMA = 0.5
    base = {svc: infl_scores.get(svc, 0) for svc in set(list(infl_scores.keys()) + list(rec_scores.keys()))}
    for svc in base:
        base[svc] += GAMMA * rec_scores.get(svc, 0)
        if propagation_scores:
            base[svc] += 0.25 * propagation_scores.get(svc, 0)
    return base

def fusion_max_pooling(infl_scores, rec_scores, propagation_scores=None):
    """最大值融合：取 infl 和 rec 加权后的最大值"""
    GAMMA = 0.5
    result = {}
    for svc in set(list(infl_scores.keys()) + list(rec_scores.keys())):
        infl_norm = infl_scores.get(svc, 0)
        rec_norm = rec_scores.get(svc, 0) * GAMMA
        result[svc] = max(infl_norm, rec_norm)
    return result

# ====== 算子注册表 ======
PROPAGATION_STRATEGIES = {
    "none": propagation_none,
    "downstream_1_hop": propagation_downstream_1_hop,
    "bidirectional": propagation_bidirectional,
}

TIME_PENALTY_STRATEGIES = {
    "linear_decay": time_penalty_linear,
    "step_function_cutoff": time_penalty_step,
    "exponential_decay": time_penalty_exponential,
}

FUSION_STRATEGIES = {
    "multiplicative": fusion_multiplicative,
    "additive": fusion_additive,
    "max_pooling": fusion_max_pooling,
}
