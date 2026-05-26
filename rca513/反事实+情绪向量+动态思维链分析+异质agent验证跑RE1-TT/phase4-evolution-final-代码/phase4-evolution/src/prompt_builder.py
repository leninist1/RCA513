from src.variant import Variant

PARAM_DOC = """
【可修改参数清单】（只能改以下参数的值，不能改其他）

标量参数：
- anomaly_z_threshold (当前=3.0): 异常检测Z分数阈值，越高越严格
- top_k (当前=7): 候选集大小
- propagation_weight (当前=0.25): 传播贡献权重
- recovery_gamma (当前=0.5): 反事实恢复权重
- soft_cf_beta (当前=0.7): 反事实混合比例
- causal_window_seconds (当前=15): 因果窗口秒数

权重参数（按故障类型）：
LOSS故障敏感:
- istio_error_weight (当前=2.5): Istio错误数权重
- node_network_receive_drop_weight (当前=1.0): 节点接收丢包权重
MEM故障敏感:
- memory_failures_weight (当前=3.0): 内存失败次数权重
- memory_usage_weight (当前=0.8): 内存使用权重
CPU故障敏感:
- cpu_usage_weight (当前=1.5): CPU使用权重
DELAY故障敏感:
- latency_p99_weight (当前=1.5): P99延迟权重
"""

def build_agent_prompt(agent_cfg: dict, champion: Variant, failure_summary: str, gen: int, eval_results_str: str) -> str:
    state = agent_cfg['internal_state']
    instructions = []
    if state['u'] > 0.65:
        instructions.append("你的不确定度较高，请优先设计能提高推理确定性的变体。")
    if state['s'] >= 3:
        instructions.append("你已经连续多代使用同类改进，请提出结构上不同的变异方向。")
    if state['e'] > 0.7:
        instructions.append("Type C 故障的准确率很低，请专门针对它提出改进方案。")

    instruction_block = "\n".join(f"- {i}" for i in instructions) if instructions else "综合优化。"

    prompt = f"""=== 实验背景 ===
当前进化第 {gen} 代，Champion: {champion.variant_id}
Champion 当前能力：{eval_results_str}

{failure_summary}

【你的内部状态】
不确定度={state['u']:.2f}, 疲惫度={state['s']:.1f}, 探索冲动={state['e']:.2f}

【进化指导】
{instruction_block}

【你的固定身份】
归因哲学：{agent_cfg['attribution_philosophy']}
搜索偏置：{agent_cfg['search_bias']}

{PARAM_DOC}

请基于上述信息，提出一个改进 Champion 的变体。你必须输出严格的 JSON 格式：
{{
  "variant_id": "V7.X-xxx",
  "parent_version": "V7.0",
  "target_operator": "算子编号（如1, 2, 3...）",
  "modification": {{
    "what_changes": "改动描述（一句话）",
    "new_params": {{
      "参数名1": 新值,
      "参数名2": 新值
    }}
  }}
}}

注意：new_params 里的参数必须来自上面的【可修改参数清单】。
"""
    return prompt

STRATEGY_DOC = """
【可选择的策略选项】（比调阈值更有影响力的改动）

传播链追踪深度(propagation_strategy):
- "none"：不追踪传播链
- "upstream_1_hop"：只追踪上游1跳
- "downstream_1_hop"：只追踪下游1跳（当前默认）
- "downstream_2_hops"：追踪下游2跳（Claude拓扑指纹偏好）
- "bidirectional"：双向追踪

时间衰减策略(time_penalty_strategy):
- "linear_decay"：线性衰减（当前默认）
- "step_function_cutoff"：超过窗口直接截断
- "exponential_decay"：指数衰减（DeepSeek最早异常偏好）
- "none"：不衰减

融合策略(fusion_strategy):
- "multiplicative"：乘法融合 infl * (1 + gamma * rec)（当前默认）
- "additive"：加法融合 infl + gamma * rec
- "max_pooling"：取最大值 max(infl, rec)
- "pareto_front"：帕累托前沿筛选

算子开关:
- enable_topology_pruning: 拓扑剪枝（减少冗余候选）
- enable_adaptive_window: 自适应时间窗口（按故障类型动态调整）
"""

# 把 PARAM_DOC 和 STRATEGY_DOC 合并到 prompt 里
PARAM_DOC = PARAM_DOC + STRATEGY_DOC
