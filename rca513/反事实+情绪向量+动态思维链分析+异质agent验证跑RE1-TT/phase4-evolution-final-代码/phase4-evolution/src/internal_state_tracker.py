from difflib import SequenceMatcher
from typing import Dict, List
from src.variant import Variant

class InternalStateTracker:
    def __init__(self):
        self.agent_history = {}  # agent_id -> list of (variant, state, metrics)

    def update(self, agent_id: str, old_state: dict, variant: Variant,
               metrics: dict, reasoning_traces: List[Dict]) -> dict:
        # 简单规则实现；后续可增强
        u = self._calc_uncertainty(reasoning_traces)
        c = self._extract_confidence(reasoning_traces)
        s = self._update_fatigue(agent_id, variant)
        e = self._update_exploration(agent_id, metrics, variant)

        new_state = {"u": u, "c": c, "s": s, "e": e}
        # 记录历史
        self.agent_history.setdefault(agent_id, []).append(
            (variant.variant_id, new_state, metrics)
        )
        return new_state

    def _calc_uncertainty(self, traces: List[Dict]) -> float:
        # 简单统计推理文本中的不确定词
        keywords = ['possibly', 'might', 'unclear', '不确定', '可能']
        matches = 0
        for t in traces:
            reasoning = t.get('reasoning', '')
            for kw in keywords:
                matches += reasoning.lower().count(kw)
        # 映射到 0~1，假设最多出现20次
        return min(1.0, matches / 20.0)

    def _extract_confidence(self, traces: List[Dict]) -> float:
        # 假设推理文本中有 "置信度:0.8" 之类的标记，这里返回默认
        return 0.5

    def _update_fatigue(self, agent_id: str, current_variant: Variant) -> float:
        if agent_id not in self.agent_history or not self.agent_history[agent_id]:
            return 0
        # 取最近两次变体的 modification 文本相似度
        last_two = self.agent_history[agent_id][-2:]  # 获取最近两次的variant_id
        # 简化：直接从历史记录中找对应 variant 的 modification
        # 由于我们存储的是 (variant_id, state, metrics)，modification需要从全局获取
        # 这里不实现复杂的相似计算，返回默认
        return 0

    def _update_exploration(self, agent_id: str, metrics: dict, variant: Variant) -> float:
        # 若 Type C 准确率低，则冲动升高
        # 目前指标里没有 Type C 细分，假设从 metrics 中获取
        type_c_acc = metrics.get('type_c_accuracy', 0.0)
        if type_c_acc < 0.5:
            return min(1.0, (1 - type_c_acc) * 2)
        return 0.0