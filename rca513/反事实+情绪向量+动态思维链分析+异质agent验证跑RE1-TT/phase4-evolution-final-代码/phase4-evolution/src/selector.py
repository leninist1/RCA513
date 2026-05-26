from typing import List
from src.variant import Variant
import random

class Selector:
    def __init__(self, k_top=3):
        self.k = k_top

    def select(self, variants: List[Variant], champion: Variant) -> List[Variant]:
        # 按 R_total 排序
        sorted_vars = sorted(variants, key=lambda v: v.metrics.get('R_total', 0), reverse=True)
        if not sorted_vars:
            return []
        selected = [sorted_vars[0]]

        # 多样性保留：选择与冠军编辑距离最大的（这里用 variant_id 的简单差异）
        # 实际可基于 modification 文本的相似度
        champion_desc = str(champion.modification)
        best_diversity = None
        best_dist = -1
        for v in sorted_vars[1:]:
            d = self._edit_distance(str(v.modification), champion_desc)
            if d > best_dist and d > 10:  # 阈值
                best_dist = d
                best_diversity = v
        if best_diversity and best_diversity not in selected:
            selected.append(best_diversity)

        # 局部最优：选 Type C 准确率最高的（假设 metrics 中有 type_c_accuracy）
        type_c_best = max(variants, key=lambda v: v.metrics.get('type_c_accuracy', 0))
        if type_c_best not in selected:
            selected.append(type_c_best)

        # 补充到 k 个
        for v in sorted_vars:
            if len(selected) >= self.k:
                break
            if v not in selected:
                selected.append(v)
        return selected[:self.k]

    def _edit_distance(self, a: str, b: str) -> int:
        # 简单的 Levenshtein 距离，可使用外部库；这里用内置序列匹配器近似
        from difflib import SequenceMatcher
        return int((1 - SequenceMatcher(None, a, b).ratio()) * max(len(a), len(b)))