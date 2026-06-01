"""Delay-pattern localization for Noise Lab v2.

This module translates TDOA-style ideas into RCA by checking whether a
candidate object consistently leads the abnormal responses of structurally
reachable objects.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Any, Dict, List, Tuple

from ..mace.graph import ObjectGraph

EPS = 1e-6


@dataclass
class DelayPatternScore:
    object_id: str
    source_time_consistency: float
    average_delay_gain: float
    observer_coverage: float
    reverse_penalty: float
    lead_pairs: List[Tuple[str, float]]


class DelayPatternLocalizer:
    """Estimate whether an object behaves like a temporally consistent source."""

    def __init__(self, max_depth: int = 2):
        self.max_depth = max_depth

    def score(self, graph: ObjectGraph) -> Dict[str, Dict[str, Any]]:
        if not graph.nodes:
            return {}
        raw: Dict[str, DelayPatternScore] = {}
        for object_id in graph.nodes:
            reachable = self._reachable_with_decay(graph, object_id)
            source_time_consistency = self._source_time_consistency(graph, object_id, reachable)
            average_delay_gain = self._average_delay_gain(graph, object_id, reachable)
            observer_coverage = self._observer_coverage(graph, reachable)
            reverse_penalty = self._reverse_penalty(graph, object_id)
            lead_pairs = self._lead_pairs(graph, object_id, reachable)
            raw[object_id] = DelayPatternScore(
                object_id=object_id,
                source_time_consistency=source_time_consistency,
                average_delay_gain=average_delay_gain,
                observer_coverage=observer_coverage,
                reverse_penalty=reverse_penalty,
                lead_pairs=lead_pairs,
            )
        normalized = self._normalize(raw)
        return {
            object_id: {
                "source_time_consistency": round(item.source_time_consistency, 6),
                "average_delay_gain": round(item.average_delay_gain, 6),
                "observer_coverage": round(item.observer_coverage, 6),
                "reverse_penalty": round(item.reverse_penalty, 6),
                "lead_pairs": [{"object_id": target, "score": round(score, 6)} for target, score in item.lead_pairs],
            }
            for object_id, item in normalized.items()
        }

    def _reachable_with_decay(self, graph: ObjectGraph, source: str) -> Dict[str, float]:
        visited: Dict[str, float] = {}
        queue = deque([(source, 0, 1.0)])
        while queue:
            node_id, depth, path_weight = queue.popleft()
            if depth >= self.max_depth:
                continue
            for child, edge_weight in graph.adjacency.get(node_id, {}).items():
                if child not in graph.nodes or child == source:
                    continue
                next_weight = path_weight * edge_weight * (0.82 ** depth)
                if next_weight <= visited.get(child, 0.0):
                    continue
                visited[child] = next_weight
                queue.append((child, depth + 1, next_weight))
        return visited

    def _source_time_consistency(self, graph: ObjectGraph, object_id: str, reachable: Dict[str, float]) -> float:
        source_ts = graph.nodes[object_id].earliest_timestamp
        if source_ts is None or not reachable:
            return 0.5
        score = 0.0
        total = 0.0
        for target, weight in reachable.items():
            target_ts = graph.nodes[target].earliest_timestamp
            if target_ts is None:
                continue
            delta = target_ts - source_ts
            score += weight * self._lead_confidence(delta)
            total += weight
        if total <= EPS:
            return 0.5
        return score / total

    def _average_delay_gain(self, graph: ObjectGraph, object_id: str, reachable: Dict[str, float]) -> float:
        source_ts = graph.nodes[object_id].earliest_timestamp
        if source_ts is None or not reachable:
            return 0.0
        score = 0.0
        total = 0.0
        for target, weight in reachable.items():
            target_ts = graph.nodes[target].earliest_timestamp
            if target_ts is None:
                continue
            delta = target_ts - source_ts
            if delta >= 0:
                score += weight * math.tanh(delta / 300.0)
            total += weight
        if total <= EPS:
            return 0.0
        return score / total

    def _observer_coverage(self, graph: ObjectGraph, reachable: Dict[str, float]) -> float:
        if not reachable:
            return 0.0
        return min(1.0, len(reachable) / max(1.0, 0.35 * len(graph.nodes)))

    def _reverse_penalty(self, graph: ObjectGraph, object_id: str) -> float:
        source_ts = graph.nodes[object_id].earliest_timestamp
        if source_ts is None:
            return 0.5
        penalty = 0.0
        total = 0.0
        for parent, children in graph.adjacency.items():
            if object_id not in children or parent not in graph.nodes:
                continue
            parent_ts = graph.nodes[parent].earliest_timestamp
            if parent_ts is None:
                continue
            weight = children[object_id]
            total += weight
            if parent_ts < source_ts:
                penalty += weight * math.tanh((source_ts - parent_ts) / 300.0)
        if total <= EPS:
            return 0.0
        return penalty / total

    def _lead_pairs(self, graph: ObjectGraph, object_id: str, reachable: Dict[str, float]) -> List[Tuple[str, float]]:
        source_ts = graph.nodes[object_id].earliest_timestamp
        if source_ts is None:
            return []
        rows: List[Tuple[str, float]] = []
        for target, weight in reachable.items():
            target_ts = graph.nodes[target].earliest_timestamp
            if target_ts is None:
                continue
            delta = target_ts - source_ts
            rows.append((target, weight * self._lead_confidence(delta)))
        rows.sort(key=lambda item: item[1], reverse=True)
        return rows[:6]

    def _lead_confidence(self, delta_seconds: float) -> float:
        if delta_seconds >= 0:
            return math.exp(-delta_seconds / 1800.0)
        return 0.10 * math.exp(delta_seconds / 600.0)

    def _normalize(self, raw: Dict[str, DelayPatternScore]) -> Dict[str, DelayPatternScore]:
        fields = [
            "source_time_consistency",
            "average_delay_gain",
            "observer_coverage",
            "reverse_penalty",
        ]
        normalized: Dict[str, Dict[str, float]] = {field: {} for field in fields}
        for field in fields:
            values = [getattr(item, field) for item in raw.values()]
            lo = min(values)
            hi = max(values)
            for object_id, item in raw.items():
                value = getattr(item, field)
                if math.isclose(lo, hi):
                    normalized[field][object_id] = 0.5
                else:
                    normalized[field][object_id] = (value - lo) / max(EPS, hi - lo)
        output: Dict[str, DelayPatternScore] = {}
        for object_id, item in raw.items():
            output[object_id] = DelayPatternScore(
                object_id=object_id,
                source_time_consistency=normalized["source_time_consistency"][object_id],
                average_delay_gain=normalized["average_delay_gain"][object_id],
                observer_coverage=normalized["observer_coverage"][object_id],
                reverse_penalty=normalized["reverse_penalty"][object_id],
                lead_pairs=item.lead_pairs,
            )
        return output
