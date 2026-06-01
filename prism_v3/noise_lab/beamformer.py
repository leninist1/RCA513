"""Structural beamforming for Noise Lab v2.

This module performs directional explanation on the object graph. A candidate
only receives credit for anomalies that are reachable along structurally and
temporally plausible paths.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Any, Dict, List, Tuple

from ..mace.graph import ObjectGraph

EPS = 1e-6


@dataclass
class BeamformedScore:
    object_id: str
    beamformed_explanation: float
    directional_focus: float
    offbeam_penalty: float
    mechanism_coherence: float
    beam_targets: List[Tuple[str, float]]


class StructuralBeamformer:
    """Focus explanation mass onto structurally valid propagation directions."""

    def __init__(self, max_depth: int = 3):
        self.max_depth = max_depth

    def score(self, graph: ObjectGraph) -> Dict[str, Dict[str, Any]]:
        if not graph.nodes:
            return {}
        raw: Dict[str, BeamformedScore] = {}
        for object_id, node in graph.nodes.items():
            beam_targets = self._beam_targets(graph, object_id)
            beamformed_explanation = self._beamformed_explanation(graph, object_id, beam_targets)
            directional_focus = self._directional_focus(graph, object_id, beam_targets)
            offbeam_penalty = self._offbeam_penalty(graph, object_id, beam_targets)
            mechanism_coherence = self._mechanism_coherence(graph, object_id, beam_targets)
            raw[object_id] = BeamformedScore(
                object_id=object_id,
                beamformed_explanation=beamformed_explanation,
                directional_focus=directional_focus,
                offbeam_penalty=offbeam_penalty,
                mechanism_coherence=mechanism_coherence,
                beam_targets=beam_targets,
            )
        normalized = self._normalize(raw)
        return {
            object_id: {
                "beamformed_explanation": round(item.beamformed_explanation, 6),
                "directional_focus": round(item.directional_focus, 6),
                "offbeam_penalty": round(item.offbeam_penalty, 6),
                "mechanism_coherence": round(item.mechanism_coherence, 6),
                "beam_targets": [{"object_id": target, "score": round(score, 6)} for target, score in item.beam_targets],
            }
            for object_id, item in normalized.items()
        }

    def _beam_targets(self, graph: ObjectGraph, source: str) -> List[Tuple[str, float]]:
        source_node = graph.nodes[source]
        source_reason = source_node.best_reason()
        source_ts = source_node.earliest_timestamp
        queue = deque([(source, 0, 1.0)])
        best: Dict[str, float] = {}
        while queue:
            node_id, depth, path_weight = queue.popleft()
            if depth >= self.max_depth:
                continue
            for child, edge_weight in graph.adjacency.get(node_id, {}).items():
                if child not in graph.nodes or child == source:
                    continue
                child_node = graph.nodes[child]
                temporal = self._temporal_gate(source_ts, child_node.earliest_timestamp)
                mechanism = self._mechanism_gate(source_reason, child_node.best_reason())
                score = path_weight * edge_weight * temporal * mechanism * (0.85 ** depth)
                if score <= best.get(child, 0.0):
                    continue
                best[child] = score
                queue.append((child, depth + 1, score))
        rows = sorted(best.items(), key=lambda item: item[1], reverse=True)
        return rows[:8]

    def _beamformed_explanation(self, graph: ObjectGraph, object_id: str, beam_targets: List[Tuple[str, float]]) -> float:
        node = graph.nodes[object_id]
        score = 0.40 * node.anomaly_score + 0.12 * node.trace_score
        for target, weight in beam_targets:
            target_node = graph.nodes[target]
            score += weight * (0.45 * target_node.anomaly_score + 0.35 * target_node.trace_score + 0.20 * target_node.metric_score)
        return score

    def _directional_focus(self, graph: ObjectGraph, object_id: str, beam_targets: List[Tuple[str, float]]) -> float:
        if not beam_targets:
            return 0.0
        total_beam = sum(score for _, score in beam_targets)
        total_downstream = sum(graph.adjacency.get(object_id, {}).values())
        if total_downstream <= EPS:
            return min(1.0, total_beam)
        return total_beam / max(EPS, total_downstream)

    def _offbeam_penalty(self, graph: ObjectGraph, object_id: str, beam_targets: List[Tuple[str, float]]) -> float:
        selected = {target for target, _ in beam_targets}
        penalty = 0.0
        total = 0.0
        for child, weight in graph.adjacency.get(object_id, {}).items():
            if child not in graph.nodes:
                continue
            total += weight
            if child not in selected:
                penalty += weight * graph.nodes[child].anomaly_score
        if total <= EPS:
            return 0.0
        return penalty / total

    def _mechanism_coherence(self, graph: ObjectGraph, object_id: str, beam_targets: List[Tuple[str, float]]) -> float:
        source_reason = graph.nodes[object_id].best_reason()
        if not beam_targets:
            return 0.2
        score = 0.0
        total = 0.0
        for target, weight in beam_targets:
            total += weight
            score += weight * self._mechanism_gate(source_reason, graph.nodes[target].best_reason())
        if total <= EPS:
            return 0.2
        return score / total

    def _temporal_gate(self, source_ts: float | None, target_ts: float | None) -> float:
        if source_ts is None or target_ts is None:
            return 0.7
        delta = target_ts - source_ts
        if delta >= 0:
            return math.exp(-delta / 2400.0)
        return 0.08 * math.exp(delta / 600.0)

    def _mechanism_gate(self, source_reason: str, target_reason: str) -> float:
        src = source_reason.lower()
        dst = target_reason.lower()
        if src == dst:
            return 1.0
        if ("cpu" in src and "disk" in dst) or ("disk" in src and "cpu" in dst):
            return 0.85
        if ("network" in src and "latency" in dst) or ("latency" in src and "network" in dst):
            return 0.85
        if ("db" in src and "memory" in dst) or ("memory" in src and "db" in dst):
            return 0.75
        return 0.55

    def _normalize(self, raw: Dict[str, BeamformedScore]) -> Dict[str, BeamformedScore]:
        fields = [
            "beamformed_explanation",
            "directional_focus",
            "offbeam_penalty",
            "mechanism_coherence",
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
        output: Dict[str, BeamformedScore] = {}
        for object_id, item in raw.items():
            output[object_id] = BeamformedScore(
                object_id=object_id,
                beamformed_explanation=normalized["beamformed_explanation"][object_id],
                directional_focus=normalized["directional_focus"][object_id],
                offbeam_penalty=normalized["offbeam_penalty"][object_id],
                mechanism_coherence=normalized["mechanism_coherence"][object_id],
                beam_targets=item.beam_targets,
            )
        return output
