"""Local graph intervention for lightweight counterfactual RCA."""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from .graph import ObjectGraph


def rank_by_local_intervention(graph: ObjectGraph, candidates: List[str], top_k: int = 5) -> List[Dict]:
    scored = [evaluate_local_intervention(graph, object_id) for object_id in candidates if object_id in graph.nodes]
    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:top_k]


def evaluate_local_intervention(graph: ObjectGraph, object_id: str, depth: int = 2, decay: float = 0.65) -> Dict:
    """Approximate causal effect by removing a local object and propagating impact downstream."""
    node = graph.nodes[object_id]
    impacted = {object_id: 1.0}
    frontier = {object_id: 1.0}

    for _ in range(depth):
        next_frontier = {}
        for src, mass in frontier.items():
            for dst, weight in graph.adjacency.get(src, {}).items():
                propagated = mass * min(1.0, weight) * decay
                if propagated <= 1e-4:
                    continue
                next_frontier[dst] = max(next_frontier.get(dst, 0.0), propagated)
                impacted[dst] = max(impacted.get(dst, 0.0), propagated)
        frontier = next_frontier
        if not frontier:
            break

    removed_mass = 0.0
    total_downstream = 0.0
    self_removed_mass = 0.0
    for dst, influence in impacted.items():
        anomaly = graph.nodes[dst].anomaly_score
        contribution = influence * anomaly
        removed_mass += contribution
        if dst != object_id:
            total_downstream += anomaly
        else:
            self_removed_mass += contribution

    incoming = graph.incoming_mass(object_id)
    exclusivity = max(0.0, node.anomaly_score + total_downstream - 0.5 * incoming)
    coverage = removed_mass / max(1e-6, node.anomaly_score + total_downstream)
    score = 0.45 * _squash(removed_mass) + 0.35 * coverage + 0.20 * _squash(exclusivity)

    return {
        "object_id": object_id,
        "score": float(score),
        "removed_mass": float(removed_mass),
        "self_removed_mass": float(self_removed_mass),
        "downstream_removed_mass": float(max(0.0, removed_mass - self_removed_mass)),
        "coverage": float(coverage),
        "exclusivity": float(exclusivity),
        "representative": node.representative,
        "impacted": {key: float(value) for key, value in impacted.items()},
    }


def _squash(value: float) -> float:
    return float(np.tanh(max(0.0, value)))
