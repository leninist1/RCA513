"""PRISM-style backbone scoring for MACE.

This module provides generic denoising and prior scoring signals that can be
consumed by MACE without hard-coding dataset-specific entities.
"""

from __future__ import annotations

from collections import defaultdict
import math
import re
from typing import Any, Dict, List

from .graph import ObjectGraph
from .intervention import evaluate_local_intervention

EPS = 1e-6


def score_prism_backbone(graph: ObjectGraph) -> Dict[str, Any]:
    if not graph.nodes:
        return {"scores": {}, "debug": {"empty": True}}

    evidence_source_counts: Dict[str, int] = defaultdict(int)
    evidence_kind_counts: Dict[str, int] = defaultdict(int)
    token_counts: Dict[str, int] = defaultdict(int)
    for node in graph.nodes.values():
        seen_sources = set()
        seen_kinds = set()
        seen_tokens = set()
        for evidence in node.evidence:
            seen_sources.add(str(evidence.source).lower())
            seen_kinds.add(str(evidence.kind).lower())
            seen_tokens.update(_text_tokens(evidence.content))
        for item in seen_sources:
            evidence_source_counts[item] += 1
        for item in seen_kinds:
            evidence_kind_counts[item] += 1
        for token in seen_tokens:
            token_counts[token] += 1

    interventions = {
        object_id: evaluate_local_intervention(graph, object_id)
        for object_id in graph.nodes
    }
    popularity_raw: Dict[str, float] = {}
    exposure_raw: Dict[str, float] = {}
    evidence_specificity_raw: Dict[str, float] = {}
    uniqueness_raw: Dict[str, float] = {}
    downstream_recovery_raw: Dict[str, float] = {}
    symptomness_raw: Dict[str, float] = {}

    for object_id, node in graph.nodes.items():
        incoming = graph.incoming_mass(object_id)
        outgoing = graph.topological_mass(object_id)
        degree = len(graph.adjacency.get(object_id, {})) + sum(
            1 for parent, children in graph.adjacency.items() if object_id in children and parent in graph.nodes
        )
        evidence_count = len(node.evidence)
        modality_presence = sum(
            1.0 for score in (node.metric_score, node.log_score, node.trace_score, node.change_score) if score > 0.05
        )
        popularity = (
            1.0
            + 0.22 * len(node.members)
            + 0.18 * evidence_count
            + 0.20 * degree
            + 0.22 * incoming
            + 0.12 * outgoing
            + 0.10 * modality_presence
        )
        popularity_raw[object_id] = popularity
        exposure_raw[object_id] = node.anomaly_score / max(1.0, popularity)

        specificity_scores: List[float] = []
        for evidence in node.evidence:
            src = str(evidence.source).lower()
            kind = str(evidence.kind).lower()
            tokens = _text_tokens(evidence.content)
            source_specificity = 1.0 / max(1, evidence_source_counts.get(src, 1))
            kind_specificity = 1.0 / max(1, evidence_kind_counts.get(kind, 1))
            if tokens:
                token_specificity = sum(1.0 / max(1, token_counts.get(token, 1)) for token in tokens[:8]) / min(len(tokens), 8)
            else:
                token_specificity = 0.5
            specificity_scores.append(
                max(0.0, min(1.0, 0.45 * source_specificity + 0.20 * kind_specificity + 0.35 * token_specificity))
                * max(0.05, float(evidence.confidence))
            )
        evidence_specificity_raw[object_id] = (
            sum(specificity_scores) / max(EPS, sum(max(0.05, float(item.confidence)) for item in node.evidence))
            if node.evidence else 0.0
        )

    for object_id, node in graph.nodes.items():
        own = interventions[object_id]
        own_impacted = own.get("impacted", {})
        overlap_penalty = 0.0
        for other_id, other in interventions.items():
            if other_id == object_id:
                continue
            overlap = _impact_overlap(own_impacted, other.get("impacted", {}), graph)
            competitor = overlap * other["score"]
            overlap_penalty = max(overlap_penalty, competitor)
        uniqueness_margin = max(0.0, own["score"] - overlap_penalty)
        downstream_ratio = own.get("downstream_removed_mass", 0.0) / max(EPS, own.get("removed_mass", 0.0))
        uniqueness_raw[object_id] = 0.55 * own["score"] + 0.25 * uniqueness_margin + 0.20 * _squash(own.get("exclusivity", 0.0))
        downstream_recovery_raw[object_id] = 0.55 * own.get("coverage", 0.0) + 0.45 * downstream_ratio

        self_locality = own.get("self_removed_mass", 0.0) / max(EPS, own.get("removed_mass", 0.0))
        incoming_dom = incoming = graph.incoming_mass(object_id)
        outgoing = graph.topological_mass(object_id)
        incoming_dom = incoming / max(EPS, incoming + outgoing)
        symptomness_raw[object_id] = (
            0.30 * incoming_dom
            + 0.24 * (1.0 - evidence_specificity_raw[object_id])
            + 0.18 * (1.0 - uniqueness_raw[object_id])
            + 0.16 * self_locality
            + 0.12 * _squash(popularity_raw[object_id] - 1.0)
            - 0.12 * exposure_raw[object_id]
            - 0.12 * downstream_recovery_raw[object_id]
        )

    popularity_bias = _normalize_map(popularity_raw, squash=True)
    exposure_signal = _normalize_map(exposure_raw, squash=False)
    evidence_specificity = _normalize_map(evidence_specificity_raw, squash=False)
    causal_uniqueness = _normalize_map(uniqueness_raw, squash=False)
    downstream_recovery = _normalize_map(downstream_recovery_raw, squash=False)
    symptomness_prior = _normalize_map(symptomness_raw, squash=True)

    scores: Dict[str, Dict[str, float]] = {}
    debug = {"objects": {}}
    for object_id in graph.nodes:
        rootness = (
            0.26 * exposure_signal[object_id]
            + 0.20 * evidence_specificity[object_id]
            + 0.24 * causal_uniqueness[object_id]
            + 0.18 * downstream_recovery[object_id]
            - 0.22 * symptomness_prior[object_id]
            - 0.12 * popularity_bias[object_id]
        )
        scores[object_id] = {
            "exposure_signal": round(exposure_signal[object_id], 6),
            "evidence_specificity": round(evidence_specificity[object_id], 6),
            "causal_uniqueness": round(causal_uniqueness[object_id], 6),
            "downstream_recovery": round(downstream_recovery[object_id], 6),
            "symptomness_prior": round(symptomness_prior[object_id], 6),
            "popularity_bias": round(popularity_bias[object_id], 6),
            "rootness_prior": round(max(0.0, min(1.0, rootness)), 6),
        }
        debug["objects"][object_id] = {
            **scores[object_id],
            "raw_popularity": round(popularity_raw[object_id], 6),
            "raw_exposure": round(exposure_raw[object_id], 6),
            "raw_specificity": round(evidence_specificity_raw[object_id], 6),
            "raw_uniqueness": round(uniqueness_raw[object_id], 6),
            "raw_symptomness": round(symptomness_raw[object_id], 6),
            "intervention": {
                "score": round(interventions[object_id]["score"], 6),
                "coverage": round(interventions[object_id]["coverage"], 6),
                "exclusivity": round(interventions[object_id]["exclusivity"], 6),
                "removed_mass": round(interventions[object_id]["removed_mass"], 6),
            },
        }
    return {"scores": scores, "debug": debug}


def _text_tokens(text: Any) -> List[str]:
    return [
        token for token in re.split(r"[^a-z0-9]+", str(text or "").lower())
        if token and len(token) > 2
    ]


def _impact_overlap(a: Dict[str, float], b: Dict[str, float], graph: ObjectGraph) -> float:
    shared = set(a) & set(b)
    if not shared:
        return 0.0
    overlap = 0.0
    total = 0.0
    for object_id, influence in a.items():
        total += influence * graph.nodes[object_id].anomaly_score
    for object_id in shared:
        overlap += min(a.get(object_id, 0.0), b.get(object_id, 0.0)) * graph.nodes[object_id].anomaly_score
    return overlap / max(EPS, total)


def _normalize_map(values: Dict[str, float], squash: bool) -> Dict[str, float]:
    if not values:
        return {}
    ordered = list(values.items())
    raw = [value for _, value in ordered]
    if squash:
        raw = [_squash(value) for value in raw]
    min_v = min(raw)
    max_v = max(raw)
    if math.isclose(max_v, min_v):
        return {key: 0.5 for key, _ in ordered}
    return {
        key: max(0.0, min(1.0, (value - min_v) / max(EPS, max_v - min_v)))
        for (key, _), value in zip(ordered, raw)
    }


def _squash(value: float) -> float:
    return math.tanh(max(0.0, value))
