"""Structured observation tools for the noise-native agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple
import itertools
import math

import numpy as np

from .fault_event import EvidenceObservation, FaultEvent


_OBS_COUNTER = itertools.count(1)


def _evidence_id(prefix: str) -> str:
    return f"{prefix}:{next(_OBS_COUNTER)}"


def _clip(value: float, lo: float = -2.0, hi: float = 2.0) -> float:
    return float(max(lo, min(hi, value)))


@dataclass
class ToolContext:
    entities: Sequence[str]
    metric_signal: np.ndarray
    log_signal: np.ndarray
    graph: np.ndarray
    metric_detail: Dict[str, Dict[str, float]]
    log_detail: Dict[str, Dict[str, Any]]
    anomaly_times: Dict[str, float]
    candidate_scores: Dict[str, Dict[str, float]] = field(default_factory=dict)
    cf_profile_cache: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    cmi_profiles: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    event_registry: Dict[str, FaultEvent] = field(default_factory=dict)

    def index(self, component: str) -> Optional[int]:
        try:
            return list(self.entities).index(component)
        except ValueError:
            return None


class NoiseNativeToolbox:
    def __init__(self, context: ToolContext) -> None:
        self.context = context

    def inspect_metric(self, event: FaultEvent) -> EvidenceObservation:
        idx = self.context.index(event.component)
        score = float(self.context.metric_signal[idx]) if idx is not None else 0.0
        metrics = self.context.metric_detail.get(event.component, {})
        delta = _clip(1.35 * score + 0.08 * min(len(metrics), 4), 0.0, 1.7)
        return EvidenceObservation(
            evidence_id=_evidence_id("metric"),
            tool_name="inspect_metric",
            event_id=event.event_id,
            component=event.component,
            factor_name="metric_likelihood",
            factor_delta=delta,
            payload={
                "metric_signal": round(score, 6),
                "top_metrics": sorted(
                    [
                        {"metric": name, "score": round(float(value), 6)}
                        for name, value in metrics.items()
                    ],
                    key=lambda item: item["score"],
                    reverse=True,
                )[:5],
            },
        )

    def inspect_log(self, event: FaultEvent) -> EvidenceObservation:
        idx = self.context.index(event.component)
        score = float(self.context.log_signal[idx]) if idx is not None else 0.0
        detail = self.context.log_detail.get(event.component, {})
        fatal_boost = float(detail.get("fatal_boost", 1.0) or 1.0)
        delta = _clip(1.10 * score + 0.12 * max(0.0, fatal_boost - 1.0), 0.0, 1.4)
        return EvidenceObservation(
            evidence_id=_evidence_id("log"),
            tool_name="inspect_log",
            event_id=event.event_id,
            component=event.component,
            factor_name="log_likelihood",
            factor_delta=delta,
            payload={
                "log_signal": round(score, 6),
                "error_count": int(detail.get("count", 0) or 0),
                "fatal_boost": round(fatal_boost, 6),
                "text_excerpt": str(detail.get("text", ""))[:240],
            },
        )

    def verify_trace(self, event: FaultEvent) -> EvidenceObservation:
        idx = self.context.index(event.component)
        if idx is None or self.context.graph.size == 0:
            out_strength = 0.0
            in_strength = 0.0
        else:
            out_strength = float(np.sum(self.context.graph[idx, :]))
            in_strength = float(np.sum(self.context.graph[:, idx]))
        direction = out_strength / max(out_strength + in_strength, 1e-9)
        delta = _clip(0.85 * direction - 0.35 * max(0.0, in_strength - out_strength), -0.7, 1.0)
        return EvidenceObservation(
            evidence_id=_evidence_id("trace"),
            tool_name="verify_trace",
            event_id=event.event_id,
            component=event.component,
            factor_name="trace_direction_likelihood",
            factor_delta=delta,
            payload={
                "out_strength": round(out_strength, 6),
                "in_strength": round(in_strength, 6),
                "source_directionality": round(direction, 6),
            },
        )

    def run_counterfactual(self, event: FaultEvent) -> EvidenceObservation:
        profile = (
            self.context.cf_profile_cache.get(event.component)
            or self.context.candidate_scores.get(event.component)
            or {}
        )
        collapse = self._residual_collapse_profile(event)
        root_score = float(profile.get("root_score", 0.0) or 0.0)
        downstream = float(profile.get("downstream_recovery", 0.0) or 0.0)
        symptom = float(profile.get("symptom_score", 0.0) or 0.0)
        broad = float(profile.get("broad_explainer", 0.0) or 0.0)
        isolation = max(0.0, float(event.factors.get("source_isolation", 0.0) or 0.0))
        hotspot = max(0.0, float(event.factors.get("hotspot_symptom", 0.0) or 0.0))
        if not profile:
            idx = self.context.index(event.component)
            metric = float(self.context.metric_signal[idx]) if idx is not None else 0.0
            log = float(self.context.log_signal[idx]) if idx is not None else 0.0
            root_score = 0.45 * metric + 0.25 * log + 0.40 * isolation
            downstream = 0.0
            symptom = 0.50 * hotspot
        cf_likelihood = _clip(
            0.55 * root_score
            + 0.65 * collapse["residual_collapse"]
            + 0.45 * collapse["downstream_collapse"]
            + 0.25 * downstream
            + 0.20 * isolation
            - 0.50 * symptom
            - 0.45 * hotspot
            - 0.75 * collapse["hotspot_self_collapse"]
            - 0.55 * collapse["alternative_explainability"]
            - 0.40 * broad
        )
        return EvidenceObservation(
            evidence_id=_evidence_id("cf"),
            tool_name="run_counterfactual",
            event_id=event.event_id,
            component=event.component,
            factor_name="counterfactual_likelihood",
            factor_delta=cf_likelihood,
            payload={
                "cf_likelihood": round(cf_likelihood, 6),
                "local_recovery": round(root_score, 6),
                "downstream_recovery": round(downstream, 6),
                "broad_explainer_penalty": round(broad, 6),
                "symptom_score": round(symptom, 6),
                "source_isolation": round(isolation, 6),
                "hotspot_symptom": round(hotspot, 6),
                "residual_collapse": round(collapse["residual_collapse"], 6),
                "downstream_collapse": round(collapse["downstream_collapse"], 6),
                "hotspot_self_collapse": round(collapse["hotspot_self_collapse"], 6),
                "alternative_explainability": round(collapse["alternative_explainability"], 6),
                "remaining_residual_ratio": round(collapse["remaining_residual_ratio"], 6),
                "collapse_scope": collapse["collapse_scope"],
                "collapse_model": "graph_residual_explainability",
                "runtime_cost": 0.0,
                "source": "cached_profile" if profile else "telemetry_proxy",
            },
        )

    def mechanism_intervention(self, event: FaultEvent) -> List[EvidenceObservation]:
        profile = self.context.cmi_profiles.get(event.component) or {}
        if not profile:
            return [
                EvidenceObservation(
                    evidence_id=_evidence_id("cmi"),
                    tool_name="mechanism_intervention",
                    event_id=event.event_id,
                    component=event.component,
                    factor_name="mechanism_break_likelihood",
                    factor_delta=0.0,
                    payload={"cmi_available": False},
                )
            ]
        payload = {
            "cmi_available": True,
            "cmi_score": round(float(profile.get("cmi_score", 0.0) or 0.0), 6),
            "cmi_root_admissible": bool(profile.get("cmi_root_admissible", False)),
            "marginal_delta_z": round(float(profile.get("marginal_delta_z", 0.0) or 0.0), 6),
            "conditional_residual_z": round(float(profile.get("conditional_residual_z", 0.0) or 0.0), 6),
            "conditional_residual_z_norm": round(float(profile.get("conditional_residual_z_norm", 0.0) or 0.0), 6),
            "parent_explainability": round(float(profile.get("parent_explainability", 0.0) or 0.0), 6),
            "counterfactual_effect_coverage": round(float(profile.get("counterfactual_effect_coverage", 0.0) or 0.0), 6),
            "counterfactual_effect_coverage_norm": round(float(profile.get("counterfactual_effect_coverage_norm", 0.0) or 0.0), 6),
            "alternative_explainability": round(float(profile.get("alternative_explainability", 0.0) or 0.0), 6),
            "repair_uniqueness": round(float(profile.get("repair_uniqueness", 0.0) or 0.0), 6),
            "repair_uniqueness_norm": round(float(profile.get("repair_uniqueness_norm", 0.0) or 0.0), 6),
            "conditioners": list(profile.get("conditioners", []) or [])[:6],
            "effect_scope": list(profile.get("effect_scope", []) or [])[:8],
            "best_effect_observer": str(profile.get("best_effect_observer", "") or ""),
            "model": "causal_mechanism_intervention",
        }
        return [
            EvidenceObservation(
                evidence_id=_evidence_id("cmi"),
                tool_name="mechanism_intervention",
                event_id=event.event_id,
                component=event.component,
                factor_name="mechanism_break_likelihood",
                factor_delta=_clip(float(profile.get("mechanism_break_likelihood", 0.0) or 0.0), -1.0, 1.6),
                payload=payload,
            ),
            EvidenceObservation(
                evidence_id=_evidence_id("cmi"),
                tool_name="mechanism_intervention",
                event_id=event.event_id,
                component=event.component,
                factor_name="mechanism_parent_refutation",
                factor_delta=_clip(float(profile.get("mechanism_parent_refutation", 0.0) or 0.0), -1.0, 1.0),
                payload=payload,
            ),
            EvidenceObservation(
                evidence_id=_evidence_id("cmi"),
                tool_name="mechanism_intervention",
                event_id=event.event_id,
                component=event.component,
                factor_name="intervention_uniqueness",
                factor_delta=_clip(float(profile.get("intervention_uniqueness", 0.0) or 0.0), -0.7, 1.3),
                payload=payload,
            ),
        ]

    def residual_collapse(self, event: FaultEvent) -> EvidenceObservation:
        collapse = self._residual_collapse_profile(event)
        delta = _clip(
            1.15 * collapse["residual_collapse"]
            + 0.75 * collapse["downstream_collapse"]
            - 0.85 * collapse["hotspot_self_collapse"]
            - 0.70 * collapse["alternative_explainability"],
            -1.2,
            1.4,
        )
        return EvidenceObservation(
            evidence_id=_evidence_id("collapse"),
            tool_name="residual_collapse",
            event_id=event.event_id,
            component=event.component,
            factor_name="residual_collapse",
            factor_delta=delta,
            payload={
                **{
                    key: (
                        round(float(value), 6)
                        if isinstance(value, (float, int))
                        else value
                    )
                    for key, value in collapse.items()
                },
                "collapse_likelihood": round(delta, 6),
                "collapse_model": "graph_residual_explainability",
            },
        )

    def compare_pair(self, left: FaultEvent, right: FaultEvent) -> List[EvidenceObservation]:
        left_score = self._pair_score(left)
        right_score = self._pair_score(right)
        gap = _clip(left_score - right_score, -1.0, 1.0)
        return [
            EvidenceObservation(
                evidence_id=_evidence_id("pair"),
                tool_name="compare_pair",
                event_id=left.event_id,
                component=left.component,
                factor_name="pairwise_verdict",
                factor_delta=gap,
                payload={
                    "contender": right.component,
                    "pairwise_verdict": "left" if gap >= 0 else "right",
                    "score_gap": round(gap, 6),
                },
            ),
            EvidenceObservation(
                evidence_id=_evidence_id("pair"),
                tool_name="compare_pair",
                event_id=right.event_id,
                component=right.component,
                factor_name="pairwise_verdict",
                factor_delta=-gap,
                payload={
                    "contender": left.component,
                    "pairwise_verdict": "right" if gap >= 0 else "left",
                    "score_gap": round(-gap, 6),
                },
            ),
        ]

    def split_event(self, event: FaultEvent) -> EvidenceObservation:
        reason_count = len([r for r in event.reason.split("|") if r.strip()])
        delta = 0.10 if reason_count > 1 else 0.0
        return EvidenceObservation(
            evidence_id=_evidence_id("split"),
            tool_name="split_event",
            event_id=event.event_id,
            component=event.component,
            factor_name="split_event",
            factor_delta=delta,
            payload={"reason_count": reason_count, "split_recommended": reason_count > 1},
        )

    def merge_events(self, left: FaultEvent, right: FaultEvent) -> List[EvidenceObservation]:
        same_component = left.component == right.component
        same_reason = bool(left.reason and left.reason == right.reason)
        should_merge = same_component and same_reason
        delta = -0.35 if should_merge else 0.0
        return [
            EvidenceObservation(
                evidence_id=_evidence_id("merge"),
                tool_name="merge_events",
                event_id=right.event_id,
                component=right.component,
                factor_name="merge_duplicate",
                factor_delta=delta,
                payload={
                    "merge_into": left.event_id if should_merge else "",
                    "same_component": same_component,
                    "same_reason": same_reason,
                },
            )
        ]

    def _pair_score(self, event: FaultEvent) -> float:
        idx = self.context.index(event.component)
        metric = float(self.context.metric_signal[idx]) if idx is not None else 0.0
        log = float(self.context.log_signal[idx]) if idx is not None else 0.0
        graph_out = float(np.sum(self.context.graph[idx, :])) if idx is not None else 0.0
        profile = self.context.candidate_scores.get(event.component, {})
        root_score = float(profile.get("root_score", 0.0) or 0.0)
        symptom = float(profile.get("symptom_score", 0.0) or 0.0)
        source = max(0.0, float(event.factors.get("source_likelihood", 0.0) or 0.0))
        isolation = max(0.0, float(event.factors.get("source_isolation", 0.0) or 0.0))
        hotspot = max(0.0, float(event.factors.get("hotspot_symptom", 0.0) or 0.0))
        return (
            metric
            + 0.55 * log
            + 0.20 * graph_out
            + 0.45 * root_score
            + 0.45 * source
            + 0.75 * isolation
            - 0.45 * symptom
            - 0.90 * hotspot
        )

    def _residual_collapse_profile(self, event: FaultEvent) -> Dict[str, Any]:
        idx = self.context.index(event.component)
        n = len(self.context.entities)
        evidence = self._evidence_vector()
        total_evidence = max(float(np.sum(evidence)), 1e-9)
        if idx is None or n == 0:
            return self._empty_collapse_profile()

        graph = np.asarray(self.context.graph, dtype=float)
        if graph.shape != (n, n):
            graph = np.zeros((n, n), dtype=float)
        descendants = self._descendants(idx, graph)
        parents = self._parents(idx, graph)
        scope = [idx] + [node for node in descendants if node != idx]
        if not scope:
            scope = [idx]

        target_evidence = float(evidence[idx])
        downstream_evidence = float(np.sum(evidence[descendants])) if descendants else 0.0
        scope_evidence = float(np.sum(evidence[scope]))

        explained_by_target = self._explained_mass(idx, graph, evidence, scope)
        alt_explained = 0.0
        for other_idx in range(n):
            if other_idx == idx:
                continue
            alt_explained += self._pair_explainability(other_idx, idx, graph, evidence)
            for node in descendants:
                if node == other_idx:
                    continue
                alt_explained += 0.35 * self._pair_explainability(other_idx, node, graph, evidence)
        alt_explained = min(scope_evidence, alt_explained)

        parent_explained = 0.0
        for parent in parents:
            parent_explained += self._pair_explainability(parent, idx, graph, evidence)
        parent_explainability = parent_explained / max(target_evidence, 1e-9)

        downstream_collapse = (
            max(0.0, explained_by_target - target_evidence)
            / max(downstream_evidence, 1e-9)
            if downstream_evidence > 1e-9
            else 0.0
        )
        residual_collapse = max(0.0, explained_by_target - alt_explained) / total_evidence
        alternative_explainability = alt_explained / max(scope_evidence, 1e-9)
        self_fraction = target_evidence / max(scope_evidence, 1e-9)
        hotspot_self_collapse = self_fraction * min(1.0, max(alternative_explainability, parent_explainability))
        remaining_residual_ratio = max(
            0.0,
            (total_evidence - max(0.0, explained_by_target - alt_explained))
            / total_evidence,
        )
        return {
            "residual_collapse": float(np.clip(residual_collapse, 0.0, 1.0)),
            "downstream_collapse": float(np.clip(downstream_collapse, 0.0, 1.0)),
            "hotspot_self_collapse": float(np.clip(hotspot_self_collapse, 0.0, 1.0)),
            "alternative_explainability": float(np.clip(alternative_explainability, 0.0, 1.0)),
            "remaining_residual_ratio": float(np.clip(remaining_residual_ratio, 0.0, 1.0)),
            "collapse_scope": [
                str(self.context.entities[node]) for node in scope[:8]
            ],
        }

    def _empty_collapse_profile(self) -> Dict[str, Any]:
        return {
            "residual_collapse": 0.0,
            "downstream_collapse": 0.0,
            "hotspot_self_collapse": 0.0,
            "alternative_explainability": 0.0,
            "remaining_residual_ratio": 1.0,
            "collapse_scope": [],
        }

    def _evidence_vector(self) -> np.ndarray:
        metric = np.asarray(self.context.metric_signal, dtype=float)
        log = np.asarray(self.context.log_signal, dtype=float)
        size = max(metric.size, log.size, len(self.context.entities))
        out = np.zeros(size, dtype=float)
        if metric.size:
            out[: metric.size] += 0.70 * np.clip(metric, 0.0, None)
        if log.size:
            out[: log.size] += 0.30 * np.clip(log, 0.0, None)
        return out[: len(self.context.entities)]

    def _descendants(self, idx: int, graph: np.ndarray) -> List[int]:
        seen = set()
        frontier = [idx]
        while frontier and len(seen) < len(self.context.entities):
            current = frontier.pop(0)
            for child in np.argsort(graph[current, :])[::-1]:
                child = int(child)
                if child == idx or child in seen or graph[current, child] <= 0:
                    continue
                seen.add(child)
                frontier.append(child)
        return list(seen)

    def _parents(self, idx: int, graph: np.ndarray) -> List[int]:
        if graph.size == 0:
            return []
        return [int(parent) for parent in np.where(graph[:, idx] > 0)[0]]

    def _explained_mass(
        self, source_idx: int, graph: np.ndarray, evidence: np.ndarray, scope: Sequence[int]
    ) -> float:
        mass = float(evidence[source_idx])
        for node in scope:
            if node == source_idx:
                continue
            mass += self._pair_explainability(source_idx, int(node), graph, evidence)
        return mass

    def _pair_explainability(
        self, source_idx: int, target_idx: int, graph: np.ndarray, evidence: np.ndarray
    ) -> float:
        if source_idx == target_idx:
            return float(evidence[target_idx])
        direct = float(graph[source_idx, target_idx]) if graph.size else 0.0
        if direct <= 0:
            return 0.0
        source_strength = float(evidence[source_idx])
        target_strength = float(evidence[target_idx])
        source_gate = source_strength / max(source_strength + target_strength, 1e-9)
        return target_strength * min(1.0, direct) * source_gate
