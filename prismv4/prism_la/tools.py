"""Structured investigation tools for PRISM-LA.

Each tool returns typed JSON evidence, never raw telemetry. The LLM agent
calls these tools and receives structured `ToolEvidence` objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple
import itertools
import math

import numpy as np

from .state import ToolEvidence

_evidence_counter = itertools.count(1)


def _next_evidence_id(prefix: str) -> str:
    return f"{prefix}:{next(_evidence_counter)}"


def _clip(value: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return float(max(lo, min(hi, value)))


@dataclass
class PRISMLAToolbox:
    entities: Sequence[str]
    entity_types: Dict[str, str]
    metric_signal: np.ndarray = field(default_factory=lambda: np.array([]))
    log_signal: np.ndarray = field(default_factory=lambda: np.array([]))
    graph: np.ndarray = field(default_factory=lambda: np.array([[]]))
    metric_detail: Dict[str, Dict[str, float]] = field(default_factory=dict)
    log_detail: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    trace_detail: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    anomaly_times: Dict[str, float] = field(default_factory=dict)
    anchor_set: List[Dict[str, Any]] = field(default_factory=list)
    candidate_scores: Dict[str, Dict[str, float]] = field(default_factory=dict)
    cmi_profiles: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    cf_profile_cache: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def _idx(self, component: str) -> Optional[int]:
        try:
            return list(self.entities).index(component)
        except ValueError:
            return None

    def find_time_anchors(self) -> Dict[str, Any]:
        sorted_anchors = sorted(
            self.anchor_set,
            key=lambda a: float(a.get("confidence", 0.0)),
            reverse=True,
        )
        return {
            "tool": "find_time_anchors",
            "anchor_count": len(sorted_anchors),
            "top_anchors": sorted_anchors[:5],
        }

    def inspect_metric(self, component: str, anchor_time: Optional[float] = None) -> ToolEvidence:
        idx = self._idx(component)
        signal = float(self.metric_signal[idx]) if idx is not None and idx < len(self.metric_signal) else 0.0
        detail = self.metric_detail.get(component, {})
        top_metrics = sorted(
            [(name, float(value)) for name, value in detail.items()],
            key=lambda item: abs(item[1]),
            reverse=True,
        )[:6]
        cs = self.candidate_scores.get(component, {})
        source_iso = float(cs.get("source_isolation", 0.0) or 0.0)
        symptom = float(cs.get("hotspot_symptom", 0.0) or 0.0) + float(cs.get("symptom_score", 0.0) or 0.0)
        root_penalty = symptom * 0.5
        support = _clip(1.5 * abs(signal) + 0.15 * source_iso - 0.5 * min(root_penalty, 1.0), 0.0, 1.2)
        against = _clip(0.3 * min(1.0, symptom), 0.0, 0.5)
        z_summary = _z_statistics(top_metrics)
        evidence_details = [
            f"{name} z={round(float(val), 2)}" for name, val in top_metrics
        ]
        limitations = []
        if not detail:
            limitations.append("no metric detail available for this component")
        if len(top_metrics) < 2:
            limitations.append("insufficient metric diversity")
        return ToolEvidence(
            evidence_id=_next_evidence_id("metric"),
            tool_name="inspect_metric",
            component=component,
            factor="metric_onset",
            support=support,
            against=against,
            timestamp=anchor_time,
            evidence_details=evidence_details,
            limitations=limitations,
            payload={
                "metric_signal": round(signal, 4),
                "top_anomaly_metrics": top_metrics,
                "z_statistics": z_summary,
            },
        )

    def inspect_log(self, component: str, anchor_time: Optional[float] = None) -> ToolEvidence:
        idx = self._idx(component)
        signal = float(self.log_signal[idx]) if idx is not None and idx < len(self.log_signal) else 0.0
        detail = self.log_detail.get(component, {})
        error_count = int(detail.get("count", 0) or 0)
        fatal_boost = float(detail.get("fatal_boost", 1.0) or 1.0)
        text_excerpt = str(detail.get("text", ""))[:200]
        fatal_keywords = [kw for kw in ("oom", "killed", "exception", "timeout", "refused", "error")
                          if kw in text_excerpt.lower()]

        support = _clip(0.85 * signal + 0.15 * max(0.0, fatal_boost - 1.0), 0.0, 1.0)
        against = 0.0
        if error_count == 0:
            support = 0.0
            against = 0.3

        evidence_details = []
        if error_count > 0:
            evidence_details.append(f"{error_count} error log entries found")
        if fatal_keywords:
            evidence_details.append(f"fatal keywords: {', '.join(fatal_keywords)}")
        if text_excerpt and not fatal_keywords:
            evidence_details.append(f"log excerpt: \"{text_excerpt[:100]}\"")

        limitations = []
        if not detail:
            limitations.append("no log data available for this component")
        if error_count == 0:
            limitations.append("zero error logs found")

        return ToolEvidence(
            evidence_id=_next_evidence_id("log"),
            tool_name="inspect_log",
            component=component,
            factor="log_error_signal",
            support=support,
            against=against,
            timestamp=anchor_time,
            evidence_details=evidence_details,
            limitations=limitations,
            payload={
                "log_signal": round(signal, 4),
                "error_count": error_count,
                "fatal_boost": round(fatal_boost, 4),
                "fatal_keywords": fatal_keywords,
                "text_excerpt": text_excerpt,
            },
        )

    def inspect_trace(self, component: str, anchor_time: Optional[float] = None) -> ToolEvidence:
        idx = self._idx(component)
        graph = np.asarray(self.graph, dtype=float)
        n = len(self.entities)
        if idx is None or idx >= n or graph.size == 0:
            out_strength = 0.0
            in_strength = 0.0
        else:
            out_strength = float(np.sum(graph[idx, :]))
            in_strength = float(np.sum(graph[:, idx]))
        total = out_strength + in_strength
        direction = out_strength / max(total, 1e-9)
        trace_detail = self.trace_detail.get(component, {})
        latency_info = trace_detail.get("median_latency", None)

        support = _clip(0.95 * direction - 0.25 * max(0.0, in_strength - out_strength) / max(total, 1.0), -0.5, 1.0)
        against = _clip(0.3 * (1.0 - direction), 0.0, 0.6)

        evidence_details = []
        if out_strength > 0:
            evidence_details.append(f"out-strength={round(out_strength, 2)}, in-strength={round(in_strength, 2)}")
            evidence_details.append(f"direction ratio={round(direction, 3)} (higher=source)")
        if latency_info is not None:
            evidence_details.append(f"median latency={round(float(latency_info), 2)}ms")

        limitations = []
        if total < 1e-6:
            limitations.append("no trace connectivity for this component")

        return ToolEvidence(
            evidence_id=_next_evidence_id("trace"),
            tool_name="inspect_trace",
            component=component,
            factor="trace_direction",
            support=support,
            against=against,
            timestamp=anchor_time,
            evidence_details=evidence_details,
            limitations=limitations,
            payload={
                "out_strength": round(out_strength, 4),
                "in_strength": round(in_strength, 4),
                "direction_ratio": round(direction, 4),
                "source_likelihood": round(direction, 4),
            },
        )

    def get_topology_neighbors(self, component: str) -> Dict[str, Any]:
        idx = self._idx(component)
        if idx is None:
            return {"tool": "get_topology_neighbors", "component": component, "inbound": [], "outbound": [], "neighbor_count": 0}
        graph = np.asarray(self.graph, dtype=float)
        n = len(self.entities)
        if idx >= n or graph.size == 0:
            return {"tool": "get_topology_neighbors", "component": component, "inbound": [], "outbound": [], "neighbor_count": 0}
        row = graph[idx, :]
        col = graph[:, idx]
        outbound = sorted(
            [{"target": str(self.entities[j]), "weight": round(float(row[j]), 4)}
             for j in range(n) if row[j] > 0 and j != idx],
            key=lambda e: e["weight"],
            reverse=True,
        )[:10]
        inbound = sorted(
            [{"source": str(self.entities[j]), "weight": round(float(col[j]), 4)}
             for j in range(n) if col[j] > 0 and j != idx],
            key=lambda e: e["weight"],
            reverse=True,
        )[:10]
        return {
            "tool": "get_topology_neighbors",
            "component": component,
            "inbound": inbound,
            "outbound": outbound,
            "neighbor_count": len(inbound) + len(outbound),
        }

    def test_counterfactual(self, component: str) -> ToolEvidence:
        profile = (
            self.cf_profile_cache.get(component)
            or self.candidate_scores.get(component)
            or {}
        )
        if profile:
            root_recovery = float(profile.get("root_score", 0.0) or 0.0)
            downstream = float(profile.get("downstream_recovery", 0.0) or 0.0)
            symptom = float(profile.get("symptom_score", 0.0) or 0.0)
            broad = float(profile.get("broad_explainer", 0.0) or 0.0)
        else:
            idx = self._idx(component)
            metric = float(self.metric_signal[idx]) if idx is not None and idx < len(self.metric_signal) else 0.0
            log = float(self.log_signal[idx]) if idx is not None and idx < len(self.log_signal) else 0.0
            candidate_info = self.candidate_scores.get(component, {})
            source_iso = float(candidate_info.get("source_isolation", 0.0) or 0.0)
            root_recovery = 0.45 * abs(metric) + 0.25 * abs(log) + 0.40 * source_iso
            downstream = 0.0
            symptom = float(candidate_info.get("hotspot_symptom", 0.0) or 0.0)
            broad = 0.0

        support = _clip(
            0.55 * root_recovery + 0.25 * downstream - 0.45 * symptom - 0.35 * broad,
            -0.8, 1.0,
        )
        against = 0.0

        evidence_details = [
            f"root recovery: {round(root_recovery, 3)}",
            f"downstream recovery: {round(downstream, 3)}",
            f"symptom penalty: {round(symptom, 3)}",
        ]

        return ToolEvidence(
            evidence_id=_next_evidence_id("cf"),
            tool_name="test_counterfactual",
            component=component,
            factor="counterfactual",
            support=support,
            against=against,
            evidence_details=evidence_details,
            limitations=[] if profile else ["counterfactual profile from proxy signal"],
            payload={
                "root_recovery": round(root_recovery, 4),
                "downstream_recovery": round(downstream, 4),
                "symptom_penalty": round(symptom, 4),
                "broad_penalty": round(broad, 4),
            },
        )

    def explain_residual(self, component: str) -> ToolEvidence:
        n = len(self.entities)
        evidence_vec = np.zeros(n, dtype=float)
        for i in range(n):
            metric_val = float(self.metric_signal[i]) if i < len(self.metric_signal) else 0.0
            log_val = float(self.log_signal[i]) if i < len(self.log_signal) else 0.0
            evidence_vec[i] = 0.70 * abs(metric_val) + 0.30 * abs(log_val)
        total = float(np.sum(evidence_vec))
        if total < 1e-9:
            return ToolEvidence(
                evidence_id=_next_evidence_id("residual"),
                tool_name="explain_residual",
                component=component,
                factor="residual",
                support=0.0,
                against=0.0,
                evidence_details=[],
                limitations=["no anomaly evidence in system"],
                payload={"residual_explained": 0.0, "total_residual": 0.0},
            )
        idx = self._idx(component)
        graph = np.asarray(self.graph, dtype=float)
        if idx is None or graph.size == 0 or graph.shape != (n, n):
            residual_ratio = float(evidence_vec[idx]) / total if idx is not None else 0.0
        else:
            explained = float(evidence_vec[idx])
            graph_norm = graph / max(float(np.max(graph)), 1e-9)
            for j in range(n):
                if j == idx:
                    continue
                spread = float(graph_norm[idx, j]) * float(evidence_vec[idx])
                explained += spread * 0.70
            explained = min(explained, total)
            residual_ratio = explained / total
        support = _clip(1.0 - residual_ratio, -0.5, 1.0)
        against = 0.0
        if residual_ratio > 0.7:
            against = _clip(residual_ratio - 0.5, 0.0, 0.5)
        return ToolEvidence(
            evidence_id=_next_evidence_id("residual"),
            tool_name="explain_residual",
            component=component,
            factor="residual",
            support=support,
            against=against,
            evidence_details=[
                f"residual explained: {round(residual_ratio, 3)}",
                f"unexplained: {round(1.0 - residual_ratio, 3)}",
            ],
            limitations=[],
            payload={
                "residual_explained": round(residual_ratio, 4),
                "total_residual": round(total, 4),
            },
        )

    def compare_events(self, event_a: Dict[str, Any], event_b: Dict[str, Any]) -> List[ToolEvidence]:
        comp_a = event_a.get("component", "")
        comp_b = event_b.get("component", "")
        score_a = self._component_score(comp_a)
        score_b = self._component_score(comp_b)
        gap = _clip(score_a - score_b, -1.0, 1.0)
        return [
            ToolEvidence(
                evidence_id=_next_evidence_id("pair"),
                tool_name="compare_events",
                component=comp_a,
                factor="pairwise_verdict",
                support=max(0.0, gap),
                against=max(0.0, -gap),
                evidence_details=[
                    f"vs {comp_b}: score_gap={round(gap, 3)}",
                    f"{comp_a} score={round(score_a, 3)}",
                    f"{comp_b} score={round(score_b, 3)}",
                ],
                limitations=[],
                payload={"contender": comp_b, "score_gap": round(gap, 4), "verdict": "left" if gap >= 0 else "right"},
            ),
            ToolEvidence(
                evidence_id=_next_evidence_id("pair"),
                tool_name="compare_events",
                component=comp_b,
                factor="pairwise_verdict",
                support=max(0.0, -gap),
                against=max(0.0, gap),
                evidence_details=[
                    f"vs {comp_a}: score_gap={round(-gap, 3)}",
                    f"{comp_b} score={round(score_b, 3)}",
                    f"{comp_a} score={round(score_a, 3)}",
                ],
                limitations=[],
                payload={"contender": comp_a, "score_gap": round(-gap, 4), "verdict": "right" if gap >= 0 else "left"},
            ),
        ]

    def _component_score(self, component: str) -> float:
        idx = self._idx(component)
        metric = float(self.metric_signal[idx]) if idx is not None and idx < len(self.metric_signal) else 0.0
        log = float(self.log_signal[idx]) if idx is not None and idx < len(self.log_signal) else 0.0
        cs = self.candidate_scores.get(component, {})
        source = float(cs.get("source_isolation", 0.0) or 0.0)
        symptom = float(cs.get("hotspot_symptom", 0.0) or 0.0) + float(cs.get("symptom_score", 0.0) or 0.0)
        root = float(cs.get("root_score", 0.0) or 0.0)
        return abs(metric) + 0.35 * abs(log) + 0.60 * source + 0.40 * root - 0.70 * symptom

    def get_entity_type(self, component: str) -> str:
        return str(self.entity_types.get(component, "unknown"))


def _z_statistics(metric_items: List[Tuple[str, float]]) -> Dict[str, Any]:
    if not metric_items:
        return {"max_z": 0.0, "mean_z": 0.0, "significant_count": 0}
    values = [abs(v) for _, v in metric_items]
    return {
        "max_z": round(float(max(values)), 2),
        "mean_z": round(float(np.mean(values)), 2) if values else 0.0,
        "significant_count": sum(1 for v in values if v >= 3.0),
    }
