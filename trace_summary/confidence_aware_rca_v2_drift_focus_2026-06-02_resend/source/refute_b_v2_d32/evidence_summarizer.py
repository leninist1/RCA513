"""Compact evidence summary cards for optional LLM handoff.

The card builder is deliberately lossy: it emits bounded aggregate patterns,
short log examples, and edge-level trace summaries, never raw metric series,
full logs, full traces, labels, or filenames.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math
import re
from typing import Any, Mapping

import pandas as pd

from refute_b_v2_d32.evidence import kpi_in_bucket
from refute_b_v2_d32.schema import reason_bucket
from refute_b_v2_d32.signature import reason_for_kpi, reason_for_log


@dataclass(frozen=True)
class EvidenceSummaryLimits:
    max_metric_patterns: int = 5
    max_log_patterns: int = 5
    max_trace_edges: int = 5
    max_counter_evidence: int = 5


LOG_EXAMPLE_LIMIT = 200
CANONICAL_REASON_BY_BUCKET = {
    "cpu": "CPU fault",
    "memory": "memory fault",
    "jvm_oom": "JVM OOM",
    "disk_io": "disk IO fault",
    "filesystem": "disk space fault",
    "network_latency": "network delay",
    "network_packet_loss": "network loss",
    "db_connection": "db connection limit",
    "process_termination": "process termination",
}
REASON_ALIASES_BY_BUCKET = {
    "cpu": ["CPU fault", "container CPU load", "node CPU load", "node CPU spike", "high CPU usage", "high JVM CPU load"],
    "memory": ["memory fault", "container memory load", "node memory consumption", "high memory usage"],
    "jvm_oom": ["JVM OOM", "JVM out of memory (OOM) heap"],
    "disk_io": ["disk IO fault", "container read I/O load", "container write I/O load", "node disk read I/O consumption", "node disk write I/O consumption"],
    "filesystem": ["disk space fault", "node disk space consumption", "high disk space usage"],
    "network_latency": ["network delay", "network latency", "container network latency"],
    "network_packet_loss": ["network loss", "network packet loss", "container packet loss", "container network packet corruption", "container network packet retransmission"],
    "db_connection": ["db connection limit", "db close"],
    "process_termination": ["process termination", "container process termination"],
}


def build_summary_cards(
    *,
    case_id: str,
    metric_df: pd.DataFrame,
    log_df: pd.DataFrame,
    trace_summary: Mapping[str, Any] | None,
    baseline: Any,
    modal_status: Mapping[str, str],
    d32_debug: Mapping[str, Any],
    window_start_ts: int | None,
    top_k: int = 5,
    limits: EvidenceSummaryLimits | None = None,
) -> list[dict[str, Any]]:
    """Build one JSON-serializable summary card for each top D32 candidate."""

    limits = limits or EvidenceSummaryLimits()
    decisions = list(d32_debug.get("all_decisions", []) or [])[: max(0, int(top_k))]
    onset_ts = _case_onset_ts(d32_debug, window_start_ts)
    signature = dict(d32_debug.get("signature", {}) or {})
    modality_availability = _modality_availability(metric_df, log_df, trace_summary, modal_status)
    case_summary = _case_summary(signature, d32_debug, metric_df, log_df, trace_summary, modality_availability, onset_ts, limits)
    global_metric = _metric_patterns(metric_df, baseline, None, None, onset_ts, limits.max_metric_patterns)
    global_log = _log_patterns(log_df, None, onset_ts, limits.max_log_patterns)
    global_trace = _trace_edges(trace_summary, None, limits.max_trace_edges)

    candidate_infos: list[dict[str, Any]] = []
    for idx, decision in enumerate(decisions, start=1):
        candidate = dict(decision.get("candidate", {}) or {})
        component = str(candidate.get("component", ""))
        reason = str(candidate.get("reason", ""))
        bucket = str(candidate.get("reason_bucket") or reason_bucket(reason))
        reason_identity = _reason_identity(reason, bucket)
        metric_support = _metric_patterns(metric_df, baseline, component, bucket, onset_ts, limits.max_metric_patterns)
        component_metric_signal = _metric_patterns(metric_df, baseline, component, None, onset_ts, limits.max_metric_patterns)
        log_support = _log_patterns(log_df, component, onset_ts, limits.max_log_patterns)
        trace_support = _trace_edges(trace_summary, component, limits.max_trace_edges)
        topology_context = _topology_context(trace_summary, component, limits.max_trace_edges)
        direct_atoms = _direct_evidence_atoms(
            component=component,
            canonical_reason=reason_identity["canonical_reason"],
            bucket=bucket,
            metric_support=metric_support,
            component_metric_signal=component_metric_signal,
            log_support=log_support,
            trace_support=trace_support,
            topology_context=topology_context,
            max_items=max(limits.max_metric_patterns, limits.max_log_patterns, limits.max_trace_edges),
        )
        counter = _counter_evidence(
            decision=decision,
            candidate_rank=idx,
            decisions=decisions,
            global_trace=global_trace,
            metric_support=metric_support,
            log_support=log_support,
            trace_support=trace_support,
            modality_availability=modality_availability,
            component=component,
            reason=reason,
            bucket=bucket,
            max_items=limits.max_counter_evidence,
        )
        competing = _competing_evidence(
            global_metric=global_metric,
            global_log=global_log,
            global_trace=global_trace,
            component=component,
            max_items=limits.max_counter_evidence,
        )
        candidate_infos.append({
            "candidate_rank": idx,
            "decision": decision,
            "component": component,
            "reason": reason,
            "bucket": bucket,
            "reason_identity": reason_identity,
            "metric_support": metric_support,
            "component_metric_signal": component_metric_signal,
            "log_support": log_support,
            "trace_support": trace_support,
            "topology_context": topology_context,
            "direct_atoms": direct_atoms,
            "direct_evidence_score": _direct_evidence_score(direct_atoms),
            "counter": counter,
            "competing": competing,
        })

    cards: list[dict[str, Any]] = []
    for info in candidate_infos:
        reason_identity = info["reason_identity"]
        cards.append({
            "case_id": str(case_id),
            "case_summary": case_summary,
            "candidate_summary": {
                "candidate_rank": info["candidate_rank"],
                "component": info["component"],
                "reason": info["reason"],
                "raw_reason": reason_identity["raw_reason"],
                "canonical_reason": reason_identity["canonical_reason"],
                "known_reason_aliases": reason_identity["known_reason_aliases"],
                "reason_bucket": info["bucket"],
                "d32_score_summary": _score_summary(info["decision"]),
                "metric_support_summary": info["metric_support"],
                "component_metric_signal_summary": info["component_metric_signal"],
                "log_support_summary": info["log_support"],
                "trace_support_summary": info["trace_support"],
                "topology_context_summary": info["topology_context"],
                "candidate_direct_evidence_atoms": info["direct_atoms"],
                "candidate_positive_evidence_summary": _positive_evidence(
                    metric_support=info["metric_support"],
                    component_metric_signal=info["component_metric_signal"],
                    log_support=info["log_support"],
                    trace_support=info["trace_support"],
                    topology_context=info["topology_context"],
                    max_items=max(limits.max_metric_patterns, limits.max_log_patterns, limits.max_trace_edges),
                ),
                "same_reason_sibling_context": _same_reason_sibling_context(info, candidate_infos),
                "counter_evidence_summary": info["counter"],
                "competing_evidence_summary": info["competing"],
                "missing_evidence_summary": _missing_evidence_summary(modality_availability, info["component"], info["reason"]),
            },
        })
    return cards


def _case_summary(
    signature: Mapping[str, Any],
    d32_debug: Mapping[str, Any],
    metric_df: pd.DataFrame,
    log_df: pd.DataFrame,
    trace_summary: Mapping[str, Any] | None,
    modality_availability: Mapping[str, bool],
    onset_ts: int | None,
    limits: EvidenceSummaryLimits,
) -> dict[str, Any]:
    dominant = list(signature.get("dominant_evidence_types", []) or [])
    reason_scores = dict(d32_debug.get("reason_posterior", {}) or {})
    top_reasons = [
        {"reason": str(reason), "score_level": _level_from_value(float(score), 1.0, 3.0)}
        for reason, score in sorted(reason_scores.items(), key=lambda item: (-float(item[1] or 0.0), str(item[0])))[:5]
    ]
    return {
        "suspected_onset": _format_ts(onset_ts),
        "dominant_symptom_type": _dominant_symptom_type(dominant, reason_scores),
        "most_affected_components": _top_components_from_signature(signature, limit=5),
        "global_top_anomaly_patterns": _global_patterns(metric_df, log_df, trace_summary, dominant, limits),
        "dominant_reason_hypotheses": top_reasons,
        "modality_availability": dict(modality_availability),
    }


def _global_patterns(
    metric_df: pd.DataFrame,
    log_df: pd.DataFrame,
    trace_summary: Mapping[str, Any] | None,
    dominant: list[Any],
    limits: EvidenceSummaryLimits,
) -> list[dict[str, Any]]:
    patterns: list[dict[str, Any]] = []
    for item in dominant:
        text = str(item)
        parts = text.split(":", 1)
        patterns.append({"modality": parts[0], "pattern": parts[1] if len(parts) > 1 else text})
    if not patterns and metric_df is not None and not metric_df.empty:
        patterns.append({"modality": "metric", "pattern": "window_metric_activity"})
    if log_df is not None and not log_df.empty:
        patterns.append({"modality": "log", "pattern": "window_log_activity"})
    if trace_summary and trace_summary.get("trace_status") == "present":
        events = trace_summary.get("events", {}) or {}
        if events.get("slow_edges"):
            patterns.append({"modality": "trace", "pattern": "slow_edges"})
        if events.get("dropped_edges"):
            patterns.append({"modality": "trace", "pattern": "dropped_edges"})
    max_items = max(limits.max_metric_patterns, limits.max_log_patterns, limits.max_trace_edges)
    return patterns[:max_items]


def _metric_patterns(
    metric_df: pd.DataFrame,
    baseline: Any,
    component: str | None,
    bucket: str | None,
    onset_ts: int | None,
    max_items: int,
) -> list[dict[str, Any]]:
    if metric_df is None or metric_df.empty or max_items <= 0:
        return []
    rows = metric_df.copy()
    if component:
        rows = rows[rows["cmdb_id"].astype(str) == str(component)]
    if rows.empty:
        return []

    groups: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows.itertuples(index=False):
        row_component = str(getattr(row, "cmdb_id", ""))
        kpi_name = str(getattr(row, "kpi_name", ""))
        if bucket and not kpi_in_bucket(kpi_name, bucket):
            continue
        try:
            value = float(getattr(row, "value"))
            result = baseline.is_anomalous(row_component, kpi_name, value, threshold="p99")
        except Exception:
            continue
        if not getattr(result, "is_anomalous", False):
            continue
        kpi_group = _kpi_group(kpi_name)
        key = (row_component, kpi_group)
        item = groups.setdefault(key, {
            "component": row_component,
            "kpi_group": kpi_group,
            "_timestamps": [],
            "_values": [],
            "_max_deviation": 0.0,
            "reason_relevance": "supports" if bucket and kpi_in_bucket(kpi_name, bucket) else "neutral",
        })
        item["_timestamps"].append(int(getattr(row, "timestamp", 0) or 0))
        item["_values"].append(value)
        item["_max_deviation"] = max(item["_max_deviation"], abs(float(getattr(result, "deviation", 0.0) or 0.0)))

    patterns = []
    for item in groups.values():
        timestamps = item["_timestamps"]
        max_deviation = item["_max_deviation"]
        patterns.append({
            "component": item["component"],
            "kpi_group": item["kpi_group"],
            "trend": _trend(timestamps, item["_values"]),
            "severity": _severity(max_deviation),
            "first_seen_relation": _relation_to_onset(min(timestamps) if timestamps else None, onset_ts),
            "reason_relevance": item["reason_relevance"],
            "_rank": (max_deviation, len(timestamps)),
        })
    patterns.sort(key=lambda row: (-row["_rank"][0], -row["_rank"][1], row["component"], row["kpi_group"]))
    return [_drop_private(row) for row in patterns[:max_items]]


def _log_patterns(log_df: pd.DataFrame, component: str | None, onset_ts: int | None, max_items: int) -> list[dict[str, Any]]:
    if log_df is None or log_df.empty or "value" not in log_df.columns or max_items <= 0:
        return []
    rows = log_df.copy()
    if component:
        rows = rows[rows["cmdb_id"].astype(str) == str(component)]
    if rows.empty:
        return []

    groups: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows.itertuples(index=False):
        text = str(getattr(row, "value", ""))
        pattern = _log_pattern(text)
        if pattern == "unknown" and reason_for_log(text) is None:
            continue
        row_component = str(getattr(row, "cmdb_id", ""))
        key = (row_component, pattern)
        item = groups.setdefault(key, {
            "component": row_component,
            "pattern": pattern,
            "_count": 0,
            "_first_ts": None,
            "short_example": "",
        })
        item["_count"] += 1
        ts = int(getattr(row, "timestamp", 0) or 0)
        if item["_first_ts"] is None or ts < item["_first_ts"]:
            item["_first_ts"] = ts
        if not item["short_example"]:
            item["short_example"] = _short_example(text)

    patterns = []
    for item in groups.values():
        patterns.append({
            "component": item["component"],
            "pattern": item["pattern"],
            "count_level": _count_level(item["_count"]),
            "first_seen_relation": _relation_to_onset(item["_first_ts"], onset_ts),
            "short_example": item["short_example"],
            "_rank": item["_count"],
        })
    patterns.sort(key=lambda row: (-row["_rank"], row["component"], row["pattern"]))
    return [_drop_private(row) for row in patterns[:max_items]]


def _trace_edges(trace_summary: Mapping[str, Any] | None, component: str | None, max_items: int) -> list[dict[str, Any]]:
    if not trace_summary or trace_summary.get("trace_status") != "present" or max_items <= 0:
        return []
    events = trace_summary.get("events", {}) or {}
    rows = []
    for edge in events.get("slow_edges", []) or []:
        rows.append((float(edge.get("slow_ratio", 0.0) or 0.0), edge, "latency_increase"))
    for edge in events.get("dropped_edges", []) or []:
        rows.append((float(edge.get("count_drop_ratio", 0.0) or 0.0), edge, "drop"))

    summaries = []
    for strength, edge, symptom in sorted(rows, key=lambda item: -item[0]):
        src = str(edge.get("src", "unknown"))
        dst = str(edge.get("dst", "unknown"))
        relation = _edge_relation(component, src, dst) if component else "unknown"
        if component and relation == "unrelated":
            continue
        summaries.append({
            "edge": f"{src} -> {dst}",
            "symptom": symptom,
            "severity": _severity(strength),
            "relation_to_candidate": relation,
        })
        if len(summaries) >= max_items:
            break
    return summaries


def _topology_context(trace_summary: Mapping[str, Any] | None, component: str, max_items: int) -> list[dict[str, Any]]:
    if not trace_summary or trace_summary.get("trace_status") != "present" or max_items <= 0:
        return []
    events = trace_summary.get("events", {}) or {}
    out: list[dict[str, Any]] = []
    first = events.get("first_anomalous_service")
    if first:
        out.append({
            "context": "trace_first_anomalous_service",
            "relation_to_candidate": "self" if str(first) == str(component) else "other_component",
        })
    for edge in _trace_edges(trace_summary, component, max_items):
        out.append({"context": "trace_edge_neighbor", "edge": edge["edge"], "relation_to_candidate": edge["relation_to_candidate"]})
        if len(out) >= max_items:
            break
    return out[:max_items]


def _reason_identity(reason: str, bucket: str) -> dict[str, Any]:
    canonical = CANONICAL_REASON_BY_BUCKET.get(str(bucket), str(reason))
    aliases = REASON_ALIASES_BY_BUCKET.get(str(bucket), [canonical])
    if str(reason) and str(reason) not in aliases:
        aliases = [str(reason)] + list(aliases)
    return {
        "raw_reason": str(reason),
        "canonical_reason": canonical,
        "known_reason_aliases": list(dict.fromkeys(str(item) for item in aliases)),
    }


def _direct_evidence_atoms(
    *,
    component: str,
    canonical_reason: str,
    bucket: str,
    metric_support: list[Mapping[str, Any]],
    component_metric_signal: list[Mapping[str, Any]],
    log_support: list[Mapping[str, Any]],
    trace_support: list[Mapping[str, Any]],
    topology_context: list[Mapping[str, Any]],
    max_items: int,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []

    def add_atom(**kwargs: Any) -> None:
        if len(out) >= max(0, int(max_items)):
            return
        atom_id = f"atom_{len(out) + 1}"
        out.append({"atom_id": atom_id, **kwargs})

    for item in metric_support:
        add_atom(
            modality="metric",
            directness="component_and_reason",
            component=str(component),
            canonical_reason=str(canonical_reason),
            reason_bucket=str(bucket),
            pattern=str(item.get("kpi_group", "")),
            severity=str(item.get("severity", "unknown")),
            first_seen_relation=str(item.get("first_seen_relation", "unknown")),
            promotion_eligible=True,
        )
    supported_metric_keys = {(str(item.get("component", "")), str(item.get("kpi_group", ""))) for item in metric_support}
    for item in component_metric_signal:
        key = (str(item.get("component", "")), str(item.get("kpi_group", "")))
        if key in supported_metric_keys:
            continue
        relation = str(item.get("first_seen_relation", "unknown"))
        severity = str(item.get("severity", "unknown"))
        reason_relevance = str(item.get("reason_relevance", "neutral"))
        add_atom(
            modality="metric",
            directness="component_only",
            component=str(component),
            canonical_reason=str(canonical_reason),
            reason_bucket=str(bucket),
            pattern=str(item.get("kpi_group", "")),
            reason_relevance=reason_relevance,
            severity=severity,
            first_seen_relation=relation,
            promotion_eligible=False,
        )
    for item in log_support:
        add_atom(
            modality="log",
            directness="component_only",
            component=str(component),
            canonical_reason=str(canonical_reason),
            reason_bucket=str(bucket),
            pattern=str(item.get("pattern", "")),
            count_level=str(item.get("count_level", "unknown")),
            first_seen_relation=str(item.get("first_seen_relation", "unknown")),
            promotion_eligible=str(item.get("count_level", "unknown")) in {"high", "medium"},
        )
    for item in trace_support:
        relation = str(item.get("relation_to_candidate", "unknown"))
        add_atom(
            modality="trace",
            directness="topology_neighbor" if relation in {"incoming", "outgoing"} else "component_only",
            component=str(component),
            canonical_reason=str(canonical_reason),
            reason_bucket=str(bucket),
            edge=str(item.get("edge", "")),
            symptom=str(item.get("symptom", "")),
            severity=str(item.get("severity", "unknown")),
            relation_to_candidate=relation,
            promotion_eligible=relation in {"incoming", "outgoing", "self"},
        )
    for item in topology_context:
        relation = str(item.get("relation_to_candidate", "unknown"))
        if relation not in {"self", "incoming", "outgoing"}:
            continue
        add_atom(
            modality="topology",
            directness="component_only" if relation == "self" else "topology_neighbor",
            component=str(component),
            canonical_reason=str(canonical_reason),
            reason_bucket=str(bucket),
            context=str(item.get("context", "")),
            edge=str(item.get("edge", "")),
            relation_to_candidate=relation,
            promotion_eligible=relation == "self",
        )
    return out


def _same_reason_sibling_context(info: Mapping[str, Any], candidate_infos: list[Mapping[str, Any]]) -> dict[str, Any]:
    same_bucket = [
        item for item in candidate_infos
        if str(item.get("bucket", "")) == str(info.get("bucket", ""))
    ]
    if not same_bucket:
        return {
            "reason_bucket": str(info.get("bucket", "")),
            "canonical_reason": str((info.get("reason_identity", {}) or {}).get("canonical_reason", "")),
            "sibling_count": 0,
            "candidate_component_affected_rank": None,
            "candidate_vs_best_sibling": "unknown",
            "has_stronger_sibling": False,
        }

    ranked = sorted(
        same_bucket,
        key=lambda item: (-float(item.get("direct_evidence_score", 0.0) or 0.0), int(item.get("candidate_rank", 9999))),
    )
    candidate_rank = next(
        (idx for idx, item in enumerate(ranked, start=1) if int(item.get("candidate_rank", -1)) == int(info.get("candidate_rank", -2))),
        None,
    )
    best = ranked[0]
    candidate_score = float(info.get("direct_evidence_score", 0.0) or 0.0)
    best_score = float(best.get("direct_evidence_score", 0.0) or 0.0)
    delta = candidate_score - best_score
    if candidate_rank == 1:
        relation = "strongest"
    elif abs(delta) <= 0.5:
        relation = "similar"
    else:
        relation = "weaker"
    return {
        "reason_bucket": str(info.get("bucket", "")),
        "canonical_reason": str((info.get("reason_identity", {}) or {}).get("canonical_reason", "")),
        "sibling_count": len(same_bucket),
        "candidate_component_affected_rank": candidate_rank,
        "candidate_direct_evidence_level": _level_from_value(candidate_score, 2.0, 5.0),
        "candidate_direct_evidence_score_level": _level_from_value(candidate_score, 2.0, 5.0),
        "best_sibling_component": str(best.get("component", "")),
        "best_sibling_original_rank": int(best.get("candidate_rank", 0) or 0),
        "best_sibling_direct_evidence_level": _level_from_value(best_score, 2.0, 5.0),
        "candidate_vs_best_sibling": relation,
        "has_stronger_sibling": bool(candidate_rank and candidate_rank > 1 and best_score > candidate_score + 0.5),
    }


def _direct_evidence_score(atoms: list[Mapping[str, Any]]) -> float:
    directness_weight = {
        "component_and_reason": 3.0,
        "component_only": 0.6,
        "topology_neighbor": 0.8,
    }
    severity_weight = {"high": 3.0, "medium": 2.0, "low": 1.0, "unknown": 0.5}
    relation_bonus = {"near_onset": 0.4, "before_onset": 0.2, "after_onset": -0.2, "unknown": 0.0}
    score = 0.0
    for atom in atoms:
        directness = str(atom.get("directness", "component_only"))
        severity = str(atom.get("severity", "unknown"))
        relation = str(atom.get("first_seen_relation", "unknown"))
        base = directness_weight.get(directness, 0.5) * severity_weight.get(severity, 0.5)
        if atom.get("promotion_eligible") is True:
            base += 0.5
        score += max(0.0, base + relation_bonus.get(relation, 0.0))
    return float(score)


def _counter_evidence(
    *,
    decision: Mapping[str, Any],
    candidate_rank: int,
    decisions: list[Mapping[str, Any]],
    global_trace: list[Mapping[str, Any]],
    metric_support: list[Mapping[str, Any]],
    log_support: list[Mapping[str, Any]],
    trace_support: list[Mapping[str, Any]],
    modality_availability: Mapping[str, bool],
    component: str,
    reason: str,
    bucket: str,
    max_items: int,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if max_items <= 0:
        return out
    if candidate_rank > 1 and decisions:
        top_candidate = dict((decisions[0].get("candidate") if decisions else {}) or {})
        out.append({
            "type": "other_candidate_ranked_higher",
            "component": str(top_candidate.get("component", "")),
            "reason": str(top_candidate.get("reason", "")),
        })
    refute_strength = float(decision.get("refute_strength", 0.0) or 0.0)
    if refute_strength > 0:
        out.append({"type": "explicit_rule_refute_signal", "severity": _severity(refute_strength)})
    if modality_availability.get("metric", False) and not metric_support and bucket not in {"db_connection", "process_termination"}:
        out.append({"type": "candidate_reason_missing_metric_support", "reason": str(reason)})
    if modality_availability.get("log", False) and not log_support and bucket in {"db_connection", "jvm_oom", "network_latency", "network_packet_loss"}:
        out.append({"type": "candidate_reason_missing_log_support", "reason": str(reason)})
    if modality_availability.get("trace", False) and not trace_support and bucket in {"network_latency", "network_packet_loss"} and global_trace:
        out.append({"type": "trace_signal_not_centered_on_candidate", "reason": str(reason)})
    return out[:max_items]


def _positive_evidence(
    *,
    metric_support: list[Mapping[str, Any]],
    component_metric_signal: list[Mapping[str, Any]],
    log_support: list[Mapping[str, Any]],
    trace_support: list[Mapping[str, Any]],
    topology_context: list[Mapping[str, Any]],
    max_items: int,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in metric_support:
        out.append({
            "type": "reason_relevant_metric_signal",
            "modality": "metric",
            "component": str(item.get("component", "")),
            "pattern": str(item.get("kpi_group", "")),
            "severity": str(item.get("severity", "unknown")),
            "first_seen_relation": str(item.get("first_seen_relation", "unknown")),
        })
    supported_metric_keys = {(str(item.get("component", "")), str(item.get("kpi_group", ""))) for item in metric_support}
    for item in component_metric_signal:
        key = (str(item.get("component", "")), str(item.get("kpi_group", "")))
        if key in supported_metric_keys:
            continue
        out.append({
            "type": "component_metric_signal",
            "modality": "metric",
            "component": str(item.get("component", "")),
            "pattern": str(item.get("kpi_group", "")),
            "severity": str(item.get("severity", "unknown")),
            "reason_relevance": str(item.get("reason_relevance", "neutral")),
            "first_seen_relation": str(item.get("first_seen_relation", "unknown")),
        })
    for item in log_support:
        out.append({
            "type": "component_log_signal",
            "modality": "log",
            "component": str(item.get("component", "")),
            "pattern": str(item.get("pattern", "")),
            "count_level": str(item.get("count_level", "unknown")),
            "first_seen_relation": str(item.get("first_seen_relation", "unknown")),
        })
    for item in trace_support:
        out.append({
            "type": "candidate_trace_edge_signal",
            "modality": "trace",
            "edge": str(item.get("edge", "")),
            "symptom": str(item.get("symptom", "")),
            "severity": str(item.get("severity", "unknown")),
            "relation_to_candidate": str(item.get("relation_to_candidate", "unknown")),
        })
    for item in topology_context:
        relation = str(item.get("relation_to_candidate", "unknown"))
        if relation in {"self", "incoming", "outgoing"}:
            out.append({
                "type": "candidate_topology_signal",
                "modality": "topology",
                "context": str(item.get("context", "")),
                "edge": str(item.get("edge", "")),
                "relation_to_candidate": relation,
            })
    return out[: max(0, int(max_items))]


def _competing_evidence(
    *,
    global_metric: list[Mapping[str, Any]],
    global_log: list[Mapping[str, Any]],
    global_trace: list[Mapping[str, Any]],
    component: str,
    max_items: int,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if max_items <= 0:
        return out
    for pattern in global_metric:
        if str(pattern.get("component")) != str(component):
            out.append({
                "type": "other_component_metric_signal",
                "component": str(pattern.get("component", "")),
                "pattern": str(pattern.get("kpi_group", "")),
                "severity": str(pattern.get("severity", "unknown")),
                "interpretation": "context_not_direct_refutation",
            })
    for pattern in global_log:
        if str(pattern.get("component")) != str(component):
            out.append({
                "type": "other_component_log_signal",
                "component": str(pattern.get("component", "")),
                "pattern": str(pattern.get("pattern", "")),
                "interpretation": "context_not_direct_refutation",
            })
    for pattern in global_trace:
        edge = str(pattern.get("edge", ""))
        if str(component) and str(component) not in edge:
            out.append({
                "type": "other_component_trace_signal",
                "edge": edge,
                "symptom": str(pattern.get("symptom", "")),
                "severity": str(pattern.get("severity", "unknown")),
                "interpretation": "context_not_direct_refutation",
            })
    return out[:max_items]


def _missing_evidence_summary(modality_availability: Mapping[str, bool], component: str, reason: str) -> list[dict[str, Any]]:
    out = []
    for modality in ("metric", "log", "trace"):
        if not modality_availability.get(modality, False):
            out.append({
                "modality": modality,
                "status": "unavailable_or_empty",
                "impact": "not_used_as_counter_evidence",
                "component": str(component),
                "reason": str(reason),
            })
    return out


def _score_summary(decision: Mapping[str, Any]) -> dict[str, str]:
    support = float(decision.get("support_strength", 0.0) or 0.0)
    refute = float(decision.get("refute_strength", 0.0) or 0.0)
    confidence = str(decision.get("confidence", "unknown")).lower()
    return {
        "support_level": _level_from_value(support, 1.0, 3.0),
        "rebuttal_level": _level_from_value(refute, 1.0, 3.0),
        "reason_confidence_level": confidence if confidence in {"high", "medium", "low"} else "unknown",
    }


def _case_onset_ts(d32_debug: Mapping[str, Any], window_start_ts: int | None) -> int | None:
    timestamps = []
    for row in d32_debug.get("selected_time_anchors", []) or []:
        try:
            timestamps.append(int((row.get("time_anchor") or {}).get("timestamp")))
        except (TypeError, ValueError):
            continue
    if timestamps:
        return min(timestamps)
    return int(window_start_ts) if window_start_ts is not None else None


def _top_components_from_signature(signature: Mapping[str, Any], limit: int) -> list[dict[str, Any]]:
    scored = []
    for service in signature.get("services", []) or []:
        score = 0.0
        modalities = []
        for modality in ("metric", "log", "trace", "topology"):
            for item in (service.get(modality, {}) or {}).values():
                score += abs(float(item.get("strength", 0.0) or 0.0)) + float(item.get("intensity", 0.0) or 0.0)
                modalities.append(modality)
        if score > 0:
            scored.append((score, str(service.get("service", "")), sorted(set(modalities))))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [
        {"component": component, "evidence_level": _level_from_value(score, 5.0, 15.0), "modalities": modalities}
        for score, component, modalities in scored[:limit]
    ]


def _modality_availability(
    metric_df: pd.DataFrame,
    log_df: pd.DataFrame,
    trace_summary: Mapping[str, Any] | None,
    modal_status: Mapping[str, str],
) -> dict[str, bool]:
    return {
        "metric": bool(modal_status.get("metric") == "present" and metric_df is not None and not metric_df.empty),
        "log": bool(modal_status.get("log") == "present" and log_df is not None and not log_df.empty),
        "trace": bool(modal_status.get("trace") == "present" and trace_summary and trace_summary.get("trace_status") == "present"),
    }


def _dominant_symptom_type(dominant: list[Any], reason_scores: Mapping[str, Any]) -> str:
    text = " ".join([str(item).lower() for item in dominant] + [str(item).lower() for item in reason_scores])
    if any(token in text for token in ("latency", "slow")):
        return "latency"
    if any(token in text for token in ("loss", "drop", "error", "connection", "timeout", "oom")):
        return "error"
    if any(token in text for token in ("cpu", "memory", "disk", "filesystem", "io")):
        return "resource"
    if any(token in text for token in ("availability", "termination", "restart")):
        return "availability"
    return "unknown"


def _kpi_group(kpi_name: str) -> str:
    reason = reason_for_kpi(kpi_name)
    if reason:
        return reason_bucket(reason)
    low = str(kpi_name).lower()
    if "cpu" in low or "load" in low:
        return "cpu"
    if "mem" in low or "heap" in low:
        return "memory"
    if "disk" in low or "read" in low or "write" in low:
        return "disk_io"
    if "net" in low or "tcp" in low or "packet" in low:
        return "network"
    return "unknown"


def _trend(timestamps: list[int], values: list[float]) -> str:
    if len(values) < 2:
        return "unknown"
    ordered = [value for _, value in sorted(zip(timestamps, values))]
    first = ordered[0]
    last = ordered[-1]
    mean = sum(ordered) / len(ordered)
    if mean == 0:
        return "unknown"
    rel_span = abs((max(ordered) - min(ordered)) / mean)
    if len(ordered) >= 4 and rel_span > 1.0:
        return "oscillation"
    rel_delta = (last - first) / (abs(first) + 1e-9)
    if rel_delta > 0.5:
        return "ramp" if len(ordered) > 3 else "step_up"
    if rel_delta < -0.5:
        return "drop"
    if max(ordered) > mean * 2.0 and len(ordered) >= 3:
        return "spike"
    return "unknown"


def _log_pattern(text: str) -> str:
    low = str(text).lower()
    if any(token in low for token in ("timeout", "timed out")):
        return "timeout"
    if any(token in low for token in ("error", "exception", "failed", "failure", "refused")):
        return "error"
    if "retry" in low:
        return "retry"
    if any(token in low for token in ("connection", "connect", "reset", "broken pipe")):
        return "connection"
    if any(token in low for token in ("oom", "memory", "heap", "cpu", "disk", "space")):
        return "resource"
    return "unknown"


def _short_example(text: str) -> str:
    compact = re.sub(r"\s+", " ", str(text)).strip()
    return compact[:LOG_EXAMPLE_LIMIT]


def _edge_relation(component: str | None, src: str, dst: str) -> str:
    if not component:
        return "unknown"
    if str(src) == str(component):
        return "outgoing"
    if str(dst) == str(component):
        return "incoming"
    return "unrelated"


def _relation_to_onset(first_ts: int | None, onset_ts: int | None) -> str:
    if first_ts is None or onset_ts is None:
        return "unknown"
    delta = int(first_ts) - int(onset_ts)
    if delta < -120:
        return "before_onset"
    if abs(delta) <= 120:
        return "near_onset"
    return "after_onset"


def _severity(value: float) -> str:
    try:
        val = abs(float(value))
    except (TypeError, ValueError):
        return "unknown"
    if val >= 10.0:
        return "high"
    if val >= 3.0:
        return "medium"
    return "low"


def _count_level(count: int) -> str:
    if count >= 20:
        return "high"
    if count >= 5:
        return "medium"
    return "low"


def _level_from_value(value: float, medium: float, high: float) -> str:
    if not math.isfinite(float(value)):
        return "unknown"
    if value >= high:
        return "high"
    if value >= medium:
        return "medium"
    if value > 0:
        return "low"
    return "unknown"


def _format_ts(ts: int | None) -> str:
    if ts is None:
        return "unknown"
    try:
        return datetime.fromtimestamp(int(ts)).strftime("%Y-%m-%d %H:%M:%S")
    except (OverflowError, OSError, ValueError):
        return "unknown"


def _drop_private(row: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): value for key, value in row.items() if not str(key).startswith("_")}
