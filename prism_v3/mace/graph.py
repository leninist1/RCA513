"""Object-centric causal graph construction for MACE-RCA."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
import math
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..config import QueryCase, UnifiedTelemetry
from ..counterfactual.graph import build_graph_from_traces, infer_graph_from_metrics


RESOURCE_REASON_MAP = {
    "cpu": "CPU fault",
    "memory": "high memory usage",
    "network": "network fault",
    "disk": "disk I/O consumption",
    "db": "db fault",
    "process": "process termination",
    "change": "configuration change",
    "latency": "network latency",
}

RESOURCE_KEYWORDS = {
    "cpu": ("cpu", "load", "throttle", "util"),
    "memory": ("mem", "memory", "rss", "heap", "oom", "gc"),
    "network": ("timeout", "latency", "network", "packet", "drop", "tcp", "rtt"),
    "disk": ("disk", "iowait", "iops", "storage", "fs", "await"),
    "db": ("db", "jdbc", "mysql", "postgres", "redis", "sql"),
    "process": ("restart", "killed", "crash", "exit", "sigkill"),
    "change": ("change", "deploy", "release", "config", "rollback"),
    "latency": ("slow", "latency", "timeout", "duration"),
}


@dataclass
class EvidenceRecord:
    kind: str
    source: str
    content: str
    confidence: float
    timestamp: Optional[float] = None


@dataclass
class ObjectNode:
    object_id: str
    members: List[str] = field(default_factory=list)
    representative: str = ""
    anomaly_score: float = 0.0
    earliest_timestamp: Optional[float] = None
    metric_score: float = 0.0
    log_score: float = 0.0
    trace_score: float = 0.0
    change_score: float = 0.0
    reason_votes: Dict[str, float] = field(default_factory=dict)
    evidence: List[EvidenceRecord] = field(default_factory=list)

    def best_reason(self) -> str:
        if not self.reason_votes:
            return "high memory usage"
        key = max(self.reason_votes, key=self.reason_votes.get)
        return RESOURCE_REASON_MAP.get(key, "high memory usage")


@dataclass
class ObjectGraph:
    nodes: Dict[str, ObjectNode]
    adjacency: Dict[str, Dict[str, float]]

    def topological_mass(self, object_id: str) -> float:
        down = self.adjacency.get(object_id, {})
        return sum(self.nodes[child].anomaly_score * weight for child, weight in down.items() if child in self.nodes)

    def incoming_mass(self, object_id: str) -> float:
        score = 0.0
        for parent, children in self.adjacency.items():
            if object_id in children and parent in self.nodes:
                score += self.nodes[parent].anomaly_score * children[object_id]
        return score


def build_object_graph(
    telemetry: UnifiedTelemetry,
    query: QueryCase,
    inject_time: Optional[float],
) -> Tuple[ObjectGraph, Dict[str, Any]]:
    """Construct an object-centric causal graph from raw telemetry."""
    metrics = telemetry.metrics if telemetry.metrics is not None else pd.DataFrame()
    logs = telemetry.logs if telemetry.logs is not None else pd.DataFrame()
    traces = telemetry.traces if telemetry.traces is not None else pd.DataFrame()

    baseline_df, fault_df = _split_temporal(metrics, inject_time)
    trace_graph = build_graph_from_traces(traces, inject_time=inject_time)
    metric_graph = infer_graph_from_metrics(
        baseline_df if not baseline_df.empty else pd.DataFrame(),
        fault_df if not fault_df.empty else pd.DataFrame(),
        telemetry.entities,
    )

    entity_stats = _collect_entity_stats(metrics, logs, traces, inject_time)
    if not entity_stats:
        nodes = {
            entity: ObjectNode(
                object_id=entity,
                members=[entity],
                representative=entity,
                anomaly_score=0.01,
            )
            for entity in telemetry.entities
        }
        return ObjectGraph(nodes=nodes, adjacency={}), {"fallback": "empty_entity_stats"}

    keep_instance_suffix = telemetry.system in {"Telecom", "Market"}
    entity_to_object = {
        entity: _canonical_object_name(entity, keep_instance_suffix=keep_instance_suffix)
        for entity in entity_stats
    }
    object_members: Dict[str, List[str]] = defaultdict(list)
    for entity, object_id in entity_to_object.items():
        object_members[object_id].append(entity)

    object_nodes: Dict[str, ObjectNode] = {}
    for object_id, members in object_members.items():
        member_stats = [entity_stats[m] for m in members]
        representative = max(members, key=lambda m: entity_stats[m]["score"])
        anomaly_score = float(max(stat["score"] for stat in member_stats))
        earliest_timestamp = _min_timestamp(stat["earliest_ts"] for stat in member_stats)
        node = ObjectNode(
            object_id=object_id,
            members=sorted(members),
            representative=representative,
            anomaly_score=anomaly_score,
            earliest_timestamp=earliest_timestamp,
            metric_score=float(max(stat["metric_score"] for stat in member_stats)),
            log_score=float(max(stat["log_score"] for stat in member_stats)),
            trace_score=float(max(stat["trace_score"] for stat in member_stats)),
            change_score=float(max(stat["change_score"] for stat in member_stats)),
            reason_votes=_merge_reason_votes([stat["reason_votes"] for stat in member_stats]),
            evidence=_merge_evidence([stat["evidence"] for stat in member_stats], limit=8),
        )
        object_nodes[object_id] = node

    adjacency = _aggregate_graphs(
        object_nodes=object_nodes,
        entity_to_object=entity_to_object,
        graphs=[trace_graph, metric_graph],
    )
    adjacency = _enforce_temporal_direction(adjacency, object_nodes)

    debug = {
        "entity_to_object": entity_to_object,
        "objects": {
            obj: {
                "members": node.members,
                "representative": node.representative,
                "anomaly_score": round(node.anomaly_score, 4),
                "metric_score": round(node.metric_score, 4),
                "log_score": round(node.log_score, 4),
                "trace_score": round(node.trace_score, 4),
                "change_score": round(node.change_score, 4),
                "reason": node.best_reason(),
            }
            for obj, node in object_nodes.items()
        },
        "query": query.task_index,
    }
    return ObjectGraph(nodes=object_nodes, adjacency=adjacency), debug


def _split_temporal(metrics: pd.DataFrame, inject_time: Optional[float]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if metrics is None or metrics.empty or inject_time is None or "timestamp" not in metrics.columns:
        return pd.DataFrame(), pd.DataFrame()
    baseline = metrics[(metrics["timestamp"] >= inject_time - 300) & (metrics["timestamp"] < inject_time)]
    fault = metrics[(metrics["timestamp"] >= inject_time) & (metrics["timestamp"] <= inject_time + 300)]
    return baseline, fault


def _collect_entity_stats(
    metrics: pd.DataFrame,
    logs: pd.DataFrame,
    traces: pd.DataFrame,
    inject_time: Optional[float],
) -> Dict[str, Dict[str, Any]]:
    stats: Dict[str, Dict[str, Any]] = {}
    for entity in _all_entities(metrics, logs, traces):
        stats[entity] = {
            "metric_score": 0.0,
            "log_score": 0.0,
            "trace_score": 0.0,
            "change_score": 0.0,
            "score": 0.0,
            "earliest_ts": None,
            "reason_votes": defaultdict(float),
            "evidence": [],
        }

    if not metrics.empty:
        for entity, group in metrics.groupby("entity"):
            entity = str(entity)
            metric_score, earliest_ts, reason_votes, evidence = _metric_entity_profile(group, inject_time)
            stats[entity]["metric_score"] = metric_score
            stats[entity]["earliest_ts"] = earliest_ts
            stats[entity]["score"] += 0.60 * metric_score
            stats[entity]["reason_votes"] = _merge_reason_votes([stats[entity]["reason_votes"], reason_votes])
            stats[entity]["evidence"].extend(evidence)

    if not logs.empty:
        for entity, group in logs.groupby("entity"):
            entity = str(entity)
            log_score, earliest_ts, reason_votes, change_score, evidence = _log_entity_profile(group, inject_time)
            stats[entity]["log_score"] = log_score
            stats[entity]["change_score"] = change_score
            stats[entity]["earliest_ts"] = _min_timestamp([stats[entity]["earliest_ts"], earliest_ts])
            stats[entity]["score"] += 0.25 * log_score + 0.10 * change_score
            stats[entity]["reason_votes"] = _merge_reason_votes([stats[entity]["reason_votes"], reason_votes])
            stats[entity]["evidence"].extend(evidence)

    if not traces.empty and "entity" in traces.columns:
        for entity, group in traces.groupby("entity"):
            entity = str(entity)
            trace_score, earliest_ts, evidence = _trace_entity_profile(group, inject_time)
            stats[entity]["trace_score"] = trace_score
            stats[entity]["earliest_ts"] = _min_timestamp([stats[entity]["earliest_ts"], earliest_ts])
            stats[entity]["score"] += 0.15 * trace_score
            stats[entity]["evidence"].extend(evidence)

    for entity, payload in stats.items():
        payload["score"] = float(min(1.0, payload["score"]))
        if payload["earliest_ts"] is None and inject_time is not None:
            payload["earliest_ts"] = inject_time
    _rescale_entity_stats(stats)
    return stats


def _metric_entity_profile(group: pd.DataFrame, inject_time: Optional[float]) -> Tuple[float, Optional[float], Dict[str, float], List[EvidenceRecord]]:
    metric_scores: List[float] = []
    earliest = None
    reason_votes: Dict[str, float] = defaultdict(float)
    evidence: List[EvidenceRecord] = []
    if "metric_name" not in group.columns or "value" not in group.columns:
        return 0.0, earliest, reason_votes, evidence

    for metric_name, metric_df in group.groupby("metric_name"):
        values = pd.to_numeric(metric_df["value"], errors="coerce").dropna()
        if len(values) < 4:
            continue
        baseline, fault = _split_series(metric_df, inject_time)
        if baseline.empty or fault.empty:
            continue
        base_mean = float(baseline.mean())
        base_std = float(baseline.std()) if len(baseline) > 1 else 0.0
        fault_mean = float(fault.mean())
        if not math.isfinite(base_mean) or not math.isfinite(fault_mean):
            continue
        z = abs(fault_mean - base_mean) / max(base_std, 1e-6)
        rel = abs(fault_mean - base_mean) / max(abs(base_mean), 1.0)
        metric_score = float(np.tanh(0.08 * z + 0.30 * rel))
        if metric_score <= 0:
            continue
        metric_scores.append(metric_score)
        ts = _first_fault_timestamp(metric_df, inject_time)
        earliest = _min_timestamp([earliest, ts])
        for family, keywords in RESOURCE_KEYWORDS.items():
            if any(token in str(metric_name).lower() for token in keywords):
                reason_votes[family] += metric_score
        evidence.append(EvidenceRecord(
            kind="metric",
            source=str(metric_name),
            content=f"{metric_name}: baseline={base_mean:.3f}, fault={fault_mean:.3f}, z={z:.2f}",
            confidence=metric_score,
            timestamp=ts,
        ))
    if metric_scores:
        top_scores = sorted(metric_scores, reverse=True)[:3]
        score = float(np.mean(top_scores))
    else:
        score = 0.0
    return min(score, 1.0), earliest, reason_votes, evidence[:6]


def _log_entity_profile(group: pd.DataFrame, inject_time: Optional[float]) -> Tuple[float, Optional[float], Dict[str, float], float, List[EvidenceRecord]]:
    reason_votes: Dict[str, float] = defaultdict(float)
    evidence: List[EvidenceRecord] = []
    if "message" not in group.columns:
        return 0.0, None, reason_votes, 0.0, evidence
    if inject_time is not None and "timestamp" in group.columns:
        group = group[(group["timestamp"] >= inject_time - 300) & (group["timestamp"] <= inject_time + 300)]
    if group.empty:
        return 0.0, None, reason_votes, 0.0, evidence

    severity = 0.0
    change_score = 0.0
    earliest = None
    for _, row in group.head(80).iterrows():
        msg = str(row.get("message", "")).lower()
        ts = float(row["timestamp"]) if pd.notna(row.get("timestamp")) else None
        line_score = 0.0
        for family, keywords in RESOURCE_KEYWORDS.items():
            hits = sum(1 for kw in keywords if kw in msg)
            if hits:
                weight = min(1.0, 0.25 * hits)
                reason_votes[family] += weight
                line_score = max(line_score, weight)
        if any(token in msg for token in ("error", "exception", "timeout", "refused", "failed", "killed", "oom")):
            line_score = max(line_score, 0.45)
        if any(token in msg for token in RESOURCE_KEYWORDS["change"]):
            change_score = max(change_score, 0.7)
        if line_score <= 0:
            continue
        earliest = _min_timestamp([earliest, ts])
        severity = max(severity, line_score)
        evidence.append(EvidenceRecord(
            kind="log",
            source="logs",
            content=str(row.get("message", ""))[:200],
            confidence=min(1.0, line_score),
            timestamp=ts,
        ))
    return min(severity, 1.0), earliest, reason_votes, min(change_score, 1.0), evidence[:6]


def _trace_entity_profile(group: pd.DataFrame, inject_time: Optional[float]) -> Tuple[float, Optional[float], List[EvidenceRecord]]:
    if inject_time is not None and "timestamp" in group.columns:
        group = group[(group["timestamp"] >= inject_time - 300) & (group["timestamp"] <= inject_time + 300)]
    if group.empty:
        return 0.0, None, []
    err_col = "status_code" if "status_code" in group.columns else None
    durations = pd.to_numeric(group.get("duration"), errors="coerce") if "duration" in group.columns else pd.Series(dtype=float)
    err_rate = 0.0
    if err_col:
        err_rate = float(np.mean(group[err_col].astype(str).str.startswith(("4", "5"))))
    p95 = float(np.nanpercentile(durations.dropna(), 95)) / 1000.0 if not durations.dropna().empty else 0.0
    score = float(np.tanh(err_rate * 3.0 + p95 / 2.0))
    earliest = float(group["timestamp"].min()) if "timestamp" in group.columns else None
    evidence = []
    if score > 0:
        evidence.append(EvidenceRecord(
            kind="trace",
            source="traces",
            content=f"trace_err_rate={err_rate:.2f}, p95_duration_s={p95:.2f}",
            confidence=score,
            timestamp=earliest,
        ))
    return min(score, 1.0), earliest, evidence


def _aggregate_graphs(
    object_nodes: Dict[str, ObjectNode],
    entity_to_object: Dict[str, str],
    graphs: List[Dict[str, Dict[str, float]]],
) -> Dict[str, Dict[str, float]]:
    adjacency: Dict[str, Dict[str, float]] = defaultdict(dict)
    for graph in graphs:
        for src, children in graph.items():
            if src not in entity_to_object:
                continue
            obj_src = entity_to_object[src]
            for dst, weight in children.items():
                if dst not in entity_to_object:
                    continue
                obj_dst = entity_to_object[dst]
                if obj_src == obj_dst or obj_src not in object_nodes or obj_dst not in object_nodes:
                    continue
                adjacency[obj_src][obj_dst] = max(adjacency[obj_src].get(obj_dst, 0.0), float(weight))
    return {src: dict(children) for src, children in adjacency.items()}


def _enforce_temporal_direction(
    adjacency: Dict[str, Dict[str, float]],
    nodes: Dict[str, ObjectNode],
) -> Dict[str, Dict[str, float]]:
    cleaned: Dict[str, Dict[str, float]] = defaultdict(dict)
    for src, children in adjacency.items():
        for dst, weight in children.items():
            t_src = nodes[src].earliest_timestamp
            t_dst = nodes[dst].earliest_timestamp
            if t_src is not None and t_dst is not None and t_src > t_dst + 120:
                continue
            cleaned[src][dst] = weight
    return {src: dict(children) for src, children in cleaned.items()}


def format_timestamp(ts: Optional[float], fallback: str) -> str:
    if ts is None:
        return fallback
    try:
        return datetime.fromtimestamp(float(ts)).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return fallback


def _canonical_object_name(entity: Any, keep_instance_suffix: bool = False) -> str:
    text = str(entity or "").strip().lower()
    text = re.sub(r"\b(pod|container|service|svc|instance|deployment|host|server)\b", "", text)
    text = re.sub(r"-[0-9a-f]{6,}$", "", text)
    if not keep_instance_suffix:
        text = re.sub(r"-\d+$", "", text)
    text = re.sub(r"\.[0-9a-f]{6,}$", "", text)
    text = re.sub(r"[_\-.]{2,}", "-", text)
    text = re.sub(r"[^a-z0-9\-_.]", "-", text)
    text = text.strip("-._")
    if not text:
        return str(entity or "unknown").lower()
    if keep_instance_suffix and re.fullmatch(r"[a-z][a-z0-9]*[-_]\d+", text):
        return text
    segments = [seg for seg in re.split(r"[._]", text) if seg]
    if keep_instance_suffix:
        filtered = [seg for seg in segments if not re.fullmatch(r"[0-9a-f]{6,}", seg)]
    else:
        filtered = [
            seg
            for seg in segments
            if not re.fullmatch(r"[0-9a-f]{6,}", seg) and not seg.isdigit()
        ]
    return ".".join(filtered[:3]) if filtered else text


def _all_entities(metrics: pd.DataFrame, logs: pd.DataFrame, traces: pd.DataFrame) -> List[str]:
    entities = set()
    for frame in (metrics, logs, traces):
        if frame is not None and not frame.empty and "entity" in frame.columns:
            entities.update(str(v) for v in frame["entity"].dropna().unique())
    return sorted(entities)


def _split_series(metric_df: pd.DataFrame, inject_time: Optional[float]) -> Tuple[pd.Series, pd.Series]:
    vals = pd.to_numeric(metric_df["value"], errors="coerce")
    if inject_time is None or "timestamp" not in metric_df.columns:
        midpoint = max(1, len(vals) // 2)
        return vals.iloc[:midpoint].dropna(), vals.iloc[midpoint:].dropna()
    baseline = vals[(metric_df["timestamp"] >= inject_time - 300) & (metric_df["timestamp"] < inject_time)]
    fault = vals[(metric_df["timestamp"] >= inject_time) & (metric_df["timestamp"] <= inject_time + 300)]
    return baseline.dropna(), fault.dropna()


def _first_fault_timestamp(metric_df: pd.DataFrame, inject_time: Optional[float]) -> Optional[float]:
    if "timestamp" not in metric_df.columns:
        return None
    if inject_time is None:
        return float(metric_df["timestamp"].min())
    frame = metric_df[(metric_df["timestamp"] >= inject_time) & (metric_df["timestamp"] <= inject_time + 300)]
    if frame.empty:
        frame = metric_df
    return float(frame["timestamp"].min()) if not frame.empty else None


def _min_timestamp(values) -> Optional[float]:
    filtered = [float(v) for v in values if v is not None and pd.notna(v)]
    return min(filtered) if filtered else None


def _merge_reason_votes(vote_dicts: List[Dict[str, float]]) -> Dict[str, float]:
    merged: Dict[str, float] = defaultdict(float)
    for payload in vote_dicts:
        for key, value in dict(payload).items():
            merged[key] += float(value)
    return dict(merged)


def _merge_evidence(evidence_lists: List[List[EvidenceRecord]], limit: int) -> List[EvidenceRecord]:
    merged = []
    for records in evidence_lists:
        merged.extend(records)
    merged.sort(key=lambda item: item.confidence, reverse=True)
    return merged[:limit]


def _rescale_entity_stats(stats: Dict[str, Dict[str, Any]]) -> None:
    raw_scores = np.array([payload["score"] for payload in stats.values()], dtype=float)
    if raw_scores.size == 0:
        return
    scale = float(np.percentile(raw_scores, 90))
    if scale <= 1e-6:
        return
    for payload in stats.values():
        payload["score"] = float(min(1.0, payload["score"] / scale))
        payload["metric_score"] = float(min(1.0, payload["metric_score"] / scale))
        payload["log_score"] = float(min(1.0, payload["log_score"] / max(0.35, scale)))
        payload["trace_score"] = float(min(1.0, payload["trace_score"] / max(0.35, scale)))
