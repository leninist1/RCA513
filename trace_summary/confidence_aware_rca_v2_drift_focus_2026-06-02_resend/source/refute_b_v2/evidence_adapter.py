"""Adapters that expose abstract evidence queries to the rule engine."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from refute_b_v2.rule_engine import QueryResult
from refute_b_v2.trace_propagation import trace_evidence_for_service

SPECIAL_KPI_PATTERNS = {
    "jvm_heap": ("JVM-Memory", "HeapMemory", "Tomcat-MEMORY"),
    "redis_memory": ("used_memory", "used_memory_peak", "used_memory_rss", "mem_fragmentation_ratio", "redis-Redis", "Redis_"),
    "mysql_memory": ("Qcache Free Memory", "max trx lock memory bytes", "Mysql-MySQL_3306_"),
}


class SummaryBackedEvidence:
    """Delegate metric/log evidence to an existing EvidenceQuery and use trace summaries for trace evidence."""

    def __init__(self, base_evidence: Any, trace_summary: dict | None = None):
        self.base_evidence = base_evidence
        self.trace_summary = trace_summary or {}

    @classmethod
    def with_summary_path(cls, base_evidence: Any, path: str | Path | None) -> "SummaryBackedEvidence":
        if not path:
            return cls(base_evidence, None)
        path = Path(path)
        if not path.exists():
            return cls(base_evidence, None)
        with path.open("r", encoding="utf-8") as f:
            return cls(base_evidence, json.load(f))

    def is_container_kpi_anomalous(self, service: str, bucket: str):
        return self.base_evidence.is_container_kpi_anomalous(service, bucket)

    def is_node_kpi_anomalous(self, service: str, bucket: str):
        return self.base_evidence.is_node_kpi_anomalous(service, bucket)

    def does_match_log_keyword(self, service: str, keywords):
        return self.base_evidence.does_match_log_keyword(service, keywords)

    def is_service_trace_anomalous(self, service: str) -> QueryResult:
        if not self.trace_summary:
            if hasattr(self.base_evidence, "is_service_trace_anomalous"):
                return self.base_evidence.is_service_trace_anomalous(service)
            return QueryResult(False, "trace summary missing", 0.0, unavailable=True)
        if self.trace_summary.get("trace_status") != "present":
            return QueryResult(False, f"trace summary {self.trace_summary.get('trace_status')}", 0.0, unavailable=True)
        stats = self.trace_summary.get("service_stats", {}).get(str(service))
        if not stats:
            return QueryResult(False, f"no trace summary rows for {service}", 0.0)
        p95 = float(stats.get("duration_p95", 0.0) or 0.0)
        return QueryResult(p95 > 0, f"{service} trace p95={p95:.2f}", p95, stats)

    def has_slow_trace_edge(self, service: str) -> QueryResult:
        if not self.trace_summary:
            if hasattr(self.base_evidence, "has_slow_trace_edge"):
                return self.base_evidence.has_slow_trace_edge(service)
            return QueryResult(False, "trace summary missing", 0.0, unavailable=True)
        return _edge_event_for_service(self.trace_summary, service, "slow_edges", "slow_ratio")

    def has_trace_edge_count_drop(self, service: str) -> QueryResult:
        return _edge_event_for_service(self.trace_summary, service, "dropped_edges", "count_drop_ratio", scale=10.0)

    def is_trace_first_anomalous_service(self, service: str) -> QueryResult:
        if not self.trace_summary:
            return QueryResult(False, "trace summary missing", 0.0, unavailable=True)
        if self.trace_summary.get("trace_status") != "present":
            return QueryResult(False, f"trace summary {self.trace_summary.get('trace_status')}", 0.0, unavailable=True)
        first = self.trace_summary.get("events", {}).get("first_anomalous_service")
        if not first:
            return QueryResult(False, "no first anomalous service in trace summary", 0.0)
        matched = str(first) == str(service)
        return QueryResult(matched, f"first anomalous trace service is {first}", 2.0 if matched else 0.0, {"first_anomalous_service": first})

    def has_trace_propagation_role(self, service: str, roles) -> QueryResult:
        if not self.trace_summary:
            return QueryResult(False, "trace summary missing", 0.0, unavailable=True)
        if self.trace_summary.get("trace_status") != "present":
            return QueryResult(False, f"trace summary {self.trace_summary.get('trace_status')}", 0.0, unavailable=True)
        info = trace_evidence_for_service(self.trace_summary, service)
        role = (info.get("role") or {}).get("role")
        allowed = {str(item) for item in roles}
        if not role:
            return QueryResult(False, f"no propagation role for {service}", 0.0, info)
        matched = str(role) in allowed
        strength = 2.0 if role == "upstream_source" else 1.0
        return QueryResult(matched, f"{service} trace propagation role is {role}", strength if matched else 0.0, info)

    def is_specialty_kpi_anomalous(self, service: str, bucket: str) -> QueryResult:
        patterns = SPECIAL_KPI_PATTERNS.get(str(bucket), ())
        if not patterns:
            return QueryResult(False, f"unknown specialty KPI bucket: {bucket}", 0.0, unavailable=True)
        base = self.base_evidence
        if getattr(base, "_modality_unavailable")("metric"):
            return QueryResult(False, f"metric modality {base.modal_status.get('metric')} for case", 0.0, unavailable=True)
        rows = base.metric_df[base.metric_df["cmdb_id"] == str(service)]
        if rows.empty:
            return QueryResult(False, f"no metric rows for {service}", 0.0)
        rows = rows[rows["kpi_name"].map(lambda k: any(p in str(k) for p in patterns)).astype(bool)]
        if rows.empty:
            return QueryResult(False, f"no {bucket} KPI rows for {service}", 0.0, {"checked_service_rows": int(len(base.metric_df[base.metric_df["cmdb_id"] == str(service)]))})
        return base._max_anomaly(rows)


def _edge_event_for_service(summary: dict, service: str, event_key: str, strength_key: str,
                            scale: float = 1.0) -> QueryResult:
    if not summary:
        return QueryResult(False, "trace summary missing", 0.0, unavailable=True)
    if summary.get("trace_status") != "present":
        return QueryResult(False, f"trace summary {summary.get('trace_status')}", 0.0, unavailable=True)
    events = summary.get("events", {}).get(event_key, [])
    service = str(service)
    hits = [
        edge for edge in events
        if str(edge.get("src")) == service or str(edge.get("dst")) == service
    ]
    if not hits:
        return QueryResult(False, f"no {event_key} involving {service}", 0.0, {"checked": len(events)})
    best = max(hits, key=lambda edge: float(edge.get(strength_key, 0.0) or 0.0))
    role = "src" if str(best.get("src")) == service else "dst"
    strength = float(best.get(strength_key, 0.0) or 0.0) * scale
    text = (
        f"{event_key} {best.get('src')}->{best.get('dst')} "
        f"{strength_key}={float(best.get(strength_key, 0.0) or 0.0):.3g}"
    )
    details = {**best, "role": role, "event": event_key}
    details["propagation"] = trace_evidence_for_service(summary, service)
    return QueryResult(True, text, strength, details)
