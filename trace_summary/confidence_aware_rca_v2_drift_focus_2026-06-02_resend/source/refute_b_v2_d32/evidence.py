"""d32 Layer 2 structured evidence queries."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import pandas as pd

from refute.src.node_container_split import is_node_level_kpi_bank, is_node_level_kpi_market
from refute_b_v2.trace_propagation import trace_evidence_for_service
from refute_b_v2_d32.bucket_resolver import row_in_bucket
from refute_b_v2_d32.signature import CPU_TOKENS, DISK_TOKENS, FS_TOKENS, MEM_TOKENS, NET_TOKENS


@dataclass(frozen=True)
class EvidenceResult:
    matched: bool
    evidence: str
    strength: float = 0.0
    details: Mapping[str, Any] | None = None
    unavailable: bool = False


def kpi_in_bucket(kpi_name: str, bucket: str) -> bool:
    text = str(kpi_name)
    low = text.lower()
    packet_loss_tokens = ("error", "err", "drop", "loss", "retrans", "reject", "rejected", "reset", "abort", "aborted")
    network_latency_tokens = ("network", "packet", "packets", "traffic", "tcp", "icmp", "ping", "recv", "send", "sent", "queue", "tnsping", "net")
    if bucket == "network_packet_loss":
        return any(token in low for token in packet_loss_tokens)
    if bucket in {"network", "network_latency"}:
        return any(token in low for token in network_latency_tokens)
    patterns = {
        "cpu": CPU_TOKENS,
        "memory": MEM_TOKENS,
        "jvm_oom": MEM_TOKENS,
        "disk_io": DISK_TOKENS,
        "filesystem": FS_TOKENS,
    }.get(bucket, (bucket,))
    return any(token in text for token in patterns)


class D32EvidenceQuery:
    def __init__(
        self,
        metric_df: pd.DataFrame,
        log_df: pd.DataFrame,
        trace_summary: Mapping[str, Any] | None,
        baseline,
        node_graph: Mapping[str, Any],
        modal_status: Mapping[str, str],
        threshold: str = "p99",
    ):
        self.metric_df = metric_df.copy() if metric_df is not None else pd.DataFrame(columns=["timestamp", "cmdb_id", "kpi_name", "value"])
        if not self.metric_df.empty:
            self.metric_df["cmdb_id"] = self.metric_df["cmdb_id"].astype(str)
            self.metric_df["kpi_name"] = self.metric_df["kpi_name"].astype(str)
            self.metric_df["value"] = pd.to_numeric(self.metric_df["value"], errors="coerce")
        self.log_df = log_df
        self.trace_summary = dict(trace_summary or {})
        self.baseline = baseline
        self.node_graph = dict(node_graph or {"containers": {}, "nodes": {}})
        self.modal_status = dict(modal_status)
        self.threshold = threshold

    def unavailable(self, name: str) -> bool:
        return self.modal_status.get(name, "present") in {"missing", "empty_window", "unloaded", "truncated_missing"}

    def container_kpi(self, service: str, bucket: str) -> EvidenceResult:
        if self.unavailable("metric"):
            return EvidenceResult(False, f"metric {self.modal_status.get('metric')} for case", unavailable=True)
        rows = self._rows(service, bucket, node_level=False)
        return self._max_anomaly(rows, f"{service} container {bucket}")

    def node_kpi(self, service: str, bucket: str) -> EvidenceResult:
        if self.unavailable("metric"):
            return EvidenceResult(False, f"metric {self.modal_status.get('metric')} for case", unavailable=True)
        node = self.node_graph.get("containers", {}).get(str(service), {}).get("node_proxy", str(service))
        if self._is_market():
            return self._max_anomaly(self._rows(node, bucket, node_level=True), f"{service} node {bucket}")
        containers = self.node_graph.get("nodes", {}).get(str(node), {}).get("hosted_containers") or [str(service)]
        frames = [self._rows(container, bucket, node_level=True) for container in containers]
        rows = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        return self._max_anomaly(rows, f"{service} node {bucket}")

    def log_keyword(self, service: str, keywords: Iterable[str]) -> EvidenceResult:
        if self.unavailable("log"):
            return EvidenceResult(False, f"log {self.modal_status.get('log')} for case", unavailable=True)
        if self.log_df is None or self.log_df.empty or "value" not in self.log_df.columns:
            return EvidenceResult(False, "no log dataframe", unavailable=True)
        rows = self.log_df[self.log_df["cmdb_id"].astype(str) == str(service)]
        if rows.empty:
            return EvidenceResult(False, f"no logs for {service}")
        values = rows["value"].dropna().astype(str)
        for text in values:
            if any(str(kw) in text for kw in keywords):
                return EvidenceResult(True, f"log keyword for {service}: {text[:160]}", 1.0, {"checked": int(len(values))})
        return EvidenceResult(False, f"no log keyword for {service}", 0.0, {"checked": int(len(values))})

    def service_trace(self, service: str) -> EvidenceResult:
        if self.unavailable("trace") or self.trace_summary.get("trace_status") != "present":
            return EvidenceResult(False, f"trace {self.modal_status.get('trace')} for case", unavailable=True)
        stats = self.trace_summary.get("service_stats", {}).get(str(service))
        if not stats:
            return EvidenceResult(False, f"no trace stats for {service}")
        p95 = float(stats.get("duration_p95", 0.0) or 0.0)
        return EvidenceResult(p95 > 0, f"{service} trace p95={p95:.2f}", p95, stats)

    def trace_slow_edge(self, service: str) -> EvidenceResult:
        return self._trace_edge(service, "slow_edges", "slow_ratio")

    def trace_edge_count_drop(self, service: str) -> EvidenceResult:
        return self._trace_edge(service, "dropped_edges", "count_drop_ratio", scale=10.0)

    def trace_first_anomalous(self, service: str) -> EvidenceResult:
        if self.unavailable("trace") or self.trace_summary.get("trace_status") != "present":
            return EvidenceResult(False, f"trace {self.modal_status.get('trace')} for case", unavailable=True)
        first = self.trace_summary.get("events", {}).get("first_anomalous_service")
        matched = str(first) == str(service)
        return EvidenceResult(matched, f"first anomalous trace service is {first}", 2.0 if matched else 0.0, {"first_anomalous_service": first})

    def trace_propagation_role(self, service: str, roles: Iterable[str]) -> EvidenceResult:
        if self.unavailable("trace") or self.trace_summary.get("trace_status") != "present":
            return EvidenceResult(False, f"trace {self.modal_status.get('trace')} for case", unavailable=True)
        info = trace_evidence_for_service(self.trace_summary, service)
        role = (info.get("role") or {}).get("role")
        matched = str(role) in {str(r) for r in roles}
        strength = 2.0 if role == "upstream_source" else 1.0
        return EvidenceResult(matched, f"{service} trace propagation role is {role}", strength if matched else 0.0, info)

    def _is_market(self) -> bool:
        return str(self.node_graph.get("dataset", "")).lower() == "market"

    def _rows(self, service: str, bucket: str, node_level: bool) -> pd.DataFrame:
        rows = self.metric_df[self.metric_df["cmdb_id"] == str(service)]
        if rows.empty:
            return rows
        rows = rows[rows.apply(lambda r: row_in_bucket(r, bucket, str(r["kpi_name"])), axis=1)]
        if rows.empty:
            return rows
        if self._is_market():
            return rows
        if node_level:
            return rows[rows["kpi_name"].map(is_node_level_kpi_bank).astype(bool)]
        return rows[~rows["kpi_name"].map(is_node_level_kpi_bank).astype(bool)]

    def _max_anomaly(self, rows: pd.DataFrame, label: str) -> EvidenceResult:
        if rows is None or rows.empty:
            return EvidenceResult(False, f"no rows for {label}", 0.0)
        best = None
        checked = 0
        for row in rows.itertuples(index=False):
            checked += 1
            result = self.baseline.is_anomalous(row.cmdb_id, row.kpi_name, row.value, threshold=self.threshold)
            if not result.is_anomalous:
                continue
            strength = abs(float(result.deviation or 0.0))
            if best is None or strength > best[0]:
                best = (strength, row, result)
        if best is None:
            return EvidenceResult(False, f"no percentile anomaly among {checked} samples for {label}", 0.0, {"checked": checked})
        strength, row, result = best
        return EvidenceResult(True, f"{row.cmdb_id}.{row.kpi_name} {result.reason} value={float(row.value):.6g}", strength, {
            "cmdb_id": row.cmdb_id,
            "kpi_name": row.kpi_name,
            "value": float(row.value),
            "deviation": float(result.deviation),
        })

    def _trace_edge(self, service: str, event_key: str, strength_key: str, scale: float = 1.0) -> EvidenceResult:
        if self.unavailable("trace") or self.trace_summary.get("trace_status") != "present":
            return EvidenceResult(False, f"trace {self.modal_status.get('trace')} for case", unavailable=True)
        events = self.trace_summary.get("events", {}).get(event_key, [])
        hits = [edge for edge in events if str(edge.get("src")) == str(service) or str(edge.get("dst")) == str(service)]
        if not hits:
            return EvidenceResult(False, f"no {event_key} involving {service}", 0.0, {"checked": len(events)})
        best = max(hits, key=lambda edge: float(edge.get(strength_key, 0.0) or 0.0))
        strength = float(best.get(strength_key, 0.0) or 0.0) * scale
        return EvidenceResult(True, f"{event_key} {best.get('src')}->{best.get('dst')} {strength_key}={best.get(strength_key)}", strength, dict(best))
