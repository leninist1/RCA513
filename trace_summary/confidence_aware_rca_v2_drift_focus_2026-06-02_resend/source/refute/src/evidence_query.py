"""
evidence_query.py -- Phase 2 / B-L2.1 MVP

Structured evidence queries used by the refutation engine.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

from refute.src.baseline_distributions import BaselineStore
from refute.src.node_container_split import is_node_level_kpi_bank
from refute.src.trace_evidence import service_trace_anomaly, slow_edges_for_service


KPI_BUCKET_PATTERNS: Dict[str, Tuple[str, ...]] = {
    "cpu": ("_CpuPercent", "OSLinux-CPU", "CPUCpuUtil", "SingleCpuUtil", "JVM_CPULoad"),
    "memory": (
        "_MemPercent", "_MemLimit", "_MemUsage",
        "MemPercent", "MemLimit", "MemUsage",
        "JVM-Memory", "HeapMemory", "Tomcat-MEMORY",
        "used_memory", "used_memory_peak", "used_memory_rss", "mem_fragmentation_ratio",
        "Qcache Free Memory", "max trx lock memory bytes",
        "PROCPPMem", "PROCPPMemPerc",
    ),
    "network": (
        "NetworkRx", "NetworkTx", "NetworkPacket", "NetworkRxBytes", "NetworkTxBytes",
        "_TCP-", "TCP-", "TotalTcpConnNum", "NETPackets", "NETKBTotalPerSec",
        "NETBandwidthUtil", "NETInErr", "NETOutErr", "rejected_connections",
        "Aborted Clients", "Aborted Connects",
    ),
    "disk": ("_DSKRead", "_DSKWrite", "_DSKBps", "_DSKTps", "_DSKPercentBusy", "blkio"),
    "filesystem": ("FILESYSTEM", "FSAvailable", "FSCapacity", "FSInode"),
    "mysql_internal": ("Mysql-MySQL_3306_",),
    "redis_internal": ("redis-Redis", "Redis_",),
    "jvm_threads": ("JVM-Threads", "ThreadCount", "Tomcat-Threads", "Tomcat-Sessions"),
    "tomcat_requests": ("Tomcat-Requests",),
}

REASON_TO_BUCKET = {
    "high memory usage": "memory",
    "memory": "memory",
    "jvm out of memory (oom) heap": "memory",
    "cpu": "cpu",
    "high cpu usage": "cpu",
    "high jvm cpu load": "cpu",
    "network latency": "network",
    "network packet loss": "network",
    "network": "network",
    "disk": "disk",
    "high disk i/o read usage": "disk",
    "high disk io read usage": "disk",
    "high disk space usage": "filesystem",
    "file system": "filesystem",
    "filesystem": "filesystem",
}

LOG_KEYWORDS = (
    "OutOfMemoryError", "OOM", "oom", "Full GC", "killed", "SIGKILL",
    "Connection refused", "connection refused", "500 Internal Server Error",
    "Timeout", "timeout", "timed out", "Exception", "exception", "ERROR", "error",
)


@dataclass(frozen=True)
class EvidenceResult:
    matched: bool
    evidence: str
    strength: float = 0.0
    details: Optional[dict] = None
    unavailable: bool = False


def reason_to_bucket(reason: str) -> str:
    key = str(reason).strip().lower()
    return REASON_TO_BUCKET.get(key, key.replace(" ", "_"))


def kpi_in_bucket(kpi_name: str, bucket: str) -> bool:
    patterns = KPI_BUCKET_PATTERNS.get(bucket, (bucket,))
    return any(pattern in str(kpi_name) for pattern in patterns)


class EvidenceQuery:
    def __init__(
        self,
        metric_df: pd.DataFrame,
        baseline_store: BaselineStore,
        node_graph: Optional[dict] = None,
        log_df: Optional[pd.DataFrame] = None,
        trace_df: Optional[pd.DataFrame] = None,
        modal_status: Optional[dict] = None,
        threshold: str = "p99",
    ):
        self.metric_df = metric_df.copy() if metric_df is not None else pd.DataFrame(columns=["timestamp", "cmdb_id", "kpi_name", "value"])
        self.metric_df["cmdb_id"] = self.metric_df["cmdb_id"].astype(str)
        self.metric_df["kpi_name"] = self.metric_df["kpi_name"].astype(str)
        self.metric_df["value"] = pd.to_numeric(self.metric_df["value"], errors="coerce")
        self.baseline_store = baseline_store
        self.node_graph = node_graph or {"containers": {}, "nodes": {}}
        self.log_df = log_df
        self.trace_df = trace_df
        self.modal_status = modal_status or {}
        self.threshold = threshold

    def _modality_unavailable(self, name: str) -> bool:
        status = self.modal_status.get(name, "present")
        return status in {"missing", "empty_window", "truncated_missing", "unloaded"}

    def _kpi_rows(self, svc: str, bucket: str, node_level: bool) -> pd.DataFrame:
        rows = self.metric_df[self.metric_df["cmdb_id"] == str(svc)]
        if rows.empty:
            return rows
        rows = rows[rows["kpi_name"].map(lambda k: kpi_in_bucket(k, bucket)).astype(bool)]
        if rows.empty:
            return rows
        if node_level:
            rows = rows[rows["kpi_name"].map(is_node_level_kpi_bank).astype(bool)]
        else:
            rows = rows[~rows["kpi_name"].map(is_node_level_kpi_bank).astype(bool)]
        return rows

    def _max_anomaly(self, rows: pd.DataFrame) -> EvidenceResult:
        best = None
        checked = 0
        for row in rows.itertuples(index=False):
            checked += 1
            result = self.baseline_store.is_anomalous(
                row.cmdb_id, row.kpi_name, row.value, threshold=self.threshold
            )
            if not result.is_anomalous:
                continue
            strength = abs(result.deviation)
            if best is None or strength > best[0]:
                best = (strength, row, result)
        if best is None:
            return EvidenceResult(False, f"no percentile anomaly among {checked} samples", 0.0, {"checked": checked})
        strength, row, result = best
        return EvidenceResult(
            True,
            f"{row.cmdb_id}.{row.kpi_name} {result.reason} value={row.value:.6g} threshold={result.threshold:.6g}",
            float(strength),
            {"cmdb_id": row.cmdb_id, "kpi_name": row.kpi_name, "value": float(row.value), "deviation": result.deviation},
        )

    def is_container_kpi_anomalous(self, svc: str, kpi_bucket: str) -> EvidenceResult:
        if self._modality_unavailable("metric"):
            return EvidenceResult(False, f"metric modality {self.modal_status.get('metric')} for case", 0.0, unavailable=True)
        return self._max_anomaly(self._kpi_rows(svc, kpi_bucket, node_level=False))

    def is_node_kpi_anomalous(self, svc_or_node_id: str, kpi_bucket: str) -> EvidenceResult:
        if self._modality_unavailable("metric"):
            return EvidenceResult(False, f"metric modality {self.modal_status.get('metric')} for case", 0.0, unavailable=True)
        containers = self.node_graph.get("nodes", {}).get(str(svc_or_node_id), {}).get("hosted_containers")
        if not containers:
            containers = [str(svc_or_node_id)]
        rows = pd.concat([
            self._kpi_rows(container, kpi_bucket, node_level=True)
            for container in containers
        ], ignore_index=True) if containers else pd.DataFrame()
        return self._max_anomaly(rows)

    def is_anomaly_explained_by_node(self, svc: str, kpi_bucket: str) -> EvidenceResult:
        node = self.node_graph.get("containers", {}).get(str(svc), {}).get("node_proxy", str(svc))
        node_result = self.is_node_kpi_anomalous(node, kpi_bucket)
        if node_result.matched:
            return EvidenceResult(True, f"node-level anomaly on {node}: {node_result.evidence}", node_result.strength, node_result.details)
        return EvidenceResult(False, f"no node-level {kpi_bucket} anomaly for {svc}", 0.0, node_result.details)

    def does_match_log_keyword(self, svc: str, keywords: Iterable[str] = LOG_KEYWORDS) -> EvidenceResult:
        if self._modality_unavailable("log"):
            return EvidenceResult(False, f"log modality {self.modal_status.get('log')} for case", 0.0, unavailable=True)
        if self.log_df is None or self.log_df.empty:
            return EvidenceResult(False, "no log dataframe", 0.0, unavailable=True)
        logs = self.log_df[self.log_df["cmdb_id"].astype(str) == str(svc)]
        if logs.empty or "value" not in logs.columns:
            return EvidenceResult(False, f"no logs for {svc}", 0.0)
        pattern = re.compile("|".join(re.escape(k) for k in keywords))
        values = logs["value"].dropna().astype(str)
        hit = values[values.map(lambda text: bool(pattern.search(text)))]
        if hit.empty:
            return EvidenceResult(False, f"no log keyword for {svc}", 0.0, {"checked": int(len(values))})
        return EvidenceResult(True, f"log keyword matched for {svc}: {hit.iloc[0][:160]}", 1.0, {"checked": int(len(values))})

    def is_service_trace_anomalous(self, svc: str) -> EvidenceResult:
        if self._modality_unavailable("trace"):
            return EvidenceResult(False, f"trace modality {self.modal_status.get('trace')} for case", 0.0, unavailable=True)
        if self.trace_df is None or self.trace_df.empty:
            return EvidenceResult(False, "no trace dataframe", 0.0, unavailable=True)
        result = service_trace_anomaly(self.trace_df, svc)
        return EvidenceResult(result.matched, result.evidence, result.strength, result.details)

    def has_slow_trace_edge(self, svc: str) -> EvidenceResult:
        if self._modality_unavailable("trace"):
            return EvidenceResult(False, f"trace modality {self.modal_status.get('trace')} for case", 0.0, unavailable=True)
        if self.trace_df is None or self.trace_df.empty:
            return EvidenceResult(False, "no trace dataframe", 0.0, unavailable=True)
        result = slow_edges_for_service(self.trace_df, svc)
        return EvidenceResult(result.matched, result.evidence, result.strength, result.details)
