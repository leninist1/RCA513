"""Portable KPI ontology used only by cross-dataset adapters.

The OpenRCA path keeps its original KPI vocabulary.  This module is an opt-in
bridge for datasets whose KPI names use common cloud/service terms such as
``rx_bytes``, ``NetworkTxBytes`` or JVM heap metrics.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import re
from typing import Any, Mapping

import pandas as pd
import numpy as np

from refute_b_v2_d32.schema import RootCandidate


CANONICAL_REASON_BY_BUCKET = {
    "cpu": "CPU fault",
    "memory": "high memory usage",
    "jvm_oom": "JVM Out of Memory (OOM) Heap",
    "disk_io": "high disk I/O read usage",
    "filesystem": "high disk space usage",
    "network_latency": "network delay",
    "network_packet_loss": "network loss",
}

PORTABLE_BUCKETS = tuple(CANONICAL_REASON_BY_BUCKET)

PORTABLE_FEATURE_NAMES = [
    name
    for bucket in PORTABLE_BUCKETS
    for name in (
        f"{bucket}_density",
        f"{bucket}_max_strength",
        f"{bucket}_total_strength_ratio",
        f"{bucket}_component_count",
        f"{bucket}_top_component_dominance",
    )
] + [
    "network_loss_token_density",
    "network_latency_to_loss_strength_ratio",
    "jvm_heap_memory_ratio",
    "trace_available",
    "trace_has_slow_edge",
    "trace_has_dropped_edge",
    "trace_has_first_anomalous",
    "trace_slow_edge_max_ratio",
    "trace_dropped_edge_max_ratio",
    "log_available",
    "log_has_oom",
    "log_has_network",
    "log_has_disk",
    "log_has_db",
    "log_component_count",
]


@dataclass
class PortableBucketSignal:
    strength: float = 0.0
    count: int = 0
    first_ts: int | None = None
    examples: list[dict[str, Any]] = field(default_factory=list)

    def add(self, *, timestamp: int, strength: float, kpi_name: str, value: float | None = None) -> None:
        self.strength += abs(float(strength))
        self.count += 1
        if self.first_ts is None or int(timestamp) < self.first_ts:
            self.first_ts = int(timestamp)
        if len(self.examples) < 3:
            item: dict[str, Any] = {
                "kpi": str(kpi_name),
                "timestamp": int(timestamp),
                "strength": float(strength),
            }
            if value is not None and math.isfinite(float(value)):
                item["value"] = float(value)
            self.examples.append(item)


def normalize_kpi_text(kpi_name: str) -> str:
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(kpi_name))
    text = text.replace("_", " ").replace("-", " ").replace(".", " ").replace("/", " ")
    text = re.sub(r"[^A-Za-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def infer_kpi_buckets(kpi_name: str, dataset: str = "") -> set[str]:
    """Infer reason buckets from a KPI name without using labels.

    Rules are intentionally lexical and conservative.  They are meant to raise
    recall for portable candidate generation, while final ranking still depends
    on per-case anomaly evidence.
    """
    text = normalize_kpi_text(kpi_name)
    compact = re.sub(r"\s+", "", text)
    tokens = set(text.split())
    buckets: set[str] = set()

    if _has_any(tokens, compact, {"cpu", "processor", "cpuload", "cpupercent"}):
        buckets.add("cpu")

    has_jvm = "jvm" in tokens or "jvm" in compact
    has_heap = "heap" in tokens or "heapmemory" in compact
    if has_jvm and (has_heap or "gc" in tokens or "outofmemory" in compact):
        buckets.add("jvm_oom")

    if _has_any(tokens, compact, {"mem", "memory", "swap", "rss", "cache", "buffer", "buffers", "heap"}):
        buckets.add("memory")

    if _has_any(tokens, compact, {"filesystem", "fscapacity", "fsavailable", "fsinode", "filespace"}):
        buckets.add("filesystem")
    if "space" in tokens and ("disk" in tokens or "fs" in tokens or "filesystem" in tokens):
        buckets.add("filesystem")
    if "diskusage" in compact or "diskused" in compact or "diskfree" in compact:
        buckets.add("filesystem")

    disk_io_terms = {
        "disk", "blkio", "iops", "iowait", "await", "svctm", "io", "read", "reads",
        "write", "writes", "written", "fsync", "fsyncs", "redo", "binlog",
    }
    if _has_any(tokens, compact, disk_io_terms) and "filesystem" not in buckets:
        buckets.add("disk_io")

    packet_loss_terms = {
        "drop", "dropped", "loss", "lost", "retrans", "retransmit", "retransmission",
        "error", "errors", "err", "reject", "rejected", "reset", "abort", "aborted",
        "failed", "failure",
    }
    network_terms = {
        "network", "net", "tcp", "udp", "icmp", "ping", "packet", "packets",
        "rx", "tx", "recv", "receive", "received", "send", "sent", "bytes",
        "traffic", "queue", "latency", "delay", "response", "responsetime",
    }
    has_packet_loss = _has_any(tokens, compact, packet_loss_terms)
    has_network = _has_any(tokens, compact, network_terms) or "rxbytes" in compact or "txbytes" in compact
    if has_packet_loss:
        buckets.add("network_packet_loss")
    elif has_network:
        buckets.add("network_latency")

    # JVM heap pressure is more specific than generic memory for OOM-like
    # candidates, but keeping memory as a sibling helps non-OOM memory faults.
    if "jvm_oom" in buckets:
        buckets.add("memory")

    return buckets


def collect_metric_bucket_signals(
    metric_df: pd.DataFrame,
    baseline: Any,
    *,
    dataset: str = "",
    threshold: str = "p99",
) -> dict[str, dict[str, PortableBucketSignal]]:
    signals: dict[str, dict[str, PortableBucketSignal]] = {}
    if metric_df is None or metric_df.empty:
        return signals
    for row in metric_df.itertuples(index=False):
        component = str(getattr(row, "cmdb_id", ""))
        kpi_name = str(getattr(row, "kpi_name", ""))
        buckets = infer_kpi_buckets(kpi_name, dataset=dataset)
        if not component or not buckets:
            continue
        try:
            value = float(getattr(row, "value"))
        except Exception:
            continue
        if not math.isfinite(value):
            continue
        try:
            result = baseline.is_anomalous(component, kpi_name, value, threshold=threshold)
        except Exception:
            continue
        if not getattr(result, "is_anomalous", False):
            continue
        strength = abs(float(getattr(result, "deviation", 0.0) or 0.0))
        bucket_rows = signals.setdefault(component, {})
        for bucket in buckets:
            bucket_rows.setdefault(bucket, PortableBucketSignal()).add(
                timestamp=int(getattr(row, "timestamp", 0) or 0),
                strength=strength,
                kpi_name=kpi_name,
                value=value,
            )
    return signals


def signal_score(signal: PortableBucketSignal, window_start_ts: int | float | None = None) -> float:
    strength = math.log1p(min(max(float(signal.strength), 0.0), 80.0))
    density = min(2.0, float(signal.count) / 8.0)
    early = 0.0
    if window_start_ts is not None and signal.first_ts is not None:
        minutes = max(0.0, (float(signal.first_ts) - float(window_start_ts)) / 60.0)
        early = 1.0 / (1.0 + minutes)
    return float(strength + density + early)


def bucket_scores_from_signals(
    signals: Mapping[str, Mapping[str, PortableBucketSignal]],
    *,
    window_start_ts: int | float | None = None,
    top_k: int = 3,
) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for component_signals in signals.values():
        for bucket, signal in component_signals.items():
            grouped.setdefault(str(bucket), []).append(signal_score(signal, window_start_ts))
    out: dict[str, float] = {}
    for bucket, scores in grouped.items():
        top = sorted(scores, reverse=True)[: max(1, int(top_k))]
        if top:
            out[bucket] = float(sum(top) / len(top))
    return out


def portable_root_candidates(
    *,
    metric_df: pd.DataFrame,
    baseline: Any,
    dataset: str,
    window_start_ts: int,
    max_per_bucket: int = 8,
    prior_scale: float = 0.16,
    prior_offset: float = 0.04,
    max_prior: float = 1.8,
) -> list[RootCandidate]:
    signals = collect_metric_bucket_signals(metric_df, baseline, dataset=dataset)
    scored: list[tuple[str, float, str, PortableBucketSignal]] = []
    for component, component_signals in signals.items():
        for bucket, signal in component_signals.items():
            if bucket not in CANONICAL_REASON_BY_BUCKET:
                continue
            scored.append((bucket, signal_score(signal, window_start_ts), component, signal))

    out: list[RootCandidate] = []
    for bucket in sorted({bucket for bucket, _, _, _ in scored}):
        rows = sorted(
            [row for row in scored if row[0] == bucket],
            key=lambda item: (-item[1], item[2]),
        )[: max(1, int(max_per_bucket))]
        for bucket, score, component, signal in rows:
            prior = min(float(max_prior), float(prior_offset) + float(prior_scale) * float(score))
            out.append(RootCandidate(
                component=component,
                reason=CANONICAL_REASON_BY_BUCKET[bucket],
                prior=prior,
                source="portable_ontology",
                details={
                    "primary_bucket": bucket,
                    "signal_strength": float(signal.strength),
                    "signal_count": int(signal.count),
                    "first_ts": signal.first_ts,
                    "portable_score": float(score),
                    "examples": list(signal.examples),
                },
            ))
    return out


def extract_portable_reason_features(
    *,
    metric_df: pd.DataFrame,
    log_df: pd.DataFrame,
    trace_summary: Mapping[str, Any] | None,
    baseline: Any,
    dataset: str = "",
) -> np.ndarray:
    """Extract portable reason-classifier features.

    The feature vector uses only telemetry and baseline anomaly results from the
    current incident window.  It does not read labels or case identifiers.
    """

    signals = collect_metric_bucket_signals(metric_df, baseline, dataset=dataset)
    feats: dict[str, float] = {name: 0.0 for name in PORTABLE_FEATURE_NAMES}

    bucket_counts = {bucket: 0.0 for bucket in PORTABLE_BUCKETS}
    bucket_strengths = {bucket: 0.0 for bucket in PORTABLE_BUCKETS}
    bucket_max = {bucket: 0.0 for bucket in PORTABLE_BUCKETS}
    bucket_components: dict[str, set[str]] = {bucket: set() for bucket in PORTABLE_BUCKETS}
    bucket_top_strength = {bucket: 0.0 for bucket in PORTABLE_BUCKETS}

    for component, component_signals in signals.items():
        for bucket, signal in component_signals.items():
            if bucket not in bucket_counts:
                continue
            bucket_counts[bucket] += float(signal.count)
            bucket_strengths[bucket] += float(signal.strength)
            bucket_max[bucket] = max(bucket_max[bucket], float(signal.strength))
            bucket_top_strength[bucket] = max(bucket_top_strength[bucket], float(signal.strength))
            bucket_components[bucket].add(str(component))

    total_count = max(1.0, sum(bucket_counts.values()))
    total_strength = max(1.0, sum(bucket_strengths.values()))
    for bucket in PORTABLE_BUCKETS:
        feats[f"{bucket}_density"] = bucket_counts[bucket] / total_count
        feats[f"{bucket}_max_strength"] = math.log1p(bucket_max[bucket])
        feats[f"{bucket}_total_strength_ratio"] = bucket_strengths[bucket] / total_strength
        feats[f"{bucket}_component_count"] = math.log1p(len(bucket_components[bucket]))
        feats[f"{bucket}_top_component_dominance"] = bucket_top_strength[bucket] / max(1.0, bucket_strengths[bucket])

    feats.update(_portable_metric_subtype_features(metric_df, baseline, dataset))
    feats.update(_portable_trace_features(trace_summary))
    feats.update(_portable_log_features(log_df))
    return np.array([feats.get(name, 0.0) for name in PORTABLE_FEATURE_NAMES], dtype=np.float64)


def _portable_metric_subtype_features(metric_df: pd.DataFrame, baseline: Any, dataset: str) -> dict[str, float]:
    out = {
        "network_loss_token_density": 0.0,
        "network_latency_to_loss_strength_ratio": 0.0,
        "jvm_heap_memory_ratio": 0.0,
    }
    if metric_df is None or metric_df.empty:
        return out
    loss_count = 0.0
    network_count = 0.0
    latency_strength = 0.0
    loss_strength = 0.0
    heap_mem_count = 0.0
    mem_count = 0.0
    loss_tokens = {"drop", "dropped", "loss", "lost", "retrans", "error", "errors", "err", "reject", "reset", "abort", "failed"}
    for row in metric_df.itertuples(index=False):
        kpi = str(getattr(row, "kpi_name", ""))
        buckets = infer_kpi_buckets(kpi, dataset=dataset)
        if not buckets:
            continue
        try:
            value = float(getattr(row, "value"))
            result = baseline.is_anomalous(str(getattr(row, "cmdb_id", "")), kpi, value, threshold="p99")
        except Exception:
            continue
        if not getattr(result, "is_anomalous", False):
            continue
        strength = abs(float(getattr(result, "deviation", 0.0) or 0.0))
        text = normalize_kpi_text(kpi)
        tokens = set(text.split())
        if "network_latency" in buckets or "network_packet_loss" in buckets:
            network_count += 1.0
            if tokens & loss_tokens:
                loss_count += 1.0
        if "network_latency" in buckets:
            latency_strength += strength
        if "network_packet_loss" in buckets:
            loss_strength += strength
        if "memory" in buckets or "jvm_oom" in buckets:
            mem_count += 1.0
            if "heap" in tokens or "heapmemory" in re.sub(r"\s+", "", text) or "jvm_oom" in buckets:
                heap_mem_count += 1.0
    out["network_loss_token_density"] = loss_count / max(1.0, network_count)
    out["network_latency_to_loss_strength_ratio"] = math.log1p(latency_strength) / max(1.0, math.log1p(loss_strength))
    out["jvm_heap_memory_ratio"] = heap_mem_count / max(1.0, mem_count)
    return out


def _portable_trace_features(trace_summary: Mapping[str, Any] | None) -> dict[str, float]:
    out = {
        "trace_available": 0.0,
        "trace_has_slow_edge": 0.0,
        "trace_has_dropped_edge": 0.0,
        "trace_has_first_anomalous": 0.0,
        "trace_slow_edge_max_ratio": 0.0,
        "trace_dropped_edge_max_ratio": 0.0,
    }
    if not trace_summary or trace_summary.get("trace_status") != "present":
        return out
    out["trace_available"] = 1.0
    events = trace_summary.get("events", {}) or {}
    slow_edges = events.get("slow_edges", []) or []
    dropped_edges = events.get("dropped_edges", []) or []
    out["trace_has_slow_edge"] = 1.0 if slow_edges else 0.0
    out["trace_has_dropped_edge"] = 1.0 if dropped_edges else 0.0
    out["trace_has_first_anomalous"] = 1.0 if events.get("first_anomalous_service") else 0.0
    out["trace_slow_edge_max_ratio"] = max((float(edge.get("slow_ratio", 0.0) or 0.0) for edge in slow_edges), default=0.0)
    out["trace_dropped_edge_max_ratio"] = max((float(edge.get("count_drop_ratio", 0.0) or 0.0) for edge in dropped_edges), default=0.0)
    return out


def _portable_log_features(log_df: pd.DataFrame) -> dict[str, float]:
    out = {
        "log_available": 0.0,
        "log_has_oom": 0.0,
        "log_has_network": 0.0,
        "log_has_disk": 0.0,
        "log_has_db": 0.0,
        "log_component_count": 0.0,
    }
    if log_df is None or log_df.empty or "value" not in log_df.columns:
        return out
    out["log_available"] = 1.0
    values = log_df["value"].dropna().astype(str)
    text = " ".join(values).lower()
    out["log_has_oom"] = float(any(token in text for token in ("outofmemory", "oom", "heap space", "java heap")))
    out["log_has_network"] = float(any(token in text for token in ("timeout", "timed out", "connection reset", "broken pipe", "retrans", "packet loss")))
    out["log_has_disk"] = float(any(token in text for token in ("no space", "disk", "i/o", "io error")))
    out["log_has_db"] = float(any(token in text for token in ("connection", "session", "too many", "mysql", "redis")))
    if "cmdb_id" in log_df.columns:
        out["log_component_count"] = math.log1p(log_df["cmdb_id"].nunique())
    return out


def _has_any(tokens: set[str], compact: str, values: set[str]) -> bool:
    if tokens & values:
        return True
    return any(value in compact for value in values if len(value) >= 3)
