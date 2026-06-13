"""High-recall joint (component, reason) candidate generation for D32."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
import math
from typing import Any, Mapping

import pandas as pd

from refute_b_v2_d32.schema import RootCandidate, reason_bucket
from refute_b_v2_d32.signature import reason_for_log


CPU_TOKENS = ("cpu", "processor_load")
MEMORY_TOKENS = ("mem", "memory", "swap", "cache", "buffers", "pga")
DISK_TOKENS = ("disk", "read", "write", "io_", "iowait", "await", "svctm", "tbs", "redo")
NETWORK_TOKENS = ("network", "packet", "packets", "icmp", "ping", "recv", "send", "sent", "queue", "traffic", "tnsping")
PACKET_LOSS_TOKENS = ("error", "err", "drop", "loss", "retrans", "reject", "rejected", "reset", "abort", "aborted")
DB_CONNECTION_TOKENS = (
    "sess",
    "session",
    "connect",
    "login",
    "tnsping",
    "hang",
    "active",
    "call_per_sec",
    "exec_per_sec",
    "tps_per_sec",
    "row_lock",
)
REASON_ORDER = ("CPU fault", "network delay", "network loss", "db connection limit", "db close")


@dataclass
class BucketSignal:
    strength: float = 0.0
    count: int = 0
    first_ts: int | None = None
    examples: list[dict[str, Any]] = field(default_factory=list)

    def add(self, *, timestamp: int, strength: float, source: str, name: str) -> None:
        self.strength += abs(float(strength))
        self.count += 1
        if self.first_ts is None or int(timestamp) < self.first_ts:
            self.first_ts = int(timestamp)
        if len(self.examples) < 3:
            self.examples.append({
                "source": source,
                "name": name,
                "timestamp": int(timestamp),
                "strength": float(strength),
            })


@dataclass(frozen=True)
class JointPrior:
    pair_counts: Counter
    reason_counts: Counter
    family_reason_counts: Counter


@dataclass(frozen=True)
class JointCandidate:
    component: str
    reason: str
    score: float
    sources: tuple[str, ...]
    details: Mapping[str, Any]

    def key(self) -> tuple[str, str]:
        return self.component, self.reason


def component_family(component: str) -> str:
    text = str(component).lower()
    if text.startswith("docker_"):
        return "docker"
    if text.startswith("os_"):
        return "os"
    if text.startswith("db_"):
        return "db"
    if text.startswith("redis_"):
        return "redis"
    return "other"


def build_joint_prior(cases: list[Mapping[str, Any]]) -> JointPrior:
    pair_counts: Counter = Counter()
    reason_counts: Counter = Counter()
    family_reason_counts: Counter = Counter()
    for row in cases:
        component = str(row.get("component", "")).strip()
        reason = str(row.get("reason", "")).strip()
        if not component or not reason:
            continue
        pair_counts[(component, reason)] += 1
        reason_counts[reason] += 1
        family_reason_counts[(reason, component_family(component))] += 1
    return JointPrior(pair_counts, reason_counts, family_reason_counts)


def primary_bucket_for_reason(reason: str) -> str:
    bucket = reason_bucket(reason)
    return "db_connection" if bucket == "db_connection" else bucket


def _kpi_buckets(component: str, kpi_name: str) -> set[str]:
    low = str(kpi_name).lower()
    buckets: set[str] = set()
    if any(token in low for token in CPU_TOKENS):
        buckets.add("cpu")
    if any(token in low for token in MEMORY_TOKENS):
        buckets.add("memory")
    if any(token in low for token in DISK_TOKENS):
        buckets.add("disk_io")
    if any(token in low for token in NETWORK_TOKENS):
        if any(token in low for token in PACKET_LOSS_TOKENS):
            buckets.add("network_packet_loss")
        else:
            buckets.add("network_latency")
    if component_family(component) == "db":
        if any(token in low for token in DB_CONNECTION_TOKENS):
            buckets.add("db_connection")
    return buckets


def _reasons_from_bucket(bucket: str) -> tuple[str, ...]:
    if bucket == "cpu":
        return ("CPU fault",)
    if bucket == "network_latency":
        return ("network delay",)
    if bucket == "network_packet_loss":
        return ("network loss",)
    if bucket == "db_connection":
        return ("db connection limit", "db close")
    return ()


def _add_signal(
    signals: dict[str, dict[str, BucketSignal]],
    component: str,
    bucket: str,
    *,
    timestamp: int,
    strength: float,
    source: str,
    name: str,
) -> None:
    signals[str(component)][bucket].add(timestamp=timestamp, strength=strength, source=source, name=name)


def _metric_signals(metric_df: pd.DataFrame, baseline) -> dict[str, dict[str, BucketSignal]]:
    signals: dict[str, dict[str, BucketSignal]] = defaultdict(lambda: defaultdict(BucketSignal))
    if metric_df is None or metric_df.empty:
        return signals
    for row in metric_df.itertuples(index=False):
        component = str(getattr(row, "cmdb_id", ""))
        kpi_name = str(getattr(row, "kpi_name", ""))
        buckets = _kpi_buckets(component, kpi_name)
        if not buckets:
            continue
        result = baseline.is_anomalous(component, kpi_name, getattr(row, "value"), threshold="p99")
        if not result.is_anomalous:
            continue
        strength = abs(float(result.deviation or 0.0))
        for bucket in buckets:
            _add_signal(
                signals,
                component,
                bucket,
                timestamp=int(getattr(row, "timestamp")),
                strength=strength,
                source="metric",
                name=kpi_name,
            )
    return signals


def _add_family_presence_signals(signals: dict[str, dict[str, BucketSignal]], metric_df: pd.DataFrame) -> None:
    if metric_df is None or metric_df.empty or "cmdb_id" not in metric_df.columns:
        return
    for component in sorted({str(item) for item in metric_df["cmdb_id"].dropna().astype(str)}):
        family = component_family(component)
        if family == "docker":
            buckets = ("cpu",)
        elif family == "os":
            buckets = ("network_latency", "network_packet_loss")
        elif family == "db":
            buckets = ("db_connection",)
        else:
            continue
        for bucket in buckets:
            if bucket in signals[component]:
                continue
            signals[component][bucket] = BucketSignal(
                strength=0.01,
                count=0,
                first_ts=None,
                examples=[{"source": "metric_presence", "name": family, "timestamp": None, "strength": 0.01}],
            )


def _add_log_signals(signals: dict[str, dict[str, BucketSignal]], log_df: pd.DataFrame) -> None:
    if log_df is None or log_df.empty or "value" not in log_df.columns:
        return
    for row in log_df.itertuples(index=False):
        reason = reason_for_log(getattr(row, "value", ""))
        if not reason:
            continue
        bucket = primary_bucket_for_reason(reason)
        _add_signal(
            signals,
            str(getattr(row, "cmdb_id", "")),
            bucket,
            timestamp=int(getattr(row, "timestamp", 0) or 0),
            strength=8.0,
            source="log",
            name=reason,
        )


def _add_trace_signals(signals: dict[str, dict[str, BucketSignal]], trace_summary: Mapping[str, Any] | None, window_start_ts: int) -> None:
    if not trace_summary or trace_summary.get("trace_status") != "present":
        return
    events = trace_summary.get("events", {}) or {}
    first = events.get("first_anomalous_service")
    if first:
        _add_signal(
            signals,
            str(first),
            "network_latency",
            timestamp=window_start_ts,
            strength=4.0,
            source="trace_first",
            name="first_anomalous_service",
        )
    for edge in events.get("slow_edges", []) or []:
        strength = float(edge.get("slow_ratio", 0.0) or 0.0) * 10.0
        for endpoint in (edge.get("src"), edge.get("dst")):
            if endpoint:
                _add_signal(
                    signals,
                    str(endpoint),
                    "network_latency",
                    timestamp=window_start_ts,
                    strength=strength,
                    source="trace_slow_edge",
                    name=f"{edge.get('src')}->{edge.get('dst')}",
                )
    for edge in events.get("dropped_edges", []) or []:
        strength = float(edge.get("count_drop_ratio", 0.0) or 0.0) * 10.0
        for endpoint in (edge.get("src"), edge.get("dst")):
            if endpoint:
                _add_signal(
                    signals,
                    str(endpoint),
                    "network_packet_loss",
                    timestamp=window_start_ts,
                    strength=strength,
                    source="trace_dropped_edge",
                    name=f"{edge.get('src')}->{edge.get('dst')}",
                )


def _type_reason_expansions(component: str, component_signals: Mapping[str, BucketSignal]) -> dict[str, str]:
    family = component_family(component)
    expansions: dict[str, str] = {}
    if family == "docker" and component_signals:
        expansions["CPU fault"] = "type_docker_any_anomaly"
    if family == "os" and component_signals and ("network_latency" in component_signals or "network_packet_loss" in component_signals):
        expansions["network delay"] = "type_os_network_anomaly"
        expansions["network loss"] = "type_os_network_anomaly"
    if family == "db" and "db_connection" in component_signals:
        expansions["db connection limit"] = "type_db_any_anomaly"
        expansions["db close"] = "type_db_any_anomaly"
    return expansions


def _type_compatibility(component: str, reason: str) -> float:
    family = component_family(component)
    bucket = primary_bucket_for_reason(reason)
    if reason == "CPU fault" and family == "docker":
        return 3.0
    if bucket in {"network_latency", "network_packet_loss"} and family == "os":
        return 3.0
    if bucket == "db_connection" and family == "db":
        return 3.0
    return 0.4


def _score_local(component: str, reason: str, component_signals: Mapping[str, BucketSignal], source: str, window_start_ts: int) -> tuple[float, dict[str, Any]]:
    bucket = primary_bucket_for_reason(reason)
    signal = component_signals.get(bucket)
    strength = signal.strength if signal else 0.0
    count = signal.count if signal else 0
    first_ts = signal.first_ts if signal else None
    if bucket == "db_connection" and signal is None:
        strength = sum(item.strength for item in component_signals.values())
        count = sum(item.count for item in component_signals.values())
        firsts = [item.first_ts for item in component_signals.values() if item.first_ts is not None]
        first_ts = min(firsts) if firsts else None
    onset = 0.0
    if first_ts is not None:
        minutes = max(0.0, (float(first_ts) - float(window_start_ts)) / 60.0)
        onset = 1.0 / (1.0 + minutes)
    score = _type_compatibility(component, reason) + math.log1p(max(0.0, strength)) + onset + min(2.0, count / 5.0)
    details: dict[str, Any] = {
        "primary_bucket": bucket,
        "signal_strength": strength,
        "signal_count": count,
        "first_ts": first_ts,
        "source": source,
    }
    if signal:
        details["examples"] = signal.examples
    return score, details


def _generate_raw_candidates(
    metric_df: pd.DataFrame,
    log_df: pd.DataFrame,
    trace_summary: Mapping[str, Any] | None,
    baseline,
    window_start_ts: int,
) -> list[JointCandidate]:
    signals = _metric_signals(metric_df, baseline)
    _add_family_presence_signals(signals, metric_df)
    _add_log_signals(signals, log_df)
    _add_trace_signals(signals, trace_summary, window_start_ts)

    out: dict[tuple[str, str], JointCandidate] = {}
    for component, component_signals in signals.items():
        reason_sources: dict[str, set[str]] = defaultdict(set)
        for bucket in component_signals:
            for reason in _reasons_from_bucket(bucket):
                reason_sources[reason].add(f"bucket:{bucket}")
        for reason, source in _type_reason_expansions(component, component_signals).items():
            reason_sources[reason].add(source)
        for reason, sources in reason_sources.items():
            source_label = ",".join(sorted(sources))
            score, details = _score_local(component, reason, component_signals, source_label, window_start_ts)
            candidate = JointCandidate(component, reason, score, tuple(sorted(sources)), details)
            old = out.get(candidate.key())
            if old is None or candidate.score > old.score:
                out[candidate.key()] = candidate
    return sorted(out.values(), key=lambda row: (-row.score, row.component, row.reason))


def extract_joint_reason_features(
    *,
    metric_df: pd.DataFrame,
    log_df: pd.DataFrame,
    trace_summary: Mapping[str, Any] | None,
    baseline,
    window_start_ts: int,
) -> list[str]:
    signals = _metric_signals(metric_df, baseline)
    _add_log_signals(signals, log_df)
    _add_trace_signals(signals, trace_summary, window_start_ts)
    features: set[str] = set()
    for component, component_signals in signals.items():
        family = component_family(component)
        for bucket, signal in component_signals.items():
            if signal.count <= 0:
                continue
            features.add(f"joint:family:{family}:bucket:{bucket}")
            if signal.count >= 5:
                features.add(f"joint:family:{family}:bucket:{bucket}:multi")
            if signal.strength >= 20.0:
                features.add(f"joint:family:{family}:bucket:{bucket}:strong")
            elif signal.strength >= 5.0:
                features.add(f"joint:family:{family}:bucket:{bucket}:medium")
    return sorted(features)


def _rerank_score(candidate: JointCandidate, prior: JointPrior) -> float:
    details = dict(candidate.details)
    signal_count = float(details.get("signal_count", 0) or 0)
    signal_strength = float(details.get("signal_strength", 0.0) or 0.0)
    family = component_family(candidate.component)
    bucket = primary_bucket_for_reason(candidate.reason)
    pair_count = prior.pair_counts.get((candidate.component, candidate.reason), 0)
    family_count = prior.family_reason_counts.get((candidate.reason, family), 0)
    local = math.log1p(min(max(signal_strength, 0.0), 100.0)) + min(1.5, signal_count / 10.0)
    prior_score = 2.5 * math.log1p(pair_count) + 0.6 * math.log1p(family_count)
    if pair_count == 0 and signal_count == 0:
        prior_score -= 0.3
    if candidate.reason == "CPU fault" and family == "docker":
        prior_score += 0.5
    elif bucket in {"network_latency", "network_packet_loss"} and family == "os":
        prior_score += 0.5
    elif bucket == "db_connection" and family == "db":
        prior_score += 0.5
    return local + prior_score


def _reason_balanced_beam(candidates: list[JointCandidate], prior: JointPrior, per_reason: int) -> list[tuple[JointCandidate, float]]:
    groups: dict[str, list[tuple[float, JointCandidate]]] = {reason: [] for reason in REASON_ORDER}
    for candidate in candidates:
        groups.setdefault(candidate.reason, []).append((_rerank_score(candidate, prior), candidate))
    sorted_groups = {
        reason: sorted(rows, key=lambda item: (-item[0], item[1].component, item[1].reason))[:per_reason]
        for reason, rows in groups.items()
    }
    out: list[tuple[JointCandidate, float]] = []
    for idx in range(per_reason):
        for reason in REASON_ORDER:
            rows = sorted_groups.get(reason, [])
            if idx < len(rows):
                score, candidate = rows[idx]
                out.append((candidate, score))
    return out


def generate_joint_root_candidates(
    *,
    metric_df: pd.DataFrame,
    log_df: pd.DataFrame,
    trace_summary: Mapping[str, Any] | None,
    baseline,
    joint_prior: JointPrior,
    window_start_ts: int,
    beam_per_reason: int = 8,
    prior_scale: float = 0.35,
    prior_offset: float = 0.2,
    max_prior: float = 3.2,
) -> list[RootCandidate]:
    raw = _generate_raw_candidates(metric_df, log_df, trace_summary, baseline, window_start_ts)
    beam = _reason_balanced_beam(raw, joint_prior, beam_per_reason)
    out: list[RootCandidate] = []
    for beam_index, (candidate, rerank_score) in enumerate(beam):
        bounded_prior = min(float(max_prior), float(prior_offset) + float(prior_scale) * float(rerank_score))
        details = {
            **dict(candidate.details),
            "joint_local_score": candidate.score,
            "joint_rerank_score": rerank_score,
            "joint_beam_index": beam_index,
            "joint_sources": list(candidate.sources),
            "weak_family_presence": int(candidate.details.get("signal_count", 0) or 0) == 0,
        }
        out.append(RootCandidate(candidate.component, candidate.reason, bounded_prior, "joint_generator", details))
    return out
