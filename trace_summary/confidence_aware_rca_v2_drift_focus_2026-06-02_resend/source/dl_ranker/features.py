"""Per-candidate feature extraction for the set-transformer ranker.

Extracts 33-dim feature vectors for each (component, reason) candidate,
covering metric, temporal/onset, trace, log, structural, and case-level signals.

Single-pass over metric_df to compute per-(component, bucket) anomaly aggregates,
then per-candidate features are fast lookups.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd

from refute_b_v2.trace_propagation import propagation_roles
from refute_b_v2_d32.signature import service_type, service_role

_OOM_KEYWORDS = ("oom", "outofmemory", "out of memory", "heap space", "gc overhead")
_GC_KEYWORDS = ("gc", "garbage", "g1", "cms", "parnew", "safepoint")
_CRASH_KEYWORDS = ("crash", "error", "fatal", "exception", "aborted", "killed")
_DB_KEYWORDS = ("db ", "connection", "query", "timeout", "jdbc", "datasource")

_BUCKETS = ("cpu", "memory", "jvm_oom", "disk_io", "filesystem", "network_latency", "network_packet_loss")


@dataclass
class _CompAgg:
    """Per-component aggregation computed once per case."""
    first_ts: float | None = None
    max_dev: float = 0.0
    anom_count: int = 0
    peak_ts: float | None = None
    bucket_counts: dict[str, int] = None
    bucket_first_ts: dict[str, float] = None
    bucket_max_dev: dict[str, float] = None

    def __post_init__(self):
        if self.bucket_counts is None:
            self.bucket_counts = {}
        if self.bucket_first_ts is None:
            self.bucket_first_ts = {}
        if self.bucket_max_dev is None:
            self.bucket_max_dev = {}


def _build_component_aggregates(
    metric_df: pd.DataFrame,
    baseline: Any,
) -> dict[str, _CompAgg]:
    """Single pass: compute per-component aggregates for all buckets."""
    from refute_b_v2_d32.bucket_resolver import row_buckets
    aggs: dict[str, _CompAgg] = {}

    if metric_df.empty or "kpi_name" not in metric_df.columns:
        return aggs

    for (cmdb_id, kpi), grp in metric_df.groupby(["cmdb_id", "kpi_name"]):
        comp = str(cmdb_id)
        kpi_str = str(kpi)
        agg = aggs.get(comp)
        if agg is None:
            agg = _CompAgg()
            aggs[comp] = agg

        buckets = row_buckets(None, kpi_str, comp) or set()
        grp_sorted = grp.sort_values("timestamp")

        for row in grp_sorted.itertuples(index=False):
            val = _safe_float(row.value)
            if val is None:
                continue
            try:
                r = baseline.is_anomalous(comp, kpi_str, val, threshold="p99")
            except Exception:
                continue
            if not r.is_anomalous:
                continue

            ts = float(row.timestamp)
            dev = abs(float(r.deviation or 0.0))

            # Global
            agg.anom_count += 1
            if dev > agg.max_dev:
                agg.max_dev = dev
                agg.peak_ts = ts
            if agg.first_ts is None or ts < agg.first_ts:
                agg.first_ts = ts

            # Per bucket
            for b in buckets:
                agg.bucket_counts[b] = agg.bucket_counts.get(b, 0) + 1
                if dev > agg.bucket_max_dev.get(b, 0.0):
                    agg.bucket_max_dev[b] = dev
                if b not in agg.bucket_first_ts or ts < agg.bucket_first_ts[b]:
                    agg.bucket_first_ts[b] = ts

    return aggs


def extract_candidate_features(
    candidate_space: list[dict[str, Any]],
    metric_df: pd.DataFrame | None,
    log_df: pd.DataFrame | None,
    trace_summary: dict[str, Any] | None,
    baseline: Any,
    reason_posterior: Mapping[str, float],
    window_start_ts: float,
    window_end_ts: float,
    gt_component: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract per-candidate features for a single case.

    Returns:
        features: (N_candidates, 33) float array
        onset_ranks: (N_candidates,) int array — rank 0 = earliest onset
        labels: (N_candidates,) int array — 1 for GT match, 0 otherwise
    """
    metric_df = _safe_metric_df(metric_df)
    trace_summary = trace_summary or {}
    window_duration = max(window_end_ts - window_start_ts, 1.0)

    # -- Single pass: compute per-component aggregates --
    comp_aggs = _build_component_aggregates(metric_df, baseline)

    # -- Onset rank map (cross-component) --
    onset_rank_map: dict[str, int] = {}
    sorted_comps = sorted(
        ((c, a.first_ts) for c, a in comp_aggs.items() if a.first_ts is not None),
        key=lambda x: x[1],
    )
    onset_rank_map = {comp: rank for rank, (comp, _) in enumerate(sorted_comps)}

    # -- Trace --
    roles = propagation_roles(trace_summary)
    trace_avail = trace_summary.get("trace_status") == "present"
    first_anomalous_svc = str(trace_summary.get("events", {}).get("first_anomalous_service", ""))
    service_stats = trace_summary.get("service_stats", {})

    # -- Log --
    log_comps: set[str] = set()
    log_texts: dict[str, str] = {}
    if log_df is not None and not log_df.empty and "cmdb_id" in log_df.columns and "value" in log_df.columns:
        grouped = log_df.groupby("cmdb_id")["value"].apply(lambda x: " ".join(x.dropna().astype(str)))
        log_comps = set(grouped.index.astype(str))
        log_texts = {str(k): str(v).lower() for k, v in grouped.items()}

    # -- Normalization factors --
    max_dur_p95 = _max_in_case(service_stats, "duration_p95")
    max_edge_count = _max_edge_count_in_case(trace_summary)

    features_list: list[list[float]] = []
    onset_list: list[int] = []
    labels_list: list[int] = []

    for cand in candidate_space:
        comp = str(cand.get("component", ""))
        bucket = str(cand.get("reason_bucket", ""))
        prior = float(cand.get("prior", 0.0))

        agg = comp_aggs.get(comp) or _CompAgg()

        # Metric (7)
        e = _earliness(agg.first_ts, window_start_ts, window_duration)
        s = math.log1p(agg.max_dev)
        d = math.log1p(agg.anom_count)
        b_cnt = agg.bucket_counts.get(bucket, 0)
        b_dens = b_cnt / max(agg.anom_count, 1)
        cross = math.log1p(len(agg.bucket_counts) - (1 if bucket in agg.bucket_counts else 0))
        sig_s = math.log1p(float(cand.get("details", {}).get("signal_strength", 0) or 0))
        sig_c = math.log1p(float(cand.get("details", {}).get("signal_count", 0) or 0))

        # Temporal (4)
        o_rank = onset_rank_map.get(comp, 99)
        o_gap = _onset_gap(agg.first_ts, comp_aggs, window_duration)
        b_rank = _bucket_onset_rank_v2(comp, bucket, agg, comp_aggs)
        o_peak = _onset_to_peak_v2(agg.first_ts, agg.peak_ts)

        # Trace (8)
        t_avail = 1.0 if trace_avail and comp in service_stats else 0.0
        role = roles.get(comp)
        t_up = 1.0 if role and role.role == "upstream_source" else 0.0
        t_mid = 1.0 if role and role.role == "middle_propagator" else 0.0
        t_down = 1.0 if role and role.role == "downstream_sink" else 0.0
        t_slow = _slow_edge_ratio(comp, trace_summary)
        t_ecnt = _edge_count_norm(comp, trace_summary, max_edge_count)
        t_first = 1.0 if comp == first_anomalous_svc else 0.0
        t_dur = _duration_norm(comp, service_stats, max_dur_p95)

        # Log (6)
        l_avail = 1.0 if comp in log_comps else 0.0
        l_oom, l_gc, l_crash, l_db = _log_keyword_match(comp, log_texts)
        l_kwcnt = (l_oom + l_gc + l_crash + l_db) / 4.0

        # Structural (5)
        stype = service_type(comp)
        s_compat = _type_compat(comp, bucket)
        s_gw = 1.0 if stype in ("gateway", "ingress") else 0.0
        s_app = 1.0 if stype == "app" else 0.0
        s_db = 1.0 if stype in ("database", "cache") else 0.0
        s_mw = 1.0 if stype == "middleware" else 0.0

        # Case-level (3)
        rp = float(reason_posterior.get(bucket, 0.0))
        rm = _reason_margin(reason_posterior, bucket)
        cp = math.log1p(max(prior, 0.0))

        feat = [
            e, s, d, sig_s, sig_c, b_dens, cross,
            o_rank, o_gap, b_rank, o_peak,
            t_avail, t_up, t_mid, t_down, t_slow, t_ecnt, t_first, t_dur,
            l_avail, l_oom, l_gc, l_crash, l_db, l_kwcnt,
            s_compat, s_gw, s_app, s_db, s_mw,
            rp, rm, cp,
        ]
        features_list.append(feat)
        onset_list.append(o_rank)
        labels_list.append(1 if comp == gt_component else 0)

    feats = np.array(features_list, dtype=np.float32)
    onsets = np.array(onset_list, dtype=np.int64)
    labels = np.array(labels_list, dtype=np.int64)

    return feats, onsets, labels


# -- metric helpers --

def _safe_metric_df(df: pd.DataFrame | None) -> pd.DataFrame:
    if df is None:
        return pd.DataFrame(columns=["timestamp", "cmdb_id", "kpi_name", "value"])
    df = df.copy()
    df["cmdb_id"] = df["cmdb_id"].astype(str)
    if "kpi_name" in df.columns:
        df["kpi_name"] = df["kpi_name"].astype(str)
    if "value" in df.columns:
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df


def _earliness(first_ts: float | None, window_start: float, window_duration: float) -> float:
    if first_ts is None or window_duration <= 0:
        return 0.0
    return max(0.0, min(1.0, 1.0 - (first_ts - window_start) / window_duration))


def _onset_gap(first_ts: float | None, aggs: dict[str, _CompAgg], window_duration: float) -> float:
    if first_ts is None or not aggs:
        return 0.0
    earliest = min(a.first_ts for a in aggs.values() if a.first_ts is not None)
    return min(1.0, (first_ts - earliest) / max(window_duration, 1.0))


def _bucket_onset_rank_v2(comp: str, bucket: str, agg: _CompAgg, aggs: dict[str, _CompAgg]) -> float:
    """1/(rank+1) among components with anomalies in this bucket."""
    bt = agg.bucket_first_ts.get(bucket)
    if bt is None:
        return 0.0

    same_bucket: list[tuple[str, float]] = []
    for c, a in aggs.items():
        ts = a.bucket_first_ts.get(bucket)
        if ts is not None:
            same_bucket.append((c, ts))
    same_bucket.sort(key=lambda x: x[1])

    for rank, (c, _) in enumerate(same_bucket):
        if c == comp:
            return 1.0 / (rank + 1.0)
    return 0.0


def _onset_to_peak_v2(first_ts: float | None, peak_ts: float | None) -> float:
    if first_ts is None or peak_ts is None or first_ts >= peak_ts:
        return 0.0
    return math.log1p(peak_ts - first_ts) / math.log1p(3600.0)


# -- trace helpers --

def _max_in_case(stats: dict, key: str) -> float:
    best = 0.0
    for svc in stats.values():
        v = float(svc.get(key, 0.0) or 0.0)
        if v > best:
            best = v
    return best


def _max_edge_count_in_case(trace_summary: dict) -> int:
    if not trace_summary:
        return 1
    edges = trace_summary.get("events", {}).get("slow_edges", []) + trace_summary.get("events", {}).get("dropped_edges", [])
    if not edges:
        return 1
    from collections import Counter
    cnt = Counter()
    for e in edges:
        cnt[str(e.get("src"))] += 1
        cnt[str(e.get("dst"))] += 1
    return max(cnt.values()) if cnt else 1


def _slow_edge_ratio(comp: str, trace_summary: dict) -> float:
    edges = trace_summary.get("events", {}).get("slow_edges", []) if trace_summary else []
    best = 0.0
    for e in edges:
        if str(e.get("src")) == comp or str(e.get("dst")) == comp:
            sr = float(e.get("slow_ratio", 0.0) or 0.0)
            if sr > best:
                best = sr
    return best


def _edge_count_norm(comp: str, trace_summary: dict, max_cnt: int) -> float:
    edges = trace_summary.get("events", {}).get("slow_edges", []) + trace_summary.get("events", {}).get("dropped_edges", []) if trace_summary else []
    cnt = sum(1 for e in edges if str(e.get("src")) == comp or str(e.get("dst")) == comp)
    return cnt / max(max_cnt, 1)


def _duration_norm(comp: str, service_stats: dict, max_dur: float) -> float:
    dur = float(service_stats.get(comp, {}).get("duration_p95", 0.0) or 0.0)
    return dur / max(max_dur, 1.0)


# -- log helpers --

def _log_keyword_match(comp: str, log_texts: dict[str, str]) -> tuple[float, float, float, float]:
    text = log_texts.get(comp, "")
    oom = 1.0 if any(kw in text for kw in _OOM_KEYWORDS) else 0.0
    gc = 1.0 if any(kw in text for kw in _GC_KEYWORDS) else 0.0
    crash = 1.0 if any(kw in text for kw in _CRASH_KEYWORDS) else 0.0
    db = 1.0 if any(kw in text for kw in _DB_KEYWORDS) else 0.0
    return (oom, gc, crash, db)


# -- structural helpers --

def _type_compat(comp: str, bucket: str) -> float:
    from refute_b_v2_d32.joint_candidates import component_family
    family = component_family(comp)
    if family == "other":
        return 0.5
    compat = {
        "docker": ("cpu", "memory", "jvm_oom", "disk_io", "filesystem"),
        "os": ("network_latency", "network_packet_loss"),
        "db": ("db_connection", "disk_io"),
        "redis": ("memory", "disk_io"),
    }
    return 1.0 if bucket in compat.get(family, ()) else 0.0


# -- case-level helpers --

def _reason_margin(posterior: Mapping[str, float], bucket: str) -> float:
    if not posterior:
        return 0.0
    probs = sorted(posterior.values(), reverse=True)
    top1 = probs[0] if probs else 0.0
    top2 = probs[1] if len(probs) > 1 else 0.0
    margin = top1 - top2
    this_prob = posterior.get(bucket, 0.0)
    return min(1.0, margin * 3.0) if this_prob == top1 else 0.0


def _safe_float(val: Any) -> float | None:
    try:
        v = float(val)
        if math.isfinite(v):
            return v
    except (ValueError, TypeError):
        pass
    return None
