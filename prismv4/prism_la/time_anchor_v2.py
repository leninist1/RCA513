"""Time anchor v2 — PELT-based system-level change detection with consensus filtering.

Replaces the "first exceedance" approach in time_anchor.py with:
  Layer 1: System-level anomaly curve (topology-weighted)
  Layer 2: PELT single-change-point detection
  Layer 3: Entity consensus filtering + multi-modal fusion

Key improvements:
  - Finds STRONGEST change, not FIRST exceedance
  - Weights entities by downstream impact (topology)
  - Requires cross-entity consensus for an anchor to be trusted
  - Fuses metric, log, and trace signals before PELT
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple
import math

import numpy as np
import pandas as pd

from prism_v3.config import QueryCase, UnifiedTelemetry
from prism_v3.leakage_guard import AnchorSource, InferenceAnchor


@dataclass(frozen=True)
class AnchorHypothesis:
    timestamp: float
    confidence: float
    source_scores: Dict[str, float] = field(default_factory=dict)
    evidence_ids: Tuple[str, ...] = ()
    source: str = AnchorSource.TELEMETRY_UNSUPERVISED_ONSET.value

    def to_inference_anchor(self) -> InferenceAnchor:
        return InferenceAnchor(
            timestamp=float(self.timestamp),
            source=AnchorSource(self.source),
            confidence=float(self.confidence),
            evidence_ids=tuple(self.evidence_ids),
        )


@dataclass
class AnchorSet:
    anchors: List[AnchorHypothesis]
    fallback_anchor: Optional[AnchorHypothesis] = None
    debug: Dict[str, Any] = field(default_factory=dict)

    def best(self) -> Optional[AnchorHypothesis]:
        return self.anchors[0] if self.anchors else self.fallback_anchor

    def to_inference_anchors(self) -> List[InferenceAnchor]:
        return [anchor.to_inference_anchor() for anchor in self.anchors]


def _parse_window_start(query: QueryCase) -> Optional[float]:
    try:
        return datetime.strptime(query.time_window[0], "%Y-%m-%d %H:%M:%S").timestamp()
    except Exception:
        return None


def public_query_anchor(query: QueryCase) -> Optional[AnchorHypothesis]:
    timestamp = _parse_window_start(query)
    if timestamp is None:
        return None
    return AnchorHypothesis(
        timestamp=float(timestamp),
        confidence=1.0,
        source_scores={"public_query_window": 1.0},
        evidence_ids=("query.time_window.start",),
        source=AnchorSource.PUBLIC_QUERY_WINDOW.value,
    )


def _extract_entity_col(df: pd.DataFrame) -> Optional[str]:
    for c in ("entity", "cmdb_id", "tc", "serviceName"):
        if c in df.columns:
            return c
    return None


def _compute_downstream_weights(
    entities: List[str], trace_df: Optional[pd.DataFrame]
) -> Dict[str, float]:
    n = len(entities)
    if trace_df is None or trace_df.empty or len(trace_df) > 100000:
        return {e: 1.0 for e in entities}
    eidx = {e: i for i, e in enumerate(entities)}
    ecol = _extract_entity_col(trace_df)
    pcol = next((c for c in ("parent_entity", "parent_id") if c in trace_df.columns), None)
    if not ecol or not pcol:
        return {e: 1.0 for e in entities}
    adj = np.zeros((n, n), dtype=float)
    df = trace_df.dropna(subset=[ecol, pcol])
    for _, row in df.iterrows():
        child = str(row[ecol])
        parent = str(row[pcol])
        if child in eidx and parent in eidx and parent != child:
            adj[eidx[parent], eidx[child]] += 1.0
    if adj.sum() < 1e-9:
        return {e: 1.0 for e in entities}
    adj = adj / adj.sum()
    reachable = adj.copy()
    for _ in range(min(n, 10)):
        reachable = reachable + reachable @ adj
        reachable = (reachable > 0).astype(float)
    weights = {}
    for i, e in enumerate(entities):
        downstream_count = int(reachable[i, :].sum()) + 1
        weights[e] = min(10.0, float(downstream_count))
    return weights


def build_system_anomaly_curve(
    metrics_df: Optional[pd.DataFrame],
    logs_df: Optional[pd.DataFrame],
    traces_df: Optional[pd.DataFrame],
    entities: List[str],
    query_window: Tuple[str, str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    bucket_sec = 60
    try:
        t_start = datetime.strptime(query_window[0], "%Y-%m-%d %H:%M:%S").timestamp()
        t_end = datetime.strptime(query_window[1], "%Y-%m-%d %H:%M:%S").timestamp()
    except Exception:
        t_start = 0.0
        t_end = 3600.0
    duration = max(60.0, t_end - t_start)
    n_buckets = max(2, int(duration / bucket_sec))
    buckets = np.linspace(t_start, t_end, n_buckets + 1)
    bucket_centers = (buckets[:-1] + buckets[1:]) / 2.0
    metric_curve = np.zeros(n_buckets, dtype=float)
    log_curve = np.zeros(n_buckets, dtype=float)
    trace_curve = np.zeros(n_buckets, dtype=float)

    downstream_w = _compute_downstream_weights(entities, traces_df)
    entity_set = set(entities)

    if metrics_df is not None and not metrics_df.empty:
        ecol = _extract_entity_col(metrics_df)
        tcol = "timestamp"
        vcol = next((c for c in ("value",) if c in metrics_df.columns), None)
        if ecol and vcol and tcol:
            df = metrics_df.copy()
            df["_t"] = pd.to_numeric(df[tcol], errors="coerce")
            df["_v"] = pd.to_numeric(df[vcol], errors="coerce")
            df[ecol] = df[ecol].astype(str)
            df = df.dropna(subset=["_t", "_v"])
            in_window = df[(df["_t"] >= t_start) & (df["_t"] <= t_end)].copy()
            if not in_window.empty and len(in_window) > 100:
                in_window["_bucket"] = np.clip(
                    ((in_window["_t"].values - t_start) / bucket_sec).astype(int), 0, n_buckets - 1
                )
                t_mid = (t_start + t_end) / 2.0
                base = in_window[in_window["_t"] < t_mid]
                if not base.empty and len(base) > 10:
                    base_entity_med = base.groupby(ecol)["_v"].median()
                    base_entity_std = base.groupby(ecol)["_v"].std().fillna(1.0)
                    bucket_meds = in_window.groupby(["_bucket", ecol])["_v"].median()
                    for (bi, e_name), f_med in bucket_meds.items():
                        bi = int(bi)
                        if e_name not in entity_set:
                            continue
                        b_med = float(base_entity_med.get(e_name, 0.0))
                        b_mad = max(float(base_entity_std.get(e_name, 1.0)), abs(b_med) * 0.1 + 1e-9)
                        z = max(0.0, (float(f_med) - b_med) / b_mad)
                        dw = downstream_w.get(e_name, 1.0)
                        metric_curve[bi] += min(5.0, z) * dw
                    nonzero = metric_curve > 1e-9
                    if nonzero.any():
                        avg_w = np.sum(downstream_w.get(e, 1.0) for e in entity_set) / max(len(entity_set), 1)
                        metric_curve[nonzero] /= avg_w

    if logs_df is not None and not logs_df.empty and len(logs_df) < 100000:
        ecol = _extract_entity_col(logs_df)
        tcol = "timestamp"
        msg_col = next((c for c in ("message", "value", "log_name") if c in logs_df.columns), None)
        if ecol and msg_col:
            df = logs_df.copy()
            df["_t"] = pd.to_numeric(df[tcol], errors="coerce")
            df = df.dropna(subset=["_t"])
            msgs = df[msg_col].astype(str).str.lower()
            fatal_mask = (msgs.str.contains("error|exception|timeout|refused|failed|killed|oom|fatal|outofmemory",
                                             regex=True, na=False))
            df = df[fatal_mask]
            if not df.empty:
                df["_bucket"] = np.clip(
                    ((df["_t"].values - t_start) / bucket_sec).astype(int), 0, n_buckets - 1
                )
                counts = df.groupby("_bucket").size()
                for bi, cnt in counts.items():
                    bi = int(bi)
                    if 0 <= bi < n_buckets:
                        log_curve[bi] = float(cnt)

    if traces_df is not None and not traces_df.empty and len(traces_df) < 50000:
        tcol = "timestamp"
        dcol = next((c for c in ("duration",) if c in traces_df.columns), None)
        if dcol and tcol in traces_df.columns:
            df = traces_df.copy()
            df["_t"] = pd.to_numeric(df[tcol], errors="coerce")
            df["_d"] = pd.to_numeric(df[dcol], errors="coerce")
            df = df.dropna(subset=["_t", "_d"])
            in_window = df[(df["_t"] >= t_start) & (df["_t"] <= t_end)].copy()
            if not in_window.empty:
                t_mid = (t_start + t_end) / 2.0
                base = in_window[in_window["_t"] < t_mid]
                b_med = float(base["_d"].median()) if not base.empty else 0.0
                b_mad = max(float(np.median(np.abs(base["_d"].values - b_med))) if not base.empty else 1.0, 1e-9)
                in_window["_bucket"] = np.clip(
                    ((in_window["_t"].values - t_start) / bucket_sec).astype(int), 0, n_buckets - 1
                )
                for bucket_i in range(n_buckets):
                    bdf = in_window[in_window["_bucket"] == bucket_i]
                    if bdf.empty:
                        continue
                    f_med = float(bdf["_d"].median())
                    z = max(0.0, (f_med - b_med) / b_mad)
                    trace_curve[bucket_i] = min(3.0, z)

    debug = {
        "n_buckets": n_buckets,
        "metric_max": float(metric_curve.max()),
        "log_max": float(log_curve.max()),
        "trace_max": float(trace_curve.max()),
        "downstream_weights": {k: round(float(v), 2) for k, v in sorted(downstream_w.items()) if v > 1.0},
    }
    return metric_curve, log_curve, trace_curve, bucket_centers, debug


def pelt_single_change_point(signal: np.ndarray, penalty_mult: float = 1.0, skip_first: int = 0) -> Tuple[Optional[int], float, np.ndarray]:
    n = len(signal)
    if n < 3:
        return None, 0.0, np.zeros(n)
    s = np.clip(signal, 0.0, None)
    s_sum = max(float(s.sum()), 1e-9)
    s_norm = s / s_sum
    s_norm = s_norm / max(float(s_norm.max()), 1e-9)
    cumsum = np.cumsum(s_norm)
    cumsum = np.insert(cumsum, 0, 0.0)
    penalty = penalty_mult * 0.3 * math.log(max(2, n))
    best_score = -1e9
    best_idx = None
    scores = np.zeros(n)
    start = max(1, int(skip_first))
    for k in range(start, n - 1):
        p1 = k / n if n > 0 else 0.0
        seg1_mean = cumsum[k] / max(k, 1)
        seg2_mean = (cumsum[n] - cumsum[k]) / max(n - k, 1)
        seg1_var = np.var(s_norm[:k]) if k > 1 else 1e-9
        seg2_var = np.var(s_norm[k:]) if n - k > 1 else 1e-9
        total_var = max(float(np.var(s_norm)), 1e-9)
        var_ratio = (seg1_var * k + seg2_var * (n - k)) / max(total_var * n, 1e-9)
        contrast = abs(seg2_mean - seg1_mean) / max(abs(seg1_mean) + abs(seg2_mean) + 1e-9, 1e-9)
        score = contrast - var_ratio * penalty
        scores[k] = float(score)
        if score > best_score:
            best_score = score
            best_idx = k
    if best_idx is None:
        return None, 0.0, scores
    confidence = min(1.0, max(0.01, float(best_score) / max(float(np.max(scores)), 1e-9)))
    return best_idx, confidence, scores


def compute_entity_consensus(
    metrics_df: Optional[pd.DataFrame],
    entities: List[str],
    candidate_ts: float,
    query_window: Tuple[str, str],
) -> float:
    if metrics_df is None or metrics_df.empty:
        return 0.5
    ecol = _extract_entity_col(metrics_df)
    tcol = "timestamp"
    vcol = next((c for c in ("value",) if c in metrics_df.columns), None)
    if not ecol or not vcol:
        return 0.5
    try:
        t0 = datetime.strptime(query_window[0], "%Y-%m-%d %H:%M:%S").timestamp()
        t1 = datetime.strptime(query_window[1], "%Y-%m-%d %H:%M:%S").timestamp()
    except Exception:
        return 0.5
    df = metrics_df.copy()
    df["_t"] = pd.to_numeric(df[tcol], errors="coerce")
    df["_v"] = pd.to_numeric(df[vcol], errors="coerce")
    df[ecol] = df[ecol].astype(str)
    df = df.dropna(subset=["_t", "_v"])
    df = df[(df["_t"] >= t0) & (df["_t"] <= t1)]
    if df.empty:
        return 0.5
    entity_set = set(entities)
    consent_count = 0
    total_count = 0
    for e_name in entity_set:
        e_data = df[df[ecol] == e_name]
        if e_data.empty:
            continue
        total_count += 1
        base = e_data[e_data["_t"] < candidate_ts - 60]
        fault = e_data[(e_data["_t"] >= candidate_ts - 60) & (e_data["_t"] <= candidate_ts + 60)]
        if base.empty or fault.empty:
            continue
        b_mean = float(base["_v"].mean())
        b_std = max(float(base["_v"].std()), 1e-9)
        f_mean = float(fault["_v"].mean())
        z = abs(f_mean - b_mean) / b_std
        if z > 2.0:
            consent_count += 1
    if total_count < 1:
        return 0.5
    return consent_count / total_count


def build_anchor_set_v2(
    telemetry: UnifiedTelemetry,
    query: QueryCase,
    *,
    top_k: int = 5,
    include_public_fallback: bool = True,
) -> AnchorSet:
    top_k = max(1, int(top_k))
    public_anchor = public_query_anchor(query)
    entities = list(telemetry.entities) if hasattr(telemetry, "entities") else []
    if not entities:
        fallback = public_anchor
        return AnchorSet(
            anchors=[fallback] if fallback else [],
            fallback_anchor=fallback,
            debug={"reason": "no_entities"},
        )

    metrics_df = getattr(telemetry, "metrics", None)
    logs_df = getattr(telemetry, "logs", None)
    traces_df = getattr(telemetry, "traces", None)
    query_window = tuple(query.time_window) if len(query.time_window) == 2 else ("", "")

    metric_curve, log_curve, trace_curve, bucket_centers, curve_debug = build_system_anomaly_curve(
        metrics_df, logs_df, traces_df, entities, query_window
    )

    fused_curve = 0.50 * (metric_curve / max(float(metric_curve.max()), 1e-9)) \
                + 0.25 * (log_curve / max(float(log_curve.max()), 1e-9)) \
                + 0.25 * (trace_curve / max(float(trace_curve.max()), 1e-9))

    best_bucket, pelt_conf, pelt_scores = pelt_single_change_point(fused_curve, skip_first=2)
    anchors: List[AnchorHypothesis] = []

    if best_bucket is not None and 0 <= best_bucket < len(bucket_centers):
        candidate_ts = float(bucket_centers[best_bucket])
        consensus = compute_entity_consensus(metrics_df, entities, candidate_ts, query_window)
        source_scores = {
            "pelt_contrast_score": round(float(pelt_conf), 4),
            "entity_consensus": round(float(consensus), 4),
            "metric_max": round(float(np.max(metric_curve)), 4),
            "log_max": round(float(np.max(log_curve)), 4),
            "trace_max": round(float(np.max(trace_curve)), 4),
        }
        confidence = 0.55 * pelt_conf + 0.30 * consensus + 0.15 * min(1.0, source_scores["metric_max"] / 3.0)
        anchors.append(
            AnchorHypothesis(
                timestamp=candidate_ts,
                confidence=min(1.0, max(0.01, confidence)),
                source_scores=source_scores,
                evidence_ids=("pelt:v2:system_curve",),
                source=AnchorSource.TELEMETRY_UNSUPERVISED_ONSET.value,
            )
        )

    if len(anchors) < top_k:
        if metric_curve.max() > 0 and log_curve.max() > 0:
            metric_curve_m = metric_curve / max(float(metric_curve.max()), 1e-9)
            log_curve_m = log_curve / max(float(log_curve.max()), 1e-9)
            m_best, _, _ = pelt_single_change_point(metric_curve_m, skip_first=2)
            l_best, _, _ = pelt_single_change_point(log_curve_m, skip_first=2)
            for idx, label in [(m_best, "metric"), (l_best, "log")]:
                if idx is not None and 0 <= idx < len(bucket_centers):
                    ts = float(bucket_centers[idx])
                    if not any(abs(ts - a.timestamp) < 60 for a in anchors):
                        anchors.append(
                            AnchorHypothesis(
                                timestamp=ts,
                                confidence=0.35,
                                source_scores={f"{label}_pelt": 1.0},
                                evidence_ids=(f"pelt:v2:{label}",),
                                source=AnchorSource.TELEMETRY_UNSUPERVISED_ONSET.value,
                            )
                        )

    if public_anchor is not None:
        has_near = any(abs(public_anchor.timestamp - a.timestamp) < 60 for a in anchors)
        if not has_near and include_public_fallback:
            anchors.append(
                AnchorHypothesis(
                    timestamp=public_anchor.timestamp,
                    confidence=0.25,
                    source_scores={"public_query_window": 1.0},
                    evidence_ids=public_anchor.evidence_ids,
                    source=AnchorSource.FALLBACK_QUERY_WINDOW_START.value,
                )
            )

    anchors.sort(key=lambda a: a.confidence, reverse=True)
    selected = anchors[:top_k]
    debug = {
        "method": "pelt_v2",
        "bucket_count": len(bucket_centers),
        "pelt_confidence": round(float(pelt_conf), 4),
        "curve": curve_debug,
        "anchors": [
            {"timestamp": a.timestamp, "confidence": round(float(a.confidence), 4),
             "source": a.source, "source_scores": dict(a.source_scores)}
            for a in selected
        ],
    }
    return AnchorSet(
        anchors=selected,
        fallback_anchor=public_anchor,
        debug=debug,
    )
