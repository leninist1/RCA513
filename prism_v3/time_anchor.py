"""Unsupervised, no-label time-anchor discovery for PRISM v3."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple
import math

import numpy as np
import pandas as pd

from .config import QueryCase, UnifiedTelemetry
from .leakage_guard import AnchorSource, InferenceAnchor


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


def build_anchor_set(
    telemetry: UnifiedTelemetry,
    query: QueryCase,
    *,
    top_k: int = 5,
    include_public_fallback: bool = True,
) -> AnchorSet:
    """Return top-k no-GT anchor hypotheses from telemetry and public query text."""
    top_k = max(1, int(top_k))
    public_anchor = public_query_anchor(query)
    candidates: List[AnchorHypothesis] = []
    public_ts = public_anchor.timestamp if public_anchor is not None else None
    candidates.extend(_metric_change_anchors(telemetry.metrics, public_ts))
    candidates.extend(_log_burst_anchors(telemetry.logs, public_ts))
    candidates.extend(_trace_shift_anchors(telemetry.traces, public_ts))
    fused = _fuse_candidates(candidates, public_anchor)
    if not fused and public_anchor is not None:
        fused = [public_anchor]
    elif include_public_fallback and public_anchor is not None:
        if all(abs(anchor.timestamp - public_anchor.timestamp) > 60 for anchor in fused):
            fallback = AnchorHypothesis(
                timestamp=public_anchor.timestamp,
                confidence=min(0.35, public_anchor.confidence),
                source_scores={"public_query_window": 1.0},
                evidence_ids=public_anchor.evidence_ids,
                source=AnchorSource.FALLBACK_QUERY_WINDOW_START.value,
            )
            fused.append(fallback)
    fused = _normalize_confidence(fused)
    fused.sort(key=lambda item: item.confidence, reverse=True)
    selected = fused[:top_k]
    fallback_anchor = public_anchor
    debug = {
        "candidate_count": len(candidates),
        "selected_count": len(selected),
        "public_query_anchor": public_anchor.timestamp if public_anchor else None,
        "anchors": [
            {
                "timestamp": anchor.timestamp,
                "confidence": round(float(anchor.confidence), 6),
                "source": anchor.source,
                "source_scores": dict(anchor.source_scores),
                "evidence_ids": list(anchor.evidence_ids),
            }
            for anchor in selected
        ],
    }
    return AnchorSet(anchors=selected, fallback_anchor=fallback_anchor, debug=debug)


def anchor_stability(
    rankings_by_anchor: MappingLike,
    *,
    top_k: int = 10,
) -> Dict[str, float]:
    """Compute entity stability across anchor-weighted rankings.

    ``rankings_by_anchor`` accepts an iterable of ``(confidence, [object_id...])``.
    """
    totals: Dict[str, float] = {}
    denom = 0.0
    for confidence, ranking in rankings_by_anchor:
        weight = max(0.0, float(confidence))
        denom += weight
        for object_id in list(ranking)[: max(1, int(top_k))]:
            totals[str(object_id)] = totals.get(str(object_id), 0.0) + weight
    if denom <= 0:
        return {key: 0.0 for key in totals}
    return {key: float(value / denom) for key, value in totals.items()}


MappingLike = Iterable[Tuple[float, Iterable[str]]]


def _metric_change_anchors(metrics: Optional[pd.DataFrame], public_ts: Optional[float]) -> List[AnchorHypothesis]:
    if metrics is None or metrics.empty or "timestamp" not in metrics.columns or "value" not in metrics.columns:
        return []
    frame = _windowed_frame(metrics, public_ts)
    if frame.empty:
        return []
    anchors: List[AnchorHypothesis] = []
    group_cols = [col for col in ("entity", "metric_name") if col in frame.columns]
    if not group_cols:
        group_cols = ["timestamp"]
    for key, group in frame.groupby(group_cols):
        if len(group) < 6:
            continue
        group = group.sort_values("timestamp")
        values = pd.to_numeric(group["value"], errors="coerce")
        valid = group.loc[values.notna()].copy()
        values = values.dropna()
        if len(values) < 6:
            continue
        median = float(values.median())
        mad = float(np.median(np.abs(values - median))) or float(values.std()) or 1.0
        z = np.abs((values.to_numpy(dtype=float) - median) / max(mad * 1.4826, 1e-6))
        persistent = _first_persistent_index(z, threshold=3.0, width=2)
        if persistent is None:
            continue
        ts = float(valid.iloc[persistent]["timestamp"])
        score = float(np.tanh(np.nanmax(z[persistent : persistent + 5]) / 6.0))
        evidence = "metric:" + ":".join(str(part) for part in (key if isinstance(key, tuple) else (key,)))
        anchors.append(
            AnchorHypothesis(
                timestamp=ts,
                confidence=score,
                source_scores={"metric_change_score": score, "metric_persistence": 1.0},
                evidence_ids=(evidence,),
                source=AnchorSource.TELEMETRY_UNSUPERVISED_ONSET.value,
            )
        )
    return sorted(anchors, key=lambda item: item.confidence, reverse=True)[:20]


def _log_burst_anchors(logs: Optional[pd.DataFrame], public_ts: Optional[float]) -> List[AnchorHypothesis]:
    if logs is None or logs.empty or "timestamp" not in logs.columns:
        return []
    frame = _windowed_frame(logs, public_ts)
    if frame.empty:
        return []
    messages = frame.get("message", pd.Series([""] * len(frame))).astype(str).str.lower()
    fatal = messages.str.contains("error|exception|timeout|oom|killed|refused|failed", regex=True, na=False)
    if not bool(fatal.any()):
        return []
    fault_logs = frame.loc[fatal].copy()
    fault_logs["bucket"] = (fault_logs["timestamp"].astype(float) // 60) * 60
    counts = fault_logs.groupby("bucket").size().sort_index()
    if counts.empty:
        return []
    baseline = float(np.median(counts.to_numpy(dtype=float))) or 1.0
    rows: List[AnchorHypothesis] = []
    for bucket, count in counts.items():
        burst = float(count) / max(baseline, 1.0)
        score = float(np.tanh(burst / 4.0))
        if score < 0.25:
            continue
        rows.append(
            AnchorHypothesis(
                timestamp=float(bucket),
                confidence=score,
                source_scores={"log_burst_score": score, "fatal_keyword_score": min(1.0, burst / 5.0)},
                evidence_ids=(f"log:bucket:{int(bucket)}",),
                source=AnchorSource.TELEMETRY_UNSUPERVISED_ONSET.value,
            )
        )
    return sorted(rows, key=lambda item: item.confidence, reverse=True)[:10]


def _trace_shift_anchors(traces: Optional[pd.DataFrame], public_ts: Optional[float]) -> List[AnchorHypothesis]:
    if traces is None or traces.empty or "timestamp" not in traces.columns:
        return []
    frame = _windowed_frame(traces, public_ts)
    if frame.empty:
        return []
    work = frame.copy()
    work["bucket"] = (work["timestamp"].astype(float) // 60) * 60
    duration = pd.to_numeric(work.get("duration", 0.0), errors="coerce").fillna(0.0)
    work["duration_value"] = duration
    grouped = work.groupby("bucket")
    latency = grouped["duration_value"].median().sort_index()
    if latency.empty:
        return []
    values = latency.to_numpy(dtype=float)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median))) or float(np.std(values)) or 1.0
    z = np.abs((values - median) / max(mad * 1.4826, 1e-6))
    idx = _first_persistent_index(z, threshold=2.5, width=1)
    anchors: List[AnchorHypothesis] = []
    if idx is not None:
        ts = float(latency.index[idx])
        score = float(np.tanh(z[idx] / 5.0))
        anchors.append(
            AnchorHypothesis(
                timestamp=ts,
                confidence=score,
                source_scores={"trace_latency_shift": score},
                evidence_ids=(f"trace:latency:{int(ts)}",),
                source=AnchorSource.TELEMETRY_UNSUPERVISED_ONSET.value,
            )
        )
    if "status_code" in work.columns:
        err = work["status_code"].astype(str).str.startswith(("4", "5"))
        err_rate = err.groupby(work["bucket"]).mean().sort_index()
        if not err_rate.empty and float(err_rate.max()) > 0:
            ts = float(err_rate.idxmax())
            score = float(min(1.0, err_rate.max() * 3.0))
            anchors.append(
                AnchorHypothesis(
                    timestamp=ts,
                    confidence=score,
                    source_scores={"trace_error_shift": score},
                    evidence_ids=(f"trace:error:{int(ts)}",),
                    source=AnchorSource.TELEMETRY_UNSUPERVISED_ONSET.value,
                )
            )
    return sorted(anchors, key=lambda item: item.confidence, reverse=True)[:10]


def _fuse_candidates(
    candidates: List[AnchorHypothesis],
    public_anchor: Optional[AnchorHypothesis],
) -> List[AnchorHypothesis]:
    buckets: List[List[AnchorHypothesis]] = []
    for candidate in sorted(candidates, key=lambda item: item.timestamp):
        for bucket in buckets:
            if abs(bucket[0].timestamp - candidate.timestamp) <= 60:
                bucket.append(candidate)
                break
        else:
            buckets.append([candidate])
    fused: List[AnchorHypothesis] = []
    for bucket in buckets:
        total = sum(max(0.0, item.confidence) for item in bucket)
        if total <= 0:
            continue
        timestamp = sum(item.timestamp * max(0.0, item.confidence) for item in bucket) / total
        source_scores: Dict[str, float] = {}
        evidence_ids: List[str] = []
        modalities = set()
        for item in bucket:
            evidence_ids.extend(item.evidence_ids)
            for key, value in item.source_scores.items():
                source_scores[key] = max(source_scores.get(key, 0.0), float(value))
                modalities.add(key.split("_", 1)[0])
        metric_score = source_scores.get("metric_change_score", 0.0)
        log_score = source_scores.get("log_burst_score", 0.0)
        trace_score = max(source_scores.get("trace_latency_shift", 0.0), source_scores.get("trace_error_shift", 0.0))
        cross_modal = min(1.0, max(0, len(modalities) - 1) / 2.0)
        isolated_penalty = 0.0 if len(modalities) > 1 else 0.15
        public_closeness = 0.0
        if public_anchor is not None:
            public_closeness = 1.0 / (1.0 + abs(timestamp - public_anchor.timestamp) / 600.0)
        score = (
            0.40 * metric_score
            + 0.20 * log_score
            + 0.20 * trace_score
            + 0.15 * cross_modal
            + 0.10 * public_closeness
            - 0.05 * isolated_penalty
        )
        fused.append(
            AnchorHypothesis(
                timestamp=float(timestamp),
                confidence=max(0.0, min(1.0, score)),
                source_scores=source_scores,
                evidence_ids=tuple(sorted(set(evidence_ids))),
                source=AnchorSource.TELEMETRY_UNSUPERVISED_ONSET.value,
            )
        )
    return fused


def _normalize_confidence(anchors: List[AnchorHypothesis]) -> List[AnchorHypothesis]:
    total = sum(max(0.0, anchor.confidence) for anchor in anchors)
    if total <= 0:
        return anchors
    return [
        AnchorHypothesis(
            timestamp=anchor.timestamp,
            confidence=max(0.01, min(1.0, anchor.confidence / total)),
            source_scores=anchor.source_scores,
            evidence_ids=anchor.evidence_ids,
            source=anchor.source,
        )
        for anchor in anchors
    ]


def _windowed_frame(frame: pd.DataFrame, public_ts: Optional[float]) -> pd.DataFrame:
    if public_ts is None or "timestamp" not in frame.columns:
        return frame.copy()
    start = float(public_ts) - 900
    end = float(public_ts) + 1800
    return frame[(frame["timestamp"].astype(float) >= start) & (frame["timestamp"].astype(float) <= end)].copy()


def _first_persistent_index(values: np.ndarray, *, threshold: float, width: int) -> Optional[int]:
    if values.size == 0:
        return None
    for idx in range(values.size):
        window = values[idx : idx + max(1, width)]
        if window.size and np.all(window >= threshold):
            return int(idx)
    return None


def _parse_window_start(query: QueryCase) -> Optional[float]:
    try:
        return datetime.strptime(query.time_window[0], "%Y-%m-%d %H:%M:%S").timestamp()
    except Exception:
        return None
