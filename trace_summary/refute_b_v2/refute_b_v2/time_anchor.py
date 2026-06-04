"""Component-conditional time anchor detection."""
from __future__ import annotations

from dataclasses import dataclass
from math import log1p
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class TimeAnchorPolicy:
    cross_metric_vote_weight: float = 1.0
    strongest_spike_bonus: float = 1.0
    sustained_onset_bonus: float = 1.0
    candidate_seed_bonus: float = 0.5
    vote_strength_threshold: float = 3.0

    def to_dict(self) -> dict:
        return {
            "cross_metric_vote_weight": self.cross_metric_vote_weight,
            "strongest_spike_bonus": self.strongest_spike_bonus,
            "sustained_onset_bonus": self.sustained_onset_bonus,
            "candidate_seed_bonus": self.candidate_seed_bonus,
            "vote_strength_threshold": self.vote_strength_threshold,
        }


@dataclass(frozen=True)
class TimeAnchor:
    timestamp: int
    score: float
    kind: str
    metric_votes: int
    details: dict

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "score": self.score,
            "kind": self.kind,
            "metric_votes": self.metric_votes,
            "details": self.details,
        }


def component_time_anchors(metric_df: pd.DataFrame, component: str, baseline,
                           candidate_timestamps: Iterable[int] = (),
                           top_k: int = 3,
                           policy: TimeAnchorPolicy | None = None) -> list[TimeAnchor]:
    policy = policy or TimeAnchorPolicy()
    candidate_timestamps = tuple(int(ts // 60 * 60) for ts in candidate_timestamps)
    if metric_df is None or metric_df.empty:
        return _candidate_seed_anchors(candidate_timestamps, policy, top_k)
    rows = metric_df[metric_df["cmdb_id"].astype(str) == str(component)].copy()
    if rows.empty:
        return _candidate_seed_anchors(candidate_timestamps, policy, top_k)
    rows["value"] = pd.to_numeric(rows["value"], errors="coerce")
    rows = rows.dropna(subset=["timestamp", "kpi_name", "value"])
    scored = []
    for row in rows.itertuples(index=False):
        try:
            result = baseline.is_anomalous(row.cmdb_id, row.kpi_name, row.value, threshold="p99")
            strength = abs(float(result.deviation or 0.0)) if result.is_anomalous else 0.0
        except Exception:
            strength = 0.0
        if strength <= 0:
            continue
        scored.append({
            "timestamp": int(row.timestamp // 60 * 60),
            "kpi_name": str(row.kpi_name),
            "strength": strength,
            "score": log1p(strength),
        })
    if not scored:
        return _candidate_seed_anchors(candidate_timestamps, policy, top_k)
    df = pd.DataFrame(scored)
    anchors: list[TimeAnchor] = []
    for ts, group in df.groupby("timestamp"):
        score = float(group["score"].sum())
        votes = int((group["strength"] >= policy.vote_strength_threshold).sum())
        anchors.append(TimeAnchor(
            timestamp=int(ts),
            score=score + votes * policy.cross_metric_vote_weight,
            kind="cross_metric_vote",
            metric_votes=votes,
            details={"kpis": sorted(group["kpi_name"].unique()), "raw_score": score, "policy": policy.to_dict()},
        ))
    strongest = df.sort_values(["strength", "timestamp"], ascending=[False, True]).iloc[0]
    anchors.append(TimeAnchor(
        timestamp=int(strongest["timestamp"]),
        score=float(log1p(strongest["strength"]) + policy.strongest_spike_bonus),
        kind="strongest_spike",
        metric_votes=1,
        details={"kpi_name": strongest["kpi_name"], "strength": float(strongest["strength"])},
    ))
    onset = _sustained_onset(df, policy)
    if onset is not None:
        anchors.append(onset)
    for ts in candidate_timestamps:
        nearest = _nearest_anchor(int(ts), anchors)
        if nearest:
            anchors.append(TimeAnchor(
                timestamp=nearest.timestamp,
                score=nearest.score + policy.candidate_seed_bonus,
                kind="candidate_seed_nearest",
                metric_votes=nearest.metric_votes,
                details={"seed_timestamp": int(ts), "base_kind": nearest.kind},
            ))
    return _dedupe_anchors(anchors)[:top_k]


def _candidate_seed_anchors(candidate_timestamps: Iterable[int], policy: TimeAnchorPolicy, top_k: int) -> list[TimeAnchor]:
    counts: dict[int, int] = {}
    for ts in candidate_timestamps:
        counts[int(ts)] = counts.get(int(ts), 0) + 1
    anchors = [
        TimeAnchor(
            timestamp=ts,
            score=float(count + policy.candidate_seed_bonus),
            kind="candidate_seed_only",
            metric_votes=0,
            details={"seed_count": count, "policy": policy.to_dict()},
        )
        for ts, count in counts.items()
    ]
    return sorted(anchors, key=lambda anchor: (-anchor.score, anchor.timestamp))[:top_k]


def _sustained_onset(df: pd.DataFrame, policy: TimeAnchorPolicy) -> TimeAnchor | None:
    by_ts = df.groupby("timestamp")["score"].sum().sort_index()
    if len(by_ts) < 2:
        return None
    threshold = max(1.0, float(by_ts.quantile(0.75)))
    ordered = list(by_ts.items())
    for idx, (ts, score) in enumerate(ordered[:-1]):
        if score >= threshold and ordered[idx + 1][1] >= threshold:
            return TimeAnchor(int(ts), float(score + policy.sustained_onset_bonus), "sustained_onset", 2, {"threshold": threshold, "policy": policy.to_dict()})
    return None


def _nearest_anchor(timestamp: int, anchors: list[TimeAnchor]) -> TimeAnchor | None:
    if not anchors:
        return None
    return min(anchors, key=lambda anchor: (abs(anchor.timestamp - timestamp), -anchor.score))


def _dedupe_anchors(anchors: list[TimeAnchor]) -> list[TimeAnchor]:
    best: dict[int, TimeAnchor] = {}
    for anchor in anchors:
        old = best.get(anchor.timestamp)
        if old is None or anchor.score > old.score:
            best[anchor.timestamp] = anchor
    return sorted(best.values(), key=lambda anchor: (-anchor.score, anchor.timestamp))
