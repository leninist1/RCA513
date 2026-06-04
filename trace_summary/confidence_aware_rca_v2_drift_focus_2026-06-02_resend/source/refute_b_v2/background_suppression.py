"""Background-suppressed component scoring.

The current runner is vulnerable to a weak-root/strong-background failure
mode: a non-root component can dominate the candidate pool with an extreme
single KPI deviation. This module converts raw anomaly events into calibrated
component scores using clipping, aggregation, and optional historical
component/hour distributions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import log1p
from statistics import median
from typing import Iterable, Mapping


@dataclass(frozen=True)
class AnomalyEvent:
    timestamp: int
    component: str
    reason: str
    strength: float
    source: str = "metric"
    details: Mapping | None = None


@dataclass(frozen=True)
class ComponentScore:
    component: str
    reason: str
    raw_max: float
    clipped_sum: float
    relative_score: float
    baseline_center: float | None
    event_count: int
    calibrated_percentile: float
    rarity_score: float
    score: float
    timestamps: tuple[int, ...] = ()

    def to_dict(self) -> dict:
        return {
            "component": self.component,
            "reason": self.reason,
            "raw_max": self.raw_max,
            "clipped_sum": self.clipped_sum,
            "relative_score": self.relative_score,
            "baseline_center": self.baseline_center,
            "event_count": self.event_count,
            "calibrated_percentile": self.calibrated_percentile,
            "rarity_score": self.rarity_score,
            "score": self.score,
            "timestamps": list(self.timestamps),
        }


@dataclass
class BackgroundCalibrator:
    history: dict[tuple[str, str, int], list[float]] = field(default_factory=dict)

    def add(self, component: str, reason: str, hour_bucket: int, score: float) -> None:
        self.history.setdefault((str(component), str(reason), int(hour_bucket)), []).append(float(score))

    def percentile(self, component: str, reason: str, hour_bucket: int, score: float) -> float:
        samples = self.history.get((str(component), str(reason), int(hour_bucket)), [])
        if not samples:
            samples = [v for (comp, rsn, _), vals in self.history.items() if comp == str(component) and rsn == str(reason) for v in vals]
        if not samples:
            return 0.5
        below = sum(1 for value in samples if value <= score)
        return below / len(samples)

    def rarity(self, component: str, reason: str, hour_bucket: int, score: float) -> float:
        p = self.percentile(component, reason, hour_bucket, score)
        return max(0.0, -1.0 * _safe_log10_tail(1.0 - p))

    def samples_for(self, component: str, reason: str, hour_bucket: int) -> list[float]:
        samples = self.history.get((str(component), str(reason), int(hour_bucket)), [])
        if samples:
            return list(samples)
        return [
            v
            for (comp, rsn, _), vals in self.history.items()
            if comp == str(component) and rsn == str(reason)
            for v in vals
        ]

    def relative_score(self, component: str, reason: str, hour_bucket: int, score: float) -> tuple[float, float | None]:
        """Return current score relative to the component's own background.

        This is the key weak-root/strong-background correction: a component that
        is always noisy should not win merely because its absolute deviation is
        large. With no component history, the function returns the clipped score
        and a missing baseline marker.
        """
        samples = self.samples_for(component, reason, hour_bucket)
        if not samples:
            return float(score), None
        center = median(samples)
        if center <= 1e-9:
            return float(score), float(center)
        return float(score) / float(center), float(center)


def _safe_log10_tail(tail: float) -> float:
    import math
    return math.log10(max(tail, 1e-6))


def hour_bucket(timestamp: int, bucket_hours: int = 1) -> int:
    return int((int(timestamp) % 86400) // (bucket_hours * 3600))


def component_scores(events: Iterable[AnomalyEvent], calibrator: BackgroundCalibrator | None = None,
                     top_timestamps: int = 5) -> list[ComponentScore]:
    calibrator = calibrator or BackgroundCalibrator()
    grouped: dict[tuple[str, str], list[AnomalyEvent]] = {}
    for event in events:
        grouped.setdefault((str(event.component), str(event.reason)), []).append(event)
    rows = []
    for (component, reason), group in grouped.items():
        raw_max = max(abs(float(event.strength or 0.0)) for event in group)
        by_ts: dict[int, float] = {}
        for event in group:
            by_ts[int(event.timestamp)] = by_ts.get(int(event.timestamp), 0.0) + log1p(abs(float(event.strength or 0.0)))
        clipped_sum = sum(by_ts.values())
        dominant_ts = sorted(by_ts, key=lambda ts: (-by_ts[ts], ts))[:top_timestamps]
        hb = hour_bucket(dominant_ts[0]) if dominant_ts else 0
        percentile = calibrator.percentile(component, reason, hb, clipped_sum)
        rarity = calibrator.rarity(component, reason, hb, clipped_sum)
        relative, center = calibrator.relative_score(component, reason, hb, clipped_sum)
        # The final score is dominated by component-specific relative deviation.
        # clipped_sum prevents single-KPI explosions; percentile/rarity break ties
        # when historical component baselines exist.
        score = relative * (0.5 + percentile) + rarity
        rows.append(ComponentScore(
            component=component,
            reason=reason,
            raw_max=raw_max,
            clipped_sum=clipped_sum,
            relative_score=relative,
            baseline_center=center,
            event_count=len(group),
            calibrated_percentile=percentile,
            rarity_score=rarity,
            score=score,
            timestamps=tuple(dominant_ts),
        ))
    return sorted(rows, key=lambda row: (-row.score, -row.relative_score, -row.clipped_sum, row.component, row.reason))


def fit_background_calibrator(windows: Iterable[Iterable[AnomalyEvent]]) -> BackgroundCalibrator:
    calibrator = BackgroundCalibrator()
    for events in windows:
        for row in component_scores(events):
            ts = row.timestamps[0] if row.timestamps else 0
            calibrator.add(row.component, row.reason, hour_bucket(ts), row.clipped_sum)
    return calibrator
