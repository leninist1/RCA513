"""Per-entity adaptive anomaly detection.

Replaces the global Z-score threshold (z_th=3.0) with entity-specific baselines.
Each entity has its own per-metric (μ, σ) learned from the baseline window,
and anomaly thresholds adapt to the entity's normal variance.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np


class PerEntityAnomalyDetector:
    """Adaptive anomaly detection with per-entity, per-metric baselines."""

    def __init__(self, global_z_threshold: float = 3.0):
        self.global_z_threshold = global_z_threshold
        self._baselines: Dict[str, Dict[str, Tuple[float, float]]] = {}

    def fit_entity(
        self,
        entity: str,
        metric_name: str,
        baseline_values: np.ndarray,
    ):
        if entity not in self._baselines:
            self._baselines[entity] = {}
        if len(baseline_values) < 5:
            self._baselines[entity][metric_name] = (
                float(np.mean(baseline_values)) if len(baseline_values) > 0 else 0.0,
                1.0,
            )
            return
        mu = float(np.median(baseline_values))
        mad = float(np.median(np.abs(baseline_values - mu)))
        sigma = max(mad * 1.4826, 0.01)
        self._baselines[entity][metric_name] = (mu, sigma)

    def compute_z_score(
        self,
        entity: str,
        metric_name: str,
        value: float,
    ) -> float:
        mu, sigma = self._baselines.get(entity, {}).get(metric_name, (0.0, 1.0))
        return abs(value - mu) / max(sigma, 0.01)

    def is_anomalous(
        self,
        entity: str,
        metric_name: str,
        fault_values: np.ndarray,
        z_threshold: Optional[float] = None,
        persist_ratio: float = 0.3,
    ) -> bool:
        thresh = z_threshold or self.global_z_threshold
        mu, sigma = self._baselines.get(entity, {}).get(metric_name, (0.0, 1.0))
        if len(fault_values) == 0:
            return False
        z_scores = np.abs(fault_values - mu) / max(sigma, 0.01)
        p95 = float(np.percentile(z_scores, 95))
        frac = float(np.sum(z_scores > thresh * 2 / 3)) / len(z_scores)
        return p95 > thresh and frac > persist_ratio

    def entity_anomaly_score(
        self,
        entity: str,
        metric_values: Dict[str, np.ndarray],
        metric_weights: Optional[Dict[str, float]] = None,
        attention_boost: Optional[Dict[str, float]] = None,
        attention_penalty: Optional[Dict[str, float]] = None,
    ) -> float:
        scores = []
        for metric_name, values in metric_values.items():
            if len(values) == 0:
                continue
            mu, sigma = self._baselines.get(entity, {}).get(metric_name, (0.0, 1.0))
            z_scores = np.abs(values - mu) / max(sigma, 0.01)
            p95 = float(np.percentile(z_scores, 95))
            p95_norm = min(p95 / 10.0, 1.0)

            weight = 1.0
            if metric_weights:
                weight *= metric_weights.get(metric_name, 1.0)
            if attention_boost and metric_name in attention_boost:
                weight *= attention_boost[metric_name]
            if attention_penalty and metric_name in attention_penalty:
                weight *= attention_penalty[metric_name]

            scores.append(p95_norm * weight)

        if not scores:
            return 0.0
        return float(np.mean(scores))

    def clear(self):
        self._baselines.clear()

    def entity_count(self) -> int:
        return len(self._baselines)
