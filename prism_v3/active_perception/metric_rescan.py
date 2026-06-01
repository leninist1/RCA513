"""Hypothesis-driven metric re-analysis (Active Perception D1).

Instead of using fixed global anomaly scores, re-analyzes metrics with:
  1. Entity-specific baseline (μ, σ) instead of global Z-threshold
  2. Hypothesis-driven attention: boost metrics matching the suspected fault type
  3. Adaptive significance testing

Input: raw baseline_df, fault_df for the target entity
Output: updated anomaly evidence specific to the current hypothesis

Cost: ~0.3s (lightweight, no counterfactual)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np


class MetricRescanAction:
    """Hypothesis-driven metric re-evaluation for a specific entity."""

    def __init__(self):
        self._baselines: Dict[str, Dict[str, Tuple[float, float]]] = {}

    def fit_baseline(
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
        sigma = float(np.median(np.abs(baseline_values - mu))) * 1.4826
        self._baselines[entity][metric_name] = (mu, max(sigma, 0.01))

    def rescan_entity(
        self,
        entity: str,
        metric_values: Dict[
            str, Tuple[np.ndarray, np.ndarray]
        ],  # (baseline, fault) per metric
        hypothesis_families: Optional[List[str]] = None,
        metric_family_map: Optional[Dict[str, str]] = None,
        base_z_threshold: float = 3.0,
    ) -> Dict[str, Dict[str, float]]:
        """Re-evaluate anomaly scores for one entity under a specific hypothesis.

        Args:
            entity: Target entity name
            metric_values: {metric_name: (baseline_values, fault_values)}
            hypothesis_families: Families to boost (e.g., ["cpu", "memory"])
            metric_family_map: {metric_name: family} map
            base_z_threshold: Base Z-score threshold

        Returns:
            {
                "z_by_metric": {metric_name: z_p95},
                "entity_score": float (0-1),
                "attention_boosted_metrics": [...],
                "attention_penalized_metrics": [...],
            }
        """
        if hypothesis_families is None:
            hypothesis_families = []

        z_scores = {}
        original_z = {}
        boosted = []
        penalized = []

        for metric_name, (baseline_vals, fault_vals) in metric_values.items():
            if metric_name not in self._baselines.get(entity, {}):
                self.fit_baseline(entity, metric_name, baseline_vals)

            mu, sigma = self._baselines[entity][metric_name]
            if len(fault_vals) == 0:
                continue

            z = np.abs(fault_vals - mu) / max(sigma, 0.01)
            p95 = float(np.percentile(z, 95))
            original_z[metric_name] = p95

            family = metric_family_map.get(metric_name, "") if metric_family_map else ""
            if hypothesis_families and family in hypothesis_families:
                p95 *= 1.5
                boosted.append(metric_name)
            elif hypothesis_families:
                p95 *= 0.6
                penalized.append(metric_name)

            z_scores[metric_name] = min(p95 / 10.0, 1.0)

        if not z_scores:
            return {
                "z_by_metric": {},
                "entity_score": 0.0,
                "attention_boosted_metrics": [],
                "attention_penalized_metrics": [],
            }

        entity_score = float(np.mean(list(z_scores.values())))
        return {
            "z_by_metric": z_scores,
            "entity_score": entity_score,
            "attention_boosted_metrics": boosted,
            "attention_penalized_metrics": penalized,
        }

    def clear(self):
        self._baselines.clear()
