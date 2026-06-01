"""Hypothesis cross-validation (Active Perception D4).

Runs lightweight comparative analysis of two competing hypotheses without
executing expensive counterfactuals.  For each candidate entity:
  - Re-analyzes metrics with hypothesis-driven attention
  - Compares log patterns for the two hypotheses
  - Produces relative likelihood ratio

Cost: ~3s (no counterfactual, no LLM)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np


class HypothesisCrossval:
    """Compare two competing root cause hypotheses with light evidence."""

    def __init__(self):
        pass

    def compare(
        self,
        entity_h1: str,
        entity_h2: str,
        h1_hypothesis: str,  # e.g., "CPU fault"
        h2_hypothesis: str,
        h1_metric_scores: Dict[str, float],
        h2_metric_scores: Dict[str, float],
        h1_log_evidence: float,
        h2_log_evidence: float,
        h1_belief: float,
        h2_belief: float,
    ) -> Dict:
        """Compare two hypotheses and return relative likelihoods.

        Returns:
            {
                "h1_relative_likelihood": float (0-1),
                "h2_relative_likelihood": float (0-1),
                "discriminative": bool,
                "winner": str,
                "confidence": float,
                "metric_ratio": float,
                "log_ratio": float,
                "prior_ratio": float,
            }
        """
        h1_metric_avg = (
            float(np.mean(list(h1_metric_scores.values()))) if h1_metric_scores else 0.0
        )
        h2_metric_avg = (
            float(np.mean(list(h2_metric_scores.values()))) if h2_metric_scores else 0.0
        )

        metric_ratio = h1_metric_avg / max(h1_metric_avg + h2_metric_avg, 1e-9)
        log_ratio = h1_log_evidence / max(h1_log_evidence + h2_log_evidence, 1e-9)
        prior_ratio = h1_belief / max(h1_belief + h2_belief, 1e-9)

        combined = 0.45 * metric_ratio + 0.30 * log_ratio + 0.25 * prior_ratio
        discriminative = abs(combined - 0.5) > 0.2
        winner = entity_h1 if combined > 0.5 else entity_h2

        return {
            "h1_relative_likelihood": round(combined, 4),
            "h2_relative_likelihood": round(1.0 - combined, 4),
            "discriminative": discriminative,
            "winner": winner,
            "confidence": round(abs(combined - 0.5) * 2.0, 4),
            "metric_ratio": round(metric_ratio, 4),
            "log_ratio": round(log_ratio, 4),
            "prior_ratio": round(prior_ratio, 4),
        }
