"""Strategy A: Broad search, shallow verification.

Best for diffuse signals (LOSS + DELAY types).
Multi-candidate parallel verification, each with lightweight recovery check.
Cost: 0 API calls (pure compute).
"""

import numpy as np
import pandas as pd
from typing import List

from .base import BaseStrategy
from ..config import Candidate, StrategyResult, ResourceBudget, ResourceCost, UnifiedTelemetry


class StrategyA(BaseStrategy):
    name = "A_broad_shallow"
    cost_profile = {"api_calls": 0, "compute_seconds": 0.1}

    def execute(
        self, candidates: List[Candidate], telemetry: UnifiedTelemetry,
        baseline_df: pd.DataFrame, fault_df: pd.DataFrame,
        budget: ResourceBudget, engine=None,
    ) -> StrategyResult:
        """Light recovery score on all candidates using only most-sensitive metric.

        Algorithm:
        1. For each candidate, run single-metric recovery (fastest metric dimension)
        2. Compute quick recovery score
        3. Rank and return top candidates
        """
        result = StrategyResult(strategy_name=self.name)

        if engine is None or not candidates:
            return result

        verified = []
        evidence_added = {}

        for c in candidates:
            # Shallow: use apply_counterfactual with single-entity restore
            cf_df = engine.apply_counterfactual(fault_df, baseline_df, c.entity)
            rec = engine.compute_recovery(
                fault_df, cf_df, baseline_df, c.entity,
                entities=telemetry.entities, logs_df=telemetry.logs
            )
            shallow_c = Candidate(
                entity=c.entity,
                anomaly_score=c.anomaly_score,
                recovery_score=rec,
                verification_depth="shallow",
                evidence=[{"type": "shallow_recovery", "score": rec}],
            )
            verified.append(shallow_c)
            evidence_added[c.entity] = [{"type": "shallow_recovery", "score": rec}]

        verified.sort(key=lambda c: c.recovery_score or 0, reverse=True)
        result.candidates_verified = verified
        result.evidence_added = evidence_added
        result.cost = ResourceCost(api_calls=0)
        return result
