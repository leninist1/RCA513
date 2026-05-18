"""Strategy B: Deep dive on top candidates.

Best for concentrated faults (CPU + MEM sudden spikes).
Full multi-modal recovery with cross-validation on top 1-2 candidates.
Cost: 0-1 API calls (optional LLM analysis).
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional

from .base import BaseStrategy
from ..config import Candidate, StrategyResult, ResourceBudget, ResourceCost, UnifiedTelemetry


class StrategyB(BaseStrategy):
    name = "B_deep_dive"
    cost_profile = {"api_calls": 1, "compute_seconds": 0.5}

    def __init__(self, llm_client=None):
        self.llm = llm_client

    def execute(
        self, candidates: List[Candidate], telemetry: UnifiedTelemetry,
        baseline_df: pd.DataFrame, fault_df: pd.DataFrame,
        budget: ResourceBudget, engine=None,
    ) -> StrategyResult:
        """Deep multi-metric verification on top 1-2 candidates.

        Algorithm:
        1. Take top K=min(2, len(candidates)) by current score
        2. For each: full multi-metric recovery + cross-validation on metric subsets
        3. Optionally call LLM to analyze recovery pattern consistency
        """
        result = StrategyResult(strategy_name=self.name)

        if engine is None or not candidates:
            return result

        top_k = min(2, len(candidates))
        focus = sorted(candidates, key=lambda c: c.recovery_score or 0, reverse=True)[:top_k]

        verified = []
        evidence_added = {}

        for c in focus:
            # Deep: full multi-metric recovery
            cf_df = engine.apply_counterfactual(fault_df, baseline_df, c.entity)
            rec = engine.compute_recovery(
                fault_df, cf_df, baseline_df, c.entity,
                entities=telemetry.entities, logs_df=telemetry.logs
            )

            # Cross-validation: compute recovery on metric subsets
            subset_scores = self._cross_validate(c.entity, baseline_df, fault_df, engine, telemetry)

            # Entity degradation breakdown
            deg_breakdown = self._degradation_breakdown(c.entity, fault_df, baseline_df, engine, telemetry)

            evidence = [
                {"type": "deep_recovery", "score": rec},
                {"type": "cross_validation", "scores": subset_scores},
                {"type": "degradation_breakdown", "details": deg_breakdown},
            ]

            # Optional LLM analysis
            if self.llm and budget.calls_remaining > 0:
                llm_analysis = self._llm_analyze(c.entity, rec, deg_breakdown)
                if llm_analysis:
                    evidence.append({"type": "llm_analysis", "content": llm_analysis})

            deep_c = Candidate(
                entity=c.entity,
                anomaly_score=c.anomaly_score,
                recovery_score=rec,
                verification_depth="deep",
                evidence=evidence,
            )
            verified.append(deep_c)
            evidence_added[c.entity] = evidence

        api_calls = 1 if (self.llm and budget.calls_remaining > 0) else 0
        result.candidates_verified = verified
        result.evidence_added = evidence_added
        result.cost = ResourceCost(api_calls=api_calls)
        return result

    def _cross_validate(
        self, entity: str, baseline_df: pd.DataFrame, fault_df: pd.DataFrame,
        engine, telemetry
    ) -> Dict[str, float]:
        """Compute recovery on different metric subsets for consistency check."""
        all_metrics = engine.metric_templates
        if len(all_metrics) <= 1:
            return {"all": 0}

        # Leave-one-out: compute recovery excluding each metric
        scores = {}
        original_templates = list(engine.metric_templates)

        # All metrics
        cf_df = engine.apply_counterfactual(fault_df, baseline_df, entity)
        scores["all"] = engine.compute_recovery(
            fault_df, cf_df, baseline_df, entity,
            entities=telemetry.entities, logs_df=telemetry.logs
        )

        # Leave-one-out per metric
        for template in all_metrics:
            engine._templates = [t for t in original_templates if t["name"] != template["name"]]
            cf_df = engine.apply_counterfactual(fault_df, baseline_df, entity)
            scores[f"without_{template['name']}"] = engine.compute_recovery(
                fault_df, cf_df, baseline_df, entity,
                entities=telemetry.entities, logs_df=telemetry.logs
            )

        engine._templates = original_templates
        return scores

    def _degradation_breakdown(
        self, entity: str, fault_df: pd.DataFrame, baseline_df: pd.DataFrame,
        engine, telemetry
    ) -> Dict:
        """Break down degradation by metric dimension."""
        breakdown = {}
        entity_f = fault_df[fault_df["entity"] == entity]
        entity_b = baseline_df[baseline_df["entity"] == entity]

        for template in engine.metric_templates:
            metric_name = template["name"]
            f_vals = entity_f[entity_f["metric_name"] == metric_name]["value"].dropna()
            b_vals = entity_b[entity_b["metric_name"] == metric_name]["value"].dropna()
            if f_vals.empty or b_vals.empty:
                continue
            b_med = np.median(b_vals)
            f_p95 = np.percentile(f_vals, 95)
            ratio = f_p95 / (b_med + 1e-9)
            breakdown[metric_name] = {
                "baseline_median": float(b_med),
                "fault_p95": float(f_p95),
                "ratio": float(ratio),
            }

        if telemetry.logs is not None and not telemetry.logs.empty:
            log_score = engine._log_semantic_score(telemetry.logs, entity)
            if log_score > 0:
                breakdown["log_score"] = float(log_score)

        return breakdown

    def _llm_analyze(self, entity: str, recovery_score: float, degradation: Dict) -> Optional[str]:
        """Use LLM to analyze recovery pattern consistency."""
        if not self.llm:
            return None
        try:
            prompt = (
                f"Entity {entity} shows recovery_score={recovery_score:.3f} when restored.\n"
                f"Degradation breakdown: {degradation}\n"
                "Is this pattern consistent with being the root cause? Answer in 1 sentence."
            )
            resp = self.llm.call(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
            )
            return resp.content if hasattr(resp, 'content') else str(resp)
        except Exception:
            return None
