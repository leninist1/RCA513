"""Strategy D: Log keyword analysis + time alignment.

Best for config/deployment failures with rich log text.
Keyword tier scoring + log burst time-alignment with metric anomaly onset.
Cost: 0-1 API calls (optional LLM log summarization).
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Dict

from .base import BaseStrategy
from ..config import Candidate, StrategyResult, ResourceBudget, ResourceCost, UnifiedTelemetry, LOG_KEYWORD_TIERS


class StrategyD(BaseStrategy):
    name = "D_log_compare"
    cost_profile = {"api_calls": 1, "compute_seconds": 0.2}

    def __init__(self, llm_client=None):
        self.llm = llm_client

    def execute(
        self, candidates: List[Candidate], telemetry: UnifiedTelemetry,
        baseline_df: pd.DataFrame, fault_df: pd.DataFrame,
        budget: ResourceBudget, engine=None,
    ) -> StrategyResult:
        """Log keyword analysis + time alignment with metric anomalies.

        Algorithm:
        1. For each candidate entity, score log messages by keyword tiers
        2. Detect log burst timing — first error spike per entity
        3. Cross-reference: entity with earliest log burst + highest keyword density
           gets highest log-based confidence boost
        4. Optionally use LLM to summarize top log messages
        """
        result = StrategyResult(strategy_name=self.name)

        if telemetry.logs is None or telemetry.logs.empty:
            return result

        verified = []
        evidence_added = {}

        # Compute log scores per entity
        log_scores = {}
        log_bursts = {}
        for c in candidates:
            entity_logs = telemetry.logs[telemetry.logs["entity"] == c.entity]
            if entity_logs.empty:
                log_scores[c.entity] = 0.0
                log_bursts[c.entity] = None
                continue

            keyword_score = self._keyword_score(entity_logs)
            log_scores[c.entity] = float(keyword_score)

            burst_time = self._detect_log_burst(entity_logs)
            log_bursts[c.entity] = burst_time

        # Normalize log scores
        max_log = max(log_scores.values()) if log_scores else 1.0
        if max_log > 0:
            log_scores = {k: v / max_log for k, v in log_scores.items()}

        # Compute time-aligned scores
        metric_onset_times = self._metric_onset_times(candidates, baseline_df, fault_df)
        time_scores = {}
        for c in candidates:
            if log_bursts[c.entity] is not None and c.entity in metric_onset_times:
                delta = abs(log_bursts[c.entity] - metric_onset_times[c.entity])
                # Entities where log burst and metric onset align closely score higher
                time_scores[c.entity] = float(np.exp(-delta / 60.0))  # decay with 60s tolerance
            else:
                time_scores[c.entity] = 0.0

        # Combined log score
        for c in candidates:
            combined_log = log_scores.get(c.entity, 0) * 0.6 + time_scores.get(c.entity, 0) * 0.4

            evidence = [
                {"type": "log_keyword_score", "score": log_scores.get(c.entity, 0)},
                {"type": "log_burst_time", "time": str(log_bursts.get(c.entity))},
                {"type": "time_alignment_score", "score": time_scores.get(c.entity, 0)},
                {"type": "combined_log_score", "score": combined_log},
            ]

            # Boost recovery score with log evidence
            boosted_score = max(c.recovery_score or 0, combined_log * 0.5)

            log_c = Candidate(
                entity=c.entity,
                anomaly_score=c.anomaly_score,
                recovery_score=boosted_score,
                verification_depth="medium",
                evidence=c.evidence + evidence,
            )
            verified.append(log_c)
            evidence_added[c.entity] = evidence

        # Optional: LLM summarizes top log messages for top entity
        api_calls = 0
        if self.llm and budget.calls_remaining > 0 and verified:
            top_entity = max(verified, key=lambda c: c.recovery_score or 0)
            top_logs = telemetry.logs[telemetry.logs["entity"] == top_entity.entity]
            if not top_logs.empty:
                summary = self._summarize_logs(top_entity.entity, top_logs)
                if summary:
                    evidence_added.setdefault(top_entity.entity, []).append(
                        {"type": "llm_log_summary", "content": summary}
                    )
                    api_calls = 1

        verified.sort(key=lambda c: c.recovery_score or 0, reverse=True)
        result.candidates_verified = verified
        result.evidence_added = evidence_added
        result.cost = ResourceCost(api_calls=api_calls)
        return result

    def _keyword_score(self, entity_logs: pd.DataFrame) -> float:
        """Score log messages by keyword tier weights."""
        score = 0.0
        messages = entity_logs["message"].astype(str)
        for keyword, weight in LOG_KEYWORD_TIERS.items():
            if messages.str.contains(keyword, case=False).any():
                score += weight
        return score

    def _detect_log_burst(self, entity_logs: pd.DataFrame) -> Optional[float]:
        """Find the timestamp of the first log burst (spike in error messages)."""
        if "timestamp" not in entity_logs.columns:
            return None
        error_msgs = entity_logs[
            entity_logs["message"].astype(str).str.contains("ERROR|error|Exception|Fail", case=False, na=False)
        ]
        if error_msgs.empty:
            return None
        return float(error_msgs["timestamp"].min())

    def _metric_onset_times(
        self, candidates: List[Candidate],
        baseline_df: pd.DataFrame, fault_df: pd.DataFrame
    ) -> Dict[str, float]:
        """Find earliest metric anomaly onset time per entity."""
        onset = {}
        for c in candidates:
            entity_b = baseline_df[baseline_df["entity"] == c.entity]
            entity_f = fault_df[fault_df["entity"] == c.entity]
            if entity_b.empty or entity_f.empty:
                continue
            # Find first timestamp where value exceeds baseline median * 2
            for metric_name in entity_f["metric_name"].unique():
                b_vals = entity_b[entity_b["metric_name"] == metric_name]["value"]
                f_data = entity_f[entity_f["metric_name"] == metric_name].sort_values("timestamp")
                if b_vals.empty or f_data.empty:
                    continue
                b_med = np.median(b_vals.dropna())
                if b_med <= 0:
                    continue
                anomaly_rows = f_data[f_data["value"] > b_med * 2]
                if not anomaly_rows.empty:
                    onset[c.entity] = float(anomaly_rows["timestamp"].min())
                    break
        return onset

    def _summarize_logs(self, entity: str, logs: pd.DataFrame) -> Optional[str]:
        """Use LLM to summarize key log messages for an entity."""
        if not self.llm:
            return None
        try:
            error_logs = logs[
                logs["message"].astype(str).str.contains("ERROR|error|Exception|Fail", case=False, na=False)
            ]
            sample = error_logs["message"].head(10).tolist()
            if not sample:
                return None
            prompt = (
                f"Summarize these error logs for service {entity} in 1-2 sentences. "
                f"What failure pattern do they suggest?\n" + "\n".join(f"- {m}" for m in sample)
            )
            resp = self.llm.call(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
            )
            return resp.content if hasattr(resp, 'content') else str(resp)
        except Exception:
            return None
