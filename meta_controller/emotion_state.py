"""Emotion vector definition and update logic for Meta-Controller Option B."""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

from ..config import StrategyResult, ResourceBudget


@dataclass
class EmotionVector:
    anxiety: float = 0.3        # [0,1] — urgency, inversely proportional to remaining budget
    confidence: float = 0.0     # [0,1] — certainty about top candidate
    uncertainty: float = 0.5    # [0,1] — ambiguity in current evidence
    curiosity: float = 0.7      # [0,1] — desire to explore unverified candidates


class EmotionState:
    """Tracks and updates emotion vectors based on verification results."""

    def __init__(self):
        self.emotion = EmotionVector()
        self._score_history: List[float] = []       # top-1 score per iteration
        self._gap_history: List[float] = []         # top1-top2 gap per iteration
        self._iteration = 0

    def update(self, result: StrategyResult, budget: ResourceBudget):
        """Update emotion vector based on latest verification results."""
        self._iteration += 1

        scores = [c.recovery_score or 0 for c in result.candidates_verified]
        top_score = max(scores) if scores else 0
        self._score_history.append(top_score)

        if len(scores) >= 2:
            sorted_scores = sorted(scores, reverse=True)
            gap = sorted_scores[0] - sorted_scores[1]
            self._gap_history.append(gap)
        elif self._gap_history:
            self._gap_history.append(self._gap_history[-1])
        else:
            self._gap_history.append(0.0)

        self._update_anxiety(budget)
        self._update_confidence()
        self._update_uncertainty(result)
        self._update_curiosity(result)

    def _update_anxiety(self, budget: ResourceBudget):
        """Anxiety rises with budget depletion and plateau."""
        budget_ratio = budget.budget_ratio
        plateau_count = self._count_plateau()
        self.emotion.anxiety = min(1.0, budget_ratio * 0.6 + plateau_count * 0.05)

    def _update_confidence(self):
        """Confidence rises when top candidate pulls away from #2 and scores converge."""
        if len(self._gap_history) >= 1:
            gap = self._gap_history[-1]
            # Sigmoid: maps gap [0,1] to [0,1] with steepness around 0.2-0.3
            gap_norm = 1.0 / (1.0 + np.exp(-(gap * 10 - 3)))
        else:
            gap_norm = 0.0

        # Convergence trend: positive correlation of recent scores = converging
        if len(self._score_history) >= 3:
            recent = self._score_history[-3:]
            if max(recent) - min(recent) < 1e-9:
                trend = 0.5
            else:
                x = np.arange(len(recent))
                y = np.array(recent)
                cov = np.cov(x, y)[0, 1] if len(x) >= 2 else 0
                var_x = np.var(x) if len(x) >= 2 else 1
                trend = max(0, cov / (var_x + 1e-9))
                trend = min(1.0, trend * 5)
        else:
            trend = 0.0

        self.emotion.confidence = min(1.0, gap_norm * 0.7 + trend * 0.3)

    def _update_uncertainty(self, result: StrategyResult):
        """Uncertainty rises when top candidates have very close scores."""
        scores = [c.recovery_score or 0 for c in result.candidates_verified]
        if len(scores) >= 2:
            top3 = sorted(scores, reverse=True)[:3]
            var = np.var(top3) if len(top3) >= 2 else 0
            score_uncertainty = 1.0 - min(1.0, var * 5)
        else:
            score_uncertainty = 0.7

        self.emotion.uncertainty = score_uncertainty

    def _update_curiosity(self, result: StrategyResult):
        """Curiosity reflects remaining unexplored candidates."""
        n_verified = sum(1 for c in result.candidates_verified if c.verification_depth != "none")
        n_total = max(1, len(result.candidates_verified))
        curiosity_base = max(0, 1.0 - n_verified / n_total)

        curiosity_boost = self.emotion.uncertainty * 0.3
        anxiety_penalty = self.emotion.anxiety * 0.5
        self.emotion.curiosity = min(1.0, max(0.0, curiosity_base + curiosity_boost - anxiety_penalty))

    def _count_plateau(self) -> int:
        """Count consecutive iterations without meaningfully improved top-1 score."""
        count = 0
        for i in range(len(self._score_history) - 1, 0, -1):
            improvement = self._score_history[i] - self._score_history[i - 1]
            if improvement < 0.01:
                count += 1
            else:
                break
        return count

    def recommend_strategy(self, case_features: dict) -> str:
        """Map emotion vector to strategy selection.

        Decision rules (priority order):
        1. anxiety > 0.7  →  Strategy A (need speed, cut losses)
        2. confidence > 0.7 AND uncertainty < 0.3  →  Strategy B (converge deeply)
        3. curiosity > 0.6 AND anxiety < 0.5  →  Strategy C (explore causally)
        4. uncertainty > 0.6 AND confidence < 0.3  →  Strategy D (resolve via logs)

        System-specific override: if has_logs=False, never select Strategy D.
        """
        e = self.emotion
        has_logs = case_features.get("has_logs", True)

        if e.anxiety > 0.7:
            return "A_broad_shallow"
        if e.confidence > 0.7 and e.uncertainty < 0.3:
            return "B_deep_dive"
        if e.curiosity > 0.6 and e.anxiety < 0.5:
            return "C_causal_trace"
        if e.uncertainty > 0.6 and e.confidence < 0.3 and has_logs:
            return "D_log_compare"
        return "A_broad_shallow"

    def should_stop(self) -> Tuple[bool, str]:
        """Decide whether to stop the controller loop."""
        e = self.emotion
        if e.confidence > 0.85:
            return True, "high_confidence"
        if e.anxiety > 0.9 and e.confidence > 0.4:
            return True, "budget_low_with_answer"
        if e.anxiety > 0.95:
            return True, "budget_exhausted"
        if self._iteration >= 8:
            return True, "max_iterations"
        return False, ""
