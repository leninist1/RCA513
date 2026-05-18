"""Base meta-controller interface and ControllerState."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

from ..config import Candidate, StrategyResult, ResourceBudget, ControllerAction


class ControllerState:
    """Shared state that evolves through the controller loop."""

    def __init__(
        self, evidence_pool: Dict[str, List[Dict]],
        candidate_scores: Dict[str, float],
        budget: ResourceBudget,
        case_features: Dict[str, Any],
    ):
        self.evidence_pool = evidence_pool
        self.candidate_scores = candidate_scores
        self.budget = budget
        self.case_features = case_features
        self.iteration = 0
        self.strategy_history: List[str] = []
        self.stop_reason = ""

    def update(self, result: StrategyResult):
        """Update state with strategy execution result."""
        self.iteration += 1
        self.strategy_history.append(result.strategy_name)

        for c in result.candidates_verified:
            if c.recovery_score is not None:
                prev = self.candidate_scores.get(c.entity, 0)
                self.candidate_scores[c.entity] = prev * 0.3 + c.recovery_score * 0.7

        for entity, evidence in result.evidence_added.items():
            self.evidence_pool.setdefault(entity, []).extend(evidence)

        self.budget.api_calls_used += result.cost.api_calls
        self.budget.tokens_used += result.cost.tokens

    def top_candidates(self, n: int = 3) -> List[Candidate]:
        """Return top N candidates by current score."""
        sorted_ents = sorted(self.candidate_scores.items(), key=lambda x: x[1], reverse=True)
        return [
            Candidate(entity=ent, recovery_score=sc, evidence=self.evidence_pool.get(ent, []))
            for ent, sc in sorted_ents[:n]
        ]

    @property
    def top_score(self) -> float:
        sorted_ents = sorted(self.candidate_scores.values(), reverse=True)
        return sorted_ents[0] if sorted_ents else 0.0

    @property
    def top2_gap(self) -> float:
        sorted_ents = sorted(self.candidate_scores.values(), reverse=True)
        if len(sorted_ents) >= 2:
            return sorted_ents[0] - sorted_ents[1]
        return 0.0

    @property
    def explored_ratio(self) -> float:
        verified = sum(1 for ev in self.evidence_pool.values() if ev)
        total = max(1, len(self.candidate_scores))
        return verified / total


class BaseMetaController(ABC):
    @abstractmethod
    def select_action(self, state: ControllerState) -> ControllerAction:
        """Given current state, decide next action."""

    @abstractmethod
    def update_state(self, result: StrategyResult):
        """Update internal state after strategy execution."""
