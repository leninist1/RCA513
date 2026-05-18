"""Abstract base strategy and shared types."""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import pandas as pd

from ..config import Candidate, StrategyResult, ResourceBudget, ResourceCost, UnifiedTelemetry


class BaseStrategy(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def cost_profile(self) -> Dict[str, float]: ...

    @abstractmethod
    def execute(
        self, candidates: List[Candidate], telemetry: UnifiedTelemetry,
        baseline_df: pd.DataFrame, fault_df: pd.DataFrame,
        budget: ResourceBudget, engine=None,
    ) -> StrategyResult: ...
