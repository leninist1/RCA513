"""Meta-Controller Option B: Emotion-vector-driven strategy selection."""

from typing import Dict

from .base import BaseMetaController, ControllerState
from .emotion_state import EmotionState
from ..config import ControllerAction, StrategyResult


class EmotionMetaController(BaseMetaController):
    def __init__(self):
        self.emotion_state = EmotionState()

    def select_action(self, state: ControllerState) -> ControllerAction:
        strategy_name = self.emotion_state.recommend_strategy(state.case_features)
        should_stop, reason = self.emotion_state.should_stop()
        depth = self._decide_depth(strategy_name)

        return ControllerAction(
            strategy=strategy_name,
            depth=depth,
            stop=should_stop,
            stop_reason=reason,
            focus_entities=[c.entity for c in state.top_candidates(3)],
        )

    def update_state(self, result: StrategyResult):
        pass  # Emotion state is updated separately via emotion_state.update()

    def update_emotion(self, result: StrategyResult, budget):
        self.emotion_state.update(result, budget)

    def _decide_depth(self, strategy_name: str) -> str:
        base_depth = {
            "A_broad_shallow": "shallow",
            "B_deep_dive": "deep",
            "C_causal_trace": "medium",
            "D_log_compare": "medium",
        }
        depth = base_depth.get(strategy_name, "shallow")
        if self.emotion_state.emotion.anxiety > 0.6:
            if depth == "deep":
                depth = "medium"
            elif depth == "medium":
                depth = "shallow"
        return depth
