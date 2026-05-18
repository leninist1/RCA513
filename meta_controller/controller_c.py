"""Meta-Controller Option C: LLM as meta-controller (optimized).

Uses response caching and shorter prompts to minimize API calls.
"""

import hashlib
import json
from typing import Dict, List, Optional, Any

from .base import BaseMetaController, ControllerState
from ..config import ControllerAction, StrategyResult, ResourceBudget
from ..utils.prompts import SYSTEM_PROMPT, build_prompt
from ..utils.llm_client import LLMClient


# Shared cache across all controller instances (per-process)
_RESPONSE_CACHE: Dict[str, dict] = {}
_CACHE_HITS = 0
_CACHE_MISSES = 0


class LLMMetaController(BaseMetaController):
    def __init__(self, llm_client: LLMClient, model: str = "deepseek-chat"):
        self.llm = llm_client
        self.model = model
        self._history: List[Dict] = []

    def select_action(self, state: ControllerState) -> ControllerAction:
        """Build prompt, call LLM (with cache), parse JSON decision."""
        global _CACHE_HITS, _CACHE_MISSES

        prompt = build_prompt(state)

        # State hash for caching (only cache if no history — first iteration)
        if not self._history:
            cache_key = hashlib.md5(prompt.encode()).hexdigest()
            if cache_key in _RESPONSE_CACHE:
                _CACHE_HITS += 1
                return self._to_action(_RESPONSE_CACHE[cache_key])

        messages = list(self._history)
        messages.append({"role": "user", "content": prompt})

        response = self.llm.call(
            model=self.model,
            messages=messages,
            system=SYSTEM_PROMPT,
            response_format={"type": "json_object"},
            max_tokens=256,  # Shorter responses
            temperature=0.1,
        )

        _CACHE_MISSES += 1
        decision = self._parse_response(response)

        # Track config
        state.budget.api_calls_used += 1
        if response and response.usage:
            state.budget.tokens_used += response.usage.get("total_tokens", 0)

        # Store in history and cache
        self._history.append({"role": "user", "content": prompt})
        self._history.append({"role": "assistant", "content": json.dumps(decision) if isinstance(decision, dict) else "{}"})

        if not self._history[:-2]:  # First iteration
            cache_key = hashlib.md5(prompt.encode()).hexdigest()
            _RESPONSE_CACHE[cache_key] = decision

        return self._to_action(decision)

    def update_state(self, result: StrategyResult):
        """Store strategy result for next prompt context."""
        pass  # Results are reflected in state passed to next select_action

    def _parse_response(self, response) -> dict:
        """Parse LLM response into decision dict, with fallback."""
        if response is None:
            return self._fallback_decision("LLM call failed")

        content = response.content

        if isinstance(content, dict):
            decision = content
        elif isinstance(content, str):
            try:
                decision = json.loads(content)
            except json.JSONDecodeError:
                decision = self._fallback_decision("json_parse_failed")
        else:
            decision = self._fallback_decision("unexpected_response_type")

        return decision

    def _to_action(self, decision: dict) -> ControllerAction:
        """Convert LLM decision dict to ControllerAction."""
        strategy_map = {
            "A": "A_broad_shallow",
            "B": "B_deep_dive",
            "C": "C_causal_trace",
            "D": "D_log_compare",
        }
        strategy_name = strategy_map.get(
            decision.get("strategy", "A"), "A_broad_shallow"
        )

        depth = decision.get("depth", "shallow")
        if depth not in ("shallow", "medium", "deep"):
            depth = "shallow"

        stop = decision.get("stop", False)
        stop_reason = decision.get("reasoning", "") if stop else ""

        focus = decision.get("next_candidates", [])

        return ControllerAction(
            strategy=strategy_name,
            depth=depth,
            stop=stop,
            stop_reason=stop_reason,
            focus_entities=focus,
        )

    @staticmethod
    def _fallback_decision(reason: str) -> dict:
        return {
            "strategy": "A",
            "depth": "shallow",
            "stop": False,
            "reasoning": f"Fallback due to {reason}",
            "next_candidates": [],
        }
