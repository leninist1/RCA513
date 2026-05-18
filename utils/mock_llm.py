"""Mock LLM client for testing Option C without API keys.

Mimics LLM behavior using emotion-vector decision logic,
so the architecture (prompts, JSON parsing, conversation) is tested.
"""

import json
import random


class MockLLMResponse:
    def __init__(self, content, usage=None):
        self.content = content
        self.usage = usage or {"total_tokens": 50, "prompt_tokens": 30, "completion_tokens": 20}


class MockLLMClient:
    """Simulates LLM API with rule-based decisions mirroring emotion vectors."""

    def __init__(self, provider="mock"):
        self.provider = provider
        self.api_key = "mock-key"
        self._iteration = 0
        self._base_url = ""

    def call(self, model="mock", messages=None, system="", response_format=None,
             max_tokens=512, temperature=0.1):
        self._iteration += 1

        # Parse user prompt to extract state
        user_msg = ""
        for m in (messages or []):
            if m.get("role") == "user":
                user_msg = m.get("content", "")
                break

        # Extract key indicators from prompt
        iteration = self._extract_int(user_msg, "Iteration", 1)
        max_iter = self._extract_int(user_msg, "max_iterations", 8)
        calls_used = self._extract_int(user_msg, "calls_used", 0)
        calls_total = self._extract_int(user_msg, "calls_total", 10)
        has_logs = "Log data available: True" in user_msg
        has_traces = "Trace data available: True" in user_msg

        # Count candidates
        num_candidates = user_msg.count("score=")

        # Decision logic mirroring emotion vectors
        budget_ratio = calls_used / max(calls_total, 1)
        anxiety = min(1.0, budget_ratio * 0.6 + max(0, iteration - 3) * 0.05)

        # Stop conditions
        if iteration >= max_iter:
            return MockLLMResponse({
                "strategy": "A", "depth": "shallow", "stop": True,
                "reasoning": "Maximum iterations reached. Stopping with best candidate.",
                "next_candidates": [],
            })

        if budget_ratio > 0.8 and iteration > 2:
            return MockLLMResponse({
                "strategy": "A", "depth": "shallow", "stop": True,
                "reasoning": "Budget nearly depleted, choosing best candidate found so far.",
                "next_candidates": [],
            })

        # Strategy selection (mimics emotion vector rules)
        if anxiety > 0.7:
            strategy = "A"
            reasoning = "High urgency due to budget consumption. Using broad shallow search."
        elif iteration <= 1 and has_traces:
            strategy = "C"
            reasoning = "Early exploration: checking causal propagation paths via trace graph."
        elif iteration <= 2 and has_logs:
            strategy = "D"
            reasoning = "Log data available: analyzing log keywords for evidence."
        elif num_candidates <= 2:
            strategy = "B"
            reasoning = "Few candidates remain: deep verification on top contenders."
        else:
            strategy = "A"
            reasoning = "Defaulting to broad shallow verification across candidates."

        depth = "shallow" if strategy == "A" else ("deep" if strategy == "B" else "medium")

        return MockLLMResponse({
            "strategy": strategy,
            "depth": depth,
            "stop": False,
            "reasoning": reasoning,
            "next_candidates": [],
        })

    @staticmethod
    def _extract_int(text, keyword, default=0):
        """Extract integer value after a keyword from prompt text."""
        import re
        # Match patterns like "Iteration: 3" or "Iteration: 3/"
        match = re.search(rf'{keyword}[:\s]*(\d+)', text)
        if match:
            return int(match.group(1))
        return default
