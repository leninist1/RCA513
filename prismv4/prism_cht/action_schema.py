"""Discriminative action protocol for PRISM-CHT.

An action must distinguish at least two competing hypotheses.
Its query signature depends only on tool_name and canonical args,
so re-phrasing a question does not permit duplicate execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Tuple

from .canonical import build_tool_call_signature, deep_freeze


@dataclass(frozen=True)
class DiscriminativeAction:
    action_id: str
    action_type: str
    target_hypothesis_ids: Tuple[str, ...]
    question: str
    tool_name: str
    args: Mapping[str, Any]
    expected_outcomes: Mapping[str, str]
    why_discriminative: str

    def __post_init__(self):
        if self.action_type != "run_discriminative_test":
            raise ValueError(
                f"action_type must be 'run_discriminative_test', "
                f"got '{self.action_type}'"
            )

        if len(self.target_hypothesis_ids) < 2:
            raise ValueError(
                "target_hypothesis_ids must contain at least two "
                "different hypotheses"
            )

        if len(set(self.target_hypothesis_ids)) != len(self.target_hypothesis_ids):
            raise ValueError("target_hypothesis_ids contains duplicate entries")

        for hid in self.target_hypothesis_ids:
            if hid not in self.expected_outcomes:
                raise ValueError(
                    f"expected_outcomes must cover hypothesis '{hid}'"
                )

        for hid, outcome_text in self.expected_outcomes.items():
            if not outcome_text or not outcome_text.strip():
                raise ValueError(
                    f"expected_outcomes for '{hid}' must be non-empty"
                )

        if not self.question or not self.question.strip():
            raise ValueError("question must be non-empty")

        if not self.why_discriminative or not self.why_discriminative.strip():
            raise ValueError("why_discriminative must be non-empty")

        if not isinstance(self.args, Mapping):
            raise ValueError(f"args must be a Mapping, got {type(self.args).__name__}")

        # Deep-freeze nested containers so external mutations cannot affect
        # this frozen action.  Use object.__setattr__ to bypass frozen=True.
        object.__setattr__(self, "args", deep_freeze(self.args))
        object.__setattr__(self, "expected_outcomes", deep_freeze(self.expected_outcomes))

    def query_signature(self) -> str:
        """Canonical signature from tool_name + canonicalized args only.

        Delegates to the single shared ``build_tool_call_signature``.
        Does NOT include action_id, question, why_discriminative, or
        expected_outcomes wording.
        """
        return build_tool_call_signature(
            tool_name=self.tool_name,
            args=self.args,
        )
