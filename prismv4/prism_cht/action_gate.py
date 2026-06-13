"""Action gate for PRISM-CHT: rejects invalid or duplicate actions.

The gate enforces that every action:
- targets at least two active, distinguishable hypotheses;
- uses an allowlisted tool;
- has meaningful expected-outcome differentiation;
- has not already been executed (by query_signature).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Mapping, Set, Tuple

from .action_schema import DiscriminativeAction
from .hypothesis import CausalHypothesis, HypothesisStatus

_ALLOWED_TOOLS: Set[str] = {
    "compare_onset_order",
    "inspect_trace_path",
    "check_propagation_consistency",
    "inspect_reason_signature",
    "find_unexplained_symptoms",
    "retrieve_raw_evidence",
}

# Hypotheses that are NOT eligible for investigation actions.
# DRAFT = not yet activated
# REFUTED, FINAL = terminal
_REJECTED_FOR_ACTION: Set[HypothesisStatus] = {
    HypothesisStatus.DRAFT,
    HypothesisStatus.REFUTED,
    HypothesisStatus.FINAL,
}

_WS_RE = re.compile(r"\s+")


def _normalize_outcome(text: str) -> str:
    return _WS_RE.sub(" ", text.strip().lower())


@dataclass(frozen=True)
class GateDecision:
    accepted: bool
    query_signature: str
    reasons: Tuple[str, ...]


class ActionGate:
    """Gate that evaluates and admits discriminative actions.

    This gate only judges whether an action is worth executing.
    It does NOT judge which hypothesis is correct.
    """

    def __init__(self) -> None:
        self._executed_signatures: Set[str] = set()

    def _check_common(
        self,
        action: DiscriminativeAction,
        hypotheses: Mapping[str, CausalHypothesis],
    ) -> Tuple[str, ...]:
        reasons: list = []

        # 1. action_type
        if action.action_type != "run_discriminative_test":
            reasons.append(
                f"action_type '{action.action_type}' is not 'run_discriminative_test'"
            )
            return tuple(reasons)

        # 2. tool_name allowlist
        if action.tool_name not in _ALLOWED_TOOLS:
            reasons.append(
                f"tool_name '{action.tool_name}' is not in allowlist"
            )
            return tuple(reasons)

        # 3. target_hypothesis_ids count
        if len(action.target_hypothesis_ids) < 2:
            reasons.append(
                "target_hypothesis_ids must contain at least two hypotheses"
            )
            return tuple(reasons)

        # 4. duplicates
        if len(set(action.target_hypothesis_ids)) != len(
            action.target_hypothesis_ids
        ):
            reasons.append("target_hypothesis_ids contains duplicate entries")
            return tuple(reasons)

        # 5. references non-existent hypotheses
        for hid in action.target_hypothesis_ids:
            if hid not in hypotheses:
                reasons.append(f"hypothesis '{hid}' does not exist")
                return tuple(reasons)

        # 6. references ineligible statuses (DRAFT, REFUTED, FINAL)
        for hid in action.target_hypothesis_ids:
            h = hypotheses[hid]
            if h.status in _REJECTED_FOR_ACTION:
                reasons.append(
                    f"hypothesis '{hid}' has status '{h.status.value}' "
                    f"which is not eligible for investigation"
                )
                return tuple(reasons)

        # 7. expected_outcomes missing coverage
        for hid in action.target_hypothesis_ids:
            if hid not in action.expected_outcomes:
                reasons.append(
                    f"expected_outcomes missing coverage for hypothesis '{hid}'"
                )
                return tuple(reasons)

        # 8. empty expected_outcomes values
        for hid, text in action.expected_outcomes.items():
            if not text or not text.strip():
                reasons.append(
                    f"expected_outcomes for '{hid}' is empty"
                )
                return tuple(reasons)

        # 9. all expected_outcomes normalize to the same string
        normalized = {
            hid: _normalize_outcome(text)
            for hid, text in action.expected_outcomes.items()
        }
        if len(set(normalized.values())) <= 1:
            reasons.append(
                "all expected_outcomes are identical after normalization; "
                "action has no discriminative power"
            )
            return tuple(reasons)

        # 10. empty question
        if not action.question or not action.question.strip():
            reasons.append("question is empty")
            return tuple(reasons)

        # 11. empty why_discriminative
        if not action.why_discriminative or not action.why_discriminative.strip():
            reasons.append("why_discriminative is empty")
            return tuple(reasons)

        # 12. args is not a Mapping (already enforced in __post_init__,
        #     but double-check)
        if not isinstance(action.args, Mapping):
            reasons.append(
                f"args must be a Mapping, got {type(action.args).__name__}"
            )
            return tuple(reasons)

        return tuple(reasons)

    def evaluate(
        self,
        action: DiscriminativeAction,
        hypotheses: Mapping[str, CausalHypothesis],
    ) -> GateDecision:
        """Check action validity without modifying internal state."""
        reasons = self._check_common(action, hypotheses)
        sig = action.query_signature()
        accepted = len(reasons) == 0
        return GateDecision(
            accepted=accepted,
            query_signature=sig,
            reasons=reasons,
        )

    def admit(
        self,
        action: DiscriminativeAction,
        hypotheses: Mapping[str, CausalHypothesis],
    ) -> GateDecision:
        """Check action validity AND record signature as executed.

        The same query_signature admitted twice will be rejected.
        """
        reasons_list = list(self._check_common(action, hypotheses))
        sig = action.query_signature()

        # 13. signature already executed
        if not reasons_list and sig in self._executed_signatures:
            reasons_list.append(
                f"query_signature '{sig[:16]}...' has already been executed"
            )

        reasons = tuple(reasons_list)
        accepted = len(reasons) == 0
        if accepted:
            self._executed_signatures.add(sig)

        return GateDecision(
            accepted=accepted,
            query_signature=sig,
            reasons=reasons,
        )
