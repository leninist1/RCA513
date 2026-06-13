"""Lead policy protocol and scripted implementation for PRISM-CHT.

The ``LeadPolicy`` protocol defines the interface that any Lead Agent
must satisfy.  ``ScriptedLeadPolicy`` provides a deterministic,
recorded-playback implementation used to test the tournament
controller without an LLM.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Union

from .action_schema import DiscriminativeAction
from .evidence_graph import EvidenceAtom
from .tournament_types import (
    EvidenceAssessment,
    LeadNomination,
    LeadTournamentSnapshot,
)

# PolicyDecision = DiscriminativeAction | LeadNomination
PolicyDecision = Union[DiscriminativeAction, LeadNomination]


class LeadPolicy(Protocol):
    """Protocol for a Lead Agent in the causal hypothesis tournament.

    The Policy interprets the current tournament snapshot and either
    proposes a new discriminative action or nominates a final
    candidate hypothesis for challenge.
    """

    def decide_next(
        self,
        *,
        snapshot: LeadTournamentSnapshot,
    ) -> PolicyDecision:
        ...

    def assess_evidence(
        self,
        *,
        snapshot: LeadTournamentSnapshot,
        action: DiscriminativeAction,
        evidence: EvidenceAtom,
    ) -> EvidenceAssessment:
        ...


@dataclass(frozen=True)
class ScriptedInvestigationTurn:
    """A pre-recorded action + assessment pair for one investigation round."""

    action: DiscriminativeAction
    assessment: EvidenceAssessment


class ScriptedLeadPolicy:
    """Deterministic, pre-configured Lead Policy for testing.

    Replays a sequence of actions and assessments.  After all turns
    have been consumed, returns the configured nomination.

    This policy does not read ground truth labels, compute scores, or
    inspect test answers.  It throws ``ValueError`` on out-of-order calls.
    """

    def __init__(
        self,
        *,
        turns: list[ScriptedInvestigationTurn],
        nomination: LeadNomination,
    ) -> None:
        self._turns = tuple(turns)
        self._nomination = nomination
        self._current_index = 0
        self._last_action: DiscriminativeAction | None = None
        self._nomination_returned = False

    def decide_next(
        self,
        *,
        snapshot: LeadTournamentSnapshot,
    ) -> PolicyDecision:
        if self._nomination_returned:
            raise ValueError(
                "ScriptedLeadPolicy: decide_next called after nomination "
                "already returned"
            )
        if self._current_index < len(self._turns):
            turn = self._turns[self._current_index]
            self._last_action = turn.action
            return turn.action
        # All turns consumed, return nomination
        self._nomination_returned = True
        return self._nomination

    def assess_evidence(
        self,
        *,
        snapshot: LeadTournamentSnapshot,
        action: DiscriminativeAction,
        evidence: EvidenceAtom,
    ) -> EvidenceAssessment:
        if self._current_index >= len(self._turns):
            raise ValueError(
                "ScriptedLeadPolicy: assess_evidence called but all turns "
                "have been consumed"
            )
        expected_turn = self._turns[self._current_index]
        if action.action_id != expected_turn.action.action_id:
            raise ValueError(
                f"ScriptedLeadPolicy: action mismatch at turn "
                f"{self._current_index}: expected "
                f"'{expected_turn.action.action_id}', "
                f"got '{action.action_id}'"
            )
        turn = self._turns[self._current_index]
        self._current_index += 1
        return turn.assessment
