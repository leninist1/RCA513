"""Challenger policy protocol and scripted implementation for PRISM-CHT.

The ``ChallengerPolicy`` protocol defines the interface that any
Challenger Agent must satisfy.  ``ScriptedChallengerPolicy`` provides
a deterministic, recorded-playback implementation used to test the
challenge controller without an LLM.
"""

from __future__ import annotations

from typing import Protocol

from .challenge_types import (
    ChallengeProposal,
    ChallengeResolution,
    ChallengeSnapshot,
)
from .evidence_graph import EvidenceAtom


class ChallengerPolicy(Protocol):
    """Protocol for a Challenger in the adversarial review.

    The policy receives a snapshot and either proposes a challenge
    or assesses the resulting evidence.
    """

    def propose_challenge(
        self,
        *,
        snapshot: ChallengeSnapshot,
    ) -> ChallengeProposal:
        ...

    def assess_challenge(
        self,
        *,
        snapshot: ChallengeSnapshot,
        proposal: ChallengeProposal,
        evidence: EvidenceAtom,
    ) -> ChallengeResolution:
        ...


class ScriptedChallengerPolicy:
    """Deterministic, pre-configured Challenger Policy for testing.

    Returns a single pre-configured proposal and resolution.
    Enforces correct call order and prevents reuse.

    This policy does not read ground truth labels, compute scores, or
    inspect test answers.  It throws ``ValueError`` on out-of-order calls
    or repeated calls.
    """

    def __init__(
        self,
        proposal: ChallengeProposal,
        resolution: ChallengeResolution,
    ) -> None:
        self._proposal = proposal
        self._resolution = resolution
        self._proposal_returned = False
        self._proposal_called = False
        self._assessment_called = False

    def propose_challenge(
        self,
        *,
        snapshot: ChallengeSnapshot,
    ) -> ChallengeProposal:
        if self._proposal_called:
            raise ValueError(
                "ScriptedChallengerPolicy: propose_challenge() called more than once"
            )
        if self._assessment_called:
            raise ValueError(
                "ScriptedChallengerPolicy: propose_challenge() called after "
                "assess_challenge()"
            )
        self._proposal_called = True
        self._proposal_returned = True
        return self._proposal

    def assess_challenge(
        self,
        *,
        snapshot: ChallengeSnapshot,
        proposal: ChallengeProposal,
        evidence: EvidenceAtom,
    ) -> ChallengeResolution:
        if not self._proposal_returned:
            raise ValueError(
                "ScriptedChallengerPolicy: assess_challenge() called before "
                "propose_challenge()"
            )
        if self._assessment_called:
            raise ValueError(
                "ScriptedChallengerPolicy: assess_challenge() called more than once"
            )
        self._assessment_called = True
        return self._resolution
