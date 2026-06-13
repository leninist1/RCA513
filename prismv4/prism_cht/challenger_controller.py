"""Single-challenge adversarial review controller for PRISM-CHT.

Orchestrates a single-round Challenger review after a Lead nomination.
The Challenger proposes a counter-claim, executes one new
discriminative investigation, and assesses the resulting evidence.
The controller enforces the shared-infrastructure constraint and
one-run-only semantics.
"""

from __future__ import annotations

from typing import Mapping

from .assessment_gate import EvidenceAssessmentGate
from .challenge_gate import ChallengeReviewGate
from .challenge_types import (
    ChallengeAuditStep,
    ChallengeReviewResult,
    ChallengeVerdict,
    build_challenge_snapshot,
)
from .challenger_policy import ChallengerPolicy
from .evidence_graph import EvidenceGraph
from .executor import InvestigationExecutor
from .hypothesis import CausalHypothesis
from .tournament_types import LeadTournamentResult


class ChallengeControllerReuseError(RuntimeError):
    """Raised when ChallengerController.run() is called more than once."""


class ChallengerController:
    """Single-challenge adversarial review controller.

    Runs exactly one challenge investigation.  Shares the same
    ``InvestigationExecutor``, ``EvidenceGraph``, and
    ``EvidenceAssessmentGate`` as the Lead phase.

    The controller does NOT interpret evidence, link evidence to
    hypotheses, compute scores, call an LLM, or implement a FINAL
    output.
    """

    def __init__(
        self,
        *,
        executor: InvestigationExecutor,
        assessment_gate: EvidenceAssessmentGate,
        challenge_gate: ChallengeReviewGate,
        graph: EvidenceGraph,
    ) -> None:
        self._executor = executor
        self._assessment_gate = assessment_gate
        self._challenge_gate = challenge_gate
        self._graph = graph
        self._has_run = False

    def run(
        self,
        *,
        lead_result: LeadTournamentResult,
        hypotheses: Mapping[str, CausalHypothesis],
        policy: ChallengerPolicy,
    ) -> ChallengeReviewResult:
        """Run a single adversarial challenge investigation.

        Returns ``ChallengeReviewResult`` with one of three statuses:
        * ``"survived_challenge"`` — nomination survives challenge
        * ``"returned_to_lead"`` — nomination is refuted, returns to Lead
        * ``"challenge_inconclusive"`` — challenge could not determine outcome
        """
        if self._has_run:
            raise ChallengeControllerReuseError(
                "ChallengerController.run() can only be called once; "
                "create a new instance for each challenge"
            )
        self._has_run = True

        # 2. Validate lead_result status
        if lead_result.status != "challenge_required":
            raise ValueError(
                f"lead_result.status must be 'challenge_required', "
                f"got '{lead_result.status}'"
            )

        # 3. Build initial ChallengeSnapshot
        snapshot = build_challenge_snapshot(
            lead_result=lead_result,
            hypotheses=hypotheses,
            graph=self._graph,
        )

        # 4. Propose challenge
        proposal = policy.propose_challenge(snapshot=snapshot)

        # 5. Validate proposal
        self._challenge_gate.validate_proposal(
            proposal=proposal,
            lead_result=lead_result,
            hypotheses=hypotheses,
            graph=self._graph,
        )

        # 6. Execute investigation (one new action)
        evidence = self._executor.execute(
            action=proposal.action,
            hypotheses=hypotheses,
        )

        # 7. Build fresh snapshot with new evidence
        fresh_snapshot = build_challenge_snapshot(
            lead_result=lead_result,
            hypotheses=hypotheses,
            graph=self._graph,
        )

        # 8. Assess challenge
        resolution = policy.assess_challenge(
            snapshot=fresh_snapshot,
            proposal=proposal,
            evidence=evidence,
        )

        # 9. Apply resolution through gate
        self._challenge_gate.apply_resolution(
            proposal=proposal,
            resolution=resolution,
            evidence=evidence,
            lead_result=lead_result,
            hypotheses=hypotheses,
            graph=self._graph,
            assessment_gate=self._assessment_gate,
        )

        # 10. Build result
        audit_step = ChallengeAuditStep(
            proposal=proposal,
            evidence_id=evidence.evidence_id,
            resolution=resolution,
        )

        verdict_to_status = {
            ChallengeVerdict.NOMINATION_SURVIVED: "survived_challenge",
            ChallengeVerdict.NOMINATION_REFUTED: "returned_to_lead",
            ChallengeVerdict.INCONCLUSIVE: "challenge_inconclusive",
        }

        return ChallengeReviewResult(
            status=verdict_to_status[resolution.verdict],
            nominated_hypothesis_id=lead_result.nominated_hypothesis_id,
            verdict=resolution.verdict,
            challenge_id=proposal.challenge_id,
            evidence_id=evidence.evidence_id,
            audit_step=audit_step,
        )
