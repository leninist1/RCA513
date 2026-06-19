"""Challenge review gate for PRISM-CHT Phase 3.

Validates that a Challenger's structured proposal and resolution
meet adversarial review constraints before evidence assessment is
mutated.  The gate does not compute scores or judge semantic
correctness.
"""

from __future__ import annotations

from typing import Mapping

from .action_schema import DiscriminativeAction
from .assessment_gate import EvidenceAssessmentGate
from .challenge_types import (
    ChallengeProposal,
    ChallengeResolution,
    ChallengeVerdict,
)
from .evidence_graph import EvidenceAtom, EvidenceGraph
from .hypothesis import CausalHypothesis, HypothesisStatus
from .tournament_types import (
    AssessmentOutcome,
    EvidenceRelation,
    LeadTournamentResult,
)


class ChallengeRejectedError(ValueError):
    """Raised when a challenge proposal or resolution fails gate validation."""


class ChallengeReviewGate:
    """Gate that validates and applies challenge proposals and resolutions.

    ``validate_proposal()`` and ``validate_resolution()`` only check —
    no state is mutated.

    ``apply_resolution()`` runs full validation and delegates to
    ``EvidenceAssessmentGate.apply()``, which handles atomic writes.
    """

    def validate_proposal(
        self,
        *,
        proposal: ChallengeProposal,
        lead_result: LeadTournamentResult,
        hypotheses: Mapping[str, CausalHypothesis],
        graph: EvidenceGraph,
    ) -> None:
        """Validate *proposal* without mutating state.

        Raises ChallengeRejectedError on the first violation.
        """

        # 1. lead_result.status must be challenge_required
        if lead_result.status != "challenge_required":
            raise ChallengeRejectedError(
                f"lead_result.status must be 'challenge_required', "
                f"got '{lead_result.status}'"
            )

        # 2. proposal nominee must match lead nominee
        if proposal.nominated_hypothesis_id != lead_result.nominated_hypothesis_id:
            raise ChallengeRejectedError(
                f"proposal.nominated_hypothesis_id "
                f"'{proposal.nominated_hypothesis_id}' does not match "
                f"lead_result.nominated_hypothesis_id "
                f"'{lead_result.nominated_hypothesis_id}'"
            )

        hid = proposal.nominated_hypothesis_id

        # 3. nominated hypothesis must exist (in graph — graph.hypotheses_by_id
        #    tracks graph-registered hypotheses; we also check in hypotheses dict)
        if hid not in graph.hypotheses_by_id:
            raise ChallengeRejectedError(
                f"Nominated hypothesis '{hid}' is not registered in graph"
            )
        nominee = graph.hypotheses_by_id[hid]

        # 4. nominated hypothesis must be SUPPORTED
        if nominee.status != HypothesisStatus.SUPPORTED:
            raise ChallengeRejectedError(
                f"Nominated hypothesis '{hid}' has status "
                f"'{nominee.status.value}', must be SUPPORTED"
            )

        # 5. competitor_hypothesis_ids must not be empty
        if not proposal.competitor_hypothesis_ids:
            raise ChallengeRejectedError(
                "competitor_hypothesis_ids must not be empty"
            )

        # 6. competitor_hypothesis_ids must not have duplicates
        if len(set(proposal.competitor_hypothesis_ids)) != len(
            proposal.competitor_hypothesis_ids
        ):
            raise ChallengeRejectedError(
                "competitor_hypothesis_ids contains duplicate entries"
            )

        # validate each competitor
        for cid in proposal.competitor_hypothesis_ids:
            # 7. competitor must exist
            if cid not in graph.hypotheses_by_id:
                raise ChallengeRejectedError(
                    f"Competitor '{cid}' is not registered in graph"
                )

            # 8. competitor must not equal nominated
            if cid == hid:
                raise ChallengeRejectedError(
                    f"Competitor '{cid}' cannot equal nominated hypothesis"
                )

            comp = graph.hypotheses_by_id[cid]

            # 9. competitor must be in lead_result.nomination.addressed_competitor_ids
            if cid not in lead_result.nomination.addressed_competitor_ids:
                raise ChallengeRejectedError(
                    f"Competitor '{cid}' is not in "
                    f"lead_result.nomination.addressed_competitor_ids"
                )

            # 10. competitor status must be WEAKENED
            if comp.status != HypothesisStatus.WEAKENED:
                raise ChallengeRejectedError(
                    f"Competitor '{cid}' has status '{comp.status.value}', "
                    f"must be WEAKENED"
                )

        # 11. action target set must equal {nominated} U competitors
        expected_targets = {hid} | set(proposal.competitor_hypothesis_ids)
        action_targets = set(proposal.action.target_hypothesis_ids)
        if action_targets != expected_targets:
            raise ChallengeRejectedError(
                f"action.target_hypothesis_ids "
                f"{sorted(action_targets)} do not match expected "
                f"{sorted(expected_targets)} "
                f"(nominated + competitors)"
            )

        # 12. challenge_claim must not be empty
        if not proposal.challenge_claim or not proposal.challenge_claim.strip():
            raise ChallengeRejectedError("challenge_claim must not be empty")

        # 13. falsification_target must not be empty
        if not proposal.falsification_target or not proposal.falsification_target.strip():
            raise ChallengeRejectedError("falsification_target must not be empty")

        # 14. rationale must not be empty
        if not proposal.rationale or not proposal.rationale.strip():
            raise ChallengeRejectedError("rationale must not be empty")

    def validate_resolution(
        self,
        *,
        proposal: ChallengeProposal,
        resolution: ChallengeResolution,
        evidence: EvidenceAtom,
        lead_result: LeadTournamentResult,
        hypotheses: Mapping[str, CausalHypothesis],
        graph: EvidenceGraph,
    ) -> None:
        """Validate *resolution* without mutating state.

        Raises ChallengeRejectedError on the first violation.
        """
        hid = proposal.nominated_hypothesis_id
        assessment = resolution.assessment
        action = proposal.action

        # --- General checks ---

        # 1. resolution challenge_id must match proposal
        if resolution.challenge_id != proposal.challenge_id:
            raise ChallengeRejectedError(
                f"resolution.challenge_id '{resolution.challenge_id}' "
                f"does not match proposal.challenge_id '{proposal.challenge_id}'"
            )

        # 2. resolution action_id must match proposal action_id
        if resolution.action_id != action.action_id:
            raise ChallengeRejectedError(
                f"resolution.action_id '{resolution.action_id}' "
                f"does not match proposal.action.action_id '{action.action_id}'"
            )

        # 3. resolution evidence_id must match evidence
        if resolution.evidence_id != evidence.evidence_id:
            raise ChallengeRejectedError(
                f"resolution.evidence_id '{resolution.evidence_id}' "
                f"does not match evidence.evidence_id '{evidence.evidence_id}'"
            )

        # 4. assessment action_id must match proposal action_id
        if assessment.action_id != action.action_id:
            raise ChallengeRejectedError(
                f"assessment.action_id '{assessment.action_id}' "
                f"does not match proposal.action.action_id '{action.action_id}'"
            )

        # 5. assessment evidence_id must match evidence
        if assessment.evidence_id != evidence.evidence_id:
            raise ChallengeRejectedError(
                f"assessment.evidence_id '{assessment.evidence_id}' "
                f"does not match evidence.evidence_id '{evidence.evidence_id}'"
            )

        # 6. evidence must exist in graph
        if evidence.evidence_id not in graph.evidence_by_id:
            raise ChallengeRejectedError(
                f"Evidence '{evidence.evidence_id}' is not in the graph"
            )

        # 7. resolution rationale must not be empty
        if not resolution.rationale or not resolution.rationale.strip():
            raise ChallengeRejectedError("resolution.rationale must not be empty")

        # 8. assessment must not contain FINAL status update
        for u in assessment.status_updates:
            if u.new_status == HypothesisStatus.FINAL:
                raise ChallengeRejectedError(
                    f"Assessment contains FINAL status update for "
                    f"'{u.hypothesis_id}'; FINAL is not allowed in challenge"
                )

        # 9. assessment link or status update must target only action targets
        action_targets = set(action.target_hypothesis_ids)
        for link in assessment.links:
            if link.hypothesis_id not in action_targets:
                raise ChallengeRejectedError(
                    f"Link targets hypothesis '{link.hypothesis_id}' "
                    f"outside action.target_hypothesis_ids"
                )
        for u in assessment.status_updates:
            if u.hypothesis_id not in action_targets:
                raise ChallengeRejectedError(
                    f"Status update targets hypothesis '{u.hypothesis_id}' "
                    f"outside action.target_hypothesis_ids"
                )

        # --- Verdict-specific checks ---

        if resolution.verdict == ChallengeVerdict.NOMINATION_SURVIVED:
            self._validate_survived(assessment, hid, proposal.competitor_hypothesis_ids)

        elif resolution.verdict == ChallengeVerdict.NOMINATION_REFUTED:
            self._validate_refuted(assessment, hid)

        elif resolution.verdict == ChallengeVerdict.INCONCLUSIVE:
            self._validate_inconclusive(assessment, hid)

    def _validate_survived(
        self,
        assessment: "EvidenceAssessment",
        hid: str,
        competitor_ids: tuple[str, ...],
    ) -> None:
        """Validate NOMINATION_SURVIVED verdict constraints."""

        # 1. outcome must be INFORMATIVE
        if assessment.outcome != AssessmentOutcome.INFORMATIVE:
            raise ChallengeRejectedError(
                f"NOMINATION_SURVIVED requires outcome=INFORMATIVE, "
                f"got '{assessment.outcome.value}'"
            )

        # 2. at least one grounded link: SUPPORTS nominated OR CONTRADICTS competitor
        has_supports_nominated = any(
            link.hypothesis_id == hid and link.relation == EvidenceRelation.SUPPORTS
            for link in assessment.links
        )
        has_contradicts_competitor = any(
            link.hypothesis_id in competitor_ids
            and link.relation == EvidenceRelation.CONTRADICTS
            for link in assessment.links
        )
        if not has_supports_nominated and not has_contradicts_competitor:
            raise ChallengeRejectedError(
                "NOMINATION_SURVIVED requires at least one grounded link: "
                "SUPPORTS nominated hypothesis OR CONTRADICTS a competitor"
            )

        # 3. no CONTRADICTS nominated hypothesis
        for link in assessment.links:
            if link.hypothesis_id == hid and link.relation == EvidenceRelation.CONTRADICTS:
                raise ChallengeRejectedError(
                    "NOMINATION_SURVIVED must not contain CONTRADICTS "
                    "for the nominated hypothesis"
                )

        # 4. exactly one nominated -> SURVIVED status update
        nominee_updates = [
            u for u in assessment.status_updates if u.hypothesis_id == hid
        ]
        if len(nominee_updates) != 1:
            raise ChallengeRejectedError(
                f"NOMINATION_SURVIVED requires exactly one status update "
                f"for nominated hypothesis '{hid}', got {len(nominee_updates)}"
            )
        if nominee_updates[0].new_status != HypothesisStatus.SURVIVED:
            raise ChallengeRejectedError(
                f"NOMINATION_SURVIVED requires nominated hypothesis "
                f"-> SURVIVED, got -> "
                f"'{nominee_updates[0].new_status.value}'"
            )

        # 5. no FINAL for any hypothesis
        for u in assessment.status_updates:
            if u.new_status == HypothesisStatus.FINAL:
                raise ChallengeRejectedError(
                    f"NOMINATION_SURVIVED must not contain FINAL "
                    f"status update for '{u.hypothesis_id}'"
                )

        # 6. no WEAKENED or REFUTED for nominated
        for u in assessment.status_updates:
            if u.hypothesis_id == hid and u.new_status in (
                HypothesisStatus.WEAKENED,
                HypothesisStatus.REFUTED,
            ):
                raise ChallengeRejectedError(
                    f"NOMINATION_SURVIVED must not update nominated "
                    f"hypothesis to '{u.new_status.value}'"
                )

    def _validate_refuted(
        self,
        assessment: "EvidenceAssessment",
        hid: str,
    ) -> None:
        """Validate NOMINATION_REFUTED verdict constraints."""

        # 1. outcome must be INFORMATIVE
        if assessment.outcome != AssessmentOutcome.INFORMATIVE:
            raise ChallengeRejectedError(
                f"NOMINATION_REFUTED requires outcome=INFORMATIVE, "
                f"got '{assessment.outcome.value}'"
            )

        # 2. at least one CONTRADICTS nominated hypothesis
        has_contradicts_nominated = any(
            link.hypothesis_id == hid
            and link.relation == EvidenceRelation.CONTRADICTS
            for link in assessment.links
        )
        if not has_contradicts_nominated:
            raise ChallengeRejectedError(
                "NOMINATION_REFUTED requires at least one "
                "CONTRADICTS nominated hypothesis link"
            )

        # 3. exactly one nominated status update
        nominee_updates = [
            u for u in assessment.status_updates if u.hypothesis_id == hid
        ]
        if len(nominee_updates) != 1:
            raise ChallengeRejectedError(
                f"NOMINATION_REFUTED requires exactly one status update "
                f"for nominated hypothesis '{hid}', got {len(nominee_updates)}"
            )

        # 4. nominated -> WEAKENED or REFUTED
        new_status = nominee_updates[0].new_status
        if new_status not in (HypothesisStatus.WEAKENED, HypothesisStatus.REFUTED):
            raise ChallengeRejectedError(
                f"NOMINATION_REFUTED requires nominated hypothesis "
                f"-> WEAKENED or REFUTED, got -> '{new_status.value}'"
            )

        # 5. no SURVIVED for nominated
        for u in assessment.status_updates:
            if u.hypothesis_id == hid and u.new_status == HypothesisStatus.SURVIVED:
                raise ChallengeRejectedError(
                    "NOMINATION_REFUTED must not update nominated "
                    "hypothesis to SURVIVED"
                )

        # 6. no FINAL
        for u in assessment.status_updates:
            if u.new_status == HypothesisStatus.FINAL:
                raise ChallengeRejectedError(
                    f"NOMINATION_REFUTED must not contain FINAL "
                    f"status update for '{u.hypothesis_id}'"
                )

    def _validate_inconclusive(
        self,
        assessment: "EvidenceAssessment",
        hid: str,
    ) -> None:
        """Validate INCONCLUSIVE verdict constraints."""

        # 1. outcome must be INCONCLUSIVE
        if assessment.outcome != AssessmentOutcome.INCONCLUSIVE:
            raise ChallengeRejectedError(
                f"INCONCLUSIVE requires outcome=INCONCLUSIVE, "
                f"got '{assessment.outcome.value}'"
            )

        # 2. links must be empty
        if assessment.links:
            raise ChallengeRejectedError(
                "INCONCLUSIVE verdict must have no links"
            )

        # 3. status_updates must be empty
        if assessment.status_updates:
            raise ChallengeRejectedError(
                "INCONCLUSIVE verdict must have no status updates"
            )

        # 4-6. nominated must remain SUPPORTED (no SURVIVED, no FINAL)
        # These are enforced by the empty status_updates above, but
        # we double-check:

    def apply_resolution(
        self,
        *,
        proposal: ChallengeProposal,
        resolution: ChallengeResolution,
        evidence: EvidenceAtom,
        lead_result: LeadTournamentResult,
        hypotheses: Mapping[str, CausalHypothesis],
        graph: EvidenceGraph,
        assessment_gate: EvidenceAssessmentGate,
    ) -> None:
        """Validate and then atomically apply the challenge result.

        Execution order:
        1. validate_proposal(...)
        2. validate_resolution(...)
        3. assessment_gate.apply(...)
        4. graph.validate_consistency()
        """
        self.validate_proposal(
            proposal=proposal,
            lead_result=lead_result,
            hypotheses=hypotheses,
            graph=graph,
        )

        self.validate_resolution(
            proposal=proposal,
            resolution=resolution,
            evidence=evidence,
            lead_result=lead_result,
            hypotheses=hypotheses,
            graph=graph,
        )

        assessment_gate.apply(
            assessment=resolution.assessment,
            action=proposal.action,
            evidence=evidence,
            hypotheses=hypotheses,
            graph=graph,
        )

        graph.validate_consistency()
