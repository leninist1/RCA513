"""Deterministic final verifier for PRISM-CHT Phase 4.

Only validates — never infers.  Only allows:
  ChallengeReviewResult(status="survived_challenge")
  → CausalHypothesis: SURVIVED → FINAL
  → FinalRCAResult(status="final_verified")
"""

from __future__ import annotations

from typing import Mapping

from .challenge_types import ChallengeReviewResult, ChallengeVerdict
from .evidence_graph import EvidenceGraph
from .final_types import FinalRCAResult
from .hypothesis import CausalHypothesis, HypothesisStatus
from .tournament_types import EvidenceRelation, LeadTournamentResult


class FinalizationRejectedError(ValueError):
    """Raised when a hypothesis cannot be finalized."""


class FinalVerifier:
    """Deterministic final verifier.

    Does not call tools, does not call agents, does not compute scores.
    Only checks that the nominated hypothesis has passed Lead Tournament
    and a single Challenger review with survived_challenge status, then
    atomically promotes SURVIVED → FINAL.
    """

    def validate(
        self,
        *,
        lead_result: LeadTournamentResult,
        challenge_result: ChallengeReviewResult,
        hypotheses: Mapping[str, CausalHypothesis],
        graph: EvidenceGraph,
    ) -> None:
        """Check all finalization invariants without mutating any state."""

        # ==================================================================
        # A. Graph basic consistency
        # ==================================================================

        try:
            graph.validate_consistency()
        except ValueError as e:
            raise FinalizationRejectedError(
                f"Graph consistency check failed: {e}"
            ) from e

        hyp_ids = set(hypotheses.keys())
        graph_ids = set(graph.hypotheses_by_id.keys())
        if hyp_ids != graph_ids:
            raise FinalizationRejectedError(
                f"hypotheses ID set {sorted(hyp_ids)} does not match "
                f"graph registered set {sorted(graph_ids)}"
            )

        for hid, h in hypotheses.items():
            if graph.hypotheses_by_id.get(hid) is not h:
                raise FinalizationRejectedError(
                    f"Hypothesis '{hid}' in hypotheses parameter is not "
                    f"the same object as registered in graph"
                )

        # ==================================================================
        # B. Lead results
        # ==================================================================

        if lead_result.status != "challenge_required":
            raise FinalizationRejectedError(
                f"LeadTournamentResult.status must be 'challenge_required', "
                f"got '{lead_result.status}'"
            )

        nominee_id = lead_result.nominated_hypothesis_id
        if not nominee_id or not nominee_id.strip():
            raise FinalizationRejectedError(
                "lead_result.nominated_hypothesis_id must be non-empty"
            )

        if lead_result.nomination.hypothesis_id != nominee_id:
            raise FinalizationRejectedError(
                f"nomination.hypothesis_id "
                f"'{lead_result.nomination.hypothesis_id}' does not match "
                f"nominated_hypothesis_id '{nominee_id}'"
            )

        lead_support_ids = lead_result.nomination.supporting_evidence_ids
        if len(lead_support_ids) == 0:
            raise FinalizationRejectedError(
                "lead_result.nomination.supporting_evidence_ids must not be empty"
            )

        if len(set(lead_support_ids)) != len(lead_support_ids):
            raise FinalizationRejectedError(
                "lead nomination supporting_evidence_ids contains duplicate entries"
            )

        for eid in lead_support_ids:
            if eid not in graph.evidence_by_id:
                raise FinalizationRejectedError(
                    f"Lead supporting evidence '{eid}' not found in graph"
                )

        sup_edges_nominee = graph.support_edges.get(nominee_id, set())
        for eid in lead_support_ids:
            if eid not in sup_edges_nominee:
                raise FinalizationRejectedError(
                    f"Lead supporting evidence '{eid}' is not linked via "
                    f"support edge to nominated hypothesis '{nominee_id}'"
                )

        competitor_ids = lead_result.nomination.addressed_competitor_ids
        if len(competitor_ids) == 0:
            raise FinalizationRejectedError(
                "lead nomination addressed_competitor_ids must not be empty"
            )

        for cid in competitor_ids:
            if cid not in graph.hypotheses_by_id:
                raise FinalizationRejectedError(
                    f"Addressed competitor '{cid}' not registered in graph"
                )

        for cid in competitor_ids:
            if cid == nominee_id:
                raise FinalizationRejectedError(
                    f"Addressed competitor '{cid}' equals nominated "
                    f"hypothesis '{nominee_id}'"
                )

        for cid in competitor_ids:
            ch = graph.hypotheses_by_id[cid]
            if len(ch.contradicting_evidence_ids) == 0:
                raise FinalizationRejectedError(
                    f"Addressed competitor '{cid}' has no contradicting evidence"
                )

        # ==================================================================
        # C. Challenger results
        # ==================================================================

        if challenge_result.status != "survived_challenge":
            raise FinalizationRejectedError(
                f"ChallengeReviewResult.status must be 'survived_challenge', "
                f"got '{challenge_result.status}'"
            )

        if challenge_result.verdict != ChallengeVerdict.NOMINATION_SURVIVED:
            raise FinalizationRejectedError(
                f"Challenge verdict must be NOMINATION_SURVIVED, "
                f"got '{challenge_result.verdict.value}'"
            )

        if challenge_result.nominated_hypothesis_id != nominee_id:
            raise FinalizationRejectedError(
                f"challenge_result.nominated_hypothesis_id "
                f"'{challenge_result.nominated_hypothesis_id}' does not "
                f"match lead nominee '{nominee_id}'"
            )

        if (
            not challenge_result.challenge_id
            or not challenge_result.challenge_id.strip()
        ):
            raise FinalizationRejectedError(
                "challenge_result.challenge_id must be non-empty"
            )

        chall_evidence_id = challenge_result.evidence_id
        if not chall_evidence_id or not chall_evidence_id.strip():
            raise FinalizationRejectedError(
                "challenge_result.evidence_id must be non-empty"
            )

        if chall_evidence_id not in graph.evidence_by_id:
            raise FinalizationRejectedError(
                f"Challenge evidence '{chall_evidence_id}' not found in graph"
            )

        audit = challenge_result.audit_step

        if audit.evidence_id != chall_evidence_id:
            raise FinalizationRejectedError(
                f"audit_step.evidence_id '{audit.evidence_id}' does not "
                f"match challenge_result.evidence_id '{chall_evidence_id}'"
            )

        resolution = audit.resolution

        if resolution.evidence_id != chall_evidence_id:
            raise FinalizationRejectedError(
                f"resolution.evidence_id '{resolution.evidence_id}' does not "
                f"match challenge_result.evidence_id '{chall_evidence_id}'"
            )

        if resolution.challenge_id != challenge_result.challenge_id:
            raise FinalizationRejectedError(
                f"resolution.challenge_id '{resolution.challenge_id}' does "
                f"not match challenge_result.challenge_id "
                f"'{challenge_result.challenge_id}'"
            )

        if resolution.verdict != ChallengeVerdict.NOMINATION_SURVIVED:
            raise FinalizationRejectedError(
                f"resolution.verdict must be NOMINATION_SURVIVED, "
                f"got '{resolution.verdict.value}'"
            )

        assessment = resolution.assessment

        has_grounding = False
        for link in assessment.links:
            if (
                link.hypothesis_id == nominee_id
                and link.relation == EvidenceRelation.SUPPORTS
            ):
                has_grounding = True
                break
            if (
                link.hypothesis_id in competitor_ids
                and link.relation == EvidenceRelation.CONTRADICTS
            ):
                has_grounding = True
                break
        if not has_grounding:
            raise FinalizationRejectedError(
                "Challenge resolution assessment has no grounding: "
                f"no link SUPPORTS nominee '{nominee_id}' or "
                f"CONTRADICTS any competitor"
            )

        for link in assessment.links:
            if link.relation == EvidenceRelation.SUPPORTS:
                edges = graph.support_edges.get(link.hypothesis_id, set())
            else:
                edges = graph.contradiction_edges.get(link.hypothesis_id, set())
            if link.evidence_id not in edges:
                raise FinalizationRejectedError(
                    f"Assessment link ({link.hypothesis_id} "
                    f"{link.relation.value} {link.evidence_id}) is not "
                    f"present in graph edges"
                )

        for link in assessment.links:
            if link.evidence_id not in graph.evidence_by_id:
                raise FinalizationRejectedError(
                    f"Assessment references non-existent evidence "
                    f"'{link.evidence_id}'"
                )

        for link in assessment.links:
            if link.evidence_id != chall_evidence_id:
                raise FinalizationRejectedError(
                    f"Assessment links reference evidence "
                    f"'{link.evidence_id}' which is not the challenge "
                    f"evidence '{chall_evidence_id}'"
                )

        # ==================================================================
        # D. Final candidate hypothesis
        # ==================================================================

        if nominee_id not in hypotheses:
            raise FinalizationRejectedError(
                f"Nominated hypothesis '{nominee_id}' not found in hypotheses"
            )

        nominee_hyp = hypotheses[nominee_id]

        if nominee_hyp.status != HypothesisStatus.SURVIVED:
            raise FinalizationRejectedError(
                f"Nominated hypothesis '{nominee_id}' status is "
                f"'{nominee_hyp.status.value}', expected 'survived'"
            )

        if not nominee_hyp.root_component or not nominee_hyp.root_component.strip():
            raise FinalizationRejectedError(
                f"Nominated hypothesis '{nominee_id}' root_component is empty"
            )

        if not nominee_hyp.reason_family or not nominee_hyp.reason_family.strip():
            raise FinalizationRejectedError(
                f"Nominated hypothesis '{nominee_id}' reason_family is empty"
            )

        if len(nominee_hyp.onset_interval) != 2:
            raise FinalizationRejectedError(
                f"Nominated hypothesis '{nominee_id}' onset_interval has "
                f"{len(nominee_hyp.onset_interval)} elements, expected 2"
            )
        start, end = nominee_hyp.onset_interval
        if start > end:
            raise FinalizationRejectedError(
                f"Nominated hypothesis '{nominee_id}' onset_interval "
                f"start ({start}) > end ({end})"
            )

        if nominee_hyp.status == HypothesisStatus.FINAL:
            raise FinalizationRejectedError(
                f"Nominated hypothesis '{nominee_id}' is already FINAL"
            )

        try:
            nominee_hyp.validate_transition_to(HypothesisStatus.FINAL)
        except ValueError as e:
            raise FinalizationRejectedError(
                f"Cannot finalize hypothesis '{nominee_id}': {e}"
            ) from e

        # ==================================================================
        # E. Triplet grounding re-validation
        # ==================================================================

        grounding = lead_result.nomination.triplet_grounding
        for dim_name, dim_value in [
            ("component", grounding.component_evidence_ids),
            ("reason", grounding.reason_evidence_ids),
            ("onset", grounding.onset_evidence_ids),
        ]:
            for eid in dim_value:
                if eid not in graph.evidence_by_id:
                    raise FinalizationRejectedError(
                        f"Triplet grounding ({dim_name}) references "
                        f"non-existent evidence '{eid}'"
                    )
                if eid not in lead_support_ids:
                    raise FinalizationRejectedError(
                        f"Triplet grounding ({dim_name}) references "
                        f"evidence '{eid}' which is not in "
                        f"lead supporting_evidence_ids"
                    )
                if eid not in sup_edges_nominee:
                    raise FinalizationRejectedError(
                        f"Triplet grounding ({dim_name}) references "
                        f"evidence '{eid}' which is not linked to "
                        f"hypothesis '{nominee_id}' via a support edge"
                    )

    def finalize(
        self,
        *,
        lead_result: LeadTournamentResult,
        challenge_result: ChallengeReviewResult,
        hypotheses: Mapping[str, CausalHypothesis],
        graph: EvidenceGraph,
    ) -> FinalRCAResult:
        self.validate(
            lead_result=lead_result,
            challenge_result=challenge_result,
            hypotheses=hypotheses,
            graph=graph,
        )

        nominee_id = lead_result.nominated_hypothesis_id
        nominee_hyp = hypotheses[nominee_id]

        with graph.relation_transaction():
            nominee_hyp.transition_to(HypothesisStatus.FINAL)

        grounding = lead_result.nomination.triplet_grounding
        seen: set[str] = set()
        ref_ids: list[str] = []
        for eid in lead_result.nomination.supporting_evidence_ids:
            if eid not in seen:
                seen.add(eid)
                ref_ids.append(eid)
        challenge_eid = challenge_result.evidence_id
        if challenge_eid not in seen:
            ref_ids.append(challenge_eid)
        for eid in grounding.component_evidence_ids:
            if eid not in seen:
                seen.add(eid)
                ref_ids.append(eid)
        for eid in grounding.reason_evidence_ids:
            if eid not in seen:
                seen.add(eid)
                ref_ids.append(eid)
        for eid in grounding.onset_evidence_ids:
            if eid not in seen:
                seen.add(eid)
                ref_ids.append(eid)

        return FinalRCAResult(
            status="final_verified",
            hypothesis_id=nominee_id,
            root_component=nominee_hyp.root_component,
            reason_family=nominee_hyp.reason_family,
            onset_interval=nominee_hyp.onset_interval,
            triplet_grounding=grounding,
            lead_supporting_evidence_ids=lead_result.nomination.supporting_evidence_ids,
            challenge_evidence_id=challenge_eid,
            referenced_evidence_ids=tuple(ref_ids),
        )
