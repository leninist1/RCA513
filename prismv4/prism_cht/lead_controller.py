"""Lead-only tournament controller for PRISM-CHT.

Orchestrates a single-agent (Lead only) causal hypothesis tournament
without an LLM.  The controller drives the investigation loop,
delegates evidence interpretation to the policy and validation to
the assessment gate, and enforces the nomination guardrail.
"""

from __future__ import annotations

from typing import Sequence

from .action_schema import DiscriminativeAction
from .assessment_gate import AssessmentRejectedError, EvidenceAssessmentGate
from .evidence_graph import EvidenceGraph
from .executor import InvestigationExecutor
from .hypothesis import CausalHypothesis, HypothesisStatus
from .lead_policy import LeadPolicy
from .tournament_types import (
    InvestigationAuditStep,
    LeadNomination,
    LeadTournamentResult,
    build_snapshot,
)


class TournamentBudgetExhaustedError(RuntimeError):
    """Raised when max_rounds is reached and another action is proposed."""


class NominationRejectedError(ValueError):
    """Raised when a nomination fails the controller's guardrails."""


class LeadTournamentController:
    """Single-agent Lead tournament runner.

    Drives the investigation loop: build snapshot → decide → execute →
    assess → gate-apply → repeat.  On nomination, validates against
    the evidence graph and returns ``LeadTournamentResult``.

    The controller does NOT interpret evidence, link evidence to
    hypotheses, compute scores, call an LLM, or implement a Challenger.
    """

    def __init__(
        self,
        *,
        executor: InvestigationExecutor,
        assessment_gate: EvidenceAssessmentGate,
        graph: EvidenceGraph,
        max_rounds: int = 4,
    ) -> None:
        if max_rounds < 1:
            raise ValueError("max_rounds must be at least 1")
        self._executor = executor
        self._assessment_gate = assessment_gate
        self._graph = graph
        self._max_rounds = max_rounds
        self._has_run = False
        self._hypotheses: dict[str, CausalHypothesis] = {}

    def run(
        self,
        *,
        initial_hypotheses: Sequence[CausalHypothesis],
        policy: LeadPolicy,
    ) -> LeadTournamentResult:
        if self._has_run:
            raise RuntimeError(
                "LeadTournamentController.run() can only be called once; "
                "create a new instance for each tournament"
            )
        self._has_run = True

        # --- validate initial state -------------------------------------------------

        if len(initial_hypotheses) < 2:
            raise ValueError(
                "At least two initial hypotheses are required, "
                f"got {len(initial_hypotheses)}"
            )

        seen_ids: set[str] = set()
        for h in initial_hypotheses:
            hid = h.hypothesis_id
            if hid in seen_ids:
                raise ValueError(
                    f"Duplicate hypothesis_id '{hid}' in initial_hypotheses"
                )
            seen_ids.add(hid)

            if h.status == HypothesisStatus.DRAFT:
                h.activate()
            elif h.status == HypothesisStatus.ACTIVE:
                pass  # already active, allowed
            else:
                raise ValueError(
                    f"Hypothesis '{hid}' has status '{h.status.value}'; "
                    f"only DRAFT or ACTIVE hypotheses are accepted as initial input"
                )

        # Graph must be clean — no pre-existing evidence or hypotheses
        if self._graph.evidence_count() != 0:
            raise ValueError(
                "EvidenceGraph must be empty of evidence at tournament start"
            )
        if self._graph.hypotheses_by_id:
            raise ValueError(
                "EvidenceGraph must be empty of registered hypotheses "
                "at tournament start"
            )

        # Register all hypotheses
        for h in initial_hypotheses:
            self._graph.register_hypothesis(h)
            self._hypotheses[h.hypothesis_id] = h

        # --- investigation loop -----------------------------------------------------

        audit_steps: list[InvestigationAuditStep] = []
        round_index = 0

        while True:
            snapshot = build_snapshot(
                round_index=round_index,
                hypotheses=self._hypotheses,
                graph=self._graph,
                audit_steps=audit_steps,
            )

            decision = policy.decide_next(snapshot=snapshot)

            if isinstance(decision, DiscriminativeAction):
                # Budget check: the action will consume a round
                if round_index >= self._max_rounds:
                    raise TournamentBudgetExhaustedError(
                        f"max_rounds ({self._max_rounds}) reached; "
                        f"policy returned another action but budget is exhausted"
                    )

                action = decision

                # Execute the action — produces an EvidenceAtom
                evidence = self._executor.execute(
                    action=action,
                    hypotheses=self._hypotheses,
                )

                # Build a snapshot that includes the new evidence
                snapshot_with_evidence = build_snapshot(
                    round_index=round_index,
                    hypotheses=self._hypotheses,
                    graph=self._graph,
                    audit_steps=audit_steps,
                )

                # Ask policy to assess the evidence
                assessment = policy.assess_evidence(
                    snapshot=snapshot_with_evidence,
                    action=action,
                    evidence=evidence,
                )

                # Apply through gate (validates then writes)
                self._assessment_gate.apply(
                    assessment=assessment,
                    action=action,
                    evidence=evidence,
                    hypotheses=self._hypotheses,
                    graph=self._graph,
                )

                # Record audit step
                audit_steps.append(
                    InvestigationAuditStep(
                        round_index=round_index,
                        action=action,
                        evidence_id=evidence.evidence_id,
                        assessment=assessment,
                    )
                )

                round_index += 1

            elif isinstance(decision, LeadNomination):
                nomination = decision
                self._validate_nomination(nomination)
                return LeadTournamentResult(
                    status="challenge_required",
                    nominated_hypothesis_id=nomination.hypothesis_id,
                    rounds_completed=round_index,
                    evidence_ids=tuple(sorted(self._graph.evidence_by_id.keys())),
                    audit_steps=tuple(audit_steps),
                    nomination=nomination,
                )

            else:
                raise TypeError(
                    f"Policy returned unexpected type "
                    f"{type(decision).__name__}"
                )

    def _validate_nomination(self, nomination: LeadNomination) -> None:
        """Validate nomination against the evidence graph and hypothesis states.

        Raises NominationRejectedError on any violation.
        """
        hid = nomination.hypothesis_id

        # 1. nominated hypothesis exists
        if hid not in self._graph.hypotheses_by_id:
            raise NominationRejectedError(
                f"Nominated hypothesis '{hid}' does not exist"
            )
        h = self._graph.hypotheses_by_id[hid]

        # 12. cannot be SURVIVED (check before SUPPORTED since SURVIVED is a
        #     successor of SUPPORTED and we want to reject it first)
        if h.status == HypothesisStatus.SURVIVED:
            raise NominationRejectedError(
                f"Nominated hypothesis '{hid}' is SURVIVED; "
                f"Lead Agent cannot output SURVIVED"
            )

        # 13. cannot be FINAL
        if h.status == HypothesisStatus.FINAL:
            raise NominationRejectedError(
                f"Nominated hypothesis '{hid}' is FINAL; "
                f"Lead Agent cannot output FINAL"
            )

        # 2. nominated hypothesis status must be SUPPORTED
        if h.status != HypothesisStatus.SUPPORTED:
            raise NominationRejectedError(
                f"Nominated hypothesis '{hid}' has status "
                f"'{h.status.value}', must be SUPPORTED"
            )

        # 3. at least one supporting evidence
        if not nomination.supporting_evidence_ids:
            raise NominationRejectedError(
                "Nomination must reference at least one supporting evidence"
            )

        # 4-5. each supporting evidence must exist and be linked
        graph_support_edges = self._graph.support_edges.get(hid, set())
        for eid in nomination.supporting_evidence_ids:
            if eid not in self._graph.evidence_by_id:
                raise NominationRejectedError(
                    f"Supporting evidence '{eid}' does not exist in graph"
                )
            if eid not in graph_support_edges:
                raise NominationRejectedError(
                    f"Supporting evidence '{eid}' is not linked to "
                    f"hypothesis '{hid}' via a support edge"
                )

        # 6. at least one addressed competitor
        if not nomination.addressed_competitor_ids:
            raise NominationRejectedError(
                "Nomination must address at least one competitor"
            )

        # 7-10. validate each competitor
        for cid in nomination.addressed_competitor_ids:
            # 7. competitor exists
            if cid not in self._graph.hypotheses_by_id:
                raise NominationRejectedError(
                    f"Addressed competitor '{cid}' does not exist"
                )

            # 8. competitor != nominated
            if cid == hid:
                raise NominationRejectedError(
                    f"Addressed competitor '{cid}' cannot be the nominated "
                    f"hypothesis itself"
                )

            comp = self._graph.hypotheses_by_id[cid]

            # 9. competitor must be WEAKENED or REFUTED
            if comp.status not in (HypothesisStatus.WEAKENED, HypothesisStatus.REFUTED):
                raise NominationRejectedError(
                    f"Addressed competitor '{cid}' has status "
                    f"'{comp.status.value}', must be WEAKENED or REFUTED"
                )

            # 10. competitor must have at least one contradiction evidence
            comp_contra_edges = self._graph.contradiction_edges.get(cid, set())
            if not comp_contra_edges:
                raise NominationRejectedError(
                    f"Addressed competitor '{cid}' has no contradiction "
                    f"evidence in graph"
                )

        # 11. rationale non-empty (already enforced by LeadNomination.__post_init__)

        # 15. triplet grounding: each dimension must reference existing,
        #     Lead-support evidence linked to the nominee
        grounding = nomination.triplet_grounding
        for dim_name, dim_value in [
            ("component", grounding.component_evidence_ids),
            ("reason", grounding.reason_evidence_ids),
            ("onset", grounding.onset_evidence_ids),
        ]:
            for eid in dim_value:
                if eid not in self._graph.evidence_by_id:
                    raise NominationRejectedError(
                        f"Nomination triplet grounding ({dim_name}) "
                        f"references non-existent evidence '{eid}'"
                    )
                if eid not in nomination.supporting_evidence_ids:
                    raise NominationRejectedError(
                        f"Nomination triplet grounding ({dim_name}) "
                        f"references evidence '{eid}' which is not in "
                        f"supporting_evidence_ids"
                    )
                if eid not in graph_support_edges:
                    raise NominationRejectedError(
                        f"Nomination triplet grounding ({dim_name}) "
                        f"references evidence '{eid}' which is not linked "
                        f"to hypothesis '{hid}' via a support edge"
                    )

        # 14. status can only be challenge_required (enforced by LeadTournamentResult)
