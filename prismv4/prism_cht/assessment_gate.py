"""Evidence assessment gate for PRISM-CHT.

Validates that a Lead Agent's structured assessment of an
EvidenceAtom references only real evidence, targets only action-
eligible hypotheses, performs legal state transitions, and
does not self-contradict.  The gate does not compute scores or
judge semantic correctness.
"""

from __future__ import annotations

from typing import Mapping

from .action_schema import DiscriminativeAction
from .evidence_graph import EvidenceAtom, EvidenceGraph
from .hypothesis import CausalHypothesis
from .tournament_types import (
    AssessmentOutcome,
    EvidenceAssessment,
    EvidenceRelation,
)


class AssessmentRejectedError(ValueError):
    """Raised when an EvidenceAssessment fails gate validation."""


class EvidenceAssessmentGate:
    """Gate that validates and applies evidence assessments.

    ``validate()`` only checks — no state is mutated.
    ``apply()`` runs full validation and writes links and status
    transitions atomically.
    """

    def validate(
        self,
        *,
        assessment: EvidenceAssessment,
        action: DiscriminativeAction,
        evidence: EvidenceAtom,
        hypotheses: Mapping[str, CausalHypothesis],
        graph: EvidenceGraph,
    ) -> None:
        """Validate *assessment* without mutating any state.

        Raises AssessmentRejectedError on the first violation.
        """
        # 1. action_id matches
        if assessment.action_id != action.action_id:
            raise AssessmentRejectedError(
                f"assessment.action_id '{assessment.action_id}' does not "
                f"match action.action_id '{action.action_id}'"
            )

        # 2. evidence_id matches
        if assessment.evidence_id != evidence.evidence_id:
            raise AssessmentRejectedError(
                f"assessment.evidence_id '{assessment.evidence_id}' does not "
                f"match evidence.evidence_id '{evidence.evidence_id}'"
            )

        # 3. evidence exists in graph
        if evidence.evidence_id not in graph.evidence_by_id:
            raise AssessmentRejectedError(
                f"evidence '{evidence.evidence_id}' is not in the graph"
            )

        # 4-7. Validate each link
        for link in assessment.links:
            # 4. link references existing hypothesis
            if link.hypothesis_id not in hypotheses:
                raise AssessmentRejectedError(
                    f"link references non-existent hypothesis "
                    f"'{link.hypothesis_id}'"
                )

            # 5. link references existing evidence
            if link.evidence_id not in graph.evidence_by_id:
                raise AssessmentRejectedError(
                    f"link references non-existent evidence "
                    f"'{link.evidence_id}'"
                )

            # 6. link evidence is the current round's evidence
            if link.evidence_id != evidence.evidence_id:
                raise AssessmentRejectedError(
                    f"link references evidence '{link.evidence_id}' "
                    f"which is not the current round's evidence "
                    f"'{evidence.evidence_id}'"
                )

            # 7. link targets a hypothesis in the action's target set
            if link.hypothesis_id not in action.target_hypothesis_ids:
                raise AssessmentRejectedError(
                    f"link targets hypothesis '{link.hypothesis_id}' "
                    f"which is not in action.target_hypothesis_ids"
                )

            # 8. same evidence cannot both SUPPORTS and CONTRADICTS same hypothesis
            if link.relation == EvidenceRelation.CONTRADICTS:
                for other in assessment.links:
                    if other is link:
                        continue
                    if (
                        other.hypothesis_id == link.hypothesis_id
                        and other.evidence_id == link.evidence_id
                        and other.relation == EvidenceRelation.SUPPORTS
                    ):
                        raise AssessmentRejectedError(
                            f"evidence '{link.evidence_id}' both SUPPORTS and "
                            f"CONTRADICTS hypothesis '{link.hypothesis_id}'"
                        )

            # 9. duplicate link is already handled by EvidenceAssessment.__post_init__
            #    (which rejects duplicate (hypothesis_id, evidence_id) pairs)

            # 18. link rationale is non-empty (already enforced in __post_init__)

        # 10-12. Validate status_updates
        for update in assessment.status_updates:
            # 10. hypothesis exists
            if update.hypothesis_id not in hypotheses:
                raise AssessmentRejectedError(
                    f"status_update references non-existent hypothesis "
                    f"'{update.hypothesis_id}'"
                )

            # 11. hypothesis is in action target set
            if update.hypothesis_id not in action.target_hypothesis_ids:
                raise AssessmentRejectedError(
                    f"status_update targets hypothesis '{update.hypothesis_id}' "
                    f"which is not in action.target_hypothesis_ids"
                )

            # 12. duplicate update already handled by EvidenceAssessment.__post_init__

            # 13. validate state transition (does not mutate)
            h = hypotheses[update.hypothesis_id]
            try:
                h.validate_transition_to(update.new_status)
            except ValueError as exc:
                raise AssessmentRejectedError(str(exc)) from exc

            # 19. status_update rationale non-empty (already enforced in __post_init__)

        # 14. INFORMATIVE must have at least one link
        if assessment.outcome == AssessmentOutcome.INFORMATIVE and not assessment.links:
            raise AssessmentRejectedError(
                "outcome is INFORMATIVE but links is empty"
            )

        # 15. INCONCLUSIVE must have zero links
        if assessment.outcome == AssessmentOutcome.INCONCLUSIVE and assessment.links:
            raise AssessmentRejectedError(
                "outcome is INCONCLUSIVE but links is non-empty"
            )

        # 16. INCONCLUSIVE must have zero status_updates
        if assessment.outcome == AssessmentOutcome.INCONCLUSIVE and assessment.status_updates:
            raise AssessmentRejectedError(
                "outcome is INCONCLUSIVE but status_updates is non-empty"
            )

        # 17. rationale non-empty (already enforced in __post_init__)

    def apply(
        self,
        *,
        assessment: EvidenceAssessment,
        action: DiscriminativeAction,
        evidence: EvidenceAtom,
        hypotheses: Mapping[str, CausalHypothesis],
        graph: EvidenceGraph,
    ) -> None:
        """Validate and then atomically apply the assessment.

        Execution order:
        1. Full validate()
        2. Write graph links (link_support / link_contradiction)
        3. Execute hypothesis.transition_to() for each status update
        4. Call graph.validate_consistency()
        """
        # 1. Validate everything first — no partial writes
        self.validate(
            assessment=assessment,
            action=action,
            evidence=evidence,
            hypotheses=hypotheses,
            graph=graph,
        )

        # 2. Write graph edges
        for link in assessment.links:
            if link.relation == EvidenceRelation.SUPPORTS:
                graph.link_support(link.hypothesis_id, link.evidence_id)
            elif link.relation == EvidenceRelation.CONTRADICTS:
                graph.link_contradiction(link.hypothesis_id, link.evidence_id)

        # 3. Execute state transitions
        for update in assessment.status_updates:
            h = hypotheses[update.hypothesis_id]
            h.transition_to(update.new_status)

        # 4. Verify consistency
        graph.validate_consistency()
