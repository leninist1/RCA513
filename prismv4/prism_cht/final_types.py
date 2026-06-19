"""Final output protocol for PRISM-CHT Phase 4.

Defines the minimal, read-only FinalRCAResult that the deterministic
FinalVerifier produces when a hypothesis survives the full adversarial
review pipeline (Lead Tournament + Challenger Review).
"""

from __future__ import annotations

from dataclasses import dataclass

from .tournament_types import TripletEvidenceCoverage


@dataclass(frozen=True)
class FinalRCAResult:
    """Minimal grounded RCA result — no scores, no graph, no raw payload.

    Only a hypothesis that has passed:
      LeadTournamentResult(status="challenge_required")
      → ChallengeReviewResult(status="survived_challenge")
    may produce a FinalRCAResult.
    """

    status: str
    hypothesis_id: str
    root_component: str
    reason_family: str
    onset_interval: tuple[float, float]
    triplet_grounding: TripletEvidenceCoverage
    lead_supporting_evidence_ids: tuple[str, ...]
    challenge_evidence_id: str
    referenced_evidence_ids: tuple[str, ...]

    def __post_init__(self):
        if self.status != "final_verified":
            raise ValueError(
                f"FinalRCAResult.status must be 'final_verified', "
                f"got '{self.status}'"
            )

        if not self.hypothesis_id or not self.hypothesis_id.strip():
            raise ValueError("hypothesis_id must be non-empty")

        if not self.root_component or not self.root_component.strip():
            raise ValueError("root_component must be non-empty")

        if not self.reason_family or not self.reason_family.strip():
            raise ValueError("reason_family must be non-empty")

        if len(self.onset_interval) != 2:
            raise ValueError(
                f"onset_interval must have exactly 2 elements, "
                f"got {len(self.onset_interval)}"
            )
        start, end = self.onset_interval
        if start > end:
            raise ValueError(
                f"onset_interval start ({start}) must be <= end ({end})"
            )

        if len(self.lead_supporting_evidence_ids) < 1:
            raise ValueError(
                "lead_supporting_evidence_ids must contain at least one item"
            )

        if len(set(self.lead_supporting_evidence_ids)) != len(
            self.lead_supporting_evidence_ids
        ):
            raise ValueError(
                "lead_supporting_evidence_ids contains duplicate entries"
            )

        if not self.challenge_evidence_id or not self.challenge_evidence_id.strip():
            raise ValueError("challenge_evidence_id must be non-empty")

        # Collect all grounding evidence IDs from triplet_grounding
        grounding_ids = set(self.triplet_grounding.component_evidence_ids)
        grounding_ids.update(self.triplet_grounding.reason_evidence_ids)
        grounding_ids.update(self.triplet_grounding.onset_evidence_ids)

        expected_set = (
            set(self.lead_supporting_evidence_ids)
            | {self.challenge_evidence_id}
            | grounding_ids
        )
        actual_set = set(self.referenced_evidence_ids)
        if not expected_set.issubset(actual_set):
            missing = sorted(expected_set - actual_set)
            raise ValueError(
                f"referenced_evidence_ids missing expected entries: {missing}"
            )

        if len(set(self.referenced_evidence_ids)) != len(
            self.referenced_evidence_ids
        ):
            raise ValueError(
                "referenced_evidence_ids contains duplicate entries"
            )
