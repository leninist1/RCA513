"""Challenge protocol types for PRISM-CHT Phase 3.

Defines the vocabulary for adversarial single-challenge review:
proposal, resolution, verdict, audit, and snapshot types.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping

from .action_schema import DiscriminativeAction
from .evidence_graph import EvidenceGraph
from .hypothesis import CausalHypothesis
from .tournament_types import (
    EvidenceAssessment,
    HypothesisSnapshot,
    InvestigationAuditStep,
    LeadNomination,
    LeadTournamentResult,
    _make_hypothesis_snapshot,
)


class ChallengeVerdict(str, Enum):
    NOMINATION_SURVIVED = "nomination_survived"
    NOMINATION_REFUTED = "nomination_refuted"
    INCONCLUSIVE = "inconclusive"


@dataclass(frozen=True)
class ChallengeProposal:
    challenge_id: str
    nominated_hypothesis_id: str
    competitor_hypothesis_ids: tuple[str, ...]
    challenge_claim: str
    falsification_target: str
    action: DiscriminativeAction
    rationale: str

    def __post_init__(self):
        if not self.challenge_id or not self.challenge_id.strip():
            raise ValueError("challenge_id must be non-empty")
        if not self.nominated_hypothesis_id or not self.nominated_hypothesis_id.strip():
            raise ValueError("nominated_hypothesis_id must be non-empty")
        if not self.competitor_hypothesis_ids:
            raise ValueError("competitor_hypothesis_ids must contain at least one item")
        if len(set(self.competitor_hypothesis_ids)) != len(self.competitor_hypothesis_ids):
            raise ValueError("competitor_hypothesis_ids contains duplicate entries")
        if self.nominated_hypothesis_id in self.competitor_hypothesis_ids:
            raise ValueError(
                f"competitor cannot equal nominated hypothesis "
                f"'{self.nominated_hypothesis_id}'"
            )
        if not self.challenge_claim or not self.challenge_claim.strip():
            raise ValueError("challenge_claim must be non-empty")
        if not self.falsification_target or not self.falsification_target.strip():
            raise ValueError("falsification_target must be non-empty")
        if not self.rationale or not self.rationale.strip():
            raise ValueError("rationale must be non-empty")


@dataclass(frozen=True)
class ChallengeResolution:
    challenge_id: str
    action_id: str
    evidence_id: str
    verdict: ChallengeVerdict
    assessment: EvidenceAssessment
    rationale: str

    def __post_init__(self):
        if not self.challenge_id or not self.challenge_id.strip():
            raise ValueError("challenge_id must be non-empty")
        if not self.action_id or not self.action_id.strip():
            raise ValueError("action_id must be non-empty")
        if not self.evidence_id or not self.evidence_id.strip():
            raise ValueError("evidence_id must be non-empty")
        if not self.rationale or not self.rationale.strip():
            raise ValueError("rationale must be non-empty")


@dataclass(frozen=True)
class ChallengeAuditStep:
    proposal: ChallengeProposal
    evidence_id: str
    resolution: ChallengeResolution

    def __post_init__(self):
        if not self.evidence_id or not self.evidence_id.strip():
            raise ValueError("evidence_id must be non-empty")


@dataclass(frozen=True)
class ChallengeReviewResult:
    status: str
    nominated_hypothesis_id: str
    verdict: ChallengeVerdict
    challenge_id: str
    evidence_id: str
    audit_step: ChallengeAuditStep

    def __post_init__(self):
        _ALLOWED_STATUSES = {
            "survived_challenge",
            "returned_to_lead",
            "challenge_inconclusive",
        }
        if self.status not in _ALLOWED_STATUSES:
            raise ValueError(
                f"ChallengeReviewResult.status must be one of "
                f"{sorted(_ALLOWED_STATUSES)}, got '{self.status}'"
            )
        if not self.nominated_hypothesis_id or not self.nominated_hypothesis_id.strip():
            raise ValueError("nominated_hypothesis_id must be non-empty")
        if not self.challenge_id or not self.challenge_id.strip():
            raise ValueError("challenge_id must be non-empty")
        if not self.evidence_id or not self.evidence_id.strip():
            raise ValueError("evidence_id must be non-empty")


@dataclass(frozen=True)
class ChallengeSnapshot:
    """Immutable, read-only view of tournament state at challenge time.

    No mutable references to graph or live hypothesis objects leak.
    """

    nominated_hypothesis: HypothesisSnapshot
    competitor_hypotheses: tuple[HypothesisSnapshot, ...]
    evidence_ids: tuple[str, ...]
    lead_nomination: LeadNomination
    lead_audit_steps: tuple[InvestigationAuditStep, ...]

    def __post_init__(self):
        nominated_id = self.nominated_hypothesis.hypothesis_id
        all_competitor_ids = [h.hypothesis_id for h in self.competitor_hypotheses]
        if len(set(all_competitor_ids)) != len(all_competitor_ids):
            raise ValueError("competitor_hypotheses contains duplicate hypothesis_id")
        if nominated_id in all_competitor_ids:
            raise ValueError(
                f"competitor_hypotheses contains the nominated hypothesis "
                f"'{nominated_id}'"
            )
        if len(set(self.evidence_ids)) != len(self.evidence_ids):
            raise ValueError("evidence_ids contains duplicate entries")


def build_challenge_snapshot(
    *,
    lead_result: LeadTournamentResult,
    hypotheses: Mapping[str, CausalHypothesis],
    graph: EvidenceGraph,
) -> ChallengeSnapshot:
    """Build a read-only challenge snapshot from the Lead result.

    - Nominated hypothesis is snapshotted from *hypotheses*.
    - Competitor hypotheses are snapshotted from *lead_result.nomination.addressed_competitor_ids*,
      sorted by hypothesis_id.
    - Evidence IDs are deduplicated and sorted from *graph*.
    - Lead nomination and audit steps are passed through directly.
    """
    hid = lead_result.nominated_hypothesis_id
    nominated_snapshot = _make_hypothesis_snapshot(hypotheses[hid])

    competitor_ids = lead_result.nomination.addressed_competitor_ids
    competitor_snapshots = tuple(
        sorted(
            (_make_hypothesis_snapshot(hypotheses[cid]) for cid in competitor_ids),
            key=lambda s: s.hypothesis_id,
        )
    )

    evidence_ids = tuple(sorted(graph.evidence_by_id.keys()))

    return ChallengeSnapshot(
        nominated_hypothesis=nominated_snapshot,
        competitor_hypotheses=competitor_snapshots,
        evidence_ids=evidence_ids,
        lead_nomination=lead_result.nomination,
        lead_audit_steps=lead_result.audit_steps,
    )
