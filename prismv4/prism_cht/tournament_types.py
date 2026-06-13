"""Agent output protocols and tournament snapshot types for PRISM-CHT.

Defines the vocabulary that Lead and Challenger agents use to
communicate structured assessments, nominations, and state
snapshots.  All types are frozen and recursively validated.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Sequence

from .action_schema import DiscriminativeAction
from .canonical import deep_freeze
from .evidence_graph import EvidenceGraph
from .hypothesis import CausalHypothesis, HypothesisStatus


# ===========================================================================
# Enums
# ===========================================================================


class EvidenceRelation(str, Enum):
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"


class AssessmentOutcome(str, Enum):
    INFORMATIVE = "informative"
    INCONCLUSIVE = "inconclusive"


# ===========================================================================
# Agent proposal types
# ===========================================================================


@dataclass(frozen=True)
class EvidenceLinkProposal:
    hypothesis_id: str
    evidence_id: str
    relation: EvidenceRelation
    rationale: str

    def __post_init__(self):
        if not self.hypothesis_id or not self.hypothesis_id.strip():
            raise ValueError("hypothesis_id must be non-empty")
        if not self.evidence_id or not self.evidence_id.strip():
            raise ValueError("evidence_id must be non-empty")
        if not self.rationale or not self.rationale.strip():
            raise ValueError("rationale must be non-empty")


@dataclass(frozen=True)
class HypothesisStatusUpdate:
    hypothesis_id: str
    new_status: HypothesisStatus
    rationale: str

    def __post_init__(self):
        if not self.hypothesis_id or not self.hypothesis_id.strip():
            raise ValueError("hypothesis_id must be non-empty")
        if not self.rationale or not self.rationale.strip():
            raise ValueError("rationale must be non-empty")


@dataclass(frozen=True)
class EvidenceAssessment:
    action_id: str
    evidence_id: str
    outcome: AssessmentOutcome
    links: tuple[EvidenceLinkProposal, ...]
    status_updates: tuple[HypothesisStatusUpdate, ...]
    rationale: str

    def __post_init__(self):
        if not self.action_id or not self.action_id.strip():
            raise ValueError("action_id must be non-empty")
        if not self.evidence_id or not self.evidence_id.strip():
            raise ValueError("evidence_id must be non-empty")
        if not self.rationale or not self.rationale.strip():
            raise ValueError("rationale must be non-empty")

        # Deduplicate links by (hypothesis_id, evidence_id, relation) triple
        seen_links: set[tuple[str, str, EvidenceRelation]] = set()
        deduped: list[EvidenceLinkProposal] = []
        for link in self.links:
            key = (link.hypothesis_id, link.evidence_id, link.relation)
            if key in seen_links:
                raise ValueError(
                    f"Duplicate link: hypothesis '{link.hypothesis_id}' "
                    f"-> evidence '{link.evidence_id}' "
                    f"({link.relation.value})"
                )
            seen_links.add(key)
            deduped.append(link)

        # Deduplicate status_updates by hypothesis_id
        seen_updates: set[str] = set()
        deduped_updates: list[HypothesisStatusUpdate] = []
        for u in self.status_updates:
            if u.hypothesis_id in seen_updates:
                raise ValueError(
                    f"Duplicate status_update for hypothesis '{u.hypothesis_id}'"
                )
            seen_updates.add(u.hypothesis_id)
            deduped_updates.append(u)


@dataclass(frozen=True)
class LeadNomination:
    hypothesis_id: str
    supporting_evidence_ids: tuple[str, ...]
    addressed_competitor_ids: tuple[str, ...]
    rationale: str

    def __post_init__(self):
        if not self.hypothesis_id or not self.hypothesis_id.strip():
            raise ValueError("hypothesis_id must be non-empty")
        if not self.rationale or not self.rationale.strip():
            raise ValueError("rationale must be non-empty")

        # Deduplicate supporting_evidence_ids
        if len(set(self.supporting_evidence_ids)) != len(self.supporting_evidence_ids):
            raise ValueError("supporting_evidence_ids contains duplicate entries")

        # Deduplicate addressed_competitor_ids
        if len(set(self.addressed_competitor_ids)) != len(self.addressed_competitor_ids):
            raise ValueError("addressed_competitor_ids contains duplicate entries")


# ===========================================================================
# Audit and snapshot types
# ===========================================================================


@dataclass(frozen=True)
class InvestigationAuditStep:
    round_index: int
    action: DiscriminativeAction
    evidence_id: str
    assessment: EvidenceAssessment

    def __post_init__(self):
        if not self.evidence_id or not self.evidence_id.strip():
            raise ValueError("evidence_id must be non-empty")


@dataclass(frozen=True)
class HypothesisSnapshot:
    """Immutable, read-only view of a CausalHypothesis at a point in time."""

    hypothesis_id: str
    root_component: str
    reason_family: str
    onset_interval: tuple[float, float]
    local_trigger: str
    propagation_path: tuple[str, ...]
    explained_symptoms: tuple[str, ...]
    predicted_observations: tuple[str, ...]
    falsifiers: tuple[str, ...]
    supporting_evidence_ids: tuple[str, ...]
    contradicting_evidence_ids: tuple[str, ...]
    unresolved_questions: tuple[str, ...]
    status: HypothesisStatus

    def __post_init__(self):
        if not self.hypothesis_id or not self.hypothesis_id.strip():
            raise ValueError("hypothesis_id must be non-empty")
        if not self.root_component or not self.root_component.strip():
            raise ValueError("root_component must be non-empty")


@dataclass(frozen=True)
class LeadTournamentSnapshot:
    """Read-only snapshot of tournament state — no mutable references leak."""

    round_index: int
    hypotheses: tuple[HypothesisSnapshot, ...]
    evidence_ids: tuple[str, ...]
    audit_steps: tuple[InvestigationAuditStep, ...]

    def __post_init__(self):
        # hypotheses: sorted by hypothesis_id
        if len({h.hypothesis_id for h in self.hypotheses}) != len(self.hypotheses):
            raise ValueError("hypotheses contains duplicate hypothesis_id")

        # evidence_ids: sorted, deduped
        if len(set(self.evidence_ids)) != len(self.evidence_ids):
            raise ValueError("evidence_ids contains duplicate entries")


@dataclass(frozen=True)
class LeadTournamentResult:
    status: str
    nominated_hypothesis_id: str
    rounds_completed: int
    evidence_ids: tuple[str, ...]
    audit_steps: tuple[InvestigationAuditStep, ...]
    nomination: LeadNomination

    def __post_init__(self):
        if self.status != "challenge_required":
            raise ValueError(
                f"LeadTournamentResult.status must be 'challenge_required', "
                f"got '{self.status}'"
            )
        if not self.nominated_hypothesis_id or not self.nominated_hypothesis_id.strip():
            raise ValueError("nominated_hypothesis_id must be non-empty")

        # evidence_ids: no duplicates
        if len(set(self.evidence_ids)) != len(self.evidence_ids):
            raise ValueError("evidence_ids contains duplicate entries")


# ===========================================================================
# Snapshot builder
# ===========================================================================


def _make_hypothesis_snapshot(h: CausalHypothesis) -> HypothesisSnapshot:
    """Build a read-only HypothesisSnapshot from a live CausalHypothesis."""
    return HypothesisSnapshot(
        hypothesis_id=h.hypothesis_id,
        root_component=h.root_component,
        reason_family=h.reason_family,
        onset_interval=tuple(h.onset_interval),
        local_trigger=h.local_trigger,
        propagation_path=tuple(h.propagation_path),
        explained_symptoms=tuple(h.explained_symptoms),
        predicted_observations=tuple(h.predicted_observations),
        falsifiers=tuple(h.falsifiers),
        supporting_evidence_ids=tuple(h.supporting_evidence_ids),
        contradicting_evidence_ids=tuple(h.contradicting_evidence_ids),
        unresolved_questions=tuple(h.unresolved_questions),
        status=h.status,
    )


def build_snapshot(
    *,
    round_index: int,
    hypotheses: Mapping[str, CausalHypothesis],
    graph: EvidenceGraph,
    audit_steps: Sequence[InvestigationAuditStep],
) -> LeadTournamentSnapshot:
    """Build a read-only tournament snapshot.

    - Hypotheses are sorted by *hypothesis_id*.
    - Evidence IDs are deduplicated and sorted.
    - No mutable references to *graph* or live *CausalHypothesis*
      objects leak into the returned snapshot.
    """
    snapshots = sorted(
        (_make_hypothesis_snapshot(h) for h in hypotheses.values()),
        key=lambda s: s.hypothesis_id,
    )

    evidence_ids = tuple(sorted(graph.evidence_by_id.keys()))

    return LeadTournamentSnapshot(
        round_index=round_index,
        hypotheses=tuple(snapshots),
        evidence_ids=evidence_ids,
        audit_steps=tuple(audit_steps),
    )
