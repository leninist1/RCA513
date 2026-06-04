"""
evidence_matrix.py -- non-scalar candidate evidence representation.

Scores may still be derived for compatibility, but the primary object is a set
of evidence cards plus an explicit vector of support/refutation/coverage fields.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List


@dataclass(frozen=True)
class EvidenceCard:
    modality: str
    kind: str
    polarity: str  # support | refute | blind
    strength: float
    text: str
    details: dict = field(default_factory=dict)
    round_index: int = 1

    def to_dict(self) -> dict:
        return {
            "modality": self.modality,
            "kind": self.kind,
            "polarity": self.polarity,
            "strength": self.strength,
            "text": self.text,
            "details": self.details,
            "round": self.round_index,
        }


@dataclass
class CandidateEvidence:
    service: str
    reason: str
    cards: List[EvidenceCard] = field(default_factory=list)
    coverage: Dict[str, str] = field(default_factory=dict)
    cluster_similarity: float | None = None
    topology_roles: List[str] = field(default_factory=list)

    def add(self, card: EvidenceCard) -> None:
        self.cards.append(card)

    @property
    def hard_support(self) -> int:
        return sum(1 for c in self.cards if c.polarity == "support")

    @property
    def hard_refute(self) -> int:
        return sum(1 for c in self.cards if c.polarity == "refute")

    @property
    def blind_count(self) -> int:
        return sum(1 for c in self.cards if c.polarity == "blind")

    @property
    def support_strength(self) -> float:
        return sum(c.strength for c in self.cards if c.polarity == "support")

    @property
    def refute_strength(self) -> float:
        return sum(c.strength for c in self.cards if c.polarity == "refute")

    def vector(self) -> dict:
        return {
            "hard_support": self.hard_support,
            "hard_refute": self.hard_refute,
            "blind_count": self.blind_count,
            "support_strength": self.support_strength,
            "refute_strength": self.refute_strength,
            "coverage": self.coverage,
            "cluster_similarity": self.cluster_similarity,
            "topology_roles": self.topology_roles,
        }

    def to_dict(self) -> dict:
        return {
            "candidate": {"service": self.service, "reason": self.reason},
            "evidence_vector": self.vector(),
            "evidence_cards": [card.to_dict() for card in self.cards],
        }


def bucket_candidates(candidates: Iterable[CandidateEvidence]) -> dict:
    high, low, blind = [], [], []
    for cand in candidates:
        if cand.hard_support > 0 and cand.hard_refute == 0:
            high.append(cand)
        elif cand.hard_support == 0 and cand.blind_count > 0:
            blind.append(cand)
        else:
            low.append(cand)
    return {
        "high_suspicion": [c.to_dict() for c in high],
        "low_suspicion": [c.to_dict() for c in low],
        "data_blind_candidates": [c.to_dict() for c in blind],
    }
