"""Evidence-matrix decision layer.

The matrix layer separates evidence dimensions instead of collapsing every
candidate into one global score. It still provides deterministic groups for
automation, but keeps support/refute/blind dimensions visible.
"""
from __future__ import annotations

from dataclasses import dataclass

from refute_b_v2.rule_engine import CandidateRuleResult


@dataclass(frozen=True)
class EvidenceVector:
    support_hard: int
    support_soft: int
    refute_hard: int
    refute_soft: int
    blind: int
    support_strength: float
    refute_strength: float
    modalities: tuple[str, ...]

    @property
    def coverage(self) -> int:
        return len(self.modalities)

    def to_dict(self) -> dict:
        return {
            "support_hard": self.support_hard,
            "support_soft": self.support_soft,
            "refute_hard": self.refute_hard,
            "refute_soft": self.refute_soft,
            "blind": self.blind,
            "support_strength": self.support_strength,
            "refute_strength": self.refute_strength,
            "modalities": list(self.modalities),
            "coverage": self.coverage,
        }


@dataclass(frozen=True)
class MatrixRow:
    result: CandidateRuleResult
    vector: EvidenceVector

    def to_dict(self) -> dict:
        return {
            **self.result.to_dict(),
            "evidence_vector_v2": self.vector.to_dict(),
        }


@dataclass(frozen=True)
class EvidenceMatrixDecision:
    high_suspicion: tuple[MatrixRow, ...]
    low_suspicion: tuple[MatrixRow, ...]
    data_blind_spots: tuple[MatrixRow, ...]
    ambiguous: tuple[MatrixRow, ...]

    def to_dict(self) -> dict:
        return {
            "high_suspicion": [row.to_dict() for row in self.high_suspicion],
            "low_suspicion": [row.to_dict() for row in self.low_suspicion],
            "data_blind_spots": [row.to_dict() for row in self.data_blind_spots],
            "ambiguous": [row.to_dict() for row in self.ambiguous],
        }


def vector_from_result(result: CandidateRuleResult) -> EvidenceVector:
    modalities = sorted({card.modality for card in result.cards if card.polarity != "blind"})
    return EvidenceVector(
        support_hard=sum(1 for c in result.cards if c.polarity == "support" and c.rule_type == "hard"),
        support_soft=sum(1 for c in result.cards if c.polarity == "support" and c.rule_type != "hard"),
        refute_hard=sum(1 for c in result.cards if c.polarity == "refute" and c.rule_type == "hard"),
        refute_soft=sum(1 for c in result.cards if c.polarity == "refute" and c.rule_type != "hard"),
        blind=sum(1 for c in result.cards if c.polarity == "blind"),
        support_strength=result.support_strength,
        refute_strength=result.refute_strength,
        modalities=tuple(modalities),
    )


def build_matrix(results: list[CandidateRuleResult]) -> list[MatrixRow]:
    return [MatrixRow(result, vector_from_result(result)) for result in results]


def decide_matrix(results: list[CandidateRuleResult]) -> EvidenceMatrixDecision:
    rows = build_matrix(results)
    high = []
    low = []
    blind = []
    ambiguous = []
    for row in rows:
        v = row.vector
        if v.refute_hard > 0 or (v.refute_strength > v.support_strength and v.support_hard == 0):
            low.append(row)
        elif v.support_strength <= 0 and v.blind > 0:
            blind.append(row)
        elif v.support_hard > 0 or (v.support_strength > 0 and v.refute_hard == 0):
            high.append(row)
        else:
            ambiguous.append(row)
    high = _pareto_front(high)
    return EvidenceMatrixDecision(
        high_suspicion=tuple(high),
        low_suspicion=tuple(low),
        data_blind_spots=tuple(blind),
        ambiguous=tuple(ambiguous),
    )


def _dominates(left: MatrixRow, right: MatrixRow) -> bool:
    a, b = left.vector, right.vector
    no_worse = (
        a.support_hard >= b.support_hard
        and a.support_soft >= b.support_soft
        and a.coverage >= b.coverage
        and a.refute_hard <= b.refute_hard
        and a.refute_soft <= b.refute_soft
        and a.blind <= b.blind
    )
    strictly_better = (
        a.support_hard > b.support_hard
        or a.support_soft > b.support_soft
        or a.coverage > b.coverage
        or a.refute_hard < b.refute_hard
        or a.refute_soft < b.refute_soft
        or a.blind < b.blind
    )
    return no_worse and strictly_better


def _pareto_front(rows: list[MatrixRow]) -> list[MatrixRow]:
    out = []
    for row in rows:
        if not any(_dominates(other, row) for other in rows if other is not row):
            out.append(row)
    return sorted(out, key=lambda r: (r.result.candidate.service, r.result.candidate.reason))
