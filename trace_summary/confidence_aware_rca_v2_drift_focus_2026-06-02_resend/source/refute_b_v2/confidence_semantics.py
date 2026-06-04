"""Semantic confidence explanations."""
from __future__ import annotations

from dataclasses import dataclass

from refute_b_v2.evidence_matrix import MatrixRow


@dataclass(frozen=True)
class ConfidenceExplanation:
    label: str
    factors: dict
    explanation: str

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "factors": self.factors,
            "explanation": self.explanation,
        }


def explain_confidence(row: MatrixRow, similar_case_count: int = 0, novelty_threshold: float = 0.3,
                       best_similarity: float = 0.0) -> ConfidenceExplanation:
    v = row.vector
    support_count = v.support_hard + v.support_soft
    refute_count = v.refute_hard + v.refute_soft
    modality_diversity = v.coverage
    novel = best_similarity < novelty_threshold
    if v.refute_hard > 0 or support_count == 0:
        label = "LOW"
    elif v.blind > 0 or modality_diversity < 2:
        label = "MEDIUM"
    elif support_count >= 2 and refute_count == 0:
        label = "HIGH"
    else:
        label = "MEDIUM"
    factors = {
        "support_count": support_count,
        "hard_support_count": v.support_hard,
        "refute_count": refute_count,
        "hard_refute_count": v.refute_hard,
        "blind_count": v.blind,
        "modality_diversity": modality_diversity,
        "similar_case_count": similar_case_count,
        "best_similarity": best_similarity,
        "novel_pattern": novel,
    }
    explanation = _sentence(row, factors, label)
    return ConfidenceExplanation(label, factors, explanation)


def _sentence(row: MatrixRow, factors: dict, label: str) -> str:
    cand = row.result.candidate
    if label == "HIGH":
        base = f"{cand.service}/{cand.reason} has convergent evidence across {factors['modality_diversity']} modalities"
    elif label == "LOW":
        base = f"{cand.service}/{cand.reason} lacks sufficient support or has hard refuting evidence"
    else:
        base = f"{cand.service}/{cand.reason} has partial evidence and needs verification"
    if factors["blind_count"]:
        base += f"; {factors['blind_count']} blind evidence areas remain"
    if factors["novel_pattern"]:
        base += "; no close historical signature match was found"
    return base + "."
