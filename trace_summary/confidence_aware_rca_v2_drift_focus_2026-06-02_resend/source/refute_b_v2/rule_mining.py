"""Small-sample rule mining helpers.

This is not full FP-growth; Bank has only 136 cases. The implementation mines
frequent singletons and pairs with confidence/lift, producing auditable rule
suggestions rather than silently injecting rules into the engine.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable


@dataclass(frozen=True)
class RuleSuggestion:
    antecedent: tuple[str, ...]
    target: str
    support: int
    confidence: float
    lift: float

    def to_dict(self) -> dict:
        return {
            "antecedent": list(self.antecedent),
            "target": self.target,
            "support": self.support,
            "confidence": self.confidence,
            "lift": self.lift,
        }


def mine_rule_suggestions(cases: Iterable[tuple[Iterable[str], str]], min_support: int = 3,
                          min_confidence: float = 0.6, max_len: int = 2) -> list[RuleSuggestion]:
    rows = [(tuple(sorted(set(features))), str(target)) for features, target in cases]
    n = len(rows)
    if n == 0:
        return []
    target_counts: dict[str, int] = {}
    item_target_counts: dict[tuple[tuple[str, ...], str], int] = {}
    item_counts: dict[tuple[str, ...], int] = {}
    for features, target in rows:
        target_counts[target] = target_counts.get(target, 0) + 1
        candidates = []
        for size in range(1, max_len + 1):
            candidates.extend(combinations(features, size))
        for item in candidates:
            item_counts[item] = item_counts.get(item, 0) + 1
            item_target_counts[(item, target)] = item_target_counts.get((item, target), 0) + 1
    suggestions = []
    for (item, target), support in item_target_counts.items():
        if support < min_support:
            continue
        confidence = support / item_counts[item]
        if confidence < min_confidence:
            continue
        prior = target_counts[target] / n
        lift = confidence / prior if prior else 0.0
        suggestions.append(RuleSuggestion(item, target, support, confidence, lift))
    return sorted(suggestions, key=lambda row: (-row.confidence, -row.lift, -row.support, row.antecedent, row.target))


def signature_features(signature: dict) -> list[str]:
    features = []
    for service in signature.get("services", []):
        for modality in ("metric", "log", "trace", "topology"):
            for name, item in service.get(modality, {}).items():
                state = item.get("state", "")
                intensity = int(item.get("intensity", 0) or 0)
                if state == "support":
                    features.append(f"{modality}:{name}:support:{min(3, intensity)}")
                elif state == "blind":
                    features.append(f"{modality}:{name}:blind")
    for blind in signature.get("blind_spots", []):
        features.append(f"blind:{blind}")
    return sorted(set(features))
