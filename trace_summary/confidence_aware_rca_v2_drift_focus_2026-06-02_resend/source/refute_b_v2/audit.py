"""Rule audit helpers."""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from refute_b_v2.rule_engine import CandidateRuleResult


def summarize_rule_results(results: Iterable[CandidateRuleResult], true_service: str | None = None) -> dict:
    rows = defaultdict(lambda: {
        "support_count": 0,
        "refute_count": 0,
        "blind_count": 0,
        "support_true_root_count": 0,
        "support_false_root_count": 0,
        "refute_true_root_count": 0,
    })
    for result in results:
        is_true = true_service is not None and result.candidate.service == true_service
        for card in result.cards:
            row = rows[card.rule_id]
            key = f"{card.polarity}_count"
            if key in row:
                row[key] += 1
            if card.polarity == "support":
                row["support_true_root_count" if is_true else "support_false_root_count"] += 1
            if card.polarity == "refute" and is_true:
                row["refute_true_root_count"] += 1
    return {rule_id: dict(values) for rule_id, values in sorted(rows.items())}
