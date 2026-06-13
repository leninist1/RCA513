"""Layer 3: d32 conflict arbitration/report boundary.

This module deliberately does not guess roots. It receives Layer 2 decisions and
may only choose among them or report conflict/blind-spot information.
"""
from __future__ import annotations

from typing import Any, Mapping

from refute_b_v2_d32.schema import RefutationDecision


def conflict_summary(decisions: list[RefutationDecision], gap: float = 0.3) -> dict[str, Any]:
    if not decisions:
        return {"status": "no_decisions", "needs_llm": False}
    if len(decisions) == 1:
        return {"status": "single_candidate", "needs_llm": False}
    first, second = decisions[0], decisions[1]
    score_gap = float(second.rebuttal_score - first.rebuttal_score)
    return {
        "status": "conflict" if score_gap < gap else "clear_winner",
        "needs_llm": score_gap < gap,
        "score_gap": score_gap,
        "top1": first.to_dict(),
        "top2": second.to_dict(),
    }


def investigation_report(result: Mapping[str, Any]) -> dict[str, Any]:
    high = result.get("high_suspicion", [])
    blind = result.get("data_blind_spots", [])
    return {
        "summary": "d32 refutation output",
        "high_suspicion_count": len(high),
        "blind_spot_count": len(blind),
        "recommended_next_action": "inspect high_suspicion evidence cards; collect blind modalities if decisive",
    }

