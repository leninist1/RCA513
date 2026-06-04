"""Iterative L2 refutation loop."""
from __future__ import annotations

from dataclasses import dataclass

from refute_b_v2.evidence_matrix import EvidenceMatrixDecision, MatrixRow, decide_matrix
from refute_b_v2.rule_engine import RuleEngine
from refute_b_v2.rules import Candidate


DEFAULT_ROUNDS = (
    {"round_1_cheap_metric"},
    {"round_1_cheap_metric", "round_2_trace_topology"},
    {"round_1_cheap_metric", "round_2_trace_topology", "round_3_log_blind"},
)


@dataclass(frozen=True)
class IterationState:
    round_index: int
    groups: tuple[str, ...]
    decision: EvidenceMatrixDecision
    stop_reason: str

    def to_dict(self) -> dict:
        return {
            "round_index": self.round_index,
            "groups": list(self.groups),
            "stop_reason": self.stop_reason,
            "decision": self.decision.to_dict(),
        }


def run_iterative_refutation(engine: RuleEngine, candidates: list[Candidate], evidence,
                             rounds: tuple[set[str], ...] = DEFAULT_ROUNDS,
                             min_high_gap: float = 3.0) -> list[IterationState]:
    states = []
    for idx, groups in enumerate(rounds, start=1):
        results = [engine.run_candidate(candidate, evidence, groups=groups) for candidate in candidates]
        decision = decide_matrix(results)
        stop_reason = _stop_reason(decision, min_high_gap)
        states.append(IterationState(idx, tuple(sorted(groups)), decision, stop_reason))
        if stop_reason != "continue":
            break
    return states


def final_decision(states: list[IterationState]) -> EvidenceMatrixDecision:
    if not states:
        raise ValueError("no iteration states")
    return states[-1].decision


def _stop_reason(decision: EvidenceMatrixDecision, min_high_gap: float) -> str:
    high = list(decision.high_suspicion)
    if len(high) == 0 and decision.data_blind_spots:
        return "data_blind_spot"
    if len(high) == 1:
        return "single_high_candidate"
    if len(high) >= 2:
        ordered = sorted(high, key=lambda row: row.vector.support_strength - row.vector.refute_strength, reverse=True)
        gap = _net(ordered[0]) - _net(ordered[1])
        if gap >= min_high_gap:
            return "support_gap"
    return "continue"


def _net(row: MatrixRow) -> float:
    return row.vector.support_strength - row.vector.refute_strength
