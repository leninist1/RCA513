"""
iterative_refutation.py -- looped L2 evidence expansion.

Round 1 uses cheap local evidence. Round 2 expands topology/trace evidence.
Round 3 adds log and blind-spot bookkeeping. Ambiguous cases can then be sent
to the L3 agent.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from refute.src.evidence_matrix import CandidateEvidence, EvidenceCard, bucket_candidates
from refute.src.evidence_query import EvidenceQuery, reason_to_bucket
from refute.src.layer2_refutation import Candidate


@dataclass(frozen=True)
class IterationState:
    round_index: int
    stop: bool
    stop_reason: str
    gap: float
    candidates: List[CandidateEvidence]

    def to_dict(self) -> dict:
        return {
            "round": self.round_index,
            "stop": self.stop,
            "stop_reason": self.stop_reason,
            "gap": self.gap,
            "candidates": [c.to_dict() for c in self.candidates],
        }


class IterativeRefutationEngine:
    def __init__(self, evidence: EvidenceQuery, gap_threshold: float = 3.0):
        self.evidence = evidence
        self.gap_threshold = gap_threshold

    def run(self, candidates: Iterable[Candidate], max_rounds: int = 3) -> dict:
        matrix = [CandidateEvidence(c.service, c.reason, coverage=dict(self.evidence.modal_status)) for c in candidates]
        rounds = []
        for round_index in range(1, max_rounds + 1):
            for cand in matrix:
                self._expand(cand, round_index)
            ranked = self._rank_for_stopping(matrix)
            gap = self._gap(ranked)
            stop = gap >= self.gap_threshold and ranked[0].hard_support > 0
            reason = "gap_converged" if stop else "continue"
            rounds.append(IterationState(round_index, stop, reason, gap, ranked).to_dict())
            if stop:
                break
        final_ranked = self._rank_for_stopping(matrix)
        needs_llm = not rounds[-1]["stop"]
        out = {
            "rounds": rounds,
            "needs_llm": needs_llm,
            "final_state": "ambiguous" if needs_llm else "resolved",
            "evidence_matrix": [c.to_dict() for c in final_ranked],
        }
        out.update(bucket_candidates(final_ranked))
        return out

    def _expand(self, cand: CandidateEvidence, round_index: int) -> None:
        bucket = reason_to_bucket(cand.reason)
        if round_index == 1:
            self._add_metric(cand, bucket, node_level=False, round_index=round_index)
            if bucket in {"disk", "filesystem", "network"}:
                self._add_metric(cand, bucket, node_level=True, round_index=round_index)
        elif round_index == 2:
            if bucket == "network":
                self._add_trace(cand, round_index)
        elif round_index == 3:
            self._add_log(cand, round_index)
            self._add_blind_cards(cand, round_index)

    def _add_metric(self, cand: CandidateEvidence, bucket: str, node_level: bool, round_index: int) -> None:
        result = (
            self.evidence.is_node_kpi_anomalous(cand.service, bucket)
            if node_level else
            self.evidence.is_container_kpi_anomalous(cand.service, bucket)
        )
        kind = f"{'node' if node_level else 'container'}_{bucket}"
        self._add_result(cand, "metric", kind, result, round_index)

    def _add_trace(self, cand: CandidateEvidence, round_index: int) -> None:
        self._add_result(cand, "trace", "service_trace", self.evidence.is_service_trace_anomalous(cand.service), round_index, weak=0.1)
        edge = self.evidence.has_slow_trace_edge(cand.service)
        if edge.matched and edge.details:
            if edge.details.get("dst") == cand.service:
                edge = edge.__class__(edge.matched, edge.evidence, edge.strength * 4.0, edge.details, edge.unavailable)
            elif edge.details.get("src") == cand.service:
                edge = edge.__class__(edge.matched, edge.evidence, edge.strength * 0.5, edge.details, edge.unavailable)
        self._add_result(cand, "trace", "slow_edge", edge, round_index)

    def _add_log(self, cand: CandidateEvidence, round_index: int) -> None:
        self._add_result(cand, "log", "keyword", self.evidence.does_match_log_keyword(cand.service), round_index)

    def _add_blind_cards(self, cand: CandidateEvidence, round_index: int) -> None:
        for modality, status in self.evidence.modal_status.items():
            if status != "present":
                cand.add(EvidenceCard(modality, "coverage", "blind", 0.0, f"{modality} is {status}", {"status": status}, round_index))

    @staticmethod
    def _add_result(cand: CandidateEvidence, modality: str, kind: str, result, round_index: int, weak: float = 1.0) -> None:
        if result.matched:
            cand.add(EvidenceCard(modality, kind, "support", result.strength * weak, result.evidence, result.details or {}, round_index))
        elif result.unavailable:
            cand.add(EvidenceCard(modality, kind, "blind", 0.0, result.evidence, result.details or {}, round_index))
        else:
            cand.add(EvidenceCard(modality, kind, "refute", 1.0, result.evidence, result.details or {}, round_index))

    @staticmethod
    def _rank_for_stopping(candidates: List[CandidateEvidence]) -> List[CandidateEvidence]:
        return sorted(candidates, key=lambda c: (-c.support_strength, c.hard_refute, -c.hard_support, c.service))

    @staticmethod
    def _gap(ranked: List[CandidateEvidence]) -> float:
        if len(ranked) < 2:
            return ranked[0].support_strength if ranked else 0.0
        return ranked[0].support_strength - ranked[1].support_strength
