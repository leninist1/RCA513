"""
layer2_refutation.py -- Phase 2 / B-L2 MVP

Rank candidates by refutation score: lower score means harder to refute.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List

from refute.src.evidence_query import EvidenceQuery, reason_to_bucket


@dataclass(frozen=True)
class Candidate:
    service: str
    reason: str


@dataclass
class RefutationReport:
    candidate: Candidate
    rebuttal_score: float
    support_strength: float = 0.0
    supporting: List[str] = field(default_factory=list)
    refuting: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "candidate": {"service": self.candidate.service, "reason": self.candidate.reason},
            "rebuttal_score": self.rebuttal_score,
            "support_strength": self.support_strength,
            "supporting": self.supporting,
            "refuting": self.refuting,
        }


class RefutationEngine:
    NODE_RESOURCE_BUCKETS = {"disk", "filesystem", "network"}

    def __init__(self, evidence: EvidenceQuery):
        self.evidence = evidence

    def evaluate_candidate(self, candidate: Candidate) -> RefutationReport:
        bucket = reason_to_bucket(candidate.reason)
        report = RefutationReport(candidate=candidate, rebuttal_score=0.0)

        container = self.evidence.is_container_kpi_anomalous(candidate.service, bucket)
        if container.matched:
            report.supporting.append(f"container_{bucket}: {container.evidence}")
            report.support_strength += container.strength

        node = self.evidence.is_anomaly_explained_by_node(candidate.service, bucket)
        if bucket in self.NODE_RESOURCE_BUCKETS:
            if node.matched:
                report.supporting.append(f"node_{bucket}: {node.evidence}")
                report.support_strength += node.strength
            if bucket == "network":
                trace_service = self.evidence.is_service_trace_anomalous(candidate.service)
                trace_edge = self.evidence.has_slow_trace_edge(candidate.service)
                if trace_service.matched:
                    report.supporting.append(f"trace_service: {trace_service.evidence}")
                    report.support_strength += 0.10 * trace_service.strength
                if trace_edge.matched:
                    report.supporting.append(f"trace_edge: {trace_edge.evidence}")
                    role_boost = 1.0
                    if trace_edge.details and trace_edge.details.get("dst") == candidate.service:
                        role_boost = 4.0
                    elif trace_edge.details and trace_edge.details.get("src") == candidate.service:
                        role_boost = 0.5
                    report.support_strength += role_boost * trace_edge.strength
            if not container.matched and not node.matched:
                if container.unavailable or node.unavailable:
                    report.refuting.append(f"resource_{bucket}: metric unavailable")
                    return report
                if bucket == "network":
                    trace_service = self.evidence.is_service_trace_anomalous(candidate.service)
                    trace_edge = self.evidence.has_slow_trace_edge(candidate.service)
                    if not trace_service.matched and not trace_edge.matched:
                        report.rebuttal_score += 1.0
                        report.refuting.append(f"resource_{bucket}: no container, node, or trace anomaly")
                else:
                    report.rebuttal_score += 1.0
                    report.refuting.append(f"resource_{bucket}: no container or node anomaly")
        else:
            if not container.matched:
                if container.unavailable:
                    report.refuting.append(f"container_{bucket}: unavailable ({container.evidence})")
                else:
                    report.rebuttal_score += 1.0
                    report.refuting.append(f"container_{bucket}: {container.evidence}")
            if node.matched and not container.matched and not container.unavailable:
                report.rebuttal_score += 1.0
                report.refuting.append(f"node_explains_{bucket}: {node.evidence}")
            elif node.matched:
                report.supporting.append(f"node_context_{bucket}: {node.evidence}")
                report.support_strength += 0.25 * node.strength

        log = self.evidence.does_match_log_keyword(candidate.service)
        if log.matched:
            report.supporting.append(f"log: {log.evidence}")
            report.support_strength += log.strength
        elif log.unavailable:
            report.refuting.append(f"log: unavailable ({log.evidence})")

        return report

    def rank_candidates(self, candidates: Iterable[Candidate]) -> List[RefutationReport]:
        reports = [self.evaluate_candidate(candidate) for candidate in candidates]
        return sorted(reports, key=lambda r: (r.rebuttal_score, -r.support_strength, -len(r.supporting), r.candidate.service))
