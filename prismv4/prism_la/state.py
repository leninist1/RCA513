"""State types for PRISM-LA: case state, event hypotheses, evidence ledger."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple
import time
import math


class HypothesisStatus(str, Enum):
    ROOT = "root"
    SYMPTOM = "symptom"
    BROAD_EXPLAINER = "broad_explainer"
    DUPLICATE = "duplicate"
    UNRESOLVED = "unresolved"
    MERGED = "merged"
    CONTRADICTED = "contradicted"


@dataclass
class ToolEvidence:
    evidence_id: str
    tool_name: str
    component: str
    factor: str
    support: float
    against: float
    timestamp: Optional[float] = None
    evidence_details: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    payload: Dict[str, Any] = field(default_factory=dict)

    def net_score(self) -> float:
        return max(-1.0, min(1.0, float(self.support) - float(self.against)))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "tool_name": self.tool_name,
            "component": self.component,
            "factor": self.factor,
            "support": round(float(self.support), 4),
            "against": round(float(self.against), 4),
            "timestamp": self.timestamp,
            "evidence_details": list(self.evidence_details),
            "limitations": list(self.limitations),
            "payload": self.payload,
        }


@dataclass
class EventHypothesis:
    event_id: str
    component: str
    reason_hypothesis: str = ""
    time_hypothesis: Optional[float] = None
    status: HypothesisStatus = HypothesisStatus.UNRESOLVED
    support_evidence: List[ToolEvidence] = field(default_factory=list)
    contradictory_evidence: List[ToolEvidence] = field(default_factory=list)
    open_questions: List[str] = field(default_factory=list)
    merged_into: str = ""
    split_from: str = ""
    posterior: float = 0.0
    resolution_note: str = ""

    @property
    def total_support(self) -> float:
        return sum(item.net_score() for item in self.support_evidence)

    @property
    def total_contradiction(self) -> float:
        return sum(abs(item.net_score()) for item in self.contradictory_evidence)

    @property
    def net_evidence(self) -> float:
        return max(-1.0, min(1.0, self.total_support - self.total_contradiction))

    @property
    def evidence_count(self) -> int:
        return len(self.support_evidence) + len(self.contradictory_evidence)

    @property
    def multi_modal(self) -> bool:
        modalities = set()
        for evidence in self.support_evidence:
            for prefix in ("metric", "log", "trace", "counterfactual", "residual"):
                if evidence.tool_name.startswith(prefix) or evidence.factor.startswith(prefix):
                    modalities.add(prefix[:3])
        return len(modalities) >= 2

    @property
    def is_active(self) -> bool:
        return self.status not in {HypothesisStatus.MERGED, HypothesisStatus.CONTRADICTED, HypothesisStatus.DUPLICATE}

    def resolved_status_text(self) -> str:
        if self.status == HypothesisStatus.ROOT:
            return f"ROOT ({self.reason_hypothesis})"
        if self.status == HypothesisStatus.SYMPTOM:
            return f"SYMPTOM of upstream ({self.resolution_note})"
        if self.status == HypothesisStatus.BROAD_EXPLAINER:
            return "BROAD EXPLAINER (not specific root cause)"
        if self.status == HypothesisStatus.DUPLICATE:
            return f"DUPLICATE of ({self.merged_into})"
        if self.status == HypothesisStatus.CONTRADICTED:
            return "CONTRADICTED (evidence disproves)"
        return "UNRESOLVED"

    def to_llm_view(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "component": self.component,
            "reason_hypothesis": self.reason_hypothesis,
            "time_hypothesis": self.time_hypothesis,
            "status": self.status.value,
            "net_evidence": round(self.net_evidence, 4),
            "multi_modal": self.multi_modal,
            "evidence_count": self.evidence_count,
            "open_questions": list(self.open_questions),
            "posterior": round(float(self.posterior), 4),
        }


@dataclass
class EvidenceLedger:
    entries: List[ToolEvidence] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)

    def add(self, evidence: ToolEvidence) -> None:
        self.entries.append(evidence)

    def add_all(self, items: Sequence[ToolEvidence]) -> None:
        self.entries.extend(items)

    def for_component(self, component: str) -> List[ToolEvidence]:
        return [item for item in self.entries if item.component == component]

    def for_event(self, event_id: str) -> List[ToolEvidence]:
        return [item for item in self.entries if item.event_id == event_id]

    def all_components_observed(self) -> List[str]:
        return sorted(set(item.component for item in self.entries))

    def to_llm_view(self) -> List[Dict[str, Any]]:
        return [item.to_dict() for item in self.entries]


@dataclass
class CaseState:
    query_id: str
    system: str
    sub_system: str
    instruction: str
    time_window: Tuple[str, str]
    entities: List[str]
    entity_types: Dict[str, str]
    anchors: List[Dict[str, Any]] = field(default_factory=list)
    candidates: List[Dict[str, Any]] = field(default_factory=list)
    topology_edges: List[Tuple[str, str, float]] = field(default_factory=list)
    has_logs: bool = True
    has_traces: bool = True
    telemetry_summary: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def entity_count(self) -> int:
        return len(self.entities)

    def neighbor_entities(self, component: str, degree: int = 1) -> List[str]:
        neighbors = set()
        frontier = {component}
        for _ in range(degree):
            next_frontier = set()
            for src, dst, weight in self.topology_edges:
                if src in frontier and dst not in neighbors:
                    next_frontier.add(dst)
                if dst in frontier and src not in neighbors:
                    next_frontier.add(src)
            neighbors.update(frontier)
            frontier = next_frontier
        return sorted(neighbors | frontier - {component})

    def to_llm_view(self) -> Dict[str, Any]:
        return {
            "query_id": self.query_id,
            "system": self.system,
            "sub_system": self.sub_system,
            "instruction": self.instruction,
            "time_window": list(self.time_window),
            "entity_count": self.entity_count,
            "entity_types": dict(self.entity_types),
            "available_modalities": {
                "metrics": True,
                "logs": self.has_logs,
                "traces": self.has_traces,
            },
            "anchor_count": len(self.anchors),
            "candidate_count": len(self.candidates),
            "telemetry_summary": self.telemetry_summary,
        }


@dataclass
class InvestigationPlan:
    step_index: int
    actions: List[Dict[str, Any]] = field(default_factory=list)
    rationale: str = ""
    expected_outcome: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_index": int(self.step_index),
            "actions": list(self.actions),
            "rationale": self.rationale,
            "expected_outcome": self.expected_outcome,
        }


@dataclass
class PRISMLAState:
    case: CaseState
    events: List[EventHypothesis] = field(default_factory=list)
    ledger: EvidenceLedger = field(default_factory=EvidenceLedger)
    plans: List[InvestigationPlan] = field(default_factory=list)
    conflicts: List[Dict[str, Any]] = field(default_factory=list)
    step_count: int = 0
    stop_reason: str = ""

    @property
    def active_events(self) -> List[EventHypothesis]:
        return [e for e in self.events if e.is_active]

    @property
    def root_candidates(self) -> List[EventHypothesis]:
        return sorted(
            (e for e in self.events if e.status == HypothesisStatus.ROOT or e.is_active),
            key=lambda e: e.net_evidence,
            reverse=True,
        )

    @property
    def top_root(self) -> Optional[EventHypothesis]:
        ranked = self.root_candidates
        return ranked[0] if ranked else None

    def event_by_id(self, event_id: str) -> Optional[EventHypothesis]:
        for event in self.events:
            if event.event_id == event_id:
                return event
        return None

    def to_llm_view(self) -> Dict[str, Any]:
        return {
            "step_count": self.step_count,
            "events": [e.to_llm_view() for e in self.events],
            "ledger_size": len(self.ledger.entries),
            "open_conflicts": self.conflicts,
            "active_event_count": len(self.active_events),
        }

    def compute_gap(self) -> float:
        ranked = self.root_candidates
        if len(ranked) < 2:
            return 1.0
        return max(0.0, float(ranked[0].net_evidence) - float(ranked[1].net_evidence))
