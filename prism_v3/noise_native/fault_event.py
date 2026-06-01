"""Event-level state for the noise-native PRISM agent."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Optional


@dataclass
class EvidenceObservation:
    evidence_id: str
    tool_name: str
    event_id: str
    component: str
    factor_name: str
    factor_delta: float
    payload: Dict[str, Any] = field(default_factory=dict)

    def to_debug(self) -> Dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "tool_name": self.tool_name,
            "event_id": self.event_id,
            "component": self.component,
            "factor_name": self.factor_name,
            "factor_delta": round(float(self.factor_delta), 6),
            "payload": self.payload,
        }


@dataclass
class PosteriorAttribution:
    event_id: str
    old_posterior: float
    new_posterior: float
    factor_name: str
    factor_delta: float
    evidence_ids: List[str] = field(default_factory=list)

    def to_debug(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "old_posterior": round(float(self.old_posterior), 6),
            "new_posterior": round(float(self.new_posterior), 6),
            "factor_name": self.factor_name,
            "factor_delta": round(float(self.factor_delta), 6),
            "evidence_ids": list(self.evidence_ids),
        }


@dataclass
class FactorState:
    value: float = 0.0
    confidence: float = 0.0
    evidence_ids: List[str] = field(default_factory=list)
    update_count: int = 0
    last_updated_step: int = -1

    def update(self, observation: EvidenceObservation, step: int = 0) -> None:
        incoming = float(observation.factor_delta)
        if self.update_count <= 0:
            self.value = incoming
        elif observation.factor_name in {
            "metric_likelihood",
            "log_likelihood",
            "trace_direction_likelihood",
            "counterfactual_likelihood",
            "residual_collapse",
            "NoiseLab_logit",
            "source_likelihood",
            "source_isolation",
            "symptomness",
            "hotspot_symptom",
            "broad_explainer",
            "mechanism_break_likelihood",
            "mechanism_parent_refutation",
            "intervention_uniqueness",
        }:
            # Tool observations are snapshots of the same latent factor. Repeating
            # the same tool should refine confidence, not linearly add evidence.
            if abs(incoming) > abs(self.value):
                self.value = incoming
            else:
                self.value = 0.85 * self.value + 0.15 * incoming
        elif observation.factor_name == "pairwise_verdict":
            self.value = 0.65 * self.value + 0.35 * incoming
        elif observation.factor_name in {"merge_duplicate", "split_event"}:
            self.value = incoming
        else:
            self.value = 0.75 * self.value + 0.25 * incoming
        self.update_count += 1
        self.last_updated_step = int(step)
        if observation.evidence_id not in self.evidence_ids:
            self.evidence_ids.append(observation.evidence_id)
        self.confidence = min(1.0, self.confidence + 0.30)

    def to_debug(self) -> Dict[str, Any]:
        return {
            "value": round(float(self.value), 6),
            "confidence": round(float(self.confidence), 6),
            "update_count": int(self.update_count),
            "last_updated_step": int(self.last_updated_step),
            "evidence_ids": list(self.evidence_ids),
        }


@dataclass
class FaultEvent:
    event_id: str
    component: str
    reason: str = ""
    time: Optional[float] = None
    posterior: float = 0.0
    status: str = "active"
    status_probs: Dict[str, float] = field(
        default_factory=lambda: {
            "root": 0.20,
            "symptom": 0.20,
            "broad_explainer": 0.10,
            "duplicate": 0.05,
            "unresolved": 0.45,
        }
    )
    evidence_ledger: List[EvidenceObservation] = field(default_factory=list)
    conflict_notes: List[str] = field(default_factory=list)
    reason_candidates: List[Dict[str, Any]] = field(default_factory=list)
    factor_states: Dict[str, FactorState] = field(default_factory=dict)
    factors: Dict[str, float] = field(default_factory=dict)
    observed_tools: List[str] = field(default_factory=list)
    candidate_id: str = ""
    merged_into: str = ""
    split_from: str = ""

    def add_observation(self, observation: EvidenceObservation, step: int = 0) -> None:
        self.evidence_ledger.append(observation)
        factor_state = self.factor_states.setdefault(
            observation.factor_name, FactorState()
        )
        factor_state.update(observation, step=step)
        self.factors[observation.factor_name] = factor_state.value
        if observation.tool_name not in self.observed_tools:
            self.observed_tools.append(observation.tool_name)

    def factor_value(self, name: str, default: float = 0.0) -> float:
        return float(self.factors.get(name, default))

    def has_tool(self, tool_name: str) -> bool:
        return tool_name in self.observed_tools

    def status_entropy(self) -> float:
        probs = [max(float(value), 1e-9) for value in self.status_probs.values()]
        total = sum(probs)
        if total <= 0:
            return 0.0
        normalized = [value / total for value in probs]
        entropy = -sum(value * math.log(value) for value in normalized)
        return float(entropy / math.log(max(2, len(normalized))))

    def root_probability(self) -> float:
        return float(self.status_probs.get("root", 0.0))

    def to_debug(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "component": self.component,
            "reason": self.reason,
            "time": self.time,
            "posterior": round(float(self.posterior), 6),
            "status": self.status,
            "status_probs": {
                key: round(float(value), 6)
                for key, value in sorted(self.status_probs.items())
            },
            "status_entropy": round(float(self.status_entropy()), 6),
            "candidate_id": self.candidate_id,
            "merged_into": self.merged_into,
            "split_from": self.split_from,
            "factors": {k: round(float(v), 6) for k, v in sorted(self.factors.items())},
            "factor_states": {
                key: state.to_debug()
                for key, state in sorted(self.factor_states.items())
            },
            "observed_tools": list(self.observed_tools),
            "conflict_notes": list(self.conflict_notes),
            "reason_candidates": list(self.reason_candidates),
            "evidence_ledger": [
                observation.to_debug() for observation in self.evidence_ledger[-12:]
            ],
        }


@dataclass
class NoiseNativeAgentState:
    events: List[FaultEvent] = field(default_factory=list)
    observations: List[EvidenceObservation] = field(default_factory=list)
    posterior_attributions: List[PosteriorAttribution] = field(default_factory=list)
    action_history: List[Dict[str, Any]] = field(default_factory=list)
    uncertainty_history: List[Dict[str, Any]] = field(default_factory=list)
    stop_reason: str = ""

    @property
    def active_events(self) -> List[FaultEvent]:
        return [event for event in self.events if event.status == "active"]

    def event_by_id(self, event_id: str) -> Optional[FaultEvent]:
        for event in self.events:
            if event.event_id == event_id:
                return event
        return None

    def to_debug(self) -> Dict[str, Any]:
        return {
            "stop_reason": self.stop_reason,
            "events": [event.to_debug() for event in self.events],
            "observations": [
                observation.to_debug() for observation in self.observations[-20:]
            ],
            "posterior_attributions": [
                attribution.to_debug()
                for attribution in self.posterior_attributions[-30:]
            ],
            "action_history": list(self.action_history),
            "uncertainty_history": list(self.uncertainty_history),
        }
