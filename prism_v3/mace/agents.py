"""Specialized agents and shared memory for MACE-RCA."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from .graph import ObjectGraph


@dataclass
class AgentProposal:
    agent: str
    object_id: str
    score: float
    confidence: float
    rationale: str
    claim: str = ""
    evidence_summary: str = ""
    reason: str = ""
    counterfactual: str = ""
    needs_graft_with: str = ""
    role: str = ""
    bridge_to: str = ""
    mechanism: str = ""


@dataclass
class SharedMemory:
    proposals: List[AgentProposal] = field(default_factory=list)
    object_scores: Dict[str, float] = field(default_factory=dict)
    object_claims: Dict[str, List[Dict[str, str]]] = field(default_factory=dict)

    def record(self, proposals: List[AgentProposal]) -> None:
        self.proposals.extend(proposals)
        for proposal in proposals:
            blended = 0.65 * proposal.score + 0.35 * proposal.confidence
            self.object_scores[proposal.object_id] = self.object_scores.get(proposal.object_id, 0.0) + blended
            self.object_claims.setdefault(proposal.object_id, []).append({
                "agent": proposal.agent,
                "claim": proposal.claim or proposal.rationale,
                "evidence_summary": proposal.evidence_summary,
                "reason": proposal.reason,
            })

    def top_objects(self, k: int = 5) -> List[str]:
        ranked = sorted(self.object_scores.items(), key=lambda item: item[1], reverse=True)
        return [name for name, _ in ranked[:k]]

    def grouped_proposals(self) -> Dict[str, List[AgentProposal]]:
        grouped: Dict[str, List[AgentProposal]] = {}
        for proposal in self.proposals:
            grouped.setdefault(proposal.object_id, []).append(proposal)
        return grouped


class BaseAgent:
    name = "base"

    def propose(self, graph: ObjectGraph, memory: SharedMemory, top_k: int = 3) -> List[AgentProposal]:
        raise NotImplementedError


class TimelineAgent(BaseAgent):
    name = "timeline"

    def propose(self, graph: ObjectGraph, memory: SharedMemory, top_k: int = 3) -> List[AgentProposal]:
        timestamps = [node.earliest_timestamp for node in graph.nodes.values() if node.earliest_timestamp is not None]
        if not timestamps:
            return []
        t_min = min(timestamps)
        t_max = max(timestamps)
        span = max(1.0, t_max - t_min)
        scored = []
        for object_id, node in graph.nodes.items():
            ts = node.earliest_timestamp if node.earliest_timestamp is not None else t_max
            lead = 1.0 - (ts - t_min) / span
            score = 0.65 * lead + 0.35 * node.anomaly_score
            scored.append((object_id, score, lead))
        return _top_proposals(self.name, scored, "earliest anomaly with strong object-level activity", top_k)


class DependencyAgent(BaseAgent):
    name = "dependency"

    def propose(self, graph: ObjectGraph, memory: SharedMemory, top_k: int = 3) -> List[AgentProposal]:
        scored = []
        for object_id, node in graph.nodes.items():
            downstream = graph.topological_mass(object_id)
            incoming = graph.incoming_mass(object_id)
            rootness = max(0.0, downstream + 0.4 * node.trace_score - 0.35 * incoming)
            score = 0.50 * node.anomaly_score + 0.50 * _squash(rootness)
            scored.append((object_id, score, rootness))
        return _top_proposals(self.name, scored, "strong downstream explanatory power under dependency constraints", top_k)


class ChangeAgent(BaseAgent):
    name = "change"

    def propose(self, graph: ObjectGraph, memory: SharedMemory, top_k: int = 3) -> List[AgentProposal]:
        scored = []
        for object_id, node in graph.nodes.items():
            novelty = 0.0 if not memory.object_scores else max(0.0, 1.0 - memory.object_scores.get(object_id, 0.0) / max(memory.object_scores.values()))
            score = 0.70 * node.change_score + 0.20 * node.log_score + 0.10 * novelty
            scored.append((object_id, score, node.change_score))
        return _top_proposals(self.name, scored, "change-like evidence or restart/configuration signals", top_k)


class ResourceAgent(BaseAgent):
    name = "resource"

    def propose(self, graph: ObjectGraph, memory: SharedMemory, top_k: int = 3) -> List[AgentProposal]:
        scored = []
        for object_id, node in graph.nodes.items():
            resource_mass = max(node.metric_score, node.log_score, sum(node.reason_votes.values()) / max(1, len(node.reason_votes)))
            score = 0.60 * node.anomaly_score + 0.40 * _squash(resource_mass)
            scored.append((object_id, score, resource_mass))
        return _top_proposals(self.name, scored, "resource or failure-pattern evidence attached to the object", top_k)


def build_default_agents() -> List[BaseAgent]:
    return [TimelineAgent(), DependencyAgent(), ChangeAgent(), ResourceAgent()]


def _top_proposals(agent: str, scored, rationale: str, top_k: int) -> List[AgentProposal]:
    if not scored:
        return []
    filtered = [item for item in scored if item[1] > 0.05]
    if not filtered:
        return []
    values = np.array([max(0.0, item[1]) for item in filtered], dtype=float)
    confs = _normalize(values)
    ranked = sorted(zip(filtered, confs), key=lambda item: item[0][1], reverse=True)[:top_k]
    proposals = []
    for (object_id, score, signal), conf in ranked:
        proposals.append(AgentProposal(
            agent=agent,
            object_id=object_id,
            score=float(score),
            confidence=float(conf),
            rationale=f"{rationale}; signal={signal:.3f}",
                claim=f"{object_id} is suspicious from {agent} view",
                evidence_summary=f"{rationale}; signal={signal:.3f}",
        ))
    return proposals


def _normalize(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    vmax = float(values.max())
    vmin = float(values.min())
    if vmax <= vmin + 1e-9:
        return np.ones_like(values) * 0.5
    return (values - vmin) / (vmax - vmin + 1e-9)


def _squash(value: float) -> float:
    return float(np.tanh(max(0.0, value)))
