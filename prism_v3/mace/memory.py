"""Structured shared memory for full MACE-RCA."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import uuid

from .agents import AgentProposal
from .graph import EvidenceRecord, ObjectGraph


@dataclass
class MemoryEvidence:
    kind: str
    source: str
    content: str
    confidence: float
    timestamp: Optional[float] = None


@dataclass
class CounterfactualRecord:
    object_id: str
    summary: str
    removed_mass: float
    coverage: float
    exclusivity: float
    confidence: float
    created_by: str
    target_claim_id: str = ""
    verdict: str = ""


@dataclass
class MemoryClaim:
    claim_id: str
    object_id: str
    agent: str
    claim: str
    reason: str
    score: float
    confidence: float
    evidence: List[MemoryEvidence] = field(default_factory=list)
    evidence_summary: str = ""
    role: str = ""
    bridge_to: str = ""
    mechanism: str = ""
    symptom_score: float = 0.0
    counterfactuals: List[CounterfactualRecord] = field(default_factory=list)
    supports: List[str] = field(default_factory=list)
    contradicts: List[str] = field(default_factory=list)
    grafted_from: List[str] = field(default_factory=list)
    vitality: float = 1.0
    status: str = "active"
    round_idx: int = 0


@dataclass
class GraftRecord:
    source_claim_id: str
    target_claim_id: str
    source_object_id: str
    target_object_id: str
    graft_type: str
    compatibility: float
    summary: str
    evidence_overlap: float = 0.0
    topology_support: float = 0.0


@dataclass
class ClaimRelation:
    source_claim_id: str
    target_claim_id: str
    relation: str
    weight: float
    summary: str


@dataclass
class DebateRecord:
    object_a: str
    object_b: str
    pro_claim_id: str
    con_claim_id: str
    winner_object_id: str
    verdict: str
    confidence: float
    summary: str


class SharedMemoryGraph:
    """Shared memory graph with vitality, grafting and debate traces."""

    def __init__(self) -> None:
        self.proposals: List[AgentProposal] = []
        self.object_scores: Dict[str, float] = {}
        self.object_priors: Dict[str, Dict[str, float]] = {}
        self.object_claims: Dict[str, List[Dict[str, str]]] = defaultdict(list)
        self.claims: Dict[str, MemoryClaim] = {}
        self.claims_by_object: Dict[str, List[str]] = defaultdict(list)
        self.claims_by_agent: Dict[str, List[str]] = defaultdict(list)
        self.grafts: List[GraftRecord] = []
        self.debates: List[DebateRecord] = []
        self.claim_relations: List[ClaimRelation] = []
        self.claim_evidence_index: Dict[str, List[str]] = defaultdict(list)
        self.archived_claims: List[str] = []

    def record(
        self,
        proposals: List[AgentProposal],
        object_graph: Optional[ObjectGraph] = None,
        round_idx: int = 0,
    ) -> None:
        self.proposals.extend(proposals)
        for proposal in proposals:
            blended = 0.65 * proposal.score + 0.35 * proposal.confidence
            self.object_scores[proposal.object_id] = self.object_scores.get(proposal.object_id, 0.0) + blended
            self.object_claims[proposal.object_id].append({
                "agent": proposal.agent,
                "claim": proposal.claim or proposal.rationale,
                "evidence_summary": proposal.evidence_summary,
                "reason": proposal.reason,
                "role": proposal.role,
                "bridge_to": proposal.bridge_to,
                "mechanism": proposal.mechanism,
            })
            claim = self._claim_from_proposal(proposal, object_graph=object_graph, round_idx=round_idx)
            self._register_claim(claim)

    def top_objects(self, k: int = 5) -> List[str]:
        ranked = sorted(self.object_scores.items(), key=lambda item: item[1], reverse=True)
        return [name for name, _ in ranked[:k]]

    def grouped_proposals(self) -> Dict[str, List[AgentProposal]]:
        grouped: Dict[str, List[AgentProposal]] = defaultdict(list)
        for proposal in self.proposals:
            grouped[proposal.object_id].append(proposal)
        return dict(grouped)

    def apply_backbone_scores(self, scores: Dict[str, Dict[str, float]], prior_weight: float = 0.75) -> None:
        self.object_priors = {key: dict(value) for key, value in scores.items()}
        for object_id, payload in scores.items():
            rootness = float(payload.get("rootness_prior", 0.0))
            self.object_scores[object_id] = self.object_scores.get(object_id, 0.0) + prior_weight * rootness

    def object_packet(self, object_id: str) -> Dict[str, Any]:
        claim_ids = self.claims_by_object.get(object_id, [])
        claims = [self.claims[cid] for cid in claim_ids if cid in self.claims]
        active_claims = [claim for claim in claims if claim.status == "active"]
        return {
            "object_id": object_id,
            "score": round(self.object_scores.get(object_id, 0.0), 4),
            "claims": [self._claim_dict(claim) for claim in active_claims[:8]],
            "counterfactuals": [asdict(cf) for claim in active_claims for cf in claim.counterfactuals[:2]],
            "role_summary": self.object_role_summary(object_id),
            "symptom_score": self.object_symptom_score(object_id),
            "mechanism_summary": self.object_mechanism_summary(object_id),
            "backbone_prior": self.object_priors.get(object_id, {}),
            "grafts": [
                asdict(graft)
                for graft in self.grafts
                if graft.source_object_id == object_id or graft.target_object_id == object_id
            ][:6],
            "relations": [
                asdict(rel)
                for rel in self.claim_relations
                if (
                    self.claims.get(rel.source_claim_id)
                    and self.claims[rel.source_claim_id].object_id == object_id
                ) or (
                    self.claims.get(rel.target_claim_id)
                    and self.claims[rel.target_claim_id].object_id == object_id
                )
            ][:8],
            "debates": [
                asdict(debate)
                for debate in self.debates
                if debate.object_a == object_id or debate.object_b == object_id
            ][:4],
        }

    def suggest_grafts(self, object_graph: Optional[ObjectGraph], top_k: int = 6) -> List[Tuple[MemoryClaim, MemoryClaim, float, Dict[str, float]]]:
        suggestions: List[Tuple[MemoryClaim, MemoryClaim, float, Dict[str, float]]] = []
        same_object: List[Tuple[MemoryClaim, MemoryClaim, float, Dict[str, float]]] = []
        cross_object: List[Tuple[MemoryClaim, MemoryClaim, float, Dict[str, float]]] = []
        active_claims = [claim for claim in self.claims.values() if claim.status == "active"]
        for i in range(len(active_claims)):
            for j in range(i + 1, len(active_claims)):
                a = active_claims[i]
                b = active_claims[j]
                if a.agent == b.agent:
                    continue
                compatibility, detail = self._claim_similarity(a, b, object_graph)
                if compatibility >= 0.30:
                    packet = (a, b, compatibility, detail)
                    if a.object_id == b.object_id:
                        same_object.append(packet)
                    else:
                        cross_object.append(packet)
        same_object.sort(key=lambda item: item[2], reverse=True)
        cross_object.sort(key=lambda item: item[2], reverse=True)
        same_take = min(len(same_object), max(2, top_k // 2))
        cross_take = min(len(cross_object), max(2, top_k - same_take))
        suggestions.extend(same_object[:same_take])
        suggestions.extend(cross_object[:cross_take])
        if len(suggestions) < top_k:
            remaining = same_object[same_take:] + cross_object[cross_take:]
            remaining.sort(key=lambda item: item[2], reverse=True)
            suggestions.extend(remaining[: top_k - len(suggestions)])
        return suggestions[:top_k]

    def infer_claim_roles(self, object_graph: Optional[ObjectGraph]) -> None:
        if object_graph is None:
            return
        for claim in self.claims.values():
            if claim.status != "active":
                continue
            if claim.role not in {"cause", "propagation", "symptom"}:
                claim.role = self._infer_role(claim.object_id, object_graph)
            if not claim.mechanism:
                claim.mechanism = self._infer_mechanism(claim, object_graph)
            claim.symptom_score = self._infer_symptom_score(claim.object_id, object_graph, claim)

    def object_role_summary(self, object_id: str) -> Dict[str, float]:
        summary = {"cause": 0.0, "propagation": 0.0, "symptom": 0.0}
        for claim_id in self.claims_by_object.get(object_id, []):
            claim = self.claims.get(claim_id)
            if claim is None or claim.status != "active":
                continue
            role = claim.role if claim.role in summary else "symptom"
            summary[role] += 0.6 * claim.score + 0.4 * claim.confidence
        return {key: round(value, 4) for key, value in summary.items()}

    def object_symptom_score(self, object_id: str) -> float:
        scores = []
        for claim_id in self.claims_by_object.get(object_id, []):
            claim = self.claims.get(claim_id)
            if claim is None or claim.status != "active":
                continue
            scores.append(claim.symptom_score)
        return round(max(scores) if scores else 0.0, 4)

    def object_mechanism_summary(self, object_id: str) -> Dict[str, float]:
        summary: Dict[str, float] = defaultdict(float)
        for claim_id in self.claims_by_object.get(object_id, []):
            claim = self.claims.get(claim_id)
            if claim is None or claim.status != "active" or not claim.mechanism:
                continue
            summary[claim.mechanism] += 0.6 * claim.score + 0.4 * claim.confidence
        ranked = sorted(summary.items(), key=lambda item: item[1], reverse=True)[:4]
        return {key: round(value, 4) for key, value in ranked}

    def top_root_objects(self, object_graph: Optional[ObjectGraph], k: int = 5) -> List[str]:
        scored = []
        for object_id in self.object_scores:
            score = self.root_adjusted_score(object_id, object_graph)
            scored.append((object_id, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return [object_id for object_id, _ in scored[:k]]

    def hard_filter_candidates(
        self,
        object_graph: Optional[ObjectGraph],
        candidate_objects: List[str],
        symptom_threshold: float = 0.72,
        keep_top_k: int = 1,
    ) -> List[str]:
        if object_graph is None:
            return candidate_objects
        filtered: List[str] = []
        protected = set(self.top_root_objects(object_graph, keep_top_k))
        for object_id in candidate_objects:
            if object_id in protected:
                filtered.append(object_id)
                continue
            symptom_score = self.object_symptom_score(object_id)
            outgoing = object_graph.topological_mass(object_id)
            incoming = object_graph.incoming_mass(object_id)
            packet = self.object_packet(object_id)
            has_propagation = any(
                rel.get("relation") in {"causes", "propagates"} for rel in packet.get("relations", [])
            ) or any(
                claim.get("bridge_to") for claim in packet.get("claims", [])
            )
            prior = self.object_priors.get(object_id, {})
            uniqueness = float(prior.get("causal_uniqueness", 0.0))
            if symptom_score >= symptom_threshold and incoming >= outgoing and not has_propagation and uniqueness < 0.45:
                continue
            filtered.append(object_id)
        return filtered or candidate_objects[:keep_top_k]

    def root_adjusted_score(self, object_id: str, object_graph: Optional[ObjectGraph]) -> float:
        base = self.object_scores.get(object_id, 0.0)
        role = self.object_role_summary(object_id)
        bridge_bonus = 0.0
        bridge_penalty = 0.0
        symptom_penalty = self.object_symptom_score(object_id)
        prior = self.object_priors.get(object_id, {})
        for claim_id in self.claims_by_object.get(object_id, []):
            claim = self.claims.get(claim_id)
            if claim is None or claim.status != "active":
                continue
            if claim.bridge_to:
                bridge_bonus += 0.08 * claim.confidence
            if claim.mechanism and claim.role == "cause":
                bridge_bonus += 0.04 * claim.confidence
        if object_graph is not None:
            bridge_penalty += 0.06 * max(0.0, object_graph.incoming_mass(object_id) - object_graph.topological_mass(object_id))
        prior_bonus = (
            0.55 * float(prior.get("rootness_prior", 0.0))
            + 0.12 * float(prior.get("causal_uniqueness", 0.0))
            + 0.08 * float(prior.get("evidence_specificity", 0.0))
            + 0.05 * float(prior.get("exposure_signal", 0.0))
        )
        prior_penalty = (
            0.24 * float(prior.get("symptomness_prior", 0.0))
            + 0.14 * float(prior.get("popularity_bias", 0.0))
        )
        return (
            base
            + 0.24 * role["cause"]
            + 0.08 * role["propagation"]
            - 0.22 * role["symptom"]
            + bridge_bonus
            - bridge_penalty
            - 0.30 * symptom_penalty
            + prior_bonus
            - prior_penalty
        )

    def apply_symptom_suppression(self, object_graph: Optional[ObjectGraph]) -> List[Dict[str, Any]]:
        if object_graph is None:
            return []
        updates: List[Dict[str, Any]] = []
        for claim in self.claims.values():
            if claim.status != "active":
                continue
            claim.symptom_score = self._infer_symptom_score(claim.object_id, object_graph, claim)
            if claim.symptom_score < 0.58:
                continue
            has_bridge = bool(claim.bridge_to)
            outgoing = object_graph.topological_mass(claim.object_id)
            incoming = object_graph.incoming_mass(claim.object_id)
            if has_bridge or outgoing > incoming:
                continue
            penalty = min(0.22, 0.18 * claim.symptom_score)
            claim.confidence = max(0.0, claim.confidence - penalty)
            claim.score = max(0.0, claim.score - 0.8 * penalty)
            self.object_scores[claim.object_id] = max(0.0, self.object_scores.get(claim.object_id, 0.0) - 0.25 * claim.symptom_score)
            updates.append({
                "claim_id": claim.claim_id,
                "object_id": claim.object_id,
                "symptom_score": round(claim.symptom_score, 4),
                "penalty": round(penalty, 4),
                "summary": "suppressed as a high-intensity symptom without outbound propagation support",
            })
        return updates

    def create_bridge_claims(
        self,
        object_graph: Optional[ObjectGraph],
        max_bridges: int = 4,
    ) -> List[MemoryClaim]:
        if object_graph is None:
            return []
        created: List[MemoryClaim] = []
        candidates: List[Tuple[float, str, str]] = []
        for src, children in object_graph.adjacency.items():
            for dst, weight in children.items():
                if src not in object_graph.nodes or dst not in object_graph.nodes:
                    continue
                if src == dst:
                    continue
                bridge_score = float(weight) * (
                    0.45 * object_graph.nodes[src].anomaly_score
                    + 0.25 * object_graph.topological_mass(src)
                    + 0.15 * object_graph.nodes[dst].anomaly_score
                    + 0.15 * max(0.0, 1.0 - object_graph.incoming_mass(src))
                )
                if bridge_score > 0.10:
                    candidates.append((bridge_score, src, dst))
        candidates.sort(reverse=True)
        used = set()
        for bridge_score, src, dst in candidates:
            if len(created) >= max_bridges:
                break
            if (src, dst) in used:
                continue
            if any(
                claim.status == "active" and claim.bridge_to == dst and claim.object_id == src
                for claim in self.claims.values()
            ):
                continue
            source_claim_id = self.best_claim_id(src)
            target_claim_id = self.best_claim_id(dst)
            if not source_claim_id or not target_claim_id:
                continue
            source_claim = self.claims[source_claim_id]
            target_claim = self.claims[target_claim_id]
            source_role = source_claim.role or self._infer_role(src, object_graph)
            target_role = target_claim.role or self._infer_role(dst, object_graph)
            if source_role == "symptom" and target_role == "cause":
                continue
            weight = float(object_graph.adjacency.get(src, {}).get(dst, 0.0))
            mechanism = self._bridge_mechanism(source_claim, target_claim, object_graph)
            claim = MemoryClaim(
                claim_id=f"claim_{uuid.uuid4().hex[:12]}",
                object_id=src,
                agent="bridge",
                claim=f"{src} propagates {mechanism} pressure to {dst} along the causal graph.",
                reason="causal bridge",
                score=min(1.0, 0.45 + bridge_score),
                confidence=min(1.0, 0.40 + 0.8 * weight),
                evidence=[
                    MemoryEvidence(
                        kind="bridge",
                        source="causal_graph",
                        content=f"{src} -> {dst} edge weight={weight:.3f}; mechanism={mechanism}; source_role={source_role}; target_role={target_role}",
                        confidence=min(1.0, max(0.3, weight)),
                    )
                ],
                evidence_summary=f"Bridge edge {src}->{dst} with weight={weight:.3f} carries {mechanism} from a {source_role} claim to a {target_role} claim.",
                role="propagation",
                bridge_to=dst,
                mechanism=mechanism,
                symptom_score=max(0.0, target_claim.symptom_score - 0.15),
            )
            self._register_claim(claim)
            source_claim.supports.append(claim.claim_id)
            claim.supports.append(source_claim.claim_id)
            target_claim.supports.append(claim.claim_id)
            self.claim_relations.append(ClaimRelation(
                source_claim_id=source_claim.claim_id,
                target_claim_id=claim.claim_id,
                relation="supports",
                weight=min(1.0, 0.5 + 0.5 * weight),
                summary=f"{src} source claim supports {mechanism} propagation bridge to {dst}.",
            ))
            self.claim_relations.append(ClaimRelation(
                source_claim_id=claim.claim_id,
                target_claim_id=target_claim.claim_id,
                relation="propagates",
                weight=min(1.0, 0.4 + 0.6 * weight),
                summary=f"Bridge claim indicates {src} propagates {mechanism} impact to {dst}.",
            ))
            self.object_scores[src] = self.object_scores.get(src, 0.0) + 0.10 * claim.confidence
            self.object_scores[dst] = max(0.0, self.object_scores.get(dst, 0.0) - 0.04 * claim.confidence)
            created.append(claim)
            used.add((src, dst))
        return created

    def apply_graft(
        self,
        source_claim_id: str,
        target_claim_id: str,
        graft_type: str,
        compatibility: float,
        summary: str,
    ) -> None:
        if source_claim_id not in self.claims or target_claim_id not in self.claims:
            return
        source = self.claims[source_claim_id]
        target = self.claims[target_claim_id]
        target.grafted_from.append(source_claim_id)
        target.supports.append(source_claim_id)
        target.vitality = min(1.5, target.vitality + 0.10 * compatibility)
        target.score = min(1.0, target.score + 0.10 * compatibility)
        target.confidence = min(1.0, target.confidence + 0.08 * compatibility)
        self.object_scores[target.object_id] = self.object_scores.get(target.object_id, 0.0) + 0.12 * compatibility
        self.grafts.append(GraftRecord(
            source_claim_id=source_claim_id,
            target_claim_id=target_claim_id,
            source_object_id=source.object_id,
            target_object_id=target.object_id,
            graft_type=graft_type,
            compatibility=compatibility,
            summary=summary,
            evidence_overlap=self._claim_evidence_overlap(source, target),
            topology_support=1.0 if source.object_id == target.object_id else 0.0,
        ))
        self.claim_relations.append(ClaimRelation(
            source_claim_id=source_claim_id,
            target_claim_id=target_claim_id,
            relation="supports",
            weight=compatibility,
            summary=summary,
        ))

    def attach_counterfactual(self, object_id: str, cf: CounterfactualRecord) -> None:
        for claim_id in self.claims_by_object.get(object_id, []):
            if claim_id in self.claims and self.claims[claim_id].status == "active":
                self.claims[claim_id].counterfactuals.append(cf)

    def attach_claim_counterfactual(self, claim_id: str, cf: CounterfactualRecord, strengthen: bool) -> None:
        claim = self.claims.get(claim_id)
        if claim is None or claim.status != "active":
            return
        cf.target_claim_id = claim_id
        claim.counterfactuals.append(cf)
        delta = 0.12 * cf.confidence
        if strengthen:
            claim.confidence = min(1.0, claim.confidence + delta)
            claim.score = min(1.0, claim.score + 0.08 * cf.confidence)
            self.object_scores[claim.object_id] = self.object_scores.get(claim.object_id, 0.0) + 0.15 * cf.confidence
            verdict = "supports"
        else:
            claim.confidence = max(0.0, claim.confidence - delta)
            claim.score = max(0.0, claim.score - 0.10 * cf.confidence)
            self.object_scores[claim.object_id] = max(0.0, self.object_scores.get(claim.object_id, 0.0) - 0.18 * cf.confidence)
            verdict = "contradicts"
        cf.verdict = verdict

    def build_debate_pairs(self, candidate_objects: List[str], top_k: int = 2) -> List[Tuple[str, str, str, str]]:
        pairs = []
        for i in range(min(len(candidate_objects), top_k + 1)):
            for j in range(i + 1, min(len(candidate_objects), top_k + 1)):
                obj_a = candidate_objects[i]
                obj_b = candidate_objects[j]
                claim_a = self.best_claim_id(obj_a)
                claim_b = self.best_claim_id(obj_b)
                if claim_a and claim_b:
                    pairs.append((obj_a, obj_b, claim_a, claim_b))
        return pairs[:top_k]

    def record_debate(self, record: DebateRecord) -> None:
        self.debates.append(record)
        loser = record.object_b if record.winner_object_id == record.object_a else record.object_a
        self.object_scores[record.winner_object_id] = self.object_scores.get(record.winner_object_id, 0.0) + 0.20 * record.confidence
        self.object_scores[loser] = max(0.0, self.object_scores.get(loser, 0.0) - 0.15 * record.confidence)
        winner_claim = self.claims.get(record.pro_claim_id if self.claims.get(record.pro_claim_id, None) and self.claims[record.pro_claim_id].object_id == record.winner_object_id else record.con_claim_id)
        loser_claim = self.claims.get(record.con_claim_id if winner_claim and winner_claim.claim_id == record.pro_claim_id else record.pro_claim_id)
        if winner_claim:
            winner_claim.vitality = min(1.5, winner_claim.vitality + 0.15 * record.confidence)
        if loser_claim:
            loser_claim.vitality = max(0.05, loser_claim.vitality - 0.20 * record.confidence)
        if winner_claim and loser_claim:
            winner_claim.contradicts = [cid for cid in winner_claim.contradicts if cid != loser_claim.claim_id]
            loser_claim.contradicts.append(winner_claim.claim_id)
            self.claim_relations.append(ClaimRelation(
                source_claim_id=winner_claim.claim_id,
                target_claim_id=loser_claim.claim_id,
                relation="contradicts",
                weight=record.confidence,
                summary=record.summary,
            ))

    def decay_vitality(self, decay: float = 0.95, archive_threshold: float = 0.15) -> None:
        for claim in self.claims.values():
            if claim.status != "active":
                continue
            claim.vitality *= decay
            if claim.vitality < archive_threshold:
                claim.status = "archived"
                self.archived_claims.append(claim.claim_id)

    def best_claim_id(self, object_id: str) -> Optional[str]:
        candidates = [
            self.claims[cid]
            for cid in self.claims_by_object.get(object_id, [])
            if cid in self.claims and self.claims[cid].status == "active"
        ]
        if not candidates:
            return None
        best = max(candidates, key=lambda item: (item.score + item.confidence + 0.2 * item.vitality))
        return best.claim_id

    def snapshot(self) -> Dict[str, Any]:
        return {
            "object_scores": {key: round(value, 4) for key, value in self.object_scores.items()},
            "object_priors": {key: {k: round(v, 4) for k, v in value.items()} for key, value in self.object_priors.items()},
            "claims": {claim_id: self._claim_dict(claim) for claim_id, claim in self.claims.items()},
            "grafts": [asdict(graft) for graft in self.grafts],
            "claim_relations": [asdict(rel) for rel in self.claim_relations],
            "debates": [asdict(debate) for debate in self.debates],
            "archived_claims": list(self.archived_claims),
        }

    def _claim_from_proposal(
        self,
        proposal: AgentProposal,
        object_graph: Optional[ObjectGraph],
        round_idx: int,
    ) -> MemoryClaim:
        evidence = []
        if object_graph is not None and proposal.object_id in object_graph.nodes:
            node = object_graph.nodes[proposal.object_id]
            for item in node.evidence[:3]:
                evidence.append(self._convert_evidence(item))
        if proposal.evidence_summary:
            evidence.append(MemoryEvidence(
                kind="llm_summary",
                source=proposal.agent,
                content=proposal.evidence_summary,
                confidence=proposal.confidence,
            ))
        counterfactuals = []
        if proposal.counterfactual:
            counterfactuals.append(CounterfactualRecord(
                object_id=proposal.object_id,
                summary=proposal.counterfactual,
                removed_mass=0.0,
                coverage=0.0,
                exclusivity=0.0,
                confidence=proposal.confidence,
                created_by=proposal.agent,
            ))
        return MemoryClaim(
            claim_id=f"claim_{uuid.uuid4().hex[:12]}",
            object_id=proposal.object_id,
            agent=proposal.agent,
            claim=proposal.claim or proposal.rationale,
            reason=proposal.reason,
            score=proposal.score,
            confidence=proposal.confidence,
            evidence=evidence,
            evidence_summary=proposal.evidence_summary,
            role=proposal.role,
            bridge_to=proposal.bridge_to,
            mechanism=proposal.mechanism,
            counterfactuals=counterfactuals,
            round_idx=round_idx,
        )

    def _register_claim(self, claim: MemoryClaim) -> None:
        self.claims[claim.claim_id] = claim
        self.claims_by_object[claim.object_id].append(claim.claim_id)
        self.claims_by_agent[claim.agent].append(claim.claim_id)
        for evidence in claim.evidence:
            for key in self._evidence_keys(evidence):
                self.claim_evidence_index[key].append(claim.claim_id)

    def _convert_evidence(self, item: EvidenceRecord) -> MemoryEvidence:
        return MemoryEvidence(
            kind=item.kind,
            source=item.source,
            content=item.content,
            confidence=item.confidence,
            timestamp=item.timestamp,
        )

    def _claim_similarity(self, a: MemoryClaim, b: MemoryClaim, object_graph: Optional[ObjectGraph]) -> Tuple[float, Dict[str, float]]:
        score = 0.0
        detail = {
            "same_object": 1.0 if a.object_id == b.object_id else 0.0,
            "reason_match": 0.0,
            "token_overlap": 0.0,
            "evidence_overlap": 0.0,
            "topology_support": 0.0,
        }
        if a.object_id == b.object_id:
            score += 0.28
        if a.reason and b.reason and a.reason.lower() == b.reason.lower():
            score += 0.18
            detail["reason_match"] = 1.0
        token_overlap = len(set(_tokens(a.claim)) & set(_tokens(b.claim)))
        token_union = max(1, len(set(_tokens(a.claim)) | set(_tokens(b.claim))))
        token_ratio = token_overlap / token_union
        detail["token_overlap"] = token_ratio
        score += 0.22 * token_ratio
        evidence_overlap = self._claim_evidence_overlap(a, b)
        detail["evidence_overlap"] = evidence_overlap
        score += 0.20 * evidence_overlap
        topology_support = self._topology_support(a.object_id, b.object_id, object_graph)
        detail["topology_support"] = topology_support
        score += 0.12 * topology_support
        return min(1.0, score), detail

    def _claim_evidence_overlap(self, a: MemoryClaim, b: MemoryClaim) -> float:
        keys_a = set()
        keys_b = set()
        for evidence in a.evidence:
            keys_a.update(self._evidence_keys(evidence))
        for evidence in b.evidence:
            keys_b.update(self._evidence_keys(evidence))
        if not keys_a or not keys_b:
            return 0.0
        return len(keys_a & keys_b) / max(1, len(keys_a | keys_b))

    def _topology_support(self, source_object: str, target_object: str, object_graph: Optional[ObjectGraph]) -> float:
        if object_graph is None:
            return 0.0
        if source_object == target_object:
            return 1.0
        if target_object in object_graph.adjacency.get(source_object, {}):
            return 0.9
        if source_object in object_graph.adjacency.get(target_object, {}):
            return 0.6
        one_hop = object_graph.adjacency.get(source_object, {})
        for mid in one_hop:
            if target_object in object_graph.adjacency.get(mid, {}):
                return 0.45
        return 0.0

    def _claim_dict(self, claim: MemoryClaim) -> Dict[str, Any]:
        return {
            "claim_id": claim.claim_id,
            "object_id": claim.object_id,
            "agent": claim.agent,
            "claim": claim.claim,
            "reason": claim.reason,
            "score": round(claim.score, 4),
            "confidence": round(claim.confidence, 4),
            "evidence_summary": claim.evidence_summary,
            "role": claim.role,
            "bridge_to": claim.bridge_to,
            "mechanism": claim.mechanism,
            "symptom_score": round(claim.symptom_score, 4),
            "evidence": [asdict(item) for item in claim.evidence[:4]],
            "counterfactuals": [asdict(item) for item in claim.counterfactuals[:2]],
            "supports": list(claim.supports),
            "contradicts": list(claim.contradicts),
            "grafted_from": list(claim.grafted_from),
            "vitality": round(claim.vitality, 4),
            "status": claim.status,
            "round_idx": claim.round_idx,
        }

    def _evidence_keys(self, evidence: MemoryEvidence) -> List[str]:
        keys = []
        if evidence.source:
            keys.append(f"src:{str(evidence.source).lower()}")
        if evidence.kind:
            keys.append(f"kind:{str(evidence.kind).lower()}")
        tokens = _tokens(evidence.content)[:8]
        keys.extend(f"tok:{token}" for token in tokens)
        return keys

    def _infer_role(self, object_id: str, object_graph: ObjectGraph) -> str:
        downstream = object_graph.topological_mass(object_id)
        incoming = object_graph.incoming_mass(object_id)
        node = object_graph.nodes.get(object_id)
        anomaly = node.anomaly_score if node is not None else 0.0
        if downstream > incoming + 0.20 and anomaly >= 0.25:
            return "cause"
        if incoming > downstream + 0.15:
            return "symptom"
        return "propagation"

    def _infer_mechanism(self, claim: MemoryClaim, object_graph: ObjectGraph) -> str:
        text = " ".join([
            claim.reason or "",
            claim.claim or "",
            claim.evidence_summary or "",
            " ".join(item.content for item in claim.evidence[:3]),
        ]).lower()
        mapping = {
            "cpu": "cpu_pressure",
            "memory": "memory_pressure",
            "network": "network_latency",
            "latency": "network_latency",
            "disk": "disk_io",
            "db": "db_backpressure",
            "mysql": "db_backpressure",
            "redis": "cache_backpressure",
            "change": "change_trigger",
        }
        for key, mechanism in mapping.items():
            if key in text:
                return mechanism
        node = object_graph.nodes.get(claim.object_id)
        if node is not None:
            reason = node.best_reason().lower()
            for key, mechanism in mapping.items():
                if key in reason:
                    return mechanism
        return "generic_propagation"

    def _bridge_mechanism(self, source_claim: MemoryClaim, target_claim: MemoryClaim, object_graph: ObjectGraph) -> str:
        if source_claim.mechanism and source_claim.mechanism != "generic_propagation":
            return source_claim.mechanism
        if target_claim.mechanism and target_claim.mechanism != "generic_propagation":
            return target_claim.mechanism
        return self._infer_mechanism(source_claim, object_graph)

    def _infer_symptom_score(self, object_id: str, object_graph: ObjectGraph, claim: Optional[MemoryClaim] = None) -> float:
        downstream = object_graph.topological_mass(object_id)
        incoming = object_graph.incoming_mass(object_id)
        node = object_graph.nodes.get(object_id)
        anomaly = node.anomaly_score if node is not None else 0.0
        bridge_support = 0.0
        if claim is not None and claim.bridge_to:
            bridge_support = 0.30
        propagation_bonus = 0.20 if claim is not None and claim.role == "propagation" else 0.0
        cause_bonus = 0.15 if claim is not None and claim.role == "cause" else 0.0
        raw = 0.55 * max(0.0, incoming - downstream) + 0.25 * anomaly + 0.20 * max(0.0, incoming)
        suppressed = raw - bridge_support - propagation_bonus - cause_bonus
        return max(0.0, min(1.0, suppressed))


def _tokens(text: str) -> List[str]:
    return [token for token in (text or "").lower().replace(",", " ").replace(".", " ").split() if token]
