"""Full MACE-RCA pipeline with memory, grafting and debate."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Dict, List, Optional

from ..config import QueryCase, UnifiedTelemetry
from ..utils.llm_client import LLMClient
from .agents import AgentProposal, build_default_agents
from .graph import ObjectGraph, build_object_graph, format_timestamp
from .intervention import rank_by_local_intervention
from .memory import CounterfactualRecord, DebateRecord, SharedMemoryGraph
from .prism_backbone import score_prism_backbone


@dataclass
class MACEConfig:
    max_rounds: int = 2
    proposals_per_agent: int = 3
    intervention_top_k: int = 5
    final_intervention_weight: float = 0.35
    final_memory_weight: float = 0.65
    llm_agent_context_top_k: int = 12
    llm_controller_top_k: int = 5
    graft_suggestion_top_k: int = 6
    bridge_top_k: int = 4
    debate_pairs: int = 2
    controller_self_consistency: int = 2
    final_self_consistency: int = 3
    tournament_top_k: int = 3
    symptom_hard_threshold: float = 0.72
    vitality_decay: float = 0.96
    vitality_archive_threshold: float = 0.10
    llm_model: str = "deepseek-chat"
    llm_temperature: float = 0.1


class MACEPipeline:
    """Multi-Agent Cognitive Ecosystem for RCA."""

    def __init__(
        self,
        system_name: str,
        config: Optional[MACEConfig] = None,
        llm_client: Optional[LLMClient] = None,
    ):
        self.system_name = system_name
        self.config = config or MACEConfig()
        self.agents = build_default_agents()
        self.llm_client = llm_client

    def run(
        self,
        telemetry: UnifiedTelemetry,
        query: QueryCase,
        inject_time: float,
    ) -> Dict[str, Any]:
        object_graph, graph_debug = build_object_graph(telemetry, query, inject_time)
        if not object_graph.nodes:
            return self._empty_result(query, "no_object_nodes")

        memory = SharedMemoryGraph()
        trace: List[Dict[str, Any]] = []
        llm_usage = {"api_calls": 0, "total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0}
        backbone_bundle = score_prism_backbone(object_graph)
        memory.apply_backbone_scores(backbone_bundle.get("scores", {}))

        if self.llm_client is not None:
            agent_debug, agent_usage = self._run_llm_agents(query, object_graph, memory, trace)
            llm_usage = self._merge_usage(llm_usage, agent_usage)
        else:
            agent_debug = {"enabled": False, "fallback": "heuristic_agents"}
            self._run_heuristic_agents(object_graph, memory, trace)
        memory.infer_claim_roles(object_graph)

        bridge_debug = self._run_bridge_generation(object_graph, memory, trace)
        suppression_debug = self._run_symptom_suppression(object_graph, memory, trace)

        graft_debug, graft_usage = self._run_grafting(query, object_graph, memory, trace)
        llm_usage = self._merge_usage(llm_usage, graft_usage)
        memory.infer_claim_roles(object_graph)

        controller_debug, controller_usage = self._llm_controller_candidates(query, object_graph, memory)
        llm_usage = self._merge_usage(llm_usage, controller_usage)
        candidate_objects = controller_debug.get("candidate_list") or memory.top_root_objects(object_graph, self.config.llm_controller_top_k)
        candidate_objects = memory.hard_filter_candidates(
            object_graph,
            candidate_objects,
            symptom_threshold=self.config.symptom_hard_threshold,
            keep_top_k=1,
        )
        if not candidate_objects:
            candidate_objects = self._fallback_candidates(object_graph)

        intervention_rank = rank_by_local_intervention(
            object_graph,
            candidate_objects,
            top_k=self.config.intervention_top_k,
        )
        self._attach_counterfactuals(memory, intervention_rank)
        cf_debug, cf_usage = self._run_claim_counterfactual_validation(
            query=query,
            object_graph=object_graph,
            memory=memory,
            candidate_objects=candidate_objects,
            intervention_rank=intervention_rank,
            trace=trace,
        )
        llm_usage = self._merge_usage(llm_usage, cf_usage)

        debate_debug, debate_usage = self._run_debates(query, object_graph, memory, candidate_objects, intervention_rank, trace)
        llm_usage = self._merge_usage(llm_usage, debate_usage)

        final_scores = self._merge_scores(memory, intervention_rank, object_graph)
        if not final_scores:
            return self._empty_result(query, "empty_final_scores")
        candidate_objects = [item[0] for item in sorted(final_scores.items(), key=lambda item: item[1], reverse=True)[:self.config.llm_controller_top_k]]
        tournament_debug, tournament_usage, candidate_objects = self._run_pairwise_tournament(
            query=query,
            object_graph=object_graph,
            memory=memory,
            candidate_objects=candidate_objects,
            intervention_rank=intervention_rank,
            trace=trace,
        )
        llm_usage = self._merge_usage(llm_usage, tournament_usage)
        candidate_objects = memory.hard_filter_candidates(
            object_graph,
            candidate_objects,
            symptom_threshold=self.config.symptom_hard_threshold,
            keep_top_k=1,
        )

        final_llm, final_usage = self._llm_final_decision(
            query=query,
            object_graph=object_graph,
            memory=memory,
            candidate_objects=candidate_objects,
            intervention_rank=intervention_rank,
            final_scores=final_scores,
        )
        llm_usage = self._merge_usage(llm_usage, final_usage)

        best_object = final_llm.get("chosen_object_id")
        if best_object not in object_graph.nodes:
            best_object = max(final_scores, key=final_scores.get)
        best_node = object_graph.nodes[best_object]
        occurrence_time = final_llm.get("time") or format_timestamp(best_node.earliest_timestamp, query.time_window[0])
        if not self._looks_like_time(occurrence_time):
            occurrence_time = format_timestamp(best_node.earliest_timestamp, query.time_window[0])
        reason = final_llm.get("reason") or best_node.best_reason()
        top_score = max(float(final_scores.get(best_object, 0.0)), float(final_llm.get("confidence", 0.0) or 0.0))

        final_candidates = [
            {
                "object_id": object_id,
                "entity": object_graph.nodes[object_id].representative,
                "score": round(score, 4),
            }
            for object_id, score in sorted(final_scores.items(), key=lambda item: item[1], reverse=True)[:5]
        ]
        return {
            "prediction": {
                "component": [best_node.representative],
                "reason": [reason],
                "time": [occurrence_time],
                "top_score": top_score,
            },
            "trace": trace,
            "stop_reason": "mace_deliberation_complete",
            "initial_candidates": [
                {
                    "object_id": object_id,
                    "entity": object_graph.nodes[object_id].representative,
                    "score": round(memory.object_scores.get(object_id, 0.0), 4),
                }
                for object_id in candidate_objects[:5]
            ],
            "final_candidates": final_candidates,
            "debug": {
                "graph": _graph_debug_payload(object_graph),
                "graph_debug": graph_debug,
                "prism_backbone": backbone_bundle.get("debug", {}),
                "memory_scores": {k: round(v, 4) for k, v in memory.object_scores.items()},
                "intervention": intervention_rank,
                "llm_agents": agent_debug,
                "bridge_generation": bridge_debug,
                "symptom_suppression": suppression_debug,
                "grafting": graft_debug,
                "llm_controller": controller_debug,
                "claim_counterfactual": cf_debug,
                "debates": debate_debug,
                "tournament": tournament_debug,
                "llm_final": final_llm,
                "memory_snapshot": memory.snapshot(),
                "grouped_proposals": {
                    object_id: [
                        {
                            "agent": proposal.agent,
                            "score": round(proposal.score, 4),
                            "confidence": round(proposal.confidence, 4),
                            "rationale": proposal.rationale,
                            "claim": proposal.claim,
                            "evidence_summary": proposal.evidence_summary,
                            "reason": proposal.reason,
                            "counterfactual": proposal.counterfactual,
                            "needs_graft_with": proposal.needs_graft_with,
                            "role": proposal.role,
                            "bridge_to": proposal.bridge_to,
                        }
                        for proposal in proposals
                    ]
                    for object_id, proposals in memory.grouped_proposals().items()
                },
            },
            "llm_usage": llm_usage,
        }

    def _merge_scores(self, memory: SharedMemoryGraph, intervention_rank: List[Dict[str, Any]], object_graph: ObjectGraph) -> Dict[str, float]:
        merged: Dict[str, float] = {}
        intervention_scores = {item["object_id"]: item["score"] for item in intervention_rank}
        adjusted_scores = {object_id: memory.root_adjusted_score(object_id, object_graph) for object_id in memory.object_scores}
        mem_max = max(adjusted_scores.values()) if adjusted_scores else 1.0
        int_max = max(intervention_scores.values()) if intervention_scores else 1.0
        all_objects = set(memory.object_scores) | set(intervention_scores)
        for object_id in all_objects:
            mem_score = adjusted_scores.get(object_id, memory.object_scores.get(object_id, 0.0)) / max(mem_max, 1e-6)
            int_score = intervention_scores.get(object_id, 0.0) / max(int_max, 1e-6)
            merged[object_id] = (
                self.config.final_memory_weight * mem_score
                + self.config.final_intervention_weight * int_score
            )
        return merged

    def _empty_result(self, query: QueryCase, stop_reason: str) -> Dict[str, Any]:
        return {
            "prediction": {
                "component": [""],
                "reason": ["high memory usage"],
                "time": [query.time_window[0]],
                "top_score": 0.0,
            },
            "trace": [],
            "stop_reason": stop_reason,
            "initial_candidates": [],
            "final_candidates": [],
            "debug": {},
            "llm_usage": {},
        }

    def _fallback_candidates(self, object_graph: ObjectGraph) -> List[str]:
        return [
            object_id
            for object_id, _ in sorted(
                object_graph.nodes.items(),
                key=lambda item: item[1].anomaly_score,
                reverse=True,
            )[:self.config.intervention_top_k]
        ]

    def _run_heuristic_agents(
        self,
        object_graph: ObjectGraph,
        memory: SharedMemoryGraph,
        trace: List[Dict[str, Any]],
    ) -> None:
        for round_idx in range(self.config.max_rounds):
            for agent in self.agents:
                proposals = agent.propose(object_graph, memory, self.config.proposals_per_agent)
                memory.record(proposals, object_graph=object_graph, round_idx=round_idx)
                trace.append({
                    "step": len(trace),
                    "action": f"AGENT_{agent.name.upper()}",
                    "round": round_idx,
                    "top_objects": [p.object_id for p in proposals[:3]],
                    "rationales": [p.rationale for p in proposals[:2]],
                })
            memory.decay_vitality(self.config.vitality_decay, self.config.vitality_archive_threshold)

    def _run_llm_agents(
        self,
        query: QueryCase,
        object_graph: ObjectGraph,
        memory: SharedMemoryGraph,
        trace: List[Dict[str, Any]],
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        debug = {"enabled": True, "agents": {}}
        usage = {"api_calls": 0, "total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0}
        for round_idx in range(self.config.max_rounds):
            for agent in self.agents:
                proposals, agent_debug, agent_usage = self._llm_agent_proposals(
                    agent_name=agent.name,
                    query=query,
                    object_graph=object_graph,
                    memory=memory,
                )
                if not proposals:
                    proposals = agent.propose(object_graph, memory, self.config.proposals_per_agent)
                    agent_debug["fallback"] = "heuristic"
                proposals = self._augment_with_backbone_candidates(
                    agent_name=agent.name,
                    proposals=proposals,
                    object_graph=object_graph,
                    memory=memory,
                )
                memory.record(proposals, object_graph=object_graph, round_idx=round_idx)
                usage = self._merge_usage(usage, agent_usage)
                debug["agents"].setdefault(agent.name, []).append(agent_debug)
                trace.append({
                    "step": len(trace),
                    "action": f"AGENT_{agent.name.upper()}",
                    "round": round_idx,
                    "top_objects": [p.object_id for p in proposals[:3]],
                    "rationales": [p.claim or p.rationale for p in proposals[:2]],
                })
            memory.decay_vitality(self.config.vitality_decay, self.config.vitality_archive_threshold)
        return debug, usage

    def _llm_agent_proposals(
        self,
        agent_name: str,
        query: QueryCase,
        object_graph: ObjectGraph,
        memory: SharedMemoryGraph,
    ) -> tuple[List[AgentProposal], Dict[str, Any], Dict[str, Any]]:
        cards = self._object_cards(object_graph, memory)[:self.config.llm_agent_context_top_k]
        prior_focus = memory.top_root_objects(object_graph, 5) if hasattr(memory, "top_root_objects") else memory.top_objects(5)
        system_prompt = (
            f"You are the {agent_name} agent in a Multi-Agent Cognitive Ecosystem for root cause analysis. "
            "Produce structured claims from your perspective. "
            "Each claim must name one object, cite evidence, include one short counterfactual note, "
            "and label the role as cause, propagation, or symptom. "
            "If the object likely propagates impact to a downstream object, set bridge_to. "
            "Provide a short mechanism label such as cpu_pressure, db_backpressure, network_latency, disk_io, or change_trigger. "
            "Use the backbone priors carefully: exposure_signal favors robust abnormal signals, evidence_specificity favors object-specific evidence, causal_uniqueness favors irreplaceable recovery, and symptomness_prior penalizes likely symptom buckets. "
            "Return strict JSON only."
        )
        user_prompt = {
            "query_instruction": query.instruction,
            "time_window": list(query.time_window),
            "agent_view": agent_name,
            "prior_focus_objects": prior_focus,
            "candidate_objects": cards,
            "output_schema": {
                "proposals": [
                    {
                        "object_id": "candidate object_id",
                        "score": "0-1 float",
                        "confidence": "0-1 float",
                        "claim": "one-sentence hypothesis",
                        "reason": "short reason label",
                        "evidence_summary": "short structured evidence",
                        "counterfactual": "if this object were not faulty, what would recover or not happen",
                        "needs_graft_with": "optional object_id to graft from",
                        "role": "cause|propagation|symptom",
                        "bridge_to": "optional downstream object_id",
                        "mechanism": "short mechanism label",
                    }
                ]
            },
        }
        resp = self.llm_client.call(
            model=self.config.llm_model,
            messages=[{"role": "user", "content": json.dumps(user_prompt, ensure_ascii=True)}],
            system=system_prompt,
            response_format={"type": "json_object"},
            max_tokens=700,
            temperature=self.config.llm_temperature,
        )
        empty_usage = self._empty_usage()
        if resp is None:
            return [], {"success": False, "error": "llm_call_failed"}, empty_usage
        payload = resp.content if isinstance(resp.content, dict) else {}
        raw_items = payload.get("proposals", [])
        if isinstance(payload.get("object_id"), str):
            raw_items = [payload]
        proposals = []
        for item in raw_items[:self.config.proposals_per_agent]:
            object_id = str(item.get("object_id", "")).strip()
            if object_id not in object_graph.nodes:
                continue
            proposals.append(AgentProposal(
                agent=agent_name,
                object_id=object_id,
                score=self._clamp_float(item.get("score"), default=0.5),
                confidence=self._clamp_float(item.get("confidence"), default=0.5),
                rationale=str(item.get("claim", "") or item.get("evidence_summary", "")),
                claim=str(item.get("claim", "")).strip(),
                evidence_summary=str(item.get("evidence_summary", "")).strip(),
                reason=str(item.get("reason", "")).strip(),
                counterfactual=str(item.get("counterfactual", "")).strip(),
                needs_graft_with=str(item.get("needs_graft_with", "")).strip(),
                role=str(item.get("role", "")).strip().lower(),
                bridge_to=str(item.get("bridge_to", "")).strip().lower(),
                mechanism=str(item.get("mechanism", "")).strip().lower(),
            ))
        return proposals, {"success": True, "payload": payload}, self._usage_from_response(resp.usage)

    def _augment_with_backbone_candidates(
        self,
        agent_name: str,
        proposals: List[AgentProposal],
        object_graph: ObjectGraph,
        memory: SharedMemoryGraph,
    ) -> List[AgentProposal]:
        existing = {proposal.object_id for proposal in proposals}
        unique_target = min(self.config.proposals_per_agent, 2)
        if len(existing) >= unique_target:
            return proposals[: self.config.proposals_per_agent]
        augmented = list(proposals)
        for object_id in memory.top_root_objects(object_graph, self.config.llm_agent_context_top_k):
            if object_id in existing or object_id not in object_graph.nodes:
                continue
            prior = memory.object_priors.get(object_id, {})
            node = object_graph.nodes[object_id]
            symptom = float(prior.get("symptomness_prior", 0.5))
            role = "cause" if symptom <= 0.35 else ("propagation" if symptom <= 0.60 else "symptom")
            augmented.append(AgentProposal(
                agent=agent_name,
                object_id=object_id,
                score=max(0.15, float(prior.get("rootness_prior", 0.25))),
                confidence=max(0.15, float(prior.get("causal_uniqueness", 0.20))),
                rationale=f"Backbone prior highlights {node.representative} via exposure-normalized signal and causal uniqueness.",
                claim=f"{node.representative} is a high-priority backbone candidate with {node.best_reason().lower()} evidence.",
                evidence_summary=f"rootness={prior.get('rootness_prior', 0.0):.3f}, specificity={prior.get('evidence_specificity', 0.0):.3f}, uniqueness={prior.get('causal_uniqueness', 0.0):.3f}",
                reason=node.best_reason(),
                counterfactual=f"If {node.representative} were not causal, its backbone uniqueness and recovery contribution would drop.",
                role=role,
                mechanism=self._reason_to_mechanism(node.best_reason()),
            ))
            existing.add(object_id)
            if len(existing) >= unique_target or len(augmented) >= self.config.proposals_per_agent:
                break
        return augmented[: self.config.proposals_per_agent]

    def _run_bridge_generation(
        self,
        object_graph: ObjectGraph,
        memory: SharedMemoryGraph,
        trace: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        created = memory.create_bridge_claims(object_graph, max_bridges=self.config.bridge_top_k)
        debug = {"created": []}
        for claim in created:
            debug["created"].append({
                "claim_id": claim.claim_id,
                "object_id": claim.object_id,
                "bridge_to": claim.bridge_to,
                "role": claim.role,
                "score": round(claim.score, 4),
                "confidence": round(claim.confidence, 4),
                "claim": claim.claim,
            })
            trace.append({
                "step": len(trace),
                "action": "BRIDGE",
                "claim_id": claim.claim_id,
                "source_object": claim.object_id,
                "target_object": claim.bridge_to,
                "summary": claim.claim,
            })
        return debug

    def _run_symptom_suppression(
        self,
        object_graph: ObjectGraph,
        memory: SharedMemoryGraph,
        trace: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        updates = memory.apply_symptom_suppression(object_graph)
        for item in updates:
            trace.append({
                "step": len(trace),
                "action": "SYMPTOM_SUPPRESS",
                "claim_id": item["claim_id"],
                "object_id": item["object_id"],
                "summary": item["summary"],
            })
        return {"updates": updates}

    def _run_grafting(
        self,
        query: QueryCase,
        object_graph: ObjectGraph,
        memory: SharedMemoryGraph,
        trace: List[Dict[str, Any]],
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        suggestions = memory.suggest_grafts(object_graph, self.config.graft_suggestion_top_k)
        debug = {"enabled": self.llm_client is not None, "suggestions": [], "applied": []}
        usage = self._empty_usage()
        for source_claim, target_claim, compatibility, detail in suggestions:
            decision = {
                "apply": compatibility > 0.55,
                "graft_type": "reference",
                "summary": "heuristic graft from aligned same-object claims",
                "compatibility": compatibility,
            }
            if self.llm_client is not None:
                decision, decision_usage = self._llm_graft_decision(query, object_graph, source_claim, target_claim, compatibility)
                usage = self._merge_usage(usage, decision_usage)
            debug["suggestions"].append({
                "source_claim_id": source_claim.claim_id,
                "target_claim_id": target_claim.claim_id,
                "compatibility": round(compatibility, 4),
                "detail": {k: round(v, 4) for k, v in detail.items()},
                "decision": decision,
            })
            if decision.get("apply"):
                memory.apply_graft(
                    source_claim_id=source_claim.claim_id,
                    target_claim_id=target_claim.claim_id,
                    graft_type=decision.get("graft_type", "reference"),
                    compatibility=compatibility,
                    summary=decision.get("summary", ""),
                )
                debug["applied"].append({
                    "source_claim_id": source_claim.claim_id,
                    "target_claim_id": target_claim.claim_id,
                    "graft_type": decision.get("graft_type", "reference"),
                    "summary": decision.get("summary", ""),
                })
                trace.append({
                    "step": len(trace),
                    "action": "GRAFT",
                    "source_claim_id": source_claim.claim_id,
                    "target_claim_id": target_claim.claim_id,
                    "summary": decision.get("summary", ""),
                })
        return debug, usage

    def _run_claim_counterfactual_validation(
        self,
        query: QueryCase,
        object_graph: ObjectGraph,
        memory: SharedMemoryGraph,
        candidate_objects: List[str],
        intervention_rank: List[Dict[str, Any]],
        trace: List[Dict[str, Any]],
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        debug = {"enabled": self.llm_client is not None, "validations": []}
        usage = self._empty_usage()
        intervention_map = {item["object_id"]: item for item in intervention_rank}
        for object_id in candidate_objects[: self.config.llm_controller_top_k]:
            claim_id = memory.best_claim_id(object_id)
            if not claim_id:
                continue
            claim = memory.claims.get(claim_id)
            if claim is None:
                continue
            if self.llm_client is None:
                strengthen = intervention_map.get(object_id, {}).get("coverage", 0.0) >= 0.55
                confidence = float(intervention_map.get(object_id, {}).get("score", 0.5))
                summary = "fallback claim-level counterfactual validation"
            else:
                result, cf_usage = self._llm_claim_counterfactual(
                    query=query,
                    object_graph=object_graph,
                    memory=memory,
                    claim_id=claim_id,
                    intervention=intervention_map.get(object_id, {}),
                )
                usage = self._merge_usage(usage, cf_usage)
                strengthen = bool(result.get("supports_claim", False))
                confidence = self._clamp_float(result.get("confidence"), default=0.5)
                summary = str(result.get("summary", "")).strip()
            cf = CounterfactualRecord(
                object_id=object_id,
                target_claim_id=claim_id,
                summary=summary,
                removed_mass=float(intervention_map.get(object_id, {}).get("removed_mass", 0.0)),
                coverage=float(intervention_map.get(object_id, {}).get("coverage", 0.0)),
                exclusivity=float(intervention_map.get(object_id, {}).get("exclusivity", 0.0)),
                confidence=confidence,
                created_by="claim_counterfactual",
            )
            memory.attach_claim_counterfactual(claim_id, cf, strengthen=strengthen)
            debug["validations"].append({
                "claim_id": claim_id,
                "object_id": object_id,
                "supports_claim": strengthen,
                "confidence": round(confidence, 4),
                "summary": summary,
            })
            trace.append({
                "step": len(trace),
                "action": "CLAIM_COUNTERFACTUAL",
                "claim_id": claim_id,
                "object_id": object_id,
                "supports_claim": strengthen,
                "summary": summary,
            })
        return debug, usage

    def _llm_claim_counterfactual(
        self,
        query: QueryCase,
        object_graph: ObjectGraph,
        memory: SharedMemoryGraph,
        claim_id: str,
        intervention: Dict[str, Any],
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        claim = memory.claims[claim_id]
        system_prompt = (
            "You are a counterfactual validator in MACE-RCA. "
            "Judge whether the local intervention evidence strengthens or weakens this specific claim. "
            "Return strict JSON only."
        )
        user_prompt = {
            "query_instruction": query.instruction,
            "claim": memory.object_packet(claim.object_id),
            "focus_claim_id": claim_id,
            "focus_claim": self._claim_packet(claim, object_graph),
            "intervention": intervention,
            "output_schema": {
                "supports_claim": "boolean",
                "confidence": "0-1 float",
                "summary": "short verdict explaining why the counterfactual supports or weakens the claim",
            },
        }
        resp = self.llm_client.call(
            model=self.config.llm_model,
            messages=[{"role": "user", "content": json.dumps(user_prompt, ensure_ascii=True)}],
            system=system_prompt,
            response_format={"type": "json_object"},
            max_tokens=260,
            temperature=self.config.llm_temperature,
        )
        if resp is None:
            return {"supports_claim": False, "confidence": 0.5, "summary": "llm call failed"}, self._empty_usage()
        payload = resp.content if isinstance(resp.content, dict) else {}
        return payload, self._usage_from_response(resp.usage)

    def _llm_graft_decision(
        self,
        query: QueryCase,
        object_graph: ObjectGraph,
        source_claim,
        target_claim,
        compatibility: float,
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        system_prompt = (
            "You are a graft compatibility judge in MACE-RCA. "
            "Decide whether the target claim should graft from the source claim. "
            "Return strict JSON only."
        )
        user_prompt = {
            "query_instruction": query.instruction,
            "source": self._claim_packet(source_claim, object_graph),
            "target": self._claim_packet(target_claim, object_graph),
            "compatibility": round(compatibility, 4),
            "output_schema": {
                "apply": "boolean",
                "graft_type": "reference|absorb|merge",
                "summary": "short justification",
            },
        }
        resp = self.llm_client.call(
            model=self.config.llm_model,
            messages=[{"role": "user", "content": json.dumps(user_prompt, ensure_ascii=True)}],
            system=system_prompt,
            response_format={"type": "json_object"},
            max_tokens=220,
            temperature=self.config.llm_temperature,
        )
        if resp is None:
            return {"apply": compatibility > 0.55, "graft_type": "reference", "summary": "fallback heuristic"}, self._empty_usage()
        payload = resp.content if isinstance(resp.content, dict) else {}
        return {
            "apply": bool(payload.get("apply", False)),
            "graft_type": str(payload.get("graft_type", "reference")).strip() or "reference",
            "summary": str(payload.get("summary", "")).strip(),
        }, self._usage_from_response(resp.usage)

    def _llm_controller_candidates(
        self,
        query: QueryCase,
        object_graph: ObjectGraph,
        memory: SharedMemoryGraph,
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        if self.llm_client is None:
            return {
                "enabled": False,
                "success": False,
                "candidate_list": memory.top_root_objects(object_graph, self.config.llm_controller_top_k),
                "summary": "fallback memory top objects",
            }, self._empty_usage()
        candidate_view = []
        for object_id in memory.top_root_objects(object_graph, self.config.llm_agent_context_top_k):
            node = object_graph.nodes[object_id]
            packet = memory.object_packet(object_id)
            candidate_view.append({
                "object_id": object_id,
                "representative": node.representative,
                "score_hint": round(memory.root_adjusted_score(object_id, object_graph), 4),
                "reason_hint": node.best_reason(),
                "role_summary": packet["role_summary"],
                "symptom_score": packet["symptom_score"],
                "mechanism_summary": packet["mechanism_summary"],
                "backbone_prior": packet["backbone_prior"],
                "claims": packet["claims"][:5],
                "grafts": packet["grafts"][:3],
                "relations": packet["relations"][:4],
                "debates": packet["debates"][:2],
                "graph_summary": {
                    "anomaly_score": round(node.anomaly_score, 4),
                    "downstream_mass": round(object_graph.topological_mass(object_id), 4),
                    "incoming_mass": round(object_graph.incoming_mass(object_id), 4),
                },
            })
        usage = self._empty_usage()
        ballots: List[Dict[str, Any]] = []
        vote_score: Dict[str, float] = {}
        for vote_idx in range(self.config.controller_self_consistency):
            payload, vote_usage = self._controller_vote_once(query, candidate_view, vote_idx)
            usage = self._merge_usage(usage, vote_usage)
            ballots.append(payload)
            selected = []
            for rank, object_id in enumerate(payload.get("candidate_list", [])[:self.config.llm_controller_top_k]):
                object_id = str(object_id).strip()
                if object_id in object_graph.nodes and object_id not in selected:
                    selected.append(object_id)
                    vote_score[object_id] = vote_score.get(object_id, 0.0) + max(0.0, self.config.llm_controller_top_k - rank)
        if not vote_score:
            return {"enabled": True, "success": False, "candidate_list": memory.top_root_objects(object_graph, self.config.llm_controller_top_k)}, usage
        selected = [
            object_id for object_id, _ in sorted(
                vote_score.items(),
                key=lambda item: (item[1], memory.root_adjusted_score(item[0], object_graph)),
                reverse=True,
            )[:self.config.llm_controller_top_k]
        ]
        selected = memory.hard_filter_candidates(
            object_graph,
            selected,
            symptom_threshold=self.config.symptom_hard_threshold,
            keep_top_k=1,
        )
        return {
            "enabled": True,
            "success": bool(selected),
            "candidate_list": selected,
            "sleeping_objects": [str(item) for ballot in ballots for item in ballot.get("sleeping_objects", [])[:3]][:5],
            "summary": ballots[0].get("controller_summary", "") if ballots else "",
            "ballots": ballots,
            "vote_score": {k: round(v, 4) for k, v in vote_score.items()},
        }, usage

    def _controller_vote_once(
        self,
        query: QueryCase,
        candidate_view: List[Dict[str, Any]],
        vote_idx: int,
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        system_prompt = (
            "You are the meta-cognitive controller of MACE-RCA. "
            "Choose the active candidate list after reading agent claims, mechanism-aware bridge claims, roles, symptom scores, PRISM backbone priors, grafts and graph context. "
            "Prefer candidates with stronger causal_uniqueness and evidence_specificity, and downweight candidates with high symptom_score or high symptomness_prior unless they have strong outbound propagation evidence. "
            "Return strict JSON only."
        )
        user_prompt = {
            "query_instruction": query.instruction,
            "consistency_round": vote_idx,
            "candidate_pool": candidate_view,
            "output_schema": {
                "candidate_list": ["ordered object_id list"],
                "sleeping_objects": ["optional object_ids"],
                "controller_summary": "why these objects remain active",
            },
        }
        resp = self.llm_client.call(
            model=self.config.llm_model,
            messages=[{"role": "user", "content": json.dumps(user_prompt, ensure_ascii=True)}],
            system=system_prompt,
            response_format={"type": "json_object"},
            max_tokens=360,
            temperature=min(0.35, self.config.llm_temperature + 0.05 * vote_idx),
        )
        if resp is None:
            return {}, self._empty_usage()
        payload = resp.content if isinstance(resp.content, dict) else {}
        return payload, self._usage_from_response(resp.usage)

    def _attach_counterfactuals(
        self,
        memory: SharedMemoryGraph,
        intervention_rank: List[Dict[str, Any]],
    ) -> None:
        for item in intervention_rank:
            summary = (
                f"If {item['object_id']} were intervened on locally, removed_mass={item['removed_mass']:.3f}, "
                f"coverage={item['coverage']:.3f}, exclusivity={item['exclusivity']:.3f}."
            )
            memory.attach_counterfactual(
                item["object_id"],
                CounterfactualRecord(
                    object_id=item["object_id"],
                    summary=summary,
                    removed_mass=float(item["removed_mass"]),
                    coverage=float(item["coverage"]),
                    exclusivity=float(item["exclusivity"]),
                    confidence=float(item["score"]),
                    created_by="local_intervention",
                ),
            )

    def _run_debates(
        self,
        query: QueryCase,
        object_graph: ObjectGraph,
        memory: SharedMemoryGraph,
        candidate_objects: List[str],
        intervention_rank: List[Dict[str, Any]],
        trace: List[Dict[str, Any]],
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        pairs = memory.build_debate_pairs(candidate_objects, self.config.debate_pairs)
        debug = {"enabled": self.llm_client is not None, "pairs": [], "records": []}
        usage = self._empty_usage()
        intervention_map = {item["object_id"]: item for item in intervention_rank}
        for obj_a, obj_b, claim_a, claim_b in pairs:
            if self.llm_client is None:
                score_a = memory.object_scores.get(obj_a, 0.0)
                score_b = memory.object_scores.get(obj_b, 0.0)
                winner = obj_a if score_a >= score_b else obj_b
                record = DebateRecord(
                    object_a=obj_a,
                    object_b=obj_b,
                    pro_claim_id=claim_a,
                    con_claim_id=claim_b,
                    winner_object_id=winner,
                    verdict="heuristic winner by memory score",
                    confidence=0.55,
                    summary="fallback heuristic debate",
                )
            else:
                record, debate_usage, raw = self._llm_debate(
                    query=query,
                    object_graph=object_graph,
                    memory=memory,
                    obj_a=obj_a,
                    obj_b=obj_b,
                    claim_a=claim_a,
                    claim_b=claim_b,
                    intervention_map=intervention_map,
                )
                usage = self._merge_usage(usage, debate_usage)
                debug["pairs"].append(raw)
            memory.record_debate(record)
            debug["records"].append({
                "object_a": obj_a,
                "object_b": obj_b,
                "winner_object_id": record.winner_object_id,
                "summary": record.summary,
                "confidence": round(record.confidence, 4),
            })
            trace.append({
                "step": len(trace),
                "action": "DEBATE",
                "objects": [obj_a, obj_b],
                "winner": record.winner_object_id,
                "summary": record.summary,
            })
        return debug, usage

    def _llm_debate(
        self,
        query: QueryCase,
        object_graph: ObjectGraph,
        memory: SharedMemoryGraph,
        obj_a: str,
        obj_b: str,
        claim_a: str,
        claim_b: str,
        intervention_map: Dict[str, Dict[str, Any]],
    ) -> tuple[DebateRecord, Dict[str, Any], Dict[str, Any]]:
        system_prompt = (
            "You are the debate arbiter in MACE-RCA. "
            "Two competing objects claim to be the root cause. "
            "Choose the stronger one using claims, mechanism-aware bridge claims, role labels, symptom scores, evidence and counterfactual validation. "
            "Prefer upstream causes over downstream symptoms when evidence is comparable, and use PRISM backbone priors to penalize isolated symptom buckets and non-unique evidence. "
            "Return strict JSON only."
        )
        user_prompt = {
            "query_instruction": query.instruction,
            "candidate_a": {
                "object_id": obj_a,
                "packet": memory.object_packet(obj_a),
                "root_adjusted_score": round(memory.root_adjusted_score(obj_a, object_graph), 4),
                "symptom_score": memory.object_symptom_score(obj_a),
                "backbone_prior": memory.object_packet(obj_a).get("backbone_prior", {}),
                "intervention": intervention_map.get(obj_a, {}),
            },
            "candidate_b": {
                "object_id": obj_b,
                "packet": memory.object_packet(obj_b),
                "root_adjusted_score": round(memory.root_adjusted_score(obj_b, object_graph), 4),
                "symptom_score": memory.object_symptom_score(obj_b),
                "backbone_prior": memory.object_packet(obj_b).get("backbone_prior", {}),
                "intervention": intervention_map.get(obj_b, {}),
            },
            "output_schema": {
                "winner_object_id": "candidate_a.object_id or candidate_b.object_id",
                "confidence": "0-1 float",
                "verdict": "short verdict",
                "summary": "short arbitration summary",
            },
        }
        resp = self.llm_client.call(
            model=self.config.llm_model,
            messages=[{"role": "user", "content": json.dumps(user_prompt, ensure_ascii=True)}],
            system=system_prompt,
            response_format={"type": "json_object"},
            max_tokens=300,
            temperature=self.config.llm_temperature,
        )
        if resp is None:
            winner = obj_a if memory.object_scores.get(obj_a, 0.0) >= memory.object_scores.get(obj_b, 0.0) else obj_b
            return DebateRecord(
                object_a=obj_a,
                object_b=obj_b,
                pro_claim_id=claim_a,
                con_claim_id=claim_b,
                winner_object_id=winner,
                verdict="fallback heuristic",
                confidence=0.55,
                summary="fallback heuristic debate",
            ), self._empty_usage(), {}
        payload = resp.content if isinstance(resp.content, dict) else {}
        winner = str(payload.get("winner_object_id", "")).strip()
        if winner not in {obj_a, obj_b}:
            winner = obj_a if memory.object_scores.get(obj_a, 0.0) >= memory.object_scores.get(obj_b, 0.0) else obj_b
        return DebateRecord(
            object_a=obj_a,
            object_b=obj_b,
            pro_claim_id=claim_a,
            con_claim_id=claim_b,
            winner_object_id=winner,
            verdict=str(payload.get("verdict", "")).strip() or "llm debate",
            confidence=self._clamp_float(payload.get("confidence"), default=0.6),
            summary=str(payload.get("summary", "")).strip(),
        ), self._usage_from_response(resp.usage), payload

    def _llm_final_decision(
        self,
        query: QueryCase,
        object_graph: ObjectGraph,
        memory: SharedMemoryGraph,
        candidate_objects: List[str],
        intervention_rank: List[Dict[str, Any]],
        final_scores: Dict[str, float],
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        if self.llm_client is None:
            return {"enabled": False}, self._empty_usage()
        intervention_map = {item["object_id"]: item for item in intervention_rank}
        candidates = []
        for object_id in candidate_objects[:self.config.llm_controller_top_k]:
            if object_id not in object_graph.nodes:
                continue
            node = object_graph.nodes[object_id]
            packet = memory.object_packet(object_id)
            candidates.append({
                "object_id": object_id,
                "representative": node.representative,
                "memory_score": round(memory.object_scores.get(object_id, 0.0), 4),
                "root_adjusted_score": round(memory.root_adjusted_score(object_id, object_graph), 4),
                "symptom_score": packet["symptom_score"],
                "mechanism_summary": packet["mechanism_summary"],
                "backbone_prior": packet["backbone_prior"],
                "heuristic_score": round(final_scores.get(object_id, 0.0), 4),
                "reason_hint": node.best_reason(),
                "packet": packet,
                "intervention": intervention_map.get(object_id, {}),
                "graph_summary": {
                    "anomaly_score": round(node.anomaly_score, 4),
                    "downstream_mass": round(object_graph.topological_mass(object_id), 4),
                    "incoming_mass": round(object_graph.incoming_mass(object_id), 4),
                    "members": node.members[:5],
                },
            })
        usage = self._empty_usage()
        ballots: List[Dict[str, Any]] = []
        vote_score: Dict[str, float] = {}
        for vote_idx in range(self.config.final_self_consistency):
            payload, vote_usage = self._final_vote_once(query, candidates, vote_idx)
            usage = self._merge_usage(usage, vote_usage)
            ballots.append(payload)
            chosen = str(payload.get("chosen_object_id", "")).strip()
            if chosen in object_graph.nodes:
                confidence = self._clamp_float(payload.get("confidence"), default=0.5)
                vote_score[chosen] = vote_score.get(chosen, 0.0) + confidence
        if not vote_score:
            return {"enabled": True, "success": False, "ballots": ballots}, usage
        chosen = max(
            vote_score,
            key=lambda object_id: (
                vote_score.get(object_id, 0.0),
                final_scores.get(object_id, 0.0),
                memory.root_adjusted_score(object_id, object_graph),
            ),
        )
        winning_ballot = next(
            (ballot for ballot in ballots if str(ballot.get("chosen_object_id", "")).strip() == chosen),
            ballots[0] if ballots else {},
        )
        return {
            "enabled": True,
            "success": True,
            "chosen_object_id": chosen,
            "reason": str(winning_ballot.get("reason", "")).strip(),
            "confidence": min(1.0, vote_score.get(chosen, 0.0) / max(1, self.config.final_self_consistency)),
            "evidence_summary": str(winning_ballot.get("evidence_summary", "")).strip(),
            "time": str(winning_ballot.get("time", "")).strip(),
            "payload": winning_ballot,
            "ballots": ballots,
            "vote_score": {k: round(v, 4) for k, v in vote_score.items()},
        }, usage

    def _final_vote_once(
        self,
        query: QueryCase,
        candidates: List[Dict[str, Any]],
        vote_idx: int,
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        system_prompt = (
            "You are the final meta-cognitive controller in full MACE-RCA. "
            "Choose the final root cause after reading evidence, mechanism-aware bridge claims, role labels, symptom scores, PRISM backbone priors, counterfactual records, grafted claims and debate results. "
            "Prefer causes that explain downstream propagation, have stronger causal uniqueness, and are not hot symptom buckets. "
            "Return strict JSON only."
        )
        user_prompt = {
            "query_instruction": query.instruction,
            "time_window": list(query.time_window),
            "consistency_round": vote_idx,
            "active_candidates": candidates,
            "output_schema": {
                "chosen_object_id": "one object_id from active_candidates",
                "reason": "short reason label",
                "confidence": "0-1 float",
                "evidence_summary": "short explanation",
                "time": "YYYY-MM-DD HH:MM:SS or empty",
            },
        }
        resp = self.llm_client.call(
            model=self.config.llm_model,
            messages=[{"role": "user", "content": json.dumps(user_prompt, ensure_ascii=True)}],
            system=system_prompt,
            response_format={"type": "json_object"},
            max_tokens=420,
            temperature=min(0.35, self.config.llm_temperature + 0.05 * vote_idx),
        )
        if resp is None:
            return {}, self._empty_usage()
        payload = resp.content if isinstance(resp.content, dict) else {}
        return payload, self._usage_from_response(resp.usage)

    def _run_pairwise_tournament(
        self,
        query: QueryCase,
        object_graph: ObjectGraph,
        memory: SharedMemoryGraph,
        candidate_objects: List[str],
        intervention_rank: List[Dict[str, Any]],
        trace: List[Dict[str, Any]],
    ) -> tuple[Dict[str, Any], Dict[str, Any], List[str]]:
        if self.llm_client is None:
            return {"enabled": False, "ranking": candidate_objects[: self.config.tournament_top_k]}, self._empty_usage(), candidate_objects
        contenders = candidate_objects[: self.config.tournament_top_k]
        if len(contenders) < 2:
            return {"enabled": False, "ranking": contenders}, self._empty_usage(), contenders
        usage = self._empty_usage()
        scores = {object_id: 0.0 for object_id in contenders}
        rounds = []
        intervention_map = {item["object_id"]: item for item in intervention_rank}
        for i in range(len(contenders)):
            for j in range(i + 1, len(contenders)):
                obj_a = contenders[i]
                obj_b = contenders[j]
                claim_a = memory.best_claim_id(obj_a)
                claim_b = memory.best_claim_id(obj_b)
                if not claim_a or not claim_b:
                    continue
                record, debate_usage, raw = self._llm_debate(
                    query=query,
                    object_graph=object_graph,
                    memory=memory,
                    obj_a=obj_a,
                    obj_b=obj_b,
                    claim_a=claim_a,
                    claim_b=claim_b,
                    intervention_map=intervention_map,
                )
                usage = self._merge_usage(usage, debate_usage)
                scores[record.winner_object_id] = scores.get(record.winner_object_id, 0.0) + record.confidence
                rounds.append({
                    "object_a": obj_a,
                    "object_b": obj_b,
                    "winner": record.winner_object_id,
                    "confidence": round(record.confidence, 4),
                    "summary": record.summary,
                    "raw": raw,
                })
                trace.append({
                    "step": len(trace),
                    "action": "TOURNAMENT",
                    "objects": [obj_a, obj_b],
                    "winner": record.winner_object_id,
                    "summary": record.summary,
                })
        ranking = [object_id for object_id, _ in sorted(
            scores.items(),
            key=lambda item: (item[1], memory.root_adjusted_score(item[0], object_graph)),
            reverse=True,
        )]
        remaining = [object_id for object_id in candidate_objects if object_id not in ranking]
        return {
            "enabled": True,
            "rounds": rounds,
            "scores": {k: round(v, 4) for k, v in scores.items()},
            "ranking": ranking,
        }, usage, ranking + remaining

    def _object_cards(self, object_graph: ObjectGraph, memory: Optional[SharedMemoryGraph] = None) -> List[Dict[str, Any]]:
        cards = []
        for object_id, node in object_graph.nodes.items():
            backbone_prior = memory.object_priors.get(object_id, {}) if memory is not None else {}
            cards.append({
                "object_id": object_id,
                "representative": node.representative,
                "reason_hint": node.best_reason(),
                "mechanism_hint": self._reason_to_mechanism(node.best_reason()),
                "backbone_prior": backbone_prior,
                "anomaly_score": round(node.anomaly_score, 4),
                "metric_score": round(node.metric_score, 4),
                "log_score": round(node.log_score, 4),
                "trace_score": round(node.trace_score, 4),
                "change_score": round(node.change_score, 4),
                "downstream_mass": round(object_graph.topological_mass(object_id), 4),
                "incoming_mass": round(object_graph.incoming_mass(object_id), 4),
                "role_hint": self._object_role_hint(object_graph, object_id),
                "members": node.members[:5],
                "evidence": [
                    {
                        "kind": item.kind,
                        "source": item.source,
                        "content": item.content[:160],
                        "confidence": round(item.confidence, 4),
                    }
                    for item in node.evidence[:3]
                ],
            })
        if memory is not None:
            cards.sort(
                key=lambda item: (
                    float(item["backbone_prior"].get("rootness_prior", 0.0)),
                    float(item["backbone_prior"].get("causal_uniqueness", 0.0)),
                    -float(item["backbone_prior"].get("symptomness_prior", 0.0)),
                    item["anomaly_score"],
                ),
                reverse=True,
            )
        else:
            cards.sort(key=lambda item: (item["anomaly_score"], item["downstream_mass"]), reverse=True)
        return cards

    def _claim_packet(self, claim, object_graph: ObjectGraph) -> Dict[str, Any]:
        node = object_graph.nodes.get(claim.object_id)
        return {
            "claim_id": claim.claim_id,
            "object_id": claim.object_id,
            "agent": claim.agent,
            "claim": claim.claim,
            "reason": claim.reason,
            "role": getattr(claim, "role", ""),
            "bridge_to": getattr(claim, "bridge_to", ""),
            "mechanism": getattr(claim, "mechanism", ""),
            "symptom_score": round(getattr(claim, "symptom_score", 0.0), 4),
            "confidence": round(claim.confidence, 4),
            "evidence_summary": claim.evidence_summary,
            "graph_summary": {
                "anomaly_score": round(node.anomaly_score, 4) if node else 0.0,
                "downstream_mass": round(object_graph.topological_mass(claim.object_id), 4) if node else 0.0,
                "incoming_mass": round(object_graph.incoming_mass(claim.object_id), 4) if node else 0.0,
            },
        }

    def _object_role_hint(self, object_graph: ObjectGraph, object_id: str) -> Dict[str, float]:
        downstream = object_graph.topological_mass(object_id)
        incoming = object_graph.incoming_mass(object_id)
        if downstream > incoming + 0.20:
            return {"cause": 1.0, "propagation": 0.5, "symptom": 0.0}
        if incoming > downstream + 0.15:
            return {"cause": 0.0, "propagation": 0.4, "symptom": 1.0}
        return {"cause": 0.3, "propagation": 1.0, "symptom": 0.3}

    def _reason_to_mechanism(self, reason: str) -> str:
        text = (reason or "").lower()
        if "cpu" in text:
            return "cpu_pressure"
        if "memory" in text:
            return "memory_pressure"
        if "network" in text or "latency" in text:
            return "network_latency"
        if "disk" in text:
            return "disk_io"
        if "db" in text:
            return "db_backpressure"
        if "change" in text:
            return "change_trigger"
        return "generic_propagation"

    def _empty_usage(self) -> Dict[str, Any]:
        return {"api_calls": 0, "total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0}

    def _usage_from_response(self, usage: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        usage = usage or {}
        return {
            "api_calls": 1,
            "total_tokens": int(usage.get("total_tokens", 0) or 0),
            "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
            "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
        }

    def _merge_usage(self, left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "api_calls": int(left.get("api_calls", 0)) + int(right.get("api_calls", 0)),
            "total_tokens": int(left.get("total_tokens", 0)) + int(right.get("total_tokens", 0)),
            "prompt_tokens": int(left.get("prompt_tokens", 0)) + int(right.get("prompt_tokens", 0)),
            "completion_tokens": int(left.get("completion_tokens", 0)) + int(right.get("completion_tokens", 0)),
        }

    def _clamp_float(self, value: Any, default: float) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return default
        return max(0.0, min(1.0, parsed))

    def _looks_like_time(self, text: str) -> bool:
        return isinstance(text, str) and len(text) >= 19 and text[4] == "-" and text[13] == ":"


def _graph_debug_payload(graph: ObjectGraph) -> Dict[str, Any]:
    return {
        "nodes": {
            object_id: {
                "representative": node.representative,
                "members": node.members,
                "anomaly_score": round(node.anomaly_score, 4),
                "metric_score": round(node.metric_score, 4),
                "log_score": round(node.log_score, 4),
                "trace_score": round(node.trace_score, 4),
                "change_score": round(node.change_score, 4),
                "reason": node.best_reason(),
            }
            for object_id, node in graph.nodes.items()
        },
        "adjacency": {
            src: {dst: round(weight, 4) for dst, weight in children.items()}
            for src, children in graph.adjacency.items()
        },
    }
