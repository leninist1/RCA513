"""PRISM-LA controller: observe-hypothesize-investigate-revise loop.

Orchestrates the LLM agent, tools, verifier, and synthesis into a
complete RCA investigation cycle.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .config import PRISMLAConfig, DEFAULT_CONFIG
from .state import (
    CaseState,
    EventHypothesis,
    EvidenceLedger,
    HypothesisStatus,
    InvestigationPlan,
    PRISMLAState,
)
from .sandbox import EvidenceSandbox, assert_no_answer_leakage
from .tools import PRISMLAToolbox, ToolEvidence
from .llm_client import LLMClient
from .prompts import AVAILABLE_TOOLS
from .planner import Planner
from .verifier import DeterministicVerifier, VerifierResult
from .synthesis import synthesize_answer

logger = logging.getLogger(__name__)


class PRISMLAController:
    def __init__(
        self,
        config: Optional[PRISMLAConfig] = None,
    ):
        self.config = config or DEFAULT_CONFIG
        self.llm = LLMClient(self.config)
        self.planner = Planner(self.config)
        self.verifier = DeterministicVerifier(self.config)
        self.toolbox: Optional[PRISMLAToolbox] = None
        self.sandbox = EvidenceSandbox()

    def run(
        self,
        query: Any,
        telemetry: Any,
        entities: Sequence[str],
        entity_types: Dict[str, str],
        topology: Sequence[Tuple[str, str, float]],
        metric_signal: Any,
        log_signal: Any,
        graph: Any,
        metric_detail: Dict[str, Dict[str, float]],
        log_detail: Dict[str, Dict[str, Any]],
        trace_detail: Dict[str, Dict[str, Any]],
        anomaly_times: Dict[str, float],
        anchor_set: List[Dict[str, Any]],
        candidate_scores: Dict[str, Dict[str, float]],
        cmi_profiles: Dict[str, Dict[str, Any]],
        cf_profile_cache: Dict[str, Dict[str, Any]],
        telemetry_meta: Optional[Dict[str, Any]] = None,
        reason_inferer: Any = None,
    ) -> Dict[str, Any]:
        assert_no_answer_leakage(query)
        import numpy as np
        metric_arr = np.asarray(metric_signal, dtype=float) if hasattr(metric_signal, '__len__') else np.array([])
        log_arr = np.asarray(log_signal, dtype=float) if hasattr(log_signal, '__len__') else np.array([])
        graph_arr = np.asarray(graph, dtype=float) if hasattr(graph, '__len__') else np.array([[]])
        entity_list = [str(e) for e in entities]
        entity_types_dict = {str(k): str(v) for k, v in entity_types.items()}

        self.toolbox = PRISMLAToolbox(
            entities=entity_list,
            entity_types=entity_types_dict,
            metric_signal=metric_arr,
            log_signal=log_arr,
            graph=graph_arr,
            metric_detail=metric_detail,
            log_detail=log_detail,
            trace_detail=trace_detail,
            anomaly_times=anomaly_times,
            anchor_set=anchor_set,
            candidate_scores=candidate_scores,
            cmi_profiles=cmi_profiles,
            cf_profile_cache=cf_profile_cache,
        )

        tm = telemetry_meta or {}
        self.sandbox.build_public_view(
            query=query,
            entities=entity_list,
            entity_types=entity_types_dict,
            topology_edges=topology,
            telemetry_meta=tm,
        )

        case = CaseState(
            query_id=str(getattr(query, "task_index", "")) or str(getattr(query, "query_id", "")),
            system=str(getattr(query, "system", "")),
            sub_system=str(getattr(query, "sub_system", "")),
            instruction=str(getattr(query, "instruction", "")),
            time_window=tuple(getattr(query, "time_window", ("", ""))),
            entities=entity_list,
            entity_types=entity_types_dict,
            anchors=anchor_set,
            candidates=[{"component": k, "score": float(v.get("root_score", 0.0) or 0.0)}
                         for k, v in sorted(candidate_scores.items(),
                                            key=lambda item: float(item[1].get("root_score", 0.0) or 0.0),
                                            reverse=True)[:15]],
            topology_edges=[(str(s), str(t), float(w)) for s, t, w in topology],
            has_logs=bool(log_arr.size > 0),
            has_traces=bool(graph_arr.size > 1),
            telemetry_summary={
                "metric_entity_count": int(metric_arr.size),
                "log_entity_count": int(np.count_nonzero(log_arr)) if log_arr.size else 0,
                "trace_span_count": int(graph_arr.sum()) if graph_arr.size else 0,
                "entity_list": entity_list[:30],
            },
        )

        state = PRISMLAState(case=case, ledger=EvidenceLedger())

        logger.info(f"PRISM-LA starting: {case.query_id} | {len(entities)} entities | anchors={len(anchor_set)}")
        self.llm.connect()
        self.llm.start_session()

        case_view = state.case.to_llm_view()
        top_candidates = [c["component"] for c in state.case.candidates if c.get("component")][:8]
        if not top_candidates:
            top_candidates = entity_list[:8]
        initial_response = self.llm.invoke_initial(case_view, top_candidates)

        state = self.planner.propose_initial_events(state, initial_response)

        # Seed initial posteriors from candidate scores so the LLM knows which
        # entities are most promising before any tool calls.
        total_root = max(
            sum(max(0.0, float(s.get("root_score", 0.0) or 0.0)) for s in candidate_scores.values()),
            1e-9,
        )
        for event in state.events:
            cs = candidate_scores.get(event.component, {})
            event.posterior = max(0.01, float(cs.get("root_score", 0.0) or 0.0) / total_root)

        for step in range(self.config.max_investigation_steps):
            state.step_count = step

            case_view = state.case.to_llm_view()
            events_view = [e.to_llm_view() for e in state.events]
            ledger_view = state.ledger.to_llm_view()

            response = self.llm.invoke_plan(
                case_view=case_view,
                events=events_view,
                ledger_entries=ledger_view,
                step=step,
            )

            action_type, plan, revisions = self.planner.parse_llm_response(
                response, state
            )

            if action_type == "stop":
                root_events = revisions or []
                final_answer = self._build_answer_from_llm_response(
                    root_events, state
                )
                verif = self.verifier.final_validate(final_answer, state)
                if verif.passed:
                    state.stop_reason = "llm_stop_verified"
                    logger.info(f"PRISM-LA stopped at step {step}: {state.stop_reason}")
                    return final_answer
                logger.warning(f"LLM stop rejected by verifier: {verif.messages}")
                state.conflicts.append({
                    "step": step,
                    "type": "verifier_rejected",
                    "messages": verif.messages,
                })
                continue

            if action_type == "revise":
                state = self.planner.apply_revisions(revisions, state)
                verif = self.verifier.normalize_and_check(state)
                if verif.has_blockers:
                    logger.warning(f"Revision blocked: {verif.blocking}")
                if self.verifier.ready_to_answer(state):
                    final_answer = synthesize_answer(state, self.config)
                    verif = self.verifier.final_validate(final_answer, state)
                    if verif.passed:
                        state.stop_reason = "revised_to_answer"
                        logger.info(f"PRISM-LA stopped at step {step}: {state.stop_reason}")
                        return final_answer
                continue

            if plan is None or not plan.actions:
                logger.warning(f"No valid plan actions at step {step}")
                if self.verifier.ready_to_answer(state):
                    final_answer = synthesize_answer(state, self.config)
                    state.stop_reason = "exhausted_plans"
                    return final_answer
                continue

            state.plans.append(plan)
            for action in plan.actions:
                tool_name = str(action.get("tool", "") or "")
                component = str(action.get("component", "") or "")
                event_id_param = str(action.get("event_id", "") or "")
                other_id = str(action.get("other_event_id", "") or "")
                anchor_time = action.get("anchor_time")

                matching_event = state.event_by_id(event_id_param) if event_id_param else None
                if not matching_event and component:
                    matching_event = self._find_event_by_component(state, component)

                try:
                    if tool_name == "find_time_anchors":
                        result = self.toolbox.find_time_anchors()
                        ev = ToolEvidence(
                            evidence_id=f"tool:{step}:anchors",
                            tool_name="find_time_anchors",
                            component="system",
                            factor="time_anchor",
                            support=1.0,
                            against=0.0,
                            payload=result,
                        )
                        state.ledger.add(ev)
                        self.sandbox.add_tool_result(result)

                    elif tool_name == "inspect_metric" and component:
                        ev = self.toolbox.inspect_metric(component, anchor_time)
                        state.ledger.add(ev)
                        self.sandbox.add_tool_result(ev.to_dict())
                        self._bind_evidence_to_event(matching_event, ev)

                    elif tool_name == "inspect_log" and component:
                        ev = self.toolbox.inspect_log(component, anchor_time)
                        state.ledger.add(ev)
                        self.sandbox.add_tool_result(ev.to_dict())
                        self._bind_evidence_to_event(matching_event, ev)

                    elif tool_name == "inspect_trace" and component:
                        ev = self.toolbox.inspect_trace(component, anchor_time)
                        state.ledger.add(ev)
                        self.sandbox.add_tool_result(ev.to_dict())
                        self._bind_evidence_to_event(matching_event, ev)

                    elif tool_name == "get_topology_neighbors" and component:
                        result = self.toolbox.get_topology_neighbors(component)
                        ev = ToolEvidence(
                            evidence_id=f"tool:{step}:topo:{component}",
                            tool_name="get_topology_neighbors",
                            component=component,
                            factor="topology",
                            support=0.0,
                            against=0.0,
                            payload=result,
                        )
                        state.ledger.add(ev)
                        self.sandbox.add_tool_result(result)

                    elif tool_name == "test_counterfactual" and component:
                        ev = self.toolbox.test_counterfactual(component)
                        state.ledger.add(ev)
                        self.sandbox.add_tool_result(ev.to_dict())
                        self._bind_evidence_to_event(matching_event, ev)

                    elif tool_name == "explain_residual" and component:
                        ev = self.toolbox.explain_residual(component)
                        state.ledger.add(ev)
                        self.sandbox.add_tool_result(ev.to_dict())
                        self._bind_evidence_to_event(matching_event, ev)

                    elif tool_name == "compare_events" and other_id:
                        event_a = {"component": component}
                        other_comp = str(action.get("other_component", ""))
                        event_b = {"component": other_comp}
                        results = self.toolbox.compare_events(event_a, event_b)
                        for idx, r in enumerate(results):
                            state.ledger.add(r)
                            self.sandbox.add_tool_result(r.to_dict())
                            target_comp = component if idx == 0 else other_comp
                            target_evt = self._find_event_by_component(state, target_comp)
                            self._bind_evidence_to_event(target_evt, r)

                except Exception as exc:
                    logger.error(f"Tool error: {tool_name} on {component}: {exc}")

            gap = state.compute_gap()
            for e in state.events:
                if e.is_active:
                    e.posterior = max(0.0, e.net_evidence) / max(1e-9, sum(
                        max(0.0, ee.net_evidence) for ee in state.events
                    ))

            if self.verifier.ready_to_answer(state):
                state.stop_reason = "verifier_ready"
                logger.info(f"PRISM-LA stopping at step {step}: verifier reports ready (gap={round(gap, 3)})")
                break

        final_answer = synthesize_answer(state, self.config)
        verif = self.verifier.final_validate(final_answer, state)
        if not verif.passed:
            logger.warning(f"Final answer validation warnings: {verif.blocking}")
        if not state.stop_reason:
            state.stop_reason = "max_steps"
        return final_answer

    def _build_answer_from_llm_response(
        self,
        root_events: List[Dict[str, Any]],
        state: PRISMLAState,
    ) -> Dict[str, Any]:
        import datetime
        components: List[str] = []
        reasons: List[str] = []
        times: List[str] = []
        result_events: List[Dict[str, Any]] = []
        for item in root_events:
            comp = str(item.get("component", "") or "")
            reason = str(item.get("reason", "") or "")
            time_raw = item.get("time", "")
            evidence_ids = list(item.get("evidence_ids", []) or [])
            if isinstance(time_raw, (int, float)):
                time_str = datetime.datetime.fromtimestamp(float(time_raw)).strftime("%Y-%m-%d %H:%M:%S")
            else:
                time_str = str(time_raw)
            if not time_str and state.case.anchors:
                ts = state.case.anchors[0].get("timestamp", 0)
                time_str = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
            components.append(comp)
            reasons.append(reason)
            times.append(time_str)
            result_events.append({
                "root cause occurrence datetime": time_str,
                "root cause component": comp,
                "root cause reason": reason,
            })
        if not components:
            fallback = synthesize_answer(state, self.config)
            return fallback
        return {
            "component": components,
            "reason": reasons,
            "time": times,
            "top_score": 0.0,
            "fault_count": len(components),
            "root_cause_events": result_events,
        }

    @staticmethod
    def _find_event_by_component(
        state: PRISMLAState, component: str
    ) -> Optional[EventHypothesis]:
        if not component:
            return None
        for event in state.events:
            if event.is_active and event.component == component:
                return event
        for event in state.events:
            if event.component == component:
                return event
        return None

    @staticmethod
    def _bind_evidence_to_event(
        event: Optional[EventHypothesis], evidence: ToolEvidence
    ) -> None:
        if event is None or evidence is None:
            return
        evidence.event_id = event.event_id
        if evidence.net_score() > 0:
            event.support_evidence.append(evidence)
        else:
            event.contradictory_evidence.append(evidence)
