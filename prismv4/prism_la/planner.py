"""LLM action planning for PRISM-LA.

Parses LLM JSON responses into executable investigation plans,
event revisions, or stop decisions.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .state import EventHypothesis, HypothesisStatus, InvestigationPlan, PRISMLAState
from .config import PRISMLAConfig


class Planner:
    def __init__(self, config: PRISMLAConfig):
        self.config = config

    def parse_llm_response(
        self,
        response: Dict[str, Any],
        state: PRISMLAState,
    ) -> Tuple[str, Optional[InvestigationPlan], Optional[List[Dict[str, Any]]]]:
        action = str(response.get("action", "plan") or "plan").lower()
        if action == "stop":
            return ("stop", None, response.get("root_cause_events", []))
        if action == "revise":
            updates = response.get("event_updates", []) or []
            return ("revise", None, updates)
        investigations = response.get("investigations", []) or []
        rationale = str(response.get("rationale", "") or "")
        plan = InvestigationPlan(
            step_index=state.step_count,
            actions=self._normalize_actions(investigations, state),
            rationale=rationale,
            expected_outcome=str(response.get("expected_outcome", "") or ""),
        )
        return ("plan", plan, None)

    def _normalize_actions(
        self,
        raw_actions: List[Dict[str, Any]],
        state: PRISMLAState,
    ) -> List[Dict[str, Any]]:
        normalized = []
        entity_set = set(state.case.entities)
        for action in raw_actions[: self.config.max_plan_actions_per_step]:
            tool = str(action.get("tool", "") or "")
            component = str(action.get("component", "") or "")
            event_id = str(action.get("event_id", "") or "")
            other_event_id = str(action.get("other_event_id", "") or "")
            anchor_index = int(action.get("anchor_index", 0) or 0)
            anchor = state.case.anchors[anchor_index] if state.case.anchors and anchor_index < len(state.case.anchors) else None

            norm: Dict[str, Any] = {"tool": tool}
            if component and component in entity_set:
                norm["component"] = component
            elif component:
                matched = self._fuzzy_match(component, entity_set)
                norm["component"] = matched if matched else component
            else:
                top_event = state.top_root
                if top_event:
                    norm["component"] = top_event.component
            if tool == "find_time_anchors":
                norm["component"] = ""
            if anchor:
                norm["anchor_time"] = anchor.get("timestamp")
                norm["anchor_index"] = anchor_index
            if event_id:
                norm["event_id"] = event_id
            if other_event_id and tool in ("compare_events",):
                norm["other_event_id"] = other_event_id
            normalized.append(norm)
        return normalized

    @staticmethod
    def _fuzzy_match(component: str, entity_set: set) -> Optional[str]:
        clean = component.lower().replace("_", "").replace("-", "").replace(" ", "")
        exact = {e: e.lower().replace("_", "").replace("-", "").replace(" ", "")
                 for e in entity_set}
        if clean in exact.values():
            for orig, normed in exact.items():
                if normed == clean:
                    return orig
        for entity in entity_set:
            if component.lower() in entity.lower() or entity.lower() in component.lower():
                return entity
        return None

    def apply_revisions(
        self,
        updates: List[Dict[str, Any]],
        state: PRISMLAState,
    ) -> PRISMLAState:
        entity_set = set(state.case.entities)
        next_event_idx = max(
            (int(e.event_id.split("_")[-1].split(":")[0])
             for e in state.events
             if e.event_id.startswith("ev_")),
            default=0,
        )
        for update in updates:
            event_id = str(update.get("event_id", "") or "")
            is_new = bool(update.get("new_hypothesis", False))
            new_status = str(update.get("new_status", "") or "").lower()
            resolution = str(update.get("resolution_note", "") or "")
            component = str(update.get("component", "") or "")
            reason = str(update.get("reason", "") or "")
            time_val = update.get("time")
            if is_new and component:
                if component not in entity_set:
                    continue
                next_event_idx += 1
                new_event = EventHypothesis(
                    event_id=f"ev_{next_event_idx}:{component}",
                    component=component,
                    reason_hypothesis=reason,
                    time_hypothesis=float(time_val) if time_val is not None else None,
                    status=HypothesisStatus.UNRESOLVED,
                )
                state.events.append(new_event)
                continue
            event = state.event_by_id(event_id)
            if not event:
                continue
            if new_status in {"root", "contradicted", "symptom", "broad_explainer", "duplicate"}:
                try:
                    event.status = HypothesisStatus(new_status)
                except ValueError:
                    pass
                event.resolution_note = resolution if resolution else event.resolution_note
            if reason:
                event.reason_hypothesis = reason
            if time_val is not None:
                event.time_hypothesis = float(time_val)
        open_conflicts = updates if any("open_conflicts" in u for u in updates) else []
        for conflict_dict in open_conflicts:
            conflict_list = conflict_dict.get("open_conflicts", []) or []
            for conflict in conflict_list:
                if isinstance(conflict, dict):
                    state.conflicts.append(dict(conflict))
        return state

    def propose_initial_events(
        self,
        state: PRISMLAState,
        initial_response: Dict[str, Any],
    ) -> PRISMLAState:
        raw_events = initial_response.get("event_updates", []) or initial_response.get("hypotheses", []) or []
        if not raw_events:
            top_candidates = list(state.case.candidates)[: self.config.max_event_hypotheses]
            for idx, candidate in enumerate(top_candidates, start=1):
                component = str(candidate.get("component", "") or "")
                reason = str(candidate.get("reason", "") or "")
                time_val = candidate.get("time")
                state.events.append(
                    EventHypothesis(
                        event_id=f"ev_{idx}:{component}",
                        component=component,
                        reason_hypothesis=reason,
                        time_hypothesis=float(time_val) if time_val is not None else None,
                        status=HypothesisStatus.UNRESOLVED,
                    )
                )
        else:
            for idx, item in enumerate(raw_events[: self.config.max_event_hypotheses], start=1):
                component = str(item.get("component", "") or "")
                if not component or component not in set(state.case.entities):
                    continue
                reason = str(item.get("reason", "") or item.get("reason_hypothesis", "") or "")
                time_val = item.get("time") or item.get("time_hypothesis")
                state.events.append(
                    EventHypothesis(
                        event_id=f"ev_{idx}:{component}",
                        component=component,
                        reason_hypothesis=reason,
                        time_hypothesis=float(time_val) if time_val is not None else None,
                        status=HypothesisStatus.UNRESOLVED,
                    )
                )
        return state
