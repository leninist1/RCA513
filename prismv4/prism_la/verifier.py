"""Deterministic verifier for PRISM-LA answers.

Validates that the LLM agent's output satisfies all sandbox,
evidence, and structural constraints before accepting it.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from .state import EventHypothesis, EvidenceLedger, HypothesisStatus, PRISMLAState
from .config import PRISMLAConfig


class VerifierResult:
    def __init__(self, passed: bool, messages: List[str]):
        self.passed = passed
        self.messages = messages
        self.blocking: List[str] = [m for m in messages if m.startswith("BLOCK")]
        self.warnings: List[str] = [m for m in messages if m.startswith("WARN")]

    @property
    def has_blockers(self) -> bool:
        return len(self.blocking) > 0


class DeterministicVerifier:
    def __init__(self, config: PRISMLAConfig):
        self.config = config

    def validate_answer(
        self,
        answer: Dict[str, Any],
        state: PRISMLAState,
    ) -> VerifierResult:
        messages: List[str] = []
        events = answer.get("root_cause_events", [])
        if not events:
            messages.append("BLOCK: no root_cause_events in answer")
            return VerifierResult(False, messages)

        entity_set = set(state.case.entities)
        ledger_ids = {item.evidence_id for item in state.ledger.entries}
        top_root = state.top_root

        for idx, event in enumerate(events):
            prefix = f"root_cause_events[{idx}]"
            component = str(
                event.get("root cause component", "")
                or event.get("component", "")
                or ""
            )
            reason = str(
                event.get("root cause reason", "")
                or event.get("reason", "")
                or ""
            )
            ev_ids = list(event.get("evidence_ids", []) or [])
            confidence = float(event.get("confidence", 0.0) or 0.0)

            if not component:
                messages.append(f"BLOCK: {prefix} missing component")
                continue
            if component not in entity_set:
                messages.append(f"BLOCK: {prefix} component '{component}' not in case entity graph")
            if reason and len(reason.strip()) < 3:
                messages.append(f"BLOCK: {prefix} reason too short: '{reason}'")

            if self.config.require_evidence_citations and not ev_ids:
                messages.append(f"BLOCK: {prefix} has no evidence citations")

            cited_in_ledger = set(ev_ids) & ledger_ids
            if ev_ids and not cited_in_ledger:
                messages.append(
                    f"BLOCK: {prefix} evidence_ids not found in ledger: "
                    f"{set(ev_ids) - ledger_ids}"
                )

            if confidence <= 0.0:
                messages.append(f"WARN: {prefix} confidence is 0.0")

        if top_root and top_root.status == HypothesisStatus.ROOT:
            answer_component = str(events[0].get("component", "")) if events else ""
            if answer_component and answer_component != top_root.component:
                messages.append(
                    f"WARN: answer component '{answer_component}' differs from agent top root "
                    f"'{top_root.component}'"
                )

        root_events_in_state = [
            e for e in state.events if e.status == HypothesisStatus.ROOT
        ]
        if top_root and not root_events_in_state:
            messages.append("WARN: no events resolved as ROOT during investigation")

        if not messages:
            messages.append("OK: answer validated successfully")

        return VerifierResult(
            passed=not any(m.startswith("BLOCK") for m in messages),
            messages=messages,
        )

    def check_event_revision(
        self,
        event: EventHypothesis,
        state: PRISMLAState,
    ) -> List[str]:
        messages: List[str] = []
        if event.status == HypothesisStatus.ROOT:
            if event.evidence_count < 2:
                messages.append(f"WARN: event {event.event_id} set to ROOT with <2 evidence items")
            if self.config.require_dual_modality and not event.multi_modal:
                messages.append(
                    f"WARN: event {event.event_id} set to ROOT without dual-modality evidence"
                )
            if event.net_evidence < self.config.min_root_margin:
                messages.append(
                    f"WARN: event {event.event_id} set to ROOT with low net_evidence "
                    f"({round(event.net_evidence, 3)})"
                )
            component_in_entities = event.component in set(state.case.entities)
            if not component_in_entities:
                messages.append(
                    f"BLOCK: event {event.event_id} component '{event.component}' not in entity graph"
                )
        if event.status == HypothesisStatus.SYMPTOM:
            if not event.resolution_note:
                messages.append(
                    f"WARN: event {event.event_id} set to SYMPTOM without resolution_note "
                    "(should indicate upstream root)"
                )
        return messages

    def normalize_and_check(
        self,
        state: PRISMLAState,
    ) -> VerifierResult:
        messages: List[str] = []
        for event in state.events:
            rev_messages = self.check_event_revision(event, state)
            messages.extend(rev_messages)
        if not messages:
            messages.append("OK: state normalized")
        return VerifierResult(
            passed=not any(m.startswith("BLOCK") for m in messages),
            messages=messages,
        )

    def ready_to_answer(self, state: PRISMLAState) -> bool:
        active = state.active_events
        if not active:
            return True
        root_events = [e for e in active if e.status == HypothesisStatus.ROOT]
        if not root_events:
            return False
        top = state.root_candidates[0] if state.root_candidates else None
        if top is None:
            return False
        gap = state.compute_gap()
        return (
            top.net_evidence > self.config.min_root_margin
            and gap > self.config.stop_gap_threshold
            and top.evidence_count >= self.config.stop_coverage_threshold
        )

    def final_validate(
        self,
        answer: Dict[str, Any],
        state: PRISMLAState,
    ) -> VerifierResult:
        return self.validate_answer(answer, state)


def validate_output_format(answer: Dict[str, Any]) -> List[str]:
    messages: List[str] = []
    events = answer.get("root_cause_events")
    if events is None:
        messages.append("BLOCK: missing root_cause_events key")
        return messages
    if not isinstance(events, list):
        messages.append("BLOCK: root_cause_events must be a list")
        return messages
    for idx, event in enumerate(events):
        if not isinstance(event, dict):
            messages.append(f"BLOCK: root_cause_events[{idx}] is not a dict")
            continue
        for key in ("component", "reason", "time"):
            if key not in event:
                messages.append(f"BLOCK: root_cause_events[{idx}] missing '{key}'")
    return messages
