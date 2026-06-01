"""Official-format answer synthesis for noise-native FaultEvents."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .fault_event import FaultEvent
from .posterior import root_selection_score


ReasonResolver = Callable[[FaultEvent], str]
TimeResolver = Callable[[FaultEvent], Tuple[str, str]]


def synthesize_event_answers(
    events: Sequence[FaultEvent],
    fault_count: int,
    reason_resolver: ReasonResolver,
    time_resolver: TimeResolver,
) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
    ranked = sorted(events, key=_selection_score, reverse=True)
    selected = [
        event
        for event in ranked
        if event.status not in {"merged", "duplicate"}
        and _selection_score(event) > 0.0
    ]
    if not selected:
        selected = ranked
    selected = selected[: max(1, int(fault_count))]

    answers: List[Dict[str, str]] = []
    time_sources: List[str] = []
    for event in selected:
        reason = reason_resolver(event) or event.reason
        time_value, time_source = time_resolver(event)
        time_sources.append(time_source)
        answers.append(
            {
                "root cause occurrence datetime": time_value,
                "root cause component": event.component,
                "root cause reason": reason,
            }
        )
    return answers, {
        "selected_event_ids": [event.event_id for event in selected],
        "time_sources": time_sources,
        "selection_scores": {
            event.event_id: round(float(_selection_score(event)), 6)
            for event in selected
        },
    }


def _selection_score(event: FaultEvent) -> float:
    return root_selection_score(event)


def event_answers_to_prediction(
    event_answers: Sequence[Dict[str, str]],
    raw_field_counts: Dict[str, int],
    field_counts: Dict[str, int],
    top_score: float,
) -> Dict[str, Any]:
    components = _field_values(
        event_answers, "root cause component", raw_field_counts, field_counts, "component"
    )
    reasons = _field_values(
        event_answers, "root cause reason", raw_field_counts, field_counts, "reason"
    )
    times = _field_values(
        event_answers, "root cause occurrence datetime", raw_field_counts, field_counts, "time"
    )
    return {
        "component": components,
        "reason": reasons,
        "time": times,
        "top_score": float(top_score),
        "fault_count": max(len(components), len(reasons), len(times), 1),
        "root_cause_events": list(event_answers),
    }


def _field_values(
    answers: Sequence[Dict[str, str]],
    answer_key: str,
    raw_field_counts: Dict[str, int],
    field_counts: Dict[str, int],
    field: str,
) -> List[str]:
    raw_count = max(1, int(raw_field_counts.get(field, len(answers)) or 1))
    unique_target = max(1, int(field_counts.get(field, raw_count) or 1))
    values: List[str] = []
    seen = set()
    for answer in answers:
        value = str(answer.get(answer_key, "") or "")
        if unique_target <= 1:
            if not values:
                values.append(value)
        elif value not in seen:
            values.append(value)
            seen.add(value)
        if len(values) >= unique_target:
            break
    while values and len(values) < raw_count and unique_target <= 1:
        values.append(values[0])
    while len(values) < raw_count:
        values.append(values[-1] if values else "")
    return values[:raw_count]
