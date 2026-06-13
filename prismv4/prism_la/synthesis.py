"""Answer synthesis for PRISM-LA.

Converts resolved event hypotheses into OpenRCA-format answer dicts.
"""

from __future__ import annotations

import datetime
from typing import Any, Dict, List, Optional, Sequence

from .state import EventHypothesis, HypothesisStatus, PRISMLAState
from .config import PRISMLAConfig


def synthesize_answer(
    state: PRISMLAState,
    config: PRISMLAConfig,
    fault_count: int = 1,
) -> Dict[str, Any]:
    root_events = [
        e for e in state.events
        if e.status == HypothesisStatus.ROOT and e.is_active
    ]
    if not root_events:
        scored = sorted(
            [e for e in state.events if e.is_active and e.component],
            key=lambda e: (
                e.net_evidence,
                e.evidence_count,
            ),
            reverse=True,
        )
        if scored and scored[0].evidence_count >= 1 and scored[0].net_evidence > 0.1:
            root_events = [scored[0]]
        elif scored:
            root_events = scored[:1]
    root_events.sort(key=lambda e: (e.net_evidence, e.evidence_count), reverse=True)
    selected = root_events[: max(1, int(fault_count))]

    if not selected:
        return {
            "component": [], "reason": [], "time": [], "top_score": 0.0,
            "fault_count": 0, "root_cause_events": [],
        }

    components: List[str] = []
    reasons: List[str] = []
    times: List[str] = []
    root_cause_events: List[Dict[str, Any]] = []
    top_scores: List[Dict[str, Any]] = []

    total_evidence = max(sum(e.net_evidence for e in selected), 1e-9)

    for event in selected:
        component = event.component
        reason = event.reason_hypothesis or _infer_reason_from_evidence(event)
        timestamp = event.time_hypothesis
        if timestamp:
            datetime_str = datetime.datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
        elif event.support_evidence or event.contradictory_evidence:
            all_ts = []
            for ev in event.support_evidence + event.contradictory_evidence:
                if ev.timestamp is not None and ev.timestamp > 0:
                    all_ts.append(float(ev.timestamp))
            if all_ts:
                ts = sorted(all_ts)[len(all_ts)//2]
                datetime_str = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
            elif state.case.anchors:
                ts = state.case.anchors[0].get("timestamp", 0)
                datetime_str = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
            else:
                datetime_str = state.case.time_window[0] if state.case.time_window else ""
        elif state.case.anchors:
            ts = state.case.anchors[0].get("timestamp", 0)
            datetime_str = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
        elif state.case.time_window:
            datetime_str = state.case.time_window[0]
        else:
            datetime_str = ""

        evidence_ids = [
            item.evidence_id
            for item in event.support_evidence + event.contradictory_evidence
        ]
        if not evidence_ids:
            for entry in state.ledger.entries:
                if entry.component == component:
                    evidence_ids.append(entry.evidence_id)

        confidence = event.net_evidence / total_evidence if total_evidence > 0 else 0.0

        components.append(component)
        reasons.append(reason)
        times.append(datetime_str)
        root_cause_events.append({
            "root cause occurrence datetime": datetime_str,
            "root cause component": component,
            "root cause reason": reason,
        })
        top_scores.append({
            "entity": component,
            "score": round(float(confidence), 4),
            "net_evidence": round(float(event.net_evidence), 4),
            "evidence_ids": evidence_ids,
        })

    top_score = top_scores[0]["score"] if top_scores else 0.0

    return {
        "component": components,
        "reason": reasons,
        "time": times,
        "top_score": round(float(top_score), 4),
        "fault_count": len(selected),
        "root_cause_events": root_cause_events,
        "top_scores": top_scores,
    }


def _infer_reason_from_evidence(event: "EventHypothesis") -> str:  # type: ignore
    all_details = []
    for ev in event.support_evidence:
        all_details.extend(ev.evidence_details)
    if not all_details and event.contradictory_evidence:
        for ev in event.contradictory_evidence:
            all_details.extend(ev.evidence_details)
    text = " ".join(all_details).lower()
    if "oom" in text or "outofmemory" in text or "killed" in text:
        return "high memory usage"
    if "cpu" in text and ("usage" in text or "throttl" in text or "load" in text):
        return "high CPU usage"
    if "disk" in text or "io" in text:
        return "high disk I/O usage"
    if "space" in text:
        return "high disk space usage"
    if "latency" in text or "timeout" in text or "slow" in text:
        return "network latency"
    if "packet" in text or "retransmit" in text:
        return "network packet loss"
    if "connection" in text and ("refused" in text or "error" in text):
        return "high memory usage"
    if "memory" in text:
        return "high memory usage"
    if "jvm" in text or "gc" in text:
        return "high JVM CPU load"
    log_ev = [e for e in event.support_evidence if e.tool_name == "inspect_log"]
    if log_ev and any("error" in d.lower() for d in log_ev[0].evidence_details):
        return "unspecified service error"
    return "unspecified anomaly"

    components: List[str] = []
    reasons: List[str] = []
    times: List[str] = []
    root_cause_events: List[Dict[str, Any]] = []
    top_scores: List[Dict[str, Any]] = []

    total_evidence = max(
        sum(e.net_evidence for e in selected),
        1e-9,
    )

    for event in selected:
        component = event.component
        reason = event.reason_hypothesis or "unspecified anomaly"
        timestamp = event.time_hypothesis
        if timestamp:
            datetime_str = datetime.datetime.fromtimestamp(
                timestamp
            ).strftime("%Y-%m-%d %H:%M:%S")
        elif state.case.anchors:
            ts = state.case.anchors[0].get("timestamp", 0)
            datetime_str = datetime.datetime.fromtimestamp(
                ts
            ).strftime("%Y-%m-%d %H:%M:%S")
        else:
            datetime_str = state.case.time_window[0] if state.case.time_window else ""

        evidence_ids = [
            item.evidence_id
            for item in event.support_evidence + event.contradictory_evidence
        ]
        confidence = event.net_evidence / total_evidence if total_evidence > 0 else 0.0

        components.append(component)
        reasons.append(reason)
        times.append(datetime_str)
        root_cause_events.append({
            "root cause occurrence datetime": datetime_str,
            "root cause component": component,
            "root cause reason": reason,
        })
        top_scores.append({
            "entity": component,
            "score": round(float(confidence), 4),
            "net_evidence": round(float(event.net_evidence), 4),
            "evidence_ids": evidence_ids,
        })

    top_score = top_scores[0]["score"] if top_scores else 0.0

    return {
        "component": components,
        "reason": reasons,
        "time": times,
        "top_score": round(float(top_score), 4),
        "fault_count": len(selected),
        "root_cause_events": root_cause_events,
        "top_scores": top_scores,
    }
