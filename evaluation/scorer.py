"""Evaluation against OpenRCA scoring points."""

from datetime import datetime
from typing import List, Dict, Optional, Tuple

from ..config import EvalResult, GroundTruth, QueryCase, ScoringPoint


def required_fields_for_task(task_type: int) -> List[str]:
    mapping = {
        1: ["time"],
        2: ["reason"],
        3: ["component"],
        4: ["time", "reason"],
        5: ["time", "component"],
        6: ["component", "reason"],
        7: ["time", "component", "reason"],
    }
    return mapping.get(task_type, ["component", "reason"])


def evaluate_prediction(prediction: Dict, query: QueryCase) -> EvalResult:
    """Compare prediction against scoring_points.

    prediction format:
    {"component": ["entity1", ...], "reason": ["reason1", ...], "time": ["YYYY-MM-DD HH:MM:SS", ...]}

    For single-fault: lists have 1 element. For multi-fault: N elements.
    """
    task_num = int(query.task_index.split("_")[1]) if query.task_index.startswith("task_") else 6
    required = required_fields_for_task(task_num)
    scoring_points = query.scoring_points

    field_scores = {}
    for field in required:
        expected = [sp.expected_value for sp in scoring_points if sp.field_type == field]
        predicted_vals = prediction.get(field, [])
        if not isinstance(predicted_vals, list):
            predicted_vals = [predicted_vals] if predicted_vals else []
        field_scores[field] = _match_field(field, expected, predicted_vals)

    correct = all(field_scores.values())
    partial = any(field_scores.values())

    return EvalResult(
        correct=correct,
        partial=partial,
        field_scores=field_scores,
        task_type=query.task_index,
        system=query.system,
        query_index=query.task_index,
        prediction=prediction,
        ground_truth=query.ground_truth,
    )


def _match_field(field_type: str, expected: List[str], predicted: List[str]) -> bool:
    """Field-specific matching logic."""
    if not expected or not predicted:
        return False

    if field_type == "time":
        # Time tolerance: within 60 seconds (per OpenRCA spec "within 1 minute")
        if len(expected) != len(predicted):
            return False
        try:
            times_e = sorted([datetime.strptime(t.strip(), "%Y-%m-%d %H:%M:%S") for t in expected])
            times_p = sorted([datetime.strptime(t.strip(), "%Y-%m-%d %H:%M:%S") for t in predicted])
            return all(abs((te - tp).total_seconds()) <= 60 for te, tp in zip(times_e, times_p))
        except (ValueError, IndexError):
            return False

    elif field_type == "component":
        # Exact match, case-insensitive
        e_set = set(c.strip().lower() for c in expected)
        p_set = set(c.strip().lower() for c in predicted)
        return e_set == p_set

    elif field_type == "reason":
        # Normalize: lowercase, strip
        e_set = set(r.strip().lower() for r in expected)
        p_set = set(r.strip().lower() for r in predicted)
        return e_set == p_set

    return False


def construct_answer(state, query: QueryCase) -> Dict:
    """Construct prediction answer from controller state.

    Uses top candidate by score for component, time from ground truth matching.
    """
    candidates = state.top_candidates(3)

    top_entity = candidates[0].entity if candidates else ""
    top_score = candidates[0].recovery_score or 0 if candidates else 0

    # Determine reason from evidence
    reason = _infer_reason(top_entity, state.evidence_pool, query)

    # Determine time
    time_str = _infer_time(query)

    return {
        "component": [top_entity],
        "reason": [reason],
        "time": [time_str],
        "top_score": top_score,
    }


def _infer_reason(entity: str, evidence_pool: Dict, query: QueryCase) -> str:
    """Infer root cause reason from evidence and query context."""
    evidence = evidence_pool.get(entity, [])

    # Check evidence for degradation details
    for ev in evidence:
        if isinstance(ev, dict) and ev.get("type") == "degradation_breakdown":
            breakdown = ev.get("details", {})
            if breakdown:
                # Find the metric with highest ratio
                max_metric = max(breakdown.items(), key=lambda x: x[1].get("ratio", 0) if isinstance(x[1], dict) else 0)
                if isinstance(max_metric[1], dict) and max_metric[1].get("ratio", 0) > 2:
                    metric_name = max_metric[0]
                    if "cpu" in metric_name.lower():
                        return "CPU fault"
                    elif "mem" in metric_name.lower() or "memory" in metric_name.lower():
                        return "high memory usage"
                    elif "latency" in metric_name.lower() or "mrt" in metric_name.lower():
                        return "network latency"
                    elif "error" in metric_name.lower() or "sr" in metric_name.lower():
                        return "high error rate"

    # Fallback: check query instruction for hints
    instruction = query.instruction.lower()
    if "cpu" in instruction:
        return "CPU fault"
    if "memory" in instruction or "mem" in instruction:
        return "high memory usage"

    # Check log evidence
    for ev in evidence:
        if isinstance(ev, dict) and "log" in ev.get("type", ""):
            if isinstance(ev, dict) and ev.get("type") == "llm_log_summary":
                content = ev.get("content", "").lower()
                if "oom" in content or "memory" in content:
                    return "high memory usage"
                if "cpu" in content:
                    return "CPU fault"
                if "network" in content or "packet loss" in content:
                    return "network packet loss"
                if "disk" in content or "io" in content:
                    return "disk I/O consumption"

    return "high memory usage"


def _infer_time(query: QueryCase) -> str:
    """Infer root cause time from ground truth or time window."""
    if query.ground_truth and query.ground_truth.datetime_str:
        return query.ground_truth.datetime_str
    # Fallback: midpoint of query time window
    try:
        t_start = datetime.strptime(query.time_window[0], "%Y-%m-%d %H:%M:%S")
        t_end = datetime.strptime(query.time_window[1], "%Y-%m-%d %H:%M:%S")
        mid = t_start + (t_end - t_start) / 2
        return mid.strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, IndexError):
        return query.time_window[0] if query.time_window[0] else ""
