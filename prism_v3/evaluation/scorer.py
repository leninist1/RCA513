"""Evaluation against OpenRCA scoring points."""

from datetime import datetime
import itertools
from typing import List, Dict, Optional, Tuple

from ..config import EvalResult, GroundTruth, QueryCase, ScoringPoint
from .reason_normalizer import explain_bank_reason


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

    official_score, official_passing, official_failing = official_evaluate_prediction(
        prediction, query
    )
    correct = official_score == 1.0
    partial = official_score > 0.0

    return EvalResult(
        correct=correct,
        partial=partial,
        field_scores=field_scores,
        official_score=official_score,
        official_passing=official_passing,
        official_failing=official_failing,
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


def official_evaluate_prediction(
    prediction: Dict, query: QueryCase
) -> Tuple[float, List[str], List[str]]:
    """Mirror microsoft/OpenRCA `main/evaluate.py` for internal predictions.

    The official evaluator scores per root-cause element, requires the number
    of predicted failure objects to match the scoring-point failure count, and
    chooses the best permutation for multi-fault cases.
    """
    scoring_points = list(getattr(query, "scoring_points", []) or [])
    expected_by_field: Dict[str, Dict[int, str]] = {
        "component": {},
        "reason": {},
        "time": {},
    }
    for sp in scoring_points:
        field = getattr(sp, "field_type", "")
        if field not in expected_by_field:
            continue
        expected_by_field[field][int(getattr(sp, "rank", 0) or 0)] = str(
            getattr(sp, "expected_value", "") or ""
        )

    components = _rank_ordered_values(expected_by_field["component"])
    reasons = _rank_ordered_values(expected_by_field["reason"])
    times = _rank_ordered_values(expected_by_field["time"])
    scoring_length = max(len(components), len(reasons), len(times))
    scores_num = len(components) + len(reasons) + len(times)
    if scoring_length <= 0 or scores_num <= 0:
        return 0.0, [], []

    predicted_records = _prediction_records(prediction)
    if len(predicted_records) != scoring_length:
        return 0.0, [], list(components + reasons + times)

    best_score = -1
    best_passing: List[str] = []
    for perm in itertools.permutations(predicted_records):
        current_score = 0
        current_passing: List[str] = []
        for idx in range(scoring_length):
            pred = perm[idx]
            if len(components) == scoring_length:
                expected = components[idx]
                if pred.get("component", "") == expected:
                    current_score += 1
                    current_passing.append(expected)
            if len(reasons) == scoring_length:
                expected = reasons[idx]
                if pred.get("reason", "") == expected:
                    current_score += 1
                    current_passing.append(expected)
            if len(times) == scoring_length:
                expected = times[idx]
                if _time_difference(expected, pred.get("time", "")):
                    current_score += 1
                    current_passing.append(expected)
        if current_score > best_score:
            best_score = current_score
            best_passing = current_passing

    failing = list(set(components + reasons + times) - set(best_passing))
    return round(best_score / scores_num, 2), best_passing, failing


def _rank_ordered_values(values_by_rank: Dict[int, str]) -> List[str]:
    return [value for _rank, value in sorted(values_by_rank.items())]


def _prediction_records(prediction: Dict) -> List[Dict[str, str]]:
    fields = {
        "component": prediction.get("component", []),
        "reason": prediction.get("reason", []),
        "time": prediction.get("time", []),
    }
    normalized: Dict[str, List[str]] = {}
    for field, values in fields.items():
        if isinstance(values, list):
            normalized[field] = [str(value or "") for value in values]
        elif values:
            normalized[field] = [str(values)]
        else:
            normalized[field] = []
    length = max([len(values) for values in normalized.values()] or [0])
    records: List[Dict[str, str]] = []
    for idx in range(length):
        records.append(
            {
                "component": normalized["component"][idx]
                if idx < len(normalized["component"])
                else "",
                "reason": normalized["reason"][idx]
                if idx < len(normalized["reason"])
                else "",
                "time": normalized["time"][idx]
                if idx < len(normalized["time"])
                else "",
            }
        )
    return records


def _time_difference(expected: str, predicted: str) -> bool:
    try:
        time_e = datetime.strptime(str(expected).strip(), "%Y-%m-%d %H:%M:%S")
        time_p = datetime.strptime(str(predicted).strip(), "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return False
    return abs((time_e - time_p).total_seconds()) <= 60


def construct_answer(state, query: QueryCase) -> Dict:
    """Construct prediction answer from controller state.

    Uses HypothesisList confidence scores when available (new Tool-based
    architecture), falling back to engine scores + verification modifiers.
    """
    # New: HypothesisList-based (Tool architecture)
    if hasattr(state, 'hypothesis_list') and state.hypothesis_list.n_active > 0:
        hl = state.hypothesis_list
        best = hl.active[0]
        best_entity = best.entity
        top_score = best.confidence

        # Infer reason from evidence + verdicts
        raw_reason = _raw_reason_from_hypothesis(best, state.evidence_pool, query)
        reason_debug = explain_bank_reason(
            entity=best_entity,
            evidence_pool=state.evidence_pool,
            query=query,
            raw_reason=raw_reason,
        )
        state.reason_debug = reason_debug
        reason = reason_debug["canonical_reason"]
        time_str = _infer_time(query)

        return {
            "component": [best_entity],
            "reason": [reason],
            "time": [time_str],
            "top_score": top_score,
        }

    # Legacy: engine scores + verification modifiers
    entities = list(state.candidate_scores.keys())
    if not entities:
        state.reason_debug = {
            "raw_reason": "unknown",
            "canonical_reason": "high memory usage",
            "matched_rule": "no_candidates",
            "evidence": [],
        }
        return {"component": [""], "reason": ["high memory usage"],
                "time": [_infer_time(query)], "top_score": 0}

    composite = {}
    for entity in entities:
        engine_score = state.candidate_scores.get(entity, 0)
        supports = state.support_count(entity) if hasattr(state, 'support_count') else 0
        refutes = state.refute_count(entity) if hasattr(state, 'refute_count') else 0
        composite[entity] = engine_score + 0.05 * (supports - refutes)

    best_entity = max(composite, key=composite.get)
    top_score = state.candidate_scores.get(best_entity, 0)
    raw_reason = _raw_reason(best_entity, state.evidence_pool, query)
    reason_debug = explain_bank_reason(
        entity=best_entity,
        evidence_pool=state.evidence_pool,
        query=query,
        raw_reason=raw_reason,
    )
    state.reason_debug = reason_debug
    reason = reason_debug["canonical_reason"]
    time_str = _infer_time(query)

    return {
        "component": [best_entity],
        "reason": [reason],
        "time": [time_str],
        "top_score": top_score,
    }


def _infer_reason_from_hypothesis(h, evidence_pool, query) -> str:
    """Infer reason from a Hypothesis object's verdicts and evidence."""
    raw_reason = _raw_reason_from_hypothesis(h, evidence_pool, query)
    return explain_bank_reason(
        entity=h.entity,
        evidence_pool=evidence_pool,
        query=query,
        raw_reason=raw_reason,
    )["canonical_reason"]


def _raw_reason_from_hypothesis(h, evidence_pool, query) -> str:
    """Infer a broad raw reason family from a Hypothesis object."""
    # Check semantic anchor verdicts first
    for dim, v in h.verdicts.items():
        if hasattr(v, 'evidence'):
            for ev in (v.evidence if isinstance(v.evidence, list) else []):
                if isinstance(ev, dict) and 'best_family' in ev:
                    fam = ev['best_family']
                    mapping = {'jvm': 'JVM Out of Memory', 'cpu': 'CPU fault',
                               'memory': 'high memory usage', 'network': 'network fault',
                               'db': 'db fault', 'process': 'process termination'}
                    return mapping.get(fam, 'high memory usage')

    # Check evidence pool
    evidence = evidence_pool.get(h.entity, [])
    for ev in evidence:
        if isinstance(ev, dict):
            if ev.get("type") == "degradation_breakdown":
                details = ev.get("details", {})
                if details:
                    max_metric = max(details.items(),
                                     key=lambda x: x[1].get("ratio", 0) if isinstance(x[1], dict) else 0)
                    if isinstance(max_metric[1], dict) and max_metric[1].get("ratio", 0) > 2:
                        mn = max_metric[0].lower()
                        if "cpu" in mn: return "CPU fault"
                        if "mem" in mn or "memory" in mn: return "high memory usage"
                        if "latency" in mn or "mrt" in mn: return "network latency"
                        if "error" in mn or "sr" in mn: return "high error rate"

    instruction = query.instruction.lower()
    if "cpu" in instruction: return "CPU fault"
    if "memory" in instruction or "mem" in instruction: return "high memory usage"

    return "high memory usage"


def _infer_reason(entity: str, evidence_pool: Dict, query: QueryCase) -> str:
    """Infer root cause reason from evidence and query context."""
    raw_reason = _raw_reason(entity, evidence_pool, query)
    return explain_bank_reason(
        entity=entity,
        evidence_pool=evidence_pool,
        query=query,
        raw_reason=raw_reason,
    )["canonical_reason"]


def _raw_reason(entity: str, evidence_pool: Dict, query: QueryCase) -> str:
    """Infer a broad raw reason family from evidence and query context."""
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
    """Infer root cause time from public query context only."""
    # Fallback: midpoint of query time window
    try:
        t_start = datetime.strptime(query.time_window[0], "%Y-%m-%d %H:%M:%S")
        t_end = datetime.strptime(query.time_window[1], "%Y-%m-%d %H:%M:%S")
        mid = t_start + (t_end - t_start) / 2
        return mid.strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, IndexError):
        return query.time_window[0] if query.time_window[0] else ""
