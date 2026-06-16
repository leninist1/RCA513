"""Conservative gated rerank using shadow LLM candidate judgments.

This module never calls the LLM and never reads raw metric/log/trace data.  It
only consumes D32 outputs plus the JSONL produced by llm_candidate_judge.
"""
from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Mapping

from refute_b_v2_d32.evidence_summarizer import CANONICAL_REASON_BY_BUCKET, REASON_ALIASES_BY_BUCKET
from refute_b_v2_d32.schema import reason_bucket


@dataclass(frozen=True)
class GatedRerankConfig:
    refute_threshold: float = 0.80
    support_threshold: float = 0.75
    min_alt_rank: int = 2
    max_alt_rank: int | None = None
    policy: str = "conservative"
    pairwise_margin_threshold: float = 0.25
    pairwise_top1_max_support: float = 0.50
    allow_pairwise_low_top1_support: bool = False
    require_pairwise_direct_evidence: bool = True
    block_pairwise_sibling_conflict: bool = True
    allow_pairwise_alias_tiebreak: bool = True
    pairwise_alias_min_support: float = 0.40
    block_db_close_active_session_alias: bool = True


def load_judgment_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            rows.append({"parse_ok": False, "error": "invalid_jsonl_row"})
            continue
        rows.append(row if isinstance(row, dict) else {"parse_ok": False, "error": "jsonl_row_not_object"})
    return rows


def apply_gated_rerank(
    *,
    completed: Mapping[int, Mapping[str, Any]],
    judgment_rows: list[Mapping[str, Any]],
    trace_path: Path,
    summary_path: Path,
    config: GatedRerankConfig,
    reason_name_map: Mapping[str, str] | None = None,
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    if config.policy != "conservative":
        raise ValueError(f"unsupported llm rerank policy: {config.policy}")

    grouped = _group_judgments(judgment_rows)
    reason_name_map = dict(reason_name_map or {})
    reranked: dict[int, dict[str, Any]] = {}
    traces: list[dict[str, Any]] = []
    parse_failure_cases = 0
    missing_judgment_cases = 0
    changed_cases = 0

    for row_id in sorted(completed):
        original_row = dict(completed[row_id])
        row = deepcopy(original_row)
        case_id = _case_id(row, row_id)
        judgments = grouped.get(case_id, {})
        case_has_parse_failure = any(not bool(item.get("parse_ok", False)) for item in judgments.values())
        if case_has_parse_failure:
            parse_failure_cases += 1

        trace = _rerank_one_case(
            case_id=case_id,
            row=row,
            judgments=judgments,
            config=config,
            reason_name_map=reason_name_map,
        )
        if trace.get("changed"):
            changed_cases += 1
        if trace.get("reason_for_keep") == "missing_judgment":
            missing_judgment_cases += 1
        traces.append(trace)
        reranked[row_id] = row

    total_cases = len(completed)
    summary = {
        "total_cases": total_cases,
        "changed_cases": changed_cases,
        "unchanged_cases": total_cases - changed_cases,
        "llm_parse_failures": parse_failure_cases,
        "missing_judgment_cases": missing_judgment_cases,
        "changed_ratio": float(changed_cases / total_cases) if total_cases else 0.0,
        "policy": config.policy,
        "refute_threshold": float(config.refute_threshold),
        "support_threshold": float(config.support_threshold),
        "min_alt_rank": int(config.min_alt_rank),
        "max_alt_rank": config.max_alt_rank,
        "rerank_judge": "candidate",
    }
    _write_trace(trace_path, traces)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return reranked, summary


def apply_pairwise_gated_rerank(
    *,
    completed: Mapping[int, Mapping[str, Any]],
    pairwise_rows: list[Mapping[str, Any]],
    trace_path: Path,
    summary_path: Path,
    config: GatedRerankConfig,
    reason_name_map: Mapping[str, str] | None = None,
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    if config.policy != "conservative":
        raise ValueError(f"unsupported llm rerank policy: {config.policy}")

    grouped = _group_pairwise_judgments(pairwise_rows)
    reason_name_map = dict(reason_name_map or {})
    reranked: dict[int, dict[str, Any]] = {}
    traces: list[dict[str, Any]] = []
    parse_failure_cases = 0
    missing_judgment_cases = 0
    changed_cases = 0

    for row_id in sorted(completed):
        original_row = dict(completed[row_id])
        row = deepcopy(original_row)
        case_id = _case_id(row, row_id)
        judgments = grouped.get(case_id, {})
        case_has_parse_failure = any(not bool(item.get("parse_ok", False)) for item in judgments.values())
        if case_has_parse_failure:
            parse_failure_cases += 1

        trace = _pairwise_rerank_one_case(
            case_id=case_id,
            row=row,
            judgments=judgments,
            config=config,
            reason_name_map=reason_name_map,
        )
        if trace.get("changed"):
            changed_cases += 1
        if trace.get("reason_for_keep") in {"missing_judgment", "missing_pairwise_judgment"}:
            missing_judgment_cases += 1
        traces.append(trace)
        reranked[row_id] = row

    total_cases = len(completed)
    summary = {
        "total_cases": total_cases,
        "changed_cases": changed_cases,
        "unchanged_cases": total_cases - changed_cases,
        "llm_parse_failures": parse_failure_cases,
        "llm_pairwise_parse_failures": parse_failure_cases,
        "missing_judgment_cases": missing_judgment_cases,
        "changed_ratio": float(changed_cases / total_cases) if total_cases else 0.0,
        "policy": config.policy,
        "rerank_judge": "pairwise",
        "refute_threshold": float(config.refute_threshold),
        "support_threshold": float(config.support_threshold),
        "pairwise_margin_threshold": float(config.pairwise_margin_threshold),
        "pairwise_top1_max_support": float(config.pairwise_top1_max_support),
        "allow_pairwise_low_top1_support": bool(config.allow_pairwise_low_top1_support),
        "require_pairwise_direct_evidence": bool(config.require_pairwise_direct_evidence),
        "block_pairwise_sibling_conflict": bool(config.block_pairwise_sibling_conflict),
        "allow_pairwise_alias_tiebreak": bool(config.allow_pairwise_alias_tiebreak),
        "pairwise_alias_min_support": float(config.pairwise_alias_min_support),
        "block_db_close_active_session_alias": bool(config.block_db_close_active_session_alias),
        "min_alt_rank": int(config.min_alt_rank),
        "max_alt_rank": config.max_alt_rank,
    }
    _write_trace(trace_path, traces)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return reranked, summary


def _rerank_one_case(
    *,
    case_id: str,
    row: dict[str, Any],
    judgments: Mapping[int, Mapping[str, Any]],
    config: GatedRerankConfig,
    reason_name_map: Mapping[str, str],
) -> dict[str, Any]:
    decisions = _decisions(row)
    original_decision = decisions[0] if decisions else {}
    original_top1 = _decision_summary(original_decision, rank=1)
    top1_judgment_row = judgments.get(1)
    trace_base = {
        "case_id": case_id,
        "changed": False,
        "policy": config.policy,
        "original_top1": original_top1,
        "original_top1_llm": _compact_llm(top1_judgment_row),
        "new_top1": None,
        "new_top1_llm": None,
    }

    if not decisions or "1" not in dict(row.get("prediction", {}) or {}):
        trace_base["reason_for_keep"] = "missing_judgment"
        _attach_trace(row, trace_base)
        return trace_base
    if top1_judgment_row is None:
        trace_base["reason_for_keep"] = "missing_judgment"
        _attach_trace(row, trace_base)
        return trace_base
    if not bool(top1_judgment_row.get("parse_ok", False)):
        trace_base["reason_for_keep"] = "llm_parse_failed"
        _attach_trace(row, trace_base)
        return trace_base

    top1_llm = _judgment(top1_judgment_row)
    if not _is_strong_refute(top1_llm, config.refute_threshold):
        trace_base["reason_for_keep"] = "top1_not_strongly_refuted"
        _attach_trace(row, trace_base)
        return trace_base

    max_alt_rank = int(config.max_alt_rank) if config.max_alt_rank is not None else max(judgments.keys() or [1])
    alternatives: list[tuple[float, float, int, Mapping[str, Any], Mapping[str, Any]]] = []
    for rank in range(max(2, int(config.min_alt_rank)), max_alt_rank + 1):
        judgment_row = judgments.get(rank)
        if judgment_row is None or not bool(judgment_row.get("parse_ok", False)):
            continue
        llm = _judgment(judgment_row)
        if _is_strong_support(llm, config.support_threshold):
            decision = decisions[rank - 1] if rank - 1 < len(decisions) else _decision_from_judgment(judgment_row)
            alternatives.append((
                _safe_float(llm.get("support_score")),
                _safe_float(llm.get("refute_score")),
                rank,
                decision,
                llm,
            ))

    if not alternatives:
        trace_base["reason_for_keep"] = "no_supported_alternative"
        _attach_trace(row, trace_base)
        return trace_base

    alternatives.sort(key=lambda item: (-item[0], item[1], item[2]))
    _, _, old_rank, new_decision, new_llm = alternatives[0]
    new_candidate = dict(new_decision.get("candidate", {}) or {})
    _replace_top1_prediction(row, new_candidate, reason_name_map)
    trace = dict(trace_base)
    trace.update({
        "changed": True,
        "new_top1": _decision_summary(new_decision, rank=old_rank, rank_key="old_rank"),
        "new_top1_llm": _compact_llm({"llm_judgment": new_llm}),
        "reason_for_change": "top1_refuted_and_alternative_supported",
    })
    _attach_trace(row, trace)
    return trace


def _pairwise_rerank_one_case(
    *,
    case_id: str,
    row: dict[str, Any],
    judgments: Mapping[int, Mapping[str, Any]],
    config: GatedRerankConfig,
    reason_name_map: Mapping[str, str],
) -> dict[str, Any]:
    decisions = _decisions(row)
    original_decision = decisions[0] if decisions else {}
    original_top1 = _decision_summary(original_decision, rank=1)
    trace_base = {
        "case_id": case_id,
        "changed": False,
        "policy": config.policy,
        "rerank_judge": "pairwise",
        "original_top1": original_top1,
        "new_top1": None,
        "selected_pairwise_llm": None,
        "pairwise_candidates_considered": len(judgments),
    }

    if not decisions or "1" not in dict(row.get("prediction", {}) or {}):
        trace_base["reason_for_keep"] = "missing_judgment"
        _attach_trace(row, trace_base)
        return trace_base
    if not judgments:
        trace_base["reason_for_keep"] = "missing_pairwise_judgment"
        _attach_trace(row, trace_base)
        return trace_base

    max_alt_rank = int(config.max_alt_rank) if config.max_alt_rank is not None else max(judgments.keys() or [1])
    alternatives_refuted: list[tuple[float, float, float, int, Mapping[str, Any], Mapping[str, Any], str]] = []
    alternatives_alias: list[tuple[float, float, float, int, Mapping[str, Any], Mapping[str, Any], str]] = []
    alternatives_low_support: list[tuple[float, float, float, int, Mapping[str, Any], Mapping[str, Any], str]] = []
    saw_parse_ok = False
    saw_preferred_alternative = False
    saw_strong_support = False
    saw_direct_alternative_evidence = False
    saw_sibling_conflict = False
    saw_alias_tiebreak_candidate = False
    for rank in range(max(2, int(config.min_alt_rank)), max_alt_rank + 1):
        pairwise_row = judgments.get(rank)
        if pairwise_row is None:
            continue
        if not bool(pairwise_row.get("parse_ok", False)):
            continue
        saw_parse_ok = True
        llm = _pairwise_judgment(pairwise_row)
        if str(llm.get("preferred_candidate", "")).lower() == "alternative":
            saw_preferred_alternative = True
        if _safe_float(llm.get("alternative_support_score")) >= float(config.support_threshold):
            saw_strong_support = True
        if _has_direct_alternative_evidence(llm, pairwise_row):
            saw_direct_alternative_evidence = True
        if _has_alternative_sibling_conflict(llm):
            saw_sibling_conflict = True
        decision = decisions[rank - 1] if rank - 1 < len(decisions) else _decision_from_pairwise(pairwise_row)
        reason_for_change = _pairwise_promote_reason(llm, config, pairwise_row)
        alias_reason_for_change = _pairwise_alias_tiebreak_reason(
            pairwise_row,
            llm,
            config,
            top1_decision=original_decision,
            alternative_decision=decision,
        )
        if alias_reason_for_change:
            saw_alias_tiebreak_candidate = True
        if reason_for_change or alias_reason_for_change:
            item = (
                _safe_float(llm.get("relative_margin")),
                _safe_float(llm.get("alternative_support_score")),
                _safe_float(llm.get("top1_refute_score")),
                rank,
                decision,
                llm,
                str(reason_for_change or alias_reason_for_change),
            )
            if reason_for_change == "pairwise_alternative_preferred_top1_refuted":
                alternatives_refuted.append(item)
            elif alias_reason_for_change:
                alternatives_alias.append(item)
            else:
                alternatives_low_support.append(item)

    alternatives = alternatives_refuted or alternatives_alias or alternatives_low_support
    if not alternatives:
        if not saw_parse_ok:
            trace_base["reason_for_keep"] = "llm_parse_failed"
        elif config.allow_pairwise_alias_tiebreak and saw_alias_tiebreak_candidate:
            trace_base["reason_for_keep"] = "pairwise_alias_tiebreak_gate_failed"
        elif not saw_preferred_alternative:
            trace_base["reason_for_keep"] = "no_pairwise_alternative_preferred"
        elif not saw_strong_support:
            trace_base["reason_for_keep"] = "no_supported_alternative"
        elif config.require_pairwise_direct_evidence and not saw_direct_alternative_evidence:
            trace_base["reason_for_keep"] = "no_direct_alternative_evidence"
        elif config.block_pairwise_sibling_conflict and saw_sibling_conflict:
            trace_base["reason_for_keep"] = "stronger_sibling_conflict"
        else:
            trace_base["reason_for_keep"] = "pairwise_margin_or_top1_support_gate_failed"
        _attach_trace(row, trace_base)
        return trace_base

    alternatives.sort(key=lambda item: (-item[0], -item[1], -item[2], item[3]))
    _, _, _, old_rank, new_decision, new_llm, reason_for_change = alternatives[0]
    new_candidate = dict(new_decision.get("candidate", {}) or {})
    _replace_top1_prediction(row, new_candidate, reason_name_map)
    trace = dict(trace_base)
    trace.update({
        "changed": True,
        "new_top1": _decision_summary(new_decision, rank=old_rank, rank_key="old_rank"),
        "selected_pairwise_llm": _compact_pairwise_llm(new_llm),
        "reason_for_change": reason_for_change,
    })
    _attach_trace(row, trace)
    return trace


def _group_judgments(rows: list[Mapping[str, Any]]) -> dict[str, dict[int, Mapping[str, Any]]]:
    grouped: dict[str, dict[int, Mapping[str, Any]]] = {}
    for row in rows:
        case_id = str(row.get("case_id", ""))
        rank = _safe_int(row.get("candidate_rank"))
        if not case_id or rank <= 0:
            continue
        grouped.setdefault(case_id, {})[rank] = row
    return grouped


def _group_pairwise_judgments(rows: list[Mapping[str, Any]]) -> dict[str, dict[int, Mapping[str, Any]]]:
    grouped: dict[str, dict[int, Mapping[str, Any]]] = {}
    for row in rows:
        case_id = str(row.get("case_id", ""))
        rank = _safe_int(row.get("alternative_rank"))
        if not case_id or rank <= 0:
            continue
        grouped.setdefault(case_id, {})[rank] = row
    return grouped


def _replace_top1_prediction(row: dict[str, Any], candidate: Mapping[str, Any], reason_name_map: Mapping[str, str]) -> None:
    prediction = deepcopy(dict(row.get("prediction", {}) or {}))
    top1 = deepcopy(dict(prediction.get("1", {}) or {}))
    top1["root cause component"] = str(candidate.get("component", top1.get("root cause component", "")))
    reason = str(candidate.get("reason", top1.get("root cause reason", "")))
    top1["root cause reason"] = reason_name_map.get(reason, reason)
    prediction["1"] = top1
    row["prediction"] = prediction


def _attach_trace(row: dict[str, Any], trace: Mapping[str, Any]) -> None:
    debug = deepcopy(dict(row.get("debug", {}) or {}))
    debug["llm_gated_rerank"] = dict(trace)
    row["debug"] = debug


def _case_id(row: Mapping[str, Any], row_id: int) -> str:
    d32_debug = (((row.get("debug", {}) or {}).get("d32_result", {}) or {}).get("debug", {}) or {})
    return str(d32_debug.get("case_id") or f"query_{int(row_id):03d}")


def _decisions(row: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    d32_debug = (((row.get("debug", {}) or {}).get("d32_result", {}) or {}).get("debug", {}) or {})
    decisions = d32_debug.get("all_decisions", []) or []
    return [item for item in decisions if isinstance(item, Mapping)]


def _decision_summary(decision: Mapping[str, Any], *, rank: int, rank_key: str = "rank") -> dict[str, Any]:
    candidate = dict(decision.get("candidate", {}) or {})
    return {
        rank_key: int(rank),
        "component": str(candidate.get("component", "")),
        "reason": str(candidate.get("reason", "")),
        "d32_score": _d32_score(decision),
        "d32_rebuttal_score": _safe_float(decision.get("rebuttal_score")),
        "d32_support_strength": _safe_float(decision.get("support_strength")),
        "d32_refute_strength": _safe_float(decision.get("refute_strength")),
    }


def _decision_from_judgment(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "candidate": dict(row.get("candidate", {}) or {}),
        "support_strength": 0.0,
        "refute_strength": 0.0,
        "rebuttal_score": 0.0,
    }


def _decision_from_pairwise(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "candidate": dict(row.get("alternative_candidate", {}) or {}),
        "support_strength": 0.0,
        "refute_strength": 0.0,
        "rebuttal_score": 0.0,
    }


def _d32_score(decision: Mapping[str, Any]) -> float:
    support = _safe_float(decision.get("support_strength"))
    refute = _safe_float(decision.get("refute_strength"))
    rebuttal = _safe_float(decision.get("rebuttal_score"))
    if rebuttal != 0.0:
        return float(-rebuttal)
    return float(support - refute)


def _compact_llm(row: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    llm = _judgment(row)
    return {
        "verdict": str(llm.get("verdict", "uncertain")),
        "support_score": _safe_float(llm.get("support_score")),
        "refute_score": _safe_float(llm.get("refute_score")),
    }


def _judgment(row: Mapping[str, Any]) -> Mapping[str, Any]:
    judgment = row.get("llm_judgment", {}) if isinstance(row, Mapping) else {}
    return judgment if isinstance(judgment, Mapping) else {}


def _pairwise_judgment(row: Mapping[str, Any]) -> Mapping[str, Any]:
    judgment = row.get("llm_pairwise_judgment", {}) if isinstance(row, Mapping) else {}
    return judgment if isinstance(judgment, Mapping) else {}


def _compact_pairwise_llm(judgment: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if judgment is None:
        return None
    return {
        "preferred_candidate": str(judgment.get("preferred_candidate", "uncertain")),
        "top1_support_score": _safe_float(judgment.get("top1_support_score")),
        "top1_refute_score": _safe_float(judgment.get("top1_refute_score")),
        "alternative_support_score": _safe_float(judgment.get("alternative_support_score")),
        "alternative_refute_score": _safe_float(judgment.get("alternative_refute_score")),
        "relative_margin": _safe_float(judgment.get("relative_margin")),
        "alternative_has_direct_evidence": _safe_bool(judgment.get("alternative_has_direct_evidence")),
        "alternative_has_stronger_sibling_conflict": _safe_bool(judgment.get("alternative_has_stronger_sibling_conflict")),
        "promotion_evidence_atom_ids": [str(item) for item in judgment.get("promotion_evidence_atom_ids", []) or []],
    }


def _is_strong_refute(judgment: Mapping[str, Any], threshold: float) -> bool:
    return str(judgment.get("verdict", "")).lower() == "refute" and _safe_float(judgment.get("refute_score")) >= float(threshold)


def _is_strong_support(judgment: Mapping[str, Any], threshold: float) -> bool:
    return str(judgment.get("verdict", "")).lower() == "support" and _safe_float(judgment.get("support_score")) >= float(threshold)


def _pairwise_promote_reason(
    judgment: Mapping[str, Any],
    config: GatedRerankConfig,
    pairwise_row: Mapping[str, Any] | None = None,
) -> str | None:
    if str(judgment.get("preferred_candidate", "")).lower() != "alternative":
        return None
    alternative_support = _safe_float(judgment.get("alternative_support_score"))
    if alternative_support < float(config.support_threshold):
        return None
    if config.require_pairwise_direct_evidence and not _has_direct_alternative_evidence(judgment, pairwise_row):
        return None
    if config.block_pairwise_sibling_conflict and _has_alternative_sibling_conflict(judgment):
        return None
    margin = _safe_float(judgment.get("relative_margin"))
    if margin < float(config.pairwise_margin_threshold):
        return None

    top1_refute = _safe_float(judgment.get("top1_refute_score"))
    if top1_refute >= float(config.refute_threshold):
        return "pairwise_alternative_preferred_top1_refuted"

    top1_support = _safe_float(judgment.get("top1_support_score"))
    if (
        config.allow_pairwise_low_top1_support
        and top1_support <= float(config.pairwise_top1_max_support)
        and (alternative_support - top1_support) >= float(config.pairwise_margin_threshold)
    ):
        return "pairwise_alternative_preferred_with_low_top1_support"
    return None


def _pairwise_alias_tiebreak_reason(
    pairwise_row: Mapping[str, Any],
    judgment: Mapping[str, Any],
    config: GatedRerankConfig,
    *,
    top1_decision: Mapping[str, Any],
    alternative_decision: Mapping[str, Any],
) -> str | None:
    if not config.allow_pairwise_alias_tiebreak:
        return None
    preferred = str(judgment.get("preferred_candidate", "")).strip().lower()
    if preferred == "top1":
        return None
    top1 = dict(pairwise_row.get("top1_candidate", {}) or {})
    alternative = dict(pairwise_row.get("alternative_candidate", {}) or {})
    top1_component = str(top1.get("component", ""))
    alternative_component = str(alternative.get("component", ""))
    if not top1_component or top1_component != alternative_component:
        return None
    top1_canonical = _canonical_reason_for_candidate(top1)
    alternative_canonical = _canonical_reason_for_candidate(alternative)
    top1_reason = str(top1.get("reason", "")).strip()
    alternative_reason = str(alternative.get("reason", "")).strip()
    if not top1_canonical or top1_canonical != alternative_canonical:
        return None
    if top1_reason == top1_canonical or alternative_reason != alternative_canonical:
        return None
    top1_bucket = _reason_bucket_for_candidate(top1)
    alternative_bucket = _reason_bucket_for_candidate(alternative)
    if top1_bucket and alternative_bucket and top1_bucket != alternative_bucket:
        return None
    alternative_support = _safe_float(judgment.get("alternative_support_score"))
    top1_support = _safe_float(judgment.get("top1_support_score"))
    if alternative_support < float(config.pairwise_alias_min_support):
        return None
    if alternative_support + 1e-9 < top1_support:
        return None
    if not _pairwise_role_has_direct_evidence(pairwise_row, judgment, "top1"):
        return None
    if not _pairwise_role_has_direct_evidence(pairwise_row, judgment, "alternative"):
        return None
    if (
        config.block_db_close_active_session_alias
        and _db_close_active_session_alias_blocked(top1, alternative, top1_decision, alternative_decision)
    ):
        return None
    return "pairwise_same_component_canonical_reason_alias_tiebreak"


def _has_direct_alternative_evidence(
    judgment: Mapping[str, Any],
    pairwise_row: Mapping[str, Any] | None = None,
) -> bool:
    if not _safe_bool(judgment.get("alternative_has_direct_evidence")):
        return False
    atom_ids = [str(item) for item in judgment.get("promotion_evidence_atom_ids", []) or []]
    if not atom_ids:
        return False
    if pairwise_row is None:
        return True
    return bool(_valid_promotion_atom_ids_from_pairwise_row(pairwise_row, atom_ids))


def _has_alternative_sibling_conflict(judgment: Mapping[str, Any]) -> bool:
    return _safe_bool(judgment.get("alternative_has_stronger_sibling_conflict"))


def _pairwise_role_has_direct_evidence(
    pairwise_row: Mapping[str, Any],
    judgment: Mapping[str, Any],
    role: str,
) -> bool:
    if _safe_bool(judgment.get(f"{role}_has_direct_evidence")):
        return True
    label = _pairwise_label_for_role(pairwise_row, role)
    if not label:
        return False
    pairwise_input = dict(pairwise_row.get("pairwise_input", {}) or {})
    cards = dict(pairwise_input.get("evidence_summary_cards", {}) or {})
    card = dict(cards.get(label, {}) or {})
    candidate_summary = dict(card.get("candidate_summary", {}) or {})
    atoms = candidate_summary.get("candidate_direct_evidence_atoms", []) or []
    if any(isinstance(atom, Mapping) for atom in atoms):
        return True
    for field in (
        "candidate_positive_evidence_summary",
        "metric_support_summary",
        "log_support_summary",
        "trace_support_summary",
    ):
        if candidate_summary.get(field):
            return True
    return False


def _pairwise_label_for_role(pairwise_row: Mapping[str, Any], role: str) -> str:
    for label in ("candidate_a", "candidate_b"):
        if str(pairwise_row.get(f"{label}_role", "")) == role:
            return label
    return ""


def _valid_promotion_atom_ids_from_pairwise_row(
    pairwise_row: Mapping[str, Any],
    atom_ids: list[str],
) -> list[str]:
    alt_label = _pairwise_label_for_role(pairwise_row, "alternative")
    if not alt_label:
        return []
    pairwise_input = dict(pairwise_row.get("pairwise_input", {}) or {})
    cards = dict(pairwise_input.get("evidence_summary_cards", {}) or {})
    alt_card = dict(cards.get(alt_label, {}) or {})
    candidate_summary = dict(alt_card.get("candidate_summary", {}) or {})
    valid = {
        str(atom.get("atom_id"))
        for atom in candidate_summary.get("candidate_direct_evidence_atoms", []) or []
        if isinstance(atom, Mapping) and atom.get("promotion_eligible") is True
    }
    return [atom_id for atom_id in atom_ids if atom_id in valid]


def _canonical_reason_for_candidate(candidate: Mapping[str, Any]) -> str:
    explicit = str(candidate.get("canonical_reason", "")).strip()
    if explicit:
        return explicit
    reason = str(candidate.get("reason", "")).strip()
    reason_lower = reason.lower()
    for bucket, aliases in REASON_ALIASES_BY_BUCKET.items():
        if reason_lower in {str(alias).lower() for alias in aliases}:
            return str(CANONICAL_REASON_BY_BUCKET.get(bucket, reason))
    bucket = _reason_bucket_for_candidate(candidate)
    return str(CANONICAL_REASON_BY_BUCKET.get(bucket, reason)).strip()


def _reason_bucket_for_candidate(candidate: Mapping[str, Any]) -> str:
    explicit = str(candidate.get("reason_bucket", "")).strip()
    if explicit:
        return explicit
    reason = str(candidate.get("reason", "")).strip()
    if not reason:
        return ""
    return str(reason_bucket(reason))


def _db_close_active_session_alias_blocked(
    top1_candidate: Mapping[str, Any],
    alternative_candidate: Mapping[str, Any],
    top1_decision: Mapping[str, Any],
    alternative_decision: Mapping[str, Any],
) -> bool:
    top1_reason = str(top1_candidate.get("reason", "")).strip().lower()
    alternative_reason = str(alternative_candidate.get("reason", "")).strip().lower()
    if top1_reason != "db close" or alternative_reason != "db connection limit":
        return False
    names = {
        name
        for decision in (top1_decision, alternative_decision)
        for name in _candidate_metric_example_names(decision)
    }
    if "Sess_Active" not in names:
        return False
    connection_limit_names = {"Login_Per_Sec", "TPS_Per_Sec", "Exec_Per_Sec"}
    return not bool(names & connection_limit_names)


def _candidate_metric_example_names(decision: Mapping[str, Any]) -> list[str]:
    candidate = dict(decision.get("candidate", {}) or {})
    details = dict(candidate.get("details", {}) or {})
    names: list[str] = []
    for example in details.get("examples", []) or []:
        if not isinstance(example, Mapping):
            continue
        name = str(example.get("name", "")).strip()
        if name:
            names.append(name)
    return names


def _write_trace(path: Path, traces: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for trace in traces:
            f.write(json.dumps(trace, ensure_ascii=False, sort_keys=True) + "\n")


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    if isinstance(value, (int, float)):
        return bool(value)
    return False
