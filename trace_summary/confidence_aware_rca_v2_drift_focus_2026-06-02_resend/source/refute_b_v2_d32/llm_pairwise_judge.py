"""Rank-blind pairwise LLM judge for conservative D32 reranking.

The judge only receives transformed Evidence Summary Cards.  The transform
removes D32 rank/score hints so the model compares candidate evidence rather
than echoing the original order.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping

from refute_b_v2_d32.llm_candidate_judge import (
    DEFAULT_RATIONALE_MAX_CHARS,
    LLMJudgeConfig,
    _chat_completions_url,
    _clamp01,
    _extract_message_content,
    _preview,
    _safe_int,
    _to_bool,
    _truncate_error,
    _truncate_rationale,
)


SYSTEM_PROMPT = """You are a constrained pairwise root-cause evidence judge.
You do not predict new root causes.
You only compare the two given candidates.
You must return JSON only.
If evidence is weak or ambiguous, return tie or uncertain.
Do not penalize a candidate only because one modality is unavailable.
Do not use external knowledge.
Do not infer from case id or file names.
The input is rank-blind: candidate_a and candidate_b are arbitrary labels, not rank signals.
Support means direct evidence is consistent with the candidate, not proof that it is globally best.
Treat competing_evidence_summary as case context, not direct refutation by itself.
Treat missing_evidence_summary as unavailable evidence, not direct refutation.
Promotion must be justified by promotion_eligible=true candidate_direct_evidence_atoms from the preferred candidate."""

USER_PROMPT_PREFIX = (
    "Compare the two candidates using only the rank-blind Evidence Summary Cards. "
    "Prefer a candidate only when its direct evidence is clearly stronger and the other candidate is weak or refuted. "
    "Prioritize candidate_positive_evidence_summary, metric_support_summary, trace_support_summary, and topology_context_summary. "
    "Use candidate_direct_evidence_atoms to decide whether a promotion has direct evidence. "
    "If you prefer a candidate for promotion, list only atom_id values whose promotion_eligible field is true. "
    "A component_only atom is context for affectedness, but it is not sufficient for promotion by itself. "
    "If the preferred candidate has same_reason_sibling_context.has_stronger_sibling=true, mark its stronger sibling conflict as true. "
    "Do not convert disabled or unavailable modalities into counter-evidence. "
    "Do not treat competing evidence from another component as direct refutation unless the card also gives direct candidate-level counter evidence. "
    "Return only JSON matching this schema: "
    '{"preferred_candidate":"candidate_a|candidate_b|tie|uncertain",'
    '"candidate_a_support_score":0.0,"candidate_a_refute_score":0.0,'
    '"candidate_b_support_score":0.0,"candidate_b_refute_score":0.0,'
    '"relative_margin":0.0,'
    '"candidate_a_has_direct_evidence":false,"candidate_b_has_direct_evidence":false,'
    '"candidate_a_has_stronger_sibling_conflict":false,"candidate_b_has_stronger_sibling_conflict":false,'
    '"promotion_evidence_atom_ids":["atom_1"],'
    '"evidence_atoms":[{"modality":"metric|log|trace|topology|case",'
    '"candidate":"candidate_a|candidate_b|both","effect":"support|refute|neutral",'
    '"summary":"short evidence atom"}],'
    '"rationale":"one short sentence"}\n\n'
)

MAX_EVIDENCE_ATOMS = 6
ATOM_SUMMARY_MAX_CHARS = 160


def default_pairwise_judgment() -> dict[str, Any]:
    return {
        "preferred_candidate": "uncertain",
        "preferred_candidate_label": "uncertain",
        "candidate_a_support_score": 0.0,
        "candidate_a_refute_score": 0.0,
        "candidate_b_support_score": 0.0,
        "candidate_b_refute_score": 0.0,
        "candidate_a_has_direct_evidence": False,
        "candidate_b_has_direct_evidence": False,
        "candidate_a_has_stronger_sibling_conflict": False,
        "candidate_b_has_stronger_sibling_conflict": False,
        "top1_support_score": 0.0,
        "top1_refute_score": 0.0,
        "alternative_support_score": 0.0,
        "alternative_refute_score": 0.0,
        "top1_has_direct_evidence": False,
        "alternative_has_direct_evidence": False,
        "top1_has_stronger_sibling_conflict": False,
        "alternative_has_stronger_sibling_conflict": False,
        "relative_margin": 0.0,
        "promotion_evidence_atom_ids": [],
        "evidence_atoms": [],
        "rationale": "LLM pairwise judgment unavailable.",
    }


def build_pairwise_requests(
    cards: list[Mapping[str, Any]],
    *,
    min_alt_rank: int = 2,
    max_alt_rank: int | None = None,
) -> list[dict[str, Any]]:
    """Build rank-blind top1-vs-alternative requests from summary cards."""

    grouped: dict[str, dict[int, Mapping[str, Any]]] = {}
    for card in cards:
        case_id = str(card.get("case_id", ""))
        candidate_summary = dict(card.get("candidate_summary", {}) or {})
        rank = _safe_int(candidate_summary.get("candidate_rank"))
        if case_id and rank > 0:
            grouped.setdefault(case_id, {})[rank] = card

    requests: list[dict[str, Any]] = []
    start_rank = max(2, int(min_alt_rank))
    for case_id in sorted(grouped):
        ranked = grouped[case_id]
        top1 = ranked.get(1)
        if top1 is None:
            continue
        end_rank = int(max_alt_rank) if max_alt_rank is not None else max(ranked.keys() or [1])
        for alt_rank in range(start_rank, end_rank + 1):
            alternative = ranked.get(alt_rank)
            if alternative is None:
                continue
            pairwise_input, role_by_label = _build_pairwise_input_and_roles(top1, alternative)
            requests.append({
                "case_id": case_id,
                "alternative_rank": alt_rank,
                "top1_candidate": _candidate_from_card(top1),
                "alternative_candidate": _candidate_from_card(alternative),
                "candidate_a_role": role_by_label["candidate_a"],
                "candidate_b_role": role_by_label["candidate_b"],
                "pairwise_input": pairwise_input,
            })
    return requests


def build_pairwise_input(top1_card: Mapping[str, Any], alternative_card: Mapping[str, Any]) -> dict[str, Any]:
    """Build the LLM input from two Evidence Summary Cards after rank blinding."""

    pairwise_input, _ = _build_pairwise_input_and_roles(top1_card, alternative_card)
    return pairwise_input


def _build_pairwise_input_and_roles(
    top1_card: Mapping[str, Any],
    alternative_card: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, str]]:
    top1_candidate = _candidate_from_card(top1_card)
    alternative_candidate = _candidate_from_card(alternative_card)
    top1_key = json.dumps(top1_candidate, ensure_ascii=False, sort_keys=True)
    alternative_key = json.dumps(alternative_candidate, ensure_ascii=False, sort_keys=True)
    if alternative_key < top1_key:
        candidate_a = alternative_candidate
        candidate_b = top1_candidate
        candidate_a_card = alternative_card
        candidate_b_card = top1_card
        role_by_label = {"candidate_a": "alternative", "candidate_b": "top1"}
    else:
        candidate_a = top1_candidate
        candidate_b = alternative_candidate
        candidate_a_card = top1_card
        candidate_b_card = alternative_card
        role_by_label = {"candidate_a": "top1", "candidate_b": "alternative"}

    pairwise_input = {
        "task": "compare_candidate_root_cause_evidence",
        "candidate_a": candidate_a,
        "candidate_b": candidate_b,
        "evidence_summary_cards": {
            "candidate_a": rank_blind_card(candidate_a_card),
            "candidate_b": rank_blind_card(candidate_b_card),
        },
    }
    return pairwise_input, role_by_label


def rank_blind_card(card: Mapping[str, Any]) -> dict[str, Any]:
    """Return a card view without case id, D32 rank, or D32 score hints."""

    out = deepcopy(dict(card))
    out.pop("case_id", None)
    candidate_summary = dict(out.get("candidate_summary", {}) or {})
    candidate_summary.pop("candidate_rank", None)
    candidate_summary.pop("d32_score_summary", None)
    counter = []
    for item in candidate_summary.get("counter_evidence_summary", []) or []:
        if not isinstance(item, Mapping):
            continue
        if str(item.get("type", "")) == "other_candidate_ranked_higher":
            continue
        counter.append(dict(item))
    candidate_summary["counter_evidence_summary"] = counter
    out["candidate_summary"] = candidate_summary
    return out


def judge_pairwise(request: Mapping[str, Any], config: LLMJudgeConfig) -> dict[str, Any]:
    try:
        content = _call_openai_compatible_pairwise(request, config)
    except Exception as exc:
        return {
            "llm_pairwise_judgment": default_pairwise_judgment(),
            "parse_ok": False,
            "error": _truncate_error(str(exc)),
            "raw_response_preview": None,
        }

    judgment, parse_ok, error = parse_pairwise_judgment(content, rationale_max_chars=config.rationale_max_chars)
    judgment = _map_judgment_to_roles(judgment, request)
    return {
        "llm_pairwise_judgment": judgment,
        "parse_ok": parse_ok,
        "error": error,
        "raw_response_preview": _preview(content) if not parse_ok else None,
    }


def write_pairwise_judgments_jsonl(
    *,
    requests: list[Mapping[str, Any]],
    out_path: Path,
    config: LLMJudgeConfig,
) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with out_path.open("w", encoding="utf-8") as f:
        if int(config.concurrency) <= 1:
            for request in requests:
                row = _pairwise_output_row(request, config)
                f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                f.flush()
                written += 1
        else:
            with ThreadPoolExecutor(max_workers=max(1, int(config.concurrency))) as executor:
                for row in executor.map(lambda request: _pairwise_output_row(request, config), requests):
                    f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                    f.flush()
                    written += 1
    return written


def parse_pairwise_judgment(
    raw_content: str,
    *,
    rationale_max_chars: int = DEFAULT_RATIONALE_MAX_CHARS,
) -> tuple[dict[str, Any], bool, str | None]:
    try:
        parsed = json.loads(raw_content)
    except Exception as exc:
        return default_pairwise_judgment(), False, f"json_parse_error: {_truncate_error(str(exc))}"
    if not isinstance(parsed, dict):
        return default_pairwise_judgment(), False, "json_parse_error: response is not an object"

    judgment = default_pairwise_judgment()
    errors: list[str] = []
    preferred = str(parsed.get("preferred_candidate", judgment["preferred_candidate"])).strip().lower()
    if preferred not in {"candidate_a", "candidate_b", "top1", "alternative", "tie", "uncertain"}:
        errors.append(f"invalid_preferred_candidate:{preferred[:40]}")
        preferred = "uncertain"
    judgment["preferred_candidate"] = preferred
    judgment["preferred_candidate_label"] = preferred

    for field in (
        "candidate_a_support_score",
        "candidate_a_refute_score",
        "candidate_b_support_score",
        "candidate_b_refute_score",
        "top1_support_score",
        "top1_refute_score",
        "alternative_support_score",
        "alternative_refute_score",
        "relative_margin",
    ):
        judgment[field] = _clamp01(parsed.get(field, judgment[field]))

    for field in (
        "candidate_a_has_direct_evidence",
        "candidate_b_has_direct_evidence",
        "candidate_a_has_stronger_sibling_conflict",
        "candidate_b_has_stronger_sibling_conflict",
        "top1_has_direct_evidence",
        "alternative_has_direct_evidence",
        "top1_has_stronger_sibling_conflict",
        "alternative_has_stronger_sibling_conflict",
    ):
        judgment[field] = _to_bool(parsed.get(field, judgment[field]))

    judgment["promotion_evidence_atom_ids"] = _sanitize_atom_ids(parsed.get("promotion_evidence_atom_ids", []))
    judgment["evidence_atoms"] = _sanitize_atoms(parsed.get("evidence_atoms", []))
    judgment["rationale"] = _truncate_rationale(parsed.get("rationale", judgment["rationale"]), rationale_max_chars)
    return judgment, True, ";".join(errors) if errors else None


def _map_judgment_to_roles(judgment: Mapping[str, Any], request: Mapping[str, Any]) -> dict[str, Any]:
    mapped = dict(judgment)
    role_by_label = {
        "candidate_a": str(request.get("candidate_a_role", "")),
        "candidate_b": str(request.get("candidate_b_role", "")),
    }
    preferred = str(mapped.get("preferred_candidate", "uncertain")).strip().lower()
    if preferred in role_by_label:
        mapped["preferred_candidate_label"] = preferred
        mapped["preferred_candidate"] = role_by_label[preferred] or "uncertain"
    else:
        mapped["preferred_candidate_label"] = preferred

    for label, role in role_by_label.items():
        if role not in {"top1", "alternative"}:
            continue
        mapped[f"{role}_support_score"] = _clamp01(mapped.get(f"{label}_support_score"))
        mapped[f"{role}_refute_score"] = _clamp01(mapped.get(f"{label}_refute_score"))
        mapped[f"{role}_has_direct_evidence"] = _to_bool(mapped.get(f"{label}_has_direct_evidence"))
        mapped[f"{role}_has_stronger_sibling_conflict"] = _to_bool(mapped.get(f"{label}_has_stronger_sibling_conflict"))
    valid_atom_ids = _valid_promotion_atom_ids(mapped, request)
    mapped["promotion_evidence_atom_ids"] = valid_atom_ids
    if valid_atom_ids:
        mapped["alternative_has_direct_evidence"] = True
    elif str(mapped.get("preferred_candidate", "")).lower() == "alternative":
        mapped["alternative_has_direct_evidence"] = False
    return mapped


def _pairwise_output_row(request: Mapping[str, Any], config: LLMJudgeConfig) -> dict[str, Any]:
    row = {
        "case_id": str(request.get("case_id", "")),
        "alternative_rank": _safe_int(request.get("alternative_rank")),
        "top1_candidate": dict(request.get("top1_candidate", {}) or {}),
        "alternative_candidate": dict(request.get("alternative_candidate", {}) or {}),
        "candidate_a_role": str(request.get("candidate_a_role", "")),
        "candidate_b_role": str(request.get("candidate_b_role", "")),
        "pairwise_input": dict(request.get("pairwise_input", {}) or {}),
    }
    try:
        result = judge_pairwise(request, config)
    except Exception as exc:
        result = {
            "llm_pairwise_judgment": default_pairwise_judgment(),
            "parse_ok": False,
            "error": _truncate_error(str(exc)),
            "raw_response_preview": None,
        }
    row.update(result)
    return row


def _call_openai_compatible_pairwise(request: Mapping[str, Any], config: LLMJudgeConfig) -> str:
    import requests

    if config.provider != "openai_compatible":
        raise ValueError(f"unsupported llm provider: {config.provider}")
    if not config.model:
        raise ValueError("missing --llm-model")
    if not config.base_url:
        raise ValueError("missing --llm-base-url")
    api_key = os.environ.get(config.api_key_env or "")
    if not api_key:
        raise ValueError(f"missing API key env var: {config.api_key_env}")

    url = _chat_completions_url(config.base_url)
    body = {
        "model": config.model,
        "temperature": float(config.temperature),
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": USER_PROMPT_PREFIX + json.dumps(request.get("pairwise_input", {}), ensure_ascii=False, sort_keys=True),
            },
        ],
        "response_format": {"type": "json_object"},
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    attempts = max(0, int(config.max_retries)) + 1
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            response = requests.post(url, headers=headers, json=body, timeout=float(config.timeout_sec))
            if response.status_code >= 400:
                raise RuntimeError(f"LLM HTTP {response.status_code}: {_preview(response.text, 500)}")
            payload = response.json()
            return _extract_message_content(payload)
        except Exception as exc:
            last_error = exc
            if attempt + 1 >= attempts:
                break
            time.sleep(min(2.0 ** attempt, 8.0))
    raise RuntimeError(_truncate_error(str(last_error or "LLM pairwise call failed")))


def _candidate_from_card(card: Mapping[str, Any]) -> dict[str, str]:
    candidate_summary = dict(card.get("candidate_summary", {}) or {})
    return {
        "component": str(candidate_summary.get("component", "")),
        "reason": str(candidate_summary.get("reason", "")),
        "canonical_reason": str(candidate_summary.get("canonical_reason", candidate_summary.get("reason", ""))),
        "reason_bucket": str(candidate_summary.get("reason_bucket", "")),
    }


def _valid_promotion_atom_ids(judgment: Mapping[str, Any], request: Mapping[str, Any]) -> list[str]:
    atom_ids = [str(item) for item in judgment.get("promotion_evidence_atom_ids", []) or []]
    if not atom_ids:
        return []
    alt_label = None
    for label in ("candidate_a", "candidate_b"):
        if str(request.get(f"{label}_role", "")) == "alternative":
            alt_label = label
            break
    if not alt_label:
        return []
    pairwise_input = dict(request.get("pairwise_input", {}) or {})
    cards = dict(pairwise_input.get("evidence_summary_cards", {}) or {})
    alt_card = dict(cards.get(alt_label, {}) or {})
    candidate_summary = dict(alt_card.get("candidate_summary", {}) or {})
    valid = {
        str(atom.get("atom_id"))
        for atom in candidate_summary.get("candidate_direct_evidence_atoms", []) or []
        if isinstance(atom, Mapping) and bool(atom.get("promotion_eligible") is True)
    }
    return [atom_id for atom_id in atom_ids if atom_id in valid]


def _sanitize_atoms(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    out: list[dict[str, str]] = []
    for item in value[:MAX_EVIDENCE_ATOMS]:
        if not isinstance(item, Mapping):
            continue
        modality = str(item.get("modality", "case")).strip().lower()
        if modality not in {"metric", "log", "trace", "topology", "case"}:
            modality = "case"
        candidate = str(item.get("candidate", "both")).strip().lower()
        if candidate not in {"candidate_a", "candidate_b", "top1", "alternative", "both"}:
            candidate = "both"
        effect = str(item.get("effect", "neutral")).strip().lower()
        if effect not in {"support", "refute", "neutral"}:
            effect = "neutral"
        summary = _truncate_rationale(item.get("summary", ""), ATOM_SUMMARY_MAX_CHARS)
        out.append({
            "modality": modality,
            "candidate": candidate,
            "effect": effect,
            "summary": summary,
        })
    return out


def _sanitize_atom_ids(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value[:8]:
        text = _truncate_rationale(item, 80)
        if text and text not in out:
            out.append(text)
    return out
