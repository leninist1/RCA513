"""Shadow-mode LLM judge for D32 candidate evidence summary cards.

The judge is intentionally constrained: it receives only an Evidence Summary
Card plus the candidate copied from that card, and its output is never fed back
into D32 ranking or predictions.
"""
from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping


SYSTEM_PROMPT = """You are a constrained root-cause evidence judge.
You do not predict new root causes.
You only judge the given candidate.
You must return JSON only.
If evidence is weak or ambiguous, return uncertain.
Do not penalize a candidate only because one modality is unavailable.
Do not use external knowledge.
Do not infer from case id or file names."""

USER_PROMPT_PREFIX = (
    "Judge whether the Evidence Summary Card supports, refutes, or is uncertain "
    "about the given candidate. Return only JSON matching this schema: "
    '{"verdict":"support|refute|uncertain","support_score":0.0,'
    '"refute_score":0.0,"component_consistency":0.0,'
    '"reason_consistency":0.0,"evidence_completeness":0.0,'
    '"should_promote":false,"should_demote":false,'
    '"rationale":"one short sentence"}\n\n'
)

DEFAULT_RATIONALE_MAX_CHARS = 240


@dataclass(frozen=True)
class LLMJudgeConfig:
    provider: str = "openai_compatible"
    model: str | None = None
    base_url: str | None = None
    api_key_env: str = "RCA_LLM_API_KEY"
    timeout_sec: float = 60.0
    temperature: float = 0.0
    max_retries: int = 2
    concurrency: int = 1
    rationale_max_chars: int = DEFAULT_RATIONALE_MAX_CHARS


def default_judgment() -> dict[str, Any]:
    return {
        "verdict": "uncertain",
        "support_score": 0.0,
        "refute_score": 0.0,
        "component_consistency": 0.0,
        "reason_consistency": 0.0,
        "evidence_completeness": 0.0,
        "should_promote": False,
        "should_demote": False,
        "rationale": "LLM judgment unavailable.",
    }


def judge_candidate(card: Mapping[str, Any], config: LLMJudgeConfig) -> dict[str, Any]:
    """Call the configured LLM once and return a JSONL-ready result fragment."""

    try:
        content = _call_openai_compatible(card, config)
    except Exception as exc:
        return {
            "llm_judgment": default_judgment(),
            "parse_ok": False,
            "error": _truncate_error(str(exc)),
            "raw_response_preview": None,
        }

    judgment, parse_ok, error = parse_judgment(content, rationale_max_chars=config.rationale_max_chars)
    return {
        "llm_judgment": judgment,
        "parse_ok": parse_ok,
        "error": error,
        "raw_response_preview": _preview(content) if not parse_ok else None,
    }


def write_shadow_judgments_jsonl(
    *,
    cards: list[Mapping[str, Any]],
    out_path: Path,
    config: LLMJudgeConfig,
) -> int:
    """Write one shadow LLM judgment per summary card.

    Failures are recorded per line and never raised to the caller after the
    output file has been opened.
    """

    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with out_path.open("w", encoding="utf-8") as f:
        if int(config.concurrency) <= 1:
            for card in cards:
                row = _judge_output_row(card, config)
                f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                f.flush()
                written += 1
        else:
            with ThreadPoolExecutor(max_workers=max(1, int(config.concurrency))) as executor:
                for row in executor.map(lambda card: _judge_output_row(card, config), cards):
                    f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                    f.flush()
                    written += 1
    return written


def build_judge_input(card: Mapping[str, Any]) -> dict[str, Any]:
    """Build the full LLM input from an Evidence Summary Card only."""

    candidate_summary = dict(card.get("candidate_summary", {}) or {})
    candidate = {
        "component": str(candidate_summary.get("component", "")),
        "reason": str(candidate_summary.get("reason", "")),
    }
    return {
        "task": "judge_candidate_root_cause",
        "candidate": candidate,
        "evidence_summary_card": dict(card),
    }


def _judge_output_row(card: Mapping[str, Any], config: LLMJudgeConfig) -> dict[str, Any]:
    row = _base_output_row(card)
    try:
        result = judge_candidate(card, config)
    except Exception as exc:
        result = {
            "llm_judgment": default_judgment(),
            "parse_ok": False,
            "error": _truncate_error(str(exc)),
            "raw_response_preview": None,
        }
    row.update(result)
    return row


def parse_judgment(raw_content: str, *, rationale_max_chars: int = DEFAULT_RATIONALE_MAX_CHARS) -> tuple[dict[str, Any], bool, str | None]:
    """Parse and sanitize the model response.

    Valid JSON objects are tolerated even when fields are missing. Invalid JSON
    returns the default uncertain judgment with parse_ok=false.
    """

    try:
        parsed = json.loads(raw_content)
    except Exception as exc:
        return default_judgment(), False, f"json_parse_error: {_truncate_error(str(exc))}"
    if not isinstance(parsed, dict):
        return default_judgment(), False, "json_parse_error: response is not an object"

    judgment = default_judgment()
    errors: list[str] = []

    verdict = str(parsed.get("verdict", judgment["verdict"])).strip().lower()
    if verdict not in {"support", "refute", "uncertain"}:
        errors.append(f"invalid_verdict:{verdict[:40]}")
        verdict = "uncertain"
    judgment["verdict"] = verdict

    for field in (
        "support_score",
        "refute_score",
        "component_consistency",
        "reason_consistency",
        "evidence_completeness",
    ):
        judgment[field] = _clamp01(parsed.get(field, judgment[field]))

    judgment["should_promote"] = _to_bool(parsed.get("should_promote", False))
    judgment["should_demote"] = _to_bool(parsed.get("should_demote", False))
    judgment["rationale"] = _truncate_rationale(parsed.get("rationale", judgment["rationale"]), rationale_max_chars)
    return judgment, True, ";".join(errors) if errors else None


def _call_openai_compatible(card: Mapping[str, Any], config: LLMJudgeConfig) -> str:
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
                "content": USER_PROMPT_PREFIX + json.dumps(build_judge_input(card), ensure_ascii=False, sort_keys=True),
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
    raise RuntimeError(_truncate_error(str(last_error or "LLM call failed")))


def _extract_message_content(payload: Mapping[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("LLM response missing choices")
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(message, dict):
        raise ValueError("LLM response missing message")
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        if parts:
            return "".join(parts)
    raise ValueError("LLM response missing text content")


def _base_output_row(card: Mapping[str, Any]) -> dict[str, Any]:
    candidate_summary = dict(card.get("candidate_summary", {}) or {})
    return {
        "case_id": str(card.get("case_id", "")),
        "candidate_rank": _safe_int(candidate_summary.get("candidate_rank", 0)),
        "candidate": {
            "component": str(candidate_summary.get("component", "")),
            "reason": str(candidate_summary.get("reason", "")),
        },
        "summary_card": dict(card),
    }


def _chat_completions_url(base_url: str) -> str:
    base = str(base_url).rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def _clamp01(value: Any) -> float:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return 0.0
    if val < 0.0:
        return 0.0
    if val > 1.0:
        return 1.0
    return val


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    if isinstance(value, (int, float)):
        return bool(value)
    return False


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _truncate_rationale(value: Any, max_chars: int) -> str:
    text = " ".join(str(value).split())
    limit = max(1, int(max_chars))
    if len(text) <= limit:
        return text
    return text[:limit].rstrip()


def _preview(value: Any, limit: int = 1000) -> str:
    text = str(value)
    return text if len(text) <= limit else text[:limit]


def _truncate_error(value: str, limit: int = 500) -> str:
    return _preview(" ".join(str(value).split()), limit)
