"""Full-fidelity LLM I/O audit wrappers.

This module intentionally records complete prompts and complete model outputs.
Use it for research/debug runs where prompt engineering is under inspection.
Do not use this recorder for public logs or shared artifacts that must redact
secrets or sensitive telemetry.
"""

from __future__ import annotations

import json
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .llm_types import ModelClient, ModelRequest, ModelResponse


@dataclass
class CostWindow:
    """Aggregates token usage across LLM calls for cost monitoring.

    Tracks per-purpose and per-case token consumption so the runner can
    report a full cost breakdown in the result JSON without reaching into
    provider internals.  All counts are cumulative across every call
    appended to the window, including retries and JSON-repair attempts.
    """

    call_count: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    _by_purpose: dict[str, dict[str, int]] = field(default_factory=dict)
    _case_marks: list[dict[str, Any]] = field(default_factory=list)

    def record(self, *, purpose: str, usage: Mapping[str, Any] | None) -> None:
        self.call_count += 1
        vals = (
            _as_int(usage, "prompt_tokens"),
            _as_int(usage, "completion_tokens"),
            _as_int(usage, "total_tokens"),
        )
        pt, ct, tt = vals
        if tt == 0 and (pt > 0 or ct > 0):
            tt = pt + ct
        self.prompt_tokens += pt
        self.completion_tokens += ct
        self.total_tokens += tt
        bucket = self._by_purpose.setdefault(
            purpose,
            {"call_count": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        )
        bucket["call_count"] += 1
        _accumulate_bucket(bucket, vals)

    def mark_case(self, *, case_id: str) -> dict[str, Any]:
        snapshot = self.snapshot()
        mark = {"case_id": case_id, **snapshot}
        self._case_marks.append(mark)
        return mark

    def snapshot(self) -> dict[str, Any]:
        return {
            "call_count": self.call_count,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "by_purpose": {
                purpose: dict(stats) for purpose, stats in self._by_purpose.items()
            },
        }

    def summary(self) -> dict[str, Any]:
        return {
            **self.snapshot(),
            "per_case": list(self._case_marks),
        }


def _as_int(usage: Mapping[str, Any] | None, key: str) -> int:
    if usage is None:
        return 0
    value = usage.get(key)
    if value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


_USAGE_KEYS = (
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
)


def _accumulate_bucket(bucket: dict[str, int], vals: tuple[int, int, int]) -> None:
    for _key, _val in zip(_USAGE_KEYS, vals):
        bucket[_key] += _val


@dataclass(frozen=True)
class LLMIORecord:
    """A single model call with full input/output payloads."""

    call_index: int
    started_at_unix: float
    elapsed_ms: float
    outcome: str
    request: Mapping[str, Any]
    response: Mapping[str, Any] | None = None
    error: Mapping[str, Any] | None = None


@dataclass
class LLMIORecorder:
    """Append-only in-memory and optional JSONL recorder for LLM calls."""

    jsonl_path: str | Path | None = None
    records: list[LLMIORecord] = field(default_factory=list)

    def append(self, record: LLMIORecord) -> None:
        self.records.append(record)
        if self.jsonl_path:
            path = Path(self.jsonl_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record_to_json(record), ensure_ascii=False))
                handle.write("\n")


class AuditedModelClient:
    """ModelClient wrapper that preserves every request and response verbatim.

    When a ``CostWindow`` is supplied, token usage from each successful call
    is accumulated into it for cost monitoring.
    """

    def __init__(
        self,
        *,
        inner: ModelClient,
        recorder: LLMIORecorder,
        cost_window: CostWindow | None = None,
    ) -> None:
        self._inner = inner
        self._recorder = recorder
        self._cost_window = cost_window
        self._call_index = 0

    @property
    def recorder(self) -> LLMIORecorder:
        return self._recorder

    @property
    def cost_window(self) -> CostWindow | None:
        return self._cost_window

    def complete(self, *, request: ModelRequest) -> ModelResponse:
        self._call_index += 1
        started = time.time()
        request_payload = request_to_json(request)
        try:
            response = self._inner.complete(request=request)
        except Exception as exc:
            elapsed_ms = (time.time() - started) * 1000.0
            self._recorder.append(
                LLMIORecord(
                    call_index=self._call_index,
                    started_at_unix=started,
                    elapsed_ms=elapsed_ms,
                    outcome="error",
                    request=request_payload,
                    error={
                        "type": type(exc).__name__,
                        "message": str(exc),
                        "traceback": traceback.format_exc(),
                    },
                )
            )
            raise

        elapsed_ms = (time.time() - started) * 1000.0
        self._recorder.append(
            LLMIORecord(
                call_index=self._call_index,
                started_at_unix=started,
                elapsed_ms=elapsed_ms,
                outcome="success",
                request=request_payload,
                response={"content": response.content, "usage": dict(response.usage) if response.usage else None},
            )
        )
        if self._cost_window is not None and response.usage:
            self._cost_window.record(purpose=request.purpose, usage=response.usage)
        return response


def request_to_json(request: ModelRequest) -> Mapping[str, Any]:
    return {
        "purpose": request.purpose,
        "attempt_index": request.attempt_index,
        "messages": [
            {
                "role": message.role,
                "content": message.content,
            }
            for message in request.messages
        ],
    }


def record_to_json(record: LLMIORecord) -> Mapping[str, Any]:
    return {
        "call_index": record.call_index,
        "started_at_unix": record.started_at_unix,
        "elapsed_ms": record.elapsed_ms,
        "outcome": record.outcome,
        "request": dict(record.request),
        "response": dict(record.response) if record.response is not None else None,
        "error": dict(record.error) if record.error is not None else None,
    }
