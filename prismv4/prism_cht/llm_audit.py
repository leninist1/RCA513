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
    """ModelClient wrapper that preserves every request and response verbatim."""

    def __init__(self, *, inner: ModelClient, recorder: LLMIORecorder) -> None:
        self._inner = inner
        self._recorder = recorder
        self._call_index = 0

    @property
    def recorder(self) -> LLMIORecorder:
        return self._recorder

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
                response={"content": response.content},
            )
        )
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
