"""OpenAI-compatible chat provider adapter for PRISM-CHT.

``OpenAICompatibleChatModelClient`` satisfies the ``ModelClient`` protocol
while converting ``ModelRequest`` to OpenAI-compatible ``/chat/completions``
HTTP requests via an injected ``HttpTransport``.  Every call produces a
``ProviderCallAudit`` record that excludes API keys, full prompts, and
full response bodies.
"""

from __future__ import annotations

import json
import time

from .http_transport import HttpRequest, HttpTransport
from .llm_types import ModelClient, ModelRequest, ModelResponse
from .provider_config import OpenAICompatibleChatConfig
from .provider_types import (
    ProviderCallAudit,
    ProviderHTTPError,
    ProviderResponseError,
    ProviderTransportError,
    ProviderUsage,
)


# ===========================================================================
# HTTP status → retryable classification
# ===========================================================================

_ALWAYS_RETRYABLE_STATUS = frozenset({408, 409, 425, 429})


def _is_retryable(status_code: int) -> bool:
    """Return True if the HTTP status is retryable.

    - 408, 409, 425, 429 → retryable
    - 500-599 → retryable
    - all others → non-retryable
    """
    if status_code in _ALWAYS_RETRYABLE_STATUS:
        return True
    if 500 <= status_code <= 599:
        return True
    return False


# ===========================================================================
# Client
# ===========================================================================


class OpenAICompatibleChatModelClient:
    """OpenAI-compatible chat provider adapter implementing ``ModelClient``.

    Must be constructed with an explicit ``HttpTransport``.  Does NOT:
    - Read environment variables
    - Provide a default transport
    - Implement retry, streaming, function calling, or provider fallback
    - Log API keys, full prompts, or full response bodies
    """

    def __init__(
        self,
        *,
        config: OpenAICompatibleChatConfig,
        transport: HttpTransport,
    ) -> None:
        self._config = config
        self._transport = transport
        self._endpoint = config.base_url.rstrip("/") + "/chat/completions"
        self._audit: list[ProviderCallAudit] = []

    # -- ModelClient interface -----------------------------------------------

    def complete(
        self,
        *,
        request: ModelRequest,
    ) -> ModelResponse:
        """Send a ``ModelRequest`` to the provider and return a ``ModelResponse``.

        Exactly one ``ProviderCallAudit`` record is appended on every call,
        including failures.
        """
        t0 = time.monotonic()
        outcome = "success"
        status_code: int | None = None
        finish_reason: str | None = None
        usage = ProviderUsage()
        request_bytes = 0
        response_bytes = 0
        body_bytes = b""

        try:
            # Build request body (deterministic JSON)
            body_dict = self._build_request_body(request)
            body_bytes = json.dumps(
                body_dict, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
            request_bytes = len(body_bytes)

            # Build HTTP headers
            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": f"Bearer {self._config.api_key}",
            }

            http_request = HttpRequest(
                method="POST",
                url=self._endpoint,
                headers=headers,
                body=body_bytes,
                timeout_seconds=self._config.timeout_seconds,
            )

            http_response = self._transport.send(
                request=http_request,
                max_response_bytes=self._config.max_response_bytes,
            )
            status_code = http_response.status_code
            response_bytes = len(http_response.body)

            if status_code < 200 or status_code >= 300:
                raise ProviderHTTPError(
                    status_code=status_code,
                    message=(
                        f"HTTP {status_code}: provider returned non-2xx "
                        f"status"
                    ),
                    retryable=_is_retryable(status_code),
                )

            # Parse successful response
            finish_reason, usage, content = self._parse_response_body(
                http_response.body
            )
            outcome = "success"

        except ProviderTransportError:
            outcome = "transport_error"
            raise
        except ProviderHTTPError:
            outcome = "http_error"
            raise
        except ProviderResponseError:
            outcome = "response_error"
            raise
        except Exception:
            outcome = "response_error"
            raise
        finally:
            elapsed_ms = (time.monotonic() - t0) * 1000.0

            audit = ProviderCallAudit(
                purpose=request.purpose,
                attempt_index=request.attempt_index,
                model=self._config.model,
                endpoint=self._endpoint,
                status_code=status_code,
                elapsed_ms=elapsed_ms,
                request_bytes=request_bytes,
                response_bytes=response_bytes,
                finish_reason=finish_reason,
                usage=usage,
                outcome=outcome,
            )
            self._audit.append(audit)

        return ModelResponse(content=content)

    # -- Internal helpers ----------------------------------------------------

    def _build_request_body(self, request: ModelRequest) -> dict:
        """Build the JSON-serializable request body.

        Includes model, messages, stream=false, optional max_tokens, and
        optional response_format.  Merges extra_body.  Never includes
        API key, purpose, or attempt_index.
        """
        messages = [
            {"role": msg.role, "content": msg.content}
            for msg in request.messages
        ]

        body: dict = {
            "model": self._config.model,
            "messages": messages,
            "stream": False,
        }
        if self._config.max_tokens is not None:
            body.update({"max_tokens": self._config.max_tokens})

        if self._config.json_mode:
            body["response_format"] = {"type": "json_object"}

        # Merge extra_body
        extra = dict(self._config.extra_body)
        body.update(extra)

        return body

    @staticmethod
    def _parse_response_body(
        body_bytes: bytes,
    ) -> tuple[str | None, ProviderUsage, str]:
        """Parse and validate the provider's JSON response body.

        Returns (finish_reason, ProviderUsage, content).
        Raises ProviderResponseError on structural violations.
        """
        # Parse JSON
        try:
            text = body_bytes.decode("utf-8")
        except UnicodeDecodeError as e:
            raise ProviderResponseError(
                f"response body is not valid UTF-8: {e}"
            )

        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise ProviderResponseError(
                f"response body is not valid JSON: {e}"
            )

        # Top-level must be object
        if not isinstance(data, dict):
            raise ProviderResponseError(
                f"response top-level must be a JSON object, "
                f"got {type(data).__name__}"
            )

        # choices must exist and be non-empty list
        choices = data.get("choices")
        if choices is None:
            raise ProviderResponseError(
                "response missing required field 'choices'"
            )
        if not isinstance(choices, list):
            raise ProviderResponseError(
                f"response 'choices' must be a list, "
                f"got {type(choices).__name__}"
            )
        if len(choices) == 0:
            raise ProviderResponseError("response 'choices' is empty")

        # choices[0] must be object
        choice0 = choices[0]
        if not isinstance(choice0, dict):
            raise ProviderResponseError(
                f"response choices[0] must be an object, "
                f"got {type(choice0).__name__}"
            )

        # message must be object
        message = choice0.get("message")
        if message is None:
            raise ProviderResponseError(
                "response choices[0] missing required field 'message'"
            )
        if not isinstance(message, dict):
            raise ProviderResponseError(
                f"response choices[0].message must be an object, "
                f"got {type(message).__name__}"
            )

        # content must be non-empty str
        content = message.get("content")
        if content is None:
            raise ProviderResponseError(
                "response choices[0].message missing required field 'content'"
            )
        if not isinstance(content, str):
            raise ProviderResponseError(
                f"response choices[0].message.content must be a string, "
                f"got {type(content).__name__}"
            )
        if not content.strip():
            raise ProviderResponseError(
                "response choices[0].message.content must be non-empty"
            )

        # finish_reason
        finish_reason = choice0.get("finish_reason")
        if finish_reason is not None:
            if not isinstance(finish_reason, str):
                raise ProviderResponseError(
                    f"response finish_reason must be str or None, "
                    f"got {type(finish_reason).__name__}"
                )
            if finish_reason == "length":
                raise ProviderResponseError(
                    "response finish_reason='length': content was truncated"
                )

        # usage (optional)
        usage = ProviderUsage()
        raw_usage = data.get("usage")
        if raw_usage is not None:
            if not isinstance(raw_usage, dict):
                raise ProviderResponseError(
                    f"response usage must be an object, "
                    f"got {type(raw_usage).__name__}"
                )
            try:
                usage_kwargs = {
                    "prompt_tokens": raw_usage.get("prompt_tokens"),
                    "completion_tokens": raw_usage.get("completion_tokens"),
                    "total_tokens": raw_usage.get("total_tokens"),
                }
                usage = ProviderUsage(**usage_kwargs)
            except ValueError as e:
                raise ProviderResponseError(str(e)) from e

        return finish_reason, usage, content

    # -- Audit accessor ------------------------------------------------------

    @property
    def audit_records(self) -> tuple[ProviderCallAudit, ...]:
        """Read-only tuple of all audit records, in call order."""
        return tuple(self._audit)
