"""Provider types — errors, usage, and audit records for PRISM-CHT.

Defines immutable value objects for provider communication audit trails
without exposing API keys, full prompts, or complete response bodies.
"""

from __future__ import annotations

from dataclasses import dataclass


# ===========================================================================
# Exceptions
# ===========================================================================


class ProviderConfigurationError(ValueError):
    """Raised when provider configuration is invalid."""


class ProviderTransportError(RuntimeError):
    """Raised when a transport-level error occurs (e.g. DNS, connection)."""


class ProviderHTTPError(RuntimeError):
    """Raised for non-2xx HTTP responses."""

    def __init__(
        self,
        *,
        status_code: int,
        message: str,
        retryable: bool,
    ) -> None:
        super().__init__(message)
        self._status_code = status_code
        self._retryable = retryable

    @property
    def status_code(self) -> int:
        return self._status_code

    @property
    def retryable(self) -> bool:
        return self._retryable


class ProviderResponseError(RuntimeError):
    """Raised when a response body fails structural validation."""


# ===========================================================================
# Usage
# ===========================================================================


@dataclass(frozen=True)
class ProviderUsage:
    """Token usage returned by the provider (all fields optional)."""

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None

    def __post_init__(self) -> None:
        for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = getattr(self, name)
            if value is None:
                continue
            if isinstance(value, bool):
                raise ValueError(
                    f"ProviderUsage.{name} must be int or None, got bool"
                )
            if not isinstance(value, int):
                raise ValueError(
                    f"ProviderUsage.{name} must be int or None, "
                    f"got {type(value).__name__}"
                )
            if value < 0:
                raise ValueError(
                    f"ProviderUsage.{name} must be >= 0, got {value}"
                )


# ===========================================================================
# Audit
# ===========================================================================


_VALID_OUTCOMES = frozenset({
    "success",
    "transport_error",
    "http_error",
    "response_error",
})


@dataclass(frozen=True)
class ProviderCallAudit:
    """Immutable audit record for a single provider call.

    No API key, Authorization header, full prompt, full response body,
    or ModelRequest object is ever stored.
    """

    purpose: str
    attempt_index: int
    model: str
    endpoint: str

    status_code: int | None
    elapsed_ms: float
    request_bytes: int
    response_bytes: int

    finish_reason: str | None
    usage: ProviderUsage
    outcome: str

    def __post_init__(self) -> None:
        if not self.purpose or not self.purpose.strip():
            raise ValueError("ProviderCallAudit.purpose must be non-empty")
        if not self.model or not self.model.strip():
            raise ValueError("ProviderCallAudit.model must be non-empty")
        if not self.endpoint or not self.endpoint.strip():
            raise ValueError("ProviderCallAudit.endpoint must be non-empty")
        if self.elapsed_ms < 0:
            raise ValueError(
                f"ProviderCallAudit.elapsed_ms must be >= 0, "
                f"got {self.elapsed_ms}"
            )
        if self.request_bytes < 0:
            raise ValueError(
                f"ProviderCallAudit.request_bytes must be >= 0, "
                f"got {self.request_bytes}"
            )
        if self.response_bytes < 0:
            raise ValueError(
                f"ProviderCallAudit.response_bytes must be >= 0, "
                f"got {self.response_bytes}"
            )
        if self.outcome not in _VALID_OUTCOMES:
            raise ValueError(
                f"ProviderCallAudit.outcome must be one of "
                f"{sorted(_VALID_OUTCOMES)}, got '{self.outcome}'"
            )
