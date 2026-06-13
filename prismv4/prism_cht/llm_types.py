"""Model client protocol types for PRISM-CHT LLM Boundary.

Defines the minimal vocabulary for communicating with a structured-output
language model client: messages, requests, responses, and the client
protocol itself.  All types are frozen and do not import any model SDK.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class ModelMessage:
    """A single message in a model conversation turn."""

    role: str
    content: str

    def __post_init__(self):
        if self.role not in ("system", "user"):
            raise ValueError(
                f"ModelMessage.role must be 'system' or 'user', "
                f"got '{self.role}'"
            )
        if not self.content or not self.content.strip():
            raise ValueError("ModelMessage.content must be non-empty")


@dataclass(frozen=True)
class ModelRequest:
    """A complete request to send to a language model."""

    purpose: str
    messages: tuple[ModelMessage, ...]
    attempt_index: int

    def __post_init__(self):
        if not self.purpose or not self.purpose.strip():
            raise ValueError("ModelRequest.purpose must be non-empty")
        if len(self.messages) < 1:
            raise ValueError("ModelRequest.messages must contain at least one message")
        if self.attempt_index < 0:
            raise ValueError(
                f"ModelRequest.attempt_index must be >= 0, "
                f"got {self.attempt_index}"
            )


@dataclass(frozen=True)
class ModelResponse:
    """The raw text response from a language model."""

    content: str

    def __post_init__(self):
        if not self.content or not self.content.strip():
            raise ValueError("ModelResponse.content must be non-empty")


class ModelClient(Protocol):
    """Protocol for a structured-output language model client.

    A concrete implementation may be a real API client or a
    deterministic fake for testing.  It must return a
    ``ModelResponse`` on success and may raise on failure.
    """

    def complete(
        self,
        *,
        request: ModelRequest,
    ) -> ModelResponse:
        ...


class StructuredOutputError(ValueError):
    """Raised when a model response fails strict JSON parsing."""


class PromptBudgetExceededError(ValueError):
    """Raised when a constructed prompt exceeds the maximum character budget."""
