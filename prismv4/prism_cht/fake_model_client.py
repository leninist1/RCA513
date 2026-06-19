"""Deterministic fake model client for PRISM-CHT testing.

``FakeModelClient`` returns pre-configured responses in FIFO order
and records every ``ModelRequest`` it receives — without sending any
network traffic, reading environment variables, or importing any model
SDK.
"""

from __future__ import annotations

from collections import deque
from typing import Sequence

from .llm_types import ModelClient, ModelRequest, ModelResponse


class FakeModelClient:
    """Deterministic, pre-configured model client for closed-loop testing.

    Responses are consumed in FIFO order.  Every call to ``complete()``
    records the request so callers can later inspect what was sent.
    """

    def __init__(
        self,
        *,
        responses: Sequence[str | ModelResponse],
    ) -> None:
        # Copy the sequence so external mutations cannot affect us.
        normalized: list[ModelResponse] = []
        for r in responses:
            if isinstance(r, str):
                normalized.append(ModelResponse(content=r))
            elif isinstance(r, ModelResponse):
                normalized.append(r)
            else:
                raise TypeError(
                    f"responses must be str or ModelResponse, "
                    f"got {type(r).__name__}"
                )
        self._responses: deque[ModelResponse] = deque(normalized)
        self._requests: list[ModelRequest] = []

    def complete(
        self,
        *,
        request: ModelRequest,
    ) -> ModelResponse:
        """Return the next pre-configured response in FIFO order.

        Raises ``RuntimeError`` if no responses remain.
        """
        if not self._responses:
            raise RuntimeError(
                "FakeModelClient: no more pre-configured responses; "
                f"exhausted after {len(self._requests)} requests"
            )
        self._requests.append(request)
        return self._responses.popleft()

    @property
    def requests(self) -> tuple[ModelRequest, ...]:
        """Read-only view of all recorded requests."""
        return tuple(self._requests)

    @property
    def remaining_response_count(self) -> int:
        """Number of pre-configured responses still available."""
        return len(self._responses)
