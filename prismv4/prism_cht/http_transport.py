"""Injectable HTTP transport for PRISM-CHT provider adapter.

Defines frozen request/response value objects, a ``HttpTransport``
protocol, and a standard-library ``UrllibHttpTransport`` implementation.
No retry, no streaming, no SDK imports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, Protocol

from .provider_types import ProviderResponseError, ProviderTransportError


# ===========================================================================
# HttpRequest
# ===========================================================================


@dataclass(frozen=True)
class HttpRequest:
    """Immutable HTTP request — no headers or body leaked in repr."""

    method: str
    url: str
    headers: Mapping[str, str] = field(repr=False)
    body: bytes = field(repr=False)
    timeout_seconds: float

    def __post_init__(self) -> None:
        if not self.method or not self.method.strip():
            raise ValueError("HttpRequest.method must be non-empty")
        if not self.url or not self.url.strip():
            raise ValueError("HttpRequest.url must be non-empty")
        if self.timeout_seconds <= 0:
            raise ValueError(
                f"HttpRequest.timeout_seconds must be > 0, "
                f"got {self.timeout_seconds}"
            )

        # Defensive copy of headers
        frozen_headers = MappingProxyType(dict(self.headers))
        object.__setattr__(self, "headers", frozen_headers)

        # Ensure body is bytes
        if isinstance(self.body, str):
            object.__setattr__(self, "body", self.body.encode("utf-8"))
        elif isinstance(self.body, bytearray):
            object.__setattr__(self, "body", bytes(self.body))
        elif isinstance(self.body, memoryview):
            object.__setattr__(self, "body", bytes(self.body))
        elif not isinstance(self.body, bytes):
            object.__setattr__(self, "body", bytes(self.body))


# ===========================================================================
# HttpResponse
# ===========================================================================


@dataclass(frozen=True)
class HttpResponse:
    """Immutable HTTP response."""

    status_code: int
    headers: Mapping[str, str] = field(repr=False)
    body: bytes = field(repr=False)

    def __post_init__(self) -> None:
        if self.status_code <= 0:
            raise ValueError(
                f"HttpResponse.status_code must be > 0, "
                f"got {self.status_code}"
            )

        # Defensive copy of headers
        frozen_headers = MappingProxyType(dict(self.headers))
        object.__setattr__(self, "headers", frozen_headers)

        # Ensure body is bytes
        if isinstance(self.body, str):
            object.__setattr__(self, "body", self.body.encode("utf-8"))
        elif isinstance(self.body, bytearray):
            object.__setattr__(self, "body", bytes(self.body))
        elif isinstance(self.body, memoryview):
            object.__setattr__(self, "body", bytes(self.body))
        elif not isinstance(self.body, bytes):
            object.__setattr__(self, "body", bytes(self.body))


# ===========================================================================
# HttpTransport protocol
# ===========================================================================


class HttpTransport(Protocol):
    """Injectable transport protocol — implement for real or fake HTTP."""

    def send(
        self,
        *,
        request: HttpRequest,
        max_response_bytes: int,
    ) -> HttpResponse:
        ...


# ===========================================================================
# UrllibHttpTransport
# ===========================================================================


class UrllibHttpTransport:
    """Real HTTP transport using only stdlib ``urllib``.

    - Reads at most ``max_response_bytes + 1`` bytes to detect overflow.
    - Converts ``urllib.error.HTTPError`` to ``HttpResponse``.
    - Converts ``urllib.error.URLError`` to ``ProviderTransportError``.
    - Does NOT retry, stream, read environment variables, or log API keys.
    """

    def send(
        self,
        *,
        request: HttpRequest,
        max_response_bytes: int,
    ) -> HttpResponse:
        import urllib.request
        import urllib.error

        data = request.body if request.body else None

        req = urllib.request.Request(
            url=request.url,
            data=data,
            headers=dict(request.headers),
            method=request.method,
        )

        try:
            with urllib.request.urlopen(req, timeout=request.timeout_seconds) as resp:
                response_body = resp.read(max_response_bytes + 1)
                if len(response_body) > max_response_bytes:
                    raise ProviderResponseError(
                        f"response body exceeded max_response_bytes "
                        f"({max_response_bytes}): received "
                        f"{len(response_body)} bytes"
                    )
                response_headers: dict[str, str] = {}
                for key, value in resp.getheaders():
                    response_headers[key] = value
                return HttpResponse(
                    status_code=resp.getcode(),
                    headers=response_headers,
                    body=response_body,
                )
        except urllib.error.HTTPError as e:
            # Read the error body (truncated) and return as HttpResponse
            try:
                error_body = e.read(max_response_bytes + 1)
            except Exception:
                error_body = b""
            if len(error_body) > max_response_bytes:
                raise ProviderResponseError(
                    f"error response body exceeded max_response_bytes "
                    f"({max_response_bytes}): received "
                    f"{len(error_body)} bytes"
                )
            error_headers: dict[str, str] = {}
            if hasattr(e, "headers"):
                for key, value in e.headers.items():
                    error_headers[key] = value
            return HttpResponse(
                status_code=e.code,
                headers=error_headers,
                body=error_body,
            )
        except urllib.error.URLError as e:
            raise ProviderTransportError(
                f"transport error: {e.reason}"
            ) from e
