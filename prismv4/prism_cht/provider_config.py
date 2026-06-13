"""Provider runtime configuration for PRISM-CHT.

Defines ``OpenAICompatibleChatConfig``, a frozen dataclass for an
OpenAI-compatible chat completions provider, and a loader that reads
configuration from an explicitly-provided ``Mapping[str, str]``.
Never reads ``os.environ`` internally, never hardcodes API keys.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

from .provider_types import ProviderConfigurationError


# ===========================================================================
# Environment variable names (for use by external entry points)
# ===========================================================================

ENV_BASE_URL = "PRISM_CHT_MODEL_BASE_URL"
ENV_MODEL = "PRISM_CHT_MODEL_NAME"
ENV_API_KEY = (
    "PRISM_CHT_MODEL_API_KEY"
)
ENV_TIMEOUT_SECONDS = "PRISM_CHT_MODEL_TIMEOUT_SECONDS"
ENV_MAX_TOKENS = (
    "PRISM_CHT_MODEL_MAX_TOKENS"
)
ENV_MAX_RESPONSE_BYTES = "PRISM_CHT_MODEL_MAX_RESPONSE_BYTES"

# ===========================================================================
# Reserved fields that extra_body must not override
# ===========================================================================

_RESERVED_FIELDS = frozenset({
    "model",
    "messages",
    "stream",
    "max_tokens",
    "response_format",
})


# ===========================================================================
# OpenAICompatibleChatConfig
# ===========================================================================


def _deep_freeze_extra_body(value: Mapping[str, Any]) -> MappingProxyType[str, Any]:
    """Recursively freeze extra_body so it is truly immutable."""
    frozen: dict[str, Any] = {}
    for k, v in value.items():
        if isinstance(v, Mapping):
            frozen[k] = _deep_freeze_extra_body(v)
        elif isinstance(v, list):
            frozen[k] = tuple(v)
        else:
            frozen[k] = v
    return MappingProxyType(frozen)


@dataclass(frozen=True)
class OpenAICompatibleChatConfig:
    """Immutable configuration for an OpenAI-compatible chat provider.

    ``api_key`` is excluded from ``repr()`` and ``extra_body`` is
    recursively frozen.  No default API key, base_url, or model is
    provided — all must be supplied at construction time.
    """

    base_url: str
    model: str
    api_key: str = field(repr=False)

    timeout_seconds: float = 60.0
    max_tokens: int = 4096
    max_response_bytes: int = 2_000_000
    json_mode: bool = True
    extra_body: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._validate_base_url()
        self._validate_model()
        self._validate_api_key()
        self._validate_numeric()
        self._validate_extra_body()

    def _validate_base_url(self) -> None:
        if not self.base_url or not self.base_url.strip():
            raise ProviderConfigurationError("base_url must be non-empty")

        if not self.base_url.startswith("https://"):
            raise ProviderConfigurationError(
                f"base_url must use https scheme, got '{self.base_url}'"
            )

        normalized = self.base_url.rstrip("/")

        # Parse URL components using regex
        pattern = re.compile(
            r'^https://'
            r'(?P<host>[^/?\#]+)'
            r'(?P<rest>.*)$'
        )
        m = pattern.match(normalized)
        if m is None:
            raise ProviderConfigurationError(
                f"base_url must contain a valid hostname: '{self.base_url}'"
            )

        host_part = m.group("host")
        rest = m.group("rest")

        # Check host part for username/password (@)
        if "@" in host_part:
            raise ProviderConfigurationError(
                "base_url must not contain username or password"
            )

        if not host_part or host_part == "":
            raise ProviderConfigurationError(
                f"base_url must contain a hostname: '{self.base_url}'"
            )

        # Check rest for query/fragment
        if "?" in rest:
            raise ProviderConfigurationError(
                "base_url must not contain query parameters"
            )
        if "#" in rest:
            raise ProviderConfigurationError(
                "base_url must not contain fragment"
            )

        # Store normalized
        object.__setattr__(self, "base_url", normalized)

    def _validate_model(self) -> None:
        if not self.model or not self.model.strip():
            raise ProviderConfigurationError("model must be non-empty")

    def _validate_api_key(self) -> None:
        if not self.api_key or not self.api_key.strip():
            raise ProviderConfigurationError("api_key must be non-empty")

    def _validate_numeric(self) -> None:
        if self.timeout_seconds <= 0:
            raise ProviderConfigurationError(
                f"timeout_seconds must be > 0, got {self.timeout_seconds}"
            )
        if isinstance(self.max_tokens, bool):
            raise ProviderConfigurationError(
                "max_tokens must be int, got bool"
            )
        if not isinstance(self.max_tokens, int):
            raise ProviderConfigurationError(
                f"max_tokens must be int, got {type(self.max_tokens).__name__}"
            )
        if self.max_tokens <= 0:
            raise ProviderConfigurationError(
                f"max_tokens must be > 0, got {self.max_tokens}"
            )
        if isinstance(self.max_response_bytes, bool):
            raise ProviderConfigurationError(
                "max_response_bytes must be int, got bool"
            )
        if not isinstance(self.max_response_bytes, int):
            raise ProviderConfigurationError(
                f"max_response_bytes must be int, "
                f"got {type(self.max_response_bytes).__name__}"
            )
        if self.max_response_bytes <= 0:
            raise ProviderConfigurationError(
                f"max_response_bytes must be > 0, got {self.max_response_bytes}"
            )

    def _validate_extra_body(self) -> None:
        if not isinstance(self.extra_body, Mapping):
            raise ProviderConfigurationError(
                f"extra_body must be a Mapping, got {type(self.extra_body).__name__}"
            )
        for key in self.extra_body:
            if key in _RESERVED_FIELDS:
                raise ProviderConfigurationError(
                    f"extra_body must not override reserved field '{key}'"
                )

        # Recursively freeze
        frozen = _deep_freeze_extra_body(self.extra_body)
        object.__setattr__(self, "extra_body", frozen)


# ===========================================================================
# Configuration loader
# ===========================================================================


def load_openai_compatible_config_from_mapping(
    env: Mapping[str, str],
) -> OpenAICompatibleChatConfig:
    """Build an ``OpenAICompatibleChatConfig`` from an explicit Mapping.

    Reads only from *env* — never calls ``os.environ``.  Raises
    ``ProviderConfigurationError`` on missing required keys or invalid
    numeric values.  Error messages never include the API key.
    """

    def _require(key: str) -> str:
        if key not in env:
            raise ProviderConfigurationError(
                f"Missing required configuration key: {key}"
            )
        value = env[key]
        if not value or not value.strip():
            raise ProviderConfigurationError(
                f"Configuration key '{key}' must be non-empty"
            )
        return value.strip()

    base_url = _require(ENV_BASE_URL)
    model = _require(ENV_MODEL)
    api_key = _require(ENV_API_KEY)

    timeout_str = env.get(ENV_TIMEOUT_SECONDS, "60.0")
    max_tokens_str = env.get(
        ENV_MAX_TOKENS, "4096"
    )
    max_response_bytes_str = env.get(ENV_MAX_RESPONSE_BYTES, "2000000")

    try:
        timeout_seconds = float(timeout_str)
    except (ValueError, TypeError):
        raise ProviderConfigurationError(
            f"Invalid numeric value for {ENV_TIMEOUT_SECONDS}: {timeout_str!r}"
        )

    try:
        max_tokens = int(max_tokens_str)
    except (ValueError, TypeError):
        raise ProviderConfigurationError(
            f"Invalid numeric value for {ENV_MAX_TOKENS}: {max_tokens_str!r}"
        )

    try:
        max_response_bytes = int(max_response_bytes_str)
    except (ValueError, TypeError):
        raise ProviderConfigurationError(
            f"Invalid numeric value for {ENV_MAX_RESPONSE_BYTES}: "
            f"{max_response_bytes_str!r}"
        )

    return OpenAICompatibleChatConfig(
        base_url=base_url,
        model=model,
        api_key=api_key,
        timeout_seconds=timeout_seconds,
        max_tokens=max_tokens,
        max_response_bytes=max_response_bytes,
    )
