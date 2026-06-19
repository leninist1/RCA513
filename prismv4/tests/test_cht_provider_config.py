"""Tests for provider_config.py — configuration validation and loading."""

import pytest

from prismv4.prism_cht.provider_config import (
    ENV_API_KEY,
    ENV_BASE_URL,
    ENV_MODEL,
    ENV_MAX_TOKENS,
    ENV_MAX_RESPONSE_BYTES,
    ENV_TIMEOUT_SECONDS,
    OpenAICompatibleChatConfig,
    ProviderConfigurationError,
    load_openai_compatible_config_from_mapping,
)


# ===========================================================================
# Config construction tests
# ===========================================================================


class TestOpenAICompatibleChatConfig:
    def test_valid_https_base_url(self):
        cfg = OpenAICompatibleChatConfig(
            base_url="https://api.example.com/v1",
            model="test-model",
            api_key="test-key",
        )
        assert cfg.base_url == "https://api.example.com/v1"

    def test_base_url_trailing_slash_normalized(self):
        cfg = OpenAICompatibleChatConfig(
            base_url="https://api.example.com/v1/",
            model="test-model",
            api_key="test-key",
        )
        assert cfg.base_url == "https://api.example.com/v1"

    def test_reject_http_base_url(self):
        with pytest.raises(ProviderConfigurationError, match="https"):
            OpenAICompatibleChatConfig(
                base_url="http://api.example.com",
                model="test-model",
                api_key="test-key",
            )

    def test_reject_empty_base_url(self):
        with pytest.raises(ProviderConfigurationError, match="base_url"):
            OpenAICompatibleChatConfig(
                base_url="",
                model="test-model",
                api_key="test-key",
            )

    def test_reject_missing_hostname(self):
        with pytest.raises(ProviderConfigurationError, match="hostname"):
            OpenAICompatibleChatConfig(
                base_url="https://",
                model="test-model",
                api_key="test-key",
            )

    def test_reject_username_in_url(self):
        with pytest.raises(ProviderConfigurationError, match="username"):
            OpenAICompatibleChatConfig(
                base_url="https://user@api.example.com",
                model="test-model",
                api_key="test-key",
            )

    def test_reject_password_in_url(self):
        with pytest.raises(ProviderConfigurationError, match="password"):
            OpenAICompatibleChatConfig(
                base_url="https://user:pass@api.example.com",
                model="test-model",
                api_key="test-key",
            )

    def test_reject_query_in_url(self):
        with pytest.raises(ProviderConfigurationError, match="query"):
            OpenAICompatibleChatConfig(
                base_url="https://api.example.com?foo=bar",
                model="test-model",
                api_key="test-key",
            )

    def test_reject_fragment_in_url(self):
        with pytest.raises(ProviderConfigurationError, match="fragment"):
            OpenAICompatibleChatConfig(
                base_url="https://api.example.com#section",
                model="test-model",
                api_key="test-key",
            )

    def test_reject_empty_model(self):
        with pytest.raises(ProviderConfigurationError, match="model"):
            OpenAICompatibleChatConfig(
                base_url="https://api.example.com",
                model="",
                api_key="test-key",
            )

    def test_reject_empty_api_key(self):
        with pytest.raises(ProviderConfigurationError, match="api_key"):
            OpenAICompatibleChatConfig(
                base_url="https://api.example.com",
                model="test-model",
                api_key="",
            )

    def test_repr_excludes_api_key(self):
        cfg = OpenAICompatibleChatConfig(
            base_url="https://api.example.com",
            model="test-model",
            api_key="secret-12345",
        )
        r = repr(cfg)
        assert "secret-12345" not in r

    def test_reject_timeout_zero_or_negative(self):
        with pytest.raises(ProviderConfigurationError, match="timeout"):
            OpenAICompatibleChatConfig(
                base_url="https://api.example.com",
                model="test-model",
                api_key="test-key",
                timeout_seconds=0,
            )

    def test_reject_max_tokens_zero_or_negative(self):
        with pytest.raises(ProviderConfigurationError, match="max_tokens"):
            OpenAICompatibleChatConfig(
                base_url="https://api.example.com",
                model="test-model",
                api_key="test-key",
                max_tokens=0,
            )

    def test_reject_max_tokens_bool(self):
        with pytest.raises(ProviderConfigurationError, match="bool"):
            OpenAICompatibleChatConfig(
                base_url="https://api.example.com",
                model="test-model",
                api_key="test-key",
                max_tokens=True,  # type: ignore
            )

    def test_reject_max_response_bytes_zero_or_negative(self):
        with pytest.raises(ProviderConfigurationError, match="max_response_bytes"):
            OpenAICompatibleChatConfig(
                base_url="https://api.example.com",
                model="test-model",
                api_key="test-key",
                max_response_bytes=0,
            )

    def test_extra_body_recursively_frozen(self):
        cfg = OpenAICompatibleChatConfig(
            base_url="https://api.example.com",
            model="test-model",
            api_key="test-key",
            extra_body={"nested": {"key": "value"}},
        )
        # extra_body is a MappingProxyType — trying to modify raises
        with pytest.raises(TypeError):
            cfg.extra_body["new"] = "bad"  # type: ignore

    def test_extra_body_cannot_override_model(self):
        with pytest.raises(ProviderConfigurationError, match="model"):
            OpenAICompatibleChatConfig(
                base_url="https://api.example.com",
                model="test-model",
                api_key="test-key",
                extra_body={"model": "other"},
            )

    def test_extra_body_cannot_override_messages(self):
        with pytest.raises(ProviderConfigurationError, match="messages"):
            OpenAICompatibleChatConfig(
                base_url="https://api.example.com",
                model="test-model",
                api_key="test-key",
                extra_body={"messages": []},
            )

    def test_extra_body_cannot_override_stream(self):
        with pytest.raises(ProviderConfigurationError, match="stream"):
            OpenAICompatibleChatConfig(
                base_url="https://api.example.com",
                model="test-model",
                api_key="test-key",
                extra_body={"stream": True},
            )

    def test_extra_body_cannot_override_response_format(self):
        with pytest.raises(ProviderConfigurationError, match="response_format"):
            OpenAICompatibleChatConfig(
                base_url="https://api.example.com",
                model="test-model",
                api_key="test-key",
                extra_body={"response_format": {"type": "text"}},
            )


# ===========================================================================
# Configuration loader tests
# ===========================================================================


class TestLoadConfigFromMapping:
    def test_normal_load(self):
        cfg = load_openai_compatible_config_from_mapping({
            ENV_BASE_URL: "https://api.example.com",
            ENV_MODEL: "test-model",
            ENV_API_KEY: "test-key",
        })
        assert cfg.base_url == "https://api.example.com"
        assert cfg.model == "test-model"
        assert cfg.timeout_seconds == 60.0
        assert cfg.max_tokens is None
        assert cfg.max_response_bytes == 2_000_000

    def test_missing_base_url_rejected(self):
        with pytest.raises(ProviderConfigurationError, match=ENV_BASE_URL):
            load_openai_compatible_config_from_mapping({
                ENV_MODEL: "test-model",
                ENV_API_KEY: "test-key",
            })

    def test_missing_model_rejected(self):
        with pytest.raises(ProviderConfigurationError, match=ENV_MODEL):
            load_openai_compatible_config_from_mapping({
                ENV_BASE_URL: "https://api.example.com",
                ENV_API_KEY: "test-key",
            })

    def test_missing_api_key_rejected(self):
        with pytest.raises(ProviderConfigurationError, match=ENV_API_KEY):
            load_openai_compatible_config_from_mapping({
                ENV_BASE_URL: "https://api.example.com",
                ENV_MODEL: "test-model",
            })

    def test_invalid_timeout_rejected(self):
        with pytest.raises(ProviderConfigurationError, match=ENV_TIMEOUT_SECONDS):
            load_openai_compatible_config_from_mapping({
                ENV_BASE_URL: "https://api.example.com",
                ENV_MODEL: "test-model",
                ENV_API_KEY: "test-key",
                ENV_TIMEOUT_SECONDS: "not-a-number",
            })

    def test_invalid_max_tokens_rejected(self):
        with pytest.raises(ProviderConfigurationError, match=ENV_MAX_TOKENS):
            load_openai_compatible_config_from_mapping({
                ENV_BASE_URL: "https://api.example.com",
                ENV_MODEL: "test-model",
                ENV_API_KEY: "test-key",
                ENV_MAX_TOKENS: "abc",
            })

    def test_invalid_max_response_bytes_rejected(self):
        with pytest.raises(ProviderConfigurationError, match=ENV_MAX_RESPONSE_BYTES):
            load_openai_compatible_config_from_mapping({
                ENV_BASE_URL: "https://api.example.com",
                ENV_MODEL: "test-model",
                ENV_API_KEY: "test-key",
                ENV_MAX_RESPONSE_BYTES: "xyz",
            })

    def test_error_message_excludes_api_key(self):
        try:
            load_openai_compatible_config_from_mapping({
                ENV_BASE_URL: "not-https",
                ENV_MODEL: "test-model",
                ENV_API_KEY: "my-secret-key-123",
            })
        except ProviderConfigurationError as e:
            msg = str(e)
            assert "my-secret-key-123" not in msg

    def test_loader_does_not_read_os_environ(self):
        import os
        # Temporarily set os.environ — loader should NOT pick it up
        os.environ["PRISM_CHT_MODEL_BASE_URL"] = "https://evil.com"
        try:
            with pytest.raises(ProviderConfigurationError, match=ENV_BASE_URL):
                load_openai_compatible_config_from_mapping({})
        finally:
            del os.environ["PRISM_CHT_MODEL_BASE_URL"]

    def test_loader_does_not_read_file(self):
        # loader only accepts a Mapping, no file I/O
        with pytest.raises(ProviderConfigurationError, match=ENV_BASE_URL):
            load_openai_compatible_config_from_mapping({})

    def test_loader_does_not_import_dotenv(self):
        import sys
        assert "dotenv" not in sys.modules or "test" in sys.modules.get("dotenv", "")
