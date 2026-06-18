"""Tests for openai_compatible_client.py — main adapter behavior."""

import json

import pytest

from prismv4.prism_cht.llm_types import ModelMessage, ModelRequest, ModelResponse
from prismv4.prism_cht.provider_config import OpenAICompatibleChatConfig
from prismv4.prism_cht.provider_types import (
    ProviderHTTPError,
    ProviderResponseError,
    ProviderTransportError,
    ProviderUsage,
)
from prismv4.prism_cht.http_transport import HttpRequest, HttpResponse
from prismv4.prism_cht.openai_compatible_client import (
    OpenAICompatibleChatModelClient,
)


# ===========================================================================
# FakeTransport for testing
# ===========================================================================


class FakeTransport:
    """Deterministic transport that returns pre-configured responses.

    Accepts either ``HttpResponse`` objects or exceptions to raise.
    Records every received ``HttpRequest``.
    """

    def __init__(self, responses):
        import collections
        self._queue = collections.deque(responses)
        self._requests: list[HttpRequest] = []
        self._max_response_bytes_received: list[int] = []

    def send(self, *, request: HttpRequest, max_response_bytes: int) -> HttpResponse:
        self._requests.append(request)
        self._max_response_bytes_received.append(max_response_bytes)
        if not self._queue:
            raise RuntimeError("FakeTransport: no more responses queued")
        item = self._queue.popleft()
        if isinstance(item, Exception):
            raise item
        return item

    @property
    def requests(self):
        return tuple(self._requests)

    @property
    def max_response_bytes_calls(self):
        return tuple(self._max_response_bytes_received)


# ===========================================================================
# Helpers
# ===========================================================================


_TEST_CONFIG = OpenAICompatibleChatConfig(
    base_url="https://api.example.com",
    model="deepseek-v4-pro",
    api_key="test-secret-not-real",
    timeout_seconds=30.0,
    max_tokens=4096,
    max_response_bytes=2_000_000,
    json_mode=True,
)


def _make_request(purpose="test-purpose", attempt_index=0):
    return ModelRequest(
        purpose=purpose,
        messages=(
            ModelMessage(role="system", content="You are a helpful assistant."),
            ModelMessage(role="user", content="Hello"),
        ),
        attempt_index=attempt_index,
    )


def _make_success_response(content="response text"):
    """Build a valid success HttpResponse."""
    body = json.dumps({
        "choices": [
            {
                "message": {"content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        },
    }).encode("utf-8")
    return HttpResponse(status_code=200, headers={}, body=body)


def _make_error_response(status_code, body_text="error"):
    body = json.dumps({"error": body_text}).encode("utf-8")
    return HttpResponse(status_code=status_code, headers={}, body=body)


# ===========================================================================
# Tests
# ===========================================================================


class TestOpenAICompatibleChatModelClient:
    def test_satisfies_model_client_protocol(self):
        transport = FakeTransport([_make_success_response("hello")])
        client = OpenAICompatibleChatModelClient(
            config=_TEST_CONFIG,
            transport=transport,
        )
        result = client.complete(request=_make_request())
        assert isinstance(result, ModelResponse)
        assert result.content == "hello"

    def test_endpoint_includes_chat_completions(self):
        transport = FakeTransport([_make_success_response("ok")])
        client = OpenAICompatibleChatModelClient(
            config=_TEST_CONFIG,
            transport=transport,
        )
        client.complete(request=_make_request())
        http_req = transport.requests[0]
        assert http_req.url.endswith("/chat/completions")

    def test_request_method_is_post(self):
        transport = FakeTransport([_make_success_response("ok")])
        client = OpenAICompatibleChatModelClient(
            config=_TEST_CONFIG,
            transport=transport,
        )
        client.complete(request=_make_request())
        assert transport.requests[0].method == "POST"

    def test_content_type_header(self):
        transport = FakeTransport([_make_success_response("ok")])
        client = OpenAICompatibleChatModelClient(
            config=_TEST_CONFIG,
            transport=transport,
        )
        client.complete(request=_make_request())
        assert transport.requests[0].headers["Content-Type"] == "application/json"

    def test_accept_header(self):
        transport = FakeTransport([_make_success_response("ok")])
        client = OpenAICompatibleChatModelClient(
            config=_TEST_CONFIG,
            transport=transport,
        )
        client.complete(request=_make_request())
        assert transport.requests[0].headers["Accept"] == "application/json"

    def test_authorization_header(self):
        transport = FakeTransport([_make_success_response("ok")])
        client = OpenAICompatibleChatModelClient(
            config=_TEST_CONFIG,
            transport=transport,
        )
        client.complete(request=_make_request())
        assert transport.requests[0].headers["Authorization"] == "Bearer test-secret-not-real"

    def test_body_model_correct(self):
        transport = FakeTransport([_make_success_response("ok")])
        client = OpenAICompatibleChatModelClient(
            config=_TEST_CONFIG,
            transport=transport,
        )
        client.complete(request=_make_request())
        body = json.loads(transport.requests[0].body.decode())
        assert body["model"] == "deepseek-v4-pro"

    def test_body_messages_correct(self):
        transport = FakeTransport([_make_success_response("ok")])
        client = OpenAICompatibleChatModelClient(
            config=_TEST_CONFIG,
            transport=transport,
        )
        client.complete(request=_make_request())
        body = json.loads(transport.requests[0].body.decode())
        assert body["messages"] == [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
        ]

    def test_body_stream_is_false(self):
        transport = FakeTransport([_make_success_response("ok")])
        client = OpenAICompatibleChatModelClient(
            config=_TEST_CONFIG,
            transport=transport,
        )
        client.complete(request=_make_request())
        body = json.loads(transport.requests[0].body.decode())
        assert body["stream"] is False

    def test_body_omits_max_tokens_by_default(self):
        transport = FakeTransport([_make_success_response("ok")])
        cfg = OpenAICompatibleChatConfig(
            base_url="https://api.example.com",
            model="m",
            api_key="k",
        )
        client = OpenAICompatibleChatModelClient(
            config=cfg,
            transport=transport,
        )
        client.complete(request=_make_request())
        body = json.loads(transport.requests[0].body.decode())
        assert "max_tokens" not in body

    def test_body_includes_explicit_max_tokens(self):
        transport = FakeTransport([_make_success_response("ok")])
        cfg = OpenAICompatibleChatConfig(
            base_url="https://api.example.com",
            model="m",
            api_key="k",
            max_tokens=4096,
        )
        client = OpenAICompatibleChatModelClient(config=cfg, transport=transport)
        client.complete(request=_make_request())
        body = json.loads(transport.requests[0].body.decode())
        assert body["max_tokens"] == 4096

    def test_json_mode_adds_response_format(self):
        transport = FakeTransport([_make_success_response("ok")])
        cfg = OpenAICompatibleChatConfig(
            base_url="https://api.example.com",
            model="m",
            api_key="k",
            json_mode=True,
        )
        client = OpenAICompatibleChatModelClient(config=cfg, transport=transport)
        client.complete(request=_make_request())
        body = json.loads(transport.requests[0].body.decode())
        assert body["response_format"] == {"type": "json_object"}

    def test_json_mode_false_omits_response_format(self):
        transport = FakeTransport([_make_success_response("ok")])
        cfg = OpenAICompatibleChatConfig(
            base_url="https://api.example.com",
            model="m",
            api_key="k",
            json_mode=False,
        )
        client = OpenAICompatibleChatModelClient(config=cfg, transport=transport)
        client.complete(request=_make_request())
        body = json.loads(transport.requests[0].body.decode())
        assert "response_format" not in body

    def test_extra_body_merged(self):
        transport = FakeTransport([_make_success_response("ok")])
        cfg = OpenAICompatibleChatConfig(
            base_url="https://api.example.com",
            model="m",
            api_key="k",
            extra_body={"temperature": 0.5},
        )
        client = OpenAICompatibleChatModelClient(config=cfg, transport=transport)
        client.complete(request=_make_request())
        body = json.loads(transport.requests[0].body.decode())
        assert body["temperature"] == 0.5

    def test_body_excludes_api_key(self):
        transport = FakeTransport([_make_success_response("ok")])
        client = OpenAICompatibleChatModelClient(
            config=_TEST_CONFIG,
            transport=transport,
        )
        client.complete(request=_make_request())
        body = json.loads(transport.requests[0].body.decode())
        assert "api_key" not in body
        assert "test-secret-not-real" not in json.dumps(body)

    def test_body_excludes_purpose(self):
        transport = FakeTransport([_make_success_response("ok")])
        client = OpenAICompatibleChatModelClient(
            config=_TEST_CONFIG,
            transport=transport,
        )
        client.complete(request=_make_request(purpose="secret-purpose"))
        body = json.loads(transport.requests[0].body.decode())
        assert "secret-purpose" not in json.dumps(body)
        assert "purpose" not in body

    def test_body_excludes_attempt_index(self):
        transport = FakeTransport([_make_success_response("ok")])
        client = OpenAICompatibleChatModelClient(
            config=_TEST_CONFIG,
            transport=transport,
        )
        client.complete(request=_make_request(attempt_index=5))
        body = json.loads(transport.requests[0].body.decode())
        assert "attempt_index" not in body

    def test_body_json_deterministic(self):
        transport1 = FakeTransport([_make_success_response("ok")])
        transport2 = FakeTransport([_make_success_response("ok")])
        client1 = OpenAICompatibleChatModelClient(config=_TEST_CONFIG, transport=transport1)
        client2 = OpenAICompatibleChatModelClient(config=_TEST_CONFIG, transport=transport2)
        client1.complete(request=_make_request())
        client2.complete(request=_make_request())
        assert transport1.requests[0].body == transport2.requests[0].body

    def test_success_returns_model_response_content(self):
        transport = FakeTransport([_make_success_response("exact content")])
        client = OpenAICompatibleChatModelClient(
            config=_TEST_CONFIG,
            transport=transport,
        )
        result = client.complete(request=_make_request())
        assert result.content == "exact content"


# ===========================================================================
# Response parsing error tests
# ===========================================================================


class TestResponseParsingErrors:
    def test_choices_missing_rejected(self):
        body = json.dumps({"no_choices": True}).encode()
        response = HttpResponse(status_code=200, headers={}, body=body)
        transport = FakeTransport([response])
        client = OpenAICompatibleChatModelClient(config=_TEST_CONFIG, transport=transport)
        with pytest.raises(ProviderResponseError, match="choices"):
            client.complete(request=_make_request())

    def test_choices_empty_rejected(self):
        body = json.dumps({"choices": []}).encode()
        response = HttpResponse(status_code=200, headers={}, body=body)
        transport = FakeTransport([response])
        client = OpenAICompatibleChatModelClient(config=_TEST_CONFIG, transport=transport)
        with pytest.raises(ProviderResponseError, match="empty"):
            client.complete(request=_make_request())

    def test_message_missing_rejected(self):
        body = json.dumps({"choices": [{}]}).encode()
        response = HttpResponse(status_code=200, headers={}, body=body)
        transport = FakeTransport([response])
        client = OpenAICompatibleChatModelClient(config=_TEST_CONFIG, transport=transport)
        with pytest.raises(ProviderResponseError, match="message"):
            client.complete(request=_make_request())

    def test_content_missing_rejected(self):
        body = json.dumps({
            "choices": [{"message": {}, "finish_reason": "stop"}]
        }).encode()
        response = HttpResponse(status_code=200, headers={}, body=body)
        transport = FakeTransport([response])
        client = OpenAICompatibleChatModelClient(config=_TEST_CONFIG, transport=transport)
        with pytest.raises(ProviderResponseError, match="content"):
            client.complete(request=_make_request())

    def test_content_empty_rejected(self):
        body = json.dumps({
            "choices": [{"message": {"content": "  "}, "finish_reason": "stop"}]
        }).encode()
        response = HttpResponse(status_code=200, headers={}, body=body)
        transport = FakeTransport([response])
        client = OpenAICompatibleChatModelClient(config=_TEST_CONFIG, transport=transport)
        with pytest.raises(ProviderResponseError, match="non-empty"):
            client.complete(request=_make_request())

    def test_top_level_not_object_rejected(self):
        body = json.dumps([1, 2, 3]).encode()
        response = HttpResponse(status_code=200, headers={}, body=body)
        transport = FakeTransport([response])
        client = OpenAICompatibleChatModelClient(config=_TEST_CONFIG, transport=transport)
        with pytest.raises(ProviderResponseError, match="object"):
            client.complete(request=_make_request())

    def test_invalid_json_body_rejected(self):
        body = b"not json at all"
        response = HttpResponse(status_code=200, headers={}, body=body)
        transport = FakeTransport([response])
        client = OpenAICompatibleChatModelClient(config=_TEST_CONFIG, transport=transport)
        with pytest.raises(ProviderResponseError, match="not valid JSON"):
            client.complete(request=_make_request())

    def test_finish_reason_length_rejected(self):
        body = json.dumps({
            "choices": [
                {"message": {"content": "truncated"}, "finish_reason": "length"}
            ]
        }).encode()
        response = HttpResponse(status_code=200, headers={}, body=body)
        transport = FakeTransport([response])
        client = OpenAICompatibleChatModelClient(config=_TEST_CONFIG, transport=transport)
        with pytest.raises(ProviderResponseError, match="truncated"):
            client.complete(request=_make_request())

    def test_usage_missing_allowed(self):
        body = json.dumps({
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]
        }).encode()
        response = HttpResponse(status_code=200, headers={}, body=body)
        transport = FakeTransport([response])
        client = OpenAICompatibleChatModelClient(config=_TEST_CONFIG, transport=transport)
        result = client.complete(request=_make_request())
        assert result.content == "ok"

    def test_usage_correctly_parsed(self):
        body = json.dumps({
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
        }).encode()
        response = HttpResponse(status_code=200, headers={}, body=body)
        transport = FakeTransport([response])
        client = OpenAICompatibleChatModelClient(config=_TEST_CONFIG, transport=transport)
        client.complete(request=_make_request())
        audit = client.audit_records[0]
        assert audit.usage.prompt_tokens == 100
        assert audit.usage.completion_tokens == 50
        assert audit.usage.total_tokens == 150

    def test_usage_negative_rejected(self):
        body = json.dumps({
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": -1, "completion_tokens": 0, "total_tokens": -1},
        }).encode()
        response = HttpResponse(status_code=200, headers={}, body=body)
        transport = FakeTransport([response])
        client = OpenAICompatibleChatModelClient(config=_TEST_CONFIG, transport=transport)
        with pytest.raises(ProviderResponseError, match=">= 0"):
            client.complete(request=_make_request())

    def test_usage_bool_rejected(self):
        body = json.dumps({
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": True, "completion_tokens": 0, "total_tokens": 0},
        }).encode()
        response = HttpResponse(status_code=200, headers={}, body=body)
        transport = FakeTransport([response])
        client = OpenAICompatibleChatModelClient(config=_TEST_CONFIG, transport=transport)
        with pytest.raises(ProviderResponseError, match="bool"):
            client.complete(request=_make_request())

    def test_reasoning_content_not_used_as_content(self):
        body = json.dumps({
            "choices": [
                {
                    "message": {
                        "content": "final answer",
                        "reasoning_content": "this is reasoning, not the answer",
                    },
                    "finish_reason": "stop",
                }
            ]
        }).encode()
        response = HttpResponse(status_code=200, headers={}, body=body)
        transport = FakeTransport([response])
        client = OpenAICompatibleChatModelClient(config=_TEST_CONFIG, transport=transport)
        result = client.complete(request=_make_request())
        assert result.content == "final answer"
        assert "reasoning" not in result.content.lower()


# ===========================================================================
# HTTP error tests
# ===========================================================================


class TestHttpErrors:
    def test_non_2xx_raises_provider_http_error(self):
        transport = FakeTransport([_make_error_response(400)])
        client = OpenAICompatibleChatModelClient(config=_TEST_CONFIG, transport=transport)
        with pytest.raises(ProviderHTTPError) as exc:
            client.complete(request=_make_request())
        assert exc.value.status_code == 400

    def test_http_429_is_retryable(self):
        transport = FakeTransport([_make_error_response(429)])
        client = OpenAICompatibleChatModelClient(config=_TEST_CONFIG, transport=transport)
        with pytest.raises(ProviderHTTPError) as exc:
            client.complete(request=_make_request())
        assert exc.value.retryable is True

    def test_http_500_is_retryable(self):
        transport = FakeTransport([_make_error_response(500)])
        client = OpenAICompatibleChatModelClient(config=_TEST_CONFIG, transport=transport)
        with pytest.raises(ProviderHTTPError) as exc:
            client.complete(request=_make_request())
        assert exc.value.retryable is True

    def test_http_400_is_not_retryable(self):
        transport = FakeTransport([_make_error_response(400)])
        client = OpenAICompatibleChatModelClient(config=_TEST_CONFIG, transport=transport)
        with pytest.raises(ProviderHTTPError) as exc:
            client.complete(request=_make_request())
        assert exc.value.retryable is False

    def test_http_408_is_retryable(self):
        transport = FakeTransport([_make_error_response(408)])
        client = OpenAICompatibleChatModelClient(config=_TEST_CONFIG, transport=transport)
        with pytest.raises(ProviderHTTPError) as exc:
            client.complete(request=_make_request())
        assert exc.value.retryable is True

    def test_http_409_is_retryable(self):
        transport = FakeTransport([_make_error_response(409)])
        client = OpenAICompatibleChatModelClient(config=_TEST_CONFIG, transport=transport)
        with pytest.raises(ProviderHTTPError) as exc:
            client.complete(request=_make_request())
        assert exc.value.retryable is True

    def test_http_425_is_retryable(self):
        transport = FakeTransport([_make_error_response(425)])
        client = OpenAICompatibleChatModelClient(config=_TEST_CONFIG, transport=transport)
        with pytest.raises(ProviderHTTPError) as exc:
            client.complete(request=_make_request())
        assert exc.value.retryable is True

    def test_client_does_not_retry(self):
        # Even for retryable status, client just raises, no auto-retry
        transport = FakeTransport([_make_error_response(429), _make_success_response("ok")])
        client = OpenAICompatibleChatModelClient(config=_TEST_CONFIG, transport=transport)
        with pytest.raises(ProviderHTTPError):
            client.complete(request=_make_request())
        # The second response was never consumed (only 1 call made)
        assert len(transport.requests) == 1

    def test_client_no_provider_fallback(self):
        transport = FakeTransport([_make_error_response(500)])
        client = OpenAICompatibleChatModelClient(config=_TEST_CONFIG, transport=transport)
        with pytest.raises(ProviderHTTPError):
            client.complete(request=_make_request())

    def test_client_does_not_modify_model_request(self):
        transport = FakeTransport([_make_success_response("ok")])
        client = OpenAICompatibleChatModelClient(config=_TEST_CONFIG, transport=transport)
        req = _make_request()
        original_purpose = req.purpose
        original_messages = req.messages
        client.complete(request=req)
        assert req.purpose == original_purpose
        assert req.messages == original_messages

    def test_transport_error_propagated(self):
        transport = FakeTransport([ProviderTransportError("network down")])
        client = OpenAICompatibleChatModelClient(config=_TEST_CONFIG, transport=transport)
        with pytest.raises(ProviderTransportError, match="network down"):
            client.complete(request=_make_request())
