"""Tests for provider audit trail — audit records on success and failure."""

import json

import pytest

from prismv4.prism_cht.llm_types import ModelMessage, ModelRequest
from prismv4.prism_cht.provider_config import OpenAICompatibleChatConfig
from prismv4.prism_cht.provider_types import (
    ProviderCallAudit,
    ProviderHTTPError,
    ProviderResponseError,
    ProviderTransportError,
    ProviderUsage,
)
from prismv4.prism_cht.http_transport import HttpResponse
from prismv4.prism_cht.openai_compatible_client import (
    OpenAICompatibleChatModelClient,
)


# ===========================================================================
# FakeTransport
# ===========================================================================


class FakeTransport:
    def __init__(self, responses):
        import collections
        self._queue = collections.deque(responses)
        self._requests = []

    def send(self, *, request, max_response_bytes):
        self._requests.append(request)
        if not self._queue:
            raise RuntimeError("FakeTransport: no more responses")
        item = self._queue.popleft()
        if isinstance(item, Exception):
            raise item
        return item

    @property
    def requests(self):
        return tuple(self._requests)


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
)


def _make_request(purpose="test-purpose", attempt_index=0):
    return ModelRequest(
        purpose=purpose,
        messages=(
            ModelMessage(role="system", content="sys"),
            ModelMessage(role="user", content="hello"),
        ),
        attempt_index=attempt_index,
    )


def _make_success_response(content="response text"):
    body = json.dumps({
        "choices": [
            {
                "message": {"content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }).encode("utf-8")
    return HttpResponse(status_code=200, headers={}, body=body)


def _make_error_response(status_code):
    body = json.dumps({"error": "bad"}).encode()
    return HttpResponse(status_code=status_code, headers={}, body=body)


# ===========================================================================
# Tests
# ===========================================================================


class TestProviderAudit:
    def test_success_writes_one_audit(self):
        transport = FakeTransport([_make_success_response("hello")])
        client = OpenAICompatibleChatModelClient(config=_TEST_CONFIG, transport=transport)
        client.complete(request=_make_request())
        assert len(client.audit_records) == 1

    def test_transport_error_writes_audit(self):
        transport = FakeTransport([ProviderTransportError("network down")])
        client = OpenAICompatibleChatModelClient(config=_TEST_CONFIG, transport=transport)
        with pytest.raises(ProviderTransportError):
            client.complete(request=_make_request())
        assert len(client.audit_records) == 1
        assert client.audit_records[0].outcome == "transport_error"

    def test_http_error_writes_audit(self):
        transport = FakeTransport([_make_error_response(500)])
        client = OpenAICompatibleChatModelClient(config=_TEST_CONFIG, transport=transport)
        with pytest.raises(ProviderHTTPError):
            client.complete(request=_make_request())
        assert len(client.audit_records) == 1
        assert client.audit_records[0].outcome == "http_error"

    def test_response_error_writes_audit(self):
        transport = FakeTransport([
            HttpResponse(status_code=200, headers={}, body=b"not json")
        ])
        client = OpenAICompatibleChatModelClient(config=_TEST_CONFIG, transport=transport)
        with pytest.raises(ProviderResponseError):
            client.complete(request=_make_request())
        assert len(client.audit_records) == 1
        assert client.audit_records[0].outcome == "response_error"

    def test_each_call_writes_exactly_one_audit(self):
        transport = FakeTransport([
            _make_success_response("a"),
            _make_success_response("b"),
            _make_success_response("c"),
        ])
        client = OpenAICompatibleChatModelClient(config=_TEST_CONFIG, transport=transport)
        client.complete(request=_make_request())
        client.complete(request=_make_request())
        client.complete(request=_make_request())
        assert len(client.audit_records) == 3

    def test_audit_order_matches_call_order(self):
        transport = FakeTransport([
            _make_success_response("first"),
            _make_success_response("second"),
        ])
        client = OpenAICompatibleChatModelClient(config=_TEST_CONFIG, transport=transport)
        client.complete(request=_make_request(purpose="first", attempt_index=0))
        client.complete(request=_make_request(purpose="second", attempt_index=1))
        assert client.audit_records[0].purpose == "first"
        assert client.audit_records[1].purpose == "second"
        assert client.audit_records[0].attempt_index == 0
        assert client.audit_records[1].attempt_index == 1

    def test_audit_records_returns_tuple(self):
        transport = FakeTransport([_make_success_response("ok")])
        client = OpenAICompatibleChatModelClient(config=_TEST_CONFIG, transport=transport)
        client.complete(request=_make_request())
        assert isinstance(client.audit_records, tuple)

    def test_audit_records_immutable(self):
        transport = FakeTransport([_make_success_response("ok")])
        client = OpenAICompatibleChatModelClient(config=_TEST_CONFIG, transport=transport)
        client.complete(request=_make_request())
        records = client.audit_records
        with pytest.raises((TypeError, AttributeError)):
            records[0] = None  # type: ignore

    def test_audit_excludes_api_key(self):
        transport = FakeTransport([_make_success_response("ok")])
        client = OpenAICompatibleChatModelClient(config=_TEST_CONFIG, transport=transport)
        client.complete(request=_make_request())
        audit = client.audit_records[0]
        # Inspect all string fields for API key
        for field_name in ["purpose", "model", "endpoint", "finish_reason", "outcome"]:
            value = getattr(audit, field_name)
            if isinstance(value, str):
                assert "test-secret-not-real" not in value
        # ProviderCallAudit has no field for api_key
        assert not hasattr(audit, "api_key")

    def test_audit_excludes_authorization_header(self):
        transport = FakeTransport([_make_success_response("ok")])
        client = OpenAICompatibleChatModelClient(config=_TEST_CONFIG, transport=transport)
        client.complete(request=_make_request())
        audit = client.audit_records[0]
        assert "Authorization" not in str(audit.__dict__)
        assert "Bearer" not in str(audit.__dict__)

    def test_audit_excludes_full_prompt(self):
        transport = FakeTransport([_make_success_response("ok")])
        client = OpenAICompatibleChatModelClient(config=_TEST_CONFIG, transport=transport)
        req = _make_request()
        client.complete(request=req)
        audit = client.audit_records[0]
        msg_text = "You are a helpful assistant"
        # Audit should not contain the prompt text beyond purpose
        found = False
        for field_name in ["purpose", "model", "endpoint", "finish_reason", "outcome"]:
            value = str(getattr(audit, field_name))
            if "You are a helpful" in value and field_name != "purpose":
                found = True
        # The audit does not carry message content at all
        assert not hasattr(audit, "messages")
        assert not hasattr(audit, "prompt")

    def test_audit_excludes_full_response_body(self):
        transport = FakeTransport([_make_success_response("exact match text")])
        client = OpenAICompatibleChatModelClient(config=_TEST_CONFIG, transport=transport)
        client.complete(request=_make_request())
        audit = client.audit_records[0]
        # Audit has no response body field
        assert not hasattr(audit, "response_body")
        assert not hasattr(audit, "body")
        # Verify raw response body not in any audit field
        audit_dict = {
            "purpose": audit.purpose,
            "model": audit.model,
            "endpoint": audit.endpoint,
            "finish_reason": audit.finish_reason,
            "outcome": audit.outcome,
        }
        assert "exact match text" not in str(audit_dict)

    def test_audit_excludes_model_request(self):
        transport = FakeTransport([_make_success_response("ok")])
        client = OpenAICompatibleChatModelClient(config=_TEST_CONFIG, transport=transport)
        client.complete(request=_make_request())
        audit = client.audit_records[0]
        assert not hasattr(audit, "request")
        assert not hasattr(audit, "model_request")

    def test_audit_records_purpose(self):
        transport = FakeTransport([_make_success_response("ok")])
        client = OpenAICompatibleChatModelClient(config=_TEST_CONFIG, transport=transport)
        client.complete(request=_make_request(purpose="lead-decision"))
        assert client.audit_records[0].purpose == "lead-decision"

    def test_audit_records_attempt_index(self):
        transport = FakeTransport([_make_success_response("ok")])
        client = OpenAICompatibleChatModelClient(config=_TEST_CONFIG, transport=transport)
        client.complete(request=_make_request(attempt_index=3))
        assert client.audit_records[0].attempt_index == 3

    def test_audit_records_model(self):
        transport = FakeTransport([_make_success_response("ok")])
        client = OpenAICompatibleChatModelClient(config=_TEST_CONFIG, transport=transport)
        client.complete(request=_make_request())
        assert client.audit_records[0].model == "deepseek-v4-pro"

    def test_audit_records_endpoint(self):
        transport = FakeTransport([_make_success_response("ok")])
        client = OpenAICompatibleChatModelClient(config=_TEST_CONFIG, transport=transport)
        client.complete(request=_make_request())
        assert client.audit_records[0].endpoint.endswith("/chat/completions")

    def test_audit_records_status_code(self):
        transport = FakeTransport([_make_success_response("ok")])
        client = OpenAICompatibleChatModelClient(config=_TEST_CONFIG, transport=transport)
        client.complete(request=_make_request())
        assert client.audit_records[0].status_code == 200

    def test_audit_records_elapsed_ms(self):
        transport = FakeTransport([_make_success_response("ok")])
        client = OpenAICompatibleChatModelClient(config=_TEST_CONFIG, transport=transport)
        client.complete(request=_make_request())
        assert client.audit_records[0].elapsed_ms >= 0

    def test_audit_records_request_bytes(self):
        transport = FakeTransport([_make_success_response("ok")])
        client = OpenAICompatibleChatModelClient(config=_TEST_CONFIG, transport=transport)
        client.complete(request=_make_request())
        assert client.audit_records[0].request_bytes > 0

    def test_audit_records_response_bytes(self):
        transport = FakeTransport([_make_success_response("ok")])
        client = OpenAICompatibleChatModelClient(config=_TEST_CONFIG, transport=transport)
        client.complete(request=_make_request())
        assert client.audit_records[0].response_bytes > 0

    def test_audit_records_finish_reason(self):
        transport = FakeTransport([_make_success_response("ok")])
        client = OpenAICompatibleChatModelClient(config=_TEST_CONFIG, transport=transport)
        client.complete(request=_make_request())
        assert client.audit_records[0].finish_reason == "stop"

    def test_audit_records_usage(self):
        transport = FakeTransport([_make_success_response("ok")])
        client = OpenAICompatibleChatModelClient(config=_TEST_CONFIG, transport=transport)
        client.complete(request=_make_request())
        assert client.audit_records[0].usage.prompt_tokens == 10

    def test_usage_none_when_missing(self):
        body = json.dumps({
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]
        }).encode()
        transport = FakeTransport([HttpResponse(status_code=200, headers={}, body=body)])
        client = OpenAICompatibleChatModelClient(config=_TEST_CONFIG, transport=transport)
        client.complete(request=_make_request())
        assert client.audit_records[0].usage.prompt_tokens is None
        assert client.audit_records[0].usage.completion_tokens is None
        assert client.audit_records[0].usage.total_tokens is None

    def test_exception_not_swallowed(self):
        transport = FakeTransport([ProviderTransportError("fail")])
        client = OpenAICompatibleChatModelClient(config=_TEST_CONFIG, transport=transport)
        with pytest.raises(ProviderTransportError):
            client.complete(request=_make_request())
        # Audit was still written
        assert len(client.audit_records) == 1


class TestProviderCallAuditValidation:
    def test_reject_empty_purpose(self):
        with pytest.raises(ValueError, match="purpose"):
            ProviderCallAudit(
                purpose="",
                attempt_index=0,
                model="m",
                endpoint="e",
                status_code=200,
                elapsed_ms=1.0,
                request_bytes=1,
                response_bytes=1,
                finish_reason=None,
                usage=ProviderUsage(),
                outcome="success",
            )

    def test_reject_invalid_outcome(self):
        with pytest.raises(ValueError, match="outcome"):
            ProviderCallAudit(
                purpose="test",
                attempt_index=0,
                model="m",
                endpoint="e",
                status_code=200,
                elapsed_ms=1.0,
                request_bytes=1,
                response_bytes=1,
                finish_reason=None,
                usage=ProviderUsage(),
                outcome="invalid",
            )

    def test_reject_negative_elapsed_ms(self):
        with pytest.raises(ValueError, match="elapsed_ms"):
            ProviderCallAudit(
                purpose="test",
                attempt_index=0,
                model="m",
                endpoint="e",
                status_code=200,
                elapsed_ms=-1.0,
                request_bytes=1,
                response_bytes=1,
                finish_reason=None,
                usage=ProviderUsage(),
                outcome="success",
            )
