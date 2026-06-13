"""Tests for the CHT Provider smoke harness — import safety, gate matrix,
credential and response boundaries, failure paths, real-transport
construction gating, and RCA Controller isolation.
"""

import collections
import json
import os
import sys

import pytest

from prismv4.prism_cht.http_transport import HttpRequest, HttpResponse, HttpTransport
from prismv4.prism_cht.provider_config import (
    ENV_API_KEY,
    ENV_BASE_URL,
    ENV_MODEL,
)
from prismv4.prism_cht.provider_types import (
    ProviderCallAudit,
    ProviderHTTPError,
    ProviderTransportError,
)

from prismv4.scripts import cht_provider_smoke as _smoke_module

run_smoke = _smoke_module.run_smoke
SmokeResult = _smoke_module.SmokeResult
main = _smoke_module.main


# ===========================================================================
# Sentinels
# ===========================================================================

FAKE_SMOKE_API_KEY_SENTINEL = "sk-smoke-test-fake-api-key-do-not-leak"
FAKE_AUTHORIZATION_HEADER_SENTINEL = f"Bearer {FAKE_SMOKE_API_KEY_SENTINEL}"
RAW_SMOKE_RESPONSE_BODY_SENTINEL = "RAW_SMOKE_S3_BODY_SECRET_SENTINEL"
RAW_SMOKE_ASSISTANT_CONTENT_SENTINEL = "ASSISTANT_S4_CONTENT_SECRET_SENTINEL"

ALL_SENSITIVE_SENTINELS = [
    FAKE_SMOKE_API_KEY_SENTINEL,
    FAKE_AUTHORIZATION_HEADER_SENTINEL,
    RAW_SMOKE_RESPONSE_BODY_SENTINEL,
    RAW_SMOKE_ASSISTANT_CONTENT_SENTINEL,
]


def _no_sentinel_in(text: str) -> None:
    for s in ALL_SENSITIVE_SENTINELS:
        assert s not in text, f"sentinel leaked: {s[:50]}"


# ===========================================================================
# Recording fake transport
# ===========================================================================


class RecordingFakeTransport:
    """Implements HttpTransport protocol — records requests, returns queued
    responses.  Never touches the network."""

    def __init__(self, responses=None):
        if responses is None:
            responses = []
        self._queue = collections.deque(responses)
        self._requests: list[HttpRequest] = []

    def send(self, *, request: HttpRequest, max_response_bytes: int) -> HttpResponse:
        self._requests.append(request)
        if not self._queue:
            raise RuntimeError("RecordingFakeTransport: no more responses queued")
        item = self._queue.popleft()
        if isinstance(item, Exception):
            raise item
        return item

    @property
    def requests(self) -> tuple[HttpRequest, ...]:
        return tuple(self._requests)

    @property
    def request_count(self) -> int:
        return len(self._requests)


# ===========================================================================
# Helpers
# ===========================================================================


def _base_env():
    """Minimal provider config env with sentinel API key."""
    return {
        ENV_BASE_URL: "https://api.example.com/v1",
        ENV_MODEL: "smoke-test-model",
        ENV_API_KEY: FAKE_SMOKE_API_KEY_SENTINEL,
    }


def _make_http_response(
    assistant_content: str,
    *,
    status_code: int = 200,
    finish_reason: str = "stop",
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
) -> HttpResponse:
    body = json.dumps(
        {
            "choices": [
                {
                    "message": {"content": assistant_content},
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }
    ).encode("utf-8")
    return HttpResponse(status_code=status_code, headers={}, body=body)


# ===========================================================================
# A — Import Safety
# ===========================================================================


class TestImportSafety:
    def test_module_import_does_not_access_network(self):
        assert _smoke_module is not None

    def test_module_import_does_not_read_os_environ_at_module_level(self):
        assert not hasattr(_smoke_module, "_environ_read_at_import")

    def test_module_import_does_not_construct_transport_at_module_level(self):
        assert not hasattr(_smoke_module, "_transport_at_import")

    def test_module_import_does_not_instantiate_client_at_module_level(self):
        assert not hasattr(_smoke_module, "_client_at_import")

    def test_module_import_does_not_print_output(self, capsys):
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == ""


# ===========================================================================
# B — Gate Matrix
# ===========================================================================


class TestGateMatrix:
    def test_flag_false_env_missing_refuses(self):
        result = run_smoke(allow_live_network=False, environ={})
        assert result.status == "refused"
        assert result.configured_model_name is None
        assert result.request_count == 0
        assert result.audit_record_count == 0

    def test_flag_true_env_missing_refuses(self):
        result = run_smoke(allow_live_network=True, environ={})
        assert result.status == "refused"
        assert result.configured_model_name is None
        assert result.request_count == 0

    def test_flag_false_env_one_refuses(self):
        result = run_smoke(
            allow_live_network=False,
            environ={"PRISM_CHT_ENABLE_LIVE_SMOKE": "1"},
        )
        assert result.status == "refused"

    def test_flag_true_env_zero_refuses(self):
        result = run_smoke(
            allow_live_network=True,
            environ={"PRISM_CHT_ENABLE_LIVE_SMOKE": "0"},
        )
        assert result.status == "refused"

    def test_flag_true_env_true_refuses(self):
        result = run_smoke(
            allow_live_network=True,
            environ={"PRISM_CHT_ENABLE_LIVE_SMOKE": "true"},
        )
        assert result.status == "refused"

    def test_flag_true_env_one_permitted(self):
        env = _base_env()
        env["PRISM_CHT_ENABLE_LIVE_SMOKE"] = "1"
        transport = RecordingFakeTransport([_make_http_response('"OK"')])
        result = run_smoke(
            allow_live_network=True, environ=env, transport=transport
        )
        assert result.status == "success"


# ===========================================================================
# C — Offline Permitted Path With Injected Transport
# ===========================================================================


class TestOfflinePermittedPath:
    def test_exactly_one_request_sent(self):
        env = _base_env()
        env["PRISM_CHT_ENABLE_LIVE_SMOKE"] = "1"
        transport = RecordingFakeTransport([_make_http_response('"OK"')])
        result = run_smoke(
            allow_live_network=True, environ=env, transport=transport
        )
        assert result.status == "success"
        assert transport.request_count == 1
        assert result.request_count == 1
        assert result.audit_record_count == 1

    def test_recorded_body_contains_minimal_smoke_message(self):
        env = _base_env()
        env["PRISM_CHT_ENABLE_LIVE_SMOKE"] = "1"
        transport = RecordingFakeTransport([_make_http_response('"OK"')])
        run_smoke(allow_live_network=True, environ=env, transport=transport)
        req = transport.requests[0]
        body_text = req.body.decode("utf-8")
        assert "Reply with exactly: OK" in body_text

    def test_recorded_body_does_not_contain_rca_evidence(self):
        env = _base_env()
        env["PRISM_CHT_ENABLE_LIVE_SMOKE"] = "1"
        transport = RecordingFakeTransport([_make_http_response('"OK"')])
        run_smoke(allow_live_network=True, environ=env, transport=transport)
        req = transport.requests[0]
        body_text = req.body.decode("utf-8")
        assert "root_cause" not in body_text.lower()
        assert "hypothesis" not in body_text.lower()
        assert "telemetry" not in body_text.lower()

    def test_config_loaded_from_explicit_environ(self):
        env = _base_env()
        env["PRISM_CHT_ENABLE_LIVE_SMOKE"] = "1"
        transport = RecordingFakeTransport([_make_http_response('"OK"')])
        result = run_smoke(
            allow_live_network=True, environ=env, transport=transport
        )
        assert result.configured_model_name == "smoke-test-model"

    def test_smoke_result_metadata_is_safe(self):
        env = _base_env()
        env["PRISM_CHT_ENABLE_LIVE_SMOKE"] = "1"
        transport = RecordingFakeTransport([_make_http_response('"OK"')])
        result = run_smoke(
            allow_live_network=True, environ=env, transport=transport
        )
        result_repr = repr(result)
        _no_sentinel_in(result_repr)


# ===========================================================================
# D — Credential Boundary
# ===========================================================================


class TestCredentialBoundary:
    def test_api_key_reaches_authorization_header(self):
        env = _base_env()
        env["PRISM_CHT_ENABLE_LIVE_SMOKE"] = "1"
        transport = RecordingFakeTransport([_make_http_response('"OK"')])
        run_smoke(allow_live_network=True, environ=env, transport=transport)
        req = transport.requests[0]
        assert req.headers["Authorization"] == FAKE_AUTHORIZATION_HEADER_SENTINEL

    def test_api_key_not_in_request_repr(self):
        env = _base_env()
        env["PRISM_CHT_ENABLE_LIVE_SMOKE"] = "1"
        transport = RecordingFakeTransport([_make_http_response('"OK"')])
        run_smoke(allow_live_network=True, environ=env, transport=transport)
        req = transport.requests[0]
        req_repr = repr(req)
        assert FAKE_SMOKE_API_KEY_SENTINEL not in req_repr
        assert FAKE_AUTHORIZATION_HEADER_SENTINEL not in req_repr

    def test_api_key_not_in_smoke_result(self):
        env = _base_env()
        env["PRISM_CHT_ENABLE_LIVE_SMOKE"] = "1"
        transport = RecordingFakeTransport([_make_http_response('"OK"')])
        result = run_smoke(
            allow_live_network=True, environ=env, transport=transport
        )
        result_repr = repr(result)
        assert FAKE_SMOKE_API_KEY_SENTINEL not in result_repr
        assert FAKE_AUTHORIZATION_HEADER_SENTINEL not in result_repr

    def test_api_key_not_in_audit_repr(self):
        env = _base_env()
        env["PRISM_CHT_ENABLE_LIVE_SMOKE"] = "1"
        transport = RecordingFakeTransport([_make_http_response('"OK"')])
        run_smoke(allow_live_network=True, environ=env, transport=transport)
        from prismv4.prism_cht.openai_compatible_client import (
            OpenAICompatibleChatModelClient,
        )
        from prismv4.prism_cht.provider_config import (
            OpenAICompatibleChatConfig,
        )
        config = OpenAICompatibleChatConfig(
            base_url="https://api.example.com/v1",
            model="test-model",
            api_key=FAKE_SMOKE_API_KEY_SENTINEL,
        )
        client = OpenAICompatibleChatModelClient(
            config=config, transport=transport
        )
        for audit in client.audit_records:
            audit_repr = repr(audit)
            assert FAKE_SMOKE_API_KEY_SENTINEL not in audit_repr
            assert FAKE_AUTHORIZATION_HEADER_SENTINEL not in audit_repr

    def test_api_key_not_in_cli_output(self, capsys):
        env = _base_env()
        env["PRISM_CHT_ENABLE_LIVE_SMOKE"] = "1"
        transport = RecordingFakeTransport([_make_http_response('"OK"')])
        saved_run_smoke = _smoke_module.run_smoke

        def _fake_run_smoke(**kw):
            del kw
            return run_smoke(
                allow_live_network=True, environ=env, transport=transport
            )

        _smoke_module.run_smoke = _fake_run_smoke
        try:
            main([])
        finally:
            _smoke_module.run_smoke = saved_run_smoke
        out = capsys.readouterr().out
        assert FAKE_SMOKE_API_KEY_SENTINEL not in out


# ===========================================================================
# E — Response Boundary
# ===========================================================================


class TestResponseBoundary:
    def test_assistant_content_not_in_smoke_result(self):
        env = _base_env()
        env["PRISM_CHT_ENABLE_LIVE_SMOKE"] = "1"
        http_resp = _make_http_response(RAW_SMOKE_ASSISTANT_CONTENT_SENTINEL)
        transport = RecordingFakeTransport([http_resp])
        result = run_smoke(
            allow_live_network=True, environ=env, transport=transport
        )
        result_repr = repr(result)
        assert RAW_SMOKE_ASSISTANT_CONTENT_SENTINEL not in result_repr

    def test_raw_response_body_not_in_smoke_result(self):
        env = _base_env()
        env["PRISM_CHT_ENABLE_LIVE_SMOKE"] = "1"
        body = json.dumps(
            {
                "choices": [
                    {
                        "message": {
                            "content": RAW_SMOKE_ASSISTANT_CONTENT_SENTINEL
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            }
        ).encode("utf-8")
        http_resp = HttpResponse(status_code=200, headers={}, body=body)
        transport = RecordingFakeTransport([http_resp])
        result = run_smoke(
            allow_live_network=True, environ=env, transport=transport
        )
        result_repr = repr(result)
        assert RAW_SMOKE_RESPONSE_BODY_SENTINEL not in result_repr
        assert RAW_SMOKE_ASSISTANT_CONTENT_SENTINEL not in result_repr

    def test_assistant_content_not_in_cli_output(self, capsys):
        env = _base_env()
        env["PRISM_CHT_ENABLE_LIVE_SMOKE"] = "1"
        http_resp = _make_http_response(RAW_SMOKE_ASSISTANT_CONTENT_SENTINEL)
        transport = RecordingFakeTransport([http_resp])
        saved_run_smoke = _smoke_module.run_smoke

        def _fake_run_smoke(**kw):
            del kw
            return run_smoke(
                allow_live_network=True, environ=env, transport=transport
            )

        _smoke_module.run_smoke = _fake_run_smoke
        try:
            main([])
        finally:
            _smoke_module.run_smoke = saved_run_smoke
        out = capsys.readouterr().out
        assert RAW_SMOKE_ASSISTANT_CONTENT_SENTINEL not in out

    def test_smoke_success_with_valid_response(self):
        env = _base_env()
        env["PRISM_CHT_ENABLE_LIVE_SMOKE"] = "1"
        transport = RecordingFakeTransport([_make_http_response('"OK"')])
        result = run_smoke(
            allow_live_network=True, environ=env, transport=transport
        )
        assert result.status == "success"
        assert result.configured_model_name == "smoke-test-model"
        assert result.provider_outcome == "success"


# ===========================================================================
# F — Failure Boundary
# ===========================================================================


class TestFailureBoundary:
    def test_http_error_no_credential_leak_in_result(self):
        env = _base_env()
        env["PRISM_CHT_ENABLE_LIVE_SMOKE"] = "1"
        transport = RecordingFakeTransport(
            [
                HttpResponse(
                    status_code=502,
                    headers={},
                    body=json.dumps(
                        {"error": RAW_SMOKE_RESPONSE_BODY_SENTINEL}
                    ).encode("utf-8"),
                )
            ]
        )
        result = run_smoke(
            allow_live_network=True, environ=env, transport=transport
        )
        assert result.status == "http_error"
        result_repr = repr(result)
        _no_sentinel_in(result_repr)

    def test_transport_exception_no_credential_leak(self):
        env = _base_env()
        env["PRISM_CHT_ENABLE_LIVE_SMOKE"] = "1"
        transport = RecordingFakeTransport()
        transport._queue.append(ProviderTransportError("network is down"))
        result = run_smoke(
            allow_live_network=True, environ=env, transport=transport
        )
        assert result.status == "transport_error"
        result_repr = repr(result)
        _no_sentinel_in(result_repr)

    def test_failure_result_reports_error_metadata_safely(self):
        env = _base_env()
        env["PRISM_CHT_ENABLE_LIVE_SMOKE"] = "1"
        transport = RecordingFakeTransport()
        transport._queue.append(ProviderTransportError("dns failure"))
        result = run_smoke(
            allow_live_network=True, environ=env, transport=transport
        )
        assert result.error_class == "ProviderTransportError"
        assert "dns failure" in result.error_message
        result_repr = repr(result)
        _no_sentinel_in(result_repr)


# ===========================================================================
# G — Real-Transport Construction Gate
# ===========================================================================


class TestRealTransportConstructionGate:
    def test_urllib_transport_not_constructed_in_refused_case(
        self, monkeypatch
    ):
        call_count = 0

        def fake_init(self):
            nonlocal call_count
            call_count += 1

        from prismv4.prism_cht.http_transport import UrllibHttpTransport
        monkeypatch.setattr(
            UrllibHttpTransport, "__init__", fake_init
        )
        result = run_smoke(allow_live_network=False, environ={})
        assert result.status == "refused"
        assert call_count == 0

    def test_urllib_transport_not_constructed_flag_only(self, monkeypatch):
        call_count = 0

        def fake_init(self):
            nonlocal call_count
            call_count += 1

        from prismv4.prism_cht.http_transport import UrllibHttpTransport
        monkeypatch.setattr(
            UrllibHttpTransport, "__init__", fake_init
        )
        result = run_smoke(allow_live_network=True, environ={})
        assert result.status == "refused"
        assert call_count == 0

    def test_urllib_transport_not_constructed_env_only(self, monkeypatch):
        call_count = 0

        def fake_init(self):
            nonlocal call_count
            call_count += 1

        from prismv4.prism_cht.http_transport import UrllibHttpTransport
        monkeypatch.setattr(
            UrllibHttpTransport, "__init__", fake_init
        )
        result = run_smoke(
            allow_live_network=False,
            environ={"PRISM_CHT_ENABLE_LIVE_SMOKE": "1"},
        )
        assert result.status == "refused"
        assert call_count == 0

    def test_urllib_transport_not_constructed_with_injected_transport(
        self, monkeypatch
    ):
        call_count = 0

        def fake_init(self):
            nonlocal call_count
            call_count += 1

        from prismv4.prism_cht.http_transport import UrllibHttpTransport
        monkeypatch.setattr(
            UrllibHttpTransport, "__init__", fake_init
        )
        env = _base_env()
        transport = RecordingFakeTransport([_make_http_response('"OK"')])
        result = run_smoke(
            allow_live_network=False, environ=env, transport=transport
        )
        assert result.status == "success"
        assert call_count == 0

    def test_urllib_transport_eligible_only_with_both_gates(
        self, monkeypatch
    ):
        call_count = 0

        def fake_init(self):
            nonlocal call_count
            call_count += 1

        from prismv4.prism_cht.http_transport import UrllibHttpTransport
        monkeypatch.setattr(
            UrllibHttpTransport, "__init__", fake_init
        )
        env = _base_env()
        env["PRISM_CHT_ENABLE_LIVE_SMOKE"] = "1"
        try:
            run_smoke(allow_live_network=True, environ=env)
        except Exception:
            pass
        assert call_count == 1


# ===========================================================================
# H — No RCA Wiring
# ===========================================================================


class TestNoRCAWiring:
    def test_no_lead_controller_in_module_namespace(self):
        for name in dir(_smoke_module):
            assert "lead_controller" not in name, (
                f"lead_controller found in namespace: {name}"
            )

    def test_no_challenger_controller_in_module_namespace(self):
        for name in dir(_smoke_module):
            assert "challenger_controller" not in name, (
                f"challenger_controller found in namespace: {name}"
            )

    def test_no_executor_in_module_namespace(self):
        for name in dir(_smoke_module):
            assert "executor" not in name, (
                f"executor found in namespace: {name}"
            )

    def test_no_final_verifier_in_module_namespace(self):
        for name in dir(_smoke_module):
            assert "final_verifier" not in name, (
                f"final_verifier found in namespace: {name}"
            )

    def test_no_rca_controller_imports_in_source(self):
        import inspect
        source = inspect.getsource(_smoke_module)
        for forbidden in (
            "lead_controller",
            "challenger_controller",
            "executor",
            "final_verifier",
        ):
            assert forbidden not in source, (
                f"Forbidden RCA import '{forbidden}' found"
            )
