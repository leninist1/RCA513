"""Tests for CHT-5.2B-2: hardened sensitive DTO repr surfaces."""

import pytest

from prismv4.prism_cht.http_transport import HttpRequest, HttpResponse
from prismv4.prism_cht.llm_types import ModelMessage, ModelRequest, ModelResponse


# ---------------------------------------------------------------------------
# Conspicuous sentinels
# ---------------------------------------------------------------------------

RAW_PROMPT_CONTENT_SENTINEL = "RAW_PROMPT_TOP_SECRET_S2_PAYLOAD_DO_NOT_LEAK"
PARSED_MODEL_PAYLOAD_SENTINEL = "PARSED_MODEL_SECRET_S4_PAYLOAD_DO_NOT_LEAK"
RAW_HTTP_RESPONSE_BODY_SENTINEL = b"RAW_HTTP_S3_BODY_DO_NOT_LEAK"
SENSITIVE_HTTP_HEADER_SENTINEL = "X-INTERNAL-SECRET-HDR-DO-NOT-LEAK"
AUTHORIZATION_SENTINEL = "Bearer sk-the-quick-brown-fox-leaked-token"


# ===========================================================================
# Helpers
# ===========================================================================


def _sentinel_not_in(text: str, sentinel: object) -> None:
    """Assert that a text representation does not contain a sentinel."""
    # For bytes sentinels, decode both
    check = text
    if isinstance(sentinel, bytes):
        assert sentinel not in check.encode("utf-8", errors="replace"), (
            f"bytes sentinel leaked: {check[:200]!r}"
        )
    else:
        assert sentinel not in check, f"sentinel leaked: {check[:200]!r}"


# ===========================================================================
# A — ModelMessage
# ===========================================================================


class TestModelMessageReprSafety:
    def test_content_still_accessible(self):
        msg = ModelMessage(role="user", content=RAW_PROMPT_CONTENT_SENTINEL)
        assert msg.content == RAW_PROMPT_CONTENT_SENTINEL

    def test_repr_excludes_content(self):
        msg = ModelMessage(role="user", content=RAW_PROMPT_CONTENT_SENTINEL)
        _sentinel_not_in(repr(msg), RAW_PROMPT_CONTENT_SENTINEL)

    def test_class_name_visible(self):
        msg = ModelMessage(role="system", content=RAW_PROMPT_CONTENT_SENTINEL)
        assert "ModelMessage" in repr(msg)

    def test_role_still_visible(self):
        msg = ModelMessage(role="system", content=RAW_PROMPT_CONTENT_SENTINEL)
        assert "system" in repr(msg)

    def test_equality_preserved(self):
        a = ModelMessage(role="user", content=RAW_PROMPT_CONTENT_SENTINEL)
        b = ModelMessage(role="user", content=RAW_PROMPT_CONTENT_SENTINEL)
        c = ModelMessage(role="user", content="different")
        assert a == b
        assert a != c

    def test_frozen_still_enforced(self):
        msg = ModelMessage(role="system", content="test")
        with pytest.raises(Exception):
            msg.role = "user"  # type: ignore[attr-defined]

    def test_validation_still_works(self):
        with pytest.raises(ValueError, match="content"):
            ModelMessage(role="system", content="")
        with pytest.raises(ValueError, match="role"):
            ModelMessage(role="assistant", content="x")

    def test_accidental_exception_formatting(self):
        msg = ModelMessage(role="user", content=RAW_PROMPT_CONTENT_SENTINEL)
        formatted = f"debug object: {msg!r}"
        _sentinel_not_in(formatted, RAW_PROMPT_CONTENT_SENTINEL)


# ===========================================================================
# B — ModelRequest
# ===========================================================================


class TestModelRequestReprSafety:
    def test_messages_still_accessible(self):
        msg = ModelMessage(role="user", content=RAW_PROMPT_CONTENT_SENTINEL)
        req = ModelRequest(purpose="test", messages=(msg,), attempt_index=0)
        assert req.messages[0].content == RAW_PROMPT_CONTENT_SENTINEL

    def test_repr_excludes_messages(self):
        msg = ModelMessage(role="user", content=RAW_PROMPT_CONTENT_SENTINEL)
        req = ModelRequest(purpose="test", messages=(msg,), attempt_index=0)
        _sentinel_not_in(repr(req), RAW_PROMPT_CONTENT_SENTINEL)

    def test_nested_message_content_not_leaked(self):
        msg = ModelMessage(role="user", content=RAW_PROMPT_CONTENT_SENTINEL)
        req = ModelRequest(purpose="test", messages=(msg,), attempt_index=0)
        # Recursive repr of the request must not expose nested message content
        request_repr = repr(req)
        _sentinel_not_in(request_repr, RAW_PROMPT_CONTENT_SENTINEL)

    def test_harmless_fields_visible(self):
        msg = ModelMessage(role="user", content="irrelevant")
        req = ModelRequest(purpose="investigate", messages=(msg,), attempt_index=42)
        req_repr = repr(req)
        assert "investigate" in req_repr
        assert "42" in req_repr
        assert "ModelRequest" in req_repr

    def test_equality_preserved(self):
        msg = ModelMessage(role="user", content=RAW_PROMPT_CONTENT_SENTINEL)
        a = ModelRequest(purpose="p", messages=(msg,), attempt_index=1)
        b = ModelRequest(purpose="p", messages=(msg,), attempt_index=1)
        assert a == b

    def test_validation_still_works(self):
        with pytest.raises(ValueError, match="purpose"):
            ModelRequest(purpose="", messages=(ModelMessage(role="system", content="x"),), attempt_index=0)
        with pytest.raises(ValueError, match="messages"):
            ModelRequest(purpose="p", messages=(), attempt_index=0)

    def test_accidental_exception_formatting(self):
        msg = ModelMessage(role="user", content=RAW_PROMPT_CONTENT_SENTINEL)
        req = ModelRequest(purpose="test", messages=(msg,), attempt_index=0)
        formatted = f"debug object: {req!r}"
        _sentinel_not_in(formatted, RAW_PROMPT_CONTENT_SENTINEL)


# ===========================================================================
# C — ModelResponse
# ===========================================================================


class TestModelResponseReprSafety:
    def test_content_still_accessible(self):
        resp = ModelResponse(content=PARSED_MODEL_PAYLOAD_SENTINEL)
        assert resp.content == PARSED_MODEL_PAYLOAD_SENTINEL

    def test_repr_excludes_content(self):
        resp = ModelResponse(content=PARSED_MODEL_PAYLOAD_SENTINEL)
        _sentinel_not_in(repr(resp), PARSED_MODEL_PAYLOAD_SENTINEL)

    def test_class_name_visible(self):
        resp = ModelResponse(content=PARSED_MODEL_PAYLOAD_SENTINEL)
        assert "ModelResponse" in repr(resp)

    def test_equality_preserved(self):
        a = ModelResponse(content=PARSED_MODEL_PAYLOAD_SENTINEL)
        b = ModelResponse(content=PARSED_MODEL_PAYLOAD_SENTINEL)
        c = ModelResponse(content="different")
        assert a == b
        assert a != c

    def test_frozen_still_enforced(self):
        resp = ModelResponse(content="test")
        with pytest.raises(Exception):
            resp.content = "other"  # type: ignore[attr-defined]

    def test_validation_still_works(self):
        with pytest.raises(ValueError, match="content"):
            ModelResponse(content="")

    def test_accidental_exception_formatting(self):
        resp = ModelResponse(content=PARSED_MODEL_PAYLOAD_SENTINEL)
        formatted = f"debug object: {resp!r}"
        _sentinel_not_in(formatted, PARSED_MODEL_PAYLOAD_SENTINEL)


# ===========================================================================
# D — HttpResponse
# ===========================================================================


class TestHttpResponseReprSafety:
    def test_headers_and_body_still_accessible(self):
        resp = HttpResponse(
            status_code=200,
            headers={SENSITIVE_HTTP_HEADER_SENTINEL: AUTHORIZATION_SENTINEL},
            body=RAW_HTTP_RESPONSE_BODY_SENTINEL,
        )
        assert resp.headers[SENSITIVE_HTTP_HEADER_SENTINEL] == AUTHORIZATION_SENTINEL
        assert resp.body == RAW_HTTP_RESPONSE_BODY_SENTINEL

    def test_repr_excludes_sensitive_header_value(self):
        resp = HttpResponse(
            status_code=200,
            headers={SENSITIVE_HTTP_HEADER_SENTINEL: AUTHORIZATION_SENTINEL},
            body=b"ok",
        )
        _sentinel_not_in(repr(resp), AUTHORIZATION_SENTINEL)

    def test_repr_excludes_body(self):
        resp = HttpResponse(
            status_code=200,
            headers={},
            body=RAW_HTTP_RESPONSE_BODY_SENTINEL,
        )
        _sentinel_not_in(repr(resp), RAW_HTTP_RESPONSE_BODY_SENTINEL)

    def test_status_code_visible(self):
        resp = HttpResponse(
            status_code=201,
            headers={"Authorization": AUTHORIZATION_SENTINEL},
            body=RAW_HTTP_RESPONSE_BODY_SENTINEL,
        )
        assert "201" in repr(resp)

    def test_class_name_visible(self):
        resp = HttpResponse(status_code=500, headers={}, body=b"")
        assert "HttpResponse" in repr(resp)

    def test_body_coerced_to_bytes(self):
        resp = HttpResponse(status_code=200, headers={}, body="string body")
        assert isinstance(resp.body, bytes)
        assert resp.body == b"string body"

    def test_headers_frozen_mapping(self):
        resp = HttpResponse(status_code=200, headers={"a": "1"}, body=b"")
        with pytest.raises(Exception):
            resp.headers["b"] = "2"  # type: ignore[index]

    def test_accidental_exception_formatting(self):
        resp = HttpResponse(
            status_code=200,
            headers={SENSITIVE_HTTP_HEADER_SENTINEL: AUTHORIZATION_SENTINEL},
            body=RAW_HTTP_RESPONSE_BODY_SENTINEL,
        )
        formatted = f"debug object: {resp!r}"
        _sentinel_not_in(formatted, AUTHORIZATION_SENTINEL)
        _sentinel_not_in(formatted, RAW_HTTP_RESPONSE_BODY_SENTINEL)


# ===========================================================================
# E — Existing HttpRequest Guard (regression)
# ===========================================================================


class TestHttpRequestReprSafetyRegression:
    def test_sensitive_headers_not_in_repr(self):
        req = HttpRequest(
            method="POST",
            url="https://api.example.com/v1/chat",
            headers={"Authorization": AUTHORIZATION_SENTINEL},
            body=b"request body",
            timeout_seconds=30.0,
        )
        _sentinel_not_in(repr(req), AUTHORIZATION_SENTINEL)

    def test_body_not_in_repr(self):
        req = HttpRequest(
            method="GET",
            url="https://api.example.com/v1/models",
            headers={},
            body=b"SECRET_REQUEST_BODY_DO_NOT_LEAK",
            timeout_seconds=10.0,
        )
        _sentinel_not_in(repr(req), b"SECRET_REQUEST_BODY_DO_NOT_LEAK")

    def test_headers_still_accessible(self):
        req = HttpRequest(
            method="POST",
            url="https://api.example.com/",
            headers={"Authorization": AUTHORIZATION_SENTINEL},
            body=b"",
            timeout_seconds=5.0,
        )
        assert req.headers["Authorization"] == AUTHORIZATION_SENTINEL

    def test_body_still_accessible(self):
        body_content = b"accessible request body"
        req = HttpRequest(
            method="POST",
            url="https://api.example.com/",
            headers={},
            body=body_content,
            timeout_seconds=5.0,
        )
        assert req.body == body_content

    def test_harmless_fields_visible(self):
        req = HttpRequest(
            method="DELETE",
            url="https://api.example.com/resource/42",
            headers={"Authorization": AUTHORIZATION_SENTINEL},
            body=b"delete body",
            timeout_seconds=7.5,
        )
        req_repr = repr(req)
        assert "DELETE" in req_repr
        assert "https://api.example.com/resource/42" in req_repr

    def test_accidental_exception_formatting(self):
        req = HttpRequest(
            method="POST",
            url="https://api.example.com/",
            headers={"Authorization": AUTHORIZATION_SENTINEL},
            body=b"TOP_SECRET_REQUEST_BODY",
            timeout_seconds=5.0,
        )
        formatted = f"debug object: {req!r}"
        _sentinel_not_in(formatted, AUTHORIZATION_SENTINEL)
        _sentinel_not_in(formatted, b"TOP_SECRET_REQUEST_BODY")


# ===========================================================================
# G — Semantic Preservation
# ===========================================================================


class TestSemanticPreservation:
    """Representation redaction must not alter runtime semantics."""

    def test_modelmessage_direct_access(self):
        msg = ModelMessage(role="user", content=RAW_PROMPT_CONTENT_SENTINEL)
        assert msg.role == "user"
        assert msg.content == RAW_PROMPT_CONTENT_SENTINEL

    def test_modelrequest_deterministic_construction(self):
        inner = ModelMessage(role="system", content="sys prompt")
        outer = ModelMessage(role="user", content="user query")
        req = ModelRequest(purpose="diagnose", messages=(inner, outer), attempt_index=5)
        assert req.purpose == "diagnose"
        assert len(req.messages) == 2
        assert req.messages[0].role == "system"
        assert req.messages[1].role == "user"
        assert req.messages[0].content == "sys prompt"
        assert req.messages[1].content == "user query"
        assert req.attempt_index == 5

    def test_modelresponse_content_unchanged(self):
        original = '{"result": 42}'
        resp = ModelResponse(content=original)
        assert resp.content == original

    def test_httpresponse_body_bytes_identity(self):
        body = b"\x00\x01\x02\xff\xfe"
        resp = HttpResponse(status_code=200, headers={}, body=body)
        assert resp.body == body
        assert isinstance(resp.body, bytes)

    def test_httpresponse_status_code_accessible(self):
        resp = HttpResponse(status_code=404, headers={}, body=b"not found")
        assert resp.status_code == 404

    def test_modelmessage_inequality_different_content(self):
        a = ModelMessage(role="user", content="alpha")
        b = ModelMessage(role="user", content="beta")
        assert a != b

    def test_modelresponse_inequality_different_content(self):
        a = ModelResponse(content="alpha")
        b = ModelResponse(content="beta")
        assert a != b

    def test_httprequest_body_coerced_from_str(self):
        req = HttpRequest(
            method="POST",
            url="https://api.example.com/",
            headers={},
            body="string body",
            timeout_seconds=1.0,
        )
        assert isinstance(req.body, bytes)
        assert req.body == b"string body"

    def test_httprequest_body_coerced_from_bytearray(self):
        req = HttpRequest(
            method="POST",
            url="https://api.example.com/",
            headers={},
            body=bytearray(b"bytearray body"),
            timeout_seconds=1.0,
        )
        assert isinstance(req.body, bytes)
        assert req.body == b"bytearray body"


# ===========================================================================
# No custom __repr__ check
# ===========================================================================


class TestNoCustomRepr:
    """Verify no custom __repr__ was added to any of the hardened types.

    The dataclass decorator always generates a __repr__ method.  A
    *custom* __repr__ (one defined in our source files) would have
    co_filename pointing to our package directory.  The auto-generated
    repr's co_filename comes from stdlib ``reprlib.py``.
    """

    _PRISM_CHT_PATH = "/prismv4/prism_cht/"

    def test_modelmessage_no_custom_repr(self):
        assert "__repr__" in vars(ModelMessage)  # auto-generated
        assert self._PRISM_CHT_PATH not in ModelMessage.__repr__.__code__.co_filename

    def test_modelrequest_no_custom_repr(self):
        assert "__repr__" in vars(ModelRequest)
        assert self._PRISM_CHT_PATH not in ModelRequest.__repr__.__code__.co_filename

    def test_modelresponse_no_custom_repr(self):
        assert "__repr__" in vars(ModelResponse)
        assert self._PRISM_CHT_PATH not in ModelResponse.__repr__.__code__.co_filename

    def test_httpresponse_no_custom_repr(self):
        assert "__repr__" in vars(HttpResponse)
        assert self._PRISM_CHT_PATH not in HttpResponse.__repr__.__code__.co_filename

    def test_httprequest_no_custom_repr(self):
        assert "__repr__" in vars(HttpRequest)
        assert self._PRISM_CHT_PATH not in HttpRequest.__repr__.__code__.co_filename
