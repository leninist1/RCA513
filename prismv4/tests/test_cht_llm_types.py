"""Tests for llm_types.py — structured types and contracts."""

import pytest

from prismv4.prism_cht.llm_types import (
    ModelClient,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    PromptBudgetExceededError,
    StructuredOutputError,
)


# ===========================================================================
# ModelMessage
# ===========================================================================


class TestModelMessage:
    def test_valid_system_message(self):
        msg = ModelMessage(role="system", content="You are a helpful assistant.")
        assert msg.role == "system"
        assert msg.content == "You are a helpful assistant."

    def test_valid_user_message(self):
        msg = ModelMessage(role="user", content="What is the answer?")
        assert msg.role == "user"

    def test_rejects_invalid_role(self):
        with pytest.raises(ValueError, match="role"):
            ModelMessage(role="assistant", content="Hello")

    def test_rejects_empty_role(self):
        with pytest.raises(ValueError, match="role"):
            ModelMessage(role="", content="Hello")

    def test_rejects_empty_content(self):
        with pytest.raises(ValueError, match="content"):
            ModelMessage(role="system", content="")

    def test_rejects_whitespace_only_content(self):
        with pytest.raises(ValueError, match="content"):
            ModelMessage(role="system", content="   \t\n  ")

    def test_equality(self):
        a = ModelMessage(role="system", content="hello")
        b = ModelMessage(role="system", content="hello")
        assert a == b

    def test_inequality_different_role(self):
        a = ModelMessage(role="system", content="hello")
        b = ModelMessage(role="user", content="hello")
        assert a != b

    def test_frozen_prevents_mutation(self):
        msg = ModelMessage(role="system", content="hello")
        with pytest.raises(Exception):
            msg.role = "user"  # type: ignore

    def test_content_excluded_from_repr(self):
        """ModelMessage.content is excluded from repr via field(repr=False)."""
        msg = ModelMessage(role="system", content="visible text")
        assert "visible text" not in repr(msg)
        assert msg.content == "visible text"

    def test_role_enum_restriction(self):
        """Only 'system' and 'user' are valid roles."""
        for bad_role in ("assistant", "tool", "function", "model"):
            with pytest.raises(ValueError, match="role"):
                ModelMessage(role=bad_role, content="x")


# ===========================================================================
# ModelRequest
# ===========================================================================


class TestModelRequest:
    def test_valid_construction(self):
        msg = ModelMessage(role="system", content="Hello")
        req = ModelRequest(purpose="test", messages=(msg,), attempt_index=0)
        assert req.purpose == "test"
        assert req.messages == (msg,)
        assert req.attempt_index == 0

    def test_rejects_empty_purpose(self):
        msg = ModelMessage(role="system", content="x")
        with pytest.raises(ValueError, match="purpose"):
            ModelRequest(purpose="", messages=(msg,), attempt_index=0)

    def test_rejects_whitespace_only_purpose(self):
        msg = ModelMessage(role="system", content="x")
        with pytest.raises(ValueError, match="purpose"):
            ModelRequest(purpose="  \n ", messages=(msg,), attempt_index=0)

    def test_rejects_empty_messages(self):
        with pytest.raises(ValueError, match="messages"):
            ModelRequest(purpose="test", messages=(), attempt_index=0)

    def test_rejects_negative_attempt_index(self):
        msg = ModelMessage(role="system", content="x")
        with pytest.raises(ValueError, match="attempt_index"):
            ModelRequest(purpose="test", messages=(msg,), attempt_index=-1)

    def test_accepts_high_attempt_index(self):
        msg = ModelMessage(role="system", content="x")
        req = ModelRequest(purpose="test", messages=(msg,), attempt_index=99)
        assert req.attempt_index == 99

    def test_frozen(self):
        msg = ModelMessage(role="system", content="x")
        req = ModelRequest(purpose="test", messages=(msg,), attempt_index=0)
        with pytest.raises(Exception):
            req.purpose = "other"  # type: ignore

    def test_multiple_messages(self):
        a = ModelMessage(role="system", content="sys")
        b = ModelMessage(role="user", content="user")
        req = ModelRequest(purpose="multi", messages=(a, b), attempt_index=1)
        assert len(req.messages) == 2
        assert req.messages[0].content == "sys"
        assert req.messages[1].content == "user"

    def test_purpose_preserved_in_repr(self):
        """ModelRequest is a plain dataclass; repr includes field values.
        This is a known limitation (deferred: no secret redaction in repr)."""
        msg = ModelMessage(role="system", content="hello")
        req = ModelRequest(purpose="investigate", messages=(msg,), attempt_index=0)
        assert "investigate" in repr(req)

    def test_equality(self):
        msg = ModelMessage(role="system", content="x")
        a = ModelRequest(purpose="test", messages=(msg,), attempt_index=0)
        b = ModelRequest(purpose="test", messages=(msg,), attempt_index=0)
        assert a == b


# ===========================================================================
# ModelResponse
# ===========================================================================


class TestModelResponse:
    def test_valid_construction(self):
        resp = ModelResponse(content='{"result": "ok"}')
        assert resp.content == '{"result": "ok"}'

    def test_rejects_empty_content(self):
        with pytest.raises(ValueError, match="content"):
            ModelResponse(content="")

    def test_rejects_whitespace_only_content(self):
        with pytest.raises(ValueError, match="content"):
            ModelResponse(content="   \n  ")

    def test_frozen(self):
        resp = ModelResponse(content="hello")
        with pytest.raises(Exception):
            resp.content = "world"  # type: ignore

    def test_equality(self):
        a = ModelResponse(content="abc")
        b = ModelResponse(content="abc")
        assert a == b

    def test_content_excluded_from_repr(self):
        """ModelResponse.content is excluded from repr via field(repr=False)."""
        resp = ModelResponse(content='{"key": "value"}')
        assert "key" not in repr(resp)
        assert resp.content == '{"key": "value"}'


# ===========================================================================
# Error types
# ===========================================================================


class TestStructuredOutputError:
    def test_is_value_error(self):
        err = StructuredOutputError("bad JSON")
        assert isinstance(err, ValueError)

    def test_message_preserved(self):
        err = StructuredOutputError("parse failed: invalid token")
        assert "invalid token" in str(err)


class TestPromptBudgetExceededError:
    def test_is_value_error(self):
        err = PromptBudgetExceededError("exceeded budget")
        assert isinstance(err, ValueError)

    def test_message_preserved(self):
        err = PromptBudgetExceededError("budget: 100001 > 100000")
        assert "100001" in str(err)


# ===========================================================================
# ModelClient protocol
# ===========================================================================


class TestModelClientProtocol:
    def test_protocol_is_defined(self):
        assert ModelClient is not None

    def test_fake_model_client_satisfies_protocol(self):
        from prismv4.prism_cht.fake_model_client import FakeModelClient

        client = FakeModelClient(responses=["hello"])
        assert hasattr(client, "complete")
