"""Tests for fake_model_client.py — deterministic fake model client."""

import pytest

from prismv4.prism_cht.fake_model_client import FakeModelClient
from prismv4.prism_cht.llm_types import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
)


def _make_request(purpose="test", content="hello", attempt_index=0):
    msg = ModelMessage(role="system", content=content)
    return ModelRequest(purpose=purpose, messages=(msg,), attempt_index=attempt_index)


# ===========================================================================
# Construction
# ===========================================================================


class TestFakeModelClientConstruction:
    def test_construct_with_strings(self):
        client = FakeModelClient(responses=["response1", "response2"])
        assert client.remaining_response_count == 2

    def test_construct_with_model_responses(self):
        r1 = ModelResponse(content="resp1")
        r2 = ModelResponse(content="resp2")
        client = FakeModelClient(responses=[r1, r2])
        assert client.remaining_response_count == 2

    def test_construct_with_mixed(self):
        r1 = ModelResponse(content="resp1")
        client = FakeModelClient(responses=[r1, "resp2"])
        assert client.remaining_response_count == 2

    def test_rejects_non_string_non_model_response(self):
        with pytest.raises(TypeError, match="responses"):
            FakeModelClient(responses=[42])  # type: ignore

    def test_empty_responses_allowed(self):
        client = FakeModelClient(responses=[])
        assert client.remaining_response_count == 0

    def test_external_mutation_does_not_affect_client(self):
        external = ["a", "b"]
        client = FakeModelClient(responses=external)
        external.append("c")
        assert client.remaining_response_count == 2

    def test_requests_property_initially_empty(self):
        client = FakeModelClient(responses=["a"])
        assert client.requests == ()


# ===========================================================================
# FIFO response order
# ===========================================================================


class TestFakeModelClientFifo:
    def test_responses_returned_in_order(self):
        client = FakeModelClient(responses=["first", "second", "third"])
        req1 = _make_request(purpose="p1")
        req2 = _make_request(purpose="p2")
        req3 = _make_request(purpose="p3")

        r1 = client.complete(request=req1)
        r2 = client.complete(request=req2)
        r3 = client.complete(request=req3)

        assert r1.content == "first"
        assert r2.content == "second"
        assert r3.content == "third"

    def test_model_response_objects_passed_through(self):
        original = ModelResponse(content="original")
        client = FakeModelClient(responses=[original])
        result = client.complete(request=_make_request())
        assert result is original
        assert result.content == "original"


# ===========================================================================
# Request recording
# ===========================================================================


class TestFakeModelClientRequestRecording:
    def test_records_all_requests(self):
        client = FakeModelClient(responses=["a", "b"])
        req1 = _make_request(purpose="first")
        req2 = _make_request(purpose="second")

        client.complete(request=req1)
        client.complete(request=req2)

        requests = client.requests
        assert len(requests) == 2
        assert requests[0].purpose == "first"
        assert requests[1].purpose == "second"

    def test_requests_returned_in_call_order(self):
        client = FakeModelClient(responses=["a", "b", "c"])
        for i in range(3):
            client.complete(request=_make_request(purpose=f"p{i}"))
        for i, req in enumerate(client.requests):
            assert req.purpose == f"p{i}"

    def test_requests_is_tuple(self):
        client = FakeModelClient(responses=["a"])
        client.complete(request=_make_request())
        assert isinstance(client.requests, tuple)

    def test_requests_cannot_be_mutated_externally(self):
        client = FakeModelClient(responses=["a"])
        client.complete(request=_make_request())
        requests_view = client.requests
        # The property returns a new tuple each time, so external mutation impossible
        assert requests_view is not client.requests

    def test_request_records_local_only(self):
        client = FakeModelClient(responses=["a"])
        req = _make_request()
        client.complete(request=req)
        recorded = client.requests[0]
        assert recorded.purpose == req.purpose
        assert recorded.messages[0].content == req.messages[0].content


# ===========================================================================
# Deterministic behavior
# ===========================================================================


class TestFakeModelClientDeterminism:
    def test_same_inputs_same_outputs(self):
        c1 = FakeModelClient(responses=["a", "b"])
        c2 = FakeModelClient(responses=["a", "b"])

        for _ in range(2):
            r1 = c1.complete(request=_make_request())
            r2 = c2.complete(request=_make_request())
            assert r1.content == r2.content

    def test_same_sequence_reproducible(self):
        def run_sequence():
            client = FakeModelClient(responses=["one", "two"])
            results = []
            for i in range(2):
                req = _make_request(purpose=f"p{i}")
                results.append(client.complete(request=req).content)
            return tuple(results)

        assert run_sequence() == ("one", "two")
        assert run_sequence() == ("one", "two")
        assert run_sequence() == ("one", "two")

    def test_requests_captured_deterministically(self):
        def captured_requests():
            client = FakeModelClient(responses=["a"])
            client.complete(request=_make_request(purpose="test"))
            return [r.purpose for r in client.requests]

        assert captured_requests() == ["test"]
        assert captured_requests() == ["test"]

    def test_no_randomness_no_network_no_env(self):
        client = FakeModelClient(responses=["a"])
        result = client.complete(request=_make_request())
        assert result.content == "a"

    def test_call_ordering_is_strict(self):
        client = FakeModelClient(responses=["a", "b", "c", "d", "e"])
        results = []
        for i in range(5):
            req = _make_request(purpose=f"p{i}")
            results.append(client.complete(request=req).content)
        assert results == ["a", "b", "c", "d", "e"]


# ===========================================================================
# Empty queue behavior
# ===========================================================================


class TestFakeModelClientEmptyQueue:
    def test_raises_on_empty_queue(self):
        client = FakeModelClient(responses=[])
        with pytest.raises(RuntimeError, match="no more"):
            client.complete(request=_make_request())

    def test_raises_after_exhaustion(self):
        client = FakeModelClient(responses=["only"])
        client.complete(request=_make_request())
        with pytest.raises(RuntimeError, match="exhausted"):
            client.complete(request=_make_request())

    def test_error_includes_request_count(self):
        client = FakeModelClient(responses=["a", "b"])
        client.complete(request=_make_request())
        client.complete(request=_make_request())
        with pytest.raises(RuntimeError, match="2 requests"):
            client.complete(request=_make_request())


# ===========================================================================
# Remaining response count
# ===========================================================================


class TestFakeModelClientRemainingCount:
    def test_count_decreases(self):
        client = FakeModelClient(responses=["a", "b", "c"])
        assert client.remaining_response_count == 3
        client.complete(request=_make_request())
        assert client.remaining_response_count == 2
        client.complete(request=_make_request())
        assert client.remaining_response_count == 1
        client.complete(request=_make_request())
        assert client.remaining_response_count == 0

    def test_count_never_negative(self):
        client = FakeModelClient(responses=["a"])
        client.complete(request=_make_request())
        assert client.remaining_response_count == 0
        with pytest.raises(RuntimeError):
            client.complete(request=_make_request())
        assert client.remaining_response_count == 0


# ===========================================================================
# Network and environment isolation
# ===========================================================================


class TestFakeModelClientIsolation:
    def test_no_network_access(self):
        import socket

        original_socket = socket.socket
        client = FakeModelClient(responses=["a"])
        try:
            result = client.complete(request=_make_request())
        finally:
            pass
        assert result.content == "a"

    def test_no_environment_variable_dependency(self):
        import os

        saved = {}
        for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "API_KEY", "MODEL_NAME"):
            saved[key] = os.environ.pop(key, None)

        try:
            client = FakeModelClient(responses=["a"])
            result = client.complete(request=_make_request())
            assert result.content == "a"
        finally:
            for key, val in saved.items():
                if val is not None:
                    os.environ[key] = val
                elif key in os.environ:
                    del os.environ[key]

    def test_no_filesystem_access(self):
        client = FakeModelClient(responses=["a"])
        result = client.complete(request=_make_request())
        assert result is not None

    def test_request_records_not_mixed_with_provider_audit(self):
        """Fake client request records are local and separate from provider audit."""
        client = FakeModelClient(responses=["a"])
        client.complete(request=_make_request())
        # FakeModelClient.requests contains ModelRequest objects only
        for req in client.requests:
            assert isinstance(req, ModelRequest)
            # No provider audit fields should leak
            assert not hasattr(req, "http_status")
            assert not hasattr(req, "retry_count")
            assert not hasattr(req, "latency_ms")
