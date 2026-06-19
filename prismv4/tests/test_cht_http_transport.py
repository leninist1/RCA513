"""Tests for http_transport.py — HttpRequest, HttpResponse, UrllibHttpTransport."""

import json

import pytest

from prismv4.prism_cht.http_transport import (
    HttpRequest,
    HttpResponse,
    UrllibHttpTransport,
)
from prismv4.prism_cht.provider_types import (
    ProviderResponseError,
    ProviderTransportError,
)


# ===========================================================================
# HttpRequest tests
# ===========================================================================


class TestHttpRequest:
    def test_defensive_copy_of_headers(self):
        headers = {"X-Foo": "bar"}
        req = HttpRequest(
            method="POST",
            url="https://example.com",
            headers=headers,
            body=b"test",
            timeout_seconds=10.0,
        )
        headers["X-Foo"] = "modified"
        assert req.headers["X-Foo"] == "bar"

    def test_body_saved_as_bytes(self):
        req = HttpRequest(
            method="POST",
            url="https://example.com",
            headers={},
            body="string-body",
            timeout_seconds=10.0,
        )
        assert isinstance(req.body, bytes)
        assert req.body == b"string-body"

    def test_repr_excludes_headers(self):
        req = HttpRequest(
            method="POST",
            url="https://example.com",
            headers={"Authorization": "Bearer secret"},
            body=b"data",
            timeout_seconds=10.0,
        )
        r = repr(req)
        assert "headers" not in r
        assert "Authorization" not in r
        assert "secret" not in r

    def test_repr_excludes_body(self):
        req = HttpRequest(
            method="POST",
            url="https://example.com",
            headers={},
            body=b"sensitive-data",
            timeout_seconds=10.0,
        )
        r = repr(req)
        assert "body" not in r
        assert "sensitive-data" not in r

    def test_repr_excludes_test_api_key(self):
        req = HttpRequest(
            method="POST",
            url="https://example.com",
            headers={"Authorization": "Bearer test-secret-not-real"},
            body=b"data",
            timeout_seconds=10.0,
        )
        r = repr(req)
        assert "test-secret-not-real" not in r

    def test_reject_empty_method(self):
        with pytest.raises(ValueError, match="method"):
            HttpRequest(
                method="",
                url="https://example.com",
                headers={},
                body=b"test",
                timeout_seconds=10.0,
            )

    def test_reject_empty_url(self):
        with pytest.raises(ValueError, match="url"):
            HttpRequest(
                method="POST",
                url="",
                headers={},
                body=b"test",
                timeout_seconds=10.0,
            )

    def test_reject_zero_timeout(self):
        with pytest.raises(ValueError, match="timeout"):
            HttpRequest(
                method="POST",
                url="https://example.com",
                headers={},
                body=b"test",
                timeout_seconds=0,
            )


# ===========================================================================
# HttpResponse tests
# ===========================================================================


class TestHttpResponse:
    def test_defensive_copy_of_headers(self):
        headers = {"Content-Type": "application/json"}
        resp = HttpResponse(
            status_code=200,
            headers=headers,
            body=b"ok",
        )
        headers["Content-Type"] = "modified"
        assert resp.headers["Content-Type"] == "application/json"

    def test_body_saved_as_bytes(self):
        resp = HttpResponse(
            status_code=200,
            headers={},
            body="string-ok",
        )
        assert isinstance(resp.body, bytes)
        assert resp.body == b"string-ok"

    def test_reject_non_positive_status(self):
        with pytest.raises(ValueError, match="status_code"):
            HttpResponse(status_code=0, headers={}, body=b"")


# ===========================================================================
# UrllibHttpTransport tests (monkeypatch only)
# ===========================================================================


class FakeUrlopenResponse:
    """A fake file-like object returned by monkeypatched urlopen."""

    def __init__(self, status_code, body_bytes, headers):
        self._status = status_code
        self._body = body_bytes
        self._headers = headers

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def getcode(self):
        return self._status

    def getheaders(self):
        return list(self._headers.items())

    def read(self, size=-1):
        if size < 0:
            return self._body
        result = self._body[:size]
        self._body = self._body[size:]
        return result


class TestUrllibHttpTransport:
    def test_success_returns_http_response(self, monkeypatch):
        response_body = json.dumps({
            "choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}],
        }).encode("utf-8")

        def fake_urlopen(req, timeout=None):
            return FakeUrlopenResponse(
                status_code=200,
                body_bytes=response_body,
                headers={"Content-Type": "application/json"},
            )

        monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

        transport = UrllibHttpTransport()
        http_req = HttpRequest(
            method="POST",
            url="https://api.example.com/chat/completions",
            headers={"Authorization": "Bearer test-key"},
            body=b'{"test":true}',
            timeout_seconds=30.0,
        )
        resp = transport.send(request=http_req, max_response_bytes=1000000)
        assert resp.status_code == 200
        assert json.loads(resp.body.decode()) == {
            "choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}],
        }

    def test_http_error_returns_http_response(self, monkeypatch):
        import urllib.error
        from io import BytesIO

        error_body = b'{"error":"bad request"}'

        def fake_urlopen(req, timeout=None):
            raise urllib.error.HTTPError(
                url="https://api.example.com",
                code=400,
                msg="Bad Request",
                hdrs={"Content-Type": "application/json"},
                fp=BytesIO(error_body),
            )

        monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

        transport = UrllibHttpTransport()
        http_req = HttpRequest(
            method="POST",
            url="https://api.example.com/chat/completions",
            headers={},
            body=b"{}",
            timeout_seconds=30.0,
        )
        resp = transport.send(request=http_req, max_response_bytes=1000000)
        assert resp.status_code == 400
        assert json.loads(resp.body.decode()) == {"error": "bad request"}

    def test_url_error_converts_to_transport_error(self, monkeypatch):
        import urllib.error

        def fake_urlopen(req, timeout=None):
            raise urllib.error.URLError("connection refused")

        monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

        transport = UrllibHttpTransport()
        http_req = HttpRequest(
            method="POST",
            url="https://api.example.com/chat/completions",
            headers={"Authorization": "Bearer test-key"},
            body=b"{}",
            timeout_seconds=30.0,
        )
        with pytest.raises(ProviderTransportError, match="transport error"):
            transport.send(request=http_req, max_response_bytes=1000000)

    def test_response_exceeds_max_bytes_rejected(self, monkeypatch):
        response_body = b"x" * 2000

        def fake_urlopen(req, timeout=None):
            return FakeUrlopenResponse(
                status_code=200,
                body_bytes=response_body,
                headers={},
            )

        monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

        transport = UrllibHttpTransport()
        http_req = HttpRequest(
            method="POST",
            url="https://api.example.com/chat/completions",
            headers={},
            body=b"{}",
            timeout_seconds=30.0,
        )
        with pytest.raises(ProviderResponseError, match="max_response_bytes"):
            transport.send(request=http_req, max_response_bytes=1000)

    def test_error_does_not_contain_authorization_header(self, monkeypatch):
        import urllib.error

        def fake_urlopen(req, timeout=None):
            raise urllib.error.URLError("connection refused")

        monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

        transport = UrllibHttpTransport()
        http_req = HttpRequest(
            method="POST",
            url="https://api.example.com",
            headers={"Authorization": "Bearer test-secret-not-real"},
            body=b"{}",
            timeout_seconds=30.0,
        )
        try:
            transport.send(request=http_req, max_response_bytes=1000000)
        except ProviderTransportError as e:
            msg = str(e)
            assert "Bearer" not in msg
            assert "test-secret-not-real" not in msg

    def test_error_does_not_contain_api_key(self, monkeypatch):
        import urllib.error

        def fake_urlopen(req, timeout=None):
            raise urllib.error.URLError("connection refused")

        monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

        transport = UrllibHttpTransport()
        http_req = HttpRequest(
            method="POST",
            url="https://api.example.com",
            headers={"Authorization": "Bearer my-real-key-123"},
            body=b"{}",
            timeout_seconds=30.0,
        )
        try:
            transport.send(request=http_req, max_response_bytes=1000000)
        except ProviderTransportError as e:
            msg = str(e)
            assert "my-real-key-123" not in msg

    def test_error_does_not_contain_full_request_body(self, monkeypatch):
        import urllib.error

        def fake_urlopen(req, timeout=None):
            raise urllib.error.URLError("connection refused")

        monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

        transport = UrllibHttpTransport()
        http_req = HttpRequest(
            method="POST",
            url="https://api.example.com",
            headers={},
            body=b"sensitive-prompt-data-here",
            timeout_seconds=30.0,
        )
        try:
            transport.send(request=http_req, max_response_bytes=1000000)
        except ProviderTransportError as e:
            msg = str(e)
            assert "sensitive-prompt-data-here" not in msg

    def test_no_retry_implementation(self):
        src = inspect.getsource(UrllibHttpTransport.send)
        assert "retry" not in src.lower()
        assert "while True" not in src


import inspect
