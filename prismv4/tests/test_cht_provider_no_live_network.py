"""Scan provider adapter code for forbidden network access.

Ensures:
- All integration tests inject FakeTransport
- No real socket connections or DNS queries in tests
- UrllibHttpTransport tests use monkeypatch only
- No live smoke tests
"""

import ast
import os


# ===========================================================================
# Provider adapter production files
# ===========================================================================

PROVIDER_FILES = [
    "provider_types.py",
    "http_transport.py",
    "provider_config.py",
    "openai_compatible_client.py",
]

PROVIDER_TEST_FILES = [
    "test_cht_provider_config.py",
    "test_cht_http_transport.py",
    "test_cht_openai_compatible_client.py",
    "test_cht_provider_audit.py",
    "test_cht_provider_integration.py",
    "test_cht_provider_no_live_network.py",
    "test_cht_provider_no_leakage.py",
]


def _prod_source_paths():
    cht_dir = os.path.join(os.path.dirname(__file__), "..", "prism_cht")
    cht_dir = os.path.abspath(cht_dir)
    for fn in PROVIDER_FILES:
        yield os.path.join(cht_dir, fn)


def _test_source_paths():
    tests_dir = os.path.dirname(os.path.abspath(__file__))
    for fn in PROVIDER_TEST_FILES:
        yield os.path.join(tests_dir, fn)


def _read_file(path):
    with open(path, "r") as f:
        return f.read()


# ===========================================================================
# Tests
# ===========================================================================


class TestNoLiveNetwork:
    def test_integration_test_injects_fake_transport(self):
        """Verify integration test uses FakeTransport, not real HTTP."""
        tests_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(tests_dir, "test_cht_provider_integration.py")
        source = _read_file(path)
        assert "class FakeTransport" in source
        # Must not import UrllibHttpTransport in test
        assert "from prismv4.prism_cht.http_transport import" not in source or \
            "UrllibHttpTransport" not in source

    def test_openai_client_test_injects_fake_transport(self):
        tests_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(tests_dir, "test_cht_openai_compatible_client.py")
        source = _read_file(path)
        assert "class FakeTransport" in source

    def test_provider_audit_test_injects_fake_transport(self):
        tests_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(tests_dir, "test_cht_provider_audit.py")
        source = _read_file(path)
        assert "class FakeTransport" in source

    def test_no_live_url_open_in_tests(self):
        """No test should directly call urllib.request.urlopen without monkeypatch."""
        for path in _test_source_paths():
            source = _read_file(path)
            # The http_transport test should use monkeypatch
            if "test_cht_http_transport" in path:
                assert "monkeypatch" in source
            else:
                # Other tests should not mention urlopen
                assert "urlopen" not in source or "monkeypatch" in source

    def test_urllib_transport_tests_use_monkeypatch(self):
        tests_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(tests_dir, "test_cht_http_transport.py")
        source = _read_file(path)
        assert "monkeypatch" in source

    def test_no_live_smoke_test(self):
        """No test file should contain 'smoke' in a function name."""
        for path in _test_source_paths():
            source = _read_file(path)
            # Skip the test file itself
            if "test_cht_provider_no_live_network" in path:
                continue
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    if "smoke" in node.name.lower() and "smoke_test" not in path:
                        raise AssertionError(
                            f"Live smoke test '{node.name}' found in "
                            f"{os.path.basename(path)}"
                        )

    def test_no_socket_import(self):
        for path in _prod_source_paths():
            source = _read_file(path)
            assert "import socket" not in source
            assert "from socket" not in source

    def test_no_real_api_key_in_tests(self):
        """Tests should only use 'test-secret-not-real' or similar test keys."""
        import re
        for path in _test_source_paths():
            source = _read_file(path)
            for match in re.finditer(r'"sk-[a-zA-Z0-9]{30,}"', source):
                raise AssertionError(
                    f"Real-looking API key (sk-...) in {os.path.basename(path)}: "
                    f"{match.group()[:40]}..."
                )

    def test_integration_test_uses_only_fake_transport(self):
        tests_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(tests_dir, "test_cht_provider_integration.py")
        source = _read_file(path)
        # Must not import UrllibHttpTransport
        assert "UrllibHttpTransport" not in source
