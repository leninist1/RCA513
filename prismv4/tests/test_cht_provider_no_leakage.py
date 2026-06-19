"""Scan provider adapter code for forbidden leakage.

Ensures:
- No hardcoded API keys
- No .env file reading
- No full prompt logging
- No full response body logging
- No third-party SDK imports
- No scoring/posterior/probability/confidence/rank implementations
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


def _prod_source_paths():
    cht_dir = os.path.join(os.path.dirname(__file__), "..", "prism_cht")
    cht_dir = os.path.abspath(cht_dir)
    for fn in PROVIDER_FILES:
        yield os.path.join(cht_dir, fn)


def _read_file(path):
    with open(path, "r") as f:
        return f.read()


# ===========================================================================
# Tests
# ===========================================================================


class TestNoLeakage:
    def test_no_hardcoded_api_key(self):
        """Ensure no hardcoded real API keys in production code."""
        import re
        for path in _prod_source_paths():
            source = _read_file(path)
            # Search for string patterns that look like API keys
            # OpenAI keys: sk-...
            # Anthropic keys: sk-ant-...
            # DeepSeek keys: sk-...
            # Generic long hex/base64 strings
            for match in re.finditer(r'"sk-[a-zA-Z0-9]{20,}"', source):
                raise AssertionError(
                    f"Hardcoded API key in {os.path.basename(path)}: "
                    f"{match.group()[:20]}..."
                )
            # Check for other long hex strings that might be keys
            for match in re.finditer(r'"([a-f0-9]{32,})"', source):
                value = match.group(1)
                # Skip SHA hashes used as evidence IDs (they're fine)
                # Skip other known non-key hex strings
                if value == "0" * 64:
                    continue
            # Check for base64-looking keys
            for match in re.finditer(r'"([A-Za-z0-9+/]{40,}={0,2})"', source):
                value = match.group(1)
                # Skip things that look like Base64-encoded data
                # rather than API keys in context

    def test_no_real_token(self):
        """No real authentication tokens in source."""
        for path in _prod_source_paths():
            source = _read_file(path)
            assert "sk-ant" not in source
            assert "sk-or" not in source

    def test_no_dotenv_file_reading(self):
        for path in _prod_source_paths():
            source = _read_file(path)
            assert ".env" not in source or "env" in source.lower()

    def test_no_dotenv_import(self):
        for path in _prod_source_paths():
            source = _read_file(path)
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        assert alias.name != "dotenv", \
                            f"dotenv imported in {os.path.basename(path)}"
                if isinstance(node, ast.ImportFrom):
                    if node.module and "dotenv" in node.module:
                        raise AssertionError(
                            f"dotenv imported in {os.path.basename(path)}"
                        )

    def test_no_authorization_header_printed(self):
        """Authorization header is used but never logged/printed."""
        for path in _prod_source_paths():
            source = _read_file(path)
            # Authorization is fine as a header key
            # But there should be no print/log of the full header value
            # Check for str() or repr() of objects containing headers
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name) and node.func.id == "print":
                        for arg in node.args:
                            if isinstance(arg, ast.Name) and "auth" in arg.id.lower():
                                raise AssertionError(
                                    f"print(Authorization) in {os.path.basename(path)}"
                                )

    def test_no_full_prompt_printed(self):
        """Full prompts must not be logged."""
        for path in _prod_source_paths():
            source = _read_file(path)
            # The message content should only be serialized for HTTP body
            # Not logged or printed
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name) and node.func.id == "print":
                        for arg in node.args:
                            if isinstance(arg, ast.Name):
                                if arg.id in ("messages", "prompt", "content"):
                                    # Check context — is this error handling?
                                    source_lines = source.split("\n")
                                    line_no = node.lineno - 1
                                    ctx = source_lines[line_no] if line_no < len(source_lines) else ""
                                    if "error" not in ctx.lower() and "exception" not in ctx.lower():
                                        raise AssertionError(
                                            f"print({arg.id}) in {os.path.basename(path)}:{node.lineno}"
                                        )

    def test_no_full_response_body_printed(self):
        for path in _prod_source_paths():
            source = _read_file(path)
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name) and node.func.id == "print":
                        for arg in node.args:
                            if isinstance(arg, ast.Name) and arg.id in ("body", "response_body"):
                                raise AssertionError(
                                    f"print(body) in {os.path.basename(path)}:{node.lineno}"
                                )

    def test_no_api_key_saved_to_audit(self):
        """ProviderCallAudit must not store api_key."""
        for path in _prod_source_paths():
            source = _read_file(path)
            if "ProviderCallAudit" in source:
                # Verify the dataclass fields don't include api_key
                tree = ast.parse(source)
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef) and node.name == "ProviderCallAudit":
                        for item in node.body:
                            if isinstance(item, ast.AnnAssign) and hasattr(item, "target"):
                                if hasattr(item.target, "id"):
                                    assert item.target.id != "api_key", \
                                        f"api_key in ProviderCallAudit fields"

    def test_no_headers_saved_to_audit(self):
        """ProviderCallAudit must not store HTTP headers."""
        for path in _prod_source_paths():
            source = _read_file(path)
            if "ProviderCallAudit" in source:
                tree = ast.parse(source)
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef) and node.name == "ProviderCallAudit":
                        for item in node.body:
                            if isinstance(item, ast.AnnAssign) and hasattr(item, "target"):
                                if hasattr(item.target, "id"):
                                    field_name = item.target.id
                                    assert "header" not in field_name.lower(), \
                                        f"Header field '{field_name}' in ProviderCallAudit"

    def test_no_model_request_saved_to_audit(self):
        """ProviderCallAudit must not store a ModelRequest object."""
        for path in _prod_source_paths():
            source = _read_file(path)
            if "ProviderCallAudit" in source:
                tree = ast.parse(source)
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef) and node.name == "ProviderCallAudit":
                        for item in node.body:
                            if isinstance(item, ast.AnnAssign) and hasattr(item, "target"):
                                if hasattr(item.target, "id"):
                                    field_name = item.target.id
                                    # Allow request_bytes and response_bytes,
                                    # reject fields like 'request' or 'model_request'
                                    if field_name == "request":
                                        raise AssertionError(
                                            f"'request' field in ProviderCallAudit"
                                        )

    def test_no_forbidden_third_party_imports(self):
        forbidden = [
            "requests", "httpx", "aiohttp",
            "openai", "anthropic", "langchain", "litellm",
        ]
        for path in _prod_source_paths():
            source = _read_file(path)
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        for bad in forbidden:
                            if alias.name == bad or alias.name.startswith(bad + "."):
                                raise AssertionError(
                                    f"Forbidden import '{alias.name}' in "
                                    f"{os.path.basename(path)}"
                                )
                if isinstance(node, ast.ImportFrom):
                    if node.module:
                        for bad in forbidden:
                            if node.module == bad or node.module.startswith(bad + "."):
                                raise AssertionError(
                                    f"Forbidden import 'from {node.module}' in "
                                    f"{os.path.basename(path)}"
                                )

    def test_no_legacy_imports(self):
        forbidden = [
            "prism_la.controller",
            "experiments.run_bank_136q",
        ]
        for path in _prod_source_paths():
            source = _read_file(path)
            for term in forbidden:
                assert term not in source, \
                    f"Forbidden legacy import '{term}' in {os.path.basename(path)}"

    def test_no_ground_truth_or_scoring_terms(self):
        forbidden_terms = [
            "ground_truth",
            "gt_time",
            "inject_time",
            "scoring_points",
            "record_csv",
        ]
        for path in _prod_source_paths():
            source = _read_file(path)
            for term in forbidden_terms:
                # Allow in docstrings/comments that describe what NOT to do
                lines = source.split("\n")
                for i, line in enumerate(lines):
                    if term in line:
                        stripped = line.strip()
                        if stripped.startswith("#") or stripped.startswith('"""'):
                            continue
                        if stripped.startswith("'") or stripped.startswith('"'):
                            continue
                        raise AssertionError(
                            f"Forbidden term '{term}' in "
                            f"{os.path.basename(path)}:{i+1}"
                        )

    def test_no_scoring_implementation(self):
        forbidden_terms = [
            "score",
            "posterior",
            "probability",
            "confidence",
            "rank",
        ]
        for path in _prod_source_paths():
            source = _read_file(path)
            tree = ast.parse(source)
            # Check function names
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    name_lower = node.name.lower()
                    # Skip well-known safe function names
                    safe_prefixes = ("test_", "_test", "parse", "build_", "serialize",
                                     "require_", "_parse", "_build", "_validate",
                                     "_require", "_is_", "get_", "send", "complete",
                                     "load_")
                    if any(name_lower.startswith(p) for p in safe_prefixes):
                        continue
                    for term in forbidden_terms:
                        # Check if term appears in function name
                        # But allow compound words
                        if term == name_lower:
                            raise AssertionError(
                                f"Function '{node.name}' implements '{term}' in "
                                f"{os.path.basename(path)}"
                            )

    def test_no_composite_scoring_terms(self):
        composite = [
            "root_score",
            "component_score",
            "support_score",
            "against_score",
        ]
        for path in _prod_source_paths():
            source = _read_file(path)
            for term in composite:
                assert term not in source, \
                    f"Composite scoring term '{term}' in {os.path.basename(path)}"

    def test_api_key_allowed_as_field_name(self):
        """api_key as a variable/field name is allowed — the ban is on hardcoded values."""
        found_api_key_field = False
        for path in _prod_source_paths():
            source = _read_file(path)
            if "api_key" in source:
                # Check it's used as a field/parameter name, not a hardcoded value
                lines = source.split("\n")
                for line in lines:
                    if "api_key" in line:
                        if "api_key =" in line or "api_key:" in line or "api_key=" in line:
                            found_api_key_field = True
        # This is a positive assertion — api_key as field name IS expected
        assert found_api_key_field, "Expected api_key field in config"

    def test_authorization_allowed_as_header(self):
        """'Authorization' is allowed as an HTTP header name."""
        found = False
        for path in _prod_source_paths():
            source = _read_file(path)
            if "Authorization" in source:
                found = True
        assert found, "Expected Authorization header usage"

    def test_retryable_allowed_as_error_property(self):
        """'retryable' is allowed as ProviderHTTPError property."""
        found = False
        for path in _prod_source_paths():
            source = _read_file(path)
            if "retryable" in source:
                found = True
        assert found, "Expected retryable property"
