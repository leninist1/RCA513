"""Tests for no-leakage guarantees in prismv4/prism_cht.

Verifies that the protocol code does not reference benchmark ground truth,
injection times, scoring points, or other evaluation-only fields.
"""

import ast
import os
import sys


# Fields that must NOT appear in prism_cht source code
FORBIDDEN_SOURCE_PATTERNS = [
    "ground_truth",
    "gt_time",
    "inject_time",
    "scoring_points",
    "record_csv",
]

# Modules / packages that prism_cht must NOT import
FORBIDDEN_IMPORTS = [
    "experiments.run_bank_136q",
    "prismv4.experiments.run_bank_136q",
]

# Legacy controller that prism_cht must NOT depend on
FORBIDDEN_CONTROLLER_IMPORTS = [
    "prismv4.prism_la.controller",
    "prism_la.controller",
]

# Benchmark result files that prism_cht must NOT read
FORBIDDEN_PATH_PATTERNS = [
    "benchmark",
    "results/prism_la",
]

# Security: must not contain hardcoded secrets
FORBIDDEN_SECRET_PATTERNS = [
    "sk-",           # OpenAI API key prefix
    "api_key",       # generic api key references
    "apiKey",
    "SECRET",
    "password",
    "token",
]


CHT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "prism_cht"
)


def _source_files():
    """Yield absolute paths to all .py files in prism_cht."""
    root = os.path.abspath(CHT_DIR)
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            if fn.endswith(".py"):
                yield os.path.join(dirpath, fn)


def _read_source(path):
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


class TestNoBenchmarkLeakage:
    """Source code must not reference benchmark ground truth fields."""

    def test_no_forbidden_patterns_in_source(self):
        for path in _source_files():
            content = _read_source(path)
            for pattern in FORBIDDEN_SOURCE_PATTERNS:
                assert pattern not in content, (
                    f"FORBIDDEN pattern '{pattern}' found in {path}"
                )

    def test_no_forbidden_imports(self):
        for path in _source_files():
            content = _read_source(path)
            try:
                tree = ast.parse(content, filename=path)
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        for forbidden in FORBIDDEN_IMPORTS:
                            assert alias.name != forbidden, (
                                f"FORBIDDEN import '{alias.name}' in {path}"
                            )
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        for forbidden in FORBIDDEN_IMPORTS:
                            assert node.module != forbidden, (
                                f"FORBIDDEN import from '{node.module}' in {path}"
                            )

    def test_no_legacy_controller_dependency(self):
        for path in _source_files():
            content = _read_source(path)
            try:
                tree = ast.parse(content, filename=path)
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        for forbidden in FORBIDDEN_CONTROLLER_IMPORTS:
                            assert alias.name != forbidden, (
                                f"FORBIDDEN controller import '{alias.name}' in {path}"
                            )
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        for forbidden in FORBIDDEN_CONTROLLER_IMPORTS:
                            assert node.module != forbidden, (
                                f"FORBIDDEN controller import from '{node.module}' in {path}"
                            )

    def test_no_benchmark_result_file_access(self):
        for path in _source_files():
            content = _read_source(path)
            for pattern in FORBIDDEN_PATH_PATTERNS:
                # Allow self-reference (results inside prism_cht) but not
                # external benchmark results.
                # Check if the pattern appears as a file path.
                assert pattern not in content, (
                    f"FORBIDDEN path pattern '{pattern}' in {path}"
                )


class TestNoSecrets:
    """Protocol code must not contain API keys or tokens."""

    def test_no_hardcoded_secrets(self):
        for path in _source_files():
            content = _read_source(path)
            contents_lower = content.lower()
            for pattern in FORBIDDEN_SECRET_PATTERNS:
                if pattern.lower() in contents_lower:
                    # Allow false positives: words like "password" in comments
                    # about validation rules are OK only if not actual secrets.
                    # We check for obvious assignment patterns.
                    lines = content.split("\n")
                    for lineno, line in enumerate(lines, 1):
                        if pattern.lower() in line.lower():
                            # Skip if it's just a docstring or comment about
                            # validation (no actual secret value).
                            stripped = line.strip()
                            if (
                                stripped.startswith("#")
                                or stripped.startswith('"""')
                                or stripped.startswith("'''")
                            ):
                                continue
                            # Check for assignment of a real-looking value
                            if "=" in stripped and (
                                "'" in stripped or '"' in stripped
                            ):
                                raise AssertionError(
                                    f"Suspicious secret pattern '{pattern}' "
                                    f"in {path}:{lineno}: {stripped}"
                                )

    def test_no_api_key_in_any_form(self):
        for path in _source_files():
            content = _read_source(path)
            # Scan for any string literals that look like API keys
            try:
                tree = ast.parse(content, filename=path)
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    val = node.value
                    if val.startswith("sk-") and len(val) > 20:
                        raise AssertionError(
                            f"API key found in {path}: ...{val[-8:]}"
                        )
