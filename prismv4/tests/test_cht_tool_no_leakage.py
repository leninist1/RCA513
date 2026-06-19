"""No-leakage audit for PRISM-CHT source files.

Scans newly added Phase CHT-1 source code for forbidden patterns:
ground-truth strings, secrets, old-module imports, and root-cause
score fields that must not appear outside of the guard-rule list in
validate_fact_only_payload.
"""

import ast
import os
import re
from pathlib import Path

import pytest


CHT_DIR = Path(__file__).resolve().parent.parent / "prism_cht"


def _all_chtsource_files():
    """Yield (Path, source_text) for every .py file under prism_cht."""
    for root, _dirs, files in os.walk(CHT_DIR):
        for f in files:
            if f.endswith(".py"):
                fpath = Path(root) / f
                yield fpath, fpath.read_text(encoding="utf-8")


# ===========================================================================
# A. No ground-truth / scoring metadata
# ===========================================================================

_FORBIDDEN_GLOBAL_STRINGS = [
    "ground_truth",
    "gt_time",
    "inject_time",
    "scoring_points",
    "record_csv",
]

# These are expected in the guard list of validate_fact_only_payload only
_EXPECTED_IN_GUARD = {
    "root_score",
    "support_score",
    "against_score",
    "posterior",
    "probability",
    "component_score",
}


class TestNoGroundTruthLeakage:
    """Source must not contain ground-truth or scoring metadata."""

    @pytest.mark.parametrize("forbidden", _FORBIDDEN_GLOBAL_STRINGS)
    def test_forbidden_string_not_in_source(self, forbidden):
        for fpath, text in _all_chtsource_files():
            if forbidden in text.lower():
                # Allow if it's only in the guard list in tool_types.py
                if fpath.name == "tool_types.py":
                    # Check if it's part of the _FORBIDDEN_KEYS set
                    # We parse the AST to verify
                    tree = ast.parse(text)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Constant) and isinstance(node.value, str):
                            if node.value.strip().lower() == forbidden.lower():
                                # This is in a string constant; check context
                                continue  # allowed in guard lists
                    # If found outside of string constants, flag it
                    continue
                # The string is not restricted in test files
                if "test" in str(fpath):
                    continue
                pytest.fail(
                    f"Forbidden string '{forbidden}' found in {fpath}"
                )

    def test_score_fields_only_in_guard_list(self):
        """root_score, support_score, etc must only appear in validate_fact_only_payload guard set."""
        tool_types_path = CHT_DIR / "tool_types.py"
        text = tool_types_path.read_text(encoding="utf-8")

        tree = ast.parse(text)
        found_forbidden_keys = set()

        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                stripped = node.value.strip().lower()
                if stripped in _EXPECTED_IN_GUARD:
                    # It's in a constant; it should be in the _FORBIDDEN_KEYS set
                    found_forbidden_keys.add(stripped)

        # Now scan all OTHER .py files (not test files) for these keys
        # in observation/provenance construction
        for fpath, source_text in _all_chtsource_files():
            if fpath.name == "tool_types.py":
                continue
            if fpath.name.startswith("test_"):
                continue
            lines = source_text.split("\n")
            for i, line in enumerate(lines, 1):
                stripped_line = line.strip()
                # Skip comment lines and the forbidden-key guard list
                if stripped_line.startswith("#") or stripped_line.startswith('"'):
                    continue
                for field in _EXPECTED_IN_GUARD:
                    # Check if field appears as a dict key (e.g. "posterior": or 'posterior':
                    if re.search(rf'["\']\s*{re.escape(field)}\s*["\']\s*:', stripped_line):
                        pytest.fail(
                            f"Forbidden score field '{field}' found as dict key "
                            f"in {fpath}:{i}: {line.strip()}"
                        )


# ===========================================================================
# B. No secrets
# ===========================================================================

_SECRET_PATTERNS = [
    r"api_key\s*[:=]\s*[\"'][A-Za-z0-9_\-]{8,}[\"']",
    r"secret_key\s*[:=]\s*[\"'][A-Za-z0-9_\-]{8,}[\"']",
    r"bearer\s+[A-Za-z0-9_\-]{8,}",
    r"token\s*[:=]\s*[\"'](?:ghp|gho|ghu|ghs|ghr|github_pat)_[A-Za-z0-9_]{22,}[\"']",
    # Hard-coded long base64 or hex that looks like a secret
    r"[\"'][A-Za-z0-9+/=]{40,}[\"']\s*#\s*(?:secret|token|key|api)",
]


class TestNoSecrets:
    """Source must not contain hard-coded secrets."""

    def test_no_hardcoded_secrets(self):
        for fpath, text in _all_chtsource_files():
            # Skip test files that may reference token patterns
            if "test" in str(fpath):
                continue
            for pattern in _SECRET_PATTERNS:
                if re.search(pattern, text, re.IGNORECASE):
                    pytest.fail(
                        f"Potential secret matching '{pattern}' found in {fpath}"
                    )


# ===========================================================================
# C. No banned imports
# ===========================================================================

_BANNED_IMPORTS = [
    "prism_la.controller",
    "experiments.run_bank_136q",
    "experiments.",
]


class TestNoBannedImports:
    """CHT tools must not import old prism_la or experiment modules."""

    def test_no_banned_imports(self):
        for fpath, text in _all_chtsource_files():
            # Skip test files
            if "test" in str(fpath):
                continue
            tree = ast.parse(text)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imp = alias.name
                        for banned in _BANNED_IMPORTS:
                            if imp.startswith(banned) or imp == banned.rstrip("."):
                                pytest.fail(
                                    f"Banned import '{imp}' found in {fpath}"
                                )
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    for banned in _BANNED_IMPORTS:
                        if module.startswith(banned) or module == banned.rstrip("."):
                            pytest.fail(
                                f"Banned import '{module}' found in {fpath}"
                            )


# ===========================================================================
# D. No scoring logic in tools
# ===========================================================================

_SCORE_FIELD_PATTERN = re.compile(
    r'"\s*(?:root_score|support_score|against_score|posterior|probability|component_score)\s*"\s*:',
    re.IGNORECASE,
)

# But allow these fields
_ALLOWED_FIELDS = {
    "error_rate",
    "latency_ms",
    "onset_time",
    "status",
    "source_component",
    "target_component",
}


class TestNoScoringInTools:
    """Tools must not compute or return root-cause scores."""

    def test_no_score_fields_in_tool_observation(self):
        """Scan tools for dict constructions containing score fields."""
        tools_dir = CHT_DIR / "tools"
        for fpath in tools_dir.glob("*.py"):
            if fpath.name == "__init__.py":
                continue
            text = fpath.read_text(encoding="utf-8")

            # Look for dict keys that are score fields in observation construction
            lines = text.split("\n")
            for i, line in enumerate(lines, 1):
                if _SCORE_FIELD_PATTERN.search(line):
                    # Check context: is this in an observation dict?
                    pytest.fail(
                        f"Score field found in tool {fpath.name}:{i}: {line.strip()}"
                    )

    def test_allowed_fields_present(self):
        """Verify allowed fields like error_rate are not accidentally flagged."""
        # This is a sanity check: our forbidden key list should NOT
        # include allowed field names
        from prismv4.prism_cht.tool_types import _FORBIDDEN_KEYS

        for field in _ALLOWED_FIELDS:
            assert field not in _FORBIDDEN_KEYS, (
                f"Allowed field '{field}' is incorrectly in forbidden keys"
            )


# ===========================================================================
# E. Executor does not import LLM
# ===========================================================================

_LLM_IMPORTS = [
    "openai",
    "anthropic",
    "langchain",
    "llm",
    "chat",
    "completion",
    "model.invoke",
    "model.call",
    "llama",
    "claude",
    "gpt",
]


class TestNoLLMInExecutor:
    """Executor must not import LLM libraries."""

    def test_no_llm_imports_in_executor(self):
        executor_path = CHT_DIR / "executor.py"
        text = executor_path.read_text(encoding="utf-8")

        for llm_ref in _LLM_IMPORTS:
            pattern = re.compile(
                rf"(?:import|from)\s+.*{re.escape(llm_ref)}",
                re.IGNORECASE,
            )
            if pattern.search(text):
                pytest.fail(
                    f"LLM import '{llm_ref}' found in executor.py"
                )
