"""Source scan: CHT-4 final verifier code must not leak ground truth, secrets, scoring, LLM calls, or prism_la imports."""

import ast
import os


def _collect_source_files():
    cht_dir = os.path.join(
        os.path.dirname(__file__), "..", "prism_cht"
    )
    target_files = {
        "final_types.py",
        "final_verifier.py",
    }
    for f in target_files:
        path = os.path.join(cht_dir, f)
        if os.path.isfile(path):
            yield f, path


def _read_file(path):
    with open(path, "r") as fh:
        return fh.read()


# ===========================================================================
# 1. No ground truth / benchmark leakage
# ===========================================================================


FORBIDDEN_WORDS = [
    "ground_truth",
    "gt_time",
    "inject_time",
    "scoring_points",
    "record_csv",
]


class TestNoBenchmarkLeakage:
    def test_no_ground_truth_terms(self):
        for name, path in _collect_source_files():
            content = _read_file(path)
            lowered = content.lower()
            for word in FORBIDDEN_WORDS:
                assert word not in lowered, (
                    f"File {name} contains forbidden term '{word}'"
                )


# ===========================================================================
# 2. No secrets or hardcoded tokens
# ===========================================================================


FORBIDDEN_SECRET_PATTERNS = [
    "api_key",
    "secret_key",
    "bearer",
]


class TestNoSecrets:
    def test_no_secret_patterns(self):
        for name, path in _collect_source_files():
            content = _read_file(path)
            lowered = content.lower()
            for pattern in FORBIDDEN_SECRET_PATTERNS:
                assert pattern not in lowered, (
                    f"File {name} contains secret pattern '{pattern}'"
                )


# ===========================================================================
# 3. No prism_la or experiments imports
# ===========================================================================


FORBIDDEN_IMPORTS = [
    "prism_la.controller",
    "experiments.run_bank_136q",
]


class TestNoBannedImports:
    def test_no_banned_imports(self):
        for name, path in _collect_source_files():
            content = _read_file(path)
            lowered = content.lower()
            for imp in FORBIDDEN_IMPORTS:
                assert imp.lower() not in lowered, (
                    f"File {name} contains banned import '{imp}'"
                )


# ===========================================================================
# 4. No scoring / probability / posterior / confidence / rank
# ===========================================================================


FORBIDDEN_SCORING_ASSIGNMENTS = [
    "score =",
    "posterior =",
    "probability =",
    "confidence =",
    "rank =",
    "root_score =",
    "component_score =",
    "support_score =",
    "against_score =",
    "def score(",
    "def posterior(",
    "def probability(",
    "def confidence(",
    "def rank(",
    "def root_score(",
    "def component_score(",
    "def support_score(",
    "def against_score(",
]


class TestNoScoring:
    def test_no_scoring_variable_or_function_definitions(self):
        for name, path in _collect_source_files():
            content = _read_file(path)
            lowered = content.lower()
            for pattern in FORBIDDEN_SCORING_ASSIGNMENTS:
                assert pattern not in lowered, (
                    f"File {name} contains scoring pattern '{pattern}'"
                )

    def test_no_score_computation(self):
        for name, path in _collect_source_files():
            content = _read_file(path)
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.Name):
                    if node.id.lower() in {
                        "score", "posterior", "probability",
                        "confidence", "rank",
                        "root_score", "component_score",
                        "support_score", "against_score",
                    }:
                        raise AssertionError(
                            f"File {name} uses variable name '{node.id}'"
                        )


# ===========================================================================
# 5. No LLM / network / API call infrastructure
# ===========================================================================


LLM_PATTERNS = [
    "openai",
    "anthropic",
    "api_call",
    "http_client",
    "requests.",
    "urllib",
    "httpx",
    "aiohttp",
]


class TestNoLLM:
    def test_no_llm_infrastructure(self):
        for name, path in _collect_source_files():
            content = _read_file(path)
            lowered = content.lower()
            for pattern in LLM_PATTERNS:
                assert pattern not in lowered, (
                    f"File {name} contains LLM/network pattern '{pattern}'"
                )
