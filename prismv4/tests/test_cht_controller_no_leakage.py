"""No-leakage audit for Phase CHT-2 source files.

Scans CHT-2 source code for forbidden patterns that would violate
the ground-truth-free, score-free, LLM-free invariants.
"""

import ast
import os
import re
from pathlib import Path

import pytest


CHT_DIR = Path(__file__).resolve().parent.parent / "prism_cht"


def _all_cht2_source_files():
    """Yield (Path, source_text) for CHT-2 source files only."""
    cht2_files = {
        "tournament_types.py",
        "assessment_gate.py",
        "lead_policy.py",
        "lead_controller.py",
        "demo_scenario.py",
    }
    for fname in cht2_files:
        fpath = CHT_DIR / fname
        if fpath.exists():
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


class TestNoGroundTruthLeakageCHT2:
    """CHT-2 source must not contain ground-truth or scoring metadata."""

    @pytest.mark.parametrize("forbidden", _FORBIDDEN_GLOBAL_STRINGS)
    def test_forbidden_string_not_in_cht2_source(self, forbidden):
        for fpath, text in _all_cht2_source_files():
            if forbidden in text.lower():
                pytest.fail(
                    f"Forbidden string '{forbidden}' found in {fpath}"
                )


# ===========================================================================
# B. No secrets
# ===========================================================================

_SECRET_PATTERNS = [
    r"api_key\s*[:=]\s*[\"'][A-Za-z0-9_\-]{8,}[\"']",
    r"secret_key\s*[:=]\s*[\"'][A-Za-z0-9_\-]{8,}[\"']",
    r"bearer\s+[A-Za-z0-9_\-]{8,}",
    r"token\s*[:=]\s*[\"'](?:ghp|gho|ghu|ghs|ghr|github_pat)_[A-Za-z0-9_]{22,}[\"']",
    r"[\"'][A-Za-z0-9+/=]{40,}[\"']\s*#\s*(?:secret|token|key|api)",
]


class TestNoSecretsCHT2:
    """CHT-2 source must not contain hard-coded secrets."""

    def test_no_hardcoded_secrets_in_cht2(self):
        for fpath, text in _all_cht2_source_files():
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


class TestNoBannedImportsCHT2:
    """CHT-2 must not import old prism_la or experiment modules."""

    def test_no_banned_imports_in_cht2(self):
        for fpath, text in _all_cht2_source_files():
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
# D. No scoring/probability fields in CHT-2 source
# ===========================================================================

_SCORE_FIELDS = {
    "score",
    "posterior",
    "probability",
    "confidence",
    "root_score",
    "component_score",
    "support_score",
    "against_score",
}


class TestNoScoringInCHT2:
    """CHT-2 source must not implement any scoring or probability logic."""

    def test_no_score_field_keys_in_cht2(self):
        for fpath, text in _all_cht2_source_files():
            lines = text.split("\n")
            for i, line in enumerate(lines, 1):
                stripped = line.strip()
                # Skip comment lines
                if stripped.startswith("#"):
                    continue
                for field in _SCORE_FIELDS:
                    if re.search(rf'["\']\s*{re.escape(field)}\s*["\']\s*:', stripped):
                        pytest.fail(
                            f"Score field '{field}' found as dict key "
                            f"in {fpath}:{i}: {line.strip()}"
                        )


# ===========================================================================
# E. No LLM imports in CHT-2
# ===========================================================================

_LLM_IMPORTS = [
    "openai",
    "anthropic",
    "langchain",
    "llm",
    "chat",
    "completion",
    "llama",
    "claude",
    "gpt",
]


class TestNoLLMInCHT2:
    """CHT-2 must not import LLM libraries."""

    def test_no_llm_imports_in_cht2(self):
        for fpath, text in _all_cht2_source_files():
            for llm_ref in _LLM_IMPORTS:
                pattern = re.compile(
                    rf"(?:import|from)\s+.*{re.escape(llm_ref)}",
                    re.IGNORECASE,
                )
                if pattern.search(text):
                    pytest.fail(
                        f"LLM import '{llm_ref}' found in {fpath}"
                    )


# ===========================================================================
# F. No forbidden status strings
# ===========================================================================

_FORBIDDEN_RESULT_STATUSES = ["final", "survived", "completed"]


class TestNoForbiddenStatusInCHT2:
    """tournament_types.py must enforce that LeadTournamentResult only allows
    'challenge_required', never 'final', 'survived', or 'completed'."""

    @pytest.mark.parametrize("bad_status", _FORBIDDEN_RESULT_STATUSES)
    def test_result_does_not_accept_bad_status(self, bad_status):
        from prismv4.prism_cht.tournament_types import (
            LeadNomination,
            LeadTournamentResult,
            TripletEvidenceCoverage,
        )
        nom = LeadNomination("H1", ("e1",), ("H2",), TripletEvidenceCoverage(("e1",), ("e1",), ("e1",)), "rationale")
        with pytest.raises(ValueError, match="must be 'challenge_required'"):
            LeadTournamentResult(
                status=bad_status,
                nominated_hypothesis_id="H1",
                rounds_completed=1,
                evidence_ids=("e1",),
                audit_steps=(),
                nomination=nom,
            )
