from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from prism_v3.leakage_guard import UntrustedArtifactError
from prism_v3.strict_main import validate_strict_argv


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _certify(path: Path, artifact_type: str) -> None:
    manifest = {
        "artifact_type": artifact_type,
        "feature_pipeline_version": "noise_lab_no_gt_v1",
        "generated_without_gt": True,
        "allowed_anchor_sources": ["public_query_window"],
        "fit_query_ids": [],
        "fit_dates": [],
        "fit_systems": [],
        "git_commit": "test",
        "artifact_sha256": _sha256(path),
        "metadata": {},
    }
    Path(f"{path}.manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_strict_main_accepts_runtime_only_mode() -> None:
    assert validate_strict_argv(["--systems", "Bank", "--prism-noise-lab"]) == {}


def test_strict_main_rejects_bare_v2_all() -> None:
    with pytest.raises(UntrustedArtifactError, match="convenience learned flags"):
        validate_strict_argv(["--systems", "Bank", "--v2-all"])


def test_strict_main_rejects_uncertified_score_csv(tmp_path: Path) -> None:
    path = tmp_path / "scores.csv"
    path.write_text("query_index,task_index,object_id,ltr_score\n0,task_1,a,1.0\n", encoding="utf-8")
    with pytest.raises(UntrustedArtifactError, match="missing artifact manifest"):
        validate_strict_argv(["--prism-noise-lab-scores", str(path)])


def test_strict_main_accepts_certified_score_csv(tmp_path: Path) -> None:
    path = tmp_path / "scores.csv"
    path.write_text("query_index,task_index,object_id,ltr_score\n0,task_1,a,1.0\n", encoding="utf-8")
    _certify(path, "no_leak_ltr_scores")
    validated = validate_strict_argv(["--prism-noise-lab-scores", str(path)])
    assert "--prism-noise-lab-scores" in validated


def test_learned_module_requires_explicit_certified_artifact() -> None:
    with pytest.raises(UntrustedArtifactError, match="requires a certified embedding"):
        validate_strict_argv(["--v2-learned-embeddings"])
