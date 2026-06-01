"""Strict no-leakage contracts for PRISM v3.

This module turns the no-leakage policy from a calling convention into
executable checks shared by runners, adapters, and tests.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple
import json


class LeakageGuardError(RuntimeError):
    """Base exception for no-leakage contract violations."""


class UnsafeInferenceQueryError(LeakageGuardError):
    """Raised when inference receives answer-bearing fields."""


class UntrustedArtifactError(LeakageGuardError):
    """Raised when an external artifact cannot prove a no-leakage lineage."""


class AnchorSource(str, Enum):
    PUBLIC_QUERY_WINDOW = "public_query_window"
    TELEMETRY_UNSUPERVISED_ONSET = "telemetry_unsupervised_onset"
    FALLBACK_QUERY_WINDOW_START = "fallback_query_window_start"


ALLOWED_ANCHOR_SOURCES = frozenset(item.value for item in AnchorSource)
TRUSTED_SCORE_ARTIFACT_TYPES = frozenset(
    {"no_leak_ltr_scores", "no_leak_noiselab_scores", "no_leak_runtime_scores"}
)


@dataclass(frozen=True)
class InferenceAnchor:
    timestamp: float
    source: AnchorSource
    confidence: float = 1.0
    evidence_ids: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("anchor confidence must be within [0, 1]")


@dataclass(frozen=True)
class InferenceQuery:
    query_id: str
    task_index: str
    system: str
    sub_system: str
    instruction: str
    time_window: Tuple[str, str]
    telemetry_date: Optional[str] = None


@dataclass(frozen=True)
class ArtifactManifest:
    artifact_type: str
    feature_pipeline_version: str
    generated_without_gt: bool
    allowed_anchor_sources: Tuple[str, ...]
    fit_query_ids: Tuple[str, ...]
    fit_dates: Tuple[str, ...]
    fit_systems: Tuple[str, ...]
    git_commit: str
    artifact_sha256: str
    metadata: Mapping[str, Any]

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ArtifactManifest":
        return cls(
            artifact_type=str(payload.get("artifact_type", "")),
            feature_pipeline_version=str(payload.get("feature_pipeline_version", "")),
            generated_without_gt=bool(payload.get("generated_without_gt", False)),
            allowed_anchor_sources=tuple(str(x) for x in payload.get("allowed_anchor_sources", ())),
            fit_query_ids=tuple(str(x) for x in payload.get("fit_query_ids", ())),
            fit_dates=tuple(str(x) for x in payload.get("fit_dates", ())),
            fit_systems=tuple(str(x) for x in payload.get("fit_systems", ())),
            git_commit=str(payload.get("git_commit", "")),
            artifact_sha256=str(payload.get("artifact_sha256", "")),
            metadata=dict(payload.get("metadata", {}) or {}),
        )

    def to_debug(self) -> Dict[str, Any]:
        return {
            "artifact_type": self.artifact_type,
            "feature_pipeline_version": self.feature_pipeline_version,
            "generated_without_gt": self.generated_without_gt,
            "allowed_anchor_sources": list(self.allowed_anchor_sources),
            "fit_query_count": len(self.fit_query_ids),
            "fit_dates": list(self.fit_dates),
            "fit_systems": list(self.fit_systems),
            "git_commit": self.git_commit,
        }


def build_query_id(query: Any) -> str:
    explicit = getattr(query, "query_id", None)
    if explicit:
        return str(explicit)
    return ":".join(
        [
            str(getattr(query, "system", "")),
            str(getattr(query, "sub_system", "")),
            str(getattr(query, "task_index", "")),
            str(getattr(query, "query_index", "")),
        ]
    )


def assert_inference_query_safe(query: Any) -> None:
    """Reject any query carrying labels or scoring points."""
    violations = []
    if getattr(query, "ground_truth", None) is not None:
        violations.append("ground_truth")
    if list(getattr(query, "scoring_points", ()) or ()):
        violations.append("scoring_points")
    if violations:
        raise UnsafeInferenceQueryError(
            "inference query contains evaluation-only fields: " + ", ".join(violations)
        )


def inference_query_view(query: Any) -> InferenceQuery:
    assert_inference_query_safe(query)
    return InferenceQuery(
        query_id=build_query_id(query),
        task_index=str(getattr(query, "task_index", "")),
        system=str(getattr(query, "system", "")),
        sub_system=str(getattr(query, "sub_system", "")),
        instruction=str(getattr(query, "instruction", "")),
        time_window=tuple(getattr(query, "time_window", ("", ""))),
        telemetry_date=getattr(query, "telemetry_date", None),
    )


def artifact_manifest_path(path: str) -> Path:
    return Path(f"{path}.manifest.json")


def file_sha256(path: str) -> str:
    digest = sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_artifact_manifest(path: str) -> ArtifactManifest:
    manifest_path = artifact_manifest_path(path)
    if not manifest_path.is_file():
        raise UntrustedArtifactError(f"missing artifact manifest: {manifest_path}")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise UntrustedArtifactError(f"invalid artifact manifest: {manifest_path}: {exc}") from exc
    return ArtifactManifest.from_mapping(payload)


def validate_external_artifact(
    path: str,
    *,
    current_query_id: str = "",
    allowed_types: Iterable[str] = TRUSTED_SCORE_ARTIFACT_TYPES,
    verify_sha256: bool = True,
) -> ArtifactManifest:
    """Validate an external artifact before it enters inference.

    Legacy files without provenance are rejected rather than silently trusted.
    Labels may be used to fit a model, but certified feature generation must be
    label-free and the current query must be out-of-fold.
    """
    artifact = Path(path)
    if not artifact.is_file():
        raise UntrustedArtifactError(f"artifact not found: {path}")
    manifest = load_artifact_manifest(path)
    allowed = set(str(item) for item in allowed_types)
    if manifest.artifact_type not in allowed:
        raise UntrustedArtifactError(
            f"untrusted artifact type: {manifest.artifact_type!r}; allowed={sorted(allowed)}"
        )
    if not manifest.feature_pipeline_version:
        raise UntrustedArtifactError("missing feature_pipeline_version")
    if not manifest.generated_without_gt:
        raise UntrustedArtifactError("artifact features were not generated by a no-GT pipeline")
    unknown_sources = set(manifest.allowed_anchor_sources) - ALLOWED_ANCHOR_SOURCES
    if unknown_sources:
        raise UntrustedArtifactError(
            "artifact uses forbidden anchor sources: " + ", ".join(sorted(unknown_sources))
        )
    if current_query_id and current_query_id in set(manifest.fit_query_ids):
        raise UntrustedArtifactError(
            f"current query {current_query_id!r} appears in artifact fit_query_ids"
        )
    if verify_sha256:
        if not manifest.artifact_sha256:
            raise UntrustedArtifactError("missing artifact_sha256")
        actual = file_sha256(path)
        if actual.lower() != manifest.artifact_sha256.lower():
            raise UntrustedArtifactError(
                f"artifact sha256 mismatch: expected={manifest.artifact_sha256} actual={actual}"
            )
    return manifest
