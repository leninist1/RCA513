"""Create or update a strict no-leakage manifest for an artifact.

Use this only after the artifact was produced by a reviewed label-free feature
pipeline and, for learned artifacts, fitted on a documented split.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import List

from .leakage_guard import ALLOWED_ANCHOR_SOURCES, file_sha256


def _split_csv(value: str) -> List[str]:
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def certify(
    path: str,
    *,
    artifact_type: str,
    feature_pipeline_version: str,
    anchor_sources: List[str],
    fit_query_ids: List[str],
    fit_dates: List[str],
    fit_systems: List[str],
    git_commit: str,
) -> Path:
    unknown = set(anchor_sources) - set(ALLOWED_ANCHOR_SOURCES)
    if unknown:
        raise ValueError("forbidden anchor sources: " + ", ".join(sorted(unknown)))
    artifact = Path(path)
    if not artifact.is_file():
        raise FileNotFoundError(path)
    manifest = {
        "artifact_type": str(artifact_type),
        "feature_pipeline_version": str(feature_pipeline_version),
        "generated_without_gt": True,
        "allowed_anchor_sources": list(anchor_sources),
        "fit_query_ids": list(fit_query_ids),
        "fit_dates": list(fit_dates),
        "fit_systems": list(fit_systems),
        "git_commit": str(git_commit),
        "artifact_sha256": file_sha256(str(artifact)),
        "metadata": {
            "certified_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "certification_note": "Review label-free feature lineage before use.",
        },
    }
    manifest_path = Path(f"{artifact}.manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=True, indent=2), encoding="utf-8")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a PRISM no-leakage artifact manifest")
    parser.add_argument("path")
    parser.add_argument("--artifact-type", required=True)
    parser.add_argument("--feature-pipeline-version", required=True)
    parser.add_argument("--anchor-sources", default="public_query_window")
    parser.add_argument("--fit-query-ids", default="")
    parser.add_argument("--fit-dates", default="")
    parser.add_argument("--fit-systems", default="")
    parser.add_argument("--git-commit", default="")
    args = parser.parse_args()
    output = certify(
        args.path,
        artifact_type=args.artifact_type,
        feature_pipeline_version=args.feature_pipeline_version,
        anchor_sources=_split_csv(args.anchor_sources),
        fit_query_ids=_split_csv(args.fit_query_ids),
        fit_dates=_split_csv(args.fit_dates),
        fit_systems=_split_csv(args.fit_systems),
        git_commit=args.git_commit,
    )
    print(output)


if __name__ == "__main__":
    main()
