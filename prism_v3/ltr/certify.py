"""Certify no-leakage LTR score artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from ..certify_artifact import certify


def main() -> None:
    parser = argparse.ArgumentParser(description="Certify a PRISM LTR artifact")
    parser.add_argument("path")
    parser.add_argument("--feature-pipeline-version", default="noise_lab_no_gt_v1")
    parser.add_argument("--anchor-sources", default="public_query_window,telemetry_unsupervised_onset")
    parser.add_argument("--fit-query-ids", default="")
    parser.add_argument("--fit-dates", default="")
    parser.add_argument("--fit-systems", default="")
    parser.add_argument("--git-commit", default="")
    args = parser.parse_args()
    manifest = certify(
        args.path,
        artifact_type="no_leak_ltr_scores",
        feature_pipeline_version=args.feature_pipeline_version,
        anchor_sources=[item.strip() for item in args.anchor_sources.split(",") if item.strip()],
        fit_query_ids=[item.strip() for item in args.fit_query_ids.split(",") if item.strip()],
        fit_dates=[item.strip() for item in args.fit_dates.split(",") if item.strip()],
        fit_systems=[item.strip() for item in args.fit_systems.split(",") if item.strip()],
        git_commit=args.git_commit,
    )
    print(manifest)


if __name__ == "__main__":
    main()
