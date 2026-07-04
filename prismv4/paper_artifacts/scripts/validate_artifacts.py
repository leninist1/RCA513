#!/usr/bin/env python3
"""Validate revised CAPE-RCA paper artifacts."""

from __future__ import annotations

import csv
from pathlib import Path

from paper_utils import ARTIFACT_ROOT, FIGURE_DIR, TABLE_DIR, REPO_ROOT


REPORT_DIR = ARTIFACT_ROOT / "reports"

REQUIRED_FILES = [
    TABLE_DIR / "rcaeval_re2_re3_summary.csv",
    TABLE_DIR / "case_level_predictions_rcaeval.csv",
    TABLE_DIR / "diagnostic_path_distribution.csv",
    TABLE_DIR / "cost_summary.csv",
    TABLE_DIR / "re2_error_analysis_before.csv",
    TABLE_DIR / "rcaeval_re2_re3_before_after.csv",
    TABLE_DIR / "second_dataset_admission.csv",
    FIGURE_DIR / "fig4_rcaeval_re2_re3_performance.png",
    FIGURE_DIR / "fig4_rcaeval_re2_re3_performance.pdf",
    FIGURE_DIR / "fig5_rcaeval_published_baseline_comparison.png",
    FIGURE_DIR / "fig5_rcaeval_published_baseline_comparison.pdf",
    FIGURE_DIR / "fig7_cost_and_path_distribution.png",
    FIGURE_DIR / "fig7_cost_and_path_distribution.pdf",
    REPORT_DIR / "context_admission_report.md",
    REPORT_DIR / "re2_failure_mode_report.md",
    REPORT_DIR / "re2_optimization_report.md",
    REPORT_DIR / "second_dataset_selection_report.md",
]

FORBIDDEN_MAIN_FILES = [
    TABLE_DIR / "eadro_summary.csv",
    TABLE_DIR / "case_level_predictions_eadro.csv",
    TABLE_DIR / "published_baselines_eadro.csv",
    TABLE_DIR / "rcaeval_all_subsets_summary.csv",
    FIGURE_DIR / "fig4_rcaeval_all_subsets_heatmap.png",
    FIGURE_DIR / "fig4_rcaeval_all_subsets_heatmap.pdf",
    FIGURE_DIR / "fig6_eadro_baseline_comparison.png",
    FIGURE_DIR / "fig6_eadro_baseline_comparison.pdf",
]


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def main() -> int:
    errors: list[str] = []
    for path in REQUIRED_FILES:
        if not path.exists():
            errors.append(f"missing required file: {path.relative_to(REPO_ROOT)}")
    for path in FORBIDDEN_MAIN_FILES:
        if path.exists():
            errors.append(f"obsolete file still in main location: {path.relative_to(REPO_ROOT)}")

    summary = TABLE_DIR / "rcaeval_re2_re3_summary.csv"
    if summary.exists():
        rows = _read_rows(summary)
        ids = {row.get("dataset_id", "") for row in rows}
        expected = {"RE2-OB", "RE2-SS", "RE2-TT", "RE3-OB", "RE3-SS", "RE3-TT"}
        if ids != expected:
            errors.append(f"unexpected RE2/RE3 summary ids: {sorted(ids)}")
        if any(row.get("dataset_id", "").startswith("RE1") for row in rows):
            errors.append("RE1 row found in main RE2/RE3 summary")

    out_report = REPORT_DIR / "artifact_validation_report.md"
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Artifact Validation Report",
        "",
        "## Result",
        "",
        "PASS" if not errors else "FAIL",
        "",
        "## Checks",
        "",
        f"- Required files checked: {len(REQUIRED_FILES)}.",
        f"- Forbidden obsolete main-location files checked: {len(FORBIDDEN_MAIN_FILES)}.",
        "- Main RCAEval summary must contain exactly RE2/RE3 systems.",
        "",
    ]
    if errors:
        lines.extend(["## Errors", ""])
        lines.extend(f"- {item}" for item in errors)
    else:
        lines.extend(["## Notes", "", "- RE1 and Eadro files are excluded from main artifact locations and retained under `paper_artifacts/obsolete/`."])
    out_report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out_report}")
    if errors:
        for item in errors:
            print(f"ERROR {item}")
        return 1
    print("artifact validation PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
