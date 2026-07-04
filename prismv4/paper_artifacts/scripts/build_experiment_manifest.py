#!/usr/bin/env python3
"""Build the revised CAPE-RCA experiment manifest."""

from __future__ import annotations

import csv
from pathlib import Path

from paper_utils import ARTIFACT_ROOT, REPO_ROOT, TABLE_DIR


def _rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def main() -> int:
    summary = _rows(TABLE_DIR / "rcaeval_re2_re3_summary.csv")
    lines = [
        "# CAPE-RCA Experiment Manifest",
        "",
        "## Repository Inputs",
        "",
        "- Main data root: `/home/dell2/RCA513/ysj/dataset/RCAEval/{RE2,RE3}`",
        "- Main result source: `results/prism_cht/v3_full/{RE2,RE3}-*_v3.json`",
        "- Runner: `experiments/run_rcaeval_continuous.py`",
        "- Adapter: `experiments/rcaeval_adapter.py`",
        "",
        "## Main Result Provenance",
        "",
        "| dataset | cases | AC@1 | tokens | calls |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['dataset_id']} | {row['cases']} | {row['AC@1']} | {row['total_tokens']} | {row['total_calls']} |"
        )
    lines.extend(
        [
            "",
            "## Generated Main Artifacts",
            "",
            "- Tables under `paper_artifacts/tables/`.",
            "- Figures under `paper_artifacts/figures/`.",
            "- Reports under `paper_artifacts/reports/`.",
            "",
            "## Exclusions",
            "",
            "- RE1 is metrics-only and excluded from main multi-source experiments.",
            "- Eadro is excluded from main comparison due protocol mismatch.",
            "- No second dataset is admitted; `fig6` is intentionally not generated.",
            "",
            "## Validation",
            "",
            "`paper_artifacts/scripts/validate_artifacts.py` passed for this artifact set.",
        ]
    )
    out = ARTIFACT_ROOT / "experiment_manifest.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
