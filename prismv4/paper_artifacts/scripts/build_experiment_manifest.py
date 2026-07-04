#!/usr/bin/env python3
"""Build experiment manifest and final report for CAPE-RCA paper artifacts."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path
from typing import Any

from compute_metrics import compute
from paper_utils import ARTIFACT_ROOT, FIGURE_DIR, RAW_DIR, REPO_ROOT, TABLE_DIR, ensure_dirs


RCAEVAL_BASELINE_COLUMNS = [
    "dataset",
    "suite",
    "system",
    "fault_subset",
    "method",
    "AC@1",
    "AC@3",
    "Avg@5",
    "source_paper",
    "source_table_or_section",
    "notes",
    "comparable_to_CAPERCA",
]

EADRO_BASELINE_COLUMNS = [
    "dataset",
    "method",
    "HR@1",
    "HR@3",
    "HR@5",
    "NDCG@3",
    "NDCG@5",
    "source_paper",
    "source_table_or_section",
    "notes",
    "comparable_to_CAPERCA",
]


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, cwd=REPO_ROOT, text=True).strip()
    except Exception as exc:  # pragma: no cover - report best effort
        return f"unavailable ({type(exc).__name__}: {exc})"


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _ensure_csv(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in columns})


def ensure_baseline_tables() -> None:
    _ensure_csv(
        TABLE_DIR / "published_baselines_rcaeval.csv",
        RCAEVAL_BASELINE_COLUMNS,
        [
            {
                "dataset": "TODO",
                "suite": "TODO",
                "system": "TODO",
                "fault_subset": "TODO",
                "method": "TODO",
                "source_paper": "TODO",
                "source_table_or_section": "TODO",
                "notes": "Reliable published/official RCAEval baseline values still need source verification.",
                "comparable_to_CAPERCA": "false",
            }
        ],
    )
    _ensure_csv(
        TABLE_DIR / "published_baselines_eadro.csv",
        EADRO_BASELINE_COLUMNS,
        [
            {
                "dataset": "TODO",
                "method": "TODO",
                "source_paper": "TODO",
                "source_table_or_section": "TODO",
                "notes": "Reliable Eadro RCA-only published baseline values still need source verification.",
                "comparable_to_CAPERCA": "false",
            }
        ],
    )


def _markdown_table(rows: list[dict[str, str]], columns: list[str], limit: int = 12) -> str:
    if not rows:
        return "_No rows._"
    shown = rows[:limit]
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in shown:
        body.append("| " + " | ".join(str(row.get(col, "")).replace("\n", " ") for col in columns) + " |")
    if len(rows) > limit:
        body.append(f"| ... | {' | '.join([''] * (len(columns) - 1))} |")
    return "\n".join([header, sep, *body])


def _figure_list() -> list[str]:
    if not FIGURE_DIR.exists():
        return []
    return sorted(str(path.relative_to(ARTIFACT_ROOT)) for path in FIGURE_DIR.iterdir() if path.suffix in {".png", ".pdf"})


def _missing_notes() -> list[str]:
    notes: list[str] = []
    for row in _read_csv(TABLE_DIR / "rcaeval_all_subsets_summary.csv"):
        if "missing_result" in row.get("notes", ""):
            notes.append(f"{row['dataset_id']}: missing CAPE-RCA run")
        if "rank_missing" in row.get("notes", ""):
            notes.append(f"{row['dataset_id']}: {row['notes']}")
        if row.get("timeout_cases") not in ("", "0", 0):
            notes.append(f"{row['dataset_id']}: timeout/budget guarded cases={row.get('timeout_cases')}")
    for row in _read_csv(TABLE_DIR / "eadro_summary.csv"):
        if "rank_missing" in row.get("notes", ""):
            notes.append(f"{row['dataset']}: {row['notes']}")
    return notes


def build_manifest() -> str:
    ensure_dirs()
    ensure_baseline_tables()
    commit = _run(["git", "rev-parse", "HEAD"])
    status = _run(["git", "status", "--short", "--", "paper_artifacts", "experiments/rcaeval_adapter.py"])
    python_version = sys.version.replace("\n", " ")
    inventory = _read_csv(RAW_DIR / "existing_result_inventory.csv")
    figures = _figure_list()
    missing = _missing_notes()

    lines = [
        "# CAPE-RCA Experiment Manifest",
        "",
        "## Repository",
        "",
        f"- Commit hash: `{commit}`",
        f"- Python version: `{python_version}`",
        "- Worktree status: see below.",
        "",
        "```text",
        status or "clean",
        "```",
        "",
        "## Data Paths",
        "",
        "- RCAEval RE1/RE2/RE3: `/home/dell2/RCA513/ysj/dataset/RCAEval/{RE1,RE2,RE3}`",
        "- Eadro adapter workdir: `/home/dell2/RCA513/syh/datasets/Eadro/.adapter_work_v22`",
        "",
        "## Result Provenance",
        "",
        _markdown_table(
            inventory,
            [
                "dataset",
                "dataset_id",
                "source_file",
                "total_cases",
                "ranking_cases",
                "ranking_missing_cases",
                "total_tokens",
                "total_calls",
            ],
            limit=20,
        ),
        "",
        "## Commands",
        "",
        "```bash",
        "python3 paper_artifacts/scripts/collect_existing_results.py",
        "python3 paper_artifacts/scripts/compute_metrics.py",
        "python3 paper_artifacts/scripts/make_figures.py",
        "python3 paper_artifacts/scripts/extract_case_study.py",
        "python3 paper_artifacts/scripts/build_experiment_manifest.py",
        "# Optional missing RE1 runs:",
        "DRY_RUN=0 PRISM_CHT_MODEL_API_KEY=... bash paper_artifacts/scripts/run_missing_experiments.sh",
        "```",
        "",
        "## Configuration",
        "",
        "- Main method name: CAPE-RCA-Full adaptive path.",
        "- RCAEval runner: `experiments/run_rcaeval_continuous.py`.",
        "- Existing RCAEval RE2/RE3 source: `results/prism_cht/v3_full/`.",
        "- Existing Eadro source: `results/prism_cht/Eadro-{TT,SN}_ambig_full.json`.",
        "- Standard RCAEval Avg@5 formula: `(AC@1 + AC@2 + AC@3 + AC@4 + AC@5) / 5`.",
        "- Eadro NDCG@k formula for one ground-truth root: `1 / log2(rank + 1)` if rank <= k, otherwise 0.",
        "",
        "## Parsed vs Newly Generated",
        "",
        "- Parsed from existing files: RCAEval RE2/RE3 and Eadro TT/SN result JSONs listed above.",
        "- Newly run by this artifact pipeline: RCAEval RE1 metrics-only adaptive shortcut/budget-guarded runs in `paper_artifacts/raw/RE1-*_cape_rca_full.json`.",
        "- Newly generated by this artifact pipeline: compact raw summaries, CSV tables, figures, manifest, final report, and case-study files.",
        "",
        "## Missing Or Incomplete Metrics",
        "",
        "\n".join(f"- {item}" for item in missing) if missing else "- None detected.",
        "",
        "## Baseline Source Policy",
        "",
        "- Published baseline tables must contain only published paper, official benchmark, or official repository values.",
        "- Unverified values are left as `TODO`; reproduced local baseline outputs are not used as published numbers.",
        "- Eadro comparisons are RCA/localization-only and are not claimed as full end-to-end anomaly detection comparisons.",
        "",
        "## Figure Files",
        "",
        "\n".join(f"- `{item}`" for item in figures) if figures else "- No figures generated yet.",
        "",
    ]
    text = "\n".join(lines)
    (ARTIFACT_ROOT / "experiment_manifest.md").write_text(text, encoding="utf-8")
    return text


def build_final_report() -> None:
    rcaeval = _read_csv(TABLE_DIR / "rcaeval_all_subsets_summary.csv")
    eadro = _read_csv(TABLE_DIR / "eadro_summary.csv")
    rca_baselines = _read_csv(TABLE_DIR / "published_baselines_rcaeval.csv")
    eadro_baselines = _read_csv(TABLE_DIR / "published_baselines_eadro.csv")
    errors = _read_csv(TABLE_DIR / "error_analysis.csv")
    figures = _figure_list()
    missing = _missing_notes()

    lines = [
        "# CAPE-RCA Final Experiment Report",
        "",
        "## 1. Context Admission Report",
        "",
        "See `paper_artifacts/context_admission_report.md`. The pipeline indexed relevant result and runner files, excluded full raw telemetry/model I/O/log dumps from context, and admitted only compact summary fields.",
        "",
        "## 2. Parsed From Existing Results",
        "",
        "- RCAEval RE2/RE3: parsed from `results/prism_cht/v3_full/*.json`.",
        "- Eadro TT/SN: parsed from `results/prism_cht/Eadro-TT_ambig_full.json` and `results/prism_cht/Eadro-SN_ambig_full.json`.",
        "",
        "## 3. Newly Run",
        "",
        "RCAEval RE1-OB, RE1-SS, and RE1-TT were newly generated with `paper_artifacts/scripts/run_missing_experiments.sh`, which calls `run_re1_metrics_only_adaptive.py`. Strong IVD consensus cases use the adaptive shortcut; non-strong RE1 metrics-only cases are marked timeout/budget-exhausted rather than filled with fabricated EG-CDA outputs.",
        "",
        "## 4. Missing Or Incompatible Experiments",
        "",
        "\n".join(f"- {item}" for item in missing) if missing else "- None detected.",
        "",
        "## 5. RCAEval All-Subset Summary",
        "",
        _markdown_table(
            rcaeval,
            ["dataset_id", "cases", "AC@1", "AC@3", "Avg@5", "shortcut_cases", "egcda_cases", "notes"],
            limit=12,
        ),
        "",
        "## 6. Eadro Summary",
        "",
        _markdown_table(
            eadro,
            ["dataset", "cases", "HR@1", "HR@3", "HR@5", "NDCG@5", "egcda_cases", "notes"],
            limit=8,
        ),
        "",
        "## 7. Published Baselines",
        "",
        "### RCAEval",
        "",
        _markdown_table(rca_baselines, RCAEVAL_BASELINE_COLUMNS, limit=12),
        "",
        "### Eadro",
        "",
        _markdown_table(eadro_baselines, EADRO_BASELINE_COLUMNS, limit=12),
        "",
        "## 8. Figure Files",
        "",
        "\n".join(f"- `{item}`" for item in figures) if figures else "- No figures generated yet.",
        "",
        "## 9. Main Observations",
        "",
        "- RCAEval RE2/RE3 existing results are available for all six multi-source subsets.",
        "- RE1 is metrics-only and currently missing; it should be interpreted as compatibility coverage after rerun, not as multi-source causal diagnosis evidence.",
        "- CAPE-RCA-Full is reported as an adaptive path: shortcut statuses and continuous/EG-CDA statuses belong to the same main method, with path distribution reported separately.",
        "- Eadro results are localization-only under known fault cases; they should not be described as matching Eadro's full end-to-end anomaly detection protocol.",
        "- Published baseline comparisons remain conservative: unresolved or incompatible values are marked TODO rather than filled from local reproductions.",
        "",
        "## 10. Error Analysis Summary",
        "",
        _markdown_table(
            errors,
            ["dataset", "case_id", "ground_truth", "prediction", "diagnostic_path", "error_type", "short_reason"],
            limit=20,
        ),
        "",
        "## 11. Reproduction Commands",
        "",
        "```bash",
        "python3 paper_artifacts/scripts/collect_existing_results.py",
        "python3 paper_artifacts/scripts/compute_metrics.py",
        "python3 paper_artifacts/scripts/make_figures.py",
        "python3 paper_artifacts/scripts/extract_case_study.py",
        "python3 paper_artifacts/scripts/build_experiment_manifest.py",
        "```",
        "",
    ]
    (ARTIFACT_ROOT / "final_experiment_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build experiment manifest and final report.")
    parser.add_argument("--skip-compute", action="store_true", help="Do not recompute metrics first.")
    args = parser.parse_args()
    ensure_dirs()
    ensure_baseline_tables()
    if not args.skip_compute:
        compute()
    build_manifest()
    build_final_report()
    print(f"wrote {ARTIFACT_ROOT / 'experiment_manifest.md'} and {ARTIFACT_ROOT / 'final_experiment_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
