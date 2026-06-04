"""Summarize selective RCA evaluation JSON files into a compact Markdown report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DISPLAY_TARGETS = [
    "strict",
    "partial",
    "actionable_any",
    "component_reason_pair_any",
    "time_any",
    "component_any",
    "reason_any",
    "component_reason_pair",
    "actionable",
    "time",
    "component",
    "reason",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lodo", required=True)
    parser.add_argument("--forward", default="")
    parser.add_argument("--out", default="confidence_aware_rca/experiments/selective_summary.md")
    parser.add_argument("--display-targets", default=",".join(DISPLAY_TARGETS))
    return parser.parse_args()


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def pct(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return "-"


def stat_cell(row: dict[str, Any], key: str) -> str:
    stat = row.get(key) or {}
    return f"{pct(stat.get('coverage'))} / {pct(stat.get('selective_accuracy'))}"


def feature_set_cell(row: dict[str, Any]) -> str:
    selected = row.get("selected_feature_sets")
    if isinstance(selected, dict) and selected:
        return ", ".join(f"{key}:{value}" for key, value in sorted(selected.items()))
    return str(row.get("selected_feature_set") or row.get("feature_set") or "-")


def lodo_table(report: dict[str, Any], display_targets: list[str]) -> list[str]:
    lines = [
        "## LODO Selective Evaluation",
        "",
        "| target | feature set | base rate | coverage@80 / acc | coverage@90 / acc | AURC | ECE |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    targets = report.get("targets", {})
    for target in display_targets:
        if target not in targets:
            continue
        row = targets[target]
        lines.append(
            f"| {target} | {feature_set_cell(row)} | {pct(row.get('base_rate'))} | "
            f"{stat_cell(row, 'coverage_at_train_acc_0.80')} | "
            f"{stat_cell(row, 'coverage_at_train_acc_0.90')} | "
            f"{float(row.get('risk_coverage', {}).get('aurc', 0.0)):.4f} | "
            f"{float(row.get('ece', 0.0)):.4f} |"
        )
    return lines


def forward_table(report: dict[str, Any], display_targets: list[str]) -> list[str]:
    lines = [
        "## Forward Holdout",
        "",
        f"Heldout dates: `{', '.join(report.get('heldout_dates', []))}`",
        "",
        "| target | feature set | test base rate | coverage@80 / acc | coverage@90 / acc | AURC |",
        "|---|---|---:|---:|---:|---:|",
    ]
    targets = report.get("targets", {})
    for target in display_targets:
        if target not in targets:
            continue
        row = targets[target]
        lines.append(
            f"| {target} | {feature_set_cell(row)} | {pct(row.get('test_base_rate'))} | "
            f"{stat_cell(row, 'coverage_at_train_acc_0.80')} | "
            f"{stat_cell(row, 'coverage_at_train_acc_0.90')} | "
            f"{float(row.get('risk_coverage_test', {}).get('aurc', 0.0)):.4f} |"
        )
    return lines


def main() -> int:
    args = parse_args()
    lodo = load_json(args.lodo)
    display_targets = [item.strip() for item in args.display_targets.split(",") if item.strip()]
    lines = [
        "# Confidence-Aware RCA Summary",
        "",
        f"LODO model: `{lodo.get('model')}`",
        f"LODO risk mode: `{lodo.get('risk_mode', 'empirical')}`",
        "",
        "Cell format for coverage columns: `coverage / selective accuracy`.",
        "",
    ]
    lines.extend(lodo_table(lodo, display_targets))
    if args.forward:
        forward = load_json(args.forward)
        lines.extend([""])
        lines.extend(forward_table(forward, display_targets))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
