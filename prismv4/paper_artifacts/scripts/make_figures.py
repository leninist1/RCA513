#!/usr/bin/env python3
"""Generate paper figures from CAPE-RCA artifact tables."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from compute_metrics import compute
from paper_utils import FIGURE_DIR, TABLE_DIR, ensure_dirs


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _float(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _save(fig: plt.Figure, name: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURE_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def _placeholder(name: str, title: str, message: str) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 3.6))
    ax.axis("off")
    ax.set_title(title, fontsize=13, pad=12)
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=11, wrap=True)
    _save(fig, name)


def fig4_heatmap() -> None:
    rows = _read_csv(TABLE_DIR / "rcaeval_all_subsets_summary.csv")
    by_id = {row["dataset_id"]: row for row in rows}
    suites = ["RE1", "RE2", "RE3"]
    systems = [("OB", "Online Boutique"), ("SS", "Sock Shop"), ("TT", "Train Ticket")]
    values: list[list[float]] = []
    labels: list[list[str]] = []
    for suite in suites:
        value_row = []
        label_row = []
        for code, _ in systems:
            value = _float((by_id.get(f"{suite}-{code}") or {}).get("AC@1"))
            value_row.append(-1.0 if value is None else value)
            label_row.append("TODO" if value is None else f"{value * 100:.1f}%")
        values.append(value_row)
        labels.append(label_row)

    cmap = ListedColormap(["#d9d9d9", "#f2f7fb", "#c6dbef", "#6baed6", "#2171b5"])
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    display = [[0.0 if item < 0 else item for item in row] for row in values]
    im = ax.imshow(display, vmin=0.0, vmax=1.0, cmap=cmap)
    ax.set_xticks(range(3), [name for _, name in systems], fontsize=10)
    ax.set_yticks(range(3), suites, fontsize=10)
    ax.set_title("CAPE-RCA performance across all RCAEval subsets", fontsize=13, pad=12)
    for i in range(3):
        for j in range(3):
            color = "black" if values[i][j] < 0 or values[i][j] < 0.68 else "white"
            ax.text(j, i, labels[i][j], ha="center", va="center", fontsize=11, color=color)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("AC@1", fontsize=10)
    ax.text(
        0.0,
        -0.22,
        "Note: RE1 is metrics-only; RE2 and RE3 use multi-source telemetry. Gray/TODO cells are missing runs.",
        transform=ax.transAxes,
        fontsize=9,
        va="top",
    )
    _save(fig, "fig4_rcaeval_all_subsets_heatmap")


def _truthy(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def fig5_rcaeval_baselines() -> None:
    baselines = [
        row
        for row in _read_csv(TABLE_DIR / "published_baselines_rcaeval.csv")
        if _truthy(row.get("comparable_to_CAPERCA", "")) and _float(row.get("AC@1")) is not None
    ]
    if not baselines:
        _placeholder(
            "fig5_rcaeval_published_baseline_comparison",
            "Comparison with published RCAEval baselines",
            "No fully comparable published RCAEval baseline value has been resolved yet. "
            "The baseline table marks unresolved entries as TODO.",
        )
        return

    summary = _read_csv(TABLE_DIR / "rcaeval_all_subsets_summary.csv")
    by_id = {row["dataset_id"]: row for row in summary}
    dataset_id = baselines[0].get("dataset") or baselines[0].get("suite") or ""
    cape_value = _float((by_id.get(dataset_id) or {}).get("AC@1"))
    labels = [row["method"] for row in baselines]
    values = [_float(row.get("AC@1")) or 0.0 for row in baselines]
    colors = ["#6baed6"] * len(labels)
    if cape_value is not None:
        labels.append("CAPE-RCA")
        values.append(cape_value)
        colors.append("#238b45")

    fig, ax = plt.subplots(figsize=(7.2, max(3.8, 0.38 * len(labels) + 1.5)))
    positions = list(range(len(labels)))
    ax.barh(positions, values, color=colors)
    ax.set_yticks(positions, labels, fontsize=10)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("AC@1", fontsize=11)
    ax.set_title(f"Comparison with published RCAEval baselines on {dataset_id}", fontsize=13, pad=10)
    for y, value in zip(positions, values):
        ax.text(min(value + 0.02, 0.98), y, f"{value * 100:.1f}%", va="center", fontsize=10)
    ax.grid(axis="x", alpha=0.25)
    _save(fig, "fig5_rcaeval_published_baseline_comparison")


def fig6_eadro_baselines() -> None:
    rows = _read_csv(TABLE_DIR / "published_baselines_eadro.csv")
    comparable = [
        row
        for row in rows
        if _truthy(row.get("comparable_to_CAPERCA", "")) and _float(row.get("HR@1")) is not None
    ]
    if not comparable:
        _placeholder(
            "fig6_eadro_baseline_comparison",
            "Eadro root-cause localization comparison",
            "No comparable Eadro published RCA-only baseline value has been resolved yet. "
            "CAPE-RCA is evaluated under known-fault localization, so end-to-end anomaly detection "
            "numbers are not plotted.",
        )
        return

    summary = {row["dataset"]: row for row in _read_csv(TABLE_DIR / "eadro_summary.csv")}
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 6.2), sharex=True)
    for ax, dataset in zip(axes, ["Eadro-TT", "Eadro-SN"]):
        data_rows = [row for row in comparable if row.get("dataset") == dataset]
        labels = [row["method"] for row in data_rows]
        values = [_float(row.get("HR@1")) or 0.0 for row in data_rows]
        cape = _float((summary.get(dataset) or {}).get("HR@1"))
        colors = ["#9ecae1"] * len(labels)
        if cape is not None:
            labels.append("CAPE-RCA")
            values.append(cape)
            colors.append("#238b45")
        positions = list(range(len(labels)))
        ax.barh(positions, values, color=colors)
        ax.set_title(dataset, fontsize=12)
        ax.set_xlim(0, 1.05)
        ax.set_yticks(positions, labels, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("HR@1", fontsize=10)
        ax.grid(axis="x", alpha=0.25)
        for y, value in zip(positions, values):
            ax.text(min(value + 0.015, 1.0), y, f"{value * 100:.1f}%", va="center", fontsize=8)
    fig.suptitle("Eadro localization comparison", fontsize=13)
    fig.text(
        0.5,
        0.02,
        "Note: CAPE-RCA is evaluated under a localization setting with known fault cases.",
        ha="center",
        fontsize=9,
    )
    fig.subplots_adjust(left=0.14, right=0.98, bottom=0.12, top=0.88, wspace=0.34)
    _save(fig, "fig6_eadro_baseline_comparison")


def fig7_cost_and_path() -> None:
    cost_rows = _read_csv(TABLE_DIR / "cost_summary.csv")
    path_rows = _read_csv(TABLE_DIR / "diagnostic_path_distribution.csv")
    if not cost_rows or not path_rows:
        _placeholder(
            "fig7_cost_and_path_distribution",
            "Diagnostic cost and path distribution",
            "Cost/path tables are empty. Run compute_metrics.py first.",
        )
        return
    path_by_label = {row["suite_or_system"]: row for row in path_rows}
    labels = [row["suite_or_system"] for row in cost_rows]
    token_values = [_float(row.get("avg_tokens_per_case")) or 0.0 for row in cost_rows]

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6))
    axes[0].bar(range(len(labels)), token_values, color="#3182bd")
    axes[0].set_xticks(range(len(labels)), labels, rotation=35, ha="right", fontsize=9)
    axes[0].set_ylabel("Average tokens per case", fontsize=11)
    axes[0].set_title("(a) Diagnostic cost", fontsize=12)
    axes[0].grid(axis="y", alpha=0.25)

    shortcut = [_float((path_by_label.get(label) or {}).get("shortcut_ratio")) or 0.0 for label in labels]
    egcda = [_float((path_by_label.get(label) or {}).get("egcda_ratio")) or 0.0 for label in labels]
    failed = [
        (_float((path_by_label.get(label) or {}).get("failed_ratio")) or 0.0)
        + (_float((path_by_label.get(label) or {}).get("timeout_ratio")) or 0.0)
        for label in labels
    ]
    x = list(range(len(labels)))
    axes[1].bar(x, shortcut, color="#74c476", label="CPSI shortcut")
    axes[1].bar(x, egcda, bottom=shortcut, color="#6baed6", label="EG-CDA loop")
    bottoms = [shortcut[i] + egcda[i] for i in range(len(labels))]
    axes[1].bar(x, failed, bottom=bottoms, color="#fb6a4a", label="Failed/timeout")
    axes[1].set_xticks(x, labels, rotation=35, ha="right", fontsize=9)
    axes[1].set_ylim(0, 1.0)
    axes[1].set_ylabel("Case ratio", fontsize=11)
    axes[1].set_title("(b) Diagnostic path distribution", fontsize=12)
    axes[1].legend(fontsize=9, loc="upper right")
    axes[1].grid(axis="y", alpha=0.25)
    fig.suptitle("CAPE-RCA cost and adaptive path distribution", fontsize=13)
    _save(fig, "fig7_cost_and_path_distribution")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate CAPE-RCA paper figures.")
    parser.add_argument("--skip-compute", action="store_true", help="Do not recompute tables before plotting.")
    args = parser.parse_args()
    ensure_dirs()
    if not args.skip_compute:
        compute()
    fig4_heatmap()
    fig5_rcaeval_baselines()
    fig6_eadro_baselines()
    fig7_cost_and_path()
    print(f"wrote figures under {FIGURE_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
