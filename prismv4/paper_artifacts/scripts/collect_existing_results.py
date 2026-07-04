#!/usr/bin/env python3
"""Collect compact CAPE-RCA result fields for paper metrics.

This script deliberately avoids printing or exporting full raw result dumps.
It emits normalized case rows, inventory summaries, and case-level prediction
CSV files needed by the paper artifact pipeline.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Any

from paper_utils import (
    RAW_DIR,
    RCAEVAL_SOURCE_FILES,
    TABLE_DIR,
    compact_json,
    ensure_dirs,
    fault_type_from_case_id,
    load_json,
    normalize_ranking,
    rank_of_ground_truth,
    relpath,
    split_rcaeval_system,
    status_category,
    write_json,
)


RCAEVAL_CASE_COLUMNS = [
    "suite",
    "system",
    "case_id",
    "fault_type",
    "ground_truth",
    "prediction_top1",
    "hit@1",
    "rank_of_ground_truth",
    "predicted_ranking",
    "diagnostic_path",
    "total_tokens",
    "total_calls",
    "status",
    "notes",
]

def _case_cost(row: dict[str, Any]) -> tuple[int, int]:
    usage = row.get("token_usage") or {}
    if not isinstance(usage, dict):
        return 0, 0
    tokens = usage.get("total_tokens") or 0
    calls = usage.get("call_count") or 0
    try:
        tokens_i = int(tokens)
    except (TypeError, ValueError):
        tokens_i = 0
    try:
        calls_i = int(calls)
    except (TypeError, ValueError):
        calls_i = 0
    return tokens_i, calls_i


def _summary_cost(data: dict[str, Any], cases: list[dict[str, Any]]) -> dict[str, int]:
    cost = data.get("cost") or {}
    if not isinstance(cost, dict):
        cost = {}
    total_tokens = cost.get("total_tokens")
    total_calls = cost.get("total_calls")
    if total_tokens is None:
        total_tokens = sum(_case_cost(row)[0] for row in cases)
    if total_calls is None:
        total_calls = sum(_case_cost(row)[1] for row in cases)
    try:
        total_tokens_i = int(total_tokens)
    except (TypeError, ValueError):
        total_tokens_i = 0
    try:
        total_calls_i = int(total_calls)
    except (TypeError, ValueError):
        total_calls_i = 0
    return {"total_tokens": total_tokens_i, "total_calls": total_calls_i}


def _normalize_case(
    *,
    raw: dict[str, Any],
    dataset: str,
    dataset_id: str,
    source_path: Path,
) -> dict[str, Any]:
    case_id = str(raw.get("case_id") or "")
    ground_truth = str(raw.get("expected_component") or raw.get("ground_truth") or "")
    prediction = str(raw.get("predicted_component") or raw.get("prediction_top1") or "")
    ranking = normalize_ranking(raw.get("predicted_ranking"))
    rank = rank_of_ground_truth(ground_truth, ranking)
    hit = bool(raw.get("hit")) if "hit" in raw else (ground_truth == prediction and bool(ground_truth))
    tokens, calls = _case_cost(raw)
    status = str(raw.get("status") or "")
    error = str(raw.get("error") or "")
    diagnostic_path = status_category(status, error)
    suite = ""
    system = dataset_id
    if dataset == "RCAEval":
        suite, system = split_rcaeval_system(dataset_id)
    notes: list[str] = []
    if not ranking:
        notes.append("predicted_ranking_missing")
    if error:
        notes.append(f"error={error}")
    if diagnostic_path == "egcda" and status and "continuous" not in status.lower():
        notes.append(f"non_shortcut_status={status}")
    return {
        "dataset": dataset,
        "dataset_id": dataset_id,
        "suite": suite,
        "system": system,
        "case_id": case_id,
        "fault_type": fault_type_from_case_id(case_id),
        "ground_truth": ground_truth,
        "prediction_top1": prediction,
        "hit@1": int(hit),
        "rank_of_ground_truth": rank,
        "predicted_ranking_list": ranking,
        "predicted_ranking": compact_json(ranking),
        "diagnostic_path": diagnostic_path,
        "total_tokens": tokens,
        "total_calls": calls,
        "status": status,
        "error": error,
        "elapsed_sec": raw.get("elapsed_sec") or 0,
        "source_file": relpath(source_path),
        "notes": "; ".join(notes),
    }


def _parse_source(
    *,
    dataset: str,
    dataset_id: str,
    source_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    if not source_path.exists():
        return [], None
    data = load_json(source_path)
    cases = data.get("results") or data.get("cases") or data.get("case_results") or []
    if isinstance(cases, dict):
        cases = list(cases.values())
    if not isinstance(cases, list):
        cases = []
    normalized = [
        _normalize_case(raw=row, dataset=dataset, dataset_id=dataset_id, source_path=source_path)
        for row in cases
        if isinstance(row, dict)
    ]
    status_counts = Counter(str(row.get("status") or "") for row in cases if isinstance(row, dict))
    path_counts = Counter(row["diagnostic_path"] for row in normalized)
    ranking_cases = sum(1 for row in normalized if row["predicted_ranking_list"])
    error_cases = [row["case_id"] for row in normalized if row.get("error")]
    cost = _summary_cost(data, cases)
    summary = {
        "dataset": dataset,
        "dataset_id": dataset_id,
        "source_file": relpath(source_path),
        "file_size_bytes": source_path.stat().st_size,
        "top_level_dataset": data.get("dataset", ""),
        "top_level_system": data.get("system", ""),
        "mode": data.get("mode", ""),
        "total_cases": len(normalized),
        "top1_hits": data.get("top1_hits", sum(row["hit@1"] for row in normalized)),
        "top1_accuracy": data.get("top1_accuracy", ""),
        "elapsed_sec": data.get("elapsed_sec", ""),
        "total_tokens": cost["total_tokens"],
        "total_calls": cost["total_calls"],
        "ranking_cases": ranking_cases,
        "ranking_missing_cases": len(normalized) - ranking_cases,
        "status_distribution": dict(status_counts),
        "diagnostic_path_distribution": dict(path_counts),
        "error_cases": error_cases,
    }
    return normalized, summary


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in columns})


def collect() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ensure_dirs()
    all_cases: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for dataset_id, source in RCAEVAL_SOURCE_FILES.items():
        cases, summary = _parse_source(dataset="RCAEval", dataset_id=dataset_id, source_path=source)
        all_cases.extend(cases)
        if summary:
            summaries.append(summary)
    write_json(RAW_DIR / "normalized_cases.json", all_cases)
    write_json(RAW_DIR / "result_file_summaries.json", summaries)
    write_json(
        RAW_DIR / "result_provenance.json",
        {
            "rcaeval_sources": {key: relpath(path) for key, path in RCAEVAL_SOURCE_FILES.items()},
            "excluded_sources": {
                "RE1": "metrics-only RCAEval subset; moved to paper_artifacts/obsolete/re1_metrics_only/",
                "Eadro": "protocol mismatch for main comparison; moved to paper_artifacts/obsolete/eadro_exploratory/",
            },
            "note": "Existing RE2/RE3 files are parsed in place; raw result files are not copied or overwritten.",
        },
    )

    inventory_columns = [
        "dataset",
        "dataset_id",
        "source_file",
        "file_size_bytes",
        "top_level_dataset",
        "top_level_system",
        "mode",
        "total_cases",
        "top1_hits",
        "top1_accuracy",
        "elapsed_sec",
        "total_tokens",
        "total_calls",
        "ranking_cases",
        "ranking_missing_cases",
        "status_distribution",
        "diagnostic_path_distribution",
        "error_cases",
    ]
    inventory_rows = []
    for row in summaries:
        out = dict(row)
        out["status_distribution"] = compact_json(out.get("status_distribution"))
        out["diagnostic_path_distribution"] = compact_json(out.get("diagnostic_path_distribution"))
        out["error_cases"] = compact_json(out.get("error_cases"))
        inventory_rows.append(out)
    _write_csv(RAW_DIR / "existing_result_inventory.csv", inventory_rows, inventory_columns)

    rcaeval_rows = [row for row in all_cases if row["dataset"] == "RCAEval"]
    _write_csv(TABLE_DIR / "case_level_predictions_rcaeval.csv", rcaeval_rows, RCAEVAL_CASE_COLUMNS)
    return all_cases, summaries


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect compact CAPE-RCA result summaries.")
    parser.parse_args()
    cases, summaries = collect()
    print(
        f"collected {len(cases)} cases from {len(summaries)} result files; "
        f"wrote {TABLE_DIR / 'case_level_predictions_rcaeval.csv'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
