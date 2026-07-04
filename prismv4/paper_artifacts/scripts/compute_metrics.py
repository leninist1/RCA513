#!/usr/bin/env python3
"""Compute paper metrics from compact CAPE-RCA result summaries."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from collect_existing_results import collect
from paper_utils import (
    EADRO_SOURCE_FILES,
    RAW_DIR,
    RCAEVAL_SYSTEMS,
    TABLE_DIR,
    avg_at_5,
    available_modalities,
    ensure_dirs,
    fmt_float,
    hit_at_k,
    load_json,
    metric_hits,
    ndcg_at_k,
    split_rcaeval_system,
)


RCAEVAL_SUMMARY_COLUMNS = [
    "suite",
    "system",
    "dataset_id",
    "available_modalities",
    "cases",
    "hits@1",
    "AC@1",
    "hits@3",
    "AC@3",
    "Avg@5",
    "avg_rank",
    "total_calls",
    "avg_calls_per_case",
    "total_tokens",
    "avg_tokens_per_case",
    "elapsed_sec",
    "avg_elapsed_sec_per_case",
    "shortcut_cases",
    "egcda_cases",
    "failed_cases",
    "timeout_cases",
    "error_cases",
    "notes",
]

EADRO_SUMMARY_COLUMNS = [
    "dataset",
    "cases",
    "HR@1",
    "HR@3",
    "HR@5",
    "NDCG@3",
    "NDCG@5",
    "total_calls",
    "avg_calls_per_case",
    "total_tokens",
    "avg_tokens_per_case",
    "elapsed_sec",
    "avg_elapsed_sec_per_case",
    "shortcut_cases",
    "egcda_cases",
    "failed_cases",
    "timeout_cases",
    "error_cases",
    "notes",
]

PATH_COLUMNS = [
    "dataset",
    "suite_or_system",
    "shortcut_cases",
    "egcda_cases",
    "failed_cases",
    "timeout_cases",
    "shortcut_ratio",
    "egcda_ratio",
    "failed_ratio",
    "timeout_ratio",
]

COST_COLUMNS = [
    "dataset",
    "suite_or_system",
    "cases",
    "total_tokens",
    "avg_tokens_per_case",
    "total_calls",
    "avg_calls_per_case",
    "elapsed_sec",
    "avg_elapsed_sec_per_case",
]

ERROR_COLUMNS = [
    "dataset",
    "suite",
    "system",
    "case_id",
    "ground_truth",
    "prediction",
    "hit@1",
    "rank_of_ground_truth",
    "diagnostic_path",
    "error_type",
    "short_reason",
    "evidence_note",
]


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in columns})


def _load_or_collect() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cases_path = RAW_DIR / "normalized_cases.json"
    summaries_path = RAW_DIR / "result_file_summaries.json"
    if not cases_path.exists() or not summaries_path.exists():
        return collect()
    return load_json(cases_path), load_json(summaries_path)


def _group_cases(cases: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in cases:
        grouped[str(row.get("dataset_id") or row.get("dataset") or "")].append(row)
    return grouped


def _summary_by_dataset(summaries: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row.get("dataset_id") or ""): row for row in summaries}


def _path_counts(cases: list[dict[str, Any]]) -> Counter:
    return Counter(str(row.get("diagnostic_path") or "") for row in cases)


def _error_case_count(cases: list[dict[str, Any]]) -> int:
    return sum(1 for row in cases if row.get("error") or row.get("diagnostic_path") in {"failed", "timeout"})


def _ranking_notes(cases: list[dict[str, Any]]) -> str:
    missing = [row.get("case_id") for row in cases if not row.get("predicted_ranking_list")]
    if not missing:
        return ""
    sample = ", ".join(str(item) for item in missing[:5])
    suffix = "" if len(missing) <= 5 else f", +{len(missing) - 5} more"
    return (
        f"rank_missing={len(missing)}; top-k/Avg@5 are lower bounds where "
        f"only top-1 is available; missing_cases={sample}{suffix}"
    )


def _avg_rank(cases: list[dict[str, Any]]) -> str:
    ranks = []
    for row in cases:
        rank = row.get("rank_of_ground_truth")
        try:
            ranks.append(int(rank))
        except (TypeError, ValueError):
            continue
    if not ranks:
        return ""
    return fmt_float(sum(ranks) / len(ranks))


def _cost_values(dataset_id: str, cases: list[dict[str, Any]], summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    summary = summaries.get(dataset_id) or {}
    total_tokens = summary.get("total_tokens")
    total_calls = summary.get("total_calls")
    elapsed_sec = summary.get("elapsed_sec")
    if total_tokens in (None, ""):
        total_tokens = sum(int(row.get("total_tokens") or 0) for row in cases)
    if total_calls in (None, ""):
        total_calls = sum(int(row.get("total_calls") or 0) for row in cases)
    if elapsed_sec in (None, ""):
        elapsed_sec = sum(float(row.get("elapsed_sec") or 0) for row in cases)
    total = len(cases)
    total_tokens = int(total_tokens or 0)
    total_calls = int(total_calls or 0)
    elapsed_sec = float(elapsed_sec or 0)
    return {
        "total_tokens": total_tokens,
        "avg_tokens_per_case": fmt_float(total_tokens / total if total else None),
        "total_calls": total_calls,
        "avg_calls_per_case": fmt_float(total_calls / total if total else None),
        "elapsed_sec": fmt_float(elapsed_sec),
        "avg_elapsed_sec_per_case": fmt_float(elapsed_sec / total if total else None),
    }


def _rcaeval_summary(grouped: dict[str, list[dict[str, Any]]], summaries: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset_id in RCAEVAL_SYSTEMS:
        cases = grouped.get(dataset_id, [])
        suite, system = split_rcaeval_system(dataset_id)
        paths = _path_counts(cases)
        hit1, total, ac1, _ = metric_hits(cases, 1)
        hit3, _, ac3, miss3 = metric_hits(cases, 3)
        avg5, miss_avg = avg_at_5(cases)
        notes: list[str] = []
        if not cases:
            notes.append("missing_result")
        ranking_note = _ranking_notes(cases)
        if ranking_note:
            notes.append(ranking_note)
        if miss3 or miss_avg:
            notes.append("top-k_missing_values_counted_as_false_lower_bound")
        if dataset_id.startswith("RE1"):
            notes.append("RE1 is metrics-only; not multi-source causal evidence")
        if paths.get("timeout", 0):
            notes.append(
                f"timeout_or_budget_guarded_cases={paths.get('timeout', 0)}; "
                "EG-CDA fallback not completed for those metrics-only cases"
            )
        cost = _cost_values(dataset_id, cases, summaries)
        rows.append(
            {
                "suite": suite,
                "system": system,
                "dataset_id": dataset_id,
                "available_modalities": available_modalities(dataset_id),
                "cases": total if cases else "",
                "hits@1": hit1 if cases else "",
                "AC@1": fmt_float(ac1),
                "hits@3": hit3 if cases else "",
                "AC@3": fmt_float(ac3),
                "Avg@5": fmt_float(avg5),
                "avg_rank": _avg_rank(cases),
                "total_calls": cost["total_calls"] if cases else "",
                "avg_calls_per_case": cost["avg_calls_per_case"] if cases else "",
                "total_tokens": cost["total_tokens"] if cases else "",
                "avg_tokens_per_case": cost["avg_tokens_per_case"] if cases else "",
                "elapsed_sec": cost["elapsed_sec"] if cases else "",
                "avg_elapsed_sec_per_case": cost["avg_elapsed_sec_per_case"] if cases else "",
                "shortcut_cases": paths.get("shortcut", 0) if cases else "",
                "egcda_cases": paths.get("egcda", 0) if cases else "",
                "failed_cases": paths.get("failed", 0) if cases else "",
                "timeout_cases": paths.get("timeout", 0) if cases else "",
                "error_cases": _error_case_count(cases) if cases else "",
                "notes": "; ".join(notes),
            }
        )
    return rows


def _mean(values: list[float | None]) -> float | None:
    clean = [value for value in values if value is not None]
    if not clean:
        return None
    return sum(clean) / len(clean)


def _eadro_summary(grouped: dict[str, list[dict[str, Any]]], summaries: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset_id in EADRO_SOURCE_FILES:
        cases = grouped.get(dataset_id, [])
        paths = _path_counts(cases)
        _, total, hr1, _ = metric_hits(cases, 1)
        _, _, hr3, miss3 = metric_hits(cases, 3)
        _, _, hr5, miss5 = metric_hits(cases, 5)
        ndcg3_values = [ndcg_at_k(row, 3) for row in cases]
        ndcg5_values = [ndcg_at_k(row, 5) for row in cases]
        ndcg3_lower = [0.0 if value is None else value for value in ndcg3_values]
        ndcg5_lower = [0.0 if value is None else value for value in ndcg5_values]
        notes: list[str] = [
            "CAPE-RCA evaluated in known-fault localization setting, not Eadro end-to-end anomaly detection"
        ]
        ranking_note = _ranking_notes(cases)
        if ranking_note:
            notes.append(ranking_note)
        if miss3 or miss5:
            notes.append("HR@3/HR@5/NDCG are lower bounds where only top-1 is available")
        cost = _cost_values(dataset_id, cases, summaries)
        rows.append(
            {
                "dataset": dataset_id,
                "cases": total if cases else "",
                "HR@1": fmt_float(hr1),
                "HR@3": fmt_float(hr3),
                "HR@5": fmt_float(hr5),
                "NDCG@3": fmt_float(sum(ndcg3_lower) / len(cases) if cases else None),
                "NDCG@5": fmt_float(sum(ndcg5_lower) / len(cases) if cases else None),
                "total_calls": cost["total_calls"] if cases else "",
                "avg_calls_per_case": cost["avg_calls_per_case"] if cases else "",
                "total_tokens": cost["total_tokens"] if cases else "",
                "avg_tokens_per_case": cost["avg_tokens_per_case"] if cases else "",
                "elapsed_sec": cost["elapsed_sec"] if cases else "",
                "avg_elapsed_sec_per_case": cost["avg_elapsed_sec_per_case"] if cases else "",
                "shortcut_cases": paths.get("shortcut", 0) if cases else "",
                "egcda_cases": paths.get("egcda", 0) if cases else "",
                "failed_cases": paths.get("failed", 0) if cases else "",
                "timeout_cases": paths.get("timeout", 0) if cases else "",
                "error_cases": _error_case_count(cases) if cases else "",
                "notes": "; ".join(notes),
            }
        )
    return rows


def _path_and_cost_rows(
    grouped: dict[str, list[dict[str, Any]]],
    summaries: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    path_rows: list[dict[str, Any]] = []
    cost_rows: list[dict[str, Any]] = []
    for dataset_id in [*RCAEVAL_SYSTEMS, *EADRO_SOURCE_FILES.keys()]:
        cases = grouped.get(dataset_id, [])
        if not cases:
            continue
        dataset = "RCAEval" if dataset_id.startswith("RE") else "Eadro"
        suite_or_system = dataset_id
        paths = _path_counts(cases)
        total = len(cases)
        path_rows.append(
            {
                "dataset": dataset,
                "suite_or_system": suite_or_system,
                "shortcut_cases": paths.get("shortcut", 0),
                "egcda_cases": paths.get("egcda", 0),
                "failed_cases": paths.get("failed", 0),
                "timeout_cases": paths.get("timeout", 0),
                "shortcut_ratio": fmt_float(paths.get("shortcut", 0) / total),
                "egcda_ratio": fmt_float(paths.get("egcda", 0) / total),
                "failed_ratio": fmt_float(paths.get("failed", 0) / total),
                "timeout_ratio": fmt_float(paths.get("timeout", 0) / total),
            }
        )
        cost = _cost_values(dataset_id, cases, summaries)
        cost_rows.append(
            {
                "dataset": dataset,
                "suite_or_system": suite_or_system,
                "cases": total,
                **cost,
            }
        )
    return path_rows, cost_rows


def _classify_error(row: dict[str, Any]) -> tuple[str, str]:
    if row.get("diagnostic_path") == "timeout":
        return "timeout_or_budget_exhausted", "case timed out or exhausted execution budget"
    if row.get("error"):
        return "tool_or_parser_failure", str(row.get("error"))[:180]
    if str(row.get("suite") or "").startswith("RE1"):
        return "metric_only_evidence_insufficient", "metrics-only evidence could not disambiguate top-1"
    if not row.get("predicted_ranking_list"):
        return "other", "top-1 mismatch and ranking/evidence details are unavailable"
    if row.get("diagnostic_path") == "shortcut":
        return "false_cpsi_shortcut", "shortcut path selected an incorrect top-1 root"
    rank = row.get("rank_of_ground_truth")
    if rank in (None, ""):
        return "true_root_missing_from_candidate_pool", "ground truth absent from predicted ranking"
    return "source_like_symptom_unresolved", "ground truth ranked below a source-like symptom candidate"


def _error_rows(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in cases:
        if int(row.get("hit@1") or 0) == 1 and row.get("diagnostic_path") not in {"failed", "timeout"}:
            continue
        error_type, reason = _classify_error(row)
        rows.append(
            {
                "dataset": row.get("dataset"),
                "suite": row.get("suite"),
                "system": row.get("system") or row.get("dataset_id"),
                "case_id": row.get("case_id"),
                "ground_truth": row.get("ground_truth"),
                "prediction": row.get("prediction_top1"),
                "hit@1": row.get("hit@1"),
                "rank_of_ground_truth": row.get("rank_of_ground_truth") or "",
                "diagnostic_path": row.get("diagnostic_path"),
                "error_type": error_type,
                "short_reason": reason,
                "evidence_note": "Automated from compact result fields; use case-study extraction for manual evidence review.",
            }
        )
    return rows


def compute() -> None:
    ensure_dirs()
    cases, file_summaries = _load_or_collect()
    grouped = _group_cases(cases)
    summaries = _summary_by_dataset(file_summaries)
    rcaeval_rows = _rcaeval_summary(grouped, summaries)
    eadro_rows = _eadro_summary(grouped, summaries)
    path_rows, cost_rows = _path_and_cost_rows(grouped, summaries)
    errors = _error_rows(cases)

    _write_csv(TABLE_DIR / "rcaeval_all_subsets_summary.csv", rcaeval_rows, RCAEVAL_SUMMARY_COLUMNS)
    _write_csv(TABLE_DIR / "eadro_summary.csv", eadro_rows, EADRO_SUMMARY_COLUMNS)
    _write_csv(TABLE_DIR / "diagnostic_path_distribution.csv", path_rows, PATH_COLUMNS)
    _write_csv(TABLE_DIR / "cost_summary.csv", cost_rows, COST_COLUMNS)
    _write_csv(TABLE_DIR / "error_analysis.csv", errors, ERROR_COLUMNS)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute CAPE-RCA paper metrics.")
    parser.parse_args()
    compute()
    print(f"wrote metric tables under {TABLE_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
