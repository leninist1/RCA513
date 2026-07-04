#!/usr/bin/env python3
"""Generate pre/post RE2 failure-mode analysis tables.

The script only reads compact result JSON files. It intentionally does not
export raw metrics/logs/traces.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from paper_utils import (
    REPO_ROOT,
    TABLE_DIR,
    available_modalities,
    fault_type_from_case_id,
    normalize_ranking,
    rank_of_ground_truth,
    split_rcaeval_system,
)


DEFAULT_RE2_RESULTS = {
    "RE2-OB": REPO_ROOT / "results/prism_cht/v3_full/RE2-OB_v3.json",
    "RE2-SS": REPO_ROOT / "results/prism_cht/v3_full/RE2-SS_v3.json",
    "RE2-TT": REPO_ROOT / "results/prism_cht/v3_full/RE2-TT_v3.json",
}

ERROR_TYPES = {
    "candidate_pool_miss",
    "wrong_resource_owner",
    "network_fault_direction_error",
    "storage_symptom_confusion",
    "downstream_symptom_selected",
    "metric_magnitude_bias",
    "missing_trace_direction",
    "weak_fault_signature",
    "false_cpsi_shortcut",
    "egcda_unresolved",
    "label_granularity_mismatch",
    "parser_or_adapter_issue",
    "other",
}

ERROR_COLUMNS = [
    "suite",
    "system",
    "case_id",
    "fault_type",
    "ground_truth",
    "prediction_top1",
    "rank_of_ground_truth",
    "diagnostic_path",
    "candidate_pool_contains_gt",
    "top_candidates",
    "status",
    "available_modalities",
    "short_reason",
    "error_type",
    "suspected_failure_mode",
]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _diagnostic_path(status: str, error: str = "") -> str:
    text = f"{status} {error}".lower()
    if "timeout" in text:
        return "timeout"
    if error or "failed" in text or "error" in text:
        return "failed"
    if "shortcut" in text or "fallback" in text:
        return "shortcut"
    return "egcda"


def _is_storage(component: str) -> bool:
    low = component.lower()
    return any(part in low for part in ("mongo", "redis", "mysql", "database")) or low.endswith("-db")


def _classify_failure(
    *,
    fault_type: str,
    ground_truth: str,
    prediction: str,
    rank: int | None,
    diagnostic_path: str,
    status: str,
) -> tuple[str, str, str]:
    if diagnostic_path == "failed":
        return (
            "parser_or_adapter_issue",
            "case failed before a reliable top-1 decision",
            "execution/parser failure in compact result",
        )
    if diagnostic_path == "timeout":
        return (
            "egcda_unresolved",
            "case timed out or exhausted the run budget",
            "full reasoning path did not complete",
        )
    if rank is None:
        return (
            "candidate_pool_miss",
            "ground truth is absent from the exported top-candidate list",
            "recall/ranking stage did not retain the true component",
        )
    if _is_storage(prediction) and not _is_storage(ground_truth):
        return (
            "storage_symptom_confusion",
            "storage/dependency symptom was selected over the service root",
            "resource pressure in dependency likely treated as source evidence",
        )
    if fault_type in {"loss", "delay"}:
        if diagnostic_path == "shortcut":
            return (
                "missing_trace_direction",
                "shortcut selected a network/latency symptom without full trace-direction checks",
                "IVD shortcut over-trusted propagation surface signals",
            )
        return (
            "network_fault_direction_error",
            "network/latency fault direction remained ambiguous",
            "directional trace evidence did not promote the injected owner",
        )
    if fault_type in {"cpu", "mem", "memory", "disk", "socket"} and prediction != ground_truth:
        if diagnostic_path == "shortcut":
            return (
                "false_cpsi_shortcut",
                "shortcut selected the wrong resource owner",
                "IVD/CPSI consensus was too permissive for a noisy multi-service case",
            )
        return (
            "wrong_resource_owner",
            "wrong component chosen for a resource fault",
            "resource-local signal was present but not ranked first",
        )
    if diagnostic_path == "shortcut":
        return (
            "false_cpsi_shortcut",
            "shortcut selected an incorrect top-1 root",
            "deterministic shortcut bypassed full causal checks",
        )
    if "fallback" in status:
        return (
            "weak_fault_signature",
            "fallback ranking had no strong causal winner",
            "deterministic evidence was weak or sparse",
        )
    return (
        "egcda_unresolved",
        "full EG-CDA reasoning ranked the ground truth below another candidate",
        "source-vs-symptom ambiguity remained unresolved",
    )


def _iter_error_rows(result_files: dict[str, Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset_id, path in result_files.items():
        if not path.exists():
            continue
        data = _load_json(path)
        suite, system = split_rcaeval_system(dataset_id)
        for raw in data.get("results", []):
            if not isinstance(raw, dict):
                continue
            ground_truth = str(raw.get("expected_component") or raw.get("ground_truth") or "")
            prediction = str(raw.get("predicted_component") or "")
            hit = bool(raw.get("hit"))
            status = str(raw.get("status") or "")
            error = str(raw.get("error") or "")
            diagnostic_path = _diagnostic_path(status, error)
            if hit and diagnostic_path not in {"failed", "timeout"}:
                continue
            ranking = normalize_ranking(raw.get("predicted_ranking"))
            rank = rank_of_ground_truth(ground_truth, ranking)
            fault_type = fault_type_from_case_id(str(raw.get("case_id") or ""))
            error_type, short_reason, mode = _classify_failure(
                fault_type=fault_type,
                ground_truth=ground_truth,
                prediction=prediction,
                rank=rank,
                diagnostic_path=diagnostic_path,
                status=status,
            )
            if error_type not in ERROR_TYPES:
                error_type = "other"
            rows.append(
                {
                    "suite": suite,
                    "system": system,
                    "case_id": raw.get("case_id"),
                    "fault_type": fault_type,
                    "ground_truth": ground_truth,
                    "prediction_top1": prediction,
                    "rank_of_ground_truth": rank if rank is not None else "",
                    "diagnostic_path": diagnostic_path,
                    "candidate_pool_contains_gt": int(rank is not None),
                    "top_candidates": json.dumps(ranking[:5], ensure_ascii=False),
                    "status": status,
                    "available_modalities": available_modalities(dataset_id),
                    "short_reason": short_reason,
                    "error_type": error_type,
                    "suspected_failure_mode": mode,
                }
            )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=ERROR_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in ERROR_COLUMNS})


def _write_report(path: Path, rows: list[dict[str, Any]], *, label: str) -> None:
    by_system: dict[str, Counter[str]] = defaultdict(Counter)
    by_error = Counter()
    by_fault = Counter()
    total_misses = len(rows)
    pool_miss = 0
    shortcut_miss = 0
    for row in rows:
        by_system[str(row["suite"]) + "-" + str(row["system"])] [str(row["error_type"])] += 1
        by_error[str(row["error_type"])] += 1
        by_fault[str(row["fault_type"])] += 1
        pool_miss += int(row["candidate_pool_contains_gt"] == 0)
        shortcut_miss += int(row["diagnostic_path"] == "shortcut")

    lines = [
        f"# RE2 Failure Mode Report ({label})",
        "",
        "## Scope",
        "",
        "This report analyzes RE2 compact result summaries only. It does not copy raw metrics, logs, or traces into the artifact set.",
        "",
        "## Aggregate Findings",
        "",
        f"- Total RE2 top-1 misses or failed cases: {total_misses}.",
        f"- Misses with ground truth absent from exported top candidates: {pool_miss}.",
        f"- Misses on shortcut/fallback paths: {shortcut_miss}.",
        "",
        "## Error Type Counts",
        "",
        "| error_type | cases |",
        "|---|---:|",
    ]
    for key, value in by_error.most_common():
        lines.append(f"| {key} | {value} |")

    lines.extend(["", "## Fault Type Counts", "", "| fault_type | cases |", "|---|---:|"])
    for key, value in by_fault.most_common():
        lines.append(f"| {key} | {value} |")

    lines.extend(["", "## System Breakdown", "", "| system | misses | dominant_error_types |", "|---|---:|---|"])
    for system in sorted(by_system):
        counts = by_system[system]
        dominant = ", ".join(f"{key}:{value}" for key, value in counts.most_common(3))
        lines.append(f"| {system} | {sum(counts.values())} | {dominant} |")

    lines.extend(
        [
            "",
            "## Optimization Implications",
            "",
            "- RE2-TT has many shortcut/fallback misses; the IVD shortcut should become more conservative on noisy multi-service cases.",
            "- Ground-truth absence in top candidates points to recall/ranking width rather than final reasoning alone.",
            "- Network and latency faults need explicit trace-direction checks before shortcut acceptance.",
            "- Resource-fault misses should be handled by owner-sensitive resource evidence, not by case or label-specific rules.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze RE2 failure modes from compact result JSON.")
    parser.add_argument("--label", default="before")
    parser.add_argument("--output", default=str(TABLE_DIR / "re2_error_analysis_before.csv"))
    parser.add_argument(
        "--report",
        default=str(REPO_ROOT / "paper_artifacts/reports/re2_failure_mode_report.md"),
    )
    parser.add_argument(
        "--result",
        action="append",
        default=[],
        help="Optional DATASET_ID=PATH override. May be passed more than once.",
    )
    args = parser.parse_args()

    result_files = dict(DEFAULT_RE2_RESULTS)
    for item in args.result:
        if "=" not in item:
            raise SystemExit(f"invalid --result value: {item}")
        key, value = item.split("=", 1)
        result_files[key] = Path(value)

    rows = _iter_error_rows(result_files)
    _write_csv(Path(args.output), rows)
    _write_report(Path(args.report), rows, label=args.label)
    print(f"wrote {args.output} with {len(rows)} rows")
    print(f"wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
