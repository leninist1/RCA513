"""Reason-conditional tolerance evaluator for OpenRCA Bank predictions.

This diagnostic keeps the official field-matching semantics, but replaces the
fixed <=1min time tolerance with a reason-aware tolerance. It is meant as a
supplementary metric, not a replacement for the official evaluator.
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from datetime import datetime
import itertools
import json
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from field_hit_diagnostics import parse_prediction, parse_truth  # noqa: E402


EVENT_LIKE = {
    "high CPU usage",
    "high JVM CPU load",
    "network latency",
    "network packet loss",
}

PROGRESSIVE_LIKE = {
    "high memory usage",
    "JVM Out of Memory (OOM) Heap",
    "high disk space usage",
}

TOLERANCE_BY_REASON_MINUTES = {
    "high CPU usage": 1,
    "high JVM CPU load": 1,
    "network latency": 2,
    "network packet loss": 2,
    "high disk I/O read usage": 5,
    "high disk space usage": 30,
    "high memory usage": 15,
    "JVM Out of Memory (OOM) Heap": 15,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred", default="logs/openrca_query_v2_trace_batch18_predictions.csv")
    parser.add_argument("--query", default="/home/yan/workspace/data/openrca/Bank/query.csv")
    parser.add_argument("--record", default="/home/yan/workspace/data/openrca/Bank/record.csv")
    parser.add_argument("--out", default="logs/reason_conditional_tolerance_eval_batch18.json")
    return parser.parse_args()


def reason_group(reason: str) -> str:
    if reason in EVENT_LIKE:
        return "event_like"
    if reason in PROGRESSIVE_LIKE:
        return "progressive_like"
    if reason == "high disk I/O read usage":
        return "io_like"
    return "other"


def parse_time(value: str) -> datetime | None:
    try:
        return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
    except (TypeError, ValueError):
        return None


def time_ok(expected: str, predicted: str, tolerance_minutes: float) -> bool:
    a = parse_time(expected)
    b = parse_time(predicted)
    if a is None or b is None:
        return False
    return abs((a - b).total_seconds()) <= tolerance_minutes * 60.0


def field_hit(field: str, truth: dict[str, str], pred: dict[str, str], tolerance_minutes: float) -> bool:
    if field not in truth:
        return False
    if field == "time":
        return time_ok(truth["time"], pred.get("time", ""), tolerance_minutes)
    return truth[field] == pred.get(field, "")


def score_assignment(
    preds: list[dict],
    truths: list[dict],
    tolerance_reasons: list[str],
    fixed_tolerance_minutes: float | None = None,
) -> tuple[dict[str, int], int, float]:
    required = sorted({field for truth in truths for field in truth})
    field_totals = {field: sum(1 for truth in truths if field in truth) for field in ("time", "component", "reason")}
    if len(preds) != len(truths) or not truths:
        return {field: 0 for field in ("time", "component", "reason")}, sum(field_totals.values()), 0.0
    best_hits = None
    best_score = -1
    for perm in itertools.permutations(preds):
        hits = {field: 0 for field in ("time", "component", "reason")}
        for idx, (truth, pred) in enumerate(zip(truths, perm)):
            tol = fixed_tolerance_minutes
            if tol is None:
                tol = TOLERANCE_BY_REASON_MINUTES.get(tolerance_reasons[idx], 5)
            for field in required:
                hits[field] += int(field_hit(field, truth, pred, tol))
        score = sum(hits.values())
        if score > best_score:
            best_score = score
            best_hits = hits
    total = sum(field_totals.values())
    return best_hits or {field: 0 for field in ("time", "component", "reason")}, total, best_score / total if total else 0.0


def load_csv(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def record_lookup(record_rows: list[dict]) -> dict[str, dict]:
    return {row["datetime"].strip(): row for row in record_rows}


def reason_for_truth(truth: dict[str, str], fallback_record: dict, by_time: dict[str, dict]) -> str:
    if truth.get("reason"):
        return truth["reason"]
    if truth.get("time") and truth["time"] in by_time:
        return by_time[truth["time"]]["reason"].strip()
    return fallback_record["reason"].strip()


def init_stats() -> dict:
    return {
        "n": 0,
        "official_strict": 0,
        "reason_conditional_strict": 0,
        "official_partial_sum": 0.0,
        "reason_conditional_partial_sum": 0.0,
        "field_totals": {"time": 0, "component": 0, "reason": 0},
        "official_field_hits": {"time": 0, "component": 0, "reason": 0},
        "reason_conditional_field_hits": {"time": 0, "component": 0, "reason": 0},
    }


def add_stats(table: dict, key: str, row: dict) -> None:
    stats = table.setdefault(key, init_stats())
    stats["n"] += 1
    stats["official_strict"] += int(row["official_score"] == 1.0)
    stats["reason_conditional_strict"] += int(row["reason_conditional_score"] == 1.0)
    stats["official_partial_sum"] += row["official_score"]
    stats["reason_conditional_partial_sum"] += row["reason_conditional_score"]
    for field in ("time", "component", "reason"):
        stats["field_totals"][field] += row["field_totals"][field]
        stats["official_field_hits"][field] += row["official_hits"][field]
        stats["reason_conditional_field_hits"][field] += row["reason_conditional_hits"][field]


def finish_stats(stats: dict) -> dict:
    n = stats["n"]
    out = {
        **stats,
        "official_strict_rate": stats["official_strict"] / n if n else 0.0,
        "reason_conditional_strict_rate": stats["reason_conditional_strict"] / n if n else 0.0,
        "official_partial_rate": stats["official_partial_sum"] / n if n else 0.0,
        "reason_conditional_partial_rate": stats["reason_conditional_partial_sum"] / n if n else 0.0,
        "official_field_rates": {},
        "reason_conditional_field_rates": {},
    }
    for field in ("time", "component", "reason"):
        total = stats["field_totals"][field]
        out["official_field_rates"][field] = stats["official_field_hits"][field] / total if total else None
        out["reason_conditional_field_rates"][field] = stats["reason_conditional_field_hits"][field] / total if total else None
    return out


def main() -> int:
    args = parse_args()
    pred_rows = load_csv(args.pred)
    if pred_rows and "row_id" in pred_rows[0]:
        pred_rows.sort(key=lambda row: int(row["row_id"]))
    query_rows = load_csv(args.query)
    record_rows = load_csv(args.record)
    if not (len(pred_rows) == len(query_rows) == len(record_rows)):
        raise ValueError(f"row mismatch: pred={len(pred_rows)} query={len(query_rows)} record={len(record_rows)}")

    by_time = record_lookup(record_rows)
    rows = []
    by_reason = {}
    by_reason_group = {}
    by_task = {}
    for idx, (pred_row, query_row, record_row) in enumerate(zip(pred_rows, query_rows, record_rows)):
        preds = parse_prediction(pred_row["prediction"])
        truths = parse_truth(query_row["scoring_points"])
        truth_reasons = [reason_for_truth(truth, record_row, by_time) for truth in truths]
        tolerance_reasons = truth_reasons
        official_hits, total, official_score = score_assignment(
            preds,
            truths,
            ["__official_1min__"] * len(truths),
            fixed_tolerance_minutes=1,
        )
        rc_hits, _, rc_score = score_assignment(preds, truths, tolerance_reasons)
        field_totals = {field: sum(1 for truth in truths if field in truth) for field in ("time", "component", "reason")}
        unique_reasons = sorted(set(truth_reasons)) or [record_row["reason"].strip()]
        unique_groups = sorted(set(reason_group(reason) for reason in unique_reasons))
        row_reason = unique_reasons[0] if len(unique_reasons) == 1 else "+".join(unique_reasons)
        row_group = unique_groups[0] if len(unique_groups) == 1 else "mixed"
        row = {
            "row_id": idx,
            "task_index": query_row.get("task_index", ""),
            "gt_component": record_row["component"].strip(),
            "gt_reason": row_reason,
            "truth_reasons": truth_reasons,
            "reason_group": row_group,
            "tolerance_minutes": [TOLERANCE_BY_REASON_MINUTES.get(reason, 5) for reason in truth_reasons],
            "field_totals": field_totals,
            "official_hits": official_hits,
            "reason_conditional_hits": rc_hits,
            "official_score": official_score,
            "reason_conditional_score": rc_score,
            "official_strict": official_score == 1.0,
            "reason_conditional_strict": rc_score == 1.0,
        }
        rows.append(row)
        for table, key in [
            (by_reason, row_reason),
            (by_reason_group, row["reason_group"]),
            (by_task, row["task_index"]),
        ]:
            add_stats(table, key, row)

    overall = init_stats()
    for row in rows:
        add_stats({"overall": overall}, "overall", row)
    report = {
        "n": len(rows),
        "tolerance_by_reason_minutes": TOLERANCE_BY_REASON_MINUTES,
        "reason_groups": {
            "event_like": sorted(EVENT_LIKE),
            "progressive_like": sorted(PROGRESSIVE_LIKE),
            "io_like": ["high disk I/O read usage"],
        },
        "overall": finish_stats(overall),
        "by_reason": {key: finish_stats(value) for key, value in sorted(by_reason.items())},
        "by_reason_group": {key: finish_stats(value) for key, value in sorted(by_reason_group.items())},
        "by_task_index": {key: finish_stats(value) for key, value in sorted(by_task.items())},
        "rows": rows,
    }
    print(json.dumps({
        "overall": report["overall"],
        "by_reason_group": report["by_reason_group"],
        "by_reason": {
            key: {
                "n": value["n"],
                "official_strict_rate": value["official_strict_rate"],
                "reason_conditional_strict_rate": value["reason_conditional_strict_rate"],
                "official_time_rate": value["official_field_rates"]["time"],
                "reason_conditional_time_rate": value["reason_conditional_field_rates"]["time"],
            }
            for key, value in report["by_reason"].items()
        },
    }, ensure_ascii=False, indent=2))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
