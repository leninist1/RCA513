"""Research-boundary diagnostics for Scheme B v2.

This script checks two questions that are not visible from strict/partial:

1. How much of P(reason | component) can be explained by ground-truth data
   structure alone?
2. Are predicted times near the official <=1min tolerance, or are they far
   enough away to indicate background-anomaly time selection?
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from datetime import datetime
import itertools
import json
import math
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from field_hit_diagnostics import parse_prediction, parse_truth  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", default="/home/yan/workspace/data/openrca/Bank/record.csv")
    parser.add_argument("--query", default="/home/yan/workspace/data/openrca/Bank/query.csv")
    parser.add_argument("--pred", default="logs/openrca_query_v2_trace_batch18_predictions.csv")
    parser.add_argument("--out", default="logs/research_boundary_diag_batch18.json")
    return parser.parse_args()


def entropy(counter: Counter) -> float:
    total = sum(counter.values())
    if total <= 0:
        return 0.0
    out = 0.0
    for count in counter.values():
        p = count / total
        out -= p * math.log2(p)
    return out


def conditional_reason_baseline(pairs: list[tuple[str, str]]) -> dict:
    by_component: dict[str, Counter] = defaultdict(Counter)
    reason_counts = Counter()
    for component, reason in pairs:
        by_component[component][reason] += 1
        reason_counts[reason] += 1
    total = len(pairs)
    majority_hits = sum(max(counts.values()) for counts in by_component.values())
    prior_majority = max(reason_counts.values()) if reason_counts else 0
    conditional_entropy = 0.0
    component_rows = {}
    for component, counts in sorted(by_component.items()):
        n = sum(counts.values())
        majority_reason, majority_count = counts.most_common(1)[0]
        conditional_entropy += (n / total) * entropy(counts) if total else 0.0
        component_rows[component] = {
            "n": n,
            "majority_reason": majority_reason,
            "majority_count": majority_count,
            "majority_rate": majority_count / n if n else 0.0,
            "reason_counts": dict(counts.most_common()),
            "entropy_bits": entropy(counts),
        }
    return {
        "n": total,
        "n_components": len(by_component),
        "reason_counts": dict(reason_counts.most_common()),
        "reason_prior_majority_hit_rate": prior_majority / total if total else 0.0,
        "component_conditional_majority_hit_rate": majority_hits / total if total else 0.0,
        "reason_entropy_bits": entropy(reason_counts),
        "conditional_reason_entropy_bits": conditional_entropy,
        "mutual_information_component_reason_bits": entropy(reason_counts) - conditional_entropy,
        "by_component": component_rows,
    }


def record_pairs(record_path: str) -> list[tuple[str, str]]:
    with open(record_path, newline="", encoding="utf-8") as f:
        return [(row["component"].strip(), row["reason"].strip()) for row in csv.DictReader(f)]


def query_pairs(query_path: str) -> list[tuple[str, str]]:
    out = []
    with open(query_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            truths = parse_truth(row["scoring_points"])
            for truth in truths:
                if "component" in truth and "reason" in truth:
                    out.append((truth["component"], truth["reason"]))
    return out


def parse_time(value: str) -> datetime | None:
    try:
        return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
    except (TypeError, ValueError):
        return None


def best_time_diffs(preds: list[dict], truths: list[dict]) -> list[float]:
    truth_times = [truth["time"] for truth in truths if "time" in truth]
    pred_times = [pred.get("time", "") for pred in preds]
    if not truth_times or len(pred_times) != len(truth_times):
        return []
    best = None
    for perm in itertools.permutations(pred_times):
        diffs = []
        ok = True
        for predicted, expected in zip(perm, truth_times):
            p = parse_time(predicted)
            t = parse_time(expected)
            if p is None or t is None:
                ok = False
                break
            diffs.append((p - t).total_seconds() / 60.0)
        if not ok:
            continue
        if best is None or sum(abs(x) for x in diffs) < sum(abs(x) for x in best):
            best = diffs
    return best or []


def bin_abs_minutes(value: float) -> str:
    x = abs(value)
    if x <= 1:
        return "<=1min"
    if x <= 3:
        return "1-3min"
    if x <= 5:
        return "3-5min"
    if x <= 10:
        return "5-10min"
    if x <= 30:
        return "10-30min"
    return ">30min"


def quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def time_error_report(pred_path: str, query_path: str) -> dict:
    with open(pred_path, newline="", encoding="utf-8") as f:
        pred_rows = list(csv.DictReader(f))
    if pred_rows and "row_id" in pred_rows[0]:
        pred_rows.sort(key=lambda row: int(row["row_id"]))
    with open(query_path, newline="", encoding="utf-8") as f:
        query_rows = list(csv.DictReader(f))
    signed = []
    row_details = []
    for idx, (pred_row, query_row) in enumerate(zip(pred_rows, query_rows)):
        preds = parse_prediction(pred_row["prediction"])
        truths = parse_truth(query_row["scoring_points"])
        diffs = best_time_diffs(preds, truths)
        if not diffs:
            continue
        signed.extend(diffs)
        row_details.append({
            "row_id": idx,
            "task_index": query_row.get("task_index", ""),
            "signed_diff_minutes": diffs,
            "abs_diff_minutes": [abs(x) for x in diffs],
            "max_abs_diff_minutes": max(abs(x) for x in diffs),
        })
    abs_values = [abs(x) for x in signed]
    bins = Counter(bin_abs_minutes(x) for x in signed)
    direction = Counter("early" if x < -1 else "late" if x > 1 else "within_1min" for x in signed)
    return {
        "n_time_items": len(signed),
        "hit_within_1min": sum(1 for x in abs_values if x <= 1),
        "within_1min_rate": sum(1 for x in abs_values if x <= 1) / len(abs_values) if abs_values else 0.0,
        "within_3min_rate": sum(1 for x in abs_values if x <= 3) / len(abs_values) if abs_values else 0.0,
        "within_5min_rate": sum(1 for x in abs_values if x <= 5) / len(abs_values) if abs_values else 0.0,
        "within_10min_rate": sum(1 for x in abs_values if x <= 10) / len(abs_values) if abs_values else 0.0,
        "within_30min_rate": sum(1 for x in abs_values if x <= 30) / len(abs_values) if abs_values else 0.0,
        "abs_diff_bins": dict(sorted(bins.items())),
        "direction_counts": dict(direction.most_common()),
        "abs_diff_summary": {
            "mean": sum(abs_values) / len(abs_values) if abs_values else None,
            "p25": quantile(abs_values, 0.25),
            "median": quantile(abs_values, 0.5),
            "p75": quantile(abs_values, 0.75),
            "p90": quantile(abs_values, 0.9),
            "max": max(abs_values) if abs_values else None,
        },
        "signed_diff_summary": {
            "mean": sum(signed) / len(signed) if signed else None,
            "p25": quantile(signed, 0.25),
            "median": quantile(signed, 0.5),
            "p75": quantile(signed, 0.75),
        },
        "rows": row_details,
    }


def main() -> int:
    args = parse_args()
    report = {
        "record_reason_given_component": conditional_reason_baseline(record_pairs(args.record)),
        "query_reason_given_component": conditional_reason_baseline(query_pairs(args.query)),
        "time_error": time_error_report(args.pred, args.query),
        "notes": [
            "record_reason_given_component uses all Bank record.csv cases.",
            "query_reason_given_component uses only official scoring points that ask for both component and reason.",
            "time_error uses best absolute-time assignment per query, independent of component/reason correctness.",
        ],
    }
    print(json.dumps({
        "record_baseline": {
            "n": report["record_reason_given_component"]["n"],
            "conditional_majority": report["record_reason_given_component"]["component_conditional_majority_hit_rate"],
            "prior_majority": report["record_reason_given_component"]["reason_prior_majority_hit_rate"],
            "mutual_information_bits": report["record_reason_given_component"]["mutual_information_component_reason_bits"],
        },
        "query_baseline": {
            "n": report["query_reason_given_component"]["n"],
            "conditional_majority": report["query_reason_given_component"]["component_conditional_majority_hit_rate"],
            "prior_majority": report["query_reason_given_component"]["reason_prior_majority_hit_rate"],
            "mutual_information_bits": report["query_reason_given_component"]["mutual_information_component_reason_bits"],
        },
        "time_error": {
            k: v for k, v in report["time_error"].items()
            if k not in {"rows"}
        },
    }, ensure_ascii=False, indent=2))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
