"""Attribution diagnostics for near vs far time errors.

The goal is to separate two failure modes:

1. near misses: the model selects an anomaly close to the ground-truth anchor;
2. far misses: the model selects a different anomaly cluster inside the query
   window.

For each group, the report summarizes reason distribution, task/query shape,
time-of-day distribution, and a metric-based background-anomaly proxy.
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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = PROJECT_ROOT.parent
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (PROJECT_ROOT, REPO_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from field_hit_diagnostics import parse_prediction, parse_truth  # noqa: E402
from reason_conditional_tolerance_eval import reason_group  # noqa: E402
from run_openrca_query_v2 import (  # noqa: E402
    DayCache,
    gather_window,
    metric_seeds,
    parse_modalities,
)
from refute.src.baseline_distributions import BaselineStore  # noqa: E402
from refute.src.data_loader import BankDataPaths  # noqa: E402
from refute_b_v2.query_windows import parse_query_window  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred", default="logs/openrca_query_v2_trace_batch18_predictions.csv")
    parser.add_argument("--query", default="/home/yan/workspace/data/openrca/Bank/query.csv")
    parser.add_argument("--record", default="/home/yan/workspace/data/openrca/Bank/record.csv")
    parser.add_argument("--debug-json", default="logs/openrca_query_v2_trace_batch18_debug.json")
    parser.add_argument("--data-root", default="/home/yan/workspace/data/openrca/Bank")
    parser.add_argument("--baseline", default="../refute/knowledge/baseline_distributions_all_metric_dates.json")
    parser.add_argument("--out", default="logs/time_bimodal_attribution_batch18.json")
    return parser.parse_args()


def load_csv(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def record_lookup(record_rows: list[dict]) -> dict[str, dict]:
    return {row["datetime"].strip(): row for row in record_rows}


def parse_time(value: str) -> datetime | None:
    try:
        return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
    except (TypeError, ValueError):
        return None


def best_time_assignments(preds: list[dict], truths: list[dict]) -> list[tuple[dict, dict, float]]:
    time_truths = [truth for truth in truths if "time" in truth]
    if not time_truths or len(preds) != len(time_truths):
        return []
    best = None
    best_abs_sum = None
    for perm in itertools.permutations(preds):
        assignments = []
        ok = True
        for truth, pred in zip(time_truths, perm):
            expected = parse_time(truth.get("time", ""))
            predicted = parse_time(pred.get("time", ""))
            if expected is None or predicted is None:
                ok = False
                break
            diff = (predicted - expected).total_seconds() / 60.0
            assignments.append((truth, pred, diff))
        if not ok:
            continue
        abs_sum = sum(abs(item[2]) for item in assignments)
        if best_abs_sum is None or abs_sum < best_abs_sum:
            best_abs_sum = abs_sum
            best = assignments
    return best or []


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


def numeric_summary(values: list[float]) -> dict:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return {
        "n": len(vals),
        "mean": sum(vals) / len(vals) if vals else None,
        "median": quantile(vals, 0.5),
        "p75": quantile(vals, 0.75),
        "p90": quantile(vals, 0.9),
        "max": max(vals) if vals else None,
    }


def error_group(abs_minutes: float) -> str:
    if abs_minutes <= 5:
        return "near_le_5min"
    if abs_minutes <= 10:
        return "mid_5_10min"
    if abs_minutes <= 30:
        return "far_10_30min"
    return "out_gt_30min"


def hour_bucket(hour: int) -> str:
    if 0 <= hour < 6:
        return "00-06"
    if 6 <= hour < 12:
        return "06-12"
    if 12 <= hour < 18:
        return "12-18"
    return "18-24"


def required_pattern(truths: list[dict]) -> str:
    fields = sorted({field for truth in truths for field in truth})
    return "+".join(fields)


def first_prediction(preds: list[dict]) -> dict:
    return preds[0] if preds else {"time": "", "component": "", "reason": ""}


def load_debug(path: str) -> dict[int, dict]:
    p = Path(path)
    if not p.exists():
        return {}
    data = json.loads(p.read_text(encoding="utf-8"))
    return {int(row["row_id"]): row for row in data.get("debug", [])}


def metric_background_features(cache: DayCache, baseline: BaselineStore, window, gt_component: str, pred_component: str) -> dict:
    metric_df, _ = gather_window(cache, window, parse_modalities("metric"))
    seeds = metric_seeds(metric_df, baseline)
    root_scores = [seed.score for seed in seeds if seed.component == gt_component]
    pred_scores = [seed.score for seed in seeds if seed.component == pred_component]
    nonroot_scores = [seed.score for seed in seeds if seed.component != gt_component]
    root_max = max(root_scores) if root_scores else 0.0
    pred_max = max(pred_scores) if pred_scores else 0.0
    nonroot_max = max(nonroot_scores) if nonroot_scores else 0.0
    return {
        "metric_seed_count": len(seeds),
        "root_metric_max": root_max,
        "pred_metric_max": pred_max,
        "nonroot_metric_max": nonroot_max,
        "nonroot_minus_root": nonroot_max - root_max,
        "nonroot_over_root": nonroot_max / root_max if root_max > 0 else None,
        "metric_winner_is_root": root_max >= nonroot_max and root_max > 0,
    }


def summarize_group(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0}
    out = {
        "n": len(rows),
        "reason_counts": dict(Counter(row["gt_reason"] for row in rows).most_common()),
        "reason_group_counts": dict(Counter(row["reason_group"] for row in rows).most_common()),
        "task_counts": dict(Counter(row["task_index"] for row in rows).most_common()),
        "required_pattern_counts": dict(Counter(row["required_pattern"] for row in rows).most_common()),
        "date_counts": dict(Counter(row["date"] for row in rows).most_common()),
        "hour_bucket_counts": dict(Counter(row["hour_bucket"] for row in rows).most_common()),
        "selected_source_counts": dict(Counter(row.get("selected_seed_source", "") for row in rows).most_common()),
        "abs_diff_minutes": numeric_summary([row["abs_diff_minutes"] for row in rows]),
        "signed_diff_minutes": numeric_summary([row["signed_diff_minutes"] for row in rows]),
        "selected_score": numeric_summary([row.get("selected_score", 0.0) for row in rows]),
        "metric_seed_count": numeric_summary([row["metric_seed_count"] for row in rows]),
        "root_metric_max": numeric_summary([row["root_metric_max"] for row in rows]),
        "nonroot_metric_max": numeric_summary([row["nonroot_metric_max"] for row in rows]),
        "nonroot_minus_root": numeric_summary([row["nonroot_minus_root"] for row in rows]),
        "nonroot_over_root": numeric_summary([row["nonroot_over_root"] for row in rows if row["nonroot_over_root"] is not None]),
        "metric_winner_is_root_rate": sum(1 for row in rows if row["metric_winner_is_root"]) / len(rows),
        "pred_component_is_gt_rate": sum(1 for row in rows if row["pred_component"] == row["gt_component"]) / len(rows),
        "pred_reason_is_gt_rate": sum(1 for row in rows if row["pred_reason"] == row["gt_reason"]) / len(rows),
    }
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

    paths = BankDataPaths.from_root(args.data_root)
    cache = DayCache(paths)
    baseline = BaselineStore.load_json(args.baseline)
    debug_rows = load_debug(args.debug_json)

    rows = []
    by_time = record_lookup(record_rows)
    for idx, (pred_row, query_row, record_row) in enumerate(zip(pred_rows, query_rows, record_rows)):
        preds = parse_prediction(pred_row["prediction"])
        truths = parse_truth(query_row["scoring_points"])
        assignments = best_time_assignments(preds, truths)
        if not assignments:
            continue
        window = parse_query_window(query_row["instruction"])
        selected = (debug_rows.get(idx, {}).get("selected") or [{}])[0]
        selected_seed = selected.get("seed", {})
        for item_idx, (truth, pred, signed) in enumerate(assignments):
            gt_time = truth.get("time", "")
            gt_record = by_time.get(gt_time, record_row)
            gt_dt = datetime.strptime(gt_time, "%Y-%m-%d %H:%M:%S")
            gt_component = truth.get("component") or gt_record["component"].strip()
            gt_reason = truth.get("reason") or gt_record["reason"].strip()
            features = metric_background_features(cache, baseline, window, gt_component, pred.get("component", ""))
            abs_diff = abs(signed)
            row = {
                "row_id": idx,
                "time_item_index": item_idx,
                "task_index": query_row.get("task_index", ""),
                "required_pattern": required_pattern(truths),
                "date": gt_dt.strftime("%Y_%m_%d"),
                "hour": gt_dt.hour,
                "hour_bucket": hour_bucket(gt_dt.hour),
                "gt_time": gt_time,
                "pred_time": pred.get("time", ""),
                "signed_diff_minutes": signed,
                "abs_diff_minutes": abs_diff,
                "error_group": error_group(abs_diff),
                "gt_component": gt_component,
                "pred_component": pred.get("component", ""),
                "gt_reason": gt_reason,
                "pred_reason": pred.get("reason", ""),
                "reason_group": reason_group(gt_reason),
                "selected_score": selected.get("score", 0.0),
                "selected_seed_source": selected_seed.get("source", ""),
                **features,
            }
            rows.append(row)

    by_group = defaultdict(list)
    by_reason_group = defaultdict(list)
    for row in rows:
        by_group[row["error_group"]].append(row)
        by_reason_group[row["reason_group"]].append(row)

    report = {
        "n_time_rows": len(rows),
        "group_definitions": {
            "near_le_5min": "absolute time error <= 5 minutes",
            "mid_5_10min": "5 < absolute time error <= 10 minutes",
            "far_10_30min": "10 < absolute time error <= 30 minutes",
            "out_gt_30min": "absolute time error > 30 minutes",
        },
        "overall": summarize_group(rows),
        "by_error_group": {key: summarize_group(value) for key, value in sorted(by_group.items())},
        "by_reason_group": {key: summarize_group(value) for key, value in sorted(by_reason_group.items())},
        "rows": rows,
        "notes": [
            "Background anomaly strength is approximated using metric anomaly seed scores in the query window.",
            "root_metric_max is the strongest metric seed on the ground-truth component.",
            "nonroot_metric_max is the strongest metric seed on any other component in the same query window.",
        ],
    }
    print(json.dumps({
        "n_time_rows": report["n_time_rows"],
        "by_error_group": {
            key: {
                "n": value["n"],
                "reason_group_counts": value.get("reason_group_counts"),
                "reason_counts": value.get("reason_counts"),
                "metric_winner_is_root_rate": value.get("metric_winner_is_root_rate"),
                "pred_component_is_gt_rate": value.get("pred_component_is_gt_rate"),
                "root_metric_max_mean": (value.get("root_metric_max") or {}).get("mean"),
                "nonroot_metric_max_mean": (value.get("nonroot_metric_max") or {}).get("mean"),
                "nonroot_minus_root_mean": (value.get("nonroot_minus_root") or {}).get("mean"),
            }
            for key, value in report["by_error_group"].items()
        },
        "by_reason_group": {
            key: {
                "n": value["n"],
                "abs_diff_median": (value.get("abs_diff_minutes") or {}).get("median"),
                "abs_diff_p75": (value.get("abs_diff_minutes") or {}).get("p75"),
            }
            for key, value in report["by_reason_group"].items()
        },
    }, ensure_ascii=False, indent=2))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
