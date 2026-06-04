"""Forward cross-date confidence holdout for v2 case outputs.

This is stricter than leave-one-date-out: calibrate on earlier dates and test
on later date(s), which better mimics deployment on future incidents.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from leave_one_date_out_calibration import aggregate_labeled, bucket_key, label_from_training, table_for  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_log")
    parser.add_argument("--out", default="logs/cross_date_holdout_calibration.json")
    parser.add_argument("--min-n", type=int, default=5)
    parser.add_argument("--test-date", default="", help="Specific held-out date. Default: last date.")
    parser.add_argument("--test-last-n", type=int, default=1, help="Hold out the last N dates when --test-date is not set.")
    return parser.parse_args()


def summarize_coverage(labels: dict, total_n: int) -> dict:
    out = {}
    for label, row in labels.items():
        out[label] = {
            **row,
            "coverage": row["n"] / total_n if total_n else 0.0,
        }
    return out


def main() -> int:
    args = parse_args()
    data = json.load(open(args.case_log, "r", encoding="utf-8"))
    cases = data.get("cases", [])
    dates = sorted({case["date"] for case in cases})
    if not dates:
        raise ValueError("no dated cases found")
    if args.test_date:
        test_dates = [args.test_date]
    else:
        test_dates = dates[-max(1, args.test_last_n):]
    first_test_date = min(test_dates)
    train_dates = [date for date in dates if date < first_test_date]
    if not train_dates:
        raise ValueError(f"no earlier train dates for test_dates={test_dates}")

    train = [case for case in cases if case["date"] in train_dates]
    test = [case for case in cases if case["date"] in test_dates]
    table = table_for(train)

    labeled = []
    for case in test:
        key = bucket_key(case)
        labeled.append({
            **case,
            "bucket_key": key,
            "predicted_confidence": label_from_training(table.get(key), args.min_n),
            "training_bucket": table.get(key),
        })
    labels = aggregate_labeled(labeled)
    report = {
        "n_train": len(train),
        "n_test": len(test),
        "dates": dates,
        "train_dates": train_dates,
        "test_dates": test_dates,
        "min_n": args.min_n,
        "labels": summarize_coverage(labels, len(test)),
        "cases": [
            {
                "date": row.get("date"),
                "datetime": row.get("datetime"),
                "reason": row.get("reason"),
                "true_component": row.get("true_component"),
                "top1_hit": row.get("top1_hit"),
                "top3_hit": row.get("top3_hit"),
                "bucket_key": row["bucket_key"],
                "predicted_confidence": row["predicted_confidence"],
                "training_bucket": row.get("training_bucket"),
            }
            for row in labeled
        ],
        "note": "Confidence labels are calibrated on dates strictly earlier than the held-out test date(s).",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "n_train": report["n_train"],
        "n_test": report["n_test"],
        "train_dates": train_dates,
        "test_dates": test_dates,
        "labels": report["labels"],
    }, ensure_ascii=False, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
