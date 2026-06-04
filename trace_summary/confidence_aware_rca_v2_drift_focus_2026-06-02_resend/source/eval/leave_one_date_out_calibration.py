"""Leave-one-date-out confidence calibration for v2 case outputs."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from refute_b_v2.rules import normalize_reason_bucket  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_log")
    parser.add_argument("--out", default="logs/lodo_calibration.json")
    parser.add_argument("--min-n", type=int, default=5)
    return parser.parse_args()


def evidence_state(case: dict) -> str:
    features = case.get("confidence_features", {})
    support = float(features.get("support_strength", 0.0) or 0.0)
    blind = int(features.get("blind_count", 0) or 0)
    if support <= 0:
        return "no_evidence"
    if blind > 0:
        return "partial_evidence"
    return "evidence"


def gap_bin(case: dict) -> str:
    gap = float(case.get("confidence_features", {}).get("top1_top2_gap", 0.0) or 0.0)
    if gap >= 10:
        return "gap_high"
    if gap >= 3:
        return "gap_mid"
    if gap > 0:
        return "gap_low"
    return "gap_none"


def bucket_key(case: dict) -> str:
    return "|".join([
        normalize_reason_bucket(case.get("reason", "")),
        evidence_state(case),
        gap_bin(case),
    ])


def table_for(cases: list[dict]) -> dict[str, dict]:
    table = defaultdict(lambda: {"n": 0, "top1": 0, "top3": 0})
    for case in cases:
        row = table[bucket_key(case)]
        row["n"] += 1
        row["top1"] += int(bool(case.get("top1_hit")))
        row["top3"] += int(bool(case.get("top3_hit")))
    out = {}
    for key, row in table.items():
        n = row["n"]
        out[key] = {
            **row,
            "top1_rate": row["top1"] / n if n else 0.0,
            "top3_rate": row["top3"] / n if n else 0.0,
        }
    return out


def label_from_training(row: dict | None, min_n: int) -> str:
    if row is None:
        return "unknown"
    if row["n"] < min_n:
        return "unknown"
    if row["top1_rate"] >= 0.80:
        return "high"
    if row["top1_rate"] >= 0.40:
        return "medium"
    return "low"


def aggregate_labeled(cases: list[dict]) -> dict:
    labels = defaultdict(lambda: {"n": 0, "top1": 0, "top3": 0})
    for case in cases:
        label = case["predicted_confidence"]
        row = labels[label]
        row["n"] += 1
        row["top1"] += int(bool(case.get("top1_hit")))
        row["top3"] += int(bool(case.get("top3_hit")))
    return {
        label: {
            **row,
            "top1_rate": row["top1"] / row["n"] if row["n"] else 0.0,
            "top3_rate": row["top3"] / row["n"] if row["n"] else 0.0,
        }
        for label, row in sorted(labels.items())
    }


def main() -> int:
    args = parse_args()
    data = json.load(open(args.case_log, "r", encoding="utf-8"))
    cases = data.get("cases", [])
    dates = sorted({case["date"] for case in cases})
    folds = []
    labeled_all = []
    for heldout in dates:
        train = [case for case in cases if case["date"] != heldout]
        test = [case for case in cases if case["date"] == heldout]
        table = table_for(train)
        labeled = []
        for case in test:
            key = bucket_key(case)
            label = label_from_training(table.get(key), args.min_n)
            row = {
                **case,
                "bucket_key": key,
                "predicted_confidence": label,
                "training_bucket": table.get(key),
            }
            labeled.append(row)
            labeled_all.append(row)
        folds.append({
            "heldout_date": heldout,
            "n": len(test),
            "labels": aggregate_labeled(labeled),
            "cases": [
                {
                    "datetime": row.get("datetime"),
                    "reason": row.get("reason"),
                    "true_component": row.get("true_component"),
                    "top5": row.get("top5", [])[:5],
                    "top1_hit": row.get("top1_hit"),
                    "top3_hit": row.get("top3_hit"),
                    "bucket_key": row["bucket_key"],
                    "predicted_confidence": row["predicted_confidence"],
                }
                for row in labeled
            ],
        })
    report = {
        "n": len(cases),
        "dates": dates,
        "min_n": args.min_n,
        "overall_lodo": aggregate_labeled(labeled_all),
        "folds": folds,
        "note": "Confidence labels are learned from all dates except the held-out date.",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "n": report["n"],
        "overall_lodo": report["overall_lodo"],
    }, ensure_ascii=False, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
