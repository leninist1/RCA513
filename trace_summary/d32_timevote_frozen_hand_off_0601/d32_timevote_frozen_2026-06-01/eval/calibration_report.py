"""Build an initial confidence calibration report from v2 case outputs."""
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
    parser.add_argument("--out", default="")
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
    reason_bucket = normalize_reason_bucket(case.get("reason", ""))
    return "|".join([reason_bucket, evidence_state(case), gap_bin(case)])


def label_bucket(n: int, top1_rate: float, min_n: int) -> str:
    if n < min_n:
        return "medium_sample_low"
    if top1_rate >= 0.80:
        return "high"
    if top1_rate >= 0.40:
        return "medium"
    return "low"


def summarize(cases: list[dict], min_n: int) -> dict:
    buckets = defaultdict(lambda: {"n": 0, "top1": 0, "top3": 0})
    by_reason = defaultdict(lambda: {"n": 0, "top1": 0, "top3": 0})
    by_state = defaultdict(lambda: {"n": 0, "top1": 0, "top3": 0})
    for case in cases:
        hit1 = int(bool(case.get("top1_hit")))
        hit3 = int(bool(case.get("top3_hit")))
        for table, key in [
            (buckets, bucket_key(case)),
            (by_reason, normalize_reason_bucket(case.get("reason", ""))),
            (by_state, evidence_state(case)),
        ]:
            table[key]["n"] += 1
            table[key]["top1"] += hit1
            table[key]["top3"] += hit3
    def finish(table):
        out = {}
        for key, row in sorted(table.items()):
            n = row["n"]
            top1_rate = row["top1"] / n if n else 0.0
            top3_rate = row["top3"] / n if n else 0.0
            out[key] = {
                **row,
                "top1_rate": top1_rate,
                "top3_rate": top3_rate,
                "confidence_label": label_bucket(n, top1_rate, min_n),
            }
        return out
    bucket_stats = finish(buckets)
    labeled_cases = []
    for case in cases:
        key = bucket_key(case)
        labeled_cases.append({
            "date": case.get("date"),
            "datetime": case.get("datetime"),
            "reason": case.get("reason"),
            "true_component": case.get("true_component"),
            "top5": case.get("top5", [])[:5],
            "top1_hit": case.get("top1_hit"),
            "top3_hit": case.get("top3_hit"),
            "bucket_key": key,
            "confidence_label": bucket_stats[key]["confidence_label"],
            "confidence_features": case.get("confidence_features", {}),
        })
    return {
        "n": len(cases),
        "min_n": min_n,
        "buckets": bucket_stats,
        "by_reason_bucket": finish(by_reason),
        "by_evidence_state": finish(by_state),
        "cases": labeled_cases,
        "note": "This is in-sample calibration. Use leave-one-date-out before reporting as final.",
    }


def main() -> int:
    args = parse_args()
    data = json.load(open(args.case_log, "r", encoding="utf-8"))
    report = summarize(data.get("cases", []), args.min_n)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(f"wrote {out}")
    print(json.dumps({
        "n": report["n"],
        "by_evidence_state": report["by_evidence_state"],
        "high_buckets": {k: v for k, v in report["buckets"].items() if v["confidence_label"] == "high"},
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
