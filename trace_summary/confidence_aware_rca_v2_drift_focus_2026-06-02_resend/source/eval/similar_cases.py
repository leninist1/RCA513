"""Signature nearest-neighbor analysis for v2 outputs.

This is analysis-only; it is not part of the decision core.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_log")
    parser.add_argument("--out", default="logs/similar_cases.json")
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def feature_set(case: dict) -> set[str]:
    out = set(case.get("signature", {}).get("dominant_evidence_types", []))
    for name, status in case.get("modal_status", {}).items():
        out.add(f"{name}:{status}")
    reason = case.get("reason")
    if reason:
        out.add(f"reason:{reason}")
    return out


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def main() -> int:
    args = parse_args()
    data = json.load(open(args.case_log, "r", encoding="utf-8"))
    cases = data.get("cases", [])
    features = [feature_set(case) for case in cases]
    rows = []
    for i, case in enumerate(cases):
        sims = []
        for j, other in enumerate(cases):
            if i == j:
                continue
            sims.append((jaccard(features[i], features[j]), j, other))
        sims.sort(key=lambda x: (-x[0], x[1]))
        rows.append({
            "index": i,
            "datetime": case.get("datetime"),
            "reason": case.get("reason"),
            "true_component": case.get("true_component"),
            "top1_hit": case.get("top1_hit"),
            "neighbors": [
                {
                    "score": score,
                    "index": j,
                    "datetime": other.get("datetime"),
                    "reason": other.get("reason"),
                    "true_component": other.get("true_component"),
                    "top1_hit": other.get("top1_hit"),
                }
                for score, j, other in sims[:args.top_k]
            ],
        })
    report = {
        "n": len(cases),
        "top_k": args.top_k,
        "rows": rows,
        "note": "Nearest neighbors are for analysis/possible future optimization, not current RCA decisions.",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"n": len(cases), "top_k": args.top_k}, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
