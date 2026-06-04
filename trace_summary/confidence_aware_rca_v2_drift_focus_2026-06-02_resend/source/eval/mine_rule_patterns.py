"""Lightweight FP-growth-like evidence pattern mining from v2 signatures."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_log")
    parser.add_argument("--out", default="logs/rule_pattern_mining.json")
    parser.add_argument("--min-support", type=int, default=3)
    return parser.parse_args()


def items_for_case(case: dict) -> set[str]:
    items = set()
    items.add(f"reason:{case.get('reason')}")
    for item in case.get("signature", {}).get("dominant_evidence_types", []):
        items.add(f"evidence:{item}")
    for name, status in case.get("modal_status", {}).items():
        items.add(f"{name}:{status}")
    features = case.get("confidence_features", {})
    if float(features.get("support_strength", 0.0) or 0.0) > 0:
        items.add("has_support")
    if bool(case.get("top1_hit")):
        items.add("top1_hit")
    return items


def main() -> int:
    args = parse_args()
    data = json.load(open(args.case_log, "r", encoding="utf-8"))
    cases = data.get("cases", [])
    single = Counter()
    pair = Counter()
    reason_evidence = defaultdict(Counter)
    for case in cases:
        items = items_for_case(case)
        single.update(items)
        pair.update(tuple(sorted(p)) for p in combinations(items, 2))
        for item in items:
            if item.startswith("evidence:"):
                reason_evidence[case.get("reason")][item] += 1
    frequent_pairs = {
        "|".join(k): v for k, v in pair.items()
        if v >= args.min_support
    }
    report = {
        "n": len(cases),
        "min_support": args.min_support,
        "frequent_items": {k: v for k, v in single.most_common() if v >= args.min_support},
        "frequent_pairs": dict(sorted(frequent_pairs.items(), key=lambda kv: (-kv[1], kv[0]))[:200]),
        "reason_evidence": {k: dict(v.most_common()) for k, v in sorted(reason_evidence.items())},
        "note": "This is pattern mining for rule discovery/audit, not an automatic rule source.",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"n": report["n"], "frequent_item_count": len(report["frequent_items"]), "frequent_pair_count": len(report["frequent_pairs"])}, ensure_ascii=False, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
