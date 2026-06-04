"""Summarize blind spots and no-evidence cases from v2 case outputs."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_log")
    parser.add_argument("--out", default="logs/blind_spot_report.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data = json.load(open(args.case_log, "r", encoding="utf-8"))
    cases = data.get("cases", [])
    modality = Counter()
    no_evidence = Counter()
    by_reason = defaultdict(lambda: {"n": 0, "no_evidence": 0, "top1": 0, "top3": 0})
    for case in cases:
        for name, status in case.get("modal_status", {}).items():
            if status != "present":
                modality[f"{name}:{status}"] += 1
        support = float(case.get("confidence_features", {}).get("support_strength", 0.0) or 0.0)
        reason = case.get("reason", "")
        by_reason[reason]["n"] += 1
        by_reason[reason]["top1"] += int(bool(case.get("top1_hit")))
        by_reason[reason]["top3"] += int(bool(case.get("top3_hit")))
        if support <= 0:
            no_evidence[reason] += 1
            by_reason[reason]["no_evidence"] += 1
    report = {
        "n": len(cases),
        "missing_modality_counts": dict(modality),
        "no_evidence_by_reason": dict(no_evidence),
        "by_reason": {
            k: {
                **v,
                "no_evidence_rate": v["no_evidence"] / v["n"] if v["n"] else 0.0,
                "top1_rate": v["top1"] / v["n"] if v["n"] else 0.0,
                "top3_rate": v["top3"] / v["n"] if v["n"] else 0.0,
            }
            for k, v in sorted(by_reason.items())
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"n": report["n"], "missing_modality_counts": report["missing_modality_counts"], "no_evidence_by_reason": report["no_evidence_by_reason"]}, ensure_ascii=False, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
