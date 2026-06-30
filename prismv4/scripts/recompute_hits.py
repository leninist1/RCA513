#!/usr/bin/env python3
"""Offline re-evaluate hits for existing lightweight result JSONs using
suffix-aware component matching. No LLM calls needed — just re-checks
predicted_component vs expected_component with the same _component_match
logic that run_rcaeval_continuous.py now uses."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_INSTANCE_SUFFIX_RE = re.compile(r"-\d+$")


def _component_match(predicted, expected):
    if predicted is None or expected is None:
        return False
    if predicted == expected:
        return True
    exp_base = _INSTANCE_SUFFIX_RE.sub("", expected)
    pred_base = _INSTANCE_SUFFIX_RE.sub("", predicted)
    return exp_base == pred_base and exp_base != expected


def recompute(path: Path, dry_run: bool = False):
    with open(path) as f:
        data = json.load(f)
    results = data.get("results", [])
    old_hits = sum(1 for c in results if c.get("hit"))
    new_hits = 0
    for c in results:
        pred = c.get("predicted_component")
        exp = c.get("expected_component")
        old = c.get("hit", False)
        new = _component_match(pred, exp)
        c["hit"] = new
        if new:
            new_hits += 1
        if old != new:
            print(f"  {'+' if new else '-'} {c.get('case_id','?'):55s} "
                  f"exp={exp:25s} pred={str(pred):25s} "
                  f"old={old} new={new}")
    total = len(results)
    data["top1_hits"] = new_hits
    data["top1_accuracy"] = new_hits / total if total else 0.0
    print(f"\n{path.name}: {old_hits}/{total} -> {new_hits}/{total} "
          f"({old_hits/total*100:.1f}% -> {new_hits/total*100:.1f}%)")
    if not dry_run:
        out = path.with_suffix(path.suffix)  # same path
        with open(out, "w") as f:
            json.dump(data, f, indent=2, default=str)
        print(f"  Written to {out}")
    return old_hits, new_hits, total


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("files", nargs="+", type=Path,
                    help="result JSON files to re-evaluate")
    ap.add_argument("--dry-run", action="store_true",
                    help="show changes but do not write")
    args = ap.parse_args()

    total_old = total_new = total_n = 0
    for p in args.files:
        if not p.exists():
            print(f"SKIP (not found): {p}")
            continue
        o, n, t = recompute(p, dry_run=args.dry_run)
        total_old += o
        total_new += n
        total_n += t
    if len(args.files) > 1:
        print(f"\n=== TOTAL: {total_old}/{total_n} -> {total_new}/{total_n} "
              f"({total_old/total_n*100:.1f}% -> {total_new/total_n*100:.1f}%) ===")


if __name__ == "__main__":
    sys.exit(main() or 0)