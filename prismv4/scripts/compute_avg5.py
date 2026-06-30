#!/usr/bin/env python3
"""Compute AC@k and Avg@5 metrics from lightweight result JSONs.

For cases that store ``predicted_ranking`` (a top-5 list), we compute exact
AC@k. For older results that only have ``predicted_component`` (top-1), we
approximate: AC@1 is exact; AC@2..5 fall back to recall_pool membership
since the full ranking wasn't stored.

Avg@5 = (AC@1 + AC@2 + AC@3 + AC@4 + AC@5) / 5
"""
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


def compute_metrics(path: Path):
    with open(path) as f:
        data = json.load(f)
    results = data.get("results", [])
    n = len(results)
    if n == 0:
        print(f"{path.name}: no results")
        return

    ac = [0] * 6  # ac[k] = count of hits at rank k (1-indexed)
    avg5_sum = 0.0

    for c in results:
        exp = c.get("expected_component")
        ranking = c.get("predicted_ranking")
        pred = c.get("predicted_component")
        pool = c.get("recall_pool", [])

        if ranking and isinstance(ranking, list) and len(ranking) > 0:
            # Exact ranking available
            hit_at = 0
            for k, comp in enumerate(ranking[:5], 1):
                if _component_match(comp, exp):
                    hit_at = k
                    break
        else:
            # Approximate: top-1 exact, rest from recall_pool
            hit_at = 1 if _component_match(pred, exp) else 0
            if hit_at == 0 and pool:
                for k, comp in enumerate(pool[:5], 1):
                    if k == 1:
                        continue  # already checked via pred
                    if _component_match(comp, exp):
                        hit_at = k
                        break

        if hit_at > 0:
            for k in range(hit_at, 6):
                ac[k] += 1
            # Avg@5 contribution = (1 + 1 + ... + 0 + ...) / 5
            # AC@j = 1 for j >= hit_at, 0 for j < hit_at
            avg5_sum += sum(1.0 for j in range(1, 6) if j >= hit_at) / 5.0

    print(f"\n{'=' * 60}")
    print(f"{path.name}  (n={n})")
    print(f"  AC@1 = {ac[1]/n:.4f}  ({ac[1]}/{n})")
    print(f"  AC@3 = {ac[3]/n:.4f}  ({ac[3]}/{n})")
    print(f"  AC@5 = {ac[5]/n:.4f}  ({ac[5]}/{n})")
    print(f"  Avg@5 = {avg5_sum/n:.4f}")
    print(f"  Top-1 (stored) = {data.get('top1_accuracy', 0):.4f}")
    return {
        "file": path.name,
        "n": n,
        "ac1": ac[1] / n,
        "ac3": ac[3] / n,
        "ac5": ac[5] / n,
        "avg5": avg5_sum / n,
        "top1_stored": data.get("top1_accuracy", 0),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("files", nargs="+", type=Path)
    args = ap.parse_args()
    all_metrics = []
    for p in args.files:
        if not p.exists():
            print(f"SKIP: {p}")
            continue
        m = compute_metrics(p)
        if m:
            all_metrics.append(m)

    if len(all_metrics) > 1:
        total_n = sum(m["n"] for m in all_metrics)
        total_ac1 = sum(m["ac1"] * m["n"] for m in all_metrics) / total_n
        total_ac3 = sum(m["ac3"] * m["n"] for m in all_metrics) / total_n
        total_ac5 = sum(m["ac5"] * m["n"] for m in all_metrics) / total_n
        total_avg5 = sum(m["avg5"] * m["n"] for m in all_metrics) / total_n
        print(f"\n{'=' * 60}")
        print(f"TOTAL  (n={total_n})")
        print(f"  AC@1 = {total_ac1:.4f}")
        print(f"  AC@3 = {total_ac3:.4f}")
        print(f"  AC@5 = {total_ac5:.4f}")
        print(f"  Avg@5 = {total_avg5:.4f}")

        # Print summary table
        print(f"\n{'Dataset':<35s} {'n':>4s} {'AC@1':>8s} {'AC@3':>8s} {'AC@5':>8s} {'Avg@5':>8s}")
        print("-" * 75)
        for m in all_metrics:
            print(f"{m['file']:<35s} {m['n']:>4d} {m['ac1']:>8.4f} {m['ac3']:>8.4f} "
                  f"{m['ac5']:>8.4f} {m['avg5']:>8.4f}")
        print("-" * 75)
        print(f"{'TOTAL':<35s} {total_n:>4d} {total_ac1:>8.4f} {total_ac3:>8.4f} "
              f"{total_ac5:>8.4f} {total_avg5:>8.4f}")


if __name__ == "__main__":
    sys.exit(main() or 0)