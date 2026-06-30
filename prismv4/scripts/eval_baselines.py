"""Compute AC@1/3/5 and Avg@5 for RCAEval baseline results.

Reads per-case JSONs from `baseline_results/{method}_{dataset}/results/*.json`.
Each JSON is either {"0": ["svc_metric", ...]} (success) or {"error": "..."} (fail).

For each case, extracts top-k service names from ranked metric list:
  - service = rank_entity.split("_")[0] (matches main.py convention)
  - removes "-db" suffix
  - deduplicates service list

Expected component is derived from filename: `{service}_{fault}_{case}.json` -> service.

Output: prints per-dataset and per-method AC@1/3/5 + Avg@5 table.
"""
import argparse
import glob
import json
import os
from collections import defaultdict
from os.path import basename, join


def _service_from_rank(ranked_entity: str) -> str:
    s = ranked_entity.split("_")[0].replace("-db", "")
    return s


def _dedup_services(ranks):
    seen = []
    for r in ranks:
        s = _service_from_rank(r)
        if s not in seen:
            seen.append(s)
    return seen


def evaluate_dir(results_dir, expected_from="filename"):
    """
    expected_from='filename': parse expected service from filename {service}_{fault}_{case}.json
    expected_from='json':    use 'expected' key stored in JSON (AIOps2021 case)
    Returns dict: {n_total, n_evaluable, AC@1, AC@3, AC@5, Avg@5}
    """
    files = sorted(glob.glob(join(results_dir, "*.json")))
    if not files:
        return None

    ac = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    n_ok = 0
    n_fail = 0
    for f in files:
        try:
            with open(f) as fh:
                d = json.load(fh)
        except Exception:
            n_fail += 1
            continue
        if "error" in d and "0" not in d:
            n_fail += 1
            continue
        ranks = d.get("0", [])
        if not ranks:
            n_fail += 1
            continue

        if expected_from == "json" and "expected" in d:
            expected = d["expected"]
        else:
            fname = basename(f).rsplit(".json", 1)[0]
            parts = fname.split("_")
            if fname.startswith("aiops2021_"):
                expected = parts[-1]
            else:
                expected = parts[0]

        services = _dedup_services(ranks)
        for k in (1, 2, 3, 4, 5):
            if expected in services[:k]:
                ac[k] += 1
        n_ok += 1

    if n_ok == 0:
        return None
    avg5 = sum(ac[k] for k in (1, 2, 3, 4, 5)) / (5 * n_ok)
    return {
        "n_total": len(files),
        "n_evaluable": n_ok,
        "n_fail": n_fail,
        "AC@1": ac[1] / n_ok,
        "AC@3": ac[3] / n_ok,
        "AC@5": ac[5] / n_ok,
        "Avg@5": avg5,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/dell2/RCA513/yyx/prismv4/results/baseline_results")
    ap.add_argument("--expected_from", default="filename",
                    choices=["filename", "json"],
                    help="auto-detect per-dir; override for special formats")
    args = ap.parse_args()

    method_dirs = sorted([d for d in os.listdir(args.root)
                          if os.path.isdir(join(args.root, d))])

    # group by method and dataset
    print(f"\n{'Method-Dataset':<30s} {'n':>4s} {'eval':>4s} {'AC@1':>7s} {'AC@3':>7s} {'AC@5':>7s} {'Avg@5':>7s}")
    print("-" * 75)

    by_method_dataset = {}
    for md in method_dirs:
        rd = join(args.root, md, "results")
        if not os.path.isdir(rd):
            continue
        # auto-detect AIOps2021 -> json
        exp_from = "json" if "aiops2021" in md.lower() else "filename"
        r = evaluate_dir(rd, expected_from=exp_from)
        if r is None:
            continue
        by_method_dataset[md] = r
        print(f"{md:<30s} {r['n_total']:>4d} {r['n_evaluable']:>4d} "
              f"{r['AC@1']:>7.4f} {r['AC@3']:>7.4f} {r['AC@5']:>7.4f} {r['Avg@5']:>7.4f}")

    # Group by method: aggregate across all its datasets
    print("\n--- Aggregated by method ---")
    by_method = defaultdict(lambda: {"n": 0, "ac1": 0, "ac3": 0, "ac5": 0, "n_eval": 0})
    for md, r in by_method_dataset.items():
        method = md.rsplit("_", 1)[0] if "_" in md else md
        # actually split: method_dataset -> for re2-tt etc., the dataset is the suffix
        # but we need careful split: baro_re2-tt -> method=baro, dataset=re2-tt
        parts = md.split("_", 1)
        if len(parts) == 2 and parts[1].startswith(("re2", "re3", "aiops2021")):
            method = parts[0]
        else:
            method = md
        by_method[method]["n"] += r["n_total"]
        by_method[method]["n_eval"] += r["n_evaluable"]
        by_method[method]["ac1"] += r["AC@1"] * r["n_evaluable"]
        by_method[method]["ac3"] += r["AC@3"] * r["n_evaluable"]
        by_method[method]["ac5"] += r["AC@5"] * r["n_evaluable"]
    for method, s in sorted(by_method.items()):
        n = s["n_eval"] or 1
        print(f"{method:<20s} (n_eval={s['n_eval']}/{s['n']}): "
              f"AC@1={s['ac1']/n:.4f} AC@3={s['ac3']/n:.4f} AC@5={s['ac5']/n:.4f} "
              f"Avg@5={(s['ac1']+s['ac3']+s['ac5'])/(3*n):.4f}")


if __name__ == "__main__":
    main()