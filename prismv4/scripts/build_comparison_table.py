"""Build consolidated comparison table: IVD vs all baselines on all datasets.

Outputs:
- Table 1: RE2 three subsets × all methods
- Table 2: RE3 three subsets × all methods
- Table 3: AIOps2021 × non-trace methods
"""
import glob
import json
import os
import re
from collections import defaultdict
from os.path import basename, join


BASELINE_ROOT = "/home/dell2/RCA513/yyx/prismv4/results/baseline_results"
IVD_ROOT = "/home/dell2/RCA513/yyx/prismv4/results/prism_cht"

_INSTANCE_RE = re.compile(r"-\d+$")


def _service_from_rank(ranked_entity: str) -> str:
    return ranked_entity.split("_")[0].replace("-db", "")


def _ivd_normalize(s):
    if s is None:
        return None
    # for OpenRCA suffix-style labels
    return _INSTANCE_RE.sub("", str(s))


def _dedup_services(ranks):
    seen = []
    for r in ranks:
        s = _service_from_rank(r)
        if s not in seen:
            seen.append(s)
    return seen


def eval_baseline_dir(results_dir, expected_from="filename"):
    files = sorted(glob.glob(join(results_dir, "*.json")))
    if not files:
        return None
    ac = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    n_ok = 0
    n_fail = 0
    for f in files:
        try:
            d = json.load(open(f))
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
        if expected_from == "json":
            expected = d.get("expected")
        else:
            fname = basename(f).rsplit(".json", 1)[0]
            parts = fname.split("_")
            expected = parts[-1] if fname.startswith("aiops2021_") else parts[0]
        services = _dedup_services(ranks)
        for k in (1, 2, 3, 4, 5):
            if expected in services[:k]:
                ac[k] += 1
        n_ok += 1
    if n_ok == 0:
        return None
    avg5 = sum(ac[k] for k in (1, 2, 3, 4, 5)) / (5 * n_ok)
    return {
        "n": n_ok, "n_total": len(files), "n_fail": n_fail,
        "AC@1": ac[1] / n_ok, "AC@3": ac[3] / n_ok, "AC@5": ac[5] / n_ok,
        "Avg@5": avg5,
    }


def eval_ivd_json(path):
    d = json.load(open(path))
    r = d["results"]
    ac = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    n = 0
    for c in r:
        rk = c.get("predicted_ranking") or []
        exp = c["expected_component"]
        exp_norm = _ivd_normalize(exp)
        seen = []
        for s in rk[:5]:
            sn = _ivd_normalize(s)
            if sn not in seen:
                seen.append(sn)
        for k in (1, 2, 3, 4, 5):
            if (exp in seen[:k]) or (exp_norm in seen[:k]):
                ac[k] += 1
        n += 1
    if n == 0:
        return None
    avg5 = sum(ac[k] for k in (1, 2, 3, 4, 5)) / (5 * n)
    return {
        "n": n, "n_total": n, "n_fail": 0,
        "AC@1": ac[1] / n, "AC@3": ac[3] / n, "AC@5": ac[5] / n,
        "Avg@5": avg5,
    }


IVD_FILES = {
    "RE2-OB": f"{IVD_ROOT}/RE2-OB_lw_v2.json",
    "RE2-SS": f"{IVD_ROOT}/RE2-SS_lw_v2.json",
    "RE2-TT": f"{IVD_ROOT}/RE2-TT_lw_v2.json",
    "RE3-OB": f"{IVD_ROOT}/RE3-OB_lw_v2.json",
    "RE3-SS": f"{IVD_ROOT}/RE3-SS_lw_v2.json",
    "RE3-TT": f"{IVD_ROOT}/RE3-TT_lw_v2.json",
    "AIOps2021": f"{IVD_ROOT}/AIOps2021-test_lw_v1.json",
}

BASELINE_KEYS = ["baro", "rcd", "e_diagnosis", "microrank", "tracerca", "pdiagnose"]
BASELINE_DS_MAP = {
    "re2-ob": "RE2-OB", "re2-ss": "RE2-SS", "re2-tt": "RE2-TT",
    "re3-ob": "RE3-OB", "re3-ss": "RE3-SS", "re3-tt": "RE3-TT",
    "aiops2021": "AIOps2021",
}


def main():
    # Gather all data
    results = {}  # results[(dataset, method)] = dict
    for ds_key, ds_name in BASELINE_DS_MAP.items():
        # IVD
        ivd_path = IVD_FILES.get(ds_name)
        if ivd_path and os.path.exists(ivd_path):
            r = eval_ivd_json(ivd_path)
            if r:
                results[(ds_name, "IVD")] = r
        # baselines
        for m in BASELINE_KEYS:
            rd = f"{BASELINE_ROOT}/{m}_{ds_key}/results"
            if os.path.isdir(rd):
                r = eval_baseline_dir(rd, expected_from=("json" if "aiops2021" in ds_key else "filename"))
                if r:
                    results[(ds_name, m)] = r

    # Print grouped tables
    for table_name, ds_list in [
        ("Table 1: RE2", ["RE2-OB", "RE2-SS", "RE2-TT"]),
        ("Table 2: RE3", ["RE3-OB", "RE3-SS", "RE3-TT"]),
        ("Table 3: AIOps2021", ["AIOps2021"]),
    ]:
        print(f"\n=== {table_name} ===")
        # Table header
        methods_order = ["baro", "rcd", "e_diagnosis", "microrank", "tracerca", "pdiagnose", "IVD"]
        print(f"{'Dataset':<12s}", end='')
        for m in methods_order:
            print(f" | {m:>11s}", end='')
        print()
        print("-" * (12 + len(methods_order) * 15))
        for ds in ds_list:
            print(f"{ds:<12s}", end='')
            for m in methods_order:
                r = results.get((ds, m))
                if r is None:
                    print(f" | {'n/a':>11s}", end='')
                else:
                    print(f" | {r['Avg@5']:>5.3f}({r['n']:>3d})", end='')
            print()
        print("\n  Legend: value = Avg@5; (n) = evaluable cases")
        # AC@1 row
        print(f"\n  AC@1 for {table_name}:")
        print(f"{'Dataset':<12s}", end='')
        for m in methods_order:
            print(f" | {m:>11s}", end='')
        print()
        print("-" * (12 + len(methods_order) * 15))
        for ds in ds_list:
            print(f"{ds:<12s}", end='')
            for m in methods_order:
                r = results.get((ds, m))
                if r is None:
                    print(f" | {'n/a':>11s}", end='')
                else:
                    print(f" | {r['AC@1']:>11.4f}", end='')
            print()


if __name__ == "__main__":
    main()