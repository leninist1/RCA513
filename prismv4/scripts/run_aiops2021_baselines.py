"""
Batch runner for baselines on AIOps2021 dataset.

For each (method, case) writes a JSON {results: {"0": ranks}} to baseline_results/{method}_aiops2021/results/{case_id}.json
Metric-only baselines: BARO/RCD/E-Diagnosis.

Usage:
  python run_aiops2021_baselines.py --method baro
  python run_aiops2021_baselines.py --method rcd
  python run_aiops2021_baselines.py --method e_diagnosis
"""
import argparse
import json
import os
import sys
import warnings
from os.path import join

warnings.filterwarnings("ignore")

# Add prismv4 path for adapter import
sys.path.insert(0, "/home/dell2/RCA513/yyx")

import numpy as np
import pandas as pd
from tqdm import tqdm

from prismv4.experiments.rcaeval_adapter import discover_aiops2021_cases, load_aiops2021_case


def _is_py38():
    return sys.version_info[:2] == (3, 8)


if _is_py38():
    from RCAEval.e2e import rcd, e_diagnosis, dummy
    METHODS = {"rcd": rcd, "e_diagnosis": e_diagnosis, "dummy": dummy}
else:
    from RCAEval.e2e import baro
    METHODS = {"baro": baro}


OUT_ROOT = "/home/dell2/RCA513/yyx/prismv4/results/baseline_results"


def run_one(method, case_spec, length=30, window_pre_min=30):
    """Run a metric-only baseline on one AIOps2021 case.

    length: minutes used for normal+anormal slicing inside the baseline (matches main.py convention)
    window_pre_min: how much pre-fault history to load (minutes worth)
    """
    if method not in METHODS:
        raise ValueError(f"method not available on this python version: {method}")

    loaded = load_aiops2021_case(case_spec, top_k=10, window_pre_min=window_pre_min)
    df = loaded.store.metrics.copy()
    if df.empty or "time" not in df.columns:
        return None, "empty metrics"
    df = df.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
    inject = loaded.store.event_time

    # roughly mirror main.py: window length in seconds
    data_length = length * 60 // 2  # main.py uses length*60//2 per side

    # rename lat-90 -> latency (not needed for AIOps, but consistent)
    for c in list(df.columns):
        if c.endswith("_latency-90"):
            df = df.rename(columns={c: c.replace("_latency-90", "_latency")})

    normal_df = df[df["time"] < inject].tail(data_length)
    anomal_df = df[df["time"] >= inject].head(data_length)
    if normal_df.empty or anomal_df.empty:
        return None, "no normal or anomal slice"
    data = pd.concat([normal_df, anomal_df], ignore_index=True)
    if len(data) < 4:
        return None, f"too few rows: {len(data)}"

    func = METHODS[method]
    try:
        out = func(
            data, inject,
            dataset="aiops2021",
            anomalies=None,
            dk_select_useful=False,
            sli=None,
            verbose=False,
            n_iter=max(1, len(data.columns) - 1),
            args=argparse.Namespace(root_path=os.getcwd(), data_path="aiops2021_case"),
        )
        ranks = out.get("ranks", [])
        return ranks, None
    except Exception as e:
        return None, str(e)[:300]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--length", type=int, default=20)
    ap.add_argument("--window_pre_min", type=int, default=30)
    ap.add_argument("--out_root", default=OUT_ROOT)
    args = ap.parse_args()

    out_dir = join(args.out_root, f"{args.method}_aiops2021", "results")
    os.makedirs(out_dir, exist_ok=True)

    cases = discover_aiops2021_cases(split=args.split, limit=args.limit)
    print(f"=== Running {args.method} on AIOps2021 ({len(cases)} cases) ===", flush=True)

    success = 0
    failed = 0
    for cs in tqdm(cases):
        gt_id = cs["groundtruth_id"]
        # Safe file name: gt_id@service
        safe_service = cs["expected_component"].replace("/", "_")
        file_name = f"aiops2021_{gt_id}_{safe_service}.json"
        out_path = join(out_dir, file_name)
        if os.path.exists(out_path):
            success += 1
            continue

        ranks, err = run_one(args.method, cs,
                             length=args.length,
                             window_pre_min=args.window_pre_min)
        if ranks is None:
            failed += 1
            with open(out_path, "w") as f:
                json.dump({"error": err or "no ranks"}, f)
            print(f"FAIL {gt_id}: {err}", flush=True)
            continue

        success += 1
        with open(out_path, "w") as f:
            json.dump({"0": ranks, "expected": cs["expected_component"],
                       "anomaly_type": cs["anomaly_type"]}, f)

    print(f"=== DONE {args.method}: {success} ok, {failed} fail ===", flush=True)


if __name__ == "__main__":
    main()