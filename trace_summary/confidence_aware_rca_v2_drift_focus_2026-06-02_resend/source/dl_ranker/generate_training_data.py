"""Generate per-case training data for the Set Transformer ranker.

Usage:
    PYTHONPATH=.:.. python3 dl_ranker/generate_training_data.py \
        --debug-json logs/.../debug.json \
        --cases-csv logs/.../cases.csv \
        --metrics-csv logs/.../metrics.csv \
        --logs-csv logs/.../logs.csv \
        --trace-dir logs/.../trace_summaries \
        --baseline-json refute/knowledge/baseline_distributions_all_metric_dates.json \
        --out-dir dl_ranker_data/
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from dl_ranker.features import extract_candidate_features


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Set Transformer training data")
    parser.add_argument("--debug-json", required=True)
    parser.add_argument("--cases-csv", required=True)
    parser.add_argument("--metrics-csv", required=True)
    parser.add_argument("--logs-csv", default="")
    parser.add_argument("--trace-dir", default="")
    parser.add_argument("--baseline-json", default="")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--train-only", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load baseline
    baseline = _load_baseline(args.baseline_json)

    # Load ground truth
    cases_df = pd.read_csv(args.cases_csv)
    gt_map = {r.case_id: (r.root_cause_component, r.failure_type, r.data_type)
              for r in cases_df.itertuples()}

    # Load debug
    with open(args.debug_json) as f:
        debug_data = json.load(f)
    debug_list = debug_data.get("debug", debug_data if isinstance(debug_data, list) else [])
    if isinstance(debug_data, dict) and "debug" not in debug_data and isinstance(debug_data, list):
        debug_list = debug_data

    # Load full metric/log once, filter per case
    metrics_full = _load_metrics_once(args.metrics_csv)
    logs_full = _load_logs_once(args.logs_csv) if args.logs_csv else None

    generated = 0
    skipped = 0

    for case_debug in debug_list:
        cid = case_debug.get("case_id", "")

        if cid not in gt_map:
            skipped += 1
            continue

        gt_comp, gt_type, data_type = gt_map[cid]
        if args.train_only and data_type != "train":
            skipped += 1
            continue

        window = case_debug.get("window", {})
        d32_result = case_debug.get("d32_result", {})
        d32_debug = d32_result.get("debug", {})
        candidate_space = d32_debug.get("candidate_space", [])
        reason_posterior = d32_debug.get("reason_posterior", {})

        if not candidate_space:
            skipped += 1
            continue

        try:
            metric_df = _filter_window(metrics_full, window)
            log_df = _filter_window(logs_full, window) if logs_full is not None else None
            trace_summary = _load_trace_summary(cid, args.trace_dir)

            feats, onsets, labels = extract_candidate_features(
                candidate_space=candidate_space,
                metric_df=metric_df,
                log_df=log_df,
                trace_summary=trace_summary,
                baseline=baseline,
                reason_posterior=reason_posterior,
                window_start_ts=float(window.get("start_ts", 0)),
                window_end_ts=float(window.get("end_ts", 0)),
                gt_component=gt_comp,
            )

            torch.save({
                "features": torch.from_numpy(feats),
                "onset_ranks": torch.from_numpy(onsets),
                "labels": torch.from_numpy(labels),
                "case_id": cid,
                "gt_component": gt_comp,
                "gt_type": gt_type,
            }, out_dir / f"case_{cid}.pt")
            generated += 1
        except Exception as e:
            skipped += 1
            continue

    print(f"Generated {generated} case files, skipped {skipped}")
    print(f"Output: {out_dir}")


# -- helpers --

def _load_baseline(baseline_json: str):
    """Load BaselineStore or return a fallback."""
    if baseline_json and Path(baseline_json).exists():
        from refute.src.baseline_distributions import BaselineStore
        return BaselineStore.load_json(baseline_json)
    return _FallbackBaseline()


def _load_metrics_once(metrics_csv: str) -> pd.DataFrame | None:
    path = Path(metrics_csv)
    if not path.exists():
        return None
    df = pd.read_csv(
        metrics_csv,
        dtype={"timestamp": float, "cmdb_id": str, "kpi_name": str, "value": str},
    )
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["cmdb_id", "kpi_name"])
    return df


def _load_logs_once(logs_csv: str) -> pd.DataFrame | None:
    path = Path(logs_csv)
    if not path.exists():
        return None
    df = pd.read_csv(logs_csv, dtype={"timestamp": str, "cmdb_id": str, "value": str})
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    return df


def _filter_window(df: pd.DataFrame | None, window: dict) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["timestamp", "cmdb_id", "kpi_name", "value"])
    start = float(window.get("start_ts", 0))
    end = float(window.get("end_ts", 0))
    margin = 3600
    return df[(df["timestamp"] >= start - margin) & (df["timestamp"] <= end + margin)]


def _load_trace_summary(case_id: str, trace_dir: str) -> dict | None:
    path = Path(trace_dir) / f"{case_id}.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    for p in Path(trace_dir).glob(f"*{case_id}.json"):
        with open(p) as f:
            return json.load(f)
    return None


class _FallbackBaseline:
    """Fallback: treat everything as anomalous with a small deviation."""

    def is_anomalous(self, cmdb_id: str, kpi_name: str, value: float, threshold: str = "p99"):
        from dataclasses import dataclass

        @dataclass
        class Result:
            is_anomalous: bool = False
            deviation: float = 0.0
            reason: str = ""

        try:
            v = float(value)
        except (ValueError, TypeError):
            return Result(False, 0.0, "non_numeric")
        if not math.isfinite(v) or v <= 0:
            return Result(False, 0.0, "zero_or_nan")
        return Result(True, v * 0.05, "fallback")


if __name__ == "__main__":
    main()
