"""End-to-end evaluation: replace component scoring with LambdaRank ranker.

Patches the debug.json's decisions with LambdaRank re-ranked versions,
then writes new predictions and evaluates.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

from dl_ranker.features import extract_candidate_features
from refute.src.baseline_distributions import BaselineStore


def _load_trace(case_id: str, trace_dir: str) -> dict | None:
    p = Path(trace_dir) / f"{case_id}.json"
    if p.exists():
        return json.loads(p.read_text())
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug-json", required=True)
    parser.add_argument("--cases-csv", required=True)
    parser.add_argument("--metrics-csv", required=True)
    parser.add_argument("--logs-csv", default="")
    parser.add_argument("--trace-dir", default="")
    parser.add_argument("--baseline-json", default="")
    parser.add_argument("--ranker-model", required=True)
    parser.add_argument("--ranker-scaler", required=True)
    parser.add_argument("--out-pred", required=True)
    parser.add_argument("--out-debug", default="")
    args = parser.parse_args()

    # Load ranker
    ranker = xgb.XGBRanker()
    ranker.load_model(args.ranker_model)
    scaler: StandardScaler = joblib.load(args.ranker_scaler)

    # Load baseline
    baseline = (
        BaselineStore.load_json(args.baseline_json)
        if args.baseline_json and Path(args.baseline_json).exists()
        else None
    )

    # Load data
    with open(args.debug_json) as f:
        debug_data = json.load(f)
    dbg_list = debug_data.get("debug", [])
    cases_df = pd.read_csv(args.cases_csv)
    gt_map = {r.case_id: r.root_cause_component for r in cases_df.itertuples()}

    # Load mertic/log once, filter per case
    print("Loading metrics...")
    metrics_full = pd.read_csv(
        args.metrics_csv,
        dtype={"timestamp": float, "cmdb_id": str, "kpi_name": str, "value": str},
    )
    metrics_full["value"] = pd.to_numeric(metrics_full["value"], errors="coerce")
    metrics_full["timestamp"] = pd.to_numeric(metrics_full["timestamp"], errors="coerce")
    metrics_full = metrics_full.dropna(subset=["cmdb_id", "kpi_name"])
    print(f"  {len(metrics_full)} rows")

    logs_full = None
    if args.logs_csv and Path(args.logs_csv).exists():
        logs_full = pd.read_csv(args.logs_csv, dtype={"timestamp": str, "cmdb_id": str, "value": str})
        logs_full["timestamp"] = pd.to_numeric(logs_full["timestamp"], errors="coerce")

    # Process each case
    new_predictions = []

    for i, case_debug in enumerate(dbg_list):
        cid = case_debug.get("case_id", "")
        window = case_debug.get("window", {})
        d32r = case_debug.get("d32_result", {})
        d32d = d32r.get("debug", {})
        candidates = d32d.get("candidate_space", [])
        rp = d32d.get("reason_posterior", {})

        if not candidates:
            new_predictions.append(dict(case_id=cid, prediction={}, gt_component=""))
            continue

        # Filter metrics/logs per case
        start = window["start_ts"] - 3600; end = window["end_ts"] + 3600
        metric_df = metrics_full[(metrics_full["timestamp"] >= start) & (metrics_full["timestamp"] <= end)]
        log_df = logs_full[(logs_full["timestamp"] >= start) & (logs_full["timestamp"] <= end)] if logs_full is not None else None
        trace_summary = _load_trace(cid, args.trace_dir) if args.trace_dir else None

        # Extract features
        feat, onset, _ = extract_candidate_features(
            candidate_space=candidates,
            metric_df=metric_df,
            log_df=log_df,
            trace_summary=trace_summary,
            baseline=baseline if baseline is not None else _FallbackBaseline(),
            reason_posterior=rp,
            window_start_ts=window.get("start_ts", 0),
            window_end_ts=window.get("end_ts", 0),
            gt_component="",
        )

        X = scaler.transform(feat)
        scores = ranker.predict(X)

        # Post-multiply by reason_posterior
        rp_arr = np.array([rp.get(c["reason_bucket"], 0.0) for c in candidates])
        final_scores = scores * rp_arr

        # Top candidate
        top_idx = int(np.argmax(final_scores))
        top_c = candidates[top_idx]
        top_comp = top_c["component"]
        top_reason = top_c.get("reason", top_c.get("reason_bucket", ""))

        # Build prediction in pipeline format
        new_pred = {
            "1": {
                "root cause occurrence datetime": case_debug.get("d32_result", {}).get("prediction", {}).get("1", {}).get("root cause occurrence datetime", ""),
                "root cause component": top_comp,
                "root cause reason": top_reason,
            }
        }

        new_predictions.append({
            "case_id": cid,
            "window": window,
            "prediction": new_pred,
            "gt_component": gt_map.get(cid, ""),
            "top_score": float(final_scores[top_idx]),
            "n_candidates": len(candidates),
        })

    # Write predictions CSV
    rows = []
    for npred in new_predictions:
        pred = npred.get("prediction", {})
        p1 = pred.get("1", {})
        rows.append({
            "case_id": npred["case_id"],
            "root_cause_component": p1.get("root cause component", ""),
            "root_cause_reason": p1.get("root cause reason", ""),
            "score": npred.get("top_score", 0.0),
            "n_candidates": npred.get("n_candidates", 0),
        })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.out_pred, index=False)

    if args.out_debug:
        with open(args.out_debug, "w") as f:
            json.dump([{
                "case_id": npred["case_id"],
                "gt": npred.get("gt_component", ""),
                "pred": npred.get("prediction", {}).get("1", {}).get("root cause component", ""),
                "score": npred.get("top_score", 0.0),
                "n_candidates": npred.get("n_candidates", 0),
            } for npred in new_predictions], f, indent=2)

    # Quick accuracy
    correct = sum(
        1 for npred in new_predictions
        if npred.get("prediction", {}).get("1", {}).get("root cause component") == npred.get("gt_component", "")
    )
    total = len(new_predictions)
    print(f"Component accuracy: {correct}/{total} = {correct/total*100:.1f}%")
    print(f"Output: {args.out_pred}")


class _FallbackBaseline:
    def is_anomalous(self, cmdb_id, kpi_name, value, threshold="p99"):
        from dataclasses import dataclass
        @dataclass
        class R:
            is_anomalous: bool = True
            deviation: float = 0.01
            reason: str = "fallback"
        return R()


if __name__ == "__main__":
    main()
