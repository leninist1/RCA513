"""Unified prediction + debug generator for learned ranker E2E evaluation.

Generates both predictions.csv AND debug.json with all_decisions populated,
so the evaluation script can compute HR@3, HR@5 properly.
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
    parser.add_argument("--two-stage", type=int, default=5, help="top-K reasons (0=disabled)")
    parser.add_argument("--out-pred", required=True)
    parser.add_argument("--out-debug", required=True)
    args = parser.parse_args()

    ranker = xgb.XGBRanker()
    ranker.load_model(args.ranker_model)
    scaler: StandardScaler = joblib.load(args.ranker_scaler)

    baseline = (
        BaselineStore.load_json(args.baseline_json)
        if args.baseline_json and Path(args.baseline_json).exists()
        else None
    )

    with open(args.debug_json) as f:
        debug_data = json.load(f)
    dbg_list = debug_data.get("debug", [])

    cases_df = pd.read_csv(args.cases_csv)

    print("Loading metrics...")
    metrics_full = pd.read_csv(
        args.metrics_csv,
        dtype={"timestamp": float, "cmdb_id": str, "kpi_name": str, "value": str},
    )
    metrics_full["value"] = pd.to_numeric(metrics_full["value"], errors="coerce")
    metrics_full["timestamp"] = pd.to_numeric(metrics_full["timestamp"], errors="coerce")
    metrics_full = metrics_full.dropna(subset=["cmdb_id", "kpi_name"])

    logs_full = None
    if args.logs_csv and Path(args.logs_csv).exists():
        logs_full = pd.read_csv(args.logs_csv, dtype={"timestamp": str, "cmdb_id": str, "value": str})
        logs_full["timestamp"] = pd.to_numeric(logs_full["timestamp"], errors="coerce")

    pred_rows = []
    debug_out_rows = []

    for i, case_debug in enumerate(dbg_list):
        cid = case_debug.get("case_id", "")
        window = case_debug.get("window", {})
        d32r = case_debug.get("d32_result", {})
        d32d = d32r.get("debug", {})
        candidates = d32d.get("candidate_space", [])
        rp = d32d.get("reason_posterior", {})

        if not candidates:
            continue

        start = window["start_ts"] - 3600
        end = window["end_ts"] + 3600
        metric_df = metrics_full[(metrics_full["timestamp"] >= start) & (metrics_full["timestamp"] <= end)]
        log_df = logs_full[(logs_full["timestamp"] >= start) & (logs_full["timestamp"] <= end)] if logs_full is not None else None
        trace_summary = _load_trace(cid, args.trace_dir)

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
        rp_arr = np.array([rp.get(c["reason_bucket"], 0.0) for c in candidates])
        final_scores = scores * rp_arr

        # Two-stage filter
        if args.two_stage > 0:
            top_buckets = set(sorted(rp, key=rp.get, reverse=True)[:args.two_stage])
            for i2 in range(len(final_scores)):
                if candidates[i2].get("reason_bucket", "") not in top_buckets:
                    final_scores[i2] = -1e9

        ranked_idx = np.argsort(-final_scores)
        top_k = min(10, len(ranked_idx))

        # Build prediction (pipeline format)
        top_idx = ranked_idx[0]
        top_c = candidates[top_idx]
        pred_dict = {
            "1": {
                "root cause occurrence datetime": case_debug.get("d32_result", {}).get("prediction", {}).get("1", {}).get("root cause occurrence datetime", ""),
                "root cause component": top_c["component"],
                "root cause reason": top_c.get("reason", top_c.get("reason_bucket", "")),
            }
        }

        pred_rows.append({
            "row_id": i, "case_id": cid,
            "prediction": json.dumps(pred_dict, ensure_ascii=False),
        })

        # Build debug row with all_decisions
        all_decisions = []
        for rank_j, idx in enumerate(ranked_idx[:top_k]):
            cand = candidates[idx]
            all_decisions.append({
                "candidate": {
                    "component": cand["component"],
                    "reason": cand.get("reason", cand.get("reason_bucket", "")),
                    "reason_bucket": cand.get("reason_bucket", ""),
                    "prior": cand.get("prior", 0.0),
                    "source": cand.get("source", "learned_ranker"),
                    "details": cand.get("details", {}),
                },
                "rebuttal_score": float(-final_scores[idx]),
                "support_strength": float(0.0),
                "refute_strength": float(0.0),
                "blind_count": 0,
                "confidence": "HIGH" if rank_j == 0 else ("MEDIUM" if rank_j < 3 else "LOW"),
                "cards": [],
            })

        debug_out_rows.append({
            "row_id": i,
            "case_id": cid,
            "dataset": "aiops2021",
            "algorithm_flow": "learned_ranker",
            "window": window,
            "modal_status": case_debug.get("modal_status", {}),
            "d32_result": {
                "prediction": pred_dict,
                "high_suspicion": [],
                "low_suspicion": [],
                "data_blind_spots": [],
                "debug": {
                    "case_id": cid,
                    "candidate_space": candidates,
                    "reason_posterior": rp,
                    "all_decisions": all_decisions,
                },
            },
        })

    # Write
    out_pred = Path(args.out_pred)
    out_pred.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(pred_rows).to_csv(out_pred, index=False)

    out_dbg = Path(args.out_debug)
    out_dbg.parent.mkdir(parents=True, exist_ok=True)
    with open(out_dbg, "w") as f:
        json.dump({"n": len(debug_out_rows), "debug": debug_out_rows}, f, ensure_ascii=False, indent=2)

    print(f"Predictions: {out_pred} ({len(pred_rows)} rows)")
    print(f"Debug:      {out_dbg} ({len(debug_out_rows)} rows)")


def _load_trace(case_id: str, trace_dir: str) -> dict | None:
    p = Path(trace_dir) / f"{case_id}.json"
    if p.exists():
        return json.loads(p.read_text())
    return None


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
