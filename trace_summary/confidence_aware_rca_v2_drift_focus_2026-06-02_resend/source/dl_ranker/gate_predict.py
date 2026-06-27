"""Residual RCA Gate: learned selector between Heuristic and Residual ST.

Trains a small classifier on case-level features to decide whether to trust
the Heuristic (H) or Residual Set Transformer (R) on each case.

Then generates full pipeline output (predictions.csv + debug.json) for
official evaluation.
"""
from __future__ import annotations

import argparse, json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from dl_ranker.model import OnsetAwareSetTransformer
from dl_ranker.features import extract_candidate_features
from refute.src.baseline_distributions import BaselineStore


def extract_case_features(cid: str, data_dir: str, debug_data: list, trace_dir: str) -> dict[str, float]:
    """Extract case-level features for gate training/prediction."""
    d = torch.load(Path(data_dir) / f"case_{cid}.pt", weights_only=True, map_location="cpu")
    feats = d["features"].numpy(); onset = d["onset_ranks"].numpy()
    n = feats.shape[0]

    rp_vals = feats[:, 30]
    rp_max = float(rp_vals.max())
    rp_ent = -float(np.sum(rp_vals * np.log(np.maximum(rp_vals, 1e-8)))) / n

    e = feats[:, 0]; s_ = feats[:, 1]; d_ = feats[:, 2]
    comp = 0.6 * e + 0.25 * s_ + 0.15 * d_
    h_scores = rp_vals * comp
    h_top = float(np.max(h_scores))
    h_top2 = float(sorted(h_scores, reverse=True)[1]) if n >= 2 else 0.0
    h_margin = h_top - h_top2

    h_top_idx = int(np.argmax(h_scores))
    n_anom = int((onset < 99).sum())
    n_anom_norm = n_anom / max(n, 1)

    # Trace stats
    n_slow = 0; n_svc = 0
    if trace_dir:
        tsp = Path(trace_dir) / f"{cid}.json"
        if tsp.exists():
            ts = json.loads(tsp.read_text())
            n_slow = len(ts.get("events", {}).get("slow_edges", []))
            n_svc = len(ts.get("service_stats", {}))

    # Heuristic's predicted component reason_posterior
    h_rp = 0.0
    for case in debug_data:
        if case.get("case_id") == cid:
            rp_dict = case.get("d32_result", {}).get("debug", {}).get("reason_posterior", {})
            decisions = case.get("d32_result", {}).get("debug", {}).get("all_decisions", [])
            if decisions:
                h_bucket = decisions[0]["candidate"].get("reason_bucket", "")
                h_rp = rp_dict.get(h_bucket, 0.0)
            break

    return {
        "rp_max": rp_max, "rp_ent": rp_ent,
        "h_top": h_top, "h_margin": h_margin,
        "h_rp": h_rp, "n_anom": n_anom_norm,
        "n_slow": float(n_slow), "n_svc": float(n_svc),
        "n_candidates": float(n),
    }


def train_gate(
    data_dir: str,
    train_ids: list[str],
    test_ids: list[str],
    debug_data: list,
    trace_dir: str,
    h_pred: dict,
    r_pred: dict,
    gt_map: dict,
    model_path: str,
    scaler_path: str,
) -> tuple[Any, Any, float]:
    """Train a gate classifier on case-level features."""
    X_g, y_g, cids_g = [], [], []

    for cid in train_ids:
        h, r = h_pred.get(cid, ""), r_pred.get(cid, "")
        if h == r:
            continue
        try:
            f = extract_case_features(cid, data_dir, debug_data, trace_dir)
            feat_vec = list(f.values())
            X_g.append(feat_vec)
            # Label: 1 = trust H (H is right), 0 = trust R (R is right or both wrong)
            y_g.append(1 if h == gt_map.get(cid, "___never___") else 0)
            cids_g.append(cid)
        except Exception:
            continue

    X_g = np.array(X_g, dtype=np.float32)
    y_g = np.array(y_g, dtype=np.int32)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_g)

    gate = xgb.XGBClassifier(
        n_estimators=50, max_depth=3, learning_rate=0.1,
        eval_metric="logloss", verbosity=0, random_state=42,
    )
    gate.fit(X_scaled, y_g)
    train_acc = gate.score(X_scaled, y_g)

    gate.save_model(model_path)
    joblib.dump(scaler, scaler_path)

    # Evaluate on test disagreement
    correct_choices = 0; total_disagree = 0
    for cid in test_ids:
        h, r = h_pred.get(cid, ""), r_pred.get(cid, "")
        if h == r:
            continue
        try:
            f = extract_case_features(cid, data_dir, debug_data, trace_dir)
            feat_vec = np.array(list(f.values())).reshape(1, -1)
            fs = scaler.transform(feat_vec)
            pred = gate.predict(fs)[0]
            # Check if gate's choice was correct
            if (pred == 1 and h == gt_map[cid]) or (pred == 0 and r == gt_map[cid]):
                correct_choices += 1
            total_disagree += 1
        except Exception:
            continue

    return gate, scaler, float(correct_choices), total_disagree


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--splits-json", required=True)
    parser.add_argument("--cases-csv", required=True)
    parser.add_argument("--debug-json", required=True)
    parser.add_argument("--trace-dir", default="")
    parser.add_argument("--residual-model", required=True)
    parser.add_argument("--out-pred", required=True)
    parser.add_argument("--out-debug", required=True)
    parser.add_argument("--out-gate-model", default="")
    parser.add_argument("--out-gate-scaler", default="")
    args = parser.parse_args()

    with open(args.splits_json) as f:
        splits = json.load(f)
    train_ids = splits["train"]
    test_ids = splits["test"]

    with open(args.debug_json) as f:
        debug_data = json.load(f).get("debug", [])

    cases_df = pd.read_csv(args.cases_csv)
    gt_map = {r.case_id: r.root_cause_component for r in cases_df.itertuples()}

    # Heuristic predictions from debug
    h_pred = {
        c["case_id"]: c.get("d32_result", {}).get("prediction", {}).get("1", {}).get("root cause component", "")
        for c in debug_data
    }
    h_score_map = {}
    for c in debug_data:
        cid = c["case_id"]
        decs = c.get("d32_result", {}).get("debug", {}).get("all_decisions", [])
        h_score_map[cid] = float(-decs[0].get("rebuttal_score", 0)) if decs else 0.0
    cs_map = {c["case_id"]: c.get("d32_result", {}).get("debug", {}).get("candidate_space", []) for c in debug_data}

    # Residual ST predictions
    st = OnsetAwareSetTransformer(n_features=34, d_model=64, n_heads=4, n_layers=2, dropout=0.3)
    st.load_state_dict(torch.load(args.residual_model, weights_only=True, map_location="cpu"))
    st.eval()

    r_pred = {}; r_conf = {}
    for cid in train_ids + test_ids:
        d = torch.load(Path(args.data_dir) / f"case_{cid}.pt", weights_only=True, map_location="cpu")
        feat = d["features"].unsqueeze(0)
        onset = d["onset_ranks"].unsqueeze(0)
        with torch.no_grad():
            probs = st.predict_proba(feat, onset)
        p = probs[0].numpy()
        rp_v = feat[0, :, 30].numpy()
        scores = p * rp_v
        top = int(np.argmax(scores))
        cs = cs_map.get(cid, [])
        r_pred[cid] = cs[top]["component"] if top < len(cs) else ""
        r_conf[cid] = float(scores[top])

    # Train gate
    gate, gate_scaler, gate_correct, gate_total = train_gate(
        args.data_dir, train_ids, test_ids, debug_data, args.trace_dir,
        h_pred, r_pred, gt_map,
        args.out_gate_model or "gate_model.json",
        args.out_gate_scaler or "gate_scaler.pkl",
    )
    print(f"Gate on test disagreement: {gate_correct}/{gate_total} correct choices")

    # Apply gate to ALL test cases
    final_pred = {}
    for cid in test_ids:
        h = h_pred.get(cid, "")
        r = r_pred.get(cid, "")
        h_score = h_score_map.get(cid, 0)

        if h == r:
            final_pred[cid] = h
        else:
            try:
                f = extract_case_features(cid, args.data_dir, debug_data, args.trace_dir)
                feat_vec = np.array(list(f.values())).reshape(1, -1)
                fs = gate_scaler.transform(feat_vec)
                pred = gate.predict(fs)[0]
                final_pred[cid] = h if pred == 1 else r
            except Exception:
                final_pred[cid] = h if h_score > 1.0 else r

    # Count accuracy
    correct = sum(1 for cid in test_ids if final_pred.get(cid, "") == gt_map[cid])
    h_base = sum(1 for cid in test_ids if h_pred.get(cid, "") == gt_map[cid])
    print(f"\nFinal gate result: {correct}/{len(test_ids)} = {correct / len(test_ids) * 100:.1f}%")
    print(f"  Heuristic baseline: {h_base}/{len(test_ids)} = {h_base / len(test_ids) * 100:.1f}%")
    print(f"  Improvement: +{correct - h_base} cases")

    # Generate full pipeline output
    pred_rows = []
    debug_rows = []

    for idx, cid in enumerate(test_ids):
        comp = final_pred.get(cid, h_pred.get(cid, ""))
        # Find the candidate with this component to get reason
        cs = cs_map.get(cid, [])
        reason = ""
        for c in cs:
            if c["component"] == comp:
                reason = c.get("reason", c.get("reason_bucket", ""))
                break

        pred_dict = {
            "1": {
                "root cause occurrence datetime": "",
                "root cause component": comp,
                "root cause reason": reason,
            }
        }
        pred_rows.append({
            "row_id": idx, "case_id": cid,
            "prediction": json.dumps(pred_dict, ensure_ascii=False),
        })

        # Build debug with all_decisions (top-10 from Residual ST)
        d = torch.load(Path(args.data_dir) / f"case_{cid}.pt", weights_only=True, map_location="cpu")
        feat = d["features"].unsqueeze(0)
        onset = d["onset_ranks"].unsqueeze(0)
        with torch.no_grad():
            probs = st.predict_proba(feat, onset)
        p = probs[0].numpy()
        rp_v = feat[0, :, 30].numpy()
        scores = p * rp_v
        ranked = np.argsort(-scores)[:10]

        all_dec = []
        for rank_j, j in enumerate(ranked):
            cand = cs[j] if j < len(cs) else {"component": "?", "reason": "?", "reason_bucket": "?"}
            all_dec.append({
                "candidate": {
                    "component": cand.get("component", ""),
                    "reason": cand.get("reason", cand.get("reason_bucket", "")),
                    "reason_bucket": cand.get("reason_bucket", ""),
                    "prior": cand.get("prior", 0.0),
                    "source": "residual_st_gate",
                    "details": cand.get("details", {}),
                },
                "rebuttal_score": float(-scores[j]),
                "support_strength": 0.0,
                "refute_strength": 0.0,
                "blind_count": 0,
                "confidence": "HIGH" if rank_j == 0 else ("MEDIUM" if rank_j < 3 else "LOW"),
                "cards": [],
            })

        debug_rows.append({
            "row_id": idx, "case_id": cid,
            "dataset": "aiops2021",
            "algorithm_flow": "residual_st_gate",
            "window": {},
            "modal_status": {"metric": "present", "log": "present", "trace": "present"},
            "d32_result": {
                "prediction": pred_dict,
                "high_suspicion": [],
                "low_suspicion": [],
                "data_blind_spots": [],
                "debug": {
                    "case_id": cid,
                    "candidate_space": cs,
                    "reason_posterior": {},
                    "all_decisions": all_dec,
                },
            },
        })

    # Write outputs
    out_pred = Path(args.out_pred)
    out_pred.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(pred_rows).to_csv(out_pred, index=False)

    out_dbg = Path(args.out_debug)
    out_dbg.parent.mkdir(parents=True, exist_ok=True)
    with open(out_dbg, "w") as f:
        json.dump({"n": len(debug_rows), "debug": debug_rows}, f, ensure_ascii=False, indent=2)

    print(f"\nPredictions: {out_pred}")
    print(f"Debug:       {out_dbg}")
    print(f"\\nRun eval: PYTHONPATH=.:.. python eval/evaluate_portable_task.py")
    print(f"  --dataset aiops2021 --cases-csv {args.cases_csv}")
    print(f"  --pred {out_pred} --debug-json {out_dbg}")


if __name__ == "__main__":
    main()
