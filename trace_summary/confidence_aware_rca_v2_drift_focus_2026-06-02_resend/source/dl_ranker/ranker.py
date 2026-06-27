"""Integration: replace heuristic component scoring with learned ranker in layer2."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

from refute_b_v2_d32.schema import RefutationDecision, RootCandidate
from refute_b_v2_d32.evidence import D32EvidenceQuery
from refute_b_v2_d32.signature import service_type, service_role
from refute_b_v2.trace_propagation import propagation_roles

from dl_ranker.features import extract_candidate_features


class LearnedComponentRanker:
    """Replace _component_earliness_strength with a learned ranker."""

    def __init__(
        self,
        model_path: str | None = None,
        scaler_path: str | None = None,
    ):
        self.model: xgb.XGBRanker | None = None
        self.scaler: StandardScaler | None = None
        if model_path:
            import joblib
            self.model = xgb.XGBRanker()
            self.model.load_model(model_path)
            if scaler_path:
                self.scaler = joblib.load(scaler_path)

    def score_candidates(
        self,
        candidates: list[dict[str, Any]],
        evidence: D32EvidenceQuery,
        reason_posterior: dict[str, float],
        window: dict,
    ) -> list[tuple[int, float]]:
        """Score all candidates and return [(index, score)], descending."""
        if self.model is None:
            return [(i, 0.0) for i in range(len(candidates))]

        metric_df = evidence.metric_df
        log_df = evidence.log_df if hasattr(evidence, 'log_df') else None
        trace_summary = evidence.trace_summary
        baseline = evidence.baseline

        feats, onsets, _ = extract_candidate_features(
            candidate_space=candidates,
            metric_df=metric_df,
            log_df=log_df,
            trace_summary=trace_summary,
            baseline=baseline,
            reason_posterior=reason_posterior,
            window_start_ts=float(window.get("start_ts", 0)),
            window_end_ts=float(window.get("end_ts", 0)),
            gt_component="",  # no GT at inference
        )

        if self.scaler:
            X = self.scaler.transform(feats)
        else:
            X = feats

        scores = self.model.predict(X)

        # Post-multiply by reason_posterior
        rp_values = np.array([reason_posterior.get(c["reason_bucket"], 0.0) for c in candidates])
        final_scores = scores * (0.5 + 0.5 * rp_values)

        ranked = sorted(enumerate(final_scores), key=lambda x: -x[1])
        return ranked


def build_learned_two_stage(
    candidates: list[dict[str, Any]],
    evidence: D32EvidenceQuery,
    reason_posterior: dict[str, float],
    ranker: LearnedComponentRanker,
    window: dict,
    top_reasons: int = 5,
) -> list[tuple[int, float, str, str]]:
    """Two-stage: top-K reasons → rank candidates within each reason.

    Returns [(candidate_index, score, component, reason_bucket)] sorted.
    """
    # Phase 1: top reasons
    rp_sorted = sorted(reason_posterior.items(), key=lambda x: -x[1])
    top_buckets = [b for b, _ in rp_sorted[:top_reasons]]

    # Phase 2: score all candidates, filter to top reasons, re-rank
    ranked = ranker.score_candidates(candidates, evidence, reason_posterior, window)

    results: list[tuple[int, float, str, str]] = []
    for idx, score in ranked:
        cand = candidates[idx]
        bucket = cand.get("reason_bucket", "")
        if bucket in top_buckets:
            results.append((idx, float(score), cand["component"], bucket))

    results.sort(key=lambda x: -x[1])
    return results


def convert_to_decisions(
    ranked: list[tuple[int, float, str, str]],
    candidates: list[dict[str, Any]],
    evidence: D32EvidenceQuery,
) -> list[RefutationDecision]:
    """Convert ranked results to RefutationDecision format for pipeline."""
    decisions = []
    for rank, (idx, score, comp, bucket) in enumerate(ranked):
        cand = candidates[idx]
        evidence_cards = [
            {
                "rule_id": "dl_ranker.two_stage_v3",
                "polarity": "selector",
                "strength": abs(score),
                "text": f"dl-rank component={comp} reason={bucket} score={score:.4f} rank={rank}",
                "details": {"rank": rank, "score": score},
            }
        ]
        decisions.append(RefutationDecision(
            candidate=RootCandidate(
                component=comp,
                reason=cand.get("reason", bucket),
                prior=float(cand.get("prior", 0.0)),
                source="dl_ranker",
                details=cand.get("details", {}),
            ),
            rebuttal_score=-score,
            support_strength=0.0,
            refute_strength=0.0,
            blind_count=0,
            confidence="HIGH" if rank == 0 else "MEDIUM" if rank <= 2 else "LOW",
            cards=evidence_cards,
        ))
    return decisions


def train_ranker(
    data_dir: str,
    splits_json: str,
    out_model_path: str,
    out_scaler_path: str,
) -> tuple[xgb.XGBRanker, StandardScaler]:
    """Train LambdaRank on training cases."""
    import torch
    from sklearn.preprocessing import StandardScaler as SS
    import joblib

    with open(splits_json) as f:
        splits = json.load(f)
    train_ids = splits["train"]

    X_all = []
    Y_all = []
    groups = []
    for cid in train_ids:
        d = torch.load(Path(data_dir) / f"case_{cid}.pt", weights_only=True, map_location="cpu")
        X_all.append(d["features"].numpy())
        Y_all.append(d["labels"].numpy())
        groups.append(d["labels"].shape[0])

    X = np.concatenate(X_all, axis=0)
    Y = np.concatenate(Y_all).astype(int)
    qid = [i for i, g in enumerate(groups) for _ in range(g)]

    scaler = SS()
    X_scaled = scaler.fit_transform(X)

    model = xgb.XGBRanker(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        objective="rank:ndcg", verbosity=1, random_state=42,
    )
    model.fit(X_scaled, Y, qid=qid)

    model.save_model(out_model_path)
    joblib.dump(scaler, out_scaler_path)

    return model, scaler


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--splits-json", required=True)
    parser.add_argument("--out-model", required=True)
    parser.add_argument("--out-scaler", required=True)
    args = parser.parse_args()

    model, scaler = train_ranker(
        args.data_dir, args.splits_json, args.out_model, args.out_scaler,
    )

    # Evaluate
    splits = json.load(open(args.splits_json))
    train_ids, test_ids = splits["train"], splits["test"]
    import torch
    correct = 0; total = 0
    for cid in test_ids:
        d = torch.load(Path(args.data_dir) / f"case_{cid}.pt", weights_only=True, map_location="cpu")
        X = d["features"].numpy(); y = d["labels"].numpy()
        pos = set(np.where(y == 1)[0])
        if not pos: continue
        X_s = scaler.transform(X); scores = model.predict(X_s)
        rp = X[:, 30]
        final = scores * (0.5 + 0.5 * rp)
        top = np.argmax(final)
        if top in pos: correct += 1
        total += 1
    print(f"Test acc@1: {correct}/{total} = {correct/total*100:.1f}%")
