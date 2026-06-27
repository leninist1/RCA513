"""Evaluate trained model on test split and compare with XGBoost baseline."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

from dl_ranker.dataset import CaseDataset, collate_variable
from dl_ranker.model import OnsetAwareSetTransformer
from dl_ranker.train import evaluate


def evaluate_xgboost_pointwise(
    data_dir: str,
    train_ids: list[str],
    test_ids: list[str],
) -> dict[str, float]:
    """Train XGBoost binary classifier and evaluate on test."""
    # Collect all candidates from train
    X_train = []
    y_train = []
    for cid in train_ids:
        d = torch.load(Path(data_dir) / f"case_{cid}.pt", weights_only=True, map_location="cpu")
        X_train.append(d["features"].numpy())
        y_train.append(d["labels"].numpy())
    X_train = np.concatenate(X_train, axis=0)
    y_train = np.concatenate(y_train, axis=0).astype(int)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    n_pos = y_train.sum()
    scale_pos_weight = (len(y_train) - n_pos) / max(n_pos, 1)
    clf = xgb.XGBClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.1,
        scale_pos_weight=scale_pos_weight, eval_metric="logloss",
        verbosity=0, random_state=42,
    )
    clf.fit(X_train_scaled, y_train)

    # Evaluate per case on test
    correct_top1 = 0
    correct_top3 = 0
    correct_top5 = 0
    total = 0
    ndcg1 = 0.0
    ndcg3 = 0.0
    mrr = 0.0

    for cid in test_ids:
        d = torch.load(Path(data_dir) / f"case_{cid}.pt", weights_only=True, map_location="cpu")
        X = d["features"].numpy()
        y = d["labels"].numpy()
        if len(y) == 0:
            continue
        X_s = scaler.transform(X)
        probs = clf.predict_proba(X_s)[:, 1]

        pos_set = set(np.where(y == 1)[0])
        if not pos_set:
            continue

        top = np.argsort(-probs)[:5]

        if top[0] in pos_set:
            correct_top1 += 1
        if pos_set & set(top[:3]):
            correct_top3 += 1
        if pos_set & set(top):
            correct_top5 += 1

        total += 1

        # NDCG
        dcg = 0.0
        for i, idx in enumerate(np.argsort(-probs)):
            if i >= 1:
                break
            if idx in pos_set:
                dcg += 1.0 / math.log2(i + 2)
        idcg = 1.0
        ndcg1 += dcg / idcg

        dcg = 0.0
        for i, idx in enumerate(np.argsort(-probs)):
            if i >= 3:
                break
            if idx in pos_set:
                dcg += 1.0 / math.log2(i + 2)
        ndcg3 += dcg / idcg

        # MRR
        ranks = [i for i, idx in enumerate(np.argsort(-probs)) if idx in pos_set]
        mrr += 1.0 / (min(ranks) + 1) if ranks else 0.0

    return {
        "acc@1": correct_top1 / total if total > 0 else 0.0,
        "acc@3": correct_top3 / total if total > 0 else 0.0,
        "acc@5": correct_top5 / total if total > 0 else 0.0,
        "ndcg@1": ndcg1 / total if total > 0 else 0.0,
        "ndcg@3": ndcg3 / total if total > 0 else 0.0,
        "mrr": mrr / total if total > 0 else 0.0,
        "n": total,
    }


def evaluate_set_transformer(
    model_path: str,
    data_dir: str,
    test_ids: list[str],
    device: str = "cpu",
) -> dict[str, float]:
    model = OnsetAwareSetTransformer(n_features=33, d_model=64, n_heads=4, n_layers=2, dropout=0.3)
    state = torch.load(model_path, weights_only=True, map_location="cpu")
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    test_ds = CaseDataset(data_dir, test_ids, augment=False)
    loader = torch.utils.data.DataLoader(
        test_ds, batch_size=4, shuffle=False, collate_fn=collate_variable,
    )
    return evaluate(model, loader, device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--splits-json", required=True)
    parser.add_argument("--model-dir", default="")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    with open(args.splits_json) as f:
        splits = json.load(f)
    train_ids = splits["train"]
    test_ids = splits["test"]

    # XGBoost baseline
    xgb_metrics = evaluate_xgboost_pointwise(args.data_dir, train_ids, test_ids)
    print("=== XGBoost baseline ===")
    for k, v in xgb_metrics.items():
        print(f"  {k}: {v:.4f}")

    # Set Transformer
    if args.model_dir:
        model_dir = Path(args.model_dir)
        for fold in range(2):
            model_path = model_dir / f"model_fold{fold}.pt"
            if model_path.exists():
                st_metrics = evaluate_set_transformer(
                    str(model_path), args.data_dir, test_ids, args.device,
                )
                print(f"\n=== Set Transformer fold{fold} ===")
                for k, v in st_metrics.items():
                    print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
