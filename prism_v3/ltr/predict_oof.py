"""Generate true incident-grouped out-of-fold LTR scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from .dataset import LTRDataset, load_ltr_dataset, numeric_feature_columns
from .metrics import summarize_oof
from .split import assert_no_group_overlap, group_k_fold
from .train import fit_linear_ranker


def generate_oof_scores(
    frame: pd.DataFrame,
    *,
    label_column: str = "label",
    group_column: str = "query_id",
    n_splits: int = 5,
    seed: int = 13,
) -> pd.DataFrame:
    frame = frame.copy()
    feature_columns = numeric_feature_columns(frame, label_column=label_column, group_column=group_column)
    scores = np.zeros(len(frame), dtype=float)
    fold_ids = np.full(len(frame), -1, dtype=int)
    splits = group_k_fold(frame[group_column].astype(str).tolist(), n_splits=n_splits, seed=seed)
    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        assert_no_group_overlap(frame[group_column].astype(str).tolist(), train_idx, test_idx)
        train_ds = LTRDataset(
            frame=frame.iloc[train_idx].copy(),
            feature_columns=feature_columns,
            label_column=label_column,
            group_column=group_column,
        )
        model = fit_linear_ranker(train_ds)
        X_test = frame.iloc[test_idx][feature_columns].astype(float).fillna(0.0).to_numpy(dtype=float)
        scores[test_idx] = model.predict(X_test)
        fold_ids[test_idx] = fold_idx
    out = frame.copy()
    out["ltr_score"] = scores
    out["oof_fold"] = fold_ids
    out["oof_group_column"] = group_column
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate PRISM no-leakage OOF LTR scores")
    parser.add_argument("--features", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()
    dataset = load_ltr_dataset(args.features)
    out = generate_oof_scores(dataset.frame, n_splits=args.folds, seed=args.seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False)
    summary = summarize_oof(out)
    print(json.dumps({"output": str(output), "rows": len(out), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
