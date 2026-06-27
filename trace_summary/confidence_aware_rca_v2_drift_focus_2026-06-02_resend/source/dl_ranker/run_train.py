"""Main training entry point for Set Transformer ranker.

Usage:
    PYTHONPATH=.:.. python3 dl_ranker/run_train.py \
        --data-dir dl_ranker_data/ \
        --out-dir dl_ranker_models/ \
        --cases-csv logs/portable_exp/aiops2021_metric_adaptive_inputs_tzfix/cases.csv \
        [--skip-pretrain] [--no-onset-bias] [--n-folds 2]
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import torch

from dl_ranker.pretrain import pretrain_and_save
from dl_ranker.train import main_casefold_train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--cases-csv", required=True)
    parser.add_argument("--skip-pretrain", action="store_true")
    parser.add_argument("--n-folds", type=int, default=2)
    parser.add_argument("--n-epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load splits
    splits_path = data_dir / "splits.json"
    if splits_path.exists():
        with open(splits_path) as f:
            splits = json.load(f)
        train_ids = splits["train"]
        test_ids = splits["test"]
    else:
        import pandas as pd
        cases = pd.read_csv(args.cases_csv)
        train_ids = cases[cases["data_type"] == "train"]["case_id"].tolist()
        test_ids = cases[cases["data_type"] == "test"]["case_id"].tolist()

    print(f"Train cases: {len(train_ids)}, Test cases: {len(test_ids)}")

    # P1: Pre-training
    pretrained_path = out_dir / "pretrained_encoder.pt"
    if not args.skip_pretrain:
        print("\n=== Pre-training encoder ===")
        pretrain_and_save(
            data_dir=str(data_dir),
            case_ids=train_ids,  # unsupervised, only train cases
            out_path=str(pretrained_path),
            n_epochs=50,
            device=args.device,
        )
        print(f"Pre-trained encoder saved to {pretrained_path}")
    else:
        print("\n=== Skipping pre-training ===")

    # P2: Case-fold training
    print(f"\n=== Training {args.n_folds}-fold ===")
    pretrained = str(pretrained_path) if pretrained_path.exists() else None
    if args.skip_pretrain:
        pretrained = None

    results = main_casefold_train(
        data_dir=str(data_dir),
        out_dir=str(out_dir),
        pretrained_encoder_path=pretrained,
        n_folds=args.n_folds,
        n_epochs=args.n_epochs,
        device=args.device,
    )

    # Print results
    print("\n=== Results ===")
    for i, r in enumerate(results):
        print(f"  Fold {i}: best_val_acc@1 = {r['best_val_acc@1']:.4f}, n_train={r['n_train']}, n_val={r['n_val']}")
    avg = sum(r["best_val_acc@1"] for r in results) / len(results)
    print(f"  Average: {avg:.4f}")

    return avg


if __name__ == "__main__":
    main()
