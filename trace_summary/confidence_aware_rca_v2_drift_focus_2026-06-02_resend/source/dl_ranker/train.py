"""Training loop for OnsetAwareSetTransformer with case-fold cross-validation."""
from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from dl_ranker.dataset import CaseDataset, collate_variable
from dl_ranker.model import OnsetAwareSetTransformer


def _set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_one_epoch(
    model: OnsetAwareSetTransformer,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    device: str,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        features = batch["features"].to(device)
        onset_ranks = batch["onset_ranks"].to(device)
        labels = batch["labels"].to(device)
        padding_mask = batch["padding_mask"].to(device)

        logits = model(features, onset_ranks, padding_mask)
        # Mask padding and use cross-entropy with the FIRST positive as target class.
        # (All positives represent the same GT component so any one is valid.)
        logits = logits.masked_fill(padding_mask, float("-inf"))
        target = labels.argmax(dim=1)  # first positive index, 0 if all negative
        loss = F.cross_entropy(logits, target, reduction="mean")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def _ndcg_at_k(scores: torch.Tensor, labels: torch.Tensor, k: int) -> float:
    _, idx = scores.sort(descending=True)
    dcg = 0.0
    for i, j in enumerate(idx[:k]):
        dcg += labels[j].item() / math.log2(i + 2)
    ideal_labels = labels.sort(descending=True)[0]
    idcg = 0.0
    for i in range(min(k, len(ideal_labels))):
        idcg += ideal_labels[i].item() / math.log2(i + 2)
    return dcg / idcg if idcg > 0 else 0.0


def evaluate(
    model: OnsetAwareSetTransformer,
    loader: DataLoader,
    device: str,
) -> dict[str, float]:
    model.eval()
    correct_top1 = 0
    correct_top3 = 0
    correct_top5 = 0
    total = 0
    ndcg1_sum = 0.0
    ndcg3_sum = 0.0
    mrr_sum = 0.0

    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device)
            onset_ranks = batch["onset_ranks"].to(device)
            labels = batch["labels"].to(device)
            padding_mask = batch["padding_mask"].to(device)

            probs = model.predict_proba(features, onset_ranks, padding_mask)

            for i in range(probs.shape[0]):
                n_valid = int((~padding_mask[i]).sum().item())
                if n_valid == 0:
                    continue
                p = probs[i, :n_valid]
                lab = labels[i, :n_valid]
                pos_idx = (lab == 1).nonzero(as_tuple=False)
                if len(pos_idx) == 0:
                    continue
                pos_set = set(pos_idx.flatten().tolist())
                _, top_idx = p.sort(descending=True)
                top = top_idx[:5].tolist()

                if top[0] in pos_set:
                    correct_top1 += 1
                if pos_set & {t for t in top[:3]}:
                    correct_top3 += 1
                if pos_set & {t for t in top[:5]}:
                    correct_top5 += 1

                # Create binary relevance for ranking metrics
                rel = torch.zeros_like(p)
                rel[list(pos_set)] = 1.0
                ndcg1_sum += _ndcg_at_k(p, rel, 1)
                ndcg3_sum += _ndcg_at_k(p, rel, 3)

                # MRR: best (lowest) rank among positives
                min_rank = min(
                    (p.argsort(descending=True) == idx).nonzero(as_tuple=False)[0].item()
                    for idx in pos_set
                )
                mrr_sum += 1.0 / (min_rank + 1)

                total += 1

    return {
        "acc@1": correct_top1 / total if total > 0 else 0.0,
        "acc@3": correct_top3 / total if total > 0 else 0.0,
        "acc@5": correct_top5 / total if total > 0 else 0.0,
        "ndcg@1": ndcg1_sum / total if total > 0 else 0.0,
        "ndcg@3": ndcg3_sum / total if total > 0 else 0.0,
        "mrr": mrr_sum / total if total > 0 else 0.0,
        "n": total,
    }


def train_case_fold(
    data_dir: str | Path,
    out_dir: str | Path,
    train_ids: list[str],
    val_ids: list[str],
    pretrained_encoder_path: str | None = None,
    n_epochs: int = 100,
    batch_size: int = 8,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 15,
    device: str = "cpu",
    fold_name: str = "fold0",
) -> dict[str, Any]:
    _set_seed(42)

    train_ds = CaseDataset(data_dir, train_ids, augment=True)
    val_ds = CaseDataset(data_dir, val_ids, augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_variable
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size * 2, shuffle=False, collate_fn=collate_variable
    )

    model = OnsetAwareSetTransformer(
        n_features=33, d_model=64, n_heads=4, n_layers=2,
        dropout=0.3,
    ).to(device)

    if pretrained_encoder_path and Path(pretrained_encoder_path).exists():
        state = torch.load(pretrained_encoder_path, weights_only=True, map_location="cpu")
        model.feature_encoder.load_state_dict(state, strict=False)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    best_metric = -1.0
    best_state: dict[str, torch.Tensor] = {}
    pat_counter = 0
    history: list[dict] = []

    for epoch in range(n_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        scheduler.step()
        val_metrics = evaluate(model, val_loader, device)

        history.append({"epoch": epoch, "train_loss": train_loss, **val_metrics})

        current = val_metrics["acc@1"]

        if current > best_metric + 1e-4:
            best_metric = current
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            pat_counter = 0
        else:
            pat_counter += 1

        if pat_counter >= patience:
            break

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / f"model_{fold_name}.pt"
    metrics_path = out_dir / f"metrics_{fold_name}.json"

    torch.save(best_state, model_path)
    with open(metrics_path, "w") as f:
        json.dump({
            "best_val_acc@1": best_metric,
            "history": history,
            "n_train": len(train_ids),
            "n_val": len(val_ids),
        }, f, indent=2)

    return {
        "best_val_acc@1": best_metric,
        "model_path": str(model_path),
        "n_train": len(train_ids),
        "n_val": len(val_ids),
    }


def main_casefold_train(
    data_dir: str | Path,
    out_dir: str | Path,
    pretrained_encoder_path: str | None = None,
    n_folds: int = 2,
    n_epochs: int = 100,
    device: str = "cpu",
) -> list[dict[str, Any]]:
    data_dir = Path(data_dir)
    all_ids = sorted([p.stem.replace("case_", "") for p in data_dir.glob("case_*.pt")])
    if not all_ids:
        raise FileNotFoundError(f"No case_*.pt files in {data_dir}")

    chunk = (len(all_ids) + n_folds - 1) // n_folds
    results = []

    for fold in range(n_folds):
        val_start = fold * chunk
        val_end = min(val_start + chunk, len(all_ids))
        val_ids = all_ids[val_start:val_end]
        train_ids = all_ids[:val_start] + all_ids[val_end:]
        fold_name = f"fold{fold}"

        result = train_case_fold(
            data_dir=data_dir,
            out_dir=out_dir,
            train_ids=train_ids,
            val_ids=val_ids,
            pretrained_encoder_path=pretrained_encoder_path,
            n_epochs=n_epochs,
            device=device,
            fold_name=fold_name,
        )
        results.append(result)

    avg = sum(r["best_val_acc@1"] for r in results) / len(results)
    summary = {"folds": results, "avg_acc@1": avg}
    with open(Path(out_dir) / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return results
