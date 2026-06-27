"""Train Set Transformer with heuristic-guided residual loss.

Core idea: the heuristic is used as a TEACHER, but only when it's correct.
  - When heuristic's top-1 is correct → penalty for deviating from it
  - When heuristic's top-1 is wrong → model is free to disagree

Result: the model learns to respect the heuristic's domain knowledge where it
works, while using rich features to correct where the heuristic fails.
"""
from __future__ import annotations

import argparse, json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from dl_ranker.dataset import CaseDataset, collate_variable
from dl_ranker.model import OnsetAwareSetTransformer
from dl_ranker.pretrain import pretrain_and_save


def set_seed(seed=42):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def heuristic_guided_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    h_idx: torch.Tensor,
    h_correct: torch.Tensor,
    alpha: float = 0.3,
    margin: float = 0.1,
) -> torch.Tensor:
    """Heuristic-guided ranking loss.

    Args:
        logits: (B, N) model scores
        labels: (B, N) ground truth (1 for correct candidates)
        h_idx: (B,) index of heuristic's top-1 candidate (-1 if not found)
        h_correct: (B,) 1 if heuristic's top-1 matches GT, 0 otherwise
        alpha: weight of heuristic agreement term
        margin: how much slack to allow before penalizing

    Returns:
        scalar loss
    """
    B, N = logits.shape

    # Primary: cross-entropy (first positive = target class)
    target = labels.argmax(dim=1)
    ce_loss = F.cross_entropy(logits.float(), target)

    # Residual: heuristic agreement
    valid = (h_idx >= 0) & h_correct.bool()
    if valid.any():
        vh_idx = h_idx[valid]
        vh_logits = logits[valid]

        # Model's top score per case
        model_top = vh_logits.max(dim=1).values
        # Model's score for heuristic's top-1 candidate
        model_h_score = vh_logits[range(len(vh_idx)), vh_idx]

        # Penalty: heuristic's correct answer should be close to model's top
        # relu(model_top - model_h_score - margin): if model's score for h's
        # answer is far below its own top score, apply penalty
        agreement = F.relu(model_top - model_h_score - margin)
        h_loss = agreement.mean() * alpha
    else:
        h_loss = torch.tensor(0.0, device=logits.device)

    return ce_loss + h_loss


class ResidualDataset(CaseDataset):
    """Extended dataset that also returns heuristic metadata."""

    def __init__(self, data_dir, case_ids, augment=False, **kwargs):
        super().__init__(data_dir, case_ids, augment, **kwargs)

    def __getitem__(self, idx):
        cid = self.case_ids[idx]
        path = Path(self.data_dir) / f"case_{cid}.pt"
        data = torch.load(path, weights_only=True, map_location="cpu")
        features = data["features"]
        onset_ranks = data["onset_ranks"]
        labels = data["labels"]
        h_idx = torch.tensor(data.get("h_idx", -1), dtype=torch.long)
        h_correct = torch.tensor(data.get("h_correct", 0), dtype=torch.long)

        if self.augment:
            features, onset_ranks, labels = self._augment(features, onset_ranks, labels)
            h_idx = torch.clamp(h_idx, -1, features.shape[0] - 1)

        return features, onset_ranks, labels, h_idx, h_correct, features.shape[0]


def residual_collate(padded_cases):
    """Collate with heuristic metadata."""
    max_n = max(f.shape[0] for f, _, _, _, _, _ in padded_cases)
    B = len(padded_cases)
    n_feat = padded_cases[0][0].shape[1]

    batched_feat = torch.zeros(B, max_n, n_feat)
    batched_onset = torch.zeros(B, max_n, dtype=torch.long)
    batched_labels = torch.zeros(B, max_n, dtype=torch.long)
    batched_mask = torch.ones(B, max_n, dtype=torch.bool)
    batched_h_idx = torch.full((B,), -1, dtype=torch.long)
    batched_h_correct = torch.zeros(B, dtype=torch.long)

    for i, (feat, onset, labels, h_idx, h_correct, _) in enumerate(padded_cases):
        n = feat.shape[0]
        batched_feat[i, :n] = feat
        batched_onset[i, :n] = onset
        batched_labels[i, :n] = labels
        batched_mask[i, :n] = False
        batched_h_idx[i] = h_idx
        batched_h_correct[i] = h_correct

    return {
        "features": batched_feat, "onset_ranks": batched_onset,
        "labels": batched_labels, "padding_mask": batched_mask,
        "h_idx": batched_h_idx, "h_correct": batched_h_correct,
    }


def train_epoch(model, loader, optimizer, device, alpha, margin):
    model.train()
    total = 0.0
    for batch in loader:
        feat = batch["features"].to(device)
        onset = batch["onset_ranks"].to(device)
        labels = batch["labels"].to(device)
        pmask = batch["padding_mask"].to(device)
        h_idx = batch["h_idx"].to(device)
        h_correct = batch["h_correct"].to(device)

        logits = model(feat, onset, pmask)
        logits = logits.masked_fill(pmask, float("-inf"))
        loss = heuristic_guided_loss(logits, labels, h_idx, h_correct, alpha, margin)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / max(len(loader), 1)


def evaluate(model, data_dir, case_ids, device):
    model.eval()
    correct = 0; total = 0
    with torch.no_grad():
        for cid in case_ids:
            d = torch.load(Path(data_dir) / f"case_{cid}.pt", weights_only=True, map_location="cpu")
            feat = d["features"].unsqueeze(0).to(device)
            onset = d["onset_ranks"].unsqueeze(0).to(device)
            labels = d["labels"].unsqueeze(0).to(device)
            probs = model.predict_proba(feat, onset)
            p = probs[0]; lab = labels[0]
            pos_set = set((lab == 1).nonzero(as_tuple=False).flatten().tolist())
            if not pos_set:
                continue
            top = p.argsort(descending=True)[0].item()
            if top in pos_set:
                correct += 1
            total += 1
    return correct / total if total > 0 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--splits-json", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--n-epochs", type=int, default=120)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=0.3, help="heuristic agreement weight")
    parser.add_argument("--margin", type=float, default=0.1, help="agreement margin")
    parser.add_argument("--skip-pretrain", action="store_true")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    set_seed(42)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.splits_json) as f:
        splits = json.load(f)
    train_ids = splits["train"]
    test_ids = splits["test"]

    # Pre-train on all cases
    pretrained_path = out_dir / "pretrained_encoder.pt"
    if not args.skip_pretrain:
        all_ids = sorted(set(train_ids + test_ids))
        print(f"Pre-training on {len(all_ids)} cases...")
        pretrain_and_save(
            data_dir=args.data_dir, case_ids=all_ids,
            out_path=str(pretrained_path), n_epochs=80, device=args.device,
        )

    # Train
    model = OnsetAwareSetTransformer(
        n_features=34, d_model=64, n_heads=4, n_layers=2, dropout=0.3,
    ).to(args.device)

    if pretrained_path.exists():
        state = torch.load(pretrained_path, weights_only=True, map_location="cpu")
        pretrain_enc = state
        current_enc = dict(model.feature_encoder.state_dict())
        for k in list(pretrain_enc.keys()):
            if k not in current_enc or pretrain_enc[k].shape != current_enc[k].shape:
                pretrain_enc.pop(k)
        model.feature_encoder.load_state_dict(pretrain_enc, strict=False)

    train_ds = ResidualDataset(args.data_dir, train_ids, augment=True)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=residual_collate,
    )

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_epochs)

    best_acc = 0.0; best_state = {}
    patience = 25; pat = 0

    for epoch in range(args.n_epochs):
        loss = train_epoch(model, train_loader, optimizer, args.device, args.alpha, args.margin)
        scheduler.step()
        acc = evaluate(model, args.data_dir, test_ids, args.device)

        if epoch % 10 == 0:
            print(f"  epoch {epoch:3d}: loss={loss:.4f}, test_acc@1={acc:.4f}")

        if acc > best_acc + 1e-4:
            best_acc = acc; best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            pat = 0
        else:
            pat += 1
        if pat >= patience:
            break

    model_path = out_dir / "model_residual.pt"
    torch.save(best_state, model_path)
    torch.save({"n_features": 34, "d_model": 64, "n_heads": 4, "n_layers": 2, "dropout": 0.3, "alpha": args.alpha, "margin": args.margin}, out_dir / "config.pt")

    print(f"\nBest test acc@1 = {best_acc:.4f}")
    print(f"Model saved to {model_path}")

    # Also evaluate with rp post-multiplication
    model.load_state_dict(best_state)
    model.eval()
    correct = 0; total = 0
    with torch.no_grad():
        for cid in test_ids:
            d = torch.load(Path(args.data_dir) / f"case_{cid}.pt", weights_only=True, map_location="cpu")
            feat = d["features"].unsqueeze(0).to(args.device)
            onset = d["onset_ranks"].unsqueeze(0).to(args.device)
            labels = d["labels"].unsqueeze(0).to(args.device)
            probs = model.predict_proba(feat, onset)
            p = probs[0]; lab = labels[0]
            rp = feat[0, :, 30]
            p = p * rp
            pos = set((lab == 1).nonzero(as_tuple=False).flatten().tolist())
            if not pos:
                continue
            top = p.argsort(descending=True)[0].item()
            if top in pos:
                correct += 1
            total += 1
    print(f"With rp post-mult: {correct}/{total} = {correct/total*100:.1f}%")


if __name__ == "__main__":
    main()
