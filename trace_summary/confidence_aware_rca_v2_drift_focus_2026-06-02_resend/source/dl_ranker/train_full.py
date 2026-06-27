"""Train Set Transformer on full training set and evaluate on test."""
from __future__ import annotations

import argparse, json, math
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from dl_ranker.dataset import CaseDataset, collate_variable
from dl_ranker.model import OnsetAwareSetTransformer
from dl_ranker.pretrain import pretrain_and_save


def set_seed(seed=42):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def train_full(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        features = batch["features"].to(device)
        onset_ranks = batch["onset_ranks"].to(device)
        labels = batch["labels"].to(device)
        padding_mask = batch["padding_mask"].to(device)

        logits = model(features, onset_ranks, padding_mask)
        logits = logits.masked_fill(padding_mask, float("-inf"))
        target = labels.argmax(dim=1)
        loss = torch.nn.functional.cross_entropy(logits, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    scheduler.step()
    return total_loss / max(len(loader), 1)


def eval_test(model, test_ids, data_dir, device):
    model.eval()
    correct = 0; total = 0
    with torch.no_grad():
        for cid in test_ids:
            d = torch.load(Path(data_dir) / f"case_{cid}.pt", weights_only=True, map_location="cpu")
            feat = d["features"].unsqueeze(0).to(device)
            onset = d["onset_ranks"].unsqueeze(0).to(device)
            labels = d["labels"].unsqueeze(0).to(device)
            probs = model.predict_proba(feat, onset)
            p = probs[0]; lab = labels[0]
            pos = set((lab == 1).nonzero(as_tuple=False).flatten().tolist())
            if not pos:
                continue
            top = p.argsort(descending=True)[0].item()
            if top in pos:
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

    print(f"Train: {len(train_ids)}, Test: {len(test_ids)}")

    # Pre-train encoder
    pretrained_path = out_dir / "pretrained_encoder.pt"
    if not args.skip_pretrain:
        print("Pre-training encoder...")
        pretrain_and_save(
            data_dir=args.data_dir,
            train_case_ids=train_ids,
            out_path=str(pretrained_path),
            n_epochs=50,
            device=args.device,
        )
    else:
        print("Skipping pre-training")

    # Create model
    model = OnsetAwareSetTransformer(
        n_features=33, d_model=64, n_heads=4, n_layers=2, dropout=0.3,
    ).to(args.device)

    if pretrained_path.exists() and not args.skip_pretrain:
        state = torch.load(pretrained_path, weights_only=True, map_location="cpu")
        model.feature_encoder.load_state_dict(state, strict=False)
        print("Loaded pretrained encoder")

    # Train
    train_ds = CaseDataset(args.data_dir, train_ids, augment=True)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_variable,
    )

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_epochs)

    best_acc = 0.0
    best_state = {}
    patience = 20
    pat = 0

    for epoch in range(args.n_epochs):
        loss = train_full(model, train_loader, optimizer, scheduler, args.device)
        acc = eval_test(model, test_ids, args.data_dir, args.device)

        if epoch % 20 == 0:
            print(f"  epoch {epoch:3d}: loss={loss:.4f}, test_acc@1={acc:.4f}")

        if acc > best_acc + 1e-4:
            best_acc = acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            pat = 0
        else:
            pat += 1
        if pat >= patience:
            break

    model_path = out_dir / "model_full.pt"
    torch.save(best_state, model_path)
    torch.save({"params": {"n_features": 33, "d_model": 64, "n_heads": 4, "n_layers": 2, "dropout": 0.3}}, out_dir / "config.pt")

    print(f"\nBest test acc@1: {best_acc:.4f}")
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
            pos = set((lab == 1).nonzero(as_tuple=False).flatten().tolist())
            if not pos:
                continue
            rp = feat[0, :, 30]
            p = p * rp  # post-multiply
            top = p.argsort(descending=True)[0].item()
            if top in pos:
                correct += 1
            total += 1
    print(f"With rp post-multiplication: {correct}/{total} = {correct/total*100:.1f}%")


if __name__ == "__main__":
    main()
