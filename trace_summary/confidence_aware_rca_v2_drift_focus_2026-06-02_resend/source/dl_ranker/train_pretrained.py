"""Train Set Transformer with pre-training: unsupervised on all cases, supervised on train."""
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


def train_epoch(model, loader, optimizer, device):
    model.train()
    total = 0.0
    for batch in loader:
        feat = batch["features"].to(device)
        onset = batch["onset_ranks"].to(device)
        labels = batch["labels"].to(device)
        pmask = batch["padding_mask"].to(device)

        logits = model(feat, onset, pmask)
        logits = logits.masked_fill(pmask, float("-inf"))
        target = labels.argmax(dim=1)
        loss = F.cross_entropy(logits, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / max(len(loader), 1)


def evaluate(model, data_dir, case_ids, device, use_rp=False):
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
            if use_rp:
                rp = feat[0, :, 30]
                p = p * rp
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
    parser.add_argument("--n-pretrain-epochs", type=int, default=80)
    parser.add_argument("--n-train-epochs", type=int, default=120)
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

    # Phase 1: Pre-train encoder on ALL cases (unsupervised)
    pretrained_path = out_dir / "pretrained_encoder.pt"
    if not args.skip_pretrain:
        all_ids = sorted(set(train_ids + test_ids))
        print(f"\n=== Pre-training encoder on {len(all_ids)} cases ===")
        pretrain_and_save(
            data_dir=args.data_dir,
            case_ids=all_ids,
            out_path=str(pretrained_path),
            n_epochs=args.n_pretrain_epochs,
            device=args.device,
        )
        print(f"Pre-trained encoder saved to {pretrained_path}")

    # Phase 2: Train Set Transformer on train cases
    print(f"\n=== Training Set Transformer on {len(train_ids)} cases ===")
    model = OnsetAwareSetTransformer(
        n_features=33, d_model=64, n_heads=4, n_layers=2, dropout=0.3,
    ).to(args.device)

    if pretrained_path.exists():
        state = torch.load(pretrained_path, weights_only=True, map_location="cpu")
        model.feature_encoder.load_state_dict(state, strict=True)
        print("Loaded pre-trained encoder weights")

    train_ds = CaseDataset(args.data_dir, train_ids, augment=True)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_variable,
    )

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_train_epochs)

    best_acc = 0.0; best_rp_acc = 0.0
    best_state = {}; best_state_rp = {}
    patience = 25; pat = 0

    for epoch in range(args.n_train_epochs):
        loss = train_epoch(model, train_loader, optimizer, args.device)
        scheduler.step()
        acc = evaluate(model, args.data_dir, test_ids, args.device, use_rp=False)
        acc_rp = evaluate(model, args.data_dir, test_ids, args.device, use_rp=True)

        if epoch % 10 == 0:
            print(f"  epoch {epoch:3d}: loss={loss:.4f}, acc={acc:.4f}, acc+rp={acc_rp:.4f}")

        if acc > best_acc + 1e-4:
            best_acc = acc; best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            pat = 0
        else:
            pat += 1
        if acc_rp > best_rp_acc + 1e-4:
            best_rp_acc = acc_rp; best_state_rp = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if pat >= patience:
            break

    for key, st, met, label in [("model_best.pt", best_state, best_acc, "CE"), ("model_best_rp.pt", best_state_rp, best_rp_acc, "CE+rp")]:
        if st:
            torch.save(st, out_dir / key)
            print(f"\n  Best {label} test acc@1 = {met:.4f} -> {out_dir / key}")

    torch.save({"n_features": 33, "d_model": 64, "n_heads": 4, "n_layers": 2, "dropout": 0.3}, out_dir / "config.pt")
    results = {"best_acc@1": best_acc, "best_acc@1_rp": best_rp_acc}
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    main()
