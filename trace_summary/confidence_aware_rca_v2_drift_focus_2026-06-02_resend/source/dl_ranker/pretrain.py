"""Self-supervised pre-training via masked feature reconstruction.

Pre-trains the feature encoder on per-component data from training cases,
so it learns feature correlations before fine-tuning on the ranking task.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path
from typing import Any


class MaskedFeatureEncoder(nn.Module):
    """Encoder + decoder for masked feature reconstruction pre-training.

    The encoder matches OnsetAwareSetTransformer.feature_encoder exactly so
    that the pretrained weights can be loaded directly.
    """

    def __init__(self, n_features: int = 33, d_model: int = 64, dropout: float = 0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_features, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, n_features),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (reconstruction, embedding)."""
        mask = torch.rand_like(x) < 0.15
        x_masked = x.clone()
        x_masked[mask] = 0.0
        h = self.encoder(x_masked)
        x_recon = self.decoder(h)
        return x_recon, h


def pretrain_encoder(
    case_ids: list[str],
    data_dir: str,
    n_epochs: int = 80,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str = "cpu",
) -> dict[str, torch.Tensor]:
    """Pre-train the feature encoder via masked feature reconstruction.

    Uses per-candidate feature vectors from ALL provided cases (no labels),
    so the encoder learns feature correlations before fine-tuning.
    """
    import numpy as np
    all_features = []
    for cid in case_ids:
        path = Path(data_dir) / f"case_{cid}.pt"
        if not path.exists():
            continue
        data = torch.load(path, weights_only=True, map_location="cpu")
        feat = data["features"].numpy()
        if len(feat) > 0:
            all_features.append(feat)
    if not all_features:
        return {}

    X = torch.from_numpy(np.concatenate(all_features, axis=0)).float().to(device)
    n_samples = X.shape[0]

    model = MaskedFeatureEncoder(n_features=33, d_model=64, dropout=0.1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] = {}
    patience = 15
    pat_counter = 0

    for epoch in range(n_epochs):
        perm = torch.randperm(n_samples)
        total_loss = 0.0
        n_batches = 0

        for i in range(0, n_samples, batch_size):
            idx = perm[i : i + batch_size]
            batch = X[idx]

            x_recon, _ = model(batch)
            mask = torch.rand_like(batch) < 0.15
            if mask.sum() == 0:
                continue
            loss = F.mse_loss(x_recon[mask], batch[mask])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)

        if (epoch + 1) % 10 == 0:
            pass

        if avg_loss < best_loss - 1e-6:
            best_loss = avg_loss
            best_state = {k: v.cpu().clone() for k, v in model.encoder.state_dict().items()}
            pat_counter = 0
        else:
            pat_counter += 1

        if pat_counter >= patience:
            break

    return best_state


def pretrain_and_save(
    data_dir: str | Path,
    case_ids: list[str],
    out_path: str | Path,
    n_epochs: int = 80,
    device: str = "cpu",
) -> None:
    """Full pre-training pipeline."""
    import json
    state = pretrain_encoder(
        case_ids=case_ids,
        data_dir=str(data_dir),
        n_epochs=n_epochs,
        device=device,
    )
    if state:
        torch.save(state, out_path)
