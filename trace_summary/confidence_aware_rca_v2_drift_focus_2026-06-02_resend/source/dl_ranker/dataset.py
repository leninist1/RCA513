"""PyTorch Dataset for Set Transformer training with data augmentation."""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def collate_variable(padded_cases: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]]) -> dict[str, torch.Tensor]:
    """Custom collate that pads variable-length candidate sets.

    Each element: (features_Nx33, onset_ranks_N, labels_N, original_length)
    Returns batched dict with padding masks.
    """
    max_n = max(f.shape[0] for f, _, _, _ in padded_cases)
    B = len(padded_cases)
    n_feat = padded_cases[0][0].shape[1]

    batched_feat = torch.zeros(B, max_n, n_feat)
    batched_onset = torch.zeros(B, max_n, dtype=torch.long)
    batched_labels = torch.zeros(B, max_n, dtype=torch.long)
    batched_mask = torch.ones(B, max_n, dtype=torch.bool)

    for i, (feat, onset, labels, _) in enumerate(padded_cases):
        n = feat.shape[0]
        batched_feat[i, :n] = feat
        batched_onset[i, :n] = onset
        batched_labels[i, :n] = labels
        batched_mask[i, :n] = False

    return {
        "features": batched_feat,
        "onset_ranks": batched_onset,
        "labels": batched_labels,
        "padding_mask": batched_mask,
    }


class CaseDataset(Dataset):
    """Dataset of RCA cases, each containing a variable number of candidates."""

    def __init__(
        self,
        data_dir: str | Path,
        case_ids: list[str] | None = None,
        augment: bool = False,
        aug_candidate_dropout: float = 0.1,
        aug_feature_noise: float = 0.05,
        aug_onset_jitter: int = 1,
    ):
        self.data_dir = Path(data_dir)
        self.augment = augment
        self.aug_candidate_dropout = aug_candidate_dropout
        self.aug_feature_noise = aug_feature_noise
        self.aug_onset_jitter = aug_onset_jitter

        if case_ids is not None:
            self.case_ids = case_ids
        else:
            self.case_ids = sorted([
                p.stem.replace("case_", "")
                for p in self.data_dir.glob("case_*.pt")
            ])

    def __len__(self) -> int:
        return len(self.case_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        cid = self.case_ids[idx]
        path = self.data_dir / f"case_{cid}.pt"
        data = torch.load(path, weights_only=True, map_location="cpu")
        features = data["features"]
        onset_ranks = data["onset_ranks"]
        labels = data["labels"]

        if self.augment:
            features, onset_ranks, labels = self._augment(features, onset_ranks, labels)

        return features, onset_ranks, labels, features.shape[0]

    def _augment(
        self,
        features: torch.Tensor,
        onset_ranks: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n = features.shape[0]
        if n == 0:
            return features, onset_ranks, labels

        keep_mask = torch.ones(n, dtype=torch.bool)
        if self.aug_candidate_dropout > 0 and n > 10:
            for i in range(n):
                if labels[i] != 1 and random.random() < self.aug_candidate_dropout:
                    keep_mask[i] = False

        noise_mask = torch.rand_like(features) < self.aug_feature_noise
        noise = torch.randn_like(features) * 0.1
        features = features + noise * noise_mask.float()

        if self.aug_onset_jitter > 0:
            jitter = torch.randint(
                -self.aug_onset_jitter, self.aug_onset_jitter + 1, onset_ranks.shape
            )
            onset_ranks = onset_ranks + jitter
            onset_ranks = onset_ranks.clamp(0, 99)

        if not keep_mask.all():
            features = features[keep_mask]
            onset_ranks = onset_ranks[keep_mask]
            labels = labels[keep_mask]

        return features, onset_ranks, labels


def load_case_data(data_dir: str | Path, case_id: str) -> dict[str, torch.Tensor]:
    path = Path(data_dir) / f"case_{case_id}.pt"
    return torch.load(path, weights_only=True, map_location="cpu")
