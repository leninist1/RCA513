"""Incident-grouped splits for no-leakage LTR."""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple
import random

import numpy as np


def group_k_fold(
    groups: Sequence[str],
    *,
    n_splits: int = 5,
    seed: int = 13,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    groups = [str(group) for group in groups]
    unique = sorted(set(groups))
    if not unique:
        return []
    n_splits = max(2, min(int(n_splits), len(unique)))
    rng = random.Random(seed)
    rng.shuffle(unique)
    fold_groups = [set() for _ in range(n_splits)]
    for idx, group in enumerate(unique):
        fold_groups[idx % n_splits].add(group)
    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    group_array = np.array(groups, dtype=object)
    indices = np.arange(len(groups))
    for held_out in fold_groups:
        test_mask = np.array([group in held_out for group in group_array], dtype=bool)
        train_idx = indices[~test_mask]
        test_idx = indices[test_mask]
        if train_idx.size == 0 or test_idx.size == 0:
            continue
        splits.append((train_idx, test_idx))
    return splits


def assert_no_group_overlap(groups: Sequence[str], train_idx: Iterable[int], test_idx: Iterable[int]) -> None:
    group_array = np.array([str(group) for group in groups], dtype=object)
    train_groups = set(group_array[list(train_idx)])
    test_groups = set(group_array[list(test_idx)])
    overlap = train_groups & test_groups
    if overlap:
        raise ValueError("group split leakage: " + ", ".join(sorted(overlap)[:10]))
