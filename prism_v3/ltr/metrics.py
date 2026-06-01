"""Ranking metrics for OOF LTR diagnostics."""

from __future__ import annotations

from typing import Dict, Iterable, Sequence
import math

import pandas as pd


def recall_at_k(frame: pd.DataFrame, *, k: int, score_col: str = "ltr_score", label_col: str = "label") -> float:
    total = 0
    hit = 0
    for _, group in frame.groupby("query_id"):
        if group[label_col].astype(float).max() <= 0:
            continue
        total += 1
        top = group.sort_values(score_col, ascending=False).head(int(k))
        hit += int(top[label_col].astype(float).max() > 0)
    return hit / total if total else 0.0


def ndcg_at_k(frame: pd.DataFrame, *, k: int, score_col: str = "ltr_score", label_col: str = "label") -> float:
    scores = []
    for _, group in frame.groupby("query_id"):
        labels = group.sort_values(score_col, ascending=False).head(int(k))[label_col].astype(float).tolist()
        ideal = sorted(group[label_col].astype(float).tolist(), reverse=True)[: int(k)]
        dcg = _dcg(labels)
        idcg = _dcg(ideal)
        if idcg > 0:
            scores.append(dcg / idcg)
    return sum(scores) / len(scores) if scores else 0.0


def summarize_oof(frame: pd.DataFrame, *, score_col: str = "ltr_score", label_col: str = "label") -> Dict[str, float]:
    return {
        "recall@5": recall_at_k(frame, k=5, score_col=score_col, label_col=label_col),
        "recall@10": recall_at_k(frame, k=10, score_col=score_col, label_col=label_col),
        "ndcg@5": ndcg_at_k(frame, k=5, score_col=score_col, label_col=label_col),
        "ndcg@10": ndcg_at_k(frame, k=10, score_col=score_col, label_col=label_col),
    }


def _dcg(labels: Sequence[float]) -> float:
    return sum((2.0 ** float(label) - 1.0) / math.log2(idx + 2.0) for idx, label in enumerate(labels))
