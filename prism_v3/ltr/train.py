"""Small, dependency-light LTR training utilities."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

from .dataset import LTRDataset, load_ltr_dataset


@dataclass
class LinearRanker:
    feature_columns: List[str]
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    bias: float = 0.0

    def predict(self, X: np.ndarray) -> np.ndarray:
        if X.size == 0:
            return np.zeros((X.shape[0],), dtype=float)
        Xn = (X - self.mean) / self.scale
        return Xn @ self.weights + float(self.bias)

    def to_json(self) -> Dict[str, Any]:
        return {
            "model_type": "linear_ranker_logistic",
            "feature_columns": list(self.feature_columns),
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "weights": self.weights.tolist(),
            "bias": float(self.bias),
        }

    @classmethod
    def from_json(cls, payload: Dict[str, Any]) -> "LinearRanker":
        return cls(
            feature_columns=[str(x) for x in payload.get("feature_columns", [])],
            mean=np.array(payload.get("mean", []), dtype=float),
            scale=np.array(payload.get("scale", []), dtype=float),
            weights=np.array(payload.get("weights", []), dtype=float),
            bias=float(payload.get("bias", 0.0)),
        )


def fit_linear_ranker(
    dataset: LTRDataset,
    *,
    lr: float = 0.05,
    epochs: int = 250,
    l2: float = 1e-3,
) -> LinearRanker:
    X = dataset.X
    y = dataset.y
    if X.size == 0 or X.shape[1] == 0:
        return LinearRanker(dataset.feature_columns, np.zeros((0,)), np.ones((0,)), np.zeros((0,)), 0.0)
    mean = X.mean(axis=0)
    scale = X.std(axis=0)
    scale[scale < 1e-9] = 1.0
    Xn = (X - mean) / scale
    weights = np.zeros(Xn.shape[1], dtype=float)
    bias = 0.0
    pos = max(float(np.sum(y > 0)), 1.0)
    neg = max(float(len(y) - pos), 1.0)
    sample_weight = np.where(y > 0, neg / pos, 1.0)
    for _ in range(max(1, int(epochs))):
        logits = Xn @ weights + bias
        pred = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
        error = (pred - y) * sample_weight
        grad_w = Xn.T @ error / len(y) + float(l2) * weights
        grad_b = float(np.mean(error))
        weights -= float(lr) * grad_w
        bias -= float(lr) * grad_b
    return LinearRanker(dataset.feature_columns, mean, scale, weights, bias)


def save_model(model: LinearRanker, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(model.to_json(), ensure_ascii=True, indent=2), encoding="utf-8")
    return output


def load_model(path: str | Path) -> LinearRanker:
    return LinearRanker.from_json(json.loads(Path(path).read_text(encoding="utf-8")))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a dependency-light PRISM LTR ranker")
    parser.add_argument("--features", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--l2", type=float, default=1e-3)
    args = parser.parse_args()
    dataset = load_ltr_dataset(args.features)
    model = fit_linear_ranker(dataset, epochs=args.epochs, lr=args.lr, l2=args.l2)
    print(save_model(model, args.output))


if __name__ == "__main__":
    main()
