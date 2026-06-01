"""Simple score calibration helpers for LTR artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def fit_temperature(scores: np.ndarray, labels: np.ndarray) -> float:
    best_temp = 1.0
    best_loss = float("inf")
    for temp in np.linspace(0.1, 5.0, 100):
        probs = 1.0 / (1.0 + np.exp(-np.clip(scores / temp, -30, 30)))
        loss = -np.mean(labels * np.log(probs + 1e-9) + (1.0 - labels) * np.log(1.0 - probs + 1e-9))
        if loss < best_loss:
            best_loss = float(loss)
            best_temp = float(temp)
    return best_temp


def apply_temperature(scores: np.ndarray, temperature: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(scores / max(float(temperature), 1e-9), -30, 30)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate PRISM LTR OOF scores")
    parser.add_argument("--scores", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--score-col", default="ltr_score")
    parser.add_argument("--label-col", default="label")
    args = parser.parse_args()
    frame = pd.read_csv(args.scores)
    temp = fit_temperature(frame[args.score_col].astype(float).to_numpy(), frame[args.label_col].astype(float).to_numpy())
    frame["ltr_calibrated"] = apply_temperature(frame[args.score_col].astype(float).to_numpy(), temp)
    frame.to_csv(args.output, index=False)
    Path(f"{args.output}.calibration.json").write_text(
        json.dumps({"temperature": temp}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"output": args.output, "temperature": temp}, indent=2))


if __name__ == "__main__":
    main()
