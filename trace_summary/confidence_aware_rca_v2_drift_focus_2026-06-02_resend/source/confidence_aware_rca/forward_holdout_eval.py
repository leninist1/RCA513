"""Forward last-N-date selective RCA evaluation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .selective_eval import (
        TARGETS,
        labels,
        load_csv,
        model_probabilities,
        numeric_columns,
        parse_feature_set_candidates,
        risk_coverage,
        select_feature_set,
        selective_stats,
        target_rows,
        threshold_for_mode,
    )
except ImportError:
    from selective_eval import (
        TARGETS,
        labels,
        load_csv,
        model_probabilities,
        numeric_columns,
        parse_feature_set_candidates,
        risk_coverage,
        select_feature_set,
        selective_stats,
        target_rows,
        threshold_for_mode,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", default="confidence_aware_rca/experiments/d32_timevote_confidence_features.csv")
    parser.add_argument("--out", default="confidence_aware_rca/experiments/d32_timevote_selective_forward_last3.json")
    parser.add_argument("--last-n-dates", type=int, default=3)
    parser.add_argument("--target-accuracies", default="0.8,0.9")
    parser.add_argument("--model", choices=["logistic", "isotonic", "score"], default="logistic")
    parser.add_argument("--risk-mode", choices=["empirical", "conformal"], default="empirical")
    parser.add_argument("--feature-set", choices=["core", "all", "target_specific", "auto"], default="core")
    parser.add_argument("--auto-candidates", default="core,target_specific")
    parser.add_argument("--selection-target-accuracy", type=float, default=0.9)
    parser.add_argument("--selection-objective", choices=["aurc", "threshold_transfer"], default="aurc")
    parser.add_argument("--selection-min-coverage", type=float, default=0.2)
    parser.add_argument("--selection-z", type=float, default=1.64)
    parser.add_argument("--drift-features", action="store_true")
    parser.add_argument("--delta", type=float, default=0.1)
    return parser.parse_args()


def evaluate_target(rows: list[dict], target: str, heldout_dates: set[str], target_accs: list[float], model_kind: str,
                    risk_mode: str = "empirical", delta: float = 0.1, feature_set: str = "core",
                    auto_candidates: list[str] | None = None, selection_target_accuracy: float = 0.9,
                    selection_objective: str = "aurc", selection_min_coverage: float = 0.2,
                    selection_z: float = 1.64, drift_features: bool = False) -> dict:
    rows = target_rows(rows, target)
    train_rows = [row for row in rows if row["heldout_date"] not in heldout_dates]
    test_rows = [row for row in rows if row["heldout_date"] in heldout_dates]
    y_train = labels(train_rows, target)
    y_test = labels(test_rows, target)
    if not test_rows:
        return {"target": target, "n_test": 0}
    selected_feature_set = feature_set
    selection = None
    if feature_set == "auto":
        selected_feature_set, selection = select_feature_set(
            train_rows,
            target,
            auto_candidates or ["core", "target_specific"],
            model_kind,
            selection_target_accuracy,
            risk_mode,
            delta,
            selection_objective,
            selection_min_coverage,
            selection_z,
            drift_features,
        )
    columns = numeric_columns(train_rows, feature_set=selected_feature_set, target=target)
    train_prob, test_prob = model_probabilities(train_rows, test_rows, y_train, columns, model_kind, drift_features)
    out = {
        "target": target,
        "model": model_kind,
        "risk_mode": risk_mode,
        "feature_set": feature_set,
        "selected_feature_set": selected_feature_set,
        "auto_candidates": auto_candidates or [],
        "selection_target_accuracy": selection_target_accuracy,
        "selection_objective": selection_objective,
        "selection_min_coverage": selection_min_coverage,
        "selection_z": selection_z,
        "drift_features": drift_features,
        "feature_selection": selection,
        "feature_count": len(columns),
        "delta": delta,
        "n_train": len(train_rows),
        "n_test": len(test_rows),
        "train_base_rate": float(y_train.mean()) if len(y_train) else 0.0,
        "test_base_rate": float(y_test.mean()) if len(y_test) else 0.0,
        "risk_coverage_test": risk_coverage(test_prob, y_test),
    }
    for target_acc in target_accs:
        threshold = threshold_for_mode(train_prob, y_train, target_acc, risk_mode, delta)
        out[f"coverage_at_train_acc_{target_acc:.2f}"] = selective_stats(test_prob, y_test, threshold)
    return out


def main() -> int:
    args = parse_args()
    rows = load_csv(args.features)
    dates = sorted({row["heldout_date"] for row in rows})
    heldout_dates = set(dates[-max(1, int(args.last_n_dates)):])
    target_accs = [float(item) for item in args.target_accuracies.split(",") if item.strip()]
    auto_candidates = parse_feature_set_candidates(args.auto_candidates)
    report = {
        "features": args.features,
        "model": args.model,
        "risk_mode": args.risk_mode,
        "feature_set": args.feature_set,
        "auto_candidates": auto_candidates,
        "selection_target_accuracy": args.selection_target_accuracy,
        "selection_objective": args.selection_objective,
        "selection_min_coverage": args.selection_min_coverage,
        "selection_z": args.selection_z,
        "drift_features": args.drift_features,
        "heldout_dates": sorted(heldout_dates),
        "target_accuracies": target_accs,
        "targets": {
            target: evaluate_target(
                rows,
                target,
                heldout_dates,
                target_accs,
                args.model,
                args.risk_mode,
                args.delta,
                args.feature_set,
                auto_candidates,
                args.selection_target_accuracy,
                args.selection_objective,
                args.selection_min_coverage,
                args.selection_z,
                args.drift_features,
            )
            for target in TARGETS
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    compact = {
        target: {
            "n_test": row.get("n_test"),
            "test_base_rate": row.get("test_base_rate"),
            "selected_feature_set": row.get("selected_feature_set"),
            "coverage@0.80": row.get("coverage_at_train_acc_0.80"),
            "coverage@0.90": row.get("coverage_at_train_acc_0.90"),
            "aurc_test": (row.get("risk_coverage_test") or {}).get("aurc"),
        }
        for target, row in report["targets"].items()
    }
    print(json.dumps({"out": args.out, "heldout_dates": sorted(heldout_dates), "summary": compact}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
