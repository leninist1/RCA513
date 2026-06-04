"""LODO selective prediction evaluation for confidence-aware RCA."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


TARGETS = {
    "strict": {"label": "strict_label", "required": None},
    "partial": {"label": "partial_label", "required": None},
    "time": {"label": "time_hit_label", "required": "time_required"},
    "time_any": {"label": "time_hit_any_label", "required": "time_required"},
    "component": {"label": "component_hit_label", "required": "component_required"},
    "component_any": {"label": "component_hit_any_label", "required": "component_required"},
    "reason": {"label": "reason_hit_label", "required": "reason_required"},
    "reason_any": {"label": "reason_hit_any_label", "required": "reason_required"},
    "component_reason_pair": {"label": "component_reason_pair_hit_label", "required": "component_reason_pair_required"},
    "component_reason_pair_any": {"label": "component_reason_pair_hit_any_label", "required": "component_reason_pair_required"},
    "actionable": {"label": "actionable_hit_label", "required": "actionable_required"},
    "actionable_any": {"label": "actionable_hit_any_label", "required": "actionable_required"},
}

CORE_FEATURE_COLUMNS = {
    "top1_rebuttal_score",
    "top1_support_strength",
    "top1_refute_strength",
    "top1_blind_count",
    "top1_prior",
    "top1_confidence_level",
    "top1_component_evidence_prior",
    "top1_cluster_similarity",
    "top1_reason_prior_weight",
    "top2_rebuttal_score",
    "rebuttal_margin_top2_minus_top1",
    "support_margin_top1_minus_top2",
    "prior_margin_top1_minus_top2",
    "decision_count",
    "high_suspicion_count",
    "low_suspicion_count",
    "candidate_space_count",
    "data_blind_spot_count",
    "support_refute_ratio",
    "support_per_blind",
    "card_count",
    "support_card_count",
    "refute_card_count",
    "blind_card_count",
    "support_card_strength",
    "refute_card_strength",
    "signature_service_count",
    "signature_dominant_count",
    "signature_blind_spot_count",
    "metric_service_count",
    "log_service_count",
    "trace_service_count",
    "metric_strength_max",
    "log_strength_max",
    "trace_strength_max",
    "top1_component_modality_agreement",
    "top1_component_bucket_strength_sum",
    "top1_component_bucket_strength_max",
    "topk_component_unique",
    "topk_reason_unique",
    "topk_reason_bucket_unique",
    "topk_component_entropy",
    "topk_reason_entropy",
    "topk_reason_bucket_entropy",
    "top1_top2_same_component",
    "top1_top2_same_reason",
    "top1_top2_same_reason_bucket",
    "time_anchor_score",
    "time_anchor_vote_count",
    "time_anchor_rejected_count",
    "time_anchor_rejected_best_score",
    "time_anchor_score_margin",
    "time_anchor_metric_votes",
    "time_anchor_log_votes",
    "time_anchor_trace_votes",
    "time_anchor_vote_source_diversity",
    "time_anchor_vote_kind_diversity",
    "metric_present",
    "log_present",
    "trace_present",
}

CORE_FEATURE_PREFIXES = (
    "matched_cluster_similarity_",
    "matched_rule_confidence_",
    "matched_rule_validation_confidence_",
    "matched_rule_validation_dates_",
    "matched_rule_validation_support_",
)

SHARED_HEAD_COLUMNS = {
    "top1_rebuttal_score",
    "top1_support_strength",
    "top1_refute_strength",
    "top1_blind_count",
    "top1_prior",
    "top1_confidence_level",
    "top2_rebuttal_score",
    "rebuttal_margin_top2_minus_top1",
    "support_margin_top1_minus_top2",
    "prior_margin_top1_minus_top2",
    "decision_count",
    "high_suspicion_count",
    "low_suspicion_count",
    "candidate_space_count",
    "data_blind_spot_count",
    "support_refute_ratio",
    "support_per_blind",
    "card_count",
    "support_card_count",
    "refute_card_count",
    "blind_card_count",
    "support_card_strength",
    "refute_card_strength",
    "signature_service_count",
    "signature_dominant_count",
    "signature_blind_spot_count",
    "metric_present",
    "log_present",
    "trace_present",
}

TIME_HEAD_COLUMNS = {
    "time_anchor_score",
    "time_anchor_vote_count",
    "time_anchor_rejected_count",
    "time_anchor_rejected_best_score",
    "time_anchor_score_margin",
    "time_anchor_metric_votes",
    "time_anchor_log_votes",
    "time_anchor_trace_votes",
    "time_anchor_vote_source_diversity",
    "time_anchor_vote_kind_diversity",
    "metric_service_count",
    "log_service_count",
    "trace_service_count",
    "metric_strength_max",
    "log_strength_max",
    "trace_strength_max",
}

COMPONENT_HEAD_COLUMNS = {
    "top1_component_evidence_prior",
    "top1_cluster_similarity",
    "top1_component_modality_agreement",
    "top1_component_bucket_strength_sum",
    "top1_component_bucket_strength_max",
    "topk_component_unique",
    "topk_component_entropy",
    "top1_top2_same_component",
    "metric_service_count",
    "log_service_count",
    "trace_service_count",
    "metric_strength_max",
    "log_strength_max",
    "trace_strength_max",
}

REASON_HEAD_COLUMNS = {
    "top1_reason_prior_weight",
    "topk_reason_unique",
    "topk_reason_bucket_unique",
    "topk_reason_entropy",
    "topk_reason_bucket_entropy",
    "top1_top2_same_reason",
    "top1_top2_same_reason_bucket",
    "metric_service_count",
    "log_service_count",
    "trace_service_count",
    "metric_strength_max",
    "log_strength_max",
    "trace_strength_max",
}

TARGET_HEAD_COLUMN_GROUPS = {
    "strict": SHARED_HEAD_COLUMNS | TIME_HEAD_COLUMNS | COMPONENT_HEAD_COLUMNS | REASON_HEAD_COLUMNS,
    "partial": SHARED_HEAD_COLUMNS | TIME_HEAD_COLUMNS | COMPONENT_HEAD_COLUMNS | REASON_HEAD_COLUMNS,
    "time": SHARED_HEAD_COLUMNS | TIME_HEAD_COLUMNS,
    "time_any": SHARED_HEAD_COLUMNS | TIME_HEAD_COLUMNS,
    "component": SHARED_HEAD_COLUMNS | COMPONENT_HEAD_COLUMNS,
    "component_any": SHARED_HEAD_COLUMNS | COMPONENT_HEAD_COLUMNS,
    "reason": SHARED_HEAD_COLUMNS | REASON_HEAD_COLUMNS,
    "reason_any": SHARED_HEAD_COLUMNS | REASON_HEAD_COLUMNS,
    "component_reason_pair": SHARED_HEAD_COLUMNS | COMPONENT_HEAD_COLUMNS | REASON_HEAD_COLUMNS,
    "component_reason_pair_any": SHARED_HEAD_COLUMNS | COMPONENT_HEAD_COLUMNS | REASON_HEAD_COLUMNS,
    "actionable": SHARED_HEAD_COLUMNS | TIME_HEAD_COLUMNS | COMPONENT_HEAD_COLUMNS | REASON_HEAD_COLUMNS,
    "actionable_any": SHARED_HEAD_COLUMNS | TIME_HEAD_COLUMNS | COMPONENT_HEAD_COLUMNS | REASON_HEAD_COLUMNS,
}

TARGET_HEAD_PREFIXES = {
    "strict": CORE_FEATURE_PREFIXES,
    "partial": CORE_FEATURE_PREFIXES,
    "time": (),
    "time_any": (),
    "component": ("matched_cluster_similarity_",),
    "component_any": ("matched_cluster_similarity_",),
    "reason": (
        "matched_rule_confidence_",
        "matched_rule_validation_confidence_",
        "matched_rule_validation_dates_",
        "matched_rule_validation_support_",
    ),
    "reason_any": (
        "matched_rule_confidence_",
        "matched_rule_validation_confidence_",
        "matched_rule_validation_dates_",
        "matched_rule_validation_support_",
    ),
    "component_reason_pair": CORE_FEATURE_PREFIXES,
    "component_reason_pair_any": CORE_FEATURE_PREFIXES,
    "actionable": CORE_FEATURE_PREFIXES,
    "actionable_any": CORE_FEATURE_PREFIXES,
}

METADATA_COLUMNS = {
    "row_id",
    "heldout_date",
    "task_index",
    "strict_label",
    "partial_label",
    "time_hit_label",
    "component_hit_label",
    "reason_hit_label",
    "time_required",
    "component_required",
    "reason_required",
    "component_reason_pair_required",
    "actionable_required",
    "fractional_score",
    "required_total",
    "hit_total",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", default="confidence_aware_rca/experiments/d32_timevote_confidence_features.csv")
    parser.add_argument("--out", default="confidence_aware_rca/experiments/d32_timevote_selective_lodo.json")
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


def load_csv(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def to_float(value: Any) -> float:
    try:
        if value == "" or value is None:
            return np.nan
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def numeric_columns(rows: list[dict[str, Any]], feature_set: str = "core", target: str | None = None) -> list[str]:
    out = []
    for key in sorted({key for row in rows for key in row}):
        if is_metadata_column(key):
            continue
        if feature_set == "core" and not is_core_feature(key):
            continue
        if feature_set == "target_specific" and not is_target_head_feature(key, target):
            continue
        values = [to_float(row.get(key)) for row in rows]
        if any(not np.isnan(value) for value in values):
            out.append(key)
    return out


def is_metadata_column(key: str) -> bool:
    return (
        key in METADATA_COLUMNS
        or key.endswith("_label")
        or key.endswith("_required")
        or key.endswith("_required_total")
        or key.endswith("_hit_count")
    )


def is_core_feature(key: str) -> bool:
    return key in CORE_FEATURE_COLUMNS or any(key.startswith(prefix) for prefix in CORE_FEATURE_PREFIXES)


def is_target_head_feature(key: str, target: str | None) -> bool:
    if target is None:
        return is_core_feature(key)
    columns = TARGET_HEAD_COLUMN_GROUPS.get(target, CORE_FEATURE_COLUMNS)
    prefixes = TARGET_HEAD_PREFIXES.get(target, CORE_FEATURE_PREFIXES)
    return key in columns or any(key.startswith(prefix) for prefix in prefixes)


def matrix(rows: list[dict[str, Any]], columns: list[str]) -> np.ndarray:
    return np.asarray([[to_float(row.get(col)) for col in columns] for row in rows], dtype=float)


def numeric_frame(rows: list[dict[str, Any]], columns: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    return pd.DataFrame({col: pd.to_numeric(frame[col] if col in frame else np.nan, errors="coerce") for col in columns}, index=frame.index)


def observed_columns(rows: list[dict[str, Any]], columns: list[str]) -> list[str]:
    frame = numeric_frame(rows, columns)
    return [col for col in columns if frame[col].notna().any()]


DRIFT_FEATURE_COLUMNS = [
    "drift_l1_to_train_median",
    "drift_l2_to_train_median",
    "drift_max_abs_z",
    "drift_missing_rate",
]


def append_drift_features(
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    columns: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    columns = observed_columns(train_rows, columns)
    if not columns:
        return train_rows, test_rows, columns
    train_df = numeric_frame(train_rows, columns)
    train_median = train_df.median(axis=0, skipna=True)
    train_iqr = (train_df.quantile(0.75) - train_df.quantile(0.25)).replace(0, np.nan).fillna(1.0)

    def augment(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not rows:
            return rows
        frame = numeric_frame(rows, columns)
        missing_rate = frame.isna().mean(axis=1).fillna(1.0)
        filled = frame.fillna(train_median)
        z = (filled - train_median) / train_iqr
        abs_z = z.abs()
        out = []
        for idx, row in enumerate(rows):
            new_row = dict(row)
            vals = abs_z.iloc[idx]
            new_row["drift_l1_to_train_median"] = float(vals.mean()) if len(vals) else 0.0
            new_row["drift_l2_to_train_median"] = float(np.sqrt((vals ** 2).mean())) if len(vals) else 0.0
            new_row["drift_max_abs_z"] = float(vals.max()) if len(vals) else 0.0
            new_row["drift_missing_rate"] = float(missing_rate.iloc[idx])
            out.append(new_row)
        return out

    return augment(train_rows), augment(test_rows), columns + DRIFT_FEATURE_COLUMNS


def labels(rows: list[dict[str, Any]], target: str) -> np.ndarray:
    return np.asarray([int(float(row[TARGETS[target]["label"]])) for row in rows], dtype=int)


def target_rows(rows: list[dict[str, Any]], target: str) -> list[dict[str, Any]]:
    required_col = TARGETS[target]["required"]
    if required_col is None:
        return rows
    return [row for row in rows if int(float(row.get(required_col, 0) or 0)) == 1]


def build_model(columns: list[str]) -> Pipeline:
    pre = ColumnTransformer([
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), columns),
    ])
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=0.7, solver="liblinear")
    return Pipeline([("pre", pre), ("clf", clf)])


def model_probabilities(
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    y_train: np.ndarray,
    columns: list[str],
    model_kind: str,
    drift_features: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    if len(set(y_train.tolist())) < 2:
        value = float(y_train.mean()) if len(y_train) else 0.0
        return np.full(len(train_rows), value), np.full(len(test_rows), value)
    if model_kind == "score":
        return score_baseline(train_rows), score_baseline(test_rows)

    columns = observed_columns(train_rows, columns)
    if not columns:
        value = float(y_train.mean()) if len(y_train) else 0.0
        return np.full(len(train_rows), value), np.full(len(test_rows), value)
    if drift_features:
        train_rows, test_rows, columns = append_drift_features(train_rows, test_rows, columns)
    model = build_model(columns)
    train_df = numeric_frame(train_rows, columns)
    test_df = numeric_frame(test_rows, columns)
    model.fit(train_df, y_train)
    train_prob = model.predict_proba(train_df)[:, 1]
    test_prob = model.predict_proba(test_df)[:, 1]
    if model_kind == "isotonic":
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(train_prob, y_train)
        train_prob = iso.predict(train_prob)
        test_prob = iso.predict(test_prob)
    return np.asarray(train_prob, dtype=float), np.asarray(test_prob, dtype=float)


def score_baseline(rows: list[dict[str, Any]]) -> np.ndarray:
    values = []
    for row in rows:
        score = (
            -to_float(row.get("top1_rebuttal_score"))
            + 0.35 * to_float(row.get("rebuttal_margin_top2_minus_top1"))
            + 0.08 * to_float(row.get("top1_support_strength"))
            + 0.35 * to_float(row.get("time_anchor_score_margin"))
            + 0.25 * to_float(row.get("matched_cluster_similarity_max"))
            + 0.15 * to_float(row.get("matched_rule_validation_confidence_max"))
            - 0.7 * to_float(row.get("top1_blind_count"))
            - 0.4 * to_float(row.get("data_blind_spot_count"))
        )
        values.append(score)
    arr = np.asarray(values, dtype=float)
    arr = np.nan_to_num(arr, nan=np.nanmedian(arr) if not np.isnan(arr).all() else 0.0)
    if float(arr.max()) == float(arr.min()):
        return np.full(len(arr), 0.5)
    return (arr - arr.min()) / (arr.max() - arr.min())


def choose_threshold(train_prob: np.ndarray, train_y: np.ndarray, target_accuracy: float) -> float:
    best_threshold = float("inf")
    best_coverage = -1
    for threshold in sorted(set(float(x) for x in train_prob), reverse=True):
        accepted = train_prob >= threshold
        coverage = int(accepted.sum())
        if coverage == 0:
            continue
        acc = float(train_y[accepted].mean())
        if acc >= target_accuracy and coverage > best_coverage:
            best_coverage = coverage
            best_threshold = threshold
    if best_coverage < 0:
        return float(train_prob.max()) + 1e-9
    return best_threshold


def choose_threshold_conformal(train_prob: np.ndarray, train_y: np.ndarray, target_accuracy: float, delta: float = 0.1) -> float:
    alpha = 1.0 - target_accuracy
    best_threshold = float("inf")
    best_coverage = -1
    for threshold in sorted(set(float(x) for x in train_prob), reverse=True):
        accepted = train_prob >= threshold
        n = int(accepted.sum())
        if n == 0:
            continue
        errors = int((1 - train_y[accepted]).sum())
        risk_hat = errors / n
        radius = float(np.sqrt(np.log(1.0 / max(delta, 1e-9)) / (2.0 * n)))
        risk_ucb = risk_hat + radius
        if risk_ucb <= alpha and n > best_coverage:
            best_coverage = n
            best_threshold = threshold
    if best_coverage < 0:
        return float(train_prob.max()) + 1e-9
    return best_threshold


def threshold_for_mode(train_prob: np.ndarray, train_y: np.ndarray, target_accuracy: float, risk_mode: str, delta: float) -> float:
    if risk_mode == "conformal":
        return choose_threshold_conformal(train_prob, train_y, target_accuracy, delta=delta)
    return choose_threshold(train_prob, train_y, target_accuracy)


def parse_feature_set_candidates(value: str) -> list[str]:
    allowed = {"core", "target_specific", "all"}
    candidates = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if item not in allowed:
            raise ValueError(f"unsupported auto feature-set candidate: {item}")
        if item not in candidates:
            candidates.append(item)
    return candidates or ["core", "target_specific"]


def selective_stats(prob: np.ndarray, y: np.ndarray, threshold: float) -> dict[str, Any]:
    accepted = prob >= threshold
    n_accept = int(accepted.sum())
    n = int(len(y))
    correct = int(y[accepted].sum()) if n_accept else 0
    return {
        "n": n,
        "accepted": n_accept,
        "correct": correct,
        "coverage": n_accept / n if n else 0.0,
        "selective_accuracy": correct / n_accept if n_accept else None,
        "threshold": threshold,
    }


def wilson_lower_bound(successes: int, total: int, z: float = 1.64) -> float:
    if total <= 0:
        return 0.0
    phat = successes / total
    denom = 1.0 + z * z / total
    center = phat + z * z / (2.0 * total)
    radius = z * np.sqrt((phat * (1.0 - phat) + z * z / (4.0 * total)) / total)
    return float(max(0.0, (center - radius) / denom))


def risk_coverage(prob: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    if len(y) == 0:
        return {"aurc": None, "curve": []}
    order = np.argsort(-prob)
    ys = y[order]
    rows = []
    risks = []
    correct = 0
    for idx, label in enumerate(ys, start=1):
        correct += int(label)
        coverage = idx / len(ys)
        accuracy = correct / idx
        risk = 1.0 - accuracy
        risks.append(risk)
        rows.append({"coverage": coverage, "accuracy": accuracy, "risk": risk, "accepted": idx})
    return {"aurc": float(np.mean(risks)), "curve": rows}


def inner_candidate_report(
    rows: list[dict[str, Any]],
    target: str,
    feature_set: str,
    model_kind: str,
    target_accuracy: float,
    risk_mode: str,
    delta: float,
    z: float = 1.64,
    drift_features: bool = False,
) -> dict[str, Any]:
    dates = sorted({row["heldout_date"] for row in rows})
    y_all = labels(rows, target)
    if len(dates) < 2 or len(rows) < 3:
        return {
            "feature_set": feature_set,
            "n": len(rows),
            "aurc": None,
            "accepted": 0,
            "correct": 0,
            "coverage": 0.0,
            "selective_accuracy": None,
            "wilson_lower": 0.0,
            "utility": -1.0,
            "feature_count_mean": 0.0,
        }

    oof_prob = np.zeros(len(rows), dtype=float)
    accepted = 0
    correct = 0
    total = 0
    feature_counts = []
    for date in dates:
        train_idx = [idx for idx, row in enumerate(rows) if row["heldout_date"] != date]
        valid_idx = [idx for idx, row in enumerate(rows) if row["heldout_date"] == date]
        train_rows = [rows[idx] for idx in train_idx]
        valid_rows = [rows[idx] for idx in valid_idx]
        y_train = labels(train_rows, target)
        y_valid = labels(valid_rows, target)
        columns = numeric_columns(train_rows, feature_set=feature_set, target=target)
        feature_counts.append(len(columns))
        train_prob, valid_prob = model_probabilities(train_rows, valid_rows, y_train, columns, model_kind, drift_features)
        for idx, prob in zip(valid_idx, valid_prob):
            oof_prob[idx] = float(prob)
        threshold = threshold_for_mode(train_prob, y_train, target_accuracy, risk_mode, delta)
        stat = selective_stats(valid_prob, y_valid, threshold)
        accepted += stat["accepted"]
        correct += stat["correct"]
        total += stat["n"]

    coverage = accepted / total if total else 0.0
    selective_accuracy = correct / accepted if accepted else None
    wilson = wilson_lower_bound(correct, accepted, z=z)
    aurc = risk_coverage(oof_prob, y_all)["aurc"]
    utility = (selective_accuracy or 0.0) * float(np.sqrt(coverage)) if coverage > 0 else 0.0
    return {
        "feature_set": feature_set,
        "n": len(rows),
        "aurc": aurc,
        "accepted": accepted,
        "correct": correct,
        "coverage": coverage,
        "selective_accuracy": selective_accuracy,
        "wilson_lower": wilson,
        "utility": utility,
        "feature_count_mean": float(np.mean(feature_counts)) if feature_counts else 0.0,
    }


def select_candidate_report(
    reports: list[dict[str, Any]],
    selection_objective: str = "aurc",
    min_coverage: float = 0.2,
) -> dict[str, Any]:
    if not reports:
        raise ValueError("no candidate reports to select from")
    if selection_objective == "threshold_transfer":
        return max(
            reports,
            key=lambda row: (
                float(row.get("coverage") or 0.0) >= min_coverage,
                float(row.get("wilson_lower") or 0.0),
                float(row.get("selective_accuracy") or 0.0),
                float(row.get("coverage") or 0.0),
                -float("inf") if row.get("aurc") is None else -float(row["aurc"]),
                str(row["feature_set"]),
            ),
        )
    return min(
        reports,
        key=lambda row: (
            float("inf") if row.get("aurc") is None else float(row["aurc"]),
            -float(row.get("utility") or 0.0),
            str(row["feature_set"]),
        ),
    )


def select_feature_set(
    train_rows: list[dict[str, Any]],
    target: str,
    candidates: list[str],
    model_kind: str,
    target_accuracy: float,
    risk_mode: str,
    delta: float,
    selection_objective: str = "aurc",
    min_coverage: float = 0.2,
    z: float = 1.64,
    drift_features: bool = False,
) -> tuple[str, dict[str, Any]]:
    reports = [
        inner_candidate_report(train_rows, target, candidate, model_kind, target_accuracy, risk_mode, delta, z=z, drift_features=drift_features)
        for candidate in candidates
    ]
    best = select_candidate_report(reports, selection_objective=selection_objective, min_coverage=min_coverage)
    return str(best["feature_set"]), {
        "selection_metric": (
            "inner_lodo_wilson_lower_then_selective_accuracy"
            if selection_objective == "threshold_transfer"
            else "inner_lodo_aurc_then_selective_utility"
        ),
        "selection_target_accuracy": target_accuracy,
        "selection_objective": selection_objective,
        "selection_min_coverage": min_coverage,
        "selection_z": z,
        "drift_features": drift_features,
        "candidates": reports,
        "selected_feature_set": best["feature_set"],
    }


def ece(prob: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    if len(y) == 0:
        return 0.0
    total = len(y)
    value = 0.0
    for idx in range(bins):
        lo = idx / bins
        hi = (idx + 1) / bins
        if idx == bins - 1:
            mask = (prob >= lo) & (prob <= hi)
        else:
            mask = (prob >= lo) & (prob < hi)
        if not mask.any():
            continue
        conf = float(prob[mask].mean())
        acc = float(y[mask].mean())
        value += float(mask.sum()) / total * abs(acc - conf)
    return value


def evaluate_target(rows: list[dict[str, Any]], target: str, target_accuracies: list[float], model_kind: str,
                    risk_mode: str = "empirical", delta: float = 0.1, feature_set: str = "core",
                    auto_candidates: list[str] | None = None, selection_target_accuracy: float = 0.9,
                    selection_objective: str = "aurc", selection_min_coverage: float = 0.2,
                    selection_z: float = 1.64, drift_features: bool = False) -> dict[str, Any]:
    rows = target_rows(rows, target)
    if not rows:
        return {"target": target, "n": 0}
    fixed_columns = numeric_columns(rows, feature_set=feature_set, target=target) if feature_set != "auto" else []
    dates = sorted({row["heldout_date"] for row in rows})
    oof_prob = np.zeros(len(rows), dtype=float)
    oof_y = labels(rows, target)
    folds = []
    selected_counts: dict[str, int] = {}
    for date in dates:
        train_idx = [idx for idx, row in enumerate(rows) if row["heldout_date"] != date]
        test_idx = [idx for idx, row in enumerate(rows) if row["heldout_date"] == date]
        train_rows = [rows[idx] for idx in train_idx]
        test_rows = [rows[idx] for idx in test_idx]
        y_train = labels(train_rows, target)
        y_test = labels(test_rows, target)
        selection = None
        selected_feature_set = feature_set
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
        else:
            columns = fixed_columns
        selected_counts[selected_feature_set] = selected_counts.get(selected_feature_set, 0) + 1
        train_prob, test_prob = model_probabilities(train_rows, test_rows, y_train, columns, model_kind, drift_features)
        for idx, prob in zip(test_idx, test_prob):
            oof_prob[idx] = float(prob)
        fold_row = {
            "heldout_date": date,
            "n_test": len(test_idx),
            "base_rate_test": float(y_test.mean()) if len(y_test) else 0.0,
            "feature_set": selected_feature_set,
            "feature_count": len(columns),
        }
        if selection:
            fold_row["feature_selection"] = selection
        for target_acc in target_accuracies:
            threshold = threshold_for_mode(train_prob, y_train, target_acc, risk_mode, delta)
            fold_row[f"target_{target_acc:.2f}"] = selective_stats(test_prob, y_test, threshold)
        folds.append(fold_row)
    summary: dict[str, Any] = {
        "target": target,
        "n": len(rows),
        "base_rate": float(oof_y.mean()),
        "positives": int(oof_y.sum()),
        "model": model_kind,
        "risk_mode": risk_mode,
        "feature_set": feature_set,
        "selected_feature_sets": selected_counts,
        "auto_candidates": auto_candidates or [],
        "selection_target_accuracy": selection_target_accuracy,
        "selection_objective": selection_objective,
        "selection_min_coverage": selection_min_coverage,
        "selection_z": selection_z,
        "drift_features": drift_features,
        "delta": delta,
        "feature_count": len(fixed_columns) if feature_set != "auto" else None,
        "brier": float(brier_score_loss(oof_y, np.clip(oof_prob, 0.0, 1.0))) if len(set(oof_y.tolist())) > 1 else None,
        "ece": ece(np.clip(oof_prob, 0.0, 1.0), oof_y),
        "risk_coverage": risk_coverage(oof_prob, oof_y),
        "folds": folds,
    }
    for target_acc in target_accuracies:
        accepted = 0
        correct = 0
        total = 0
        for fold in folds:
            stat = fold[f"target_{target_acc:.2f}"]
            accepted += stat["accepted"]
            correct += stat["correct"]
            total += stat["n"]
        summary[f"coverage_at_train_acc_{target_acc:.2f}"] = {
            "accepted": accepted,
            "correct": correct,
            "n": total,
            "coverage": accepted / total if total else 0.0,
            "selective_accuracy": correct / accepted if accepted else None,
        }
    for target_acc in target_accuracies:
        # Oracle threshold on OOF probabilities, useful as an upper diagnostic.
        threshold = choose_threshold(oof_prob, oof_y, target_acc)
        summary[f"oracle_oof_at_acc_{target_acc:.2f}"] = selective_stats(oof_prob, oof_y, threshold)
    return summary


def main() -> int:
    args = parse_args()
    rows = load_csv(args.features)
    target_accuracies = [float(item) for item in args.target_accuracies.split(",") if item.strip()]
    auto_candidates = parse_feature_set_candidates(args.auto_candidates)
    report = {
        "features": args.features,
        "n_rows": len(rows),
        "model": args.model,
        "risk_mode": args.risk_mode,
        "feature_set": args.feature_set,
        "auto_candidates": auto_candidates,
        "selection_target_accuracy": args.selection_target_accuracy,
        "selection_objective": args.selection_objective,
        "selection_min_coverage": args.selection_min_coverage,
        "selection_z": args.selection_z,
        "drift_features": args.drift_features,
        "target_accuracies": target_accuracies,
        "targets": {
            target: evaluate_target(
                rows,
                target,
                target_accuracies,
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
            "n": row.get("n"),
            "base_rate": row.get("base_rate"),
            "coverage@0.80": row.get("coverage_at_train_acc_0.80"),
            "coverage@0.90": row.get("coverage_at_train_acc_0.90"),
            "oracle@0.80": row.get("oracle_oof_at_acc_0.80"),
            "oracle@0.90": row.get("oracle_oof_at_acc_0.90"),
            "aurc": (row.get("risk_coverage") or {}).get("aurc"),
            "ece": row.get("ece"),
            "brier": row.get("brier"),
        }
        for target, row in report["targets"].items()
    }
    print(json.dumps({"out": args.out, "summary": compact}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
