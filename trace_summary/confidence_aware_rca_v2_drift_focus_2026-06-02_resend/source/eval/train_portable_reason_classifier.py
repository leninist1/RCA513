#!/usr/bin/env python3
"""Train a portable reason classifier from labeled training splits.

The classifier is trained from normalized telemetry features and the label
column only.  It is an offline artifact; `run_portable_d32.py` still receives
no labels during prediction.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
for path in (PROJECT_ROOT, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from refute.src.baseline_distributions import BaselineStore  # noqa: E402
from refute_b_v2_d32.adaptive_baseline import AdaptiveBaselineConfig, build_adaptive_baseline  # noqa: E402
from refute_b_v2_d32.portable_adapters import aiops2021_tabular_adapter, eadro_tabular_adapter  # noqa: E402
from refute_b_v2_d32.portable_ontology import PORTABLE_BUCKETS, PORTABLE_FEATURE_NAMES, extract_portable_reason_features  # noqa: E402
from refute_b_v2_d32.portable_schema import NormalizedIncident, normalize_metric_frame  # noqa: E402
from refute_b_v2_d32.reason_classifier import ReasonClassifier, evaluate_classifier  # noqa: E402
from refute_b_v2_d32.schema import reason_bucket  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["eadro", "aiops2021"], required=True)
    parser.add_argument("--cases-csv", required=True)
    parser.add_argument("--metrics-csv", required=True)
    parser.add_argument("--logs-csv", default=None)
    parser.add_argument("--trace-summary-dir", default=None)
    parser.add_argument("--topology-json", default=None)
    parser.add_argument("--baseline-mode", choices=["historical", "adaptive"], default="adaptive")
    parser.add_argument("--historical-baseline-json", default=None)
    parser.add_argument("--pre-baseline-sec", type=int, default=0)
    parser.add_argument("--train-split-column", default=None)
    parser.add_argument("--train-split-value", default=None)
    parser.add_argument("--eval-split-value", default=None)
    parser.add_argument("--label-column", default="failure_type")
    parser.add_argument("--model", choices=["logistic", "random_forest"], default="logistic")
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary-json", default=None)
    parser.add_argument("--max-cases", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cases = pd.read_csv(args.cases_csv)
    metrics_raw = pd.read_csv(args.metrics_csv)
    logs_raw = pd.read_csv(args.logs_csv) if args.logs_csv else None
    trace_summaries = _load_trace_summaries(args.trace_summary_dir)
    topology = _load_json(args.topology_json, {})
    historical = _load_historical_baseline(args)
    all_metrics = normalize_metric_frame(metrics_raw)

    train_cases = _filter_cases(cases, args.train_split_column, args.train_split_value)
    if args.max_cases is not None:
        train_cases = train_cases.head(max(0, int(args.max_cases))).copy()
    X_train, y_train, train_case_ids = _extract_xy(
        dataset=args.dataset,
        cases=train_cases,
        metrics_raw=metrics_raw,
        logs_raw=logs_raw,
        trace_summaries=trace_summaries,
        topology=topology,
        historical=historical,
        all_metrics=all_metrics,
        args=args,
    )
    if len(X_train) == 0:
        raise RuntimeError("no training examples extracted")
    if len(set(y_train.tolist())) < 2:
        raise RuntimeError(f"need at least 2 classes to train, got {sorted(set(y_train.tolist()))}")

    clf = _train_classifier(X_train, y_train, model_name=args.model)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    clf.save(out_path)

    train_metrics = evaluate_classifier(clf, X_train, y_train, dataset_name=f"{args.dataset}_train", verbose=False)
    summary: dict[str, Any] = {
        "dataset": args.dataset,
        "model": args.model,
        "classifier_path": str(out_path),
        "n_train": int(len(X_train)),
        "train_case_ids": [str(item) for item in train_case_ids],
        "train_label_counts": {str(k): int(v) for k, v in Counter(y_train).items()},
        "train_metrics": train_metrics,
        "feature_names": list(PORTABLE_FEATURE_NAMES),
        "split": {
            "column": args.train_split_column,
            "train_value": args.train_split_value,
            "eval_value": args.eval_split_value,
        },
    }

    if args.eval_split_value is not None:
        eval_cases = _filter_cases(cases, args.train_split_column, args.eval_split_value)
        X_eval, y_eval, eval_case_ids = _extract_xy(
            dataset=args.dataset,
            cases=eval_cases,
            metrics_raw=metrics_raw,
            logs_raw=logs_raw,
            trace_summaries=trace_summaries,
            topology=topology,
            historical=historical,
            all_metrics=all_metrics,
            args=args,
        )
        summary["n_eval"] = int(len(X_eval))
        summary["eval_case_ids"] = [str(item) for item in eval_case_ids]
        summary["eval_label_counts"] = {str(k): int(v) for k, v in Counter(y_eval).items()}
        summary["eval_metrics"] = evaluate_classifier(clf, X_eval, y_eval, dataset_name=f"{args.dataset}_eval", verbose=False)

    summary_path = Path(args.summary_json) if args.summary_json else out_path.with_name(out_path.stem + "_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "classifier_path": str(out_path),
        "summary_json": str(summary_path),
        "n_train": int(len(X_train)),
        "train_label_counts": summary["train_label_counts"],
        "train_accuracy": train_metrics.get("accuracy", 0.0),
        "eval_accuracy": (summary.get("eval_metrics") or {}).get("accuracy"),
    }, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _train_classifier(X: np.ndarray, y: np.ndarray, *, model_name: str) -> ReasonClassifier:
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.astype(np.float64))
    if model_name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier

        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=1,
        )
    else:
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(
            solver="lbfgs",
            max_iter=2000,
            C=0.7,
            class_weight="balanced",
            random_state=42,
        )
    model.fit(X_scaled, y)
    return ReasonClassifier(
        model_=model,
        classes_=[str(item) for item in model.classes_],
        feature_mean_=scaler.mean_.astype(np.float64),
        feature_std_=scaler.scale_.astype(np.float64),
        feature_names_=list(PORTABLE_FEATURE_NAMES),
    )


def _extract_xy(
    *,
    dataset: str,
    cases: pd.DataFrame,
    metrics_raw: pd.DataFrame,
    logs_raw: pd.DataFrame | None,
    trace_summaries: Mapping[str, Mapping[str, Any]],
    topology: Mapping[str, Any],
    historical: BaselineStore | None,
    all_metrics: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    adapter = _build_adapter(
        dataset=dataset,
        cases=cases,
        metrics=metrics_raw,
        logs=logs_raw,
        trace_summaries=trace_summaries,
        topology=topology,
    )
    features: list[np.ndarray] = []
    labels: list[str] = []
    case_ids: list[str] = []
    for incident in adapter.iter_incidents():
        label = _label_for_incident(incident, args.label_column)
        if label not in PORTABLE_BUCKETS:
            continue
        baseline = _baseline_for_incident(
            incident=incident,
            args=args,
            historical=historical,
            all_metrics=all_metrics,
        )
        features.append(extract_portable_reason_features(
            metric_df=incident.metric_df,
            log_df=incident.log_df,
            trace_summary=incident.trace_summary,
            baseline=baseline,
            dataset=dataset,
        ))
        labels.append(label)
        case_ids.append(incident.case_id)
    X = np.stack(features, axis=0) if features else np.empty((0, len(PORTABLE_FEATURE_NAMES)))
    return X, np.array(labels, dtype=str), case_ids


def _baseline_for_incident(
    *,
    incident: NormalizedIncident,
    args: argparse.Namespace,
    historical: BaselineStore | None,
    all_metrics: pd.DataFrame,
):
    if args.baseline_mode == "historical":
        if historical is None:
            raise ValueError("--baseline-mode historical requires --historical-baseline-json")
        return historical
    baseline_frame = incident.metric_df
    if args.pre_baseline_sec > 0 and all_metrics is not None and not all_metrics.empty:
        start = int(incident.window_start_ts)
        pre_rows = all_metrics[
            (all_metrics["timestamp"] >= start - int(args.pre_baseline_sec))
            & (all_metrics["timestamp"] < start)
        ].copy()
        baseline_frame = pd.concat([pre_rows, incident.metric_df], ignore_index=True)
    return build_adaptive_baseline(
        baseline_frame,
        historical=historical,
        window_start_ts=int(incident.window_start_ts),
        config=AdaptiveBaselineConfig(
            enable_pre_fault=args.pre_baseline_sec > 0,
            enable_cross_sectional=True,
            enable_within_window=False,
        ),
    )


def _label_for_incident(incident: NormalizedIncident, label_column: str) -> str:
    value = incident.labels.get(label_column)
    if value is None:
        value = incident.labels.get("故障类型")
    if value is None:
        value = incident.labels.get("fault_type")
    return reason_bucket(str(value or ""))


def _filter_cases(cases: pd.DataFrame, split_column: str | None, split_value: str | None) -> pd.DataFrame:
    if not split_column or split_value is None:
        return cases.copy()
    if split_column not in cases.columns:
        raise ValueError(f"split column not found in cases CSV: {split_column}")
    return cases[cases[split_column].astype(str) == str(split_value)].copy()


def _build_adapter(
    *,
    dataset: str,
    cases: pd.DataFrame,
    metrics: pd.DataFrame,
    logs: pd.DataFrame | None,
    trace_summaries: Mapping[str, Mapping[str, Any]],
    topology: Mapping[str, Any],
):
    if dataset == "eadro":
        return eadro_tabular_adapter(
            cases=cases,
            metrics=metrics,
            logs=logs,
            trace_summaries=trace_summaries,
            topology=topology,
        )
    if dataset == "aiops2021":
        return aiops2021_tabular_adapter(
            cases=cases,
            metrics=metrics,
            logs=logs,
            trace_summaries=trace_summaries,
            topology=topology,
        )
    raise ValueError(f"unsupported dataset: {dataset}")


def _load_historical_baseline(args: argparse.Namespace) -> BaselineStore | None:
    if args.historical_baseline_json:
        return BaselineStore.load_json(args.historical_baseline_json)
    if args.baseline_mode == "historical":
        raise ValueError("--baseline-mode historical requires --historical-baseline-json")
    return None


def _load_trace_summaries(trace_summary_dir: str | None) -> dict[str, Mapping[str, Any]]:
    if not trace_summary_dir:
        return {}
    base = Path(trace_summary_dir)
    if not base.exists():
        raise FileNotFoundError(f"trace summary dir not found: {base}")
    out: dict[str, Mapping[str, Any]] = {}
    for path in sorted(base.glob("*.json")):
        with path.open("r", encoding="utf-8") as f:
            out[path.stem] = json.load(f)
    return out


def _load_json(path: str | None, default: Any) -> Any:
    if not path:
        return default
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    raise SystemExit(main())
