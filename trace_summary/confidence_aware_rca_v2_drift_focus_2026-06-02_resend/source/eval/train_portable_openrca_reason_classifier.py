#!/usr/bin/env python3
"""Train a portable-dataset reason classifier with OpenRCA D32 features.

This script adapts Eadro/AIOps2021 normalized inputs into the same feature
extractor used by the OpenRCA D32 selector:
`refute_b_v2_d32.reason_classifier.extract_features`.
Labels are used only offline for training/evaluation splits.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
for path in (PROJECT_ROOT, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from refute.src.baseline_distributions import BaselineStore  # noqa: E402
from refute_b_v2_d32.adaptive_baseline import AdaptiveBaselineConfig, build_adaptive_baseline  # noqa: E402
from refute_b_v2_d32.joint_candidates import build_joint_prior, generate_joint_root_candidates  # noqa: E402
from refute_b_v2_d32.layer1 import D32Knowledge  # noqa: E402
from refute_b_v2_d32.layer2 import D32PipelineConfig, D32RefutationPipeline  # noqa: E402
from refute_b_v2_d32.portable_adapters import aiops2021_tabular_adapter, eadro_tabular_adapter  # noqa: E402
from refute_b_v2_d32.portable_kpi_canonicalizer import canonicalize_metric_kpis  # noqa: E402
from refute_b_v2_d32.portable_schema import NormalizedIncident, normalize_metric_frame  # noqa: E402
from refute_b_v2_d32.reason_classifier import (  # noqa: E402
    AIOPS2021_SCHEMA,
    FEATURE_NAMES,
    OPENRCA_SCHEMA,
    TARGET_BUCKETS,
    FeatureSchema,
    _joint_features_from_raw,
    evaluate_classifier,
    extract_features,
    train_reason_classifier,
)
from refute_b_v2_d32.schema import reason_bucket  # noqa: E402
from refute_b_v2_d32.signature import build_case_signature  # noqa: E402


CANONICAL_REASON_BY_BUCKET = {
    "cpu": "CPU fault",
    "memory": "high memory usage",
    "jvm_oom": "JVM Out of Memory (OOM) Heap",
    "disk_io": "high disk I/O read usage",
    "filesystem": "high disk space usage",
    "network_latency": "network delay",
    "network_packet_loss": "network loss",
    "db_connection": "db connection limit",
}

_DATASET_SCHEMAS: dict[str, FeatureSchema] = {
    "aiops2021": AIOPS2021_SCHEMA,
    "eadro": OPENRCA_SCHEMA,
}


def _schema_for_dataset(dataset: str) -> FeatureSchema:
    return _DATASET_SCHEMAS.get(dataset, OPENRCA_SCHEMA)


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
    parser.add_argument("--kpi-canonicalization", choices=["none", "openrca"], default="none")
    parser.add_argument("--joint-feature-mode", choices=["raw", "candidate"], default="raw",
                        help="raw uses metric-derived training joint features; candidate matches online D32 joint candidates.")
    parser.add_argument("--knowledge-json", default=None,
                        help="D32 knowledge JSON (same as run script --knowledge-json). "
                             "When provided with --joint-feature-mode candidate, training features "
                             "are built through the online _candidate_space (with _put_best dedup) "
                             "so they match the inference-time feature distribution exactly.")
    parser.add_argument("--candidate-space-alignment", action="store_true",
                        help="Force online _candidate_space alignment when --knowledge-json is given "
                             "(default: on when knowledge-json + candidate mode).")
    parser.add_argument("--joint-beam-per-reason", type=int, default=8)
    parser.add_argument("--joint-prior-scale", type=float, default=0.15)
    parser.add_argument("--joint-prior-offset", type=float, default=0.05)
    parser.add_argument("--max-joint-prior", type=float, default=1.8)
    parser.add_argument("--train-split-column", default=None)
    parser.add_argument("--train-split-value", default=None)
    parser.add_argument("--eval-split-value", default=None)
    parser.add_argument("--label-column", default="failure_type")
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
    all_metrics = canonicalize_metric_kpis(
        normalize_metric_frame(metrics_raw),
        dataset=args.dataset,
        mode=args.kpi_canonicalization,
    )

    train_cases = _filter_cases(cases, args.train_split_column, args.train_split_value)
    if args.max_cases is not None:
        train_cases = train_cases.head(max(0, int(args.max_cases))).copy()
    joint_prior = build_joint_prior(_prior_cases(train_cases, args.label_column))
    knowledge = _load_knowledge(args.knowledge_json)
    schema = _schema_for_dataset(args.dataset)
    X_train, y_train, train_case_ids = _extract_xy(
        dataset=args.dataset,
        cases=train_cases,
        metrics_raw=metrics_raw,
        logs_raw=logs_raw,
        trace_summaries=trace_summaries,
        topology=topology,
        historical=historical,
        all_metrics=all_metrics,
        joint_prior=joint_prior,
        knowledge=knowledge,
        args=args,
        schema=schema,
    )
    if len(X_train) == 0:
        raise RuntimeError("no training examples extracted")
    if len(set(y_train.tolist())) < 2:
        raise RuntimeError(f"need at least 2 classes to train, got {sorted(set(y_train.tolist()))}")

    clf = train_reason_classifier(X_train, y_train, schema=schema)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    clf.save(out_path)

    train_metrics = evaluate_classifier(clf, X_train, y_train, dataset_name=f"{args.dataset}_train", verbose=False)
    summary: dict[str, Any] = {
        "dataset": args.dataset,
        "design": "portable_dataset_with_openrca_reason_features",
        "classifier_path": str(out_path),
        "n_train": int(len(X_train)),
        "train_case_ids": [str(item) for item in train_case_ids],
        "train_label_counts": {str(k): int(v) for k, v in Counter(y_train).items()},
        "train_metrics": train_metrics,
        "train_topk": _topk_metrics(clf, X_train, y_train),
        "feature_names": list(schema.feature_names),
        "feature_schema": schema.to_dict(),
        "joint_feature_mode": args.joint_feature_mode,
        "split": {
            "column": args.train_split_column,
            "train_value": args.train_split_value,
            "eval_value": args.eval_split_value,
        },
        "leakage_scope": "training_split_only",
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
            joint_prior=joint_prior,
            knowledge=knowledge,
            args=args,
            schema=schema,
        )
        summary["n_eval"] = int(len(X_eval))
        summary["eval_case_ids"] = [str(item) for item in eval_case_ids]
        summary["eval_label_counts"] = {str(k): int(v) for k, v in Counter(y_eval).items()}
        summary["eval_metrics"] = evaluate_classifier(clf, X_eval, y_eval, dataset_name=f"{args.dataset}_eval", verbose=False)
        summary["eval_topk"] = _topk_metrics(clf, X_eval, y_eval)
        summary["eval_confusion_top1"] = _confusion_top1(clf, X_eval, y_eval)

    summary_path = Path(args.summary_json) if args.summary_json else out_path.with_name(out_path.stem + "_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "classifier_path": str(out_path),
        "summary_json": str(summary_path),
        "n_train": int(len(X_train)),
        "train_accuracy": train_metrics.get("accuracy", 0.0),
        "eval_accuracy": (summary.get("eval_metrics") or {}).get("accuracy"),
        "eval_recall_at_3": (summary.get("eval_topk") or {}).get("recall@3"),
    }, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


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
    joint_prior: Any,
    knowledge: D32Knowledge | None,
    args: argparse.Namespace,
    schema: FeatureSchema | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    schema = schema or OPENRCA_SCHEMA
    adapter = _build_adapter(
        dataset=dataset,
        cases=cases,
        metrics=metrics_raw,
        logs=logs_raw,
        trace_summaries=trace_summaries,
        topology=topology,
        kpi_canonicalization=args.kpi_canonicalization,
    )
    use_candidate_space = _use_candidate_space_alignment(args, knowledge)
    pipeline_config = _training_pipeline_config(dataset, args) if use_candidate_space else None
    services = _known_services(adapter, topology) if use_candidate_space else []
    features: list[np.ndarray] = []
    labels: list[str] = []
    case_ids: list[str] = []
    for incident in adapter.iter_incidents():
        label = _label_for_incident(incident, args.label_column)
        if label not in TARGET_BUCKETS:
            continue
        baseline = _baseline_for_incident(
            incident=incident,
            args=args,
            historical=historical,
            all_metrics=all_metrics,
        )
        if args.joint_feature_mode == "candidate":
            joint_candidates = _candidate_mode_candidates(
                use_candidate_space=use_candidate_space,
                knowledge=knowledge,
                pipeline_config=pipeline_config,
                services=services,
                topology=topology,
                incident=incident,
                baseline=baseline,
                joint_prior=joint_prior,
                args=args,
            )
            features.append(extract_features(
                metric_df=incident.metric_df,
                log_df=incident.log_df,
                trace_summary=incident.trace_summary,
                baseline=baseline,
                joint_candidates=joint_candidates,
                schema=schema,
            ))
        else:
            raw_joint = _joint_features_from_raw(incident.metric_df, baseline, schema)
            features.append(extract_features(
                metric_df=incident.metric_df,
                log_df=incident.log_df,
                trace_summary=incident.trace_summary,
                baseline=baseline,
                raw_joint_features=raw_joint,
                schema=schema,
            ))
        labels.append(label)
        case_ids.append(incident.case_id)
    X = np.stack(features, axis=0) if features else np.empty((0, len(schema.feature_names)))
    return X, np.array(labels, dtype=str), case_ids


def _use_candidate_space_alignment(args: argparse.Namespace, knowledge: D32Knowledge | None) -> bool:
    if args.joint_feature_mode != "candidate":
        return False
    if knowledge is None:
        return False
    return bool(args.candidate_space_alignment) or args.knowledge_json is not None


def _training_pipeline_config(dataset: str, args: argparse.Namespace) -> D32PipelineConfig:
    return D32PipelineConfig(
        dataset_name=dataset,
        enable_two_stage_selector=False,
        enable_joint_candidates=True,
        joint_beam_per_reason=int(args.joint_beam_per_reason),
        joint_prior_scale=float(args.joint_prior_scale),
        joint_prior_offset=float(args.joint_prior_offset),
        max_joint_prior=float(args.max_joint_prior),
        REASON_NAME_MAPS={dataset: {}},
        DATASET_FAMILY_MAPS={dataset: {}},
        portable_ontology_enabled=False,
    )


def _candidate_mode_candidates(
    *,
    use_candidate_space: bool,
    knowledge: D32Knowledge | None,
    pipeline_config: D32PipelineConfig | None,
    services: list[str],
    topology: Mapping[str, Any],
    incident: NormalizedIncident,
    baseline,
    joint_prior: Any,
    args: argparse.Namespace,
):
    if use_candidate_space and knowledge is not None and pipeline_config is not None:
        pipeline = D32RefutationPipeline(
            knowledge,
            {"rules": []},
            baseline,
            dict(topology.get("topology", topology)),
            services,
            config=pipeline_config,
        )
        d32 = incident.d32_inputs()
        signature = build_case_signature(
            incident.case_id,
            d32["metric_df"],
            d32["log_df"],
            d32["trace_summary"],
            baseline,
            d32["modal_status"],
        )
        clusters = knowledge.match_clusters(signature, case_id=incident.case_id, k=3)
        mined_rules = knowledge.mined_reason_priors(signature)
        return pipeline._candidate_space(
            clusters,
            mined_rules,
            signature,
            d32["metric_df"],
            d32["log_df"],
            d32["trace_summary"],
            int(incident.window_start_ts),
        )
    return generate_joint_root_candidates(
        metric_df=incident.metric_df,
        log_df=incident.log_df,
        trace_summary=incident.trace_summary,
        baseline=baseline,
        joint_prior=joint_prior,
        window_start_ts=int(incident.window_start_ts),
        beam_per_reason=int(args.joint_beam_per_reason),
        prior_scale=float(args.joint_prior_scale),
        prior_offset=float(args.joint_prior_offset),
        max_prior=float(args.max_joint_prior),
    )


def _known_services(adapter, topology: Mapping[str, Any]) -> list[str]:
    services: set[str] = set()
    for svc in topology.get("services", []) or []:
        if isinstance(svc, str):
            services.add(svc)
        elif isinstance(svc, Mapping) and svc.get("service"):
            services.add(str(svc["service"]))
    for incident in adapter.iter_incidents():
        for comp in incident.metric_df["cmdb_id"].dropna().astype(str).unique() if not incident.metric_df.empty else []:
            services.add(comp)
    return sorted(services)


def _load_knowledge(path: str | None) -> D32Knowledge | None:
    if not path:
        return None
    return D32Knowledge.load_json(path)


def _topk_metrics(clf, X: np.ndarray, y: np.ndarray, ks: tuple[int, ...] = (1, 2, 3, 5)) -> dict[str, Any]:
    if len(X) == 0:
        return {f"recall@{k}": 0.0 for k in ks} | {"n_samples": 0, "mean_rank": None}
    ranks: list[int | None] = []
    for xi, yi in zip(X, y):
        proba = clf.predict_proba(xi)
        ordered = [label for label, _ in sorted(proba.items(), key=lambda item: -float(item[1]))]
        ranks.append(ordered.index(str(yi)) + 1 if str(yi) in ordered else None)
    out: dict[str, Any] = {"n_samples": int(len(X))}
    for k in ks:
        out[f"recall@{k}"] = float(sum(rank is not None and rank <= k for rank in ranks) / len(ranks))
    valid = [rank for rank in ranks if rank is not None]
    out["mean_rank"] = float(np.mean(valid)) if valid else None
    out["missing_class"] = int(sum(rank is None for rank in ranks))
    return out


def _confusion_top1(clf, X: np.ndarray, y: np.ndarray) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for xi, yi in zip(X, y):
        pred = str(clf.predict(xi))
        gt = str(yi)
        out.setdefault(gt, {})
        out[gt][pred] = out[gt].get(pred, 0) + 1
    return out


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
    for key in (label_column, "failure_type", "fault_type", "故障类型"):
        value = incident.labels.get(key)
        if value is not None and str(value).strip():
            return reason_bucket(str(value))
    return ""


def _prior_cases(cases: pd.DataFrame, label_column: str) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for row in cases.to_dict(orient="records"):
        component = _first_nonempty(row, ("root_cause_component", "root_cause_service", "root_cause", "cmdb_id"))
        label = ""
        for key in (label_column, "failure_type", "fault_type", "故障类型"):
            value = row.get(key)
            if value is not None and str(value).strip():
                label = reason_bucket(str(value))
                break
        reason = CANONICAL_REASON_BY_BUCKET.get(label, label)
        if component and reason:
            out.append({"component": str(component), "reason": str(reason)})
    return out


def _first_nonempty(row: Mapping[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip() and str(value).strip().lower() != "nan":
            return str(value).strip()
    return ""


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
    kpi_canonicalization: str = "none",
):
    if dataset == "eadro":
        return eadro_tabular_adapter(
            cases=cases,
            metrics=metrics,
            logs=logs,
            trace_summaries=trace_summaries,
            topology=topology,
            kpi_canonicalization=kpi_canonicalization,
        )
    if dataset == "aiops2021":
        return aiops2021_tabular_adapter(
            cases=cases,
            metrics=metrics,
            logs=logs,
            trace_summaries=trace_summaries,
            topology=topology,
            kpi_canonicalization=kpi_canonicalization,
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
