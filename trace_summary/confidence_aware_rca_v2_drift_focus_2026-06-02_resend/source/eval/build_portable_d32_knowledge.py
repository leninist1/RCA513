#!/usr/bin/env python3
"""Build leakage-scoped D32 knowledge for portable datasets.

This is the portable counterpart of OpenRCA's Layer-1 knowledge construction.
It may use labels from an explicit training split/fold, but the online portable
runner consumes only the saved JSON artifact.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
for path in (PROJECT_ROOT, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from refute.src.baseline_distributions import BaselineStore  # noqa: E402
from refute_b_v2_d32.adaptive_baseline import AdaptiveBaselineConfig, build_adaptive_baseline  # noqa: E402
from refute_b_v2_d32.layer1 import (  # noqa: E402
    KnowledgeBuildConfig,
    _case_public,
    _cluster_cases,
    _mine_rules,
    flatten_signature,
)
from refute_b_v2_d32.portable_adapters import aiops2021_tabular_adapter, eadro_tabular_adapter  # noqa: E402
from refute_b_v2_d32.portable_ontology import collect_metric_bucket_signals  # noqa: E402
from refute_b_v2_d32.portable_schema import NormalizedIncident, normalize_metric_frame  # noqa: E402
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
}


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
    parser.add_argument("--modalities", default="metric,log,trace")
    parser.add_argument("--train-split-column", default=None)
    parser.add_argument("--train-split-value", default=None)
    parser.add_argument("--exclude-split-value", default=None,
                        help="Optional held-out value for fold knowledge, e.g. one Eadro source_file.")
    parser.add_argument("--label-column", default="failure_type")
    parser.add_argument("--component-label-column", default="root_cause_component")
    parser.add_argument("--out", required=True)
    parser.add_argument("--cluster-threshold", type=float, default=0.42)
    parser.add_argument("--min-rule-support", type=int, default=2)
    parser.add_argument("--min-rule-confidence", type=float, default=0.60)
    parser.add_argument("--disable-rule-validation", action="store_true")
    parser.add_argument("--max-cases", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    modalities = {part.strip() for part in str(args.modalities).split(",") if part.strip()}
    cases = pd.read_csv(args.cases_csv)
    cases = _filter_cases(cases, args.train_split_column, args.train_split_value, args.exclude_split_value)
    if args.max_cases is not None:
        cases = cases.head(max(0, int(args.max_cases))).copy()
    metrics_raw = pd.read_csv(args.metrics_csv) if "metric" in modalities else None
    logs_raw = pd.read_csv(args.logs_csv) if args.logs_csv and "log" in modalities else None
    trace_summaries = _load_trace_summaries(args.trace_summary_dir) if "trace" in modalities else {}
    topology = _load_json(args.topology_json, {})
    historical = _load_historical_baseline(args)
    all_metrics = normalize_metric_frame(metrics_raw) if metrics_raw is not None else pd.DataFrame(columns=["timestamp", "cmdb_id", "kpi_name", "value"])

    adapter = _build_adapter(
        dataset=args.dataset,
        cases=cases,
        metrics=metrics_raw,
        logs=logs_raw,
        trace_summaries=trace_summaries,
        topology=topology,
    )

    knowledge_cases: list[dict[str, Any]] = []
    label_counts: Counter[str] = Counter()
    for row_id, incident in enumerate(adapter.iter_incidents()):
        component = _component_label(incident, args.component_label_column)
        bucket = _reason_label_bucket(incident, args.label_column)
        reason = CANONICAL_REASON_BY_BUCKET.get(bucket, bucket)
        if not component or not reason:
            continue
        baseline = _baseline_for_incident(
            incident=incident,
            args=args,
            historical=historical,
            all_metrics=all_metrics,
        )
        signature = build_case_signature(
            incident.case_id,
            incident.metric_df,
            incident.log_df,
            incident.trace_summary,
            baseline,
            incident.modal_status,
        )
        _augment_signature_with_portable_metrics(signature, incident, baseline, args.dataset)
        joint_features = _portable_joint_features(incident, baseline, args.dataset)
        features = flatten_signature(signature)
        knowledge_cases.append({
            "case_id": incident.case_id,
            "row_id": row_id,
            "component": component,
            "reason": reason,
            "task_index": str(incident.metadata.get("source_id", incident.case_id)),
            "date_key": _case_group_key(incident),
            "signature": signature,
            "joint_features": joint_features,
            "features": features,
        })
        label_counts[reason] += 1

    config = KnowledgeBuildConfig(
        cluster_threshold=float(args.cluster_threshold),
        min_rule_support=int(args.min_rule_support),
        min_rule_confidence=float(args.min_rule_confidence),
        validate_rules_lodo=not bool(args.disable_rule_validation),
    )
    clusters = _cluster_cases(knowledge_cases, config.cluster_threshold)
    mined_rules = _mine_rules(knowledge_cases, config)
    out = {
        "version": 1,
        "design": "portable_d32_layer1_knowledge",
        "config": config.__dict__,
        "cases": [_case_public(row) for row in knowledge_cases],
        "clusters": clusters,
        "mined_rules": mined_rules,
        "metadata": {
            "dataset": args.dataset,
            "n_cases": len(knowledge_cases),
            "modalities": sorted(modalities),
            "split": {
                "column": args.train_split_column,
                "value": args.train_split_value,
                "exclude_value": args.exclude_split_value,
            },
            "label_counts": dict(label_counts),
            "leakage_scope": "training_split_or_fold_only",
        },
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "out": str(out_path),
        "n_cases": len(knowledge_cases),
        "n_clusters": len(clusters),
        "n_mined_rules": len(mined_rules),
        "label_counts": dict(label_counts),
    }, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _augment_signature_with_portable_metrics(
    signature: dict[str, Any],
    incident: NormalizedIncident,
    baseline: Any,
    dataset: str,
) -> None:
    service_rows = {str(row.get("service", "")): row for row in signature.get("services", []) if isinstance(row, dict)}
    signals = collect_metric_bucket_signals(incident.metric_df, baseline, dataset=dataset)
    for component, component_signals in signals.items():
        service = service_rows.setdefault(component, {"service": component, "metric": {}, "log": {}, "trace": {}, "topology": {}})
        if service not in signature.setdefault("services", []):
            signature["services"].append(service)
        metric = service.setdefault("metric", {})
        for bucket, signal in component_signals.items():
            item = metric.setdefault(bucket, {"state": "support", "intensity": 0, "strength": 0.0, "examples": []})
            item["state"] = "support"
            item["strength"] = max(float(item.get("strength", 0.0) or 0.0), float(signal.strength))
            item["intensity"] = max(int(item.get("intensity", 0) or 0), min(3, int(signal.count // 3) + 1))
            examples = item.setdefault("examples", [])
            for example in signal.examples:
                if len(examples) >= 3:
                    break
                examples.append(dict(example))


def _portable_joint_features(incident: NormalizedIncident, baseline: Any, dataset: str) -> list[str]:
    signals = collect_metric_bucket_signals(incident.metric_df, baseline, dataset=dataset)
    features: set[str] = set()
    for _component, component_signals in signals.items():
        for bucket, signal in component_signals.items():
            if signal.count <= 0:
                continue
            features.add(f"portable:bucket:{bucket}")
            if signal.count >= 5:
                features.add(f"portable:bucket:{bucket}:multi")
            if signal.strength >= 20.0:
                features.add(f"portable:bucket:{bucket}:strong")
            elif signal.strength >= 5.0:
                features.add(f"portable:bucket:{bucket}:medium")
    return sorted(features)


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


def _build_adapter(
    *,
    dataset: str,
    cases: pd.DataFrame,
    metrics: pd.DataFrame | None,
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


def _filter_cases(
    cases: pd.DataFrame,
    split_column: str | None,
    split_value: str | None,
    exclude_value: str | None,
) -> pd.DataFrame:
    if not split_column:
        return cases.copy()
    if split_column not in cases.columns:
        raise ValueError(f"split column not found in cases CSV: {split_column}")
    if split_value is None and exclude_value is None:
        return cases.copy()
    out = cases.copy()
    if split_value is not None:
        out = out[out[split_column].astype(str) == str(split_value)].copy()
    if exclude_value is not None:
        out = out[out[split_column].astype(str) != str(exclude_value)].copy()
    return out


def _component_label(incident: NormalizedIncident, label_column: str) -> str:
    for key in (label_column, "root_cause_component", "root_cause_service", "root_cause", "cmdb_id"):
        value = incident.labels.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _reason_label_bucket(incident: NormalizedIncident, label_column: str) -> str:
    for key in (label_column, "failure_type", "fault_type", "故障类型"):
        value = incident.labels.get(key)
        if value is not None and str(value).strip():
            return reason_bucket(str(value))
    return ""


def _case_group_key(incident: NormalizedIncident) -> str:
    for key in ("source_file", "source_id", "fault_index", "data_type"):
        value = incident.metadata.get(key) or incident.labels.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return str(incident.case_id).split("_", 1)[0]


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
