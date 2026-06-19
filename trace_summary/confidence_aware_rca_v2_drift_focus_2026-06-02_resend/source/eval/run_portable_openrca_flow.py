#!/usr/bin/env python3
"""Run Eadro/AIOps2021 through the base OpenRCA D32 pipeline.

Only dataset loading/baseline construction are adapted. Candidate generation,
reason-first selector, component scorer, and time anchor are the inherited
`D32RefutationPipeline` implementations.
"""
from __future__ import annotations

import argparse
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
from refute_b_v2_d32.entity_roles import infer_roles_for_metric_frame  # noqa: E402
from refute_b_v2_d32.layer1 import D32Knowledge  # noqa: E402
from refute_b_v2_d32.layer2 import D32PipelineConfig, D32RefutationPipeline  # noqa: E402
from refute_b_v2_d32.portable_adapters import aiops2021_tabular_adapter, eadro_tabular_adapter  # noqa: E402
from refute_b_v2_d32.portable_kpi_canonicalizer import canonicalize_metric_kpis  # noqa: E402
from refute_b_v2_d32.portable_schema import NormalizedIncident, normalize_metric_frame  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["eadro", "aiops2021"], required=True)
    parser.add_argument("--cases-csv", required=True)
    parser.add_argument("--metrics-csv", default=None)
    parser.add_argument("--logs-csv", default=None)
    parser.add_argument("--trace-summary-dir", default=None)
    parser.add_argument("--topology-json", default=None)
    parser.add_argument("--rules", default="knowledge/refutation_rules_v2.json")
    parser.add_argument("--knowledge-json", default=None)
    parser.add_argument("--baseline-mode", choices=["historical", "adaptive"], default="adaptive")
    parser.add_argument("--historical-baseline-json", default=None)
    parser.add_argument("--pre-baseline-sec", type=int, default=0)
    parser.add_argument("--kpi-canonicalization", choices=["none", "openrca"], default="none")
    parser.add_argument("--case-filter-column", default=None)
    parser.add_argument("--case-filter-value", default=None)
    parser.add_argument("--case-exclude-value", default=None)
    parser.add_argument("--modalities", default="metric,log,trace")
    parser.add_argument("--failure-count", type=int, default=1)
    parser.add_argument("--out", default="logs/portable_openrca_flow_predictions.csv")
    parser.add_argument("--debug-json", default="logs/portable_openrca_flow_debug.json")
    parser.add_argument("--baseline-reliability-json", default="logs/portable_openrca_flow_baseline_reliability.json")
    parser.add_argument("--input-audit-jsonl", default="logs/portable_openrca_flow_input_audit.jsonl")
    parser.add_argument("--reason-classifier-path", default=None)
    parser.add_argument("--reason-classifier-fold", type=int, default=None)
    parser.add_argument("--casefold-classifier-dir", default="knowledge/casefold_classifiers")
    parser.add_argument("--strict-reason-classifier", action="store_true")
    parser.add_argument("--use-openrca-default-family-map", action="store_true",
                        help="Use D32 default docker/os/db family filters. Default uses an empty dataset family map as data ontology adaptation.")
    parser.add_argument("--disable-joint-candidates", action="store_true")
    parser.add_argument("--joint-beam-per-reason", type=int, default=8)
    parser.add_argument("--joint-prior-scale", type=float, default=0.15)
    parser.add_argument("--joint-prior-offset", type=float, default=0.05)
    parser.add_argument("--max-joint-prior", type=float, default=1.8)
    parser.add_argument("--legacy-component-evidence-scale-with-joint", type=float, default=1.0)
    parser.add_argument("--disable-window-reason-scores", action="store_true")
    parser.add_argument("--window-reason-score-credit", type=float, default=0.65)
    parser.add_argument("--window-reason-top-k", type=int, default=3)
    parser.add_argument("--window-reason-weak-penalty", type=float, default=0.8)
    parser.add_argument("--window-reason-type-bonus", type=float, default=0.6)
    parser.add_argument("--window-reason-type-penalty", type=float, default=0.4)
    parser.add_argument("--symptom-reason-score-credit", type=float, default=1.0)
    parser.add_argument("--symptom-reason-top-k", type=int, default=8)
    parser.add_argument("--time-anchor-early-bonus", type=float, default=0.25)
    parser.add_argument("--time-anchor-sustained-bonus", type=float, default=2.0)
    parser.add_argument("--onset-first", dest="onset_first", action="store_true", default=True)
    parser.add_argument("--no-onset-first", dest="onset_first", action="store_false")
    parser.add_argument("--onset-first-weight", type=float, default=4.0)
    parser.add_argument("--onset-decay-seconds", type=float, default=300.0)
    parser.add_argument("--component-onset-bonus", type=float, default=5.0)
    parser.add_argument("--max-cases", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _validate_reason_classifier_args(args)
    modalities = {part.strip() for part in str(args.modalities).split(",") if part.strip()}
    cases = pd.read_csv(args.cases_csv)
    cases = _filter_cases(cases, args.case_filter_column, args.case_filter_value, args.case_exclude_value)
    metrics_raw = _read_optional_csv(args.metrics_csv) if "metric" in modalities else None
    logs_raw = _read_optional_csv(args.logs_csv) if "log" in modalities else None
    trace_summaries = _load_trace_summaries(args.trace_summary_dir) if "trace" in modalities else {}
    topology = _load_json(args.topology_json, default={})
    rules = _load_json(args.rules, default={"rules": []})
    historical = _load_historical_baseline(args)
    all_metrics = (
        canonicalize_metric_kpis(
            normalize_metric_frame(metrics_raw),
            dataset=args.dataset,
            mode=args.kpi_canonicalization,
        )
        if metrics_raw is not None
        else pd.DataFrame(columns=["timestamp", "cmdb_id", "kpi_name", "value"])
    )

    adapter = _build_adapter(
        dataset=args.dataset,
        cases=cases,
        metrics=metrics_raw if "metric" in modalities else None,
        logs=logs_raw if "log" in modalities else None,
        trace_summaries=trace_summaries if "trace" in modalities else {},
        topology=topology,
        kpi_canonicalization=args.kpi_canonicalization,
    )
    incidents = list(adapter.iter_incidents())
    if args.max_cases is not None:
        incidents = incidents[: max(0, int(args.max_cases))]

    knowledge = _load_knowledge(args.knowledge_json)
    services = _known_services(incidents, topology)
    config = _pipeline_config(args)
    completed: dict[int, dict[str, Any]] = {}
    baseline_audit: list[dict[str, Any]] = []
    input_audit_rows: list[dict[str, Any]] = []

    for row_id, incident in enumerate(incidents):
        baseline = _baseline_for_incident(
            args=args,
            incident=incident,
            historical=historical,
            all_metrics=all_metrics,
        )
        pipeline = D32RefutationPipeline(
            knowledge,
            rules,
            baseline,
            _node_graph_for_incident(incident, topology),
            services,
            config=config,
        )
        d32_inputs = incident.d32_inputs()
        result = pipeline.select(
            case_id=incident.case_id,
            metric_df=d32_inputs["metric_df"],
            log_df=d32_inputs["log_df"],
            trace_summary=d32_inputs["trace_summary"],
            modal_status=d32_inputs["modal_status"],
            failure_count=max(1, int(args.failure_count)),
            window_start_ts=int(incident.window_start_ts),
        )
        completed[row_id] = {
            "row_id": row_id,
            "case_id": incident.case_id,
            "prediction": result.prediction,
            "debug": {
                "row_id": row_id,
                "case_id": incident.case_id,
                "dataset": incident.dataset,
                "algorithm_flow": "base_openrca_d32",
                "window": {
                    "start_ts": int(incident.window_start_ts),
                    "end_ts": int(incident.window_end_ts),
                    "failure_count": max(1, int(args.failure_count)),
                },
                "modal_status": dict(incident.modal_status),
                "d32_result": result.to_dict(),
            },
        }
        reliability = _baseline_reliability(baseline, incident.metric_df)
        baseline_audit.append({
            "case_id": incident.case_id,
            "baseline_mode": args.baseline_mode,
            "baseline_reliability": reliability,
        })
        input_audit_rows.append(_input_audit_row(incident, reliability))
        print(f"processed {row_id + 1}/{len(incidents)} {incident.case_id}", flush=True)

    _write_predictions_csv(completed, Path(args.out))
    _write_debug_json(completed, Path(args.debug_json))
    _write_json(Path(args.baseline_reliability_json), {
        "dataset": args.dataset,
        "algorithm_flow": "base_openrca_d32",
        "family_map_mode": "openrca_default" if args.use_openrca_default_family_map else "empty_dataset_map",
        "kpi_canonicalization": args.kpi_canonicalization,
        "knowledge_json": args.knowledge_json,
        "reason_classifier_path": args.reason_classifier_path,
        "baseline_mode": args.baseline_mode,
        "n_cases": len(incidents),
        "cases": baseline_audit,
    })
    _write_jsonl(Path(args.input_audit_jsonl), input_audit_rows)
    print(f"wrote {args.out}")
    print(f"wrote {args.debug_json}")
    print(f"wrote {args.baseline_reliability_json}")
    print(f"wrote {args.input_audit_jsonl}")
    return 0


def _pipeline_config(args: argparse.Namespace) -> D32PipelineConfig:
    family_kwargs: dict[str, Any] = {}
    if not args.use_openrca_default_family_map:
        family_kwargs = {"DATASET_FAMILY_MAPS": {args.dataset: {}}}
    classifier_path = args.reason_classifier_path or "knowledge/__portable_openrca_reason_classifier_disabled__.json"
    return D32PipelineConfig(
        dataset_name=args.dataset,
        reason_classifier_fold=args.reason_classifier_fold,
        casefold_classifier_dir=args.casefold_classifier_dir,
        strict_reason_classifier=bool(args.strict_reason_classifier),
        reason_classifier_path=classifier_path,
        enable_joint_candidates=not args.disable_joint_candidates,
        joint_beam_per_reason=args.joint_beam_per_reason,
        joint_prior_scale=args.joint_prior_scale,
        joint_prior_offset=args.joint_prior_offset,
        max_joint_prior=args.max_joint_prior,
        legacy_component_evidence_scale_with_joint=args.legacy_component_evidence_scale_with_joint,
        enable_window_reason_scores=not args.disable_window_reason_scores,
        window_reason_score_credit=args.window_reason_score_credit,
        window_reason_top_k=args.window_reason_top_k,
        window_reason_weak_penalty=args.window_reason_weak_penalty,
        window_reason_type_bonus=args.window_reason_type_bonus,
        window_reason_type_penalty=args.window_reason_type_penalty,
        symptom_reason_score_credit=args.symptom_reason_score_credit,
        symptom_reason_top_k=args.symptom_reason_top_k,
        time_anchor_early_bonus=args.time_anchor_early_bonus,
        time_anchor_sustained_bonus=args.time_anchor_sustained_bonus,
        onset_first=args.onset_first,
        onset_first_weight=args.onset_first_weight,
        onset_decay_seconds=args.onset_decay_seconds,
        component_onset_bonus=args.component_onset_bonus,
        REASON_NAME_MAPS={args.dataset: {}},
        portable_ontology_enabled=False,
        **family_kwargs,
    )


def _validate_reason_classifier_args(args: argparse.Namespace) -> None:
    if args.reason_classifier_path:
        path = Path(args.reason_classifier_path)
        if not path.exists():
            raise FileNotFoundError(f"--reason-classifier-path not found: {path}")
        if not path.with_suffix(".pkl").exists():
            raise FileNotFoundError(f"reason classifier pickle not found: {path.with_suffix('.pkl')}")


def _filter_cases(
    cases: pd.DataFrame,
    column: str | None,
    value: str | None,
    exclude_value: str | None,
) -> pd.DataFrame:
    if not column:
        return cases.copy()
    if column not in cases.columns:
        raise ValueError(f"case filter column not found in cases CSV: {column}")
    out = cases.copy()
    if value is not None:
        out = out[out[column].astype(str) == str(value)].copy()
    if exclude_value is not None:
        out = out[out[column].astype(str) != str(exclude_value)].copy()
    return out


def _build_adapter(
    *,
    dataset: str,
    cases: pd.DataFrame,
    metrics: pd.DataFrame | None,
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


def _load_knowledge(path: str | None) -> D32Knowledge:
    if path:
        return D32Knowledge.load_json(path)
    return D32Knowledge({
        "version": 1,
        "design": "d32_empty_no_gt_knowledge",
        "cases": [],
        "clusters": [],
        "mined_rules": [],
        "metadata": {"source": "empty_default_for_portable_openrca_flow_runner"},
    })


def _load_historical_baseline(args: argparse.Namespace) -> BaselineStore | None:
    if args.historical_baseline_json:
        return BaselineStore.load_json(args.historical_baseline_json)
    if args.baseline_mode == "historical":
        raise ValueError("--baseline-mode historical requires --historical-baseline-json")
    return None


def _baseline_for_incident(
    *,
    args: argparse.Namespace,
    incident: NormalizedIncident,
    historical: BaselineStore | None,
    all_metrics: pd.DataFrame,
):
    if args.baseline_mode == "historical":
        return historical
    baseline_frame = incident.metric_df
    if args.pre_baseline_sec > 0 and all_metrics is not None and not all_metrics.empty:
        start = int(incident.window_start_ts)
        pre_start = start - int(args.pre_baseline_sec)
        pre_rows = all_metrics[(all_metrics["timestamp"] >= pre_start) & (all_metrics["timestamp"] < start)].copy()
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


def _baseline_reliability(baseline: Any, metric_df: pd.DataFrame) -> list[dict[str, Any]]:
    if hasattr(baseline, "reliability_report"):
        return list(baseline.reliability_report(metric_df))
    total = len(baseline) if baseline is not None else 0
    eligible = sum(1 for stats in getattr(baseline, "stats", {}).values() if stats.eligible) if baseline is not None else 0
    return [{
        "source": "historical",
        "n_stats": total,
        "eligible_stats": eligible,
        "coverage": _historical_coverage(baseline, metric_df),
    }]


def _historical_coverage(baseline: BaselineStore | None, metric_df: pd.DataFrame) -> float:
    if baseline is None or metric_df is None or metric_df.empty:
        return 0.0
    keys = {
        (str(row.cmdb_id), str(row.kpi_name))
        for row in metric_df[["cmdb_id", "kpi_name"]].dropna().itertuples(index=False)
    }
    if not keys:
        return 0.0
    covered = sum(1 for key in keys if baseline.get(*key) is not None and baseline.get(*key).eligible)
    return covered / len(keys)


def _known_services(incidents: Iterable[NormalizedIncident], topology: Mapping[str, Any]) -> list[str]:
    services: set[str] = set()
    for incident in incidents:
        if incident.metric_df is not None and not incident.metric_df.empty:
            services.update(str(item) for item in incident.metric_df["cmdb_id"].dropna().astype(str).unique())
        if incident.log_df is not None and not incident.log_df.empty and "cmdb_id" in incident.log_df.columns:
            services.update(str(item) for item in incident.log_df["cmdb_id"].dropna().astype(str).unique())
    services.update(str(item) for item in dict(topology.get("containers", {}) or {}).keys())
    services.update(str(item) for item in dict(topology.get("nodes", {}) or {}).keys())
    return sorted(services) or ["unknown_component"]


def _node_graph_for_incident(incident: NormalizedIncident, topology: Mapping[str, Any]) -> dict[str, Any]:
    graph = dict(topology or {})
    graph.setdefault("dataset", incident.dataset)
    containers = dict(graph.get("containers", {}) or {})
    roles = infer_roles_for_metric_frame(incident.metric_df, graph)
    component_ids = {
        str(item)
        for item in incident.metric_df["cmdb_id"].dropna().astype(str).unique()
    } if incident.metric_df is not None and not incident.metric_df.empty else set()
    for component in sorted(component_ids):
        role = roles.get(component)
        row = dict(containers.get(component, {}) or {})
        row.setdefault("node_proxy", component)
        if role is not None:
            row.setdefault("portable_role", role.role)
            row.setdefault("portable_role_confidence", role.confidence)
        containers[component] = row
    graph["containers"] = containers
    graph.setdefault("nodes", {})
    return graph


def _input_audit_row(incident: NormalizedIncident, baseline_reliability: list[dict[str, Any]]) -> dict[str, Any]:
    roles = infer_roles_for_metric_frame(incident.metric_df, incident.topology)
    return {
        "case_id": incident.case_id,
        "dataset": incident.dataset,
        "window_start_ts": int(incident.window_start_ts),
        "window_end_ts": int(incident.window_end_ts),
        "modal_status": dict(incident.modal_status),
        "metric_rows": int(len(incident.metric_df)),
        "log_rows": int(len(incident.log_df)),
        "trace_status": str((incident.trace_summary or {}).get("trace_status", "unloaded")),
        "labels_present": sorted(incident.labels),
        "role_summary": {key: role.to_dict() for key, role in sorted(roles.items())},
        "baseline_reliability": baseline_reliability,
    }


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


def _read_optional_csv(path: str | None) -> pd.DataFrame | None:
    if not path:
        return None
    return pd.read_csv(path)


def _load_json(path: str | None, default: Any) -> Any:
    if not path:
        return default
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_predictions_csv(completed: Mapping[int, Mapping[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "row_id": row["row_id"],
            "case_id": row["case_id"],
            "prediction": json.dumps(row["prediction"], ensure_ascii=False),
        }
        for _, row in sorted(completed.items())
    ]
    pd.DataFrame(rows).to_csv(out_path, index=False)


def _write_debug_json(completed: Mapping[int, Mapping[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ordered = [row for _, row in sorted(completed.items())]
    out_path.write_text(
        json.dumps({"n": len(ordered), "debug": [row["debug"] for row in ordered]}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
