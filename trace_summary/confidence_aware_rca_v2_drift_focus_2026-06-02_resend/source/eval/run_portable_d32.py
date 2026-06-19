"""Opt-in portable D32 runner for Eadro/AIOps2021-style datasets.

This runner does not replace or modify the OpenRCA LODO entrypoint.  It uses a
normalized incident boundary, keeps labels out of online inputs, and records
baseline/adapter audit information for cross-dataset diagnosis.
"""
from __future__ import annotations

import argparse
import json
import math
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
from refute_b_v2_d32.portable_ontology import (  # noqa: E402
    CANONICAL_REASON_BY_BUCKET,
    bucket_scores_from_signals,
    collect_metric_bucket_signals,
    extract_portable_reason_features,
    infer_kpi_buckets,
    portable_root_candidates,
)
from refute_b_v2_d32.portable_schema import NormalizedIncident, normalize_metric_frame  # noqa: E402
from refute_b_v2_d32.reason_classifier import ReasonClassifier  # noqa: E402
from refute_b_v2_d32.schema import D32Result, reason_bucket  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["eadro", "aiops2021"], required=True)
    parser.add_argument("--portable-profile", choices=["metric-only", "full-d32"], default="metric-only",
                        help="full-d32 requires split/fold-scoped knowledge and enables an aligned D32 artifact path.")
    parser.add_argument("--cases-csv", required=True)
    parser.add_argument("--metrics-csv", default=None)
    parser.add_argument("--logs-csv", default=None)
    parser.add_argument("--trace-summary-dir", default=None)
    parser.add_argument("--topology-json", default=None)
    parser.add_argument("--rules", default="knowledge/refutation_rules_v2.json")
    parser.add_argument("--knowledge-json", default=None,
                        help="Optional external D32 knowledge. Default is empty no-GT knowledge.")
    parser.add_argument("--baseline-mode", choices=["historical", "adaptive"], default="adaptive")
    parser.add_argument("--historical-baseline-json", default=None)
    parser.add_argument("--pre-baseline-sec", type=int, default=0,
                        help="Seconds of metric context before case start used only for adaptive baseline.")
    parser.add_argument("--case-filter-column", default=None)
    parser.add_argument("--case-filter-value", default=None)
    parser.add_argument("--case-exclude-value", default=None)
    parser.add_argument("--modalities", default="metric,log,trace")
    parser.add_argument("--failure-count", type=int, default=1)
    parser.add_argument("--out", default="logs/portable_d32_predictions.csv")
    parser.add_argument("--debug-json", default="logs/portable_d32_debug.json")
    parser.add_argument("--baseline-reliability-json", default="logs/portable_d32_baseline_reliability.json")
    parser.add_argument("--input-audit-jsonl", default="logs/portable_d32_input_audit.jsonl")
    parser.add_argument("--disable-joint-candidates", action="store_true")
    parser.add_argument("--enable-family-filter", action="store_true",
                        help="Opt into existing reason-family filters. Default disables them via an empty dataset map.")
    parser.add_argument("--reason-classifier-path", default=None,
                        help="Optional portable reason classifier JSON. Must have matching .pkl.")
    parser.add_argument("--reason-classifier-fold", type=int, default=None,
                        help="Optional same-dataset casefold classifier index.")
    parser.add_argument("--casefold-classifier-dir", default="knowledge/casefold_classifiers")
    parser.add_argument("--strict-reason-classifier", action="store_true",
                        help="Fail if the requested classifier cannot be loaded.")
    parser.add_argument("--reason-classifier-usage", choices=["selector", "label-only"], default="selector",
                        help="selector changes D32 ranking; label-only keeps D32 ranking and replaces only the output reason.")
    parser.add_argument("--disable-portable-ontology", action="store_true",
                        help="Disable portable ontology candidate/score expansion inside the portable D32 selector.")
    parser.add_argument("--max-cases", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _validate_reason_classifier_args(args)
    _validate_profile_args(args)
    modalities = {part.strip() for part in str(args.modalities).split(",") if part.strip()}
    cases = pd.read_csv(args.cases_csv)
    cases = _filter_cases(cases, args.case_filter_column, args.case_filter_value, args.case_exclude_value)
    metrics_raw = _read_optional_csv(args.metrics_csv)
    logs_raw = _read_optional_csv(args.logs_csv)
    trace_summaries = _load_trace_summaries(args.trace_summary_dir)
    topology = _load_json(args.topology_json, default={})
    rules = _load_json(args.rules, default={"rules": []})
    historical = _load_historical_baseline(args)
    all_metrics = normalize_metric_frame(metrics_raw) if metrics_raw is not None else pd.DataFrame(columns=["timestamp", "cmdb_id", "kpi_name", "value"])

    adapter = _build_adapter(
        dataset=args.dataset,
        cases=cases,
        metrics=metrics_raw if "metric" in modalities else None,
        logs=logs_raw if "log" in modalities else None,
        trace_summaries=trace_summaries if "trace" in modalities else {},
        topology=topology,
    )
    incidents = list(adapter.iter_incidents())
    if args.max_cases is not None:
        incidents = incidents[: max(0, int(args.max_cases))]

    knowledge = _load_knowledge(args.knowledge_json)
    services = _known_services(incidents, topology)
    config = _pipeline_config(args)
    reason_labeler = _load_reason_labeler(args) if args.reason_classifier_usage == "label-only" else None
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
        pipeline = _PortableD32Pipeline(
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
        if reason_labeler is not None:
            result = _apply_label_only_reason_classifier(
                result=result,
                classifier=reason_labeler,
                incident=incident,
                baseline=baseline,
                dataset=args.dataset,
            )
        completed[row_id] = {
            "row_id": row_id,
            "case_id": incident.case_id,
            "prediction": result.prediction,
            "debug": {
                "row_id": row_id,
                "case_id": incident.case_id,
                "dataset": incident.dataset,
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


class _PortableD32Pipeline(D32RefutationPipeline):
    """Keep portable fallback posterior keyed by reason bucket.

    The regular OpenRCA path usually uses a trained same-dataset classifier that
    already returns bucket keys.  Portable runs default to no-GT empty knowledge,
    so the inherited heuristic may return reason names.  This adapter-local
    normalization avoids touching the OpenRCA pipeline implementation.
    """

    def _candidate_space(
        self,
        clusters,
        mined_rules,
        signature,
        metric_df,
        log_df,
        trace_summary,
        window_start_ts,
    ):
        candidates = list(super()._candidate_space(
            clusters,
            mined_rules,
            signature,
            metric_df,
            log_df,
            trace_summary,
            window_start_ts,
        ))
        if not bool(getattr(self.config, "portable_ontology_enabled", False)):
            return candidates
        out = {candidate.key(): candidate for candidate in candidates}
        for candidate in portable_root_candidates(
            metric_df=metric_df,
            baseline=self.baseline,
            dataset=self.config.dataset_name,
            window_start_ts=int(window_start_ts),
        ):
            self._put_best(out, candidate)
        return sorted(out.values(), key=lambda row: (-row.prior, row.component, row.reason))[:80]

    def _compute_reason_posterior(self, candidates, evidence, signature) -> dict[str, float]:
        raw = super()._compute_reason_posterior(candidates, evidence, signature)
        reason_to_bucket = {str(candidate.reason): str(candidate.reason_bucket) for candidate in candidates}
        bucket_scores: dict[str, float] = {}
        for name, score in raw.items():
            bucket = reason_to_bucket.get(str(name), reason_bucket(str(name)))
            bucket_scores[bucket] = max(bucket_scores.get(bucket, 0.0), float(score or 0.0))
        for candidate in candidates:
            bucket_scores.setdefault(str(candidate.reason_bucket), 0.005)
        if bool(getattr(self.config, "portable_ontology_enabled", False)):
            signals = collect_metric_bucket_signals(evidence.metric_df, evidence.baseline, dataset=self.config.dataset_name)
            for bucket, score in bucket_scores_from_signals(signals, window_start_ts=_window_start_from_signature(signature)).items():
                bucket_scores[bucket] = max(bucket_scores.get(bucket, 0.0), 0.15 * float(score))
        return bucket_scores

    def _learned_reason_posterior(self, candidates, evidence) -> dict[str, float]:
        feats = extract_portable_reason_features(
            metric_df=evidence.metric_df,
            log_df=evidence.log_df,
            trace_summary=evidence.trace_summary,
            baseline=evidence.baseline,
            dataset=self.config.dataset_name,
        )
        if self._reason_classifier is None:
            return {}
        proba = self._reason_classifier.predict_proba(feats)
        seen_buckets = {str(candidate.reason_bucket) for candidate in candidates}
        for bucket in sorted(seen_buckets):
            proba.setdefault(bucket, 0.005)
        return proba

    def _component_earliness_strength(self, component, reason_bucket_name, evidence):
        df = evidence.metric_df
        if not bool(getattr(self.config, "portable_ontology_enabled", False)):
            return super()._component_earliness_strength(component, reason_bucket_name, evidence)
        if df is None or df.empty or "kpi_name" not in df.columns:
            return (0.0, 0.0, 0.0)
        comp_rows = df[df["cmdb_id"].astype(str) == str(component)]
        if comp_rows.empty:
            return (0.0, 0.0, 0.0)

        first_ts = None
        max_dev = 0.0
        anom_count = 0
        window_start = float(comp_rows["timestamp"].min())
        window_end = float(comp_rows["timestamp"].max())
        window_duration = max(window_end - window_start, 1.0)
        bucket = reason_bucket(str(reason_bucket_name))

        for row in comp_rows.itertuples(index=False):
            kpi = str(row.kpi_name)
            if bucket not in infer_kpi_buckets(kpi, dataset=self.config.dataset_name):
                continue
            try:
                value = float(row.value)
            except Exception:
                continue
            if not math.isfinite(value):
                continue
            try:
                result = evidence.baseline.is_anomalous(str(row.cmdb_id), kpi, value, threshold="p99")
            except Exception:
                continue
            if not result.is_anomalous:
                continue
            dev = abs(float(result.deviation or 0.0))
            anom_count += 1
            max_dev = max(max_dev, dev)
            ts = float(row.timestamp)
            if first_ts is None or ts < first_ts:
                first_ts = ts

        if first_ts is None:
            return (0.0, 0.0, 0.0)
        earliness = 1.0 - min(1.0, (first_ts - window_start) / window_duration)
        return (earliness, math.log1p(max_dev), math.log1p(anom_count))


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


def _pipeline_config(args: argparse.Namespace) -> D32PipelineConfig:
    family_maps = {args.dataset: {}}
    if args.enable_family_filter:
        family_maps = {}
    return D32PipelineConfig(
        dataset_name=args.dataset,
        reason_classifier_fold=args.reason_classifier_fold,
        casefold_classifier_dir=args.casefold_classifier_dir,
        strict_reason_classifier=bool(args.strict_reason_classifier),
        reason_classifier_path=(
            args.reason_classifier_path
            if args.reason_classifier_usage == "selector"
            else "knowledge/__portable_reason_classifier_disabled__.json"
        ) or "knowledge/__portable_reason_classifier_disabled__.json",
        enable_joint_candidates=not args.disable_joint_candidates,
        DATASET_FAMILY_MAPS=family_maps,
        REASON_NAME_MAPS={args.dataset: {}},
        portable_ontology_enabled=not bool(args.disable_portable_ontology),
    )


def _validate_reason_classifier_args(args: argparse.Namespace) -> None:
    if args.reason_classifier_path:
        path = Path(args.reason_classifier_path)
        if not path.exists():
            raise FileNotFoundError(f"--reason-classifier-path not found: {path}")
        if not path.with_suffix(".pkl").exists():
            raise FileNotFoundError(f"reason classifier pickle not found: {path.with_suffix('.pkl')}")


def _validate_profile_args(args: argparse.Namespace) -> None:
    if args.portable_profile == "full-d32" and not args.knowledge_json:
        raise ValueError("--portable-profile full-d32 requires --knowledge-json built from a training split/fold")


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


def _load_reason_labeler(args: argparse.Namespace) -> ReasonClassifier | None:
    if not args.reason_classifier_path:
        return None
    return ReasonClassifier.load(args.reason_classifier_path)


def _apply_label_only_reason_classifier(
    *,
    result: D32Result,
    classifier: ReasonClassifier,
    incident: NormalizedIncident,
    baseline: Any,
    dataset: str,
) -> D32Result:
    features = extract_portable_reason_features(
        metric_df=incident.metric_df,
        log_df=incident.log_df,
        trace_summary=incident.trace_summary,
        baseline=baseline,
        dataset=dataset,
    )
    posterior = classifier.predict_proba(features)
    if not posterior:
        return result
    bucket = max(posterior, key=posterior.get)
    reason = CANONICAL_REASON_BY_BUCKET.get(str(bucket), str(bucket))
    prediction = {
        rank: {
            **dict(item),
            "root cause reason": reason,
        }
        for rank, item in result.prediction.items()
    }
    debug = {
        **dict(result.debug),
        "portable_reason_label_only": {
            "selected_bucket": str(bucket),
            "selected_reason": reason,
            "posterior": {str(k): float(v) for k, v in posterior.items()},
        },
    }
    return D32Result(
        prediction=prediction,
        high_suspicion=result.high_suspicion,
        low_suspicion=result.low_suspicion,
        data_blind_spots=result.data_blind_spots,
        debug=debug,
    )


def _window_start_from_signature(signature: Mapping[str, Any]) -> int | None:
    services = signature.get("services", []) if isinstance(signature, Mapping) else []
    for service in services:
        for modality in ("metric", "log", "trace"):
            rows = service.get(modality, {}) if isinstance(service, Mapping) else {}
            for item in rows.values() if isinstance(rows, Mapping) else []:
                for example in item.get("examples", []) if isinstance(item, Mapping) else []:
                    ts = example.get("timestamp") if isinstance(example, Mapping) else None
                    if ts is not None:
                        return int(ts)
    return None


def _load_knowledge(path: str | None) -> D32Knowledge:
    if path:
        return D32Knowledge.load_json(path)
    return D32Knowledge({
        "version": 1,
        "design": "d32_empty_no_gt_knowledge",
        "cases": [],
        "clusters": [],
        "mined_rules": [],
        "metadata": {"source": "empty_default_for_portable_runner"},
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
