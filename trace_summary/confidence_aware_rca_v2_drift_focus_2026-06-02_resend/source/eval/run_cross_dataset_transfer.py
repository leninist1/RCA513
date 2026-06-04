"""Cross-dataset D32 knowledge transfer experiment.

Builds fault-cluster knowledge and mined rules from source datasets,
then evaluates on a target dataset.

Usage examples:
  # Train on Market+Telecom, test on Bank (metric+log+trace)
  python eval/run_cross_dataset_transfer.py \
    --sources market_cloudbed_1 market_cloudbed_2 telecom \
    --target bank \
    --modalities metric,log,trace

  # Train on Market only, test on Bank (metric+log+trace)
  python eval/run_cross_dataset_transfer.py \
    --sources market_cloudbed_1 market_cloudbed_2 \
    --target bank \
    --modalities metric,log,trace

  # Train on Telecom only, test on Bank
  python eval/run_cross_dataset_transfer.py \
    --sources telecom \
    --target bank \
    --modalities metric,trace
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT, PROJECT_ROOT.parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

os.environ.setdefault("TZ", "Asia/Shanghai")

from refute.src.baseline_distributions import BaselineStore  # noqa: E402
from refute.src.data_loader import BankDataPaths  # noqa: E402
from refute.src.data_loader import load_log_day, load_metric_day  # noqa: E402
from refute_b_v2.query_windows import parse_query_window  # noqa: E402
from refute_b_v2_d32.layer1 import (  # noqa: E402
    D32Knowledge,
    KnowledgeBuildConfig,
    build_knowledge,
    case_rows_from_openrca,
    flatten_signature,
    load_trace_summary,
    save_knowledge,
    _centroid,
    _cluster_cases,
    _mine_rules,
    _mine_rules_once,
)
from refute_b_v2_d32.layer2 import D32PipelineConfig, D32RefutationPipeline  # noqa: E402
from refute_b_v2_d32.schema import reason_bucket  # noqa: E402
from refute_b_v2_d32.signature import build_case_signature  # noqa: E402


# ──────────────────────────── dataset registry ────────────────────────────
DEFAULT_DATA_BASE = Path("/home/dell2/RCA513/ysj/dataset/openrca")
DEFAULT_SOURCE_BASE = Path("/home/dell2/RCA513/yyx/trace_summary/confidence_aware_rca_v2_drift_focus_2026-06-02_resend/source")


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    data_root: Path
    query_csv: Path
    record_csv: Path
    baseline_path: Path
    node_graph_path: Path
    trace_summary_dir: Path
    rules_path: Path = Path("knowledge/refutation_rules_v2.json")

    @classmethod
    def for_name(cls, name: str, base: Path = DEFAULT_SOURCE_BASE,
                 data_base: Path = DEFAULT_DATA_BASE) -> "DatasetConfig":
        registry = {
            "bank": cls(
                name="bank",
                data_root=data_base / "Bank",
                query_csv=data_base / "Bank" / "query.csv",
                record_csv=data_base / "Bank" / "record.csv",
                baseline_path=base / "refute" / "knowledge" / "baseline_distributions_all_metric_dates.json",
                node_graph_path=base / "refute" / "knowledge" / "node_container_graph_2021_03_10.json",
                trace_summary_dir=base / "trace_summaries" / "openrca_queries_full_batch18",
            ),
            "market_cloudbed_1": cls(
                name="market_cloudbed_1",
                data_root=data_base / "Market" / "cloudbed-1",
                query_csv=data_base / "Market" / "cloudbed-1" / "query.csv",
                record_csv=data_base / "Market" / "cloudbed-1" / "record.csv",
                baseline_path=base / "knowledge" / "market_cloudbed_1_baseline_distributions_all_metric_dates.json",
                node_graph_path=base / "knowledge" / "market_cloudbed_1_node_container_graph.json",
                trace_summary_dir=base / "trace_summaries" / "market_cloudbed_1_queries_full",
            ),
            "market_cloudbed_2": cls(
                name="market_cloudbed_2",
                data_root=data_base / "Market" / "cloudbed-2",
                query_csv=data_base / "Market" / "cloudbed-2" / "query.csv",
                record_csv=data_base / "Market" / "cloudbed-2" / "record.csv",
                baseline_path=base / "knowledge" / "market_cloudbed_2_baseline_distributions_all_metric_dates.json",
                node_graph_path=base / "knowledge" / "market_cloudbed_2_node_container_graph.json",
                trace_summary_dir=base / "trace_summaries" / "market_cloudbed_2_queries_full",
            ),
            "telecom": cls(
                name="telecom",
                data_root=data_base / "Telecom",
                query_csv=data_base / "Telecom" / "query.csv",
                record_csv=data_base / "Telecom" / "record.csv",
                baseline_path=base / "knowledge" / "telecom_baseline_distributions_all_metric_dates.json",
                node_graph_path=base / "knowledge" / "telecom_node_container_graph.json",
                trace_summary_dir=base / "trace_summaries" / "telecom_queries_full",
            ),
        }
        if name not in registry:
            raise KeyError(f"unknown dataset '{name}'; choose from {sorted(registry)}")
        return registry[name]


# ──────────────────────────── day cache ────────────────────────────
class DayCache:
    def __init__(self, paths: BankDataPaths):
        self.paths = paths
        self.metric: dict[str, pd.DataFrame] = {}
        self.log: dict[str, pd.DataFrame] = {}

    def metric_day(self, date_key: str) -> pd.DataFrame:
        if date_key not in self.metric:
            self.metric[date_key] = load_metric_day(self.paths, date_key)
        return self.metric[date_key]

    def log_day(self, date_key: str) -> pd.DataFrame:
        if date_key not in self.log:
            self.log[date_key] = load_log_day(self.paths, date_key)
        return self.log[date_key]


def filter_between(df: pd.DataFrame, start_ts: int, end_ts: int) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    return df[(df["timestamp"] >= start_ts) & (df["timestamp"] < end_ts)].copy()


def gather_window(cache: DayCache, window, modalities: set[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = []
    logs = []
    for date_key in window.date_keys:
        if "metric" in modalities:
            metrics.append(filter_between(cache.metric_day(date_key), window.start_ts, window.end_ts))
        if "log" in modalities:
            logs.append(filter_between(cache.log_day(date_key), window.start_ts, window.end_ts))
    metric_df = pd.concat(metrics, ignore_index=True) if metrics else pd.DataFrame(columns=["timestamp", "cmdb_id", "kpi_name", "value"])
    log_df = pd.concat(logs, ignore_index=True) if logs else pd.DataFrame(columns=["timestamp", "cmdb_id", "value"])
    return metric_df, log_df


def known_services(node_graph: dict) -> list[str]:
    services = sorted(str(s) for s in node_graph.get("containers", {}).keys())
    return services or []


# ──────────────────────────── knowledge building ────────────────────────────
def build_source_knowledge(cfg: DatasetConfig, modalities: set[str]) -> dict[str, Any]:
    """Build D32 knowledge from all cases of a single source dataset."""
    paths = BankDataPaths.from_root(str(cfg.data_root))
    cache = DayCache(paths)
    baseline = BaselineStore.load_json(str(cfg.baseline_path))
    case_rows = case_rows_from_openrca(str(cfg.query_csv), str(cfg.record_csv))

    config = KnowledgeBuildConfig(validate_rules_lodo=False)

    cases = []
    for row in case_rows:
        row_id = int(row["row_id"])
        window = parse_query_window(str(row["instruction"]))
        metric_df, log_df = gather_window(cache, window, modalities)
        trace_summary = load_trace_summary(str(cfg.trace_summary_dir), row_id) if "trace" in modalities else None
        modal_status = {
            "metric": "present" if "metric" in modalities and not metric_df.empty else (
                "empty_window" if "metric" in modalities else "disabled"
            ),
            "log": "present" if "log" in modalities and not log_df.empty else (
                "empty_window" if "log" in modalities else "disabled"
            ),
            "trace": (trace_summary or {}).get("trace_status", "unloaded") if "trace" in modalities else "disabled",
        }
        signature = build_case_signature(
            f"{cfg.name}_{row_id:03d}", metric_df, log_df, trace_summary, baseline, modal_status
        )
        cases.append({
            "case_id": f"{cfg.name}_{row_id:03d}",
            "row_id": row_id,
            "component": str(row["component"]),
            "reason": str(row["reason"]),
            "task_index": str(row.get("task_index", "")),
            "date_key": str(row.get("date_key", "")),
            "signature": signature,
            "features": flatten_signature(signature),
        })

    clusters = _cluster_cases(cases, config.cluster_threshold)
    mined_rules = _mine_rules_once(
        cases, config.min_rule_support, config.min_rule_confidence, config.max_rule_len
    )
    from refute_b_v2_d32.textbook import Textbook
    textbook = Textbook.build(cases)

    return {
        "version": 1,
        "design": "d32_layer1_knowledge_cross_dataset",
        "config": config.__dict__,
        "cases": [
            {
                "case_id": row["case_id"],
                "row_id": row["row_id"],
                "component": row["component"],
                "reason": row["reason"],
                "task_index": row["task_index"],
                "date_key": row["date_key"],
                "signature": row["signature"],
            }
            for row in cases
        ],
        "clusters": clusters,
        "mined_rules": mined_rules,
        "textbook": textbook.to_dict(),
        "metadata": {
            "n_cases": len(cases),
            "modalities": sorted(modalities),
            "source_dataset": cfg.name,
        },
    }


def merge_knowledge(parts: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge knowledge dicts from multiple source datasets."""
    if len(parts) == 1:
        return parts[0]

    all_cases = []
    all_clusters = []
    all_rules: dict[tuple[tuple[str, ...], str], dict[str, Any]] = {}

    for part in parts:
        all_cases.extend(part.get("cases", []))
        clusters = part.get("clusters", [])
        # renumber cluster ids to avoid collisions
        offset = len(all_clusters)
        for c in clusters:
            c["cluster_id"] = f"cluster_{offset:03d}"
            offset += 1
        all_clusters.extend(clusters)

        for rule in part.get("mined_rules", []):
            key = (
                tuple(sorted(rule.get("antecedent", []))),
                str(rule.get("target_reason", "")),
            )
            if key in all_rules:
                existing = all_rules[key]
                existing["support"] = max(existing.get("support", 0), rule.get("support", 0))
                existing["confidence"] = max(existing.get("confidence", 0.0), rule.get("confidence", 0.0))
                existing["lift"] = max(existing.get("lift", 0.0), rule.get("lift", 0.0))
            else:
                all_rules[key] = dict(rule)

    merged_rules = sorted(
        all_rules.values(),
        key=lambda r: (-r.get("confidence", 0.0), -r.get("lift", 0.0), -r.get("support", 0)),
    )[:500]

    total_cases = sum(p["metadata"].get("n_cases", 0) for p in parts)
    all_modalities = sorted(set(
        m for p in parts for m in p.get("metadata", {}).get("modalities", [])
    ))
    source_names = [p.get("metadata", {}).get("source_dataset", "?") for p in parts]

    return {
        "version": 1,
        "design": "d32_layer1_knowledge_cross_dataset_merged",
        "config": parts[0].get("config", {}),
        "cases": all_cases,
        "clusters": all_clusters,
        "mined_rules": merged_rules,
        "metadata": {
            "n_cases": total_cases,
            "modalities": all_modalities,
            "source_datasets": source_names,
        },
    }


# ──────────────────────────── prediction ────────────────────────────
def run_predictions(
    knowledge_data: dict[str, Any],
    target_cfg: DatasetConfig,
    modalities: set[str],
) -> list[dict[str, Any]]:
    """Run d32 predictions on all cases of the target dataset using transferred knowledge."""
    knowledge = D32Knowledge(knowledge_data)
    paths = BankDataPaths.from_root(str(target_cfg.data_root))
    cache = DayCache(paths)
    baseline = BaselineStore.load_json(str(target_cfg.baseline_path))
    node_graph = json.loads(Path(target_cfg.node_graph_path).read_text(encoding="utf-8"))
    rules = json.loads(Path(target_cfg.rules_path).read_text(encoding="utf-8"))
    services = known_services(node_graph)
    pipeline = D32RefutationPipeline(
        knowledge, rules, baseline, node_graph, services, config=D32PipelineConfig()
    )

    query_df = pd.read_csv(str(target_cfg.query_csv))
    results = []

    for idx in range(len(query_df)):
        qrow = query_df.iloc[idx]
        window = parse_query_window(str(qrow["instruction"]))
        metric_df, log_df = gather_window(cache, window, modalities)
        trace_summary = load_trace_summary(
            str(target_cfg.trace_summary_dir), idx
        ) if "trace" in modalities else None
        modal_status = {
            "metric": "present" if "metric" in modalities and not metric_df.empty else (
                "empty_window" if "metric" in modalities else "disabled"
            ),
            "log": "present" if "log" in modalities and not log_df.empty else (
                "empty_window" if "log" in modalities else "disabled"
            ),
            "trace": (trace_summary or {}).get("trace_status", "unloaded") if "trace" in modalities else "disabled",
        }
        result = pipeline.select(
            case_id=f"target_{idx:03d}",
            metric_df=metric_df,
            log_df=log_df,
            trace_summary=trace_summary,
            modal_status=modal_status,
            failure_count=window.failure_count,
            window_start_ts=window.start_ts,
        )
        results.append({
            "row_id": idx,
            "prediction": result.prediction,
            "debug": result.debug,
        })
        print(f"target case {idx + 1}/{len(query_df)} done", flush=True)

    return results


# ──────────────────────────── main ────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sources", nargs="+", required=True,
                        choices=["bank", "market_cloudbed_1", "market_cloudbed_2", "telecom"],
                        help="Source datasets for building transfer knowledge")
    parser.add_argument("--target", required=True,
                        choices=["bank", "market_cloudbed_1", "market_cloudbed_2", "telecom"],
                        help="Target dataset for evaluation")
    parser.add_argument("--modalities", default="metric,log,trace",
                        help="Comma-separated modalities (default: metric,log,trace)")
    parser.add_argument("--base", default=str(DEFAULT_SOURCE_BASE),
                        help="Project source base directory")
    parser.add_argument("--data-base", default=str(DEFAULT_DATA_BASE),
                        help="Raw data base directory")
    parser.add_argument("--out-dir", default="logs/cross_transfer",
                        help="Output directory for knowledge, predictions, eval results")
    parser.add_argument("--skip-build", action="store_true",
                        help="Skip knowledge building (use existing merged knowledge file)")
    parser.add_argument("--skip-predict", action="store_true",
                        help="Skip prediction (use existing predictions CSV)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base = Path(args.base)
    data_base = Path(args.data_base)
    modalities = {part.strip() for part in args.modalities.split(",") if part.strip()}
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    source_configs = [DatasetConfig.for_name(s, base, data_base) for s in args.sources]
    target_cfg = DatasetConfig.for_name(args.target, base, data_base)

    source_names = "+".join(args.sources)
    mod_name = args.modalities.replace(",", "_")
    knowledge_path = out_dir / f"cross_knowledge_{source_names}_to_{args.target}_{mod_name}.json"
    pred_path = out_dir / f"cross_predictions_{source_names}_to_{args.target}_{mod_name}.csv"

    # ── Phase 1: Build knowledge ──
    if not (args.skip_build and knowledge_path.exists()):
        print(f"=== Phase 1: building knowledge from {args.sources} ({modalities}) ===")
        parts = []
        for cfg in source_configs:
            print(f"  building knowledge for {cfg.name}...")
            part = build_source_knowledge(cfg, modalities)
            print(f"    {cfg.name}: {part['metadata']['n_cases']} cases, "
                  f"{len(part['clusters'])} clusters, {len(part['mined_rules'])} rules")
            parts.append(part)

        merged = merge_knowledge(parts)
        print(f"  merged: {merged['metadata']['n_cases']} total cases, "
              f"{len(merged['clusters'])} clusters, {len(merged['mined_rules'])} rules")
        knowledge_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  wrote {knowledge_path}")
    else:
        print(f"=== Phase 1: SKIPPED (knowledge exists at {knowledge_path}) ===")
        merged = json.loads(knowledge_path.read_text(encoding="utf-8"))

    # ── Phase 2: Predict on target ──
    if not (args.skip_predict and pred_path.exists()):
        print(f"\n=== Phase 2: predicting on {args.target} ({modalities}) ===")
        results = run_predictions(merged, target_cfg, modalities)
        pd.DataFrame([
            {"row_id": r["row_id"], "prediction": json.dumps(r["prediction"], ensure_ascii=False)}
            for r in results
        ]).to_csv(str(pred_path), index=False)
        print(f"  wrote {pred_path}")
    else:
        print(f"\n=== Phase 2: SKIPPED (predictions exist at {pred_path}) ===")

    # ── Phase 3: Evaluate ──
    print(f"\n=== Phase 3: evaluating ===")
    from eval.field_hit_diagnostics import build_component_expansion
    import eval.field_hit_diagnostics as fhd
    fhd.COMPONENT_EXPANSION = build_component_expansion(
        json.loads(Path(target_cfg.node_graph_path).read_text(encoding="utf-8"))
    )

    from eval.openrca_official_case_eval import evaluate
    official_path = out_dir / f"cross_eval_{source_names}_to_{args.target}_{mod_name}_official.json"
    report = evaluate(str(pred_path), str(target_cfg.query_csv))
    official_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  strict_rate = {report['strict_rate']:.4f}  partial_rate = {report['partial_rate']:.4f}")
    print(f"  wrote {official_path}")

    # Field diagnostics
    from eval.field_hit_diagnostics import summarize, load_rows
    field_path = out_dir / f"cross_eval_{source_names}_to_{args.target}_{mod_name}_field_diag.json"
    field_report = summarize(load_rows(str(pred_path), str(target_cfg.query_csv)))
    field_path.write_text(json.dumps(field_report, ensure_ascii=False, indent=2), encoding="utf-8")
    for field in ("time", "component", "reason"):
        info = field_report["field_item"].get(field, {})
        print(f"  {field}_hit_rate = {info.get('hit_rate', 0.0):.4f}")
    print(f"  wrote {field_path}")

    # Reason-conditional
    from eval.reason_conditional_tolerance_eval import score_assignment
    rc_path = out_dir / f"cross_eval_{source_names}_to_{args.target}_{mod_name}_reason_cond.json"

    pred_rows_rc = []
    with open(pred_path, newline="", encoding="utf-8") as f:
        import csv
        pred_rows_rc = list(csv.DictReader(f))
    query_rows_rc = []
    with open(target_cfg.query_csv, newline="", encoding="utf-8") as f:
        import csv
        query_rows_rc = list(csv.DictReader(f))
    record_rows_rc = []
    with open(target_cfg.record_csv, newline="", encoding="utf-8") as f:
        import csv
        record_rows_rc = list(csv.DictReader(f))

    from eval.field_hit_diagnostics import parse_prediction as pp, parse_truth as pt
    from eval.reason_conditional_tolerance_eval import TOLERANCE_BY_REASON_MINUTES, reason_group

    rc_rows = []
    by_time = {row["datetime"].strip(): row for row in record_rows_rc}
    strict_c = 0
    partial_c = 0
    for idx, (pr, qr, rr) in enumerate(zip(pred_rows_rc, query_rows_rc, record_rows_rc)):
        preds = pp(pr["prediction"])
        truths = pt(qr["scoring_points"])
        truth_reasons = []
        for truth in truths:
            if truth.get("reason"):
                truth_reasons.append(truth["reason"])
            elif truth.get("time") and truth["time"] in by_time:
                truth_reasons.append(by_time[truth["time"]]["reason"].strip())
            else:
                truth_reasons.append(rr["reason"].strip())
        official_hits, total, official_score = score_assignment(
            preds, truths, ["__official_1min__"] * len(truths), fixed_tolerance_minutes=1,
        )
        rc_hits, _, rc_score = score_assignment(preds, truths, truth_reasons)
        strict_c += int(official_score == 1.0)
        partial_c += int(rc_score == 1.0)

    n = len(pred_rows_rc)
    rc_report = {
        "n": n,
        "official_strict_rate": strict_c / n if n else 0.0,
        "reason_conditional_strict_rate": partial_c / n if n else 0.0,
    }
    rc_path.write_text(json.dumps(rc_report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  reason_conditional_strict_rate = {rc_report['reason_conditional_strict_rate']:.4f}")
    print(f"  wrote {rc_path}")

    print(f"\n=== Cross-dataset transfer complete ===")
    print(f"  Source: {args.sources}  Target: {args.target}  Modalities: {args.modalities}")
    print(f"  Official strict: {report['strict_rate']:.4f}  Partial: {report['partial_rate']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
