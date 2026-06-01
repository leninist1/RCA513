#!/usr/bin/env python3
"""PRISM v3 runner.

This package-local runner merges:
- yyx PRISM v2 feature switches,
- ysj Bank reason canonicalization,
- syh runtime/top-k counterfactual controls.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .config import EvalResult, QueryCase, SYSTEM_PATHS
from .data.loader import OpenRCALoader
from .evaluation.aggregator import EvalAggregator
from .evaluation.scorer import evaluate_prediction
from .prism import PRISMConfig, PRISMPipeline


_SHARED_TELEMETRY = None


def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=True, default=str))
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def parse_epoch_window(time_window: Tuple[str, str]) -> Tuple[Optional[float], Optional[float]]:
    try:
        start = datetime.strptime(time_window[0], "%Y-%m-%d %H:%M:%S").timestamp()
        end = datetime.strptime(time_window[1], "%Y-%m-%d %H:%M:%S").timestamp()
        if end <= start:
            return start, None
        return start, end
    except Exception:
        return None, None


def infer_telemetry_anchor_time(telemetry: Any, query: QueryCase, fallback_time: float) -> Tuple[float, Dict[str, Any]]:
    """Estimate a no-label fault anchor from public query window and telemetry.

    The original pipeline relies heavily on a baseline/fault split.  We keep
    that mechanism, but replace record.csv-derived inject_time with an
    unsupervised onset estimate from metrics inside the query window.
    """
    start, end = parse_epoch_window(query.time_window)
    if start is None or end is None:
        return fallback_time, {"source": "fallback_query_window_start", "reason": "bad_query_window"}

    metrics = getattr(telemetry, "metrics", None)
    if metrics is None or getattr(metrics, "empty", True):
        return fallback_time, {"source": "fallback_query_window_start", "reason": "no_metrics"}
    required = {"timestamp", "entity", "metric_name", "value"}
    if not required.issubset(set(metrics.columns)):
        return fallback_time, {
            "source": "fallback_query_window_start",
            "reason": "missing_metric_columns",
        }

    baseline_window = 600.0
    baseline_start = max(0.0, float(start) - baseline_window)
    frame = metrics.loc[
        (metrics["timestamp"] >= baseline_start) & (metrics["timestamp"] <= end),
        ["timestamp", "entity", "metric_name", "value"],
    ].copy()
    if frame.empty:
        return fallback_time, {"source": "fallback_query_window_start", "reason": "empty_metric_window"}
    frame["value_num"] = pd.to_numeric(frame["value"], errors="coerce")
    frame = frame.dropna(subset=["timestamp", "entity", "metric_name", "value_num"])
    if frame.empty:
        return fallback_time, {"source": "fallback_query_window_start", "reason": "non_numeric_metrics"}

    baseline = frame[(frame["timestamp"] >= baseline_start) & (frame["timestamp"] < start)]
    candidate = frame[(frame["timestamp"] >= start) & (frame["timestamp"] <= end)]
    if baseline.empty or candidate.empty:
        return fallback_time, {
            "source": "fallback_query_window_start",
            "reason": "empty_baseline_or_candidate",
            "baseline_rows": int(len(baseline)),
            "candidate_rows": int(len(candidate)),
        }

    key_cols = ["entity", "metric_name"]
    stats = (
        baseline.groupby(key_cols)["value_num"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    stats = stats[stats["count"] >= 2]
    merged = candidate.merge(stats, on=key_cols, how="inner")
    if merged.empty:
        return fallback_time, {
            "source": "fallback_query_window_start",
            "reason": "no_candidate_baseline_overlap",
            "baseline_rows": int(len(baseline)),
            "candidate_rows": int(len(candidate)),
        }

    denom = merged["std"].fillna(0.0).abs()
    mean_abs = merged["mean"].fillna(0.0).abs()
    denom = denom.where(denom > 1e-6, mean_abs * 0.02 + 1e-3)
    merged["abs_z"] = ((merged["value_num"] - merged["mean"]).abs() / denom).clip(upper=50.0)
    merged["metric_weight"] = merged["metric_name"].map(anchor_metric_weight)
    merged = merged[merged["metric_weight"] > 0.0]
    if merged.empty:
        return fallback_time, {"source": "fallback_query_window_start", "reason": "no_relevant_metrics"}
    merged["weighted_z"] = merged["abs_z"] * merged["metric_weight"]
    merged["over3"] = (merged["weighted_z"] >= 3.0).astype(float)
    merged["energy"] = merged["weighted_z"].clip(upper=12.0)

    by_ts = (
        merged.groupby("timestamp")
        .agg(max_z=("weighted_z", "max"), over3=("over3", "sum"), energy=("energy", "sum"))
        .reset_index()
    )
    if by_ts.empty:
        return fallback_time, {"source": "fallback_query_window_start", "reason": "empty_timestamp_scores"}
    by_ts["score"] = by_ts["max_z"] + 0.15 * by_ts["over3"].clip(upper=25.0) + 0.015 * by_ts["energy"].clip(upper=250.0)
    by_ts = by_ts.sort_values("timestamp")
    max_idx = by_ts["score"].idxmax()
    max_score = float(by_ts.loc[max_idx, "score"])
    if not math.isfinite(max_score) or max_score < 2.0:
        return fallback_time, {
            "source": "fallback_query_window_start",
            "reason": "weak_metric_anomaly",
            "max_score": round(max_score, 6) if math.isfinite(max_score) else None,
        }
    threshold = max(3.0, 0.55 * max_score)
    strong = by_ts[by_ts["score"] >= threshold]
    selected = strong.iloc[0] if not strong.empty else by_ts.loc[max_idx]
    anchor = float(selected["timestamp"])
    return anchor, {
        "source": "telemetry_metric_onset",
        "fallback_source": "query_window_start",
        "baseline_start": datetime.fromtimestamp(baseline_start).strftime("%Y-%m-%d %H:%M:%S"),
        "query_start": query.time_window[0],
        "query_end": query.time_window[1],
        "anchor_time": datetime.fromtimestamp(anchor).strftime("%Y-%m-%d %H:%M:%S"),
        "selected_score": round(float(selected["score"]), 6),
        "max_score": round(max_score, 6),
        "threshold": round(threshold, 6),
        "baseline_rows": int(len(baseline)),
        "candidate_rows": int(len(candidate)),
        "scored_timestamps": int(len(by_ts)),
    }


def anchor_metric_weight(metric_name: Any) -> float:
    name = str(metric_name or "").lower()
    if not name:
        return 0.0
    exclude = (
        "hostname",
        "uptime",
        "filesystem",
        "fsavailablespace",
        "fsusedspace",
        "fsinode",
        "filesize",
        "dirsize",
        "zabbix",
        "open_tables",
        "questions",
        "bytes_sent",
        "bytes_received",
        "handler_read",
        "handler_write",
    )
    if any(item in name for item in exclude):
        return 0.0
    high_signal = (
        "cpu",
        "memory",
        "mem",
        "heap",
        "jvm",
        "latency",
        "packet",
        "error",
        "timeout",
        "mrt",
        "sr",
        "rr",
        "thread",
        "lock",
        "dsk",
        "disk",
        "innodb",
        "mysql",
    )
    return 1.0 if any(item in name for item in high_signal) else 0.35


def percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    vals = sorted(float(v) for v in values)
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * (pct / 100.0)
    lo = int(pos)
    hi = min(lo + 1, len(vals) - 1)
    frac = pos - lo
    return vals[lo] * (1.0 - frac) + vals[hi] * frac


def summarize_runtime(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    keys = [
        "load_telemetry_sec",
        "object_induction_sec",
        "split_temporal_sec",
        "signal_graph_sec",
        "noise_lab_prior_sec",
        "attribution_sec",
        "cf_profile_orig_degradation_sec",
        "cf_profile_apply_counterfactual_sec",
        "cf_profile_cf_degradation_sec",
        "state_loop_sec",
        "final_cf_sec",
        "total_sec",
        "memory_start_rss_mb",
        "memory_end_rss_mb",
        "memory_rss_delta_mb",
        "memory_peak_worker_mb",
    ]
    summary: Dict[str, Any] = {}
    for key in keys:
        vals = []
        for row in results:
            dbg = row.get("runtime_debug", {}) or {}
            if key in dbg:
                try:
                    vals.append(float(dbg[key]))
                except Exception:
                    pass
        if vals:
            summary[key] = {
                "mean": round(sum(vals) / len(vals), 6),
                "p50": round(percentile(vals, 50), 6),
                "p95": round(percentile(vals, 95), 6),
                "total": round(sum(vals), 6),
            }
    return summary


def reason_diagnostics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    confusion: Dict[str, Dict[str, int]] = {}
    gt_counts: Dict[str, int] = {}
    pred_counts: Dict[str, int] = {}
    rule_hits: Dict[str, int] = {}
    wrong_examples: List[Dict[str, Any]] = []
    total = 0
    correct = 0
    for item in results:
        gt = str((item.get("ground_truth") or {}).get("reason", "") or "")
        pred_vals = (item.get("prediction") or {}).get("reason") or []
        pred = pred_vals[0] if isinstance(pred_vals, list) and pred_vals else pred_vals
        pred = str(pred or "")
        if not gt and not pred:
            continue
        total += 1
        if gt:
            gt_counts[gt] = gt_counts.get(gt, 0) + 1
        if pred:
            pred_counts[pred] = pred_counts.get(pred, 0) + 1
        confusion.setdefault(gt, {})
        confusion[gt][pred] = confusion[gt].get(pred, 0) + 1
        reason_debug = (item.get("debug") or {}).get("reason_debug") or {}
        rule = str(reason_debug.get("matched_rule", "") or "")
        if rule:
            rule_hits[rule] = rule_hits.get(rule, 0) + 1
        if gt and pred and gt.strip().lower() == pred.strip().lower():
            correct += 1
        elif len(wrong_examples) < 30:
            wrong_examples.append(
                {
                    "query_id": item.get("query_id", ""),
                    "task_index": item.get("task_index", ""),
                    "gt_reason": gt,
                    "pred_reason": pred,
                    "matched_rule": rule,
                    "evidence": reason_debug.get("evidence", [])[:5],
                }
            )
    return {
        "total_with_reason": total,
        "exact_correct": correct,
        "exact_rate": round(correct / total * 100, 2) if total else 0.0,
        "gt_counts": dict(sorted(gt_counts.items())),
        "pred_counts": dict(sorted(pred_counts.items())),
        "confusion_matrix": {gt: dict(sorted(preds.items())) for gt, preds in sorted(confusion.items())},
        "rule_hits": dict(sorted(rule_hits.items())),
        "wrong_examples": wrong_examples,
    }


def compact_belief(prism_result: Dict[str, Any], k: int = 20) -> List[Dict[str, Any]]:
    state = prism_result.get("state")
    entities = list(getattr(state, "entities", []) or [])
    belief = prism_result.get("belief")
    if belief is None or not entities:
        return []
    pairs = []
    try:
        for idx, entity in enumerate(entities):
            pairs.append((entity, float(belief[idx])))
    except Exception:
        return []
    pairs.sort(key=lambda item: item[1], reverse=True)
    return [{"entity": entity, "prob": round(prob, 6)} for entity, prob in pairs[:k]]


def compact_graph_edges(prism_result: Dict[str, Any], k: int = 80) -> Dict[str, Any]:
    state = prism_result.get("state")
    entities = list(getattr(state, "entities", []) or [])
    graph = prism_result.get("graph_matrix")
    if graph is None or not entities:
        return {"edge_count": 0, "top_edges": []}
    edges = []
    try:
        n = min(len(entities), len(graph))
        for i in range(n):
            row = graph[i]
            for j in range(min(len(entities), len(row))):
                weight = float(row[j])
                if weight > 0:
                    edges.append((entities[i], entities[j], weight))
    except Exception:
        return {"edge_count": 0, "top_edges": []}
    edges.sort(key=lambda item: item[2], reverse=True)
    return {
        "edge_count": len(edges),
        "top_edges": [
            {"source": src, "target": dst, "weight": round(weight, 6)}
            for src, dst, weight in edges[:k]
        ],
    }


def make_prism_config(args: argparse.Namespace) -> PRISMConfig:
    cfg = PRISMConfig(
        noise_lab_enabled=bool(args.prism_noise_lab),
        noise_lab_scores_csv=str(args.prism_noise_lab_scores or ""),
        noise_lab_strategy=str(args.prism_noise_lab_strategy or "ltr_full"),
        noise_lab_prior_weight=float(args.prism_noise_prior_weight),
        noise_lab_final_weight=float(args.prism_noise_final_weight),
        noise_native_agent_enabled=not bool(
            getattr(args, "prism_disable_noise_native_agent", False)
        ),
        noise_native_max_events=int(getattr(args, "prism_noise_native_max_events", 10)),
        noise_native_max_rounds=int(getattr(args, "prism_noise_native_max_rounds", 2)),
        noise_native_w_noise=float(getattr(args, "prism_noise_native_w_noise", 1.0)),
        noise_native_w_metric=float(getattr(args, "prism_noise_native_w_metric", 0.85)),
        noise_native_w_log=float(getattr(args, "prism_noise_native_w_log", 0.65)),
        noise_native_w_trace=float(getattr(args, "prism_noise_native_w_trace", 0.45)),
        noise_native_w_counterfactual=float(
            getattr(args, "prism_noise_native_w_counterfactual", 0.55)
        ),
        noise_native_w_pairwise=float(
            getattr(args, "prism_noise_native_w_pairwise", 0.22)
        ),
        noise_native_w_symptom=float(getattr(args, "prism_noise_native_w_symptom", 0.80)),
        noise_native_w_broad=float(getattr(args, "prism_noise_native_w_broad", 0.55)),
        noise_native_w_structural=float(
            getattr(args, "prism_noise_native_w_structural", 0.25)
        ),
        noise_native_cmi_enabled=not bool(
            getattr(args, "prism_disable_noise_native_cmi", False)
        ),
        noise_native_cmi_max_conditioners=int(
            getattr(args, "prism_noise_native_cmi_max_conditioners", 6)
        ),
        noise_native_cmi_max_effect_scope=int(
            getattr(args, "prism_noise_native_cmi_max_effect_scope", 10)
        ),
        cf_profile_top_k=int(args.prism_cf_profile_top_k),
        cf_profiles_max_calls_per_query=int(args.prism_cf_profiles_max_calls),
        cf_degradation_entity_top_k=int(args.prism_cf_degradation_entity_top_k),
        final_cf_top_k=int(args.prism_final_cf_top_k),
        final_scope_prism_top_k=int(args.prism_final_scope_prism_top_k),
        final_scope_noise_lab_top_k=int(args.prism_final_scope_noise_top_k),
        final_scope_min_candidates=int(args.prism_final_scope_min_candidates),
        final_scope_max_candidates=int(args.prism_final_scope_max_candidates),
        final_cf_noise_lab_skip_gap=float(args.prism_final_cf_skip_gap),
        final_cf_only_on_conflict=bool(args.prism_final_cf_only_on_conflict),
        object_induction_enabled=not bool(args.prism_disable_object_induction),
        object_induction_cache_enabled=not bool(args.prism_disable_object_induction_cache),
        object_induction_cache_max_entries=int(args.prism_object_cache_max_entries),
        memory_debug_enabled=not bool(args.prism_disable_memory_debug),
        use_learned_embeddings=bool(args.v2_all or args.v2_learned or args.v2_learned_embeddings),
        use_learned_likelihood=bool(args.v2_all or args.v2_learned or args.v2_learned_likelihood),
        use_learned_classifier=bool(args.v2_all or args.v2_learned or args.v2_learned_classifier),
        use_adaptive_anomaly=bool(args.v2_all or args.v2_learned or args.v2_adaptive_anomaly),
        learned_embedding_path=str(args.v2_learned_embedding_path or ""),
        learned_likelihood_path=str(args.v2_learned_likelihood_path or ""),
        learned_classifier_path=str(args.v2_learned_classifier_path or ""),
        use_hierarchical_prior=bool(args.v2_all or args.v2_hier_prior),
        use_system_prototype=bool(args.v2_all or args.v2_hier_prior or args.v2_system_proto),
        use_entity_profiles=bool(args.v2_all or args.v2_hier_prior or args.v2_entity_profiles),
        entity_profile_path=str(args.v2_entity_profile_path or ""),
        hierarchical_prior_weight=float(args.v2_hier_prior_weight),
        use_active_inference=bool(args.v2_all or args.v2_active_inference),
        ai_risk_aversion=float(args.v2_ai_risk),
        ai_efe_stop_threshold=float(args.v2_ai_efe_stop),
        use_mcts=bool(args.v2_all or args.v2_mcts),
        use_hierarchical_mcts=bool(args.v2_all or args.v2_hier_mcts),
        mcts_n_simulations=int(args.v2_mcts_sims),
        mcts_c_uct=float(args.v2_mcts_uct),
        mcts_rollout_depth=int(args.v2_mcts_rollout),
        use_active_perception=bool(args.v2_all or args.v2_active_perception),
        use_metric_rescan=bool(args.v2_all or args.v2_active_perception or args.v2_metric_rescan),
        use_log_hypothesis_search=bool(args.v2_all or args.v2_active_perception or args.v2_log_search),
        use_trace_subgraph=bool(args.v2_all or args.v2_active_perception or args.v2_trace_subgraph),
        use_hypothesis_crossval=bool(args.v2_all or args.v2_active_perception or args.v2_crossval),
    )
    if args.prism_fast:
        cfg.t_max = 1
        cfg.n_cf_max = 0
        cfg.unexplained_trigger = 2.0
        cfg.probe_edge_budget = 0
        cfg.cf_profiles_enabled = False
        cfg.final_counterfactual_enabled = False
    elif args.prism_mid:
        cfg.t_max = 2
        cfg.n_cf_max = 2
        cfg.unexplained_trigger = 0.35
        cfg.probe_edge_budget = 2
        cfg.cf_profiles_enabled = True
        cfg.final_counterfactual_enabled = False
        cfg.pool_cf_eval_size = int(args.prism_cf_profile_top_k)
    return cfg


def process_query_worker(payload: Dict[str, Any]) -> Dict[str, Any]:
    telemetry = _SHARED_TELEMETRY
    result = {
        "query_id": f"{payload['system_name']}_{payload['task_index']}_{payload['query_index']}",
        "system": payload["system_name"],
        "task_index": payload["task_index"],
        "correct": False,
        "partial": False,
        "error": None,
        "iterations": 0,
        "strategies_used": [],
    }
    try:
        query = QueryCase(
            task_index=payload["task_index"],
            system=payload["system_name"],
            sub_system=payload["sub_system"],
            instruction=payload["instruction"],
            time_window=(payload["time_window_start"], payload["time_window_end"]),
            scoring_points=[],
            inject_time=payload["inference_time"],
            telemetry_date=payload["date_str"],
            ground_truth=None,
        )
        query.query_index = payload["query_index"]
        inference_time, anchor_debug = infer_telemetry_anchor_time(
            telemetry=telemetry,
            query=query,
            fallback_time=float(payload["inference_time"]),
        )
        query.inject_time = inference_time
        cfg = make_prism_config(argparse.Namespace(**payload["config_args"]))
        prism = PRISMPipeline(payload["system_name"], config=cfg)
        prism_result = prism.run(
            telemetry=telemetry,
            query=query,
            inject_time=inference_time,
        )
        prediction = prism_result["prediction"]

        loader = OpenRCALoader(payload["system_name"])
        eval_queries = loader.load_queries(payload["sub_system"])
        eval_query = eval_queries[int(payload["query_index"])]
        eval_query.query_index = payload["query_index"]
        gt_records, eval_inject_time = loader.match_query_to_records(eval_query)
        if not gt_records or eval_inject_time is None:
            raise RuntimeError("evaluation labels unavailable after inference")
        eval_query.ground_truth = gt_records[0]
        eval_query.inject_time = eval_inject_time
        eval_query.telemetry_date = payload["date_str"]

        eval_result = evaluate_prediction(prediction, eval_query)
        result.update(
            {
                "correct": eval_result.correct,
                "partial": eval_result.partial,
                "field_scores": dict(eval_result.field_scores),
                "official_score": float(eval_result.official_score),
                "official_passing": list(eval_result.official_passing),
                "official_failing": list(eval_result.official_failing),
                "iterations": len(prism_result.get("trace", [])),
                "strategies_used": [item["action"] for item in prism_result.get("trace", [])],
                "prediction": prediction,
                "top_candidates": prism_result.get("final_candidates", []),
                "stop_reason": prism_result.get("stop_reason", ""),
                "emotion_trajectory": prism_result.get("emotion_trajectory", []),
                "hypothesis_summary": {
                    "initial_candidates": prism_result.get("initial_candidates", []),
                    "final_candidates": prism_result.get("final_candidates", []),
                },
                "prism_trace": prism_result.get("trace", []),
                "debug": prism_result.get("debug", {}),
                "runtime_debug": dict(prism_result.get("runtime_debug", {}) or {}),
                "prism_config_runtime": prism_result.get("prism_config_runtime", {}),
                "belief_top20": compact_belief(prism_result, k=20),
                "graph_summary": compact_graph_edges(prism_result, k=80),
            }
        )
        result["runtime_debug"]["load_telemetry_sec"] = round(float(payload.get("load_telemetry_sec", 0.0)), 6)
        result["runtime_debug"]["inference_anchor"] = anchor_debug
        result["runtime_debug"]["leakage_guard"] = {
            "inference_query_has_ground_truth": bool(query.ground_truth),
            "inference_query_scoring_points": len(query.scoring_points),
            "inference_time_source": anchor_debug.get(
                "source", payload.get("inference_time_source", "")
            ),
            "evaluation_labels_loaded_after_inference": True,
        }
        if eval_query.ground_truth:
            result["ground_truth"] = {
                "component": eval_query.ground_truth.component,
                "reason": eval_query.ground_truth.reason,
                "datetime": eval_query.ground_truth.datetime_str,
            }
    except Exception as exc:
        import traceback

        result["error"] = str(exc)
        result["traceback"] = traceback.format_exc()
    return result


def config_dict(args: argparse.Namespace) -> Dict[str, Any]:
    keys = [
        "option",
        "systems",
        "max_queries",
        "date",
        "workers",
        "prism_noise_lab",
        "prism_noise_lab_scores",
        "prism_noise_lab_strategy",
        "prism_noise_prior_weight",
        "prism_noise_final_weight",
        "prism_disable_noise_native_agent",
        "prism_noise_native_max_events",
        "prism_noise_native_max_rounds",
        "prism_noise_native_w_noise",
        "prism_noise_native_w_metric",
        "prism_noise_native_w_log",
        "prism_noise_native_w_trace",
        "prism_noise_native_w_counterfactual",
        "prism_noise_native_w_pairwise",
        "prism_noise_native_w_symptom",
        "prism_noise_native_w_broad",
        "prism_noise_native_w_structural",
        "prism_disable_noise_native_cmi",
        "prism_noise_native_cmi_max_conditioners",
        "prism_noise_native_cmi_max_effect_scope",
        "prism_fast",
        "prism_mid",
        "prism_cf_profile_top_k",
        "prism_cf_profiles_max_calls",
        "prism_cf_degradation_entity_top_k",
        "prism_final_cf_top_k",
        "prism_final_scope_prism_top_k",
        "prism_final_scope_noise_top_k",
        "prism_final_scope_min_candidates",
        "prism_final_scope_max_candidates",
        "prism_final_cf_skip_gap",
        "prism_final_cf_only_on_conflict",
        "prism_disable_object_induction",
        "prism_disable_object_induction_cache",
        "prism_object_cache_max_entries",
        "prism_disable_memory_debug",
        "v2_all",
        "v2_learned",
        "v2_hier_prior",
        "v2_active_inference",
        "v2_mcts",
        "v2_active_perception",
    ]
    return {key: getattr(args, key) for key in keys}


def add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--option", choices=["PRISM"], default="PRISM")
    parser.add_argument("--systems", nargs="+", default=["Bank"])
    parser.add_argument("--output", default="results")
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--date", type=str, default=None)
    parser.add_argument("--prism-noise-lab", action="store_true")
    parser.add_argument("--prism-noise-lab-scores", default="")
    parser.add_argument("--prism-noise-lab-strategy", default="ltr_full")
    parser.add_argument("--prism-noise-prior-weight", type=float, default=0.30)
    parser.add_argument("--prism-noise-final-weight", type=float, default=0.35)
    parser.add_argument("--prism-disable-noise-native-agent", action="store_true")
    parser.add_argument("--prism-noise-native-max-events", type=int, default=10)
    parser.add_argument("--prism-noise-native-max-rounds", type=int, default=2)
    parser.add_argument("--prism-noise-native-w-noise", type=float, default=1.00)
    parser.add_argument("--prism-noise-native-w-metric", type=float, default=0.85)
    parser.add_argument("--prism-noise-native-w-log", type=float, default=0.65)
    parser.add_argument("--prism-noise-native-w-trace", type=float, default=0.45)
    parser.add_argument("--prism-noise-native-w-counterfactual", type=float, default=0.55)
    parser.add_argument("--prism-noise-native-w-pairwise", type=float, default=0.22)
    parser.add_argument("--prism-noise-native-w-symptom", type=float, default=0.80)
    parser.add_argument("--prism-noise-native-w-broad", type=float, default=0.55)
    parser.add_argument("--prism-noise-native-w-structural", type=float, default=0.25)
    parser.add_argument("--prism-disable-noise-native-cmi", action="store_true")
    parser.add_argument("--prism-noise-native-cmi-max-conditioners", type=int, default=6)
    parser.add_argument("--prism-noise-native-cmi-max-effect-scope", type=int, default=10)
    parser.add_argument("--prism-fast", action="store_true")
    parser.add_argument("--prism-mid", action="store_true")
    parser.add_argument("--prism-cf-profile-top-k", type=int, default=2)
    parser.add_argument("--prism-cf-profiles-max-calls", type=int, default=1)
    parser.add_argument("--prism-cf-degradation-entity-top-k", type=int, default=8)
    parser.add_argument("--prism-final-cf-top-k", type=int, default=5)
    parser.add_argument("--prism-final-scope-prism-top-k", type=int, default=5)
    parser.add_argument("--prism-final-scope-noise-top-k", type=int, default=5)
    parser.add_argument("--prism-final-scope-min-candidates", type=int, default=6)
    parser.add_argument("--prism-final-scope-max-candidates", type=int, default=10)
    parser.add_argument("--prism-final-cf-skip-gap", type=float, default=0.25)
    parser.add_argument("--prism-final-cf-only-on-conflict", action="store_true")
    parser.add_argument("--prism-disable-object-induction", action="store_true")
    parser.add_argument("--prism-disable-object-induction-cache", action="store_true")
    parser.add_argument("--prism-object-cache-max-entries", type=int, default=8)
    parser.add_argument("--prism-disable-memory-debug", action="store_true")
    parser.add_argument("--v2-learned", action="store_true")
    parser.add_argument("--v2-learned-embeddings", action="store_true")
    parser.add_argument("--v2-learned-likelihood", action="store_true")
    parser.add_argument("--v2-learned-classifier", action="store_true")
    parser.add_argument("--v2-adaptive-anomaly", action="store_true")
    parser.add_argument("--v2-learned-embedding-path", default="")
    parser.add_argument("--v2-learned-likelihood-path", default="")
    parser.add_argument("--v2-learned-classifier-path", default="")
    parser.add_argument("--v2-hier-prior", action="store_true")
    parser.add_argument("--v2-system-proto", action="store_true")
    parser.add_argument("--v2-entity-profiles", action="store_true")
    parser.add_argument("--v2-entity-profile-path", default="")
    parser.add_argument("--v2-hier-prior-weight", type=float, default=0.35)
    parser.add_argument("--v2-active-inference", action="store_true")
    parser.add_argument("--v2-ai-risk", type=float, default=0.3)
    parser.add_argument("--v2-ai-efe-stop", type=float, default=0.015)
    parser.add_argument("--v2-mcts", action="store_true")
    parser.add_argument("--v2-hier-mcts", action="store_true")
    parser.add_argument("--v2-mcts-sims", type=int, default=50)
    parser.add_argument("--v2-mcts-uct", type=float, default=1.4)
    parser.add_argument("--v2-mcts-rollout", type=int, default=3)
    parser.add_argument("--v2-active-perception", action="store_true")
    parser.add_argument("--v2-metric-rescan", action="store_true")
    parser.add_argument("--v2-log-search", action="store_true")
    parser.add_argument("--v2-trace-subgraph", action="store_true")
    parser.add_argument("--v2-crossval", action="store_true")
    parser.add_argument("--v2-all", action="store_true")


def main() -> None:
    parser = argparse.ArgumentParser(description="PRISM v3 OpenRCA runner")
    add_args(parser)
    args = parser.parse_args()

    results_dir = Path(args.output)
    results_dir.mkdir(parents=True, exist_ok=True)
    aggregator = EvalAggregator()
    all_results: List[Dict[str, Any]] = []
    config_args = vars(args).copy()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_pid{os.getpid()}"
    query_checkpoint_file = results_dir / f"query_checkpoint_{args.option}_{run_id}.jsonl"
    append_jsonl(
        query_checkpoint_file,
        {
            "record_type": "run_start",
            "run_id": run_id,
            "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "config": config_dict(args),
        },
    )
    print(f"Query checkpoint: {query_checkpoint_file}", flush=True)

    for system_name in args.systems:
        if system_name not in SYSTEM_PATHS:
            print(f"Unknown system: {system_name}, skipping", flush=True)
            continue
        print(f"\n{'=' * 60}\nSYSTEM: {system_name} | Option: {args.option}\n{'=' * 60}", flush=True)
        loader = OpenRCALoader(system_name)
        for sub in SYSTEM_PATHS[system_name]["sub_systems"]:
            queries = loader.load_queries(sub)
            if args.max_queries:
                queries = queries[: args.max_queries]
            date_groups: Dict[str, List[QueryCase]] = {}
            skipped = 0
            for query_index, query in enumerate(queries):
                query.query_index = query_index
                inference_time = loader.infer_query_anchor_time(query)
                if inference_time is None:
                    skipped += 1
                    continue
                query.ground_truth = None
                query.scoring_points = []
                query.inject_time = inference_time
                date_str = loader.resolve_telemetry_date(query)
                query.telemetry_date = date_str
                if date_str:
                    date_groups.setdefault(date_str, []).append(query)
                else:
                    skipped += 1
            print(
                f"  Sub-system: {sub or 'default'} "
                f"({sum(len(v) for v in date_groups.values())} valid in {len(date_groups)} dates, {skipped} skipped)",
                flush=True,
            )
            total_processed = 0
            for date_str, date_queries in sorted(date_groups.items()):
                if args.date and date_str != args.date:
                    continue
                print(f"    Date {date_str}: loading telemetry...", end=" ", flush=True)
                t0 = time.time()
                telemetry = loader.load_telemetry(date_str, sub)
                load_telemetry_sec = time.time() - t0
                print(f"({load_telemetry_sec:.1f}s, {len(date_queries)} queries)", flush=True)
                global _SHARED_TELEMETRY
                _SHARED_TELEMETRY = telemetry
                worker_payloads = []
                for idx, query in enumerate(date_queries):
                    worker_payloads.append(
                        {
                            "system_name": system_name,
                            "sub_system": sub,
                            "date_str": date_str,
                            "inference_time": query.inject_time,
                            "inference_time_source": "query_window_start_fallback",
                            "task_index": query.task_index,
                            "instruction": query.instruction,
                            "time_window_start": query.time_window[0],
                            "time_window_end": query.time_window[1],
                            "query_index": getattr(query, "query_index", -1),
                            "config_args": config_args,
                            "load_telemetry_sec": load_telemetry_sec if idx == 0 else 0.0,
                        }
                    )
                n_workers = max(1, min(args.workers, len(worker_payloads)))
                iterator = map(process_query_worker, worker_payloads)
                if n_workers > 1:
                    pool = Pool(processes=n_workers)
                    iterator = pool.imap_unordered(process_query_worker, worker_payloads)
                else:
                    pool = None
                try:
                    for row in iterator:
                        total_processed += 1
                        all_results.append(row)
                        if not row.get("error"):
                            aggregator.add(
                                EvalResult(
                                    correct=bool(row.get("correct")),
                                    partial=bool(row.get("partial")),
                                    field_scores=row.get("field_scores", {}),
                                    official_score=float(row.get("official_score", 0.0)),
                                    official_passing=row.get("official_passing", []),
                                    official_failing=row.get("official_failing", []),
                                    task_type=row.get("task_index", ""),
                                    system=row.get("system", ""),
                                    prediction=row.get("prediction", {}),
                                )
                            )
                        status = "OK" if row.get("correct") else ("PART" if row.get("partial") else ("ERR" if row.get("error") else "NO"))
                        err = f" err={row.get('error')}" if row.get("error") else ""
                        append_jsonl(
                            query_checkpoint_file,
                            {
                                "record_type": "query_result",
                                "run_id": run_id,
                                "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "sequence": total_processed,
                                "result": row,
                                "summary_so_far": aggregator.summary(),
                            },
                        )
                        print(f"    [{total_processed}] {status}{err}", flush=True)
                finally:
                    if pool is not None:
                        pool.close()
                        pool.join()

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            partial_file = results_dir / f"option_{args.option}_{system_name}_{timestamp}_partial.json"
            with open(partial_file, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "config": config_dict(args),
                        "summary": aggregator.summary(),
                        "runtime_summary": summarize_runtime(all_results),
                        "reason_diagnostics": reason_diagnostics(all_results),
                        "per_query": all_results,
                    },
                    f,
                    indent=2,
                    default=str,
                )
            print(f"  [Saved partial results to {partial_file}]", flush=True)

    summary = aggregator.summary()
    runtime_summary = summarize_runtime(all_results)
    print(f"\n{'=' * 60}\nEVALUATION SUMMARY - Option {args.option}\n{'=' * 60}", flush=True)
    print(f"Total: {summary.get('total', 0)}", flush=True)
    print(f"Strict Correct: {summary.get('correct', 0)} ({summary.get('correct_pct', 0.0)}%)", flush=True)
    print(f"Official Score: {summary.get('official_score_pct', 0.0)}%", flush=True)
    print(f"Nonzero Partial: {summary.get('partial', 0)} ({summary.get('partial_pct', 0.0)}%)", flush=True)
    print(f"Any Nonzero Score: {summary.get('correct_or_partial_pct', 0.0)}%", flush=True)
    if runtime_summary:
        print("\nRuntime summary:", flush=True)
        for key, stats in runtime_summary.items():
            print(
                f"  {key}: mean={stats['mean']}s p50={stats['p50']}s p95={stats['p95']}s total={stats['total']}s",
                flush=True,
            )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = results_dir / f"option_{args.option}_{timestamp}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": config_dict(args),
                "summary": summary,
                "runtime_summary": runtime_summary,
                "reason_diagnostics": reason_diagnostics(all_results),
                "per_query": all_results,
            },
            f,
            indent=2,
            default=str,
        )
    print(f"\nSaved to: {output_file}", flush=True)


if __name__ == "__main__":
    main()
