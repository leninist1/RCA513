"""Learning-to-rank experiments over Noise Lab candidate features.

This module intentionally stays outside the default runner path. It builds a
candidate-level feature table from the existing hand-built scorers, then trains
a query-grouped ranker to learn when those signals are trustworthy.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from xgboost import XGBRanker

from ..config import SYSTEM_PATHS
from ..data.loader import OpenRCALoader
from ..mace.graph import _canonical_object_name, build_object_graph
from .beamformer import StructuralBeamformer
from .delay_localizer import DelayPatternLocalizer
from .noise_field import NoiseFieldScorer
from .reverb_mask import ReverbSuppressionMask
from .runner import _ground_truth_rank, _rank_objects
from .structural_encoder import StructuralObjectEncoder
from .subspace import SourceNoiseSubspaceDecomposer


RESULT_DIR = Path("/home/dell2/RCA513/yyx/rca513/results")

DEFAULT_BLEND_ALPHAS = (0.25, 0.5, 0.75, 1.0, 1.5)
DEFAULT_GUARDED_MARGINS = (0.25, 0.5, 0.75, 1.0)
DEFAULT_PRIMARY_STRATEGY = "ltr_full"

NOISE_FIELDS = [
    "local_noise",
    "resonance_mass",
    "source_signal",
    "reverb_mass",
    "unexplained_residual",
    "collapse_gain",
    "collapse_recovery_score",
    "exclusive_explanation",
    "temporal_source_score",
    "multi_lead_consistency",
    "hotspot_bias",
    "root_source_score",
]

STRUCTURE_FIELDS = [
    "upstream_context",
    "downstream_context",
    "temporal_lead",
    "mechanism_focus",
    "propagation_signature",
    "source_likelihood",
    "symptom_likelihood",
    "structural_uniqueness",
    "propagation_role_score",
    "hard_negative_resistance",
    "multi_view_consistency",
    "topological_eccentricity",
    "structural_score",
]

DELAY_FIELDS = [
    "source_time_consistency",
    "average_delay_gain",
    "observer_coverage",
    "reverse_penalty",
]

BEAM_FIELDS = [
    "beamformed_explanation",
    "directional_focus",
    "offbeam_penalty",
    "mechanism_coherence",
]

SUBSPACE_FIELDS = [
    "local_residual_source_energy",
    "sector_source_ratio",
    "sector_reverb_ratio",
    "residual_distinctiveness",
    "local_common_mode_alignment",
    "replaceability",
    "sector_size",
]

MASK_FIELDS = [
    "gated_reverb_penalty",
    "source_protection",
    "replaceability_penalty",
    "local_hub_pressure",
]

PREFIXES = [
    "tomcat",
    "mg",
    "ig",
    "apache",
    "mysql",
    "redis",
    "servicetest",
    "node",
    "adservice",
    "frontend",
    "checkoutservice",
    "shippingservice",
    "paymentservice",
    "recommendationservice",
    "emailservice",
    "currencyservice",
    "productcatalogservice",
    "cartservice",
]


def build_candidate_table(
    system_name: str,
    max_queries: Optional[int] = None,
    top_k: int = 20,
    variant: str = "ltr_features",
    telemetry_cache_size: int = 16,
    progress_every: int = 10,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    loader = OpenRCALoader(system_name)
    noise_scorer = NoiseFieldScorer()
    structural_encoder = StructuralObjectEncoder()
    delay_localizer = DelayPatternLocalizer()
    beamformer = StructuralBeamformer()
    subspace_decomposer = SourceNoiseSubspaceDecomposer()
    reverb_mask = ReverbSuppressionMask()

    queries = loader.load_all_queries()
    if max_queries is not None:
        queries = queries[:max_queries]

    rows: List[Dict[str, Any]] = []
    per_query: List[Dict[str, Any]] = []
    telemetry_cache: Dict[Tuple[str, str], Any] = {}
    telemetry_cache_order: List[Tuple[str, str]] = []
    cache_hits = 0
    cache_misses = 0

    for query_idx, query in enumerate(queries):
        gts, inject_time = loader.match_query_to_records(query)
        if not gts:
            per_query.append({
                "query_id": f"{system_name}_{query.task_index}_{query_idx}",
                "status": "skipped",
                "reason": "no_matching_record",
            })
            continue
        query.ground_truth = gts[0]
        query.inject_time = inject_time
        query.telemetry_date = loader.resolve_telemetry_date(query)
        if not query.telemetry_date:
            per_query.append({
                "query_id": f"{system_name}_{query.task_index}_{query_idx}",
                "status": "skipped",
                "reason": "missing_telemetry_date",
            })
            continue

        telemetry_key = (str(query.telemetry_date), str(query.sub_system or ""))
        telemetry = None
        if telemetry_cache_size > 0 and telemetry_key in telemetry_cache:
            telemetry = telemetry_cache[telemetry_key]
            telemetry_cache_order.remove(telemetry_key)
            telemetry_cache_order.append(telemetry_key)
            cache_hits += 1
        if telemetry is None:
            telemetry = loader.load_telemetry(query.telemetry_date, query.sub_system)
            cache_misses += 1
            if telemetry_cache_size > 0:
                telemetry_cache[telemetry_key] = telemetry
                telemetry_cache_order.append(telemetry_key)
                while len(telemetry_cache_order) > telemetry_cache_size:
                    evicted = telemetry_cache_order.pop(0)
                    telemetry_cache.pop(evicted, None)
        object_graph, _ = build_object_graph(telemetry, query, inject_time)
        noise_scores = noise_scorer.score(object_graph)
        structural_scores = structural_encoder.encode(object_graph)
        delay_scores = delay_localizer.score(object_graph)
        beam_scores = beamformer.score(object_graph)
        subspace_scores = subspace_decomposer.score(object_graph, beam_scores=beam_scores)
        mask_scores = reverb_mask.score(
            object_graph,
            noise_scores=noise_scores,
            structural_scores=structural_scores,
            delay_scores=delay_scores,
            beam_scores=beam_scores,
            subspace_scores=subspace_scores,
        )
        ranking = _rank_objects(
            object_graph,
            noise_scores,
            structural_scores,
            delay_scores,
            beam_scores,
            subspace_scores,
            mask_scores,
        )
        gt_component = str(query.ground_truth.component or "")
        base_gt_rank = _ground_truth_rank(ranking, gt_component)
        query_key = _query_key(system_name, query, query_idx)
        per_query.append({
            "query_id": query_key,
            "status": "ok",
            "ground_truth": gt_component,
            "base_gt_rank": base_gt_rank,
            "num_candidates": len(ranking),
        })

        for base_rank, candidate in enumerate(ranking[:top_k], start=1):
            object_id = str(candidate["object_id"])
            node = object_graph.nodes[object_id]
            label = 1 if _candidate_matches_gt(candidate, gt_component) else 0
            rows.append(_candidate_features(
                system_name=system_name,
                query_key=query_key,
                query_index=query_idx,
                query=query,
                inject_time=inject_time,
                candidate=candidate,
                node=node,
                graph=object_graph,
                base_rank=base_rank,
                label=label,
            ))
        if progress_every > 0 and (query_idx + 1) % progress_every == 0:
            print(
                "[features] "
                f"{system_name} query={query_idx + 1}/{len(queries)} "
                f"rows={len(rows)} cache_hits={cache_hits} cache_misses={cache_misses}",
                flush=True,
            )

    df = pd.DataFrame(rows)
    metadata = {
        "system": system_name,
        "max_queries": max_queries,
        "top_k": top_k,
        "variant": variant,
        "telemetry_cache_size": telemetry_cache_size,
        "telemetry_cache_hits": cache_hits,
        "telemetry_cache_misses": cache_misses,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "per_query": per_query,
    }
    return df, metadata


def evaluate_ranker_cv(
    df: pd.DataFrame,
    folds: int = 5,
    random_state: int = 13,
    blend_alphas: Sequence[float] = DEFAULT_BLEND_ALPHAS,
    guarded_margins: Sequence[float] = DEFAULT_GUARDED_MARGINS,
    primary_strategy: str = DEFAULT_PRIMARY_STRATEGY,
    exclude_feature_regex: Optional[str] = None,
    rank_objective: str = "rank:pairwise",
    train_negative_window: Optional[int] = None,
) -> Dict[str, Any]:
    if df.empty:
        return {"summary": _empty_summary(), "per_query": []}

    feature_cols = _feature_columns(df)
    if exclude_feature_regex:
        pattern = re.compile(exclude_feature_regex)
        feature_cols = [col for col in feature_cols if not pattern.search(col)]
    split_ids = list(dict.fromkeys(df["fold_key"].astype(str)))
    if len(split_ids) < 2:
        return {
            "summary": _empty_summary(),
            "base_summary": _empty_summary(),
            "strategies": {},
            "strategy_deltas": {},
            "per_query": [],
            "feature_columns": list(feature_cols),
            "feature_importance": [],
            "scored_table": df.copy(),
            "warning": "need_at_least_two_fold_keys",
        }
    n_splits = max(2, min(folds, len(split_ids)))
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    predictions: List[pd.DataFrame] = []
    importances: List[Dict[str, float]] = []
    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(split_ids), start=1):
        train_split_ids = {split_ids[i] for i in train_idx}
        test_split_ids = {split_ids[i] for i in test_idx}
        train = df[df["fold_key"].isin(train_split_ids)].copy()
        test = df[df["fold_key"].isin(test_split_ids)].copy()
        model = _fit_ranker(
            train,
            feature_cols,
            random_state + fold_idx,
            objective=rank_objective,
            train_negative_window=train_negative_window,
        )
        test["ltr_score"] = model.predict(_matrix(test, feature_cols))
        test["fold"] = fold_idx
        predictions.append(test)
        importances.append(_feature_importance(model, feature_cols))

    scored = pd.concat(predictions, ignore_index=True)
    result = _summarize_predictions(
        scored,
        feature_cols,
        blend_alphas=blend_alphas,
        guarded_margins=guarded_margins,
        primary_strategy=primary_strategy,
    )
    result["feature_importance"] = _aggregate_feature_importance(importances)
    result["scored_table"] = _add_strategy_scores(scored, blend_alphas=blend_alphas)
    return result


def _fit_ranker(
    train: pd.DataFrame,
    feature_cols: Sequence[str],
    random_state: int,
    objective: str,
    train_negative_window: Optional[int],
) -> XGBRanker:
    train = _filter_training_hard_negatives(train, train_negative_window)
    train = train.sort_values(["query_id", "base_rank"]).reset_index(drop=True)
    groups = train.groupby("query_id", sort=False).size().tolist()
    model = XGBRanker(
        objective=objective,
        n_estimators=180,
        max_depth=3,
        learning_rate=0.045,
        subsample=0.88,
        colsample_bytree=0.82,
        reg_lambda=2.0,
        min_child_weight=1.0,
        random_state=random_state,
        n_jobs=2,
        tree_method="hist",
    )
    model.fit(_matrix(train, feature_cols), train["label"].astype(int), group=groups, verbose=False)
    return model


def _filter_training_hard_negatives(train: pd.DataFrame, window: Optional[int]) -> pd.DataFrame:
    if window is None or window <= 0 or train.empty:
        return train
    kept: List[pd.DataFrame] = []
    for _, group in train.groupby("query_id", sort=False):
        group = group.copy()
        mask = (group["label"].astype(int) == 1) | (group["base_rank"].astype(float) <= float(window))
        filtered = group[mask]
        if filtered["label"].astype(int).sum() <= 0:
            filtered = group
        if len(filtered) < 2 and len(group) >= 2:
            filtered = group.sort_values("base_rank").head(2)
        kept.append(filtered)
    return pd.concat(kept, ignore_index=True) if kept else train


def _summarize_predictions(
    scored: pd.DataFrame,
    feature_cols: Sequence[str],
    blend_alphas: Sequence[float],
    guarded_margins: Sequence[float],
    primary_strategy: str,
) -> Dict[str, Any]:
    scored = _add_strategy_scores(scored, blend_alphas=blend_alphas)
    strategy_names = _strategy_names(blend_alphas=blend_alphas, guarded_margins=guarded_margins)
    if primary_strategy not in strategy_names:
        primary_strategy = DEFAULT_PRIMARY_STRATEGY

    per_query: List[Dict[str, Any]] = []
    summaries = {name: _empty_summary() for name in strategy_names}
    rank_rows: List[Dict[str, Optional[int]]] = []

    for query_id, group in scored.groupby("query_id", sort=False):
        group = group.copy()
        ranks: Dict[str, Optional[int]] = {}
        for name in strategy_names:
            ordered = _order_group(group, strategy=name)
            rank = _rank_from_sorted(ordered)
            ranks[name] = rank
            _accumulate_summary(summaries[name], rank, len(group))

        primary_group = _order_group(group, strategy=primary_strategy)
        top_candidates = [
            {
                "object_id": str(row.object_id),
                "strategy_score": round(float(_row_strategy_score(row, primary_strategy)), 6),
                "ltr_score": round(float(getattr(row, "ltr_score", 0.0)), 6),
                "base_rank": int(row.base_rank),
                "base_score": round(float(row.base_score), 6),
                "label": int(row.label),
            }
            for row in primary_group.head(5).itertuples(index=False)
        ]
        gt = str(group["ground_truth"].iloc[0])
        base_rank = ranks["base"]
        primary_rank = ranks[primary_strategy]
        per_query.append({
            "query_id": query_id,
            "ground_truth": gt,
            "base_gt_rank": base_rank,
            "ltr_gt_rank": ranks.get(DEFAULT_PRIMARY_STRATEGY),
            "primary_strategy": primary_strategy,
            "primary_gt_rank": primary_rank,
            "ranks": ranks,
            "top_candidates": top_candidates,
        })
        rank_rows.append({"query_id": query_id, **ranks})

    for summary in summaries.values():
        _finalize_summary(summary)

    deltas = {
        name: _strategy_delta(rank_rows, baseline="base", strategy=name)
        for name in strategy_names
        if name != "base"
    }

    return {
        "summary": summaries[primary_strategy],
        "base_summary": summaries["base"],
        "strategies": summaries,
        "strategy_deltas": deltas,
        "per_query": per_query,
        "feature_columns": list(feature_cols),
    }


def _rank_from_sorted(group: pd.DataFrame) -> Optional[int]:
    for rank, row in enumerate(group.itertuples(index=False), start=1):
        if int(row.label) == 1:
            return rank
    return None


def _strategy_names(blend_alphas: Sequence[float], guarded_margins: Sequence[float]) -> List[str]:
    names = ["base", DEFAULT_PRIMARY_STRATEGY]
    names.extend(_blend_name(alpha) for alpha in blend_alphas)
    names.extend(_guarded_name(margin) for margin in guarded_margins)
    return list(dict.fromkeys(names))


def _order_group(group: pd.DataFrame, strategy: str) -> pd.DataFrame:
    if strategy == "base":
        return group.sort_values(["base_rank"], ascending=[True])
    if strategy == DEFAULT_PRIMARY_STRATEGY:
        return group.sort_values(["ltr_score", "base_score"], ascending=[False, False])
    if strategy.startswith("blend_a"):
        return group.sort_values([strategy, "base_score"], ascending=[False, False])
    if strategy.startswith("guarded_m"):
        margin = _guarded_margin_from_name(strategy)
        return _guarded_promotion_order(group, margin=margin)
    raise ValueError(f"unknown ranking strategy: {strategy}")


def _guarded_promotion_order(group: pd.DataFrame, margin: float, promotion_window: int = 30) -> pd.DataFrame:
    base = group.sort_values(["base_rank"], ascending=[True]).copy()
    if base.empty:
        return base

    base_top = base.iloc[0]
    promoted_mask = (
        (base["base_rank"] > 1)
        & (base["base_rank"] <= promotion_window)
        & (base["ltr_z"] >= float(base_top["ltr_z"]) + margin)
    )
    promoted = base[promoted_mask].sort_values(["ltr_z", "base_score"], ascending=[False, False])
    remainder = base[~promoted_mask].sort_values(["base_rank"], ascending=[True])
    return pd.concat([promoted, remainder], axis=0)


def _add_strategy_scores(scored: pd.DataFrame, blend_alphas: Sequence[float]) -> pd.DataFrame:
    scored = scored.copy()
    scored["base_z"] = scored.groupby("query_id", sort=False)["base_score"].transform(_zscore)
    scored["ltr_z"] = scored.groupby("query_id", sort=False)["ltr_score"].transform(_zscore)
    for alpha in blend_alphas:
        scored[_blend_name(alpha)] = scored["base_z"] + float(alpha) * scored["ltr_z"]
    return scored


def _zscore(values: pd.Series) -> pd.Series:
    std = float(values.std(ddof=0))
    if not np.isfinite(std) or std < 1e-9:
        return pd.Series(np.zeros(len(values), dtype=float), index=values.index)
    return (values - float(values.mean())) / std


def _blend_name(alpha: float) -> str:
    return f"blend_a{_format_float_token(alpha)}"


def _guarded_name(margin: float) -> str:
    return f"guarded_m{_format_float_token(margin)}"


def _format_float_token(value: float) -> str:
    return f"{float(value):g}".replace("-", "neg").replace(".", "p")


def _guarded_margin_from_name(name: str) -> float:
    token = name.split("guarded_m", 1)[1]
    return float(token.replace("neg", "-").replace("p", "."))


def _row_strategy_score(row: Any, strategy: str) -> float:
    if strategy == "base":
        return -float(getattr(row, "base_rank", 0.0))
    if strategy == DEFAULT_PRIMARY_STRATEGY:
        return float(getattr(row, "ltr_score", 0.0))
    return float(getattr(row, strategy, 0.0))


def _strategy_delta(
    rank_rows: Sequence[Dict[str, Optional[int]]],
    baseline: str,
    strategy: str,
) -> Dict[str, Any]:
    improved = regressed = same = 0
    deltas: List[float] = []
    for row in rank_rows:
        base_rank = row.get(baseline)
        rank = row.get(strategy)
        observed = [
            int(value or 0)
            for key, value in row.items()
            if key != "query_id" and value is not None
        ]
        fallback = (max(observed) + 1) if observed else 1
        base_value = float(base_rank or fallback)
        rank_value = float(rank or fallback)
        delta = base_value - rank_value
        deltas.append(delta)
        if delta > 0:
            improved += 1
        elif delta < 0:
            regressed += 1
        else:
            same += 1
    return {
        "improved": improved,
        "regressed": regressed,
        "same": same,
        "mean_rank_delta": round(float(np.mean(deltas)) if deltas else 0.0, 4),
    }


def _empty_summary() -> Dict[str, Any]:
    return {
        "total": 0,
        "top1": 0,
        "top3": 0,
        "top5": 0,
        "avg_gt_rank": 0.0,
    }


def _accumulate_summary(summary: Dict[str, Any], rank: Optional[int], fallback: int) -> None:
    summary["total"] += 1
    if rank == 1:
        summary["top1"] += 1
    if rank is not None and rank <= 3:
        summary["top3"] += 1
    if rank is not None and rank <= 5:
        summary["top5"] += 1
    summary["avg_gt_rank"] += float(rank or (fallback + 1))


def _finalize_summary(summary: Dict[str, Any]) -> None:
    total = max(1, int(summary["total"]))
    summary["top1_pct"] = round(100.0 * summary["top1"] / total, 2)
    summary["top3_pct"] = round(100.0 * summary["top3"] / total, 2)
    summary["top5_pct"] = round(100.0 * summary["top5"] / total, 2)
    summary["avg_gt_rank"] = round(float(summary["avg_gt_rank"]) / total, 4)


def _feature_importance(model: XGBRanker, feature_cols: Sequence[str]) -> Dict[str, float]:
    booster = model.get_booster()
    raw_scores = booster.get_score(importance_type="gain")
    mapped: Dict[str, float] = {}
    for key, value in raw_scores.items():
        feature_name = key
        if key.startswith("f") and key[1:].isdigit():
            idx = int(key[1:])
            if idx < len(feature_cols):
                feature_name = feature_cols[idx]
        mapped[feature_name] = float(value)
    return mapped


def _aggregate_feature_importance(importances: Sequence[Dict[str, float]], limit: int = 30) -> List[Dict[str, Any]]:
    if not importances:
        return []
    features = sorted({feature for scores in importances for feature in scores})
    rows: List[Dict[str, Any]] = []
    for feature in features:
        values = [scores.get(feature, 0.0) for scores in importances]
        rows.append({
            "feature": feature,
            "mean_gain": round(float(np.mean(values)), 8),
            "nonzero_folds": int(sum(1 for value in values if value > 0)),
        })
    rows.sort(key=lambda row: (row["mean_gain"], row["nonzero_folds"]), reverse=True)
    return rows[:limit]


def _candidate_features(
    system_name: str,
    query_key: str,
    query_index: int,
    query: Any,
    inject_time: Optional[float],
    candidate: Dict[str, Any],
    node: Any,
    graph: Any,
    base_rank: int,
    label: int,
) -> Dict[str, Any]:
    object_id = str(candidate["object_id"])
    degree_out = len(graph.adjacency.get(object_id, {}))
    degree_in = sum(1 for _, children in graph.adjacency.items() if object_id in children)
    earliest_offset = 0.0
    has_earliest = 0.0
    if inject_time is not None and node.earliest_timestamp is not None:
        earliest_offset = float(node.earliest_timestamp - inject_time)
        has_earliest = 1.0

    row: Dict[str, Any] = {
        "system": system_name,
        "query_id": query_key,
        "fold_key": f"{system_name}:{query.ground_truth.component}:{query.ground_truth.datetime_str}",
        "query_index": query_index,
        "task_index": str(query.task_index),
        "telemetry_date": str(query.telemetry_date or ""),
        "ground_truth": str(query.ground_truth.component or ""),
        "object_id": object_id,
        "base_rank": int(base_rank),
        "base_score": float(candidate.get("score", 0.0)),
        "label": int(label),
        "rank_recip": 1.0 / max(1, base_rank),
        "rank_log": math.log1p(base_rank),
        "anomaly_score": float(node.anomaly_score),
        "metric_score": float(node.metric_score),
        "log_score": float(node.log_score),
        "trace_score": float(node.trace_score),
        "change_score": float(node.change_score),
        "degree_in": float(degree_in),
        "degree_out": float(degree_out),
        "degree_total": float(degree_in + degree_out),
        "is_isolated": 1.0 if degree_in == 0 and degree_out == 0 else 0.0,
        "member_count": float(len(node.members)),
        "member_log": math.log1p(len(node.members)),
        "incoming_mass": float(graph.incoming_mass(object_id)),
        "topological_mass": float(graph.topological_mass(object_id)),
        "has_earliest": has_earliest,
        "earliest_offset": _clip(earliest_offset, -3600.0, 3600.0) / 3600.0,
    }

    for name, fields in (
        ("noise", NOISE_FIELDS),
        ("structure", STRUCTURE_FIELDS),
        ("delay", DELAY_FIELDS),
        ("beam", BEAM_FIELDS),
        ("subspace", SUBSPACE_FIELDS),
        ("mask", MASK_FIELDS),
    ):
        payload = candidate.get("reverb_mask" if name == "mask" else name, {})
        for field in fields:
            row[f"{name}_{field}"] = float(payload.get(field, 0.0))

    reason = str(candidate.get("reason", "")).lower()
    row.update({
        "reason_cpu": 1.0 if "cpu" in reason else 0.0,
        "reason_memory": 1.0 if "memory" in reason else 0.0,
        "reason_disk": 1.0 if "disk" in reason else 0.0,
        "reason_db": 1.0 if "db" in reason else 0.0,
        "reason_network": 1.0 if "network" in reason or "latency" in reason else 0.0,
        "reason_process": 1.0 if "process" in reason else 0.0,
    })

    normalized_id = object_id.lower()
    for prefix in PREFIXES:
        row[f"prefix_{prefix}"] = 1.0 if normalized_id.startswith(prefix) else 0.0

    return row


def _candidate_matches_gt(candidate: Dict[str, Any], ground_truth_component: str) -> bool:
    gt_raw = str(ground_truth_component).strip().lower()
    keep_instance_suffix = bool(re.search(r"(?:^|[._-])[a-z]+[-_]\d+(?:$|[._-])", gt_raw))
    gt_canonical = _canonical_object_name(gt_raw, keep_instance_suffix=keep_instance_suffix)
    entity = str(candidate.get("entity", "")).strip().lower()
    obj_id = str(candidate.get("object_id", "")).strip().lower()
    obj_canonical = _canonical_object_name(obj_id, keep_instance_suffix=keep_instance_suffix)
    entity_canonical = _canonical_object_name(entity, keep_instance_suffix=keep_instance_suffix)
    if entity == gt_raw or obj_id == gt_raw:
        return True
    if entity_canonical == gt_canonical or obj_canonical == gt_canonical:
        return True
    return obj_canonical.startswith(gt_canonical + ".") or obj_canonical.startswith(gt_canonical + "-")


def _query_key(system_name: str, query: Any, query_idx: int) -> str:
    gt = str(query.ground_truth.component or "")
    ts = str(query.ground_truth.datetime_str or "")
    return f"{system_name}:{query_idx}:{query.task_index}:{gt}:{ts}"


def _feature_columns(df: pd.DataFrame) -> List[str]:
    excluded = {
        "system",
        "query_id",
        "fold_key",
        "query_index",
        "task_index",
        "telemetry_date",
        "ground_truth",
        "object_id",
        "label",
        "fold",
        "ltr_score",
        "base_z",
        "ltr_z",
    }
    cols = [
        col for col in df.columns
        if (
            col not in excluded
            and not col.startswith("blend_a")
            and not col.startswith("guarded_m")
            and pd.api.types.is_numeric_dtype(df[col])
        )
    ]
    return sorted(cols)


def _matrix(df: pd.DataFrame, feature_cols: Sequence[str]) -> np.ndarray:
    if not feature_cols:
        return np.zeros((len(df), 1), dtype=float)
    matrix = df.loc[:, feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return matrix.to_numpy(dtype=float)


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _parse_float_list(raw: str) -> List[float]:
    values: List[float] = []
    for token in raw.split(","):
        token = token.strip()
        if token:
            values.append(float(token))
    return values


def _load_candidate_table(path: Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    table = pd.read_csv(path)
    system = ""
    if "system" in table and not table.empty:
        system = str(table["system"].mode().iloc[0])
    metadata = {
        "system": system,
        "source_features_csv": str(path),
        "num_rows": int(len(table)),
        "num_queries": int(table["query_id"].nunique()) if "query_id" in table else 0,
        "top_k": int(table.groupby("query_id").size().max()) if "query_id" in table and not table.empty else 0,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    return table, metadata


def _write_outputs(
    payload: Dict[str, Any],
    table: pd.DataFrame,
    variant: str,
    system: str,
    scored_table: Optional[pd.DataFrame] = None,
) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = RESULT_DIR / f"noise_lab_ltr_{variant}_{system}_{ts}.json"
    table_path = RESULT_DIR / f"noise_lab_ltr_features_{variant}_{system}_{ts}.csv"
    scores_path = RESULT_DIR / f"noise_lab_ltr_scores_{variant}_{system}_{ts}.csv"
    result_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2))
    table.to_csv(table_path, index=False)
    if scored_table is not None:
        scored_table.to_csv(scores_path, index=False)
    print(json.dumps({
        "primary_strategy": payload["config"].get("primary_strategy"),
        "summary": payload["summary"],
        "base_summary": payload["base_summary"],
        "strategy_deltas": payload.get("strategy_deltas", {}),
    }, ensure_ascii=True, indent=2))
    print(f"result={result_path}")
    print(f"features={table_path}")
    if scored_table is not None:
        print(f"scores={scores_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Noise Lab candidate learning-to-rank experiment")
    parser.add_argument("--system", choices=list(SYSTEM_PATHS))
    parser.add_argument("--max-queries", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--variant", default="xgbrank")
    parser.add_argument("--features-csv", type=Path)
    parser.add_argument("--blend-alphas", default=",".join(str(value) for value in DEFAULT_BLEND_ALPHAS))
    parser.add_argument("--guarded-margins", default=",".join(str(value) for value in DEFAULT_GUARDED_MARGINS))
    parser.add_argument("--primary-strategy", default=DEFAULT_PRIMARY_STRATEGY)
    parser.add_argument("--exclude-feature-regex")
    parser.add_argument("--telemetry-cache-size", type=int, default=16)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument(
        "--rank-objective",
        choices=["rank:pairwise", "rank:ndcg", "rank:map"],
        default="rank:pairwise",
    )
    parser.add_argument("--train-negative-window", type=int)
    args = parser.parse_args()

    if args.features_csv:
        table, metadata = _load_candidate_table(args.features_csv)
        system = args.system or str(metadata.get("system") or "")
        if not system:
            parser.error("--system is required when --features-csv has no system column")
    else:
        if not args.system:
            parser.error("--system is required unless --features-csv is provided")
        system = args.system
        table, metadata = build_candidate_table(
            system_name=system,
            max_queries=args.max_queries,
            top_k=args.top_k,
            variant=args.variant,
            telemetry_cache_size=args.telemetry_cache_size,
            progress_every=args.progress_every,
        )

    blend_alphas = _parse_float_list(args.blend_alphas)
    guarded_margins = _parse_float_list(args.guarded_margins)
    result = evaluate_ranker_cv(
        table,
        folds=args.folds,
        blend_alphas=blend_alphas,
        guarded_margins=guarded_margins,
        primary_strategy=args.primary_strategy,
        exclude_feature_regex=args.exclude_feature_regex,
        rank_objective=args.rank_objective,
        train_negative_window=args.train_negative_window,
    )
    scored_table = result.pop("scored_table", None)
    payload = {
        "config": {
            "system": system,
            "max_queries": args.max_queries,
            "top_k": args.top_k,
            "folds": args.folds,
            "variant": args.variant,
            "features_csv": str(args.features_csv) if args.features_csv else None,
            "blend_alphas": blend_alphas,
            "guarded_margins": guarded_margins,
            "primary_strategy": args.primary_strategy,
            "exclude_feature_regex": args.exclude_feature_regex,
            "telemetry_cache_size": args.telemetry_cache_size,
            "progress_every": args.progress_every,
            "rank_objective": args.rank_objective,
            "train_negative_window": args.train_negative_window,
            "timestamp": metadata["timestamp"],
        },
        "summary": result["summary"],
        "base_summary": result["base_summary"],
        "strategies": result.get("strategies", {}),
        "strategy_deltas": result.get("strategy_deltas", {}),
        "feature_columns": result["feature_columns"],
        "feature_importance": result.get("feature_importance", []),
        "per_query": result["per_query"],
        "feature_metadata": metadata,
    }
    _write_outputs(payload, table, args.variant, system, scored_table=scored_table)


if __name__ == "__main__":
    main()
