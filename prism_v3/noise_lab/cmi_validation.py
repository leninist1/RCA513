"""Offline CMI validation table for noise-native PRISM candidates.

CMI means causal mechanism intervention in this module:
1. condition a candidate on plausible mechanism parents and measure whether its
   fault-window residual remains high;
2. repair the candidate signal to its normal mechanism prediction and estimate
   how much affected-neighbor anomaly would collapse;
3. penalize candidates whose anomaly is better explained by parents or by
   alternative candidates.

The output is diagnostic.  Labels are written only for offline analysis and are
not used by any feature or score computation.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..config import SYSTEM_PATHS
from ..data.loader import OpenRCALoader
from ..mace.graph import ObjectGraph, ObjectNode, _canonical_object_name, build_object_graph


DEFAULT_OUTPUT_DIR = Path("/home/dell2/RCA513/yyx/prism_v3/results/cmi_validation")
EPS = 1e-9


@dataclass
class CandidateRef:
    raw_name: str
    object_id: str
    source: str
    in_result_pool: bool
    current_rank: Optional[int] = None
    current_score: Optional[float] = None
    posterior: Optional[float] = None
    root_probability: Optional[float] = None
    status: str = ""
    event_factors: Dict[str, float] = None

    def __post_init__(self) -> None:
        if self.event_factors is None:
            self.event_factors = {}


def build_cmi_table(
    system_name: str,
    result_json: Optional[Path],
    max_queries: Optional[int],
    top_k: int,
    baseline_window: int,
    fault_window: int,
    max_conditioners: int,
    max_effect_scope: int,
    telemetry_cache_size: int,
    progress_every: int,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    loader = OpenRCALoader(system_name)
    result_index = _load_result_index(result_json) if result_json else {}
    queries = loader.load_all_queries()
    if max_queries is not None:
        queries = queries[:max_queries]

    rows: List[Dict[str, Any]] = []
    per_query: List[Dict[str, Any]] = []
    telemetry_cache: Dict[Tuple[str, str], Any] = {}
    telemetry_order: List[Tuple[str, str]] = []
    cache_hits = 0
    cache_misses = 0
    t_start = time.time()

    for query_idx, query in enumerate(queries):
        query_id = f"{system_name}_{query.task_index}_{query_idx}"
        gt_records, inject_time = loader.match_query_to_records(query)
        if not gt_records or inject_time is None:
            per_query.append({"query_id": query_id, "status": "skipped", "reason": "no_ground_truth"})
            continue
        query.ground_truth = gt_records[0]
        query.inject_time = inject_time
        query.telemetry_date = loader.resolve_telemetry_date(query)
        if not query.telemetry_date:
            per_query.append({"query_id": query_id, "status": "skipped", "reason": "missing_telemetry_date"})
            continue

        telemetry_key = (str(query.telemetry_date), str(query.sub_system or ""))
        telemetry = telemetry_cache.get(telemetry_key)
        if telemetry is not None:
            cache_hits += 1
            telemetry_order.remove(telemetry_key)
            telemetry_order.append(telemetry_key)
        else:
            telemetry = loader.load_telemetry(query.telemetry_date, query.sub_system)
            cache_misses += 1
            telemetry_cache[telemetry_key] = telemetry
            telemetry_order.append(telemetry_key)
            while telemetry_cache_size > 0 and len(telemetry_order) > telemetry_cache_size:
                evicted = telemetry_order.pop(0)
                telemetry_cache.pop(evicted, None)

        result_item = result_index.get(query_id, {})
        graph, graph_debug = _graph_from_result_item(result_item, system_name)
        if graph is None:
            graph, graph_debug = build_object_graph(telemetry, query, inject_time)
        signal, signal_debug = _build_object_signal(
            telemetry.metrics if telemetry.metrics is not None else pd.DataFrame(),
            graph=graph,
            entity_to_object=graph_debug.get("entity_to_object", {}),
            system_name=system_name,
            inject_time=float(inject_time),
            baseline_window=baseline_window,
            fault_window=fault_window,
        )
        if signal.empty:
            per_query.append({"query_id": query_id, "status": "skipped", "reason": "empty_signal"})
            continue

        candidates = _candidate_refs_from_result(
            result_item=result_item,
            graph=graph,
            system_name=system_name,
            top_k=top_k,
        )
        if not candidates:
            candidates = _fallback_candidate_refs(graph, top_k=top_k)

        gt_component = str(query.ground_truth.component or "")
        gt_object = _resolve_object_id(gt_component, graph, system_name)
        if gt_object and all(ref.object_id != gt_object for ref in candidates):
            candidates.append(CandidateRef(
                raw_name=gt_component,
                object_id=gt_object,
                source="ground_truth_diagnostic",
                in_result_pool=False,
            ))

        candidate_ids = [ref.object_id for ref in candidates if ref.object_id in graph.nodes]
        pair_cache: Dict[Tuple[str, str], Dict[str, float]] = {}
        for ref in candidates:
            if ref.object_id not in graph.nodes:
                continue
            label = 1 if _matches_ground_truth(ref.object_id, ref.raw_name, gt_component, system_name) else 0
            row = _candidate_cmi_features(
                query_id=query_id,
                query_index=query_idx,
                task_index=str(query.task_index),
                telemetry_date=str(query.telemetry_date or ""),
                ground_truth=gt_component,
                inject_time=float(inject_time),
                ref=ref,
                graph=graph,
                signal=signal,
                label=label,
                candidate_ids=candidate_ids,
                pair_cache=pair_cache,
                max_conditioners=max_conditioners,
                max_effect_scope=max_effect_scope,
            )
            rows.append(row)

        per_query.append({
            "query_id": query_id,
            "status": "ok",
            "task_index": str(query.task_index),
            "ground_truth": gt_component,
            "telemetry_date": str(query.telemetry_date or ""),
            "num_candidates": len(candidates),
            "gt_in_result_pool": any(row.get("query_id") == query_id and row.get("label") == 1 and row.get("in_result_pool") == 1 for row in rows),
            "signal_rows": int(signal_debug.get("signal_rows", 0)),
            "signal_objects": int(signal_debug.get("signal_objects", 0)),
        })
        if progress_every > 0 and (query_idx + 1) % progress_every == 0:
            print(
                "[cmi] "
                f"{system_name} query={query_idx + 1}/{len(queries)} "
                f"rows={len(rows)} cache_hits={cache_hits} cache_misses={cache_misses}",
                flush=True,
            )

    table = pd.DataFrame(rows)
    if not table.empty:
        table = _score_and_rank_cmi(table)
    metadata = {
        "system": system_name,
        "result_json": str(result_json) if result_json else "",
        "max_queries": max_queries,
        "top_k": top_k,
        "baseline_window": baseline_window,
        "fault_window": fault_window,
        "max_conditioners": max_conditioners,
        "max_effect_scope": max_effect_scope,
        "telemetry_cache_size": telemetry_cache_size,
        "telemetry_cache_hits": cache_hits,
        "telemetry_cache_misses": cache_misses,
        "elapsed_sec": round(time.time() - t_start, 3),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "per_query": per_query,
    }
    return table, metadata


def _load_result_index(path: Path) -> Dict[str, Dict[str, Any]]:
    payload = json.loads(path.read_text())
    if isinstance(payload, dict):
        rows = payload.get("per_query", [])
    elif isinstance(payload, list):
        rows = payload
    else:
        rows = []
    return {
        str(row.get("query_id", "")): row
        for row in rows
        if isinstance(row, dict) and row.get("query_id")
    }


def _graph_from_result_item(result_item: Dict[str, Any], system_name: str) -> Tuple[Optional[ObjectGraph], Dict[str, Any]]:
    if not isinstance(result_item, dict) or not result_item:
        return None, {}
    debug = result_item.get("debug", {}) if isinstance(result_item.get("debug"), dict) else {}
    induction = debug.get("object_induction", {}) if isinstance(debug.get("object_induction"), dict) else {}
    object_to_members = induction.get("object_to_members", {}) if isinstance(induction.get("object_to_members"), dict) else {}
    raw_to_object = induction.get("raw_to_object", {}) if isinstance(induction.get("raw_to_object"), dict) else {}
    metric_signal = debug.get("metric_signal", {}) if isinstance(debug.get("metric_signal"), dict) else {}
    log_signal = debug.get("log_signal", {}) if isinstance(debug.get("log_signal"), dict) else {}
    graph_summary = result_item.get("graph_summary", {}) if isinstance(result_item.get("graph_summary"), dict) else {}
    top_edges = graph_summary.get("top_edges", []) if isinstance(graph_summary.get("top_edges"), list) else []
    if not object_to_members and not metric_signal and not log_signal:
        return None, {}

    object_ids = set(str(obj) for obj in object_to_members)
    object_ids.update(str(key) for key in metric_signal)
    object_ids.update(str(key) for key in log_signal)
    for edge in top_edges:
        if isinstance(edge, dict):
            object_ids.add(str(edge.get("source", "")))
            object_ids.add(str(edge.get("target", "")))
    object_ids = {obj for obj in object_ids if obj and obj.lower() != "nan"}
    if not object_ids:
        return None, {}

    nodes: Dict[str, ObjectNode] = {}
    inject_ts = _result_inject_timestamp(result_item)
    for object_id in sorted(object_ids):
        members = [str(member) for member in object_to_members.get(object_id, [object_id])]
        metric_score = _lookup_signal(metric_signal, object_id)
        log_score = _lookup_signal(log_signal, object_id)
        anomaly = _clip(0.70 * metric_score + 0.30 * log_score, 0.0, 1.0)
        if anomaly <= 0.0:
            anomaly = max(metric_score, log_score, 0.01)
        nodes[object_id] = ObjectNode(
            object_id=object_id,
            members=members,
            representative=members[0] if members else object_id,
            anomaly_score=float(anomaly),
            earliest_timestamp=inject_ts,
            metric_score=float(metric_score),
            log_score=float(log_score),
            trace_score=0.0,
            change_score=0.0,
        )

    adjacency: Dict[str, Dict[str, float]] = {}
    for edge in top_edges:
        if not isinstance(edge, dict):
            continue
        src = str(edge.get("source", "") or "")
        dst = str(edge.get("target", "") or "")
        if src not in nodes or dst not in nodes or src == dst:
            continue
        adjacency.setdefault(src, {})[dst] = max(adjacency.setdefault(src, {}).get(dst, 0.0), _safe_float(edge.get("weight", 0.0)))
    entity_to_object = {str(raw): str(obj) for raw, obj in raw_to_object.items()}
    if not entity_to_object:
        for object_id, node in nodes.items():
            for member in node.members:
                entity_to_object[str(member)] = object_id
    return ObjectGraph(nodes=nodes, adjacency=adjacency), {
        "entity_to_object": entity_to_object,
        "source": "result_json_debug",
        "system": system_name,
    }


def _result_inject_timestamp(result_item: Dict[str, Any]) -> Optional[float]:
    gt = result_item.get("ground_truth", {}) if isinstance(result_item.get("ground_truth"), dict) else {}
    raw = gt.get("datetime")
    if raw:
        try:
            return datetime.strptime(str(raw), "%Y-%m-%d %H:%M:%S").timestamp()
        except ValueError:
            pass
    pred = result_item.get("prediction", {}) if isinstance(result_item.get("prediction"), dict) else {}
    times = pred.get("time", [])
    if isinstance(times, list) and times:
        try:
            return datetime.strptime(str(times[0]), "%Y-%m-%d %H:%M:%S").timestamp()
        except ValueError:
            return None
    return None


def _lookup_signal(payload: Dict[str, Any], object_id: str) -> float:
    if object_id in payload:
        return _safe_float(payload.get(object_id))
    lowered = object_id.lower()
    for key, value in payload.items():
        if str(key).lower() == lowered:
            return _safe_float(value)
    return 0.0


def _candidate_refs_from_result(
    result_item: Dict[str, Any],
    graph: ObjectGraph,
    system_name: str,
    top_k: int,
) -> List[CandidateRef]:
    refs: List[CandidateRef] = []
    by_object: Dict[str, CandidateRef] = {}
    event_payloads = _event_payloads_by_component(result_item)

    def add(
        raw_name: Any,
        source: str,
        rank: Optional[int],
        score: Optional[float] = None,
        posterior: Optional[float] = None,
        root_probability: Optional[float] = None,
        status: str = "",
    ) -> None:
        raw = str(raw_name or "").strip()
        if not raw:
            return
        object_id = _resolve_object_id(raw, graph, system_name)
        if not object_id:
            return
        event = event_payloads.get(object_id, {})
        factors = {
            str(key): _safe_float(value)
            for key, value in dict(event.get("factors", {}) or {}).items()
            if _is_number(value)
        }
        current_rank = rank if source == "prediction_top_scores" else None
        ref = by_object.get(object_id)
        if ref is None:
            ref = CandidateRef(
                raw_name=raw,
                object_id=object_id,
                source=source,
                in_result_pool=True,
                current_rank=current_rank,
                current_score=score,
                posterior=posterior if posterior is not None else _safe_optional(event.get("posterior")),
                root_probability=root_probability,
                status=status or str(event.get("status", "") or ""),
                event_factors=factors,
            )
            refs.append(ref)
            by_object[object_id] = ref
            return
        ref.source = f"{ref.source}|{source}" if source not in ref.source.split("|") else ref.source
        if current_rank is not None and (ref.current_rank is None or current_rank < ref.current_rank):
            ref.current_rank = current_rank
        if ref.current_score is None and score is not None:
            ref.current_score = score
        if ref.posterior is None and posterior is not None:
            ref.posterior = posterior
        if ref.root_probability is None and root_probability is not None:
            ref.root_probability = root_probability
        if not ref.status and status:
            ref.status = status
        if not ref.event_factors and factors:
            ref.event_factors = factors

    prediction = result_item.get("prediction", {}) if isinstance(result_item, dict) else {}
    top_scores = prediction.get("top_scores", []) if isinstance(prediction, dict) else []
    for idx, item in enumerate(top_scores[:top_k], start=1):
        if not isinstance(item, dict):
            continue
        add(
            item.get("entity"),
            "prediction_top_scores",
            idx,
            score=_safe_optional(item.get("score")),
            posterior=_safe_optional(item.get("posterior")),
            root_probability=_safe_optional(item.get("root_probability")),
            status=str(item.get("status", "") or ""),
        )

    for source, path in (
        ("top_candidates", result_item.get("top_candidates", [])),
        ("belief_top20", result_item.get("belief_top20", [])),
    ):
        for idx, item in enumerate(path[:top_k], start=1):
            if isinstance(item, dict):
                add(item.get("entity") or item.get("object_id"), source, idx, score=_safe_optional(item.get("score") or item.get("prob")))

    debug = result_item.get("debug", {}) if isinstance(result_item, dict) else {}
    scope = debug.get("final_decision_scope", []) if isinstance(debug, dict) else []
    for idx, name in enumerate(scope[:top_k], start=1):
        add(name, "final_decision_scope", idx)

    for idx, event in enumerate(event_payloads.values(), start=1):
        add(event.get("component"), "noise_native_event", idx, posterior=_safe_optional(event.get("posterior")), status=str(event.get("status", "") or ""))

    refs.sort(key=lambda item: item.current_rank if item.current_rank is not None else top_k + 100)
    return refs[:top_k]


def _event_payloads_by_component(result_item: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    debug = result_item.get("debug", {}) if isinstance(result_item, dict) else {}
    agent = debug.get("noise_native_agent", {}) if isinstance(debug, dict) else {}
    state = agent.get("state", {}) if isinstance(agent, dict) else {}
    events = state.get("events", []) if isinstance(state, dict) else []
    if not isinstance(events, list):
        return out
    for event in events:
        if not isinstance(event, dict):
            continue
        component = str(event.get("component", "") or "")
        if not component:
            continue
        out[component] = event
        out[component.lower()] = event
        out[_canonical_object_name(component)] = event
    return out


def _fallback_candidate_refs(graph: ObjectGraph, top_k: int) -> List[CandidateRef]:
    ranked = sorted(
        graph.nodes.values(),
        key=lambda node: (float(node.anomaly_score), float(node.metric_score), float(node.log_score)),
        reverse=True,
    )
    return [
        CandidateRef(
            raw_name=node.representative or node.object_id,
            object_id=node.object_id,
            source="graph_anomaly_fallback",
            in_result_pool=True,
            current_rank=idx,
            current_score=float(node.anomaly_score),
        )
        for idx, node in enumerate(ranked[:top_k], start=1)
    ]


def _build_object_signal(
    metrics: pd.DataFrame,
    graph: ObjectGraph,
    entity_to_object: Dict[str, str],
    system_name: str,
    inject_time: float,
    baseline_window: int,
    fault_window: int,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if metrics is None or metrics.empty or "timestamp" not in metrics.columns:
        return pd.DataFrame(), {"signal_rows": 0, "signal_objects": 0}
    start = inject_time - float(baseline_window)
    end = inject_time + float(fault_window)
    cols = ["timestamp", "entity", "metric_name", "value"]
    frame = metrics.loc[
        (metrics["timestamp"] >= start) & (metrics["timestamp"] <= end),
        [col for col in cols if col in metrics.columns],
    ].copy()
    if frame.empty or not set(cols).issubset(frame.columns):
        return pd.DataFrame(), {"signal_rows": 0, "signal_objects": 0}

    frame["entity"] = frame["entity"].astype(str)
    keep_instance_suffix = system_name in {"Telecom", "Market"}
    frame["object_id"] = frame["entity"].map(entity_to_object)
    missing = frame["object_id"].isna()
    if missing.any():
        frame.loc[missing, "object_id"] = frame.loc[missing, "entity"].map(
            lambda value: _canonical_object_name(value, keep_instance_suffix=keep_instance_suffix)
        )
    frame = frame[frame["object_id"].isin(graph.nodes)].copy()
    if frame.empty:
        return pd.DataFrame(), {"signal_rows": 0, "signal_objects": 0}

    baseline = frame[frame["timestamp"] < inject_time]
    if baseline.empty:
        return pd.DataFrame(), {"signal_rows": 0, "signal_objects": 0}
    stats = (
        baseline.groupby(["entity", "metric_name"])["value"]
        .agg(["mean", "std"])
        .reset_index()
    )
    frame = frame.merge(stats, on=["entity", "metric_name"], how="left")
    frame = frame.dropna(subset=["mean"])
    if frame.empty:
        return pd.DataFrame(), {"signal_rows": 0, "signal_objects": 0}
    denom = np.maximum(
        pd.to_numeric(frame["std"], errors="coerce").fillna(0.0).to_numpy(dtype=float),
        np.maximum(np.abs(pd.to_numeric(frame["mean"], errors="coerce").fillna(0.0).to_numpy(dtype=float)) * 0.05, 1e-6),
    )
    values = pd.to_numeric(frame["value"], errors="coerce").to_numpy(dtype=float)
    means = pd.to_numeric(frame["mean"], errors="coerce").to_numpy(dtype=float)
    z = np.clip((values - means) / denom, -20.0, 20.0)
    frame["z"] = z
    frame["abs_z"] = np.abs(z)
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=["z", "abs_z"])
    if frame.empty:
        return pd.DataFrame(), {"signal_rows": 0, "signal_objects": 0}

    idx = frame.groupby(["timestamp", "object_id"])["abs_z"].idxmax()
    dominant = frame.loc[idx, ["timestamp", "object_id", "z"]].copy()
    signal = dominant.pivot_table(index="timestamp", columns="object_id", values="z", aggfunc="mean")
    signal = signal.sort_index().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    for object_id in graph.nodes:
        if object_id not in signal.columns:
            signal[object_id] = 0.0
    signal = signal.loc[:, sorted(signal.columns)]
    return signal, {
        "signal_rows": int(len(signal)),
        "signal_objects": int(len(signal.columns)),
    }


def _candidate_cmi_features(
    query_id: str,
    query_index: int,
    task_index: str,
    telemetry_date: str,
    ground_truth: str,
    inject_time: float,
    ref: CandidateRef,
    graph: ObjectGraph,
    signal: pd.DataFrame,
    label: int,
    candidate_ids: Sequence[str],
    pair_cache: Dict[Tuple[str, str], Dict[str, float]],
    max_conditioners: int,
    max_effect_scope: int,
) -> Dict[str, Any]:
    object_id = ref.object_id
    node = graph.nodes[object_id]
    baseline_mask = signal.index.to_numpy(dtype=float) < inject_time
    fault_mask = signal.index.to_numpy(dtype=float) >= inject_time
    y = _series(signal, object_id)
    base_y = y[baseline_mask]
    fault_y = y[fault_mask]
    marginal_fault_energy = _mean_abs(fault_y)
    marginal_base_energy = _mean_abs(base_y)
    marginal_delta_z = abs(_safe_mean(fault_y) - _safe_mean(base_y)) / max(float(np.std(base_y)) if base_y.size else 0.0, 1.0)
    earliest_offset = 0.0
    if node.earliest_timestamp is not None:
        earliest_offset = _clip((float(node.earliest_timestamp) - inject_time) / 3600.0, -1.0, 1.0)

    conditioners = _conditioner_scope(graph, object_id, max_conditioners=max_conditioners)
    cond = _conditional_residual(signal, object_id, conditioners, baseline_mask, fault_mask)
    parent_explainability = _clip(
        (marginal_delta_z - cond["conditional_residual_z"]) / max(marginal_delta_z, 1e-6),
        0.0,
        1.0,
    )

    effect_scope = _effect_scope(graph, object_id, max_scope=max_effect_scope)
    effect = _intervention_effect_features(
        signal=signal,
        source=object_id,
        effect_scope=effect_scope,
        candidate_ids=candidate_ids,
        baseline_mask=baseline_mask,
        fault_mask=fault_mask,
        pair_cache=pair_cache,
    )

    degree_out = len(graph.adjacency.get(object_id, {}))
    degree_in = sum(1 for _, children in graph.adjacency.items() if object_id in children)
    event_factors = ref.event_factors or {}
    hotspot_proxy = max(
        _safe_float(event_factors.get("hotspot_symptom", 0.0)),
        _clip(parent_explainability * max(marginal_fault_energy, marginal_delta_z) / 4.0, 0.0, 1.0),
    )

    return {
        "system": "Bank" if query_id.startswith("Bank_") else "",
        "query_id": query_id,
        "query_index": int(query_index),
        "task_index": task_index,
        "telemetry_date": telemetry_date,
        "ground_truth": ground_truth,
        "label": int(label),
        "candidate_raw": ref.raw_name,
        "object_id": object_id,
        "representative": node.representative,
        "candidate_source": ref.source,
        "in_result_pool": 1 if ref.in_result_pool else 0,
        "current_rank": ref.current_rank,
        "current_score": ref.current_score,
        "current_posterior": ref.posterior,
        "current_root_probability": ref.root_probability,
        "current_status": ref.status,
        "event_factor_source_isolation": _safe_optional(event_factors.get("source_isolation")),
        "event_factor_hotspot_symptom": _safe_optional(event_factors.get("hotspot_symptom")),
        "event_factor_residual_collapse": _safe_optional(event_factors.get("residual_collapse")),
        "event_factor_counterfactual_likelihood": _safe_optional(event_factors.get("counterfactual_likelihood")),
        "graph_anomaly_score": float(node.anomaly_score),
        "metric_score": float(node.metric_score),
        "log_score": float(node.log_score),
        "trace_score": float(node.trace_score),
        "degree_in": float(degree_in),
        "degree_out": float(degree_out),
        "incoming_mass": float(graph.incoming_mass(object_id)),
        "topological_mass": float(graph.topological_mass(object_id)),
        "earliest_offset_hr": earliest_offset,
        "marginal_base_energy": marginal_base_energy,
        "marginal_fault_energy": marginal_fault_energy,
        "marginal_delta_z": float(marginal_delta_z),
        "conditioner_count": len(conditioners),
        "conditioners": "|".join(conditioners),
        "conditional_residual_z": cond["conditional_residual_z"],
        "conditional_fault_residual_energy": cond["fault_residual_energy"],
        "conditional_base_residual_std": cond["base_residual_std"],
        "conditional_model_r2": cond["baseline_r2"],
        "parent_explainability": parent_explainability,
        "effect_scope_size": len(effect_scope),
        "effect_scope": "|".join(item[0] for item in effect_scope),
        "effect_observer_energy": effect["observer_energy"],
        "counterfactual_effect_coverage": effect["effect_coverage"],
        "alternative_explainability": effect["alternative_explainability"],
        "repair_uniqueness": effect["repair_uniqueness"],
        "best_effect_observer": effect["best_observer"],
        "hotspot_symptom_proxy": hotspot_proxy,
    }


def _conditional_residual(
    signal: pd.DataFrame,
    target: str,
    conditioners: Sequence[str],
    baseline_mask: np.ndarray,
    fault_mask: np.ndarray,
) -> Dict[str, float]:
    y = _series(signal, target)
    X_cols = [cond for cond in conditioners if cond in signal.columns and cond != target]
    X = np.column_stack([_series(signal, cond) for cond in X_cols]) if X_cols else np.zeros((len(y), 0))
    lag_y = np.roll(y, 1)
    lag_y[0] = 0.0
    X = np.column_stack([X, lag_y])
    beta = _fit_ridge(X[baseline_mask], y[baseline_mask], alpha=1.0)
    if beta is None:
        residual = y - _safe_mean(y[baseline_mask])
        base_resid = residual[baseline_mask]
        fault_resid = residual[fault_mask]
        return {
            "conditional_residual_z": abs(_safe_mean(fault_resid) - _safe_mean(base_resid)) / max(float(np.std(base_resid)) if base_resid.size else 0.0, 1.0),
            "fault_residual_energy": _mean_abs(fault_resid),
            "base_residual_std": float(np.std(base_resid)) if base_resid.size else 0.0,
            "baseline_r2": 0.0,
        }
    pred = _predict_ridge(beta, X)
    residual = y - pred
    base_resid = residual[baseline_mask]
    fault_resid = residual[fault_mask]
    denom = max(float(np.std(base_resid)) if base_resid.size else 0.0, 1.0)
    cond_z = abs(_safe_mean(fault_resid) - _safe_mean(base_resid)) / denom
    y_base = y[baseline_mask]
    pred_base = pred[baseline_mask]
    sse = float(np.sum((y_base - pred_base) ** 2))
    sst = float(np.sum((y_base - _safe_mean(y_base)) ** 2))
    r2 = 1.0 - sse / max(sst, 1e-9)
    return {
        "conditional_residual_z": float(max(0.0, cond_z)),
        "fault_residual_energy": _mean_abs(fault_resid),
        "base_residual_std": float(np.std(base_resid)) if base_resid.size else 0.0,
        "baseline_r2": _clip(r2, -1.0, 1.0),
    }


def _intervention_effect_features(
    signal: pd.DataFrame,
    source: str,
    effect_scope: Sequence[Tuple[str, float]],
    candidate_ids: Sequence[str],
    baseline_mask: np.ndarray,
    fault_mask: np.ndarray,
    pair_cache: Dict[Tuple[str, str], Dict[str, float]],
) -> Dict[str, Any]:
    if not effect_scope:
        return {
            "observer_energy": 0.0,
            "effect_coverage": 0.0,
            "alternative_explainability": 0.0,
            "repair_uniqueness": 0.0,
            "best_observer": "",
        }
    total_energy = 0.0
    own_explained = 0.0
    alt_explained = 0.0
    best_observer = ""
    best_amount = -1.0
    for target, weight in effect_scope:
        if target not in signal.columns:
            continue
        target_energy = _mean_abs(_series(signal, target)[fault_mask])
        if target_energy <= 1e-9:
            continue
        own = _pair_effect(signal, source, target, baseline_mask, fault_mask, pair_cache)
        alt_amount = 0.0
        for alt in candidate_ids:
            if alt == source or alt == target or alt not in signal.columns:
                continue
            alt_payload = _pair_effect(signal, alt, target, baseline_mask, fault_mask, pair_cache)
            alt_amount = max(alt_amount, alt_payload["explained_amount"])
        weighted_energy = float(weight) * target_energy
        own_amount = float(weight) * own["explained_amount"]
        total_energy += weighted_energy
        own_explained += own_amount
        alt_explained += float(weight) * alt_amount
        if own_amount > best_amount:
            best_amount = own_amount
            best_observer = target
    if total_energy <= 1e-9:
        return {
            "observer_energy": 0.0,
            "effect_coverage": 0.0,
            "alternative_explainability": 0.0,
            "repair_uniqueness": 0.0,
            "best_observer": "",
        }
    effect_coverage = _clip(own_explained / total_energy, 0.0, 1.0)
    alternative = _clip(alt_explained / total_energy, 0.0, 1.0)
    relative_advantage = own_explained / max(own_explained + alt_explained, 1e-9)
    return {
        "observer_energy": float(total_energy),
        "effect_coverage": effect_coverage,
        "alternative_explainability": alternative,
        "repair_uniqueness": _clip(effect_coverage * relative_advantage, 0.0, 1.0),
        "best_observer": best_observer,
    }


def _pair_effect(
    signal: pd.DataFrame,
    source: str,
    target: str,
    baseline_mask: np.ndarray,
    fault_mask: np.ndarray,
    cache: Dict[Tuple[str, str], Dict[str, float]],
) -> Dict[str, float]:
    key = (source, target)
    if key in cache:
        return cache[key]
    if source not in signal.columns or target not in signal.columns or source == target:
        cache[key] = {"coverage": 0.0, "explained_amount": 0.0, "beta": 0.0}
        return cache[key]
    x = _series(signal, source)
    y = _series(signal, target)
    lag_y = np.roll(y, 1)
    lag_y[0] = 0.0
    X = np.column_stack([x, lag_y])
    beta = _fit_ridge(X[baseline_mask], y[baseline_mask], alpha=1.0)
    if beta is None:
        cache[key] = {"coverage": 0.0, "explained_amount": 0.0, "beta": 0.0}
        return cache[key]
    source_beta = float(beta[1])
    dx = _safe_mean(x[fault_mask]) - _safe_mean(x[baseline_mask])
    dy = _safe_mean(y[fault_mask]) - _safe_mean(y[baseline_mask])
    explained_delta = source_beta * dx
    if abs(dy) <= 1e-9 or explained_delta * dy <= 0:
        coverage = 0.0
    else:
        coverage = min(abs(explained_delta), abs(dy)) / max(abs(dy), 1.0)
    marginal = abs(_safe_mean(y[fault_mask]) - _safe_mean(y[baseline_mask])) / max(float(np.std(y[baseline_mask])) if y[baseline_mask].size else 0.0, 1.0)
    cond = _conditional_residual(signal, target, [source], baseline_mask, fault_mask)
    conditioned_collapse = _clip((marginal - cond["conditional_residual_z"]) / max(marginal, 1e-6), 0.0, 1.0)
    coverage = max(coverage, conditioned_collapse)
    explained_amount = coverage * _mean_abs(y[fault_mask])
    cache[key] = {
        "coverage": _clip(coverage, 0.0, 1.0),
        "explained_amount": float(max(0.0, explained_amount)),
        "beta": source_beta,
    }
    return cache[key]


def _score_and_rank_cmi(table: pd.DataFrame) -> pd.DataFrame:
    table = table.copy()
    norm_cols = [
        "marginal_delta_z",
        "marginal_fault_energy",
        "conditional_residual_z",
        "conditional_fault_residual_energy",
        "counterfactual_effect_coverage",
        "repair_uniqueness",
    ]
    for col in norm_cols:
        table[f"{col}_norm"] = 0.0
    table["temporal_lead_norm"] = 0.5
    table["cmi_root_admissible"] = 0

    chunks = []
    for _, group in table.groupby("query_id", sort=False):
        group = group.copy()
        for col in norm_cols:
            group[f"{col}_norm"] = _minmax(group[col].astype(float).to_numpy())
        earliest = group["earliest_offset_hr"].astype(float).to_numpy()
        if len(earliest) > 1 and np.nanmax(earliest) > np.nanmin(earliest):
            group["temporal_lead_norm"] = 1.0 - _minmax(earliest)
        else:
            group["temporal_lead_norm"] = 0.5
        cond_bar = max(float(group["conditional_residual_z_norm"].median()), 0.50)
        repair_bar = max(float(group["repair_uniqueness_norm"].median()), 0.35)
        group["cmi_root_admissible"] = (
            (group["conditional_residual_z_norm"] >= cond_bar)
            & (group["repair_uniqueness_norm"] >= repair_bar)
            & (group["parent_explainability"] <= 0.65)
            & (group["alternative_explainability"] <= 0.75)
        ).astype(int)
        chunks.append(group)
    table = pd.concat(chunks, ignore_index=True) if chunks else table
    mechanism_break = table["conditional_residual_z_norm"] * (1.0 - 0.6 * table["parent_explainability"].clip(0.0, 1.0))
    intervention_unique = table["repair_uniqueness_norm"]
    table["cmi_mechanism_break_score"] = mechanism_break
    table["cmi_intervention_uniqueness_score"] = intervention_unique
    table["cmi_score"] = (
        0.42 * table["conditional_residual_z_norm"]
        + 0.28 * intervention_unique
        + 0.18 * table["counterfactual_effect_coverage_norm"]
        + 0.10 * table["temporal_lead_norm"]
        + 0.08 * table["conditional_model_r2"].clip(0.0, 1.0)
        - 0.28 * table["parent_explainability"].clip(0.0, 1.0)
        - 0.22 * table["alternative_explainability"].clip(0.0, 1.0)
        - 0.10 * table["hotspot_symptom_proxy"].clip(0.0, 1.0)
    )
    table["cmi_rank_with_gt"] = table.groupby("query_id")["cmi_score"].rank(method="first", ascending=False).astype(int)
    table["cmi_rank_result_pool"] = np.nan
    pool = table[table["in_result_pool"] == 1].copy()
    if not pool.empty:
        ranks = pool.groupby("query_id")["cmi_score"].rank(method="first", ascending=False)
        table.loc[pool.index, "cmi_rank_result_pool"] = ranks
    table.sort_values(["query_index", "cmi_rank_with_gt", "current_rank"], inplace=True)
    return table


def summarize_table(table: pd.DataFrame, metadata: Dict[str, Any]) -> Dict[str, Any]:
    if table.empty:
        return {"num_rows": 0, "num_queries": 0}
    query_summaries: List[Dict[str, Any]] = []
    for query_id, group in table.groupby("query_id", sort=False):
        gt_rows = group[group["label"] == 1]
        pool_gt = gt_rows[gt_rows["in_result_pool"] == 1]
        current_top = group[group["current_rank"] == 1]
        current_top_correct = bool(not current_top.empty and int(current_top.iloc[0]["label"]) == 1)
        cmi_top = group.sort_values("cmi_rank_with_gt").head(1)
        cmi_pool = group[group["in_result_pool"] == 1].sort_values("cmi_rank_result_pool").head(1)
        gt_rank_current = _nullable_min(pool_gt["current_rank"]) if not pool_gt.empty else None
        gt_rank_cmi_with_gt = _nullable_min(gt_rows["cmi_rank_with_gt"]) if not gt_rows.empty else None
        gt_rank_cmi_pool = _nullable_min(pool_gt["cmi_rank_result_pool"]) if not pool_gt.empty else None
        query_summaries.append({
            "query_id": query_id,
            "task_index": str(group["task_index"].iloc[0]),
            "ground_truth": str(group["ground_truth"].iloc[0]),
            "gt_in_result_pool": bool(not pool_gt.empty),
            "current_top_correct": current_top_correct,
            "cmi_top_with_gt_correct": bool(not cmi_top.empty and int(cmi_top.iloc[0]["label"]) == 1),
            "cmi_top_pool_correct": bool(not cmi_pool.empty and int(cmi_pool.iloc[0]["label"]) == 1),
            "current_gt_rank": gt_rank_current,
            "cmi_gt_rank_with_gt": gt_rank_cmi_with_gt,
            "cmi_gt_rank_result_pool": gt_rank_cmi_pool,
        })
    qdf = pd.DataFrame(query_summaries)
    selected_wrong = table[(table["current_rank"] == 1) & (table["label"] == 0)]
    gt_rows = table[table["label"] == 1]
    summary = {
        "num_rows": int(len(table)),
        "num_queries": int(table["query_id"].nunique()),
        "gt_in_result_pool": int(qdf["gt_in_result_pool"].sum()) if not qdf.empty else 0,
        "current_top1_component_acc": _mean_bool(qdf["current_top_correct"]) if not qdf.empty else 0.0,
        "cmi_top1_with_gt_diagnostic_acc": _mean_bool(qdf["cmi_top_with_gt_correct"]) if not qdf.empty else 0.0,
        "cmi_top1_result_pool_acc": _mean_bool(qdf["cmi_top_pool_correct"]) if not qdf.empty else 0.0,
        "current_gt_rank_mean_observed_pool": _mean_non_null(qdf["current_gt_rank"]) if not qdf.empty else None,
        "cmi_gt_rank_mean_with_gt": _mean_non_null(qdf["cmi_gt_rank_with_gt"]) if not qdf.empty else None,
        "cmi_gt_rank_mean_result_pool": _mean_non_null(qdf["cmi_gt_rank_result_pool"]) if not qdf.empty else None,
        "gt_conditional_residual_z_mean": _mean_non_null(gt_rows["conditional_residual_z"]) if not gt_rows.empty else None,
        "gt_repair_uniqueness_mean": _mean_non_null(gt_rows["repair_uniqueness"]) if not gt_rows.empty else None,
        "selected_wrong_conditional_residual_z_mean": _mean_non_null(selected_wrong["conditional_residual_z"]) if not selected_wrong.empty else None,
        "selected_wrong_repair_uniqueness_mean": _mean_non_null(selected_wrong["repair_uniqueness"]) if not selected_wrong.empty else None,
        "elapsed_sec": metadata.get("elapsed_sec"),
    }
    return {"summary": summary, "per_query": query_summaries}


def _conditioner_scope(graph: ObjectGraph, object_id: str, max_conditioners: int) -> List[str]:
    node = graph.nodes[object_id]
    weights: Dict[str, float] = {}
    for child, weight in graph.adjacency.get(object_id, {}).items():
        if child in graph.nodes:
            weights[child] = max(weights.get(child, 0.0), 1.0 + min(1.0, float(weight)))
    incoming = _incoming_neighbors(graph, object_id)
    for parent, weight in incoming:
        if parent not in graph.nodes:
            continue
        parent_ts = graph.nodes[parent].earliest_timestamp
        node_ts = node.earliest_timestamp
        temporal_bonus = 0.45 if parent_ts is not None and node_ts is not None and parent_ts <= node_ts - 30.0 else 0.0
        if temporal_bonus <= 0.0:
            continue
        weights[parent] = max(weights.get(parent, 0.0), 0.45 + temporal_bonus + min(0.4, float(weight)))
    ranked = sorted(
        weights,
        key=lambda obj: (weights[obj], graph.nodes[obj].anomaly_score),
        reverse=True,
    )
    return [obj for obj in ranked if obj != object_id][:max_conditioners]


def _effect_scope(graph: ObjectGraph, object_id: str, max_scope: int) -> List[Tuple[str, float]]:
    weights: Dict[str, float] = {}
    for parent, weight in _incoming_neighbors(graph, object_id):
        weights[parent] = max(weights.get(parent, 0.0), 1.0 + min(1.0, float(weight)))
        for grandparent, gweight in _incoming_neighbors(graph, parent):
            if grandparent != object_id:
                weights[grandparent] = max(weights.get(grandparent, 0.0), 0.45 + 0.25 * min(1.0, float(gweight)))
    for child, weight in graph.adjacency.get(object_id, {}).items():
        if child != object_id and child in graph.nodes:
            weights[child] = max(weights.get(child, 0.0), 0.55 + 0.35 * min(1.0, float(weight)))
            for grandchild, gweight in graph.adjacency.get(child, {}).items():
                if grandchild != object_id and grandchild in graph.nodes:
                    weights[grandchild] = max(weights.get(grandchild, 0.0), 0.25 + 0.20 * min(1.0, float(gweight)))
    ranked = sorted(
        [(obj, weight) for obj, weight in weights.items() if obj in graph.nodes and obj != object_id],
        key=lambda item: (item[1] * max(graph.nodes[item[0]].anomaly_score, 0.05), item[1]),
        reverse=True,
    )
    return ranked[:max_scope]


def _incoming_neighbors(graph: ObjectGraph, object_id: str) -> List[Tuple[str, float]]:
    rows: List[Tuple[str, float]] = []
    for parent, children in graph.adjacency.items():
        if object_id in children:
            rows.append((parent, float(children[object_id])))
    return rows


def _resolve_object_id(raw_name: str, graph: ObjectGraph, system_name: str) -> Optional[str]:
    raw = str(raw_name or "").strip()
    if not raw:
        return None
    if raw in graph.nodes:
        return raw
    lowered = raw.lower()
    lower_map = {obj.lower(): obj for obj in graph.nodes}
    if lowered in lower_map:
        return lower_map[lowered]
    keep_instance_suffix = system_name in {"Telecom", "Market"} or bool(
        re.search(r"(?:^|[._-])[a-z]+[-_]\d+(?:$|[._-])", lowered)
    )
    canonical = _canonical_object_name(lowered, keep_instance_suffix=keep_instance_suffix)
    if canonical in graph.nodes:
        return canonical
    if canonical in lower_map:
        return lower_map[canonical]
    for object_id, node in graph.nodes.items():
        rep = str(node.representative or "").lower()
        members = {str(member).lower() for member in node.members}
        if lowered == rep or lowered in members:
            return object_id
        obj_can = _canonical_object_name(object_id, keep_instance_suffix=keep_instance_suffix)
        rep_can = _canonical_object_name(rep, keep_instance_suffix=keep_instance_suffix)
        if canonical in {obj_can, rep_can}:
            return object_id
    return None


def _matches_ground_truth(object_id: str, candidate_raw: str, ground_truth: str, system_name: str) -> bool:
    gt_raw = str(ground_truth or "").strip().lower()
    if not gt_raw:
        return False
    keep_instance_suffix = system_name in {"Telecom", "Market"} or bool(
        re.search(r"(?:^|[._-])[a-z]+[-_]\d+(?:$|[._-])", gt_raw)
    )
    gt_canonical = _canonical_object_name(gt_raw, keep_instance_suffix=keep_instance_suffix)
    cand_raw = str(candidate_raw or "").strip().lower()
    obj_raw = str(object_id or "").strip().lower()
    cand_can = _canonical_object_name(cand_raw, keep_instance_suffix=keep_instance_suffix)
    obj_can = _canonical_object_name(obj_raw, keep_instance_suffix=keep_instance_suffix)
    return (
        cand_raw == gt_raw
        or obj_raw == gt_raw
        or cand_can == gt_canonical
        or obj_can == gt_canonical
        or obj_can.startswith(gt_canonical + ".")
        or obj_can.startswith(gt_canonical + "-")
    )


def _series(signal: pd.DataFrame, column: str) -> np.ndarray:
    if column not in signal.columns:
        return np.zeros(len(signal), dtype=float)
    return pd.to_numeric(signal[column], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)


def _fit_ridge(X: np.ndarray, y: np.ndarray, alpha: float) -> Optional[np.ndarray]:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    mask = np.isfinite(y)
    if X.size:
        mask &= np.all(np.isfinite(X), axis=1)
    X = X[mask]
    y = y[mask]
    if len(y) < max(4, X.shape[1] + 2):
        return None
    design = np.column_stack([np.ones(len(y)), X])
    penalty = alpha * np.eye(design.shape[1])
    penalty[0, 0] = 0.0
    try:
        return np.linalg.solve(design.T @ design + penalty, design.T @ y)
    except np.linalg.LinAlgError:
        try:
            return np.linalg.lstsq(design.T @ design + penalty, design.T @ y, rcond=None)[0]
        except np.linalg.LinAlgError:
            return None


def _predict_ridge(beta: np.ndarray, X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    design = np.column_stack([np.ones(len(X)), X])
    return design @ beta


def _minmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = np.where(np.isfinite(values), values, 0.0)
    lo = float(np.min(values)) if values.size else 0.0
    hi = float(np.max(values)) if values.size else 0.0
    if hi <= lo + 1e-12:
        return np.zeros_like(values, dtype=float)
    return (values - lo) / (hi - lo)


def _safe_mean(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if values.size else 0.0


def _mean_abs(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.mean(np.abs(values))) if values.size else 0.0


def _clip(value: float, low: float, high: float) -> float:
    return float(max(low, min(high, value)))


def _is_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return 0.0
    return out if math.isfinite(out) else 0.0


def _safe_optional(value: Any) -> Optional[float]:
    if value is None:
        return None
    return _safe_float(value) if _is_number(value) else None


def _nullable_min(values: Iterable[Any]) -> Optional[float]:
    filtered = [_safe_float(value) for value in values if _is_number(value)]
    return float(min(filtered)) if filtered else None


def _mean_non_null(values: Iterable[Any]) -> Optional[float]:
    filtered = [_safe_float(value) for value in values if _is_number(value)]
    return round(float(np.mean(filtered)), 6) if filtered else None


def _mean_bool(values: Iterable[Any]) -> float:
    vals = [1.0 if bool(value) else 0.0 for value in values]
    return round(float(np.mean(vals)), 6) if vals else 0.0


def _write_outputs(table: pd.DataFrame, metadata: Dict[str, Any], output_dir: Path, variant: str, system: str) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    table_path = output_dir / f"cmi_validation_{variant}_{system}_{ts}.csv"
    summary_path = output_dir / f"cmi_validation_{variant}_{system}_{ts}.json"
    summary_payload = {
        "config": {key: value for key, value in metadata.items() if key != "per_query"},
        "diagnostics": summarize_table(table, metadata),
        "build_per_query": metadata.get("per_query", []),
    }
    table.to_csv(table_path, index=False)
    summary_path.write_text(json.dumps(summary_payload, ensure_ascii=True, indent=2))
    return {"table": str(table_path), "summary": str(summary_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an offline CMI validation table")
    parser.add_argument("--system", choices=list(SYSTEM_PATHS), default="Bank")
    parser.add_argument("--result-json", type=Path)
    parser.add_argument("--max-queries", type=int)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--baseline-window", type=int, default=600)
    parser.add_argument("--fault-window", type=int, default=600)
    parser.add_argument("--max-conditioners", type=int, default=6)
    parser.add_argument("--max-effect-scope", type=int, default=10)
    parser.add_argument("--telemetry-cache-size", type=int, default=8)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--variant", default="cmi_offline")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    table, metadata = build_cmi_table(
        system_name=args.system,
        result_json=args.result_json,
        max_queries=args.max_queries,
        top_k=args.top_k,
        baseline_window=args.baseline_window,
        fault_window=args.fault_window,
        max_conditioners=args.max_conditioners,
        max_effect_scope=args.max_effect_scope,
        telemetry_cache_size=args.telemetry_cache_size,
        progress_every=args.progress_every,
    )
    paths = _write_outputs(table, metadata, args.output_dir, args.variant, args.system)
    diagnostics = summarize_table(table, metadata)
    print(json.dumps(diagnostics["summary"], ensure_ascii=True, indent=2))
    print(f"table={paths['table']}")
    print(f"summary={paths['summary']}")


if __name__ == "__main__":
    main()
