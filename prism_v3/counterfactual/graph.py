"""Dependency graph construction from traces or metric anomaly timing."""

from collections import defaultdict
from typing import Dict, List, Set
import numpy as np
import pandas as pd

from ..config import EDGE_IMPORTANCE_MIN_CALLS


def build_graph_from_traces(
    traces_df: pd.DataFrame,
    inject_time: float = None,
    max_traces: int = 5000,
) -> Dict[str, Dict[str, float]]:
    """Build weighted dependency graph from trace span data.

    Uses parent-child span relationships to infer call edges.
    Samples traces for efficiency with large datasets.
    """
    if traces_df is None or traces_df.empty:
        return {}

    required_cols = {"trace_id", "span_id", "entity"}
    if not required_cols.issubset(set(traces_df.columns)):
        return {}

    if "parent_id" not in traces_df.columns:
        return {}

    df = traces_df.copy()

    # Time filter: focus on traces near inject_time if provided
    if inject_time is not None and "timestamp" in df.columns:
        time_window = 600  # 10 min around inject
        df = df[
            (df["timestamp"] >= inject_time - time_window) &
            (df["timestamp"] <= inject_time + time_window)
        ]
        if df.empty:
            return {}

    # Sample trace IDs to limit processing
    all_trace_ids = df["trace_id"].dropna().unique()
    if len(all_trace_ids) > max_traces:
        sampled_ids = set(np.random.choice(all_trace_ids, size=max_traces, replace=False))
        df = df[df["trace_id"].isin(sampled_ids)]
    if df.empty:
        return {}

    edges = defaultdict(list)

    for trace_id, group in df.groupby("trace_id"):
        id_to_entity = {}
        for _, row in group.iterrows():
            sid = str(row["span_id"])
            ent = str(row["entity"])
            if sid:
                id_to_entity[sid] = ent

        for _, row in group.iterrows():
            pid = str(row.get("parent_id", ""))
            if not pid or pid == "nan" or pid not in id_to_entity:
                continue
            parent_ent = id_to_entity[pid]
            child_ent = str(row["entity"])
            if parent_ent == child_ent:
                continue
            dur = float(row.get("duration", 0)) if pd.notna(row.get("duration")) else 0
            is_err = 1 if _is_error_status(row) else 0
            edges[(parent_ent, child_ent)].append((dur, is_err))

    graph = defaultdict(dict)
    for (parent, child), records in edges.items():
        if len(records) < EDGE_IMPORTANCE_MIN_CALLS:
            continue
        durs = [r[0] for r in records]
        err_rate = np.mean([r[1] for r in records])
        weight = (np.percentile(durs, 95) / 1000.0 if durs else 0) + err_rate * 10
        if weight > 0:
            graph[parent][child] = weight

    return dict(graph)


def infer_graph_from_metrics(
    baseline_df: pd.DataFrame, fault_df: pd.DataFrame,
    entities: List[str], time_tolerance: int = 60
) -> Dict[str, Dict[str, float]]:
    """Infer dependency graph from anomaly onset timing when traces unavailable.

    Services that become anomalous in sequence (within tolerance) are linked.
    """
    if fault_df.empty:
        return {}

    anomaly_times = {}
    for entity in entities:
        entity_data = fault_df[fault_df["entity"] == entity]
        if entity_data.empty:
            continue
        for metric_name in entity_data["metric_name"].unique():
            metric_data = entity_data[entity_data["metric_name"] == metric_name]
            vals = metric_data["value"].dropna()
            if len(vals) < 2:
                continue
            median_val = np.median(vals)
            if median_val == 0:
                continue
            high_rows = metric_data[metric_data["value"] > median_val * 2]
            if not high_rows.empty:
                anomaly_times[entity] = high_rows["timestamp"].min()
                break

    if len(anomaly_times) < 2:
        return {}

    sorted_entities = sorted(anomaly_times, key=anomaly_times.get)
    graph = defaultdict(dict)
    for i, up in enumerate(sorted_entities):
        for down in sorted_entities[i + 1:]:
            if anomaly_times[down] - anomaly_times[up] <= time_tolerance:
                graph[up][down] = 1.0
    return dict(graph)


def expand_candidates_upstream(
    candidates: List[str], graph: Dict[str, Dict[str, float]],
    all_entities: List[str]
) -> List[str]:
    """Expand candidate set to include upstream callers."""
    if not graph:
        return candidates

    reverse_graph = defaultdict(set)
    for parent in graph:
        for child in graph[parent]:
            reverse_graph[child].add(parent)

    expanded = set(candidates)
    for svc in candidates:
        if svc in reverse_graph:
            for upstream in reverse_graph[svc]:
                if upstream in all_entities:
                    expanded.add(upstream)

    return list(expanded)


def _is_error_status(row) -> bool:
    """Check if a trace span has error status."""
    if "status_code" not in row:
        return False
    sc = str(row["status_code"])
    return sc.startswith(("5", "4")) or sc.lower() in ("error", "failed")
