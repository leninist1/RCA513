"""Bounded trace summary primitives for the refactored network evidence layer."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class TraceSummaryConfig:
    slow_edge_ratio: float = 3.0
    edge_drop_threshold: float = 0.5
    min_edge_count: int = 3


def summarize_trace_window(trace_df: pd.DataFrame, baseline_df: pd.DataFrame | None = None,
                           config: TraceSummaryConfig | None = None) -> dict[str, Any]:
    config = config or TraceSummaryConfig()
    current = _normalize(trace_df)
    baseline = _normalize(baseline_df) if baseline_df is not None else pd.DataFrame(columns=current.columns)
    service_stats = _service_stats(current)
    edge_stats = _edge_stats(current, baseline, config)
    slow_edges = [edge for edge in edge_stats if edge.get("slow_ratio", 0.0) >= config.slow_edge_ratio]
    dropped_edges = [edge for edge in edge_stats if edge.get("count_drop_ratio", 0.0) >= config.edge_drop_threshold]
    first_service = _first_anomalous_service(current, slow_edges)
    return {
        "service_stats": service_stats,
        "edge_stats": edge_stats,
        "events": {
            "slow_edges": slow_edges,
            "dropped_edges": dropped_edges,
            "first_anomalous_service": first_service,
        },
    }


def _normalize(df: pd.DataFrame | None) -> pd.DataFrame:
    cols = ["timestamp", "cmdb_id", "parent_id", "span_id", "trace_id", "duration"]
    if df is None or df.empty:
        return pd.DataFrame(columns=cols)
    out = df.copy()
    for col in cols:
        if col not in out.columns:
            out[col] = None
    for col in ["cmdb_id", "parent_id", "span_id", "trace_id"]:
        out[col] = out[col].astype(str)
    out["duration"] = pd.to_numeric(out["duration"], errors="coerce")
    out["timestamp"] = pd.to_numeric(out["timestamp"], errors="coerce")
    return out.dropna(subset=["timestamp", "cmdb_id", "span_id", "trace_id", "duration"])[cols]


def _service_stats(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    if df.empty:
        return {}
    grouped = df.groupby("cmdb_id")["duration"]
    out = {}
    for service, values in grouped:
        out[str(service)] = {
            "span_count": int(values.count()),
            "duration_p50": float(values.quantile(0.50)),
            "duration_p95": float(values.quantile(0.95)),
            "duration_p99": float(values.quantile(0.99)),
        }
    return out


def _trace_edges(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["trace_id", "src", "dst", "child_duration", "child_timestamp"])
    parents = df[["trace_id", "span_id", "cmdb_id"]].rename(columns={"span_id": "parent_id", "cmdb_id": "src"})
    children = df[["trace_id", "parent_id", "cmdb_id", "duration", "timestamp"]].rename(columns={
        "cmdb_id": "dst",
        "duration": "child_duration",
        "timestamp": "child_timestamp",
    })
    edges = children.merge(parents, on=["trace_id", "parent_id"], how="inner")
    edges = edges[edges["src"] != edges["dst"]]
    return edges[["trace_id", "src", "dst", "child_duration", "child_timestamp"]]


def _edge_stats(current: pd.DataFrame, baseline: pd.DataFrame, config: TraceSummaryConfig) -> list[dict[str, Any]]:
    cur_edges = _trace_edges(current)
    base_edges = _trace_edges(baseline)
    if cur_edges.empty and base_edges.empty:
        return []
    cur = _aggregate_edges(cur_edges)
    base = _aggregate_edges(base_edges)
    keys = sorted(set(cur) | set(base))
    rows = []
    for key in keys:
        src, dst = key
        c = cur.get(key, {"count": 0, "duration_p95": 0.0})
        b = base.get(key, {"count": 0, "duration_p95": 0.0})
        baseline_count = int(b["count"])
        current_count = int(c["count"])
        drop = 0.0
        if baseline_count >= config.min_edge_count:
            drop = max(0.0, (baseline_count - current_count) / baseline_count)
        baseline_p95 = float(b["duration_p95"])
        current_p95 = float(c["duration_p95"])
        slow_ratio = current_p95 / baseline_p95 if baseline_p95 > 0 else 0.0
        rows.append({
            "src": src,
            "dst": dst,
            "count": current_count,
            "baseline_count": baseline_count,
            "count_drop_ratio": float(drop),
            "duration_p95": current_p95,
            "baseline_duration_p95": baseline_p95,
            "slow_ratio": float(slow_ratio),
            "first_timestamp": c.get("first_timestamp"),
        })
    return rows


def _aggregate_edges(edges: pd.DataFrame) -> dict[tuple[str, str], dict[str, float]]:
    if edges.empty:
        return {}
    out = {}
    grouped = edges.groupby(["src", "dst"])["child_duration"]
    first_ts = edges.groupby(["src", "dst"])["child_timestamp"].min()
    for key, values in grouped:
        out[(str(key[0]), str(key[1]))] = {
            "count": int(values.count()),
            "duration_p95": float(values.quantile(0.95)),
            "first_timestamp": int(first_ts.loc[key]),
        }
    return out


def _first_anomalous_service(df: pd.DataFrame, slow_edges: list[dict[str, Any]]) -> str | None:
    if df.empty or not slow_edges:
        return None
    services = {edge["src"] for edge in slow_edges} | {edge["dst"] for edge in slow_edges}
    rows = df[df["cmdb_id"].isin(services)].sort_values("timestamp")
    if rows.empty:
        return None
    return str(rows.iloc[0]["cmdb_id"])
