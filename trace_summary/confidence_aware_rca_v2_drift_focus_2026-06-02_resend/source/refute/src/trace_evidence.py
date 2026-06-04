"""
trace_evidence.py -- bounded trace evidence for network cases.

Trace files are large, so callers should pass a window-filtered dataframe built
with chunked loading. This module only works on that bounded window.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TraceEvidenceResult:
    matched: bool
    evidence: str
    strength: float = 0.0
    details: Optional[dict] = None


def normalize_trace_df(trace_df: pd.DataFrame) -> pd.DataFrame:
    if trace_df is None or trace_df.empty:
        return pd.DataFrame(columns=["timestamp", "cmdb_id", "parent_id", "span_id", "trace_id", "duration"])
    df = trace_df.copy()
    for col in ["cmdb_id", "parent_id", "span_id", "trace_id"]:
        df[col] = df[col].astype(str)
    df["duration"] = pd.to_numeric(df["duration"], errors="coerce")
    df = df.dropna(subset=["cmdb_id", "span_id", "trace_id", "duration"])
    return df


def service_duration_summary(trace_df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_trace_df(trace_df)
    if df.empty:
        return pd.DataFrame(columns=["cmdb_id", "count", "p50", "p95", "max"])
    grouped = df.groupby("cmdb_id")["duration"]
    out = grouped.agg(count="count", p50="median", max="max").reset_index()
    out["p95"] = grouped.quantile(0.95).values
    return out[["cmdb_id", "count", "p50", "p95", "max"]]


def service_trace_anomaly(trace_df: pd.DataFrame, svc: str, min_spans: int = 5) -> TraceEvidenceResult:
    summary = service_duration_summary(trace_df)
    if summary.empty:
        return TraceEvidenceResult(False, "no trace rows in window", 0.0)
    svc_rows = summary[summary["cmdb_id"] == str(svc)]
    if svc_rows.empty:
        return TraceEvidenceResult(False, f"no trace rows for {svc}", 0.0)
    row = svc_rows.iloc[0]
    if int(row["count"]) < min_spans:
        return TraceEvidenceResult(False, f"insufficient trace spans for {svc}: {int(row['count'])}", 0.0)
    vals = summary[summary["count"] >= min_spans]["p95"].to_numpy(dtype=float)
    if len(vals) < 3:
        return TraceEvidenceResult(False, "insufficient peer services for trace comparison", 0.0)
    median = float(np.median(vals))
    mad = float(np.median(np.abs(vals - median)))
    if mad < 1e-9:
        return TraceEvidenceResult(False, "near-constant trace duration baseline", 0.0)
    z = float((float(row["p95"]) - median) / mad)
    if z <= 3.0:
        return TraceEvidenceResult(False, f"trace p95 not high for {svc}: z={z:.2f}", abs(z), row.to_dict())
    return TraceEvidenceResult(
        True,
        f"{svc} trace p95 duration high: p95={float(row['p95']):.2f}, peer_median={median:.2f}, z={z:.2f}",
        z,
        row.to_dict(),
    )


def build_trace_edges(trace_df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_trace_df(trace_df)
    if df.empty:
        return pd.DataFrame(columns=["trace_id", "src", "dst", "child_duration", "child_timestamp"])
    parents = df[["trace_id", "span_id", "cmdb_id"]].rename(columns={
        "span_id": "parent_id",
        "cmdb_id": "src",
    })
    children = df[["trace_id", "parent_id", "cmdb_id", "duration", "timestamp"]].rename(columns={
        "cmdb_id": "dst",
        "duration": "child_duration",
        "timestamp": "child_timestamp",
    })
    edges = children.merge(parents, on=["trace_id", "parent_id"], how="inner")
    edges = edges[edges["src"] != edges["dst"]]
    return edges[["trace_id", "src", "dst", "child_duration", "child_timestamp"]]


def slow_edges_for_service(trace_df: pd.DataFrame, svc: str, min_edges: int = 3) -> TraceEvidenceResult:
    edges = build_trace_edges(trace_df)
    if edges.empty:
        return TraceEvidenceResult(False, "no cross-service trace edges in window", 0.0)
    svc_edges = edges[(edges["src"] == str(svc)) | (edges["dst"] == str(svc))]
    if len(svc_edges) < min_edges:
        return TraceEvidenceResult(False, f"insufficient trace edges for {svc}: {len(svc_edges)}", 0.0)
    edge_summary = (
        svc_edges.groupby(["src", "dst"])["child_duration"]
        .agg(count="count", p95=lambda s: float(np.percentile(s, 95)), max="max")
        .reset_index()
    )
    all_summary = (
        edges.groupby(["src", "dst"])["child_duration"]
        .agg(count="count", p95=lambda s: float(np.percentile(s, 95)))
        .reset_index()
    )
    peers = all_summary[all_summary["count"] >= min_edges]["p95"].to_numpy(dtype=float)
    if len(peers) < 3:
        return TraceEvidenceResult(False, "insufficient peer edges for trace comparison", 0.0)
    median = float(np.median(peers))
    mad = float(np.median(np.abs(peers - median)))
    if mad < 1e-9:
        return TraceEvidenceResult(False, "near-constant edge duration baseline", 0.0)
    edge_summary["z"] = (edge_summary["p95"] - median) / mad
    best = edge_summary.sort_values("z", ascending=False).iloc[0]
    if float(best["z"]) <= 3.0:
        return TraceEvidenceResult(False, f"no slow edge involving {svc}: best_z={float(best['z']):.2f}", abs(float(best["z"])), best.to_dict())
    return TraceEvidenceResult(
        True,
        f"slow trace edge {best['src']}->{best['dst']}: p95={float(best['p95']):.2f}, z={float(best['z']):.2f}",
        float(best["z"]),
        best.to_dict(),
    )
