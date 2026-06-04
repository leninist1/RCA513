"""Cross-service baseline scoring.

Historical baselines can be dirty or sparse. Cross-service scoring compares
services at the same timestamp for the same KPI, which helps distinguish a
global background shift from a service-local deviation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CrossServiceConfig:
    min_services: int = 5
    mad_eps: float = 1e-9
    clip_abs_z: float = 20.0


def cross_service_scores(metric_df: pd.DataFrame, config: CrossServiceConfig | None = None) -> pd.DataFrame:
    config = config or CrossServiceConfig()
    if metric_df is None or metric_df.empty:
        return pd.DataFrame(columns=["timestamp", "cmdb_id", "kpi_name", "value", "cross_z", "eligible", "peer_count"])
    df = metric_df[["timestamp", "cmdb_id", "kpi_name", "value"]].copy()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["timestamp", "cmdb_id", "kpi_name", "value"])
    rows = []
    for (timestamp, kpi), group in df.groupby(["timestamp", "kpi_name"], sort=False):
        values = group["value"].to_numpy(dtype=float)
        peer_count = int(np.isfinite(values).sum())
        if peer_count < config.min_services:
            for row in group.itertuples(index=False):
                rows.append({**row._asdict(), "cross_z": 0.0, "eligible": False, "peer_count": peer_count})
            continue
        median = float(np.nanmedian(values))
        mad = float(np.nanmedian(np.abs(values - median)))
        scale = mad
        if scale < config.mad_eps:
            q25, q75 = np.nanpercentile(values, [25, 75])
            scale = float(q75 - q25)
        if scale < config.mad_eps:
            scale = float(np.nanstd(values))
        if scale < config.mad_eps:
            for row in group.itertuples(index=False):
                rows.append({**row._asdict(), "cross_z": 0.0, "eligible": False, "peer_count": peer_count})
            continue
        for row in group.itertuples(index=False):
            z = float((float(row.value) - median) / scale)
            z = max(-config.clip_abs_z, min(config.clip_abs_z, z))
            rows.append({**row._asdict(), "cross_z": z, "eligible": True, "peer_count": peer_count})
    return pd.DataFrame(rows)


def component_cross_intensity(scored_df: pd.DataFrame, service: str, kpi_names: Iterable[str] | None = None) -> dict:
    if scored_df is None or scored_df.empty:
        return {"service": service, "max_abs_cross_z": 0.0, "vote_count": 0, "timestamps": []}
    rows = scored_df[scored_df["cmdb_id"].astype(str) == str(service)]
    if kpi_names is not None:
        names = {str(name) for name in kpi_names}
        rows = rows[rows["kpi_name"].astype(str).isin(names)]
    rows = rows[rows["eligible"].astype(bool)]
    if rows.empty:
        return {"service": service, "max_abs_cross_z": 0.0, "vote_count": 0, "timestamps": []}
    rows = rows.assign(abs_cross_z=rows["cross_z"].abs())
    by_ts = rows.groupby("timestamp")["abs_cross_z"].sum().sort_values(ascending=False)
    return {
        "service": str(service),
        "max_abs_cross_z": float(rows["abs_cross_z"].max()),
        "vote_count": int((rows["abs_cross_z"] >= 3.0).sum()),
        "timestamps": [int(ts) for ts in by_ts.head(5).index],
        "timestamp_scores": {str(int(ts)): float(score) for ts, score in by_ts.head(10).items()},
    }
