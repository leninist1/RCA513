"""
cross_sectional_baseline.py -- Phase 1 / B-L1.3

Compute same-timestamp, same-KPI robust deviation across services.

This is a contrastive baseline: it does not assume the historical window is
healthy. A KPI is scored only when enough services report the same KPI at the
same timestamp; otherwise the safe default is to leave it ineligible.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

try:
    from refute.src.node_container_split import classify_kpi
except ImportError:  # pragma: no cover
    from node_container_split import classify_kpi


DEFAULT_MIN_SERVICES = 5
DEFAULT_MAD_EPS = 1e-9


@dataclass(frozen=True)
class CrossSectionalConfig:
    dataset: str = "Bank"
    include_node_level: bool = False
    min_services: int = DEFAULT_MIN_SERVICES
    mad_eps: float = DEFAULT_MAD_EPS


def _median_abs_deviation(values: np.ndarray, median: float) -> float:
    return float(np.median(np.abs(values - median)))


def _filter_kpi_level(df: pd.DataFrame, config: CrossSectionalConfig) -> pd.DataFrame:
    if config.include_node_level:
        return df
    keep = df["kpi_name"].map(lambda k: classify_kpi(str(k), dataset=config.dataset) == "container")
    return df[keep]


def compute_cross_sectional_scores(
    df: pd.DataFrame,
    config: Optional[CrossSectionalConfig] = None,
) -> pd.DataFrame:
    required = {"timestamp", "cmdb_id", "kpi_name", "value"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")

    config = config or CrossSectionalConfig()
    clean = df.dropna(subset=["timestamp", "cmdb_id", "kpi_name", "value"]).copy()
    clean["cmdb_id"] = clean["cmdb_id"].astype(str)
    clean["kpi_name"] = clean["kpi_name"].astype(str)
    clean["value"] = pd.to_numeric(clean["value"], errors="coerce")
    clean = clean.dropna(subset=["value"])
    clean = _filter_kpi_level(clean, config)

    if clean.empty:
        return pd.DataFrame(columns=[
            "timestamp", "cmdb_id", "kpi_name", "value", "n_services",
            "median", "mad", "z", "eligible", "reason",
        ])

    keys = ["timestamp", "kpi_name"]
    per_service = (
        clean.groupby(keys + ["cmdb_id"], as_index=False, sort=True)["value"]
        .mean()
    )
    grouped = per_service.groupby(keys, sort=False)["value"]
    per_service["n_services"] = grouped.transform("count").astype(int)
    per_service["median"] = grouped.transform("median")
    per_service["_abs_dev"] = (per_service["value"] - per_service["median"]).abs()
    per_service["mad"] = (
        per_service.groupby(keys, sort=False)["_abs_dev"]
        .transform("median")
    )
    per_service["eligible"] = (
        (per_service["n_services"] >= config.min_services)
        & (per_service["mad"] >= config.mad_eps)
    )
    per_service["reason"] = "ok"
    per_service.loc[per_service["n_services"] < config.min_services, "reason"] = "insufficient_services"
    per_service.loc[
        (per_service["n_services"] >= config.min_services)
        & (per_service["mad"] < config.mad_eps),
        "reason",
    ] = "near_constant_mad"
    per_service["z"] = 0.0
    eligible = per_service["eligible"]
    per_service.loc[eligible, "z"] = (
        (per_service.loc[eligible, "value"] - per_service.loc[eligible, "median"])
        / per_service.loc[eligible, "mad"]
    )

    return per_service[[
        "timestamp", "cmdb_id", "kpi_name", "value", "n_services",
        "median", "mad", "z", "eligible", "reason",
    ]]


def summarize_scores(scores: pd.DataFrame) -> dict:
    if scores.empty:
        return {"rows": 0, "eligible_rows": 0, "groups": 0, "eligible_groups": 0, "reasons": {}}
    group_cols = ["timestamp", "kpi_name"]
    groups = scores.drop_duplicates(group_cols)
    reasons: Dict[str, int] = {}
    for reason, count in groups["reason"].value_counts().items():
        reasons[str(reason)] = int(count)
    return {
        "rows": int(len(scores)),
        "eligible_rows": int(scores["eligible"].sum()),
        "groups": int(len(groups)),
        "eligible_groups": int(groups["eligible"].sum()),
        "reasons": reasons,
    }
