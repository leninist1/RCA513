"""Dataset-portable incident schema for D32-style RCA.

This module is intentionally independent from the OpenRCA runner.  It provides
the normalized boundary that Eadro, AIOps2021, and future datasets can target
without changing the existing OpenRCA code path.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Protocol

import pandas as pd


METRIC_COLUMNS = ("timestamp", "cmdb_id", "kpi_name", "value")
LOG_COLUMNS = ("timestamp", "cmdb_id", "value")
# Optional precomputed per-row reason-bucket assignment (a frozenset[str] of
# bucket names).  Populated by dataset adapters that supply a dataset-native
# KPI->bucket mapping (e.g. AIOps2021 via portable_bucket_assignment).  When
# absent, the algorithm-side bucketing functions fall back to OpenRCA's
# original token matching, so the OpenRCA dataset path stays byte-identical.
BUCKETS_COLUMN = "buckets"


@dataclass(frozen=True)
class NormalizedIncident:
    dataset: str
    case_id: str
    window_start_ts: int
    window_end_ts: int
    metric_df: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=METRIC_COLUMNS))
    log_df: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=LOG_COLUMNS))
    trace_summary: Mapping[str, Any] | None = None
    topology: Mapping[str, Any] = field(default_factory=dict)
    modal_status: Mapping[str, str] = field(default_factory=dict)
    labels: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def d32_inputs(self) -> dict[str, Any]:
        """Return the input shape expected by the current D32 pipeline."""

        return {
            "metric_df": self.metric_df.copy(),
            "log_df": self.log_df.copy(),
            "trace_summary": dict(self.trace_summary or {}),
            "modal_status": dict(self.modal_status),
            "window_start_ts": int(self.window_start_ts),
        }


class DatasetAdapter(Protocol):
    """Small protocol for dataset-specific loaders.

    Adapters should do only schema conversion and dataset-local parsing.  They
    must not inject ground-truth labels into candidate generation or scoring.
    """

    dataset: str

    def iter_incidents(self) -> Iterable[NormalizedIncident]:
        ...


def normalize_metric_frame(
    frame: pd.DataFrame | None,
    *,
    column_map: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Normalize common metric table schemas to D32 columns.

    Supported source aliases cover OpenRCA-style tables plus common Eadro and
    AIOps challenge variants such as ``service``/``instance`` for component,
    and ``metric``/``name`` for KPI name.
    """

    if frame is None or frame.empty:
        return pd.DataFrame(columns=METRIC_COLUMNS)
    df = frame.copy()
    df = _rename_columns(df, column_map or {}, {
        "timestamp": ("timestamp", "time", "ts", "start_time", "datetime"),
        "cmdb_id": ("cmdb_id", "service", "service_name", "instance", "pod", "container", "node"),
        "kpi_name": ("kpi_name", "metric", "metric_name", "name", "kpi"),
        "value": ("value", "val"),
    })
    _require_columns(df, METRIC_COLUMNS, "metric")
    keep = list(METRIC_COLUMNS)
    if BUCKETS_COLUMN in df.columns:
        keep.append(BUCKETS_COLUMN)
    out = df.loc[:, keep].copy()
    out["timestamp"] = _normalize_timestamp_series(out["timestamp"])
    out["cmdb_id"] = out["cmdb_id"].astype(str)
    out["kpi_name"] = out["kpi_name"].astype(str)
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    if BUCKETS_COLUMN in out.columns:
        out[BUCKETS_COLUMN] = out[BUCKETS_COLUMN].map(_normalize_buckets_cell)
    out = out.dropna(subset=["timestamp", "cmdb_id", "kpi_name", "value"])
    out["timestamp"] = out["timestamp"].astype("int64")
    return out.sort_values(["timestamp", "cmdb_id", "kpi_name"], kind="mergesort").reset_index(drop=True)


def normalize_log_frame(
    frame: pd.DataFrame | None,
    *,
    column_map: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Normalize common log table schemas to D32 columns."""

    if frame is None or frame.empty:
        return pd.DataFrame(columns=LOG_COLUMNS)
    df = frame.copy()
    df = _rename_columns(df, column_map or {}, {
        "timestamp": ("timestamp", "time", "ts", "datetime"),
        "cmdb_id": ("cmdb_id", "service", "service_name", "instance", "pod", "container", "node"),
        "value": ("value", "message", "msg", "log", "content", "template"),
    })
    _require_columns(df, LOG_COLUMNS, "log")
    out = df.loc[:, list(LOG_COLUMNS)].copy()
    out["timestamp"] = _normalize_timestamp_series(out["timestamp"])
    out["cmdb_id"] = out["cmdb_id"].astype(str)
    out["value"] = out["value"].astype(str)
    out = out.dropna(subset=["timestamp", "cmdb_id", "value"])
    out["timestamp"] = out["timestamp"].astype("int64")
    return out.sort_values(["timestamp", "cmdb_id"], kind="mergesort").reset_index(drop=True)


def normalize_modal_status(
    *,
    metric_df: pd.DataFrame | None,
    log_df: pd.DataFrame | None,
    trace_summary: Mapping[str, Any] | None,
    enabled: Iterable[str] = ("metric", "log", "trace"),
) -> dict[str, str]:
    enabled_set = {str(item) for item in enabled}
    return {
        "metric": _status_from_frame(metric_df) if "metric" in enabled_set else "disabled",
        "log": _status_from_frame(log_df) if "log" in enabled_set else "disabled",
        "trace": str((trace_summary or {}).get("trace_status", "unloaded")) if "trace" in enabled_set else "disabled",
    }


def make_incident(
    *,
    dataset: str,
    case_id: str,
    window_start_ts: int,
    window_end_ts: int,
    metric_df: pd.DataFrame | None = None,
    log_df: pd.DataFrame | None = None,
    trace_summary: Mapping[str, Any] | None = None,
    topology: Mapping[str, Any] | None = None,
    labels: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
    enabled_modalities: Iterable[str] = ("metric", "log", "trace"),
) -> NormalizedIncident:
    metrics = normalize_metric_frame(metric_df)
    logs = normalize_log_frame(log_df)
    status = normalize_modal_status(
        metric_df=metrics,
        log_df=logs,
        trace_summary=trace_summary,
        enabled=enabled_modalities,
    )
    return NormalizedIncident(
        dataset=str(dataset),
        case_id=str(case_id),
        window_start_ts=int(window_start_ts),
        window_end_ts=int(window_end_ts),
        metric_df=metrics,
        log_df=logs,
        trace_summary=dict(trace_summary or {}),
        topology=dict(topology or {}),
        modal_status=status,
        labels=dict(labels or {}),
        metadata=dict(metadata or {}),
    )


def _rename_columns(df: pd.DataFrame, explicit: Mapping[str, str], aliases: Mapping[str, tuple[str, ...]]) -> pd.DataFrame:
    rename: dict[str, str] = {}
    for target, source in explicit.items():
        if source in df.columns:
            rename[str(source)] = str(target)
    lower_to_original = {str(col).lower(): col for col in df.columns}
    for target, candidates in aliases.items():
        if target in df.columns or target in rename.values():
            continue
        for candidate in candidates:
            original = lower_to_original.get(candidate.lower())
            if original is not None:
                rename[str(original)] = target
                break
    return df.rename(columns=rename)


def _require_columns(df: pd.DataFrame, columns: tuple[str, ...], table_name: str) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"{table_name} frame missing required normalized columns: {missing}")


def _normalize_timestamp_series(values: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(values):
        return (values.astype("int64") // 1_000_000_000).astype("int64")
    parsed = pd.to_numeric(values, errors="coerce")
    if parsed.notna().any():
        finite = parsed.dropna()
        median = float(finite.abs().median()) if not finite.empty else 0.0
        # Millisecond epoch timestamps are common in AIOps data exports.
        if median > 10_000_000_000:
            parsed = parsed / 1000.0
        return parsed.round().astype("Int64")
    dt = pd.to_datetime(values, errors="coerce")
    return (dt.astype("int64") // 1_000_000_000).astype("Int64")


def _status_from_frame(df: pd.DataFrame | None) -> str:
    if df is None:
        return "missing"
    if df.empty:
        return "empty_window"
    return "present"


def _normalize_buckets_cell(value: Any) -> frozenset[str]:
    """Coerce a buckets cell to a frozenset[str].

    Accepts None/NaN (-> empty), a single bucket string, or any iterable of
    bucket strings.  Returns an empty frozenset for non-fault / excluded KPIs.
    """
    if value is None:
        return frozenset()
    if isinstance(value, str):
        stripped = value.strip()
        return frozenset({stripped}) if stripped else frozenset()
    try:
        return frozenset(str(item) for item in value)
    except TypeError:
        return frozenset()
