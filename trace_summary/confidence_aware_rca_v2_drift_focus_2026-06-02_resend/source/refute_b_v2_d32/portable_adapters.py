"""Reusable adapters for tabular RCA benchmark datasets.

The adapters here are intentionally conservative: they normalize telemetry and
case windows, but they do not use labels while constructing D32 inputs.  Dataset
specific loaders for Eadro/AIOps2021 can subclass or wrap these helpers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

import pandas as pd

from refute_b_v2_d32.portable_schema import (
    BUCKETS_COLUMN,
    DatasetAdapter,
    NormalizedIncident,
    make_incident,
    normalize_log_frame,
    normalize_metric_frame,
)
from refute_b_v2_d32.portable_kpi_canonicalizer import canonicalize_metric_kpis
from refute_b_v2_d32.portable_bucket_assignment import assign_kpi_buckets


@dataclass(frozen=True)
class TabularCaseSpec:
    case_id_col: str = "case_id"
    start_ts_col: str = "start_ts"
    end_ts_col: str = "end_ts"
    label_cols: tuple[str, ...] = ()
    metadata_cols: tuple[str, ...] = ()


@dataclass
class TabularIncidentAdapter(DatasetAdapter):
    dataset: str
    cases: pd.DataFrame
    metrics: pd.DataFrame | None = None
    logs: pd.DataFrame | None = None
    trace_summaries: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    topology: Mapping[str, Any] = field(default_factory=dict)
    case_spec: TabularCaseSpec = field(default_factory=TabularCaseSpec)
    metric_column_map: Mapping[str, str] = field(default_factory=dict)
    log_column_map: Mapping[str, str] = field(default_factory=dict)
    enabled_modalities: tuple[str, ...] = ("metric", "log", "trace")
    kpi_canonicalization: str = "none"
    # When True, attach a precomputed dataset-native ``buckets`` column to each
    # metric row (via portable_bucket_assignment) so the algorithm reads the
    # dataset-native KPI->bucket mapping instead of OpenRCA token matching.
    # OpenRCA-native datasets leave this False and stay byte-identical.
    native_bucket_assignment: bool = False

    def iter_incidents(self) -> Iterable[NormalizedIncident]:
        cases = self.cases.copy()
        spec = self.case_spec
        for col in (spec.case_id_col, spec.start_ts_col, spec.end_ts_col):
            if col not in cases.columns:
                raise ValueError(f"cases frame missing required column: {col}")

        metrics = canonicalize_metric_kpis(
            normalize_metric_frame(self.metrics, column_map=self.metric_column_map),
            dataset=self.dataset,
            mode=self.kpi_canonicalization,
        )
        if self.native_bucket_assignment:
            metrics = _attach_native_buckets(metrics)
        logs = normalize_log_frame(self.logs, column_map=self.log_column_map)
        for row in cases.itertuples(index=False):
            case = _case_row_dict(row, cases.columns)
            case_id = str(case[spec.case_id_col])
            start_ts = _to_int_ts(case[spec.start_ts_col])
            end_ts = _to_int_ts(case[spec.end_ts_col])
            labels = {col: case[col] for col in spec.label_cols if col in case}
            metadata = {col: case[col] for col in spec.metadata_cols if col in case}
            yield make_incident(
                dataset=self.dataset,
                case_id=case_id,
                window_start_ts=start_ts,
                window_end_ts=end_ts,
                metric_df=_slice_window(metrics, start_ts, end_ts),
                log_df=_slice_window(logs, start_ts, end_ts),
                trace_summary=self.trace_summaries.get(case_id, {}),
                topology=self.topology,
                labels=labels,
                metadata=metadata,
                enabled_modalities=self.enabled_modalities,
            )


def eadro_tabular_adapter(
    *,
    cases: pd.DataFrame,
    metrics: pd.DataFrame | None = None,
    logs: pd.DataFrame | None = None,
    trace_summaries: Mapping[str, Mapping[str, Any]] | None = None,
    topology: Mapping[str, Any] | None = None,
    kpi_canonicalization: str = "none",
) -> TabularIncidentAdapter:
    """Create a schema-tolerant adapter for Eadro-like benchmark exports."""

    return TabularIncidentAdapter(
        dataset="eadro",
        cases=cases,
        metrics=metrics,
        logs=logs,
        trace_summaries=dict(trace_summaries or {}),
        topology=dict(topology or {}),
        kpi_canonicalization=kpi_canonicalization,
        case_spec=TabularCaseSpec(
            case_id_col=_first_existing(cases, ("case_id", "incident_id", "failure_id")),
            start_ts_col=_first_existing(cases, ("start_ts", "start_time", "timestamp")),
            end_ts_col=_first_existing(cases, ("end_ts", "end_time", "timestamp_end")),
            label_cols=("root_cause", "root_cause_service", "root_cause_component", "failure_type"),
            metadata_cols=("system", "scenario", "fault_type", "source_file", "fault_index"),
        ),
    )


def aiops2021_tabular_adapter(
    *,
    cases: pd.DataFrame,
    metrics: pd.DataFrame | None = None,
    logs: pd.DataFrame | None = None,
    trace_summaries: Mapping[str, Mapping[str, Any]] | None = None,
    topology: Mapping[str, Any] | None = None,
    kpi_canonicalization: str = "none",
) -> TabularIncidentAdapter:
    """Create a schema-tolerant adapter for AIOps2021-like exports."""

    return TabularIncidentAdapter(
        dataset="aiops2021",
        cases=cases,
        metrics=metrics,
        logs=logs,
        trace_summaries=dict(trace_summaries or {}),
        topology=dict(topology or {}),
        kpi_canonicalization=kpi_canonicalization,
        native_bucket_assignment=True,
        case_spec=TabularCaseSpec(
            case_id_col=_first_existing(cases, ("case_id", "故障编号", "incident_id", "id")),
            start_ts_col=_first_existing(cases, ("start_ts", "start_time", "故障开始时间", "timestamp")),
            end_ts_col=_first_existing(cases, ("end_ts", "end_time", "故障结束时间", "timestamp_end")),
            label_cols=("root_cause", "root_cause_service", "root_cause_component", "cmdb_id", "failure_type", "故障类型"),
            metadata_cols=("app", "system", "dataset", "scenario", "data_type", "source_id"),
        ),
    )


def _slice_window(df: pd.DataFrame, start_ts: int, end_ts: int) -> pd.DataFrame:
    if df is None or df.empty:
        return df.copy() if df is not None else pd.DataFrame()
    return df[(df["timestamp"] >= int(start_ts)) & (df["timestamp"] < int(end_ts))].copy()


def _attach_native_buckets(metrics: pd.DataFrame) -> pd.DataFrame:
    """Attach a precomputed ``buckets`` column (frozenset[str]) per row.

    Uses the dataset-native assignment from ``portable_bucket_assignment`` so
    the algorithm reads dataset-native bucket membership instead of inferring
    it from KPI name tokens.  KPIs with no fault evidence receive an empty
    frozenset and enter no bucket.
    """
    if metrics is None or metrics.empty or "kpi_name" not in metrics.columns:
        return metrics
    out = metrics.copy()
    cmdb = out.get("cmdb_id")
    if cmdb is None:
        cmdb = pd.Series([""] * len(out), index=out.index)
    out[BUCKETS_COLUMN] = [
        assign_kpi_buckets(kpi, comp)
        for kpi, comp in zip(out["kpi_name"].astype(str), cmdb.astype(str))
    ]
    return out


def _case_row_dict(row: Any, columns: Iterable[str]) -> dict[str, Any]:
    values = row if isinstance(row, tuple) else tuple(row)
    return {str(col): values[idx] for idx, col in enumerate(columns)}


def _to_int_ts(value: Any) -> int:
    if pd.isna(value):
        raise ValueError("case timestamp is missing")
    if isinstance(value, pd.Timestamp):
        return int(value.timestamp())
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return int(pd.to_datetime(value).timestamp())
    if abs(numeric) > 10_000_000_000:
        numeric /= 1000.0
    return int(round(numeric))


def _first_existing(df: pd.DataFrame, candidates: tuple[str, ...]) -> str:
    lower_to_original = {str(col).lower(): str(col) for col in df.columns}
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
        original = lower_to_original.get(candidate.lower())
        if original:
            return original
    return candidates[0]
