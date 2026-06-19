"""Adaptive baseline strategies for datasets without clean normal days.

OpenRCA's default runner continues to use ``BaselineStore`` directly.  This
module exposes the same ``is_anomalous`` API for future Eadro/AIOps2021 adapters
where historical, per-case pre-fault, and cross-sectional references may have
different reliability.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from refute.src.baseline_distributions import AnomalyResult, BaselineStats, BaselineStore


DEFAULT_MIN_SAMPLES = 8
DEFAULT_IQR_EPS = 1e-9


@dataclass(frozen=True)
class BaselineReliability:
    source: str
    n_stats: int
    eligible_stats: int
    coverage: float

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "n_stats": self.n_stats,
            "eligible_stats": self.eligible_stats,
            "coverage": self.coverage,
        }


@dataclass(frozen=True)
class AdaptiveBaselineConfig:
    min_samples: int = DEFAULT_MIN_SAMPLES
    iqr_eps: float = DEFAULT_IQR_EPS
    prefer_historical: bool = True
    enable_pre_fault: bool = True
    enable_cross_sectional: bool = True
    enable_within_window: bool = False


class AdaptiveBaselineStore:
    """Historical-first baseline with explicit fallbacks.

    The object is compatible with current D32 callers.  It never mutates or
    reinterprets a supplied historical baseline; if historical evidence is
    eligible for a key, that answer wins.
    """

    def __init__(
        self,
        *,
        historical: BaselineStore | None = None,
        pre_fault: BaselineStore | None = None,
        cross_sectional: BaselineStore | None = None,
        within_window: BaselineStore | None = None,
        metadata: Mapping[str, object] | None = None,
    ):
        self.historical = historical
        self.pre_fault = pre_fault
        self.cross_sectional = cross_sectional
        self.within_window = within_window
        self.metadata = dict(metadata or {})

    def is_anomalous(
        self,
        cmdb_id: str,
        kpi_name: str,
        value: float,
        threshold: str = "p99",
        two_sided: bool = True,
    ) -> AnomalyResult:
        for source, store in self._stores():
            if store is None:
                continue
            result = store.is_anomalous(cmdb_id, kpi_name, value, threshold=threshold, two_sided=two_sided)
            if _usable_result(result):
                return _with_source(result, source)
        # Preserve the safe D32 default: no reliable baseline means no anomaly.
        return AnomalyResult(False, 0.0, "none", None, "missing_reliable_baseline", None)

    def reliability_report(self, metric_df: pd.DataFrame | None = None) -> list[dict]:
        keys = _metric_keys(metric_df)
        out = []
        for source, store in self._stores():
            if store is None:
                continue
            total = len(store)
            eligible = sum(1 for stats in store.stats.values() if stats.eligible)
            if keys:
                covered = sum(1 for key in keys if store.get(*key) is not None and store.get(*key).eligible)
                coverage = covered / max(1, len(keys))
            else:
                coverage = eligible / max(1, total)
            out.append(BaselineReliability(source, total, eligible, float(coverage)).to_dict())
        return out

    def _stores(self):
        yield "historical", self.historical
        yield "pre_fault", self.pre_fault
        yield "cross_sectional", self.cross_sectional
        yield "within_window", self.within_window


def build_adaptive_baseline(
    metric_df: pd.DataFrame,
    *,
    historical: BaselineStore | None = None,
    window_start_ts: int | None = None,
    config: AdaptiveBaselineConfig | None = None,
) -> AdaptiveBaselineStore:
    """Build an adaptive baseline object for one normalized incident window.

    ``window_start_ts`` marks the injection/query start when available.  Rows
    earlier than this timestamp become the per-case pre-fault baseline, a useful
    option for Eadro/AIOps exports that include warm-up context but no separate
    normal day.
    """

    cfg = config or AdaptiveBaselineConfig()
    metric_df = _clean_metric_frame(metric_df)
    pre_fault = None
    cross_sectional = None
    within_window = None

    if cfg.enable_pre_fault and window_start_ts is not None and not metric_df.empty:
        pre_rows = metric_df[metric_df["timestamp"] < int(window_start_ts)]
        pre_fault = _build_pair_store(
            pre_rows,
            source="pre_fault",
            min_samples=cfg.min_samples,
            iqr_eps=cfg.iqr_eps,
        )
    if cfg.enable_cross_sectional and not metric_df.empty:
        cross_sectional = _build_cross_sectional_store(
            metric_df,
            min_samples=cfg.min_samples,
            iqr_eps=cfg.iqr_eps,
        )
    if cfg.enable_within_window and not metric_df.empty:
        within_window = _build_pair_store(
            metric_df,
            source="within_window",
            min_samples=cfg.min_samples,
            iqr_eps=cfg.iqr_eps,
        )

    return AdaptiveBaselineStore(
        historical=historical if cfg.prefer_historical else None,
        pre_fault=pre_fault,
        cross_sectional=cross_sectional,
        within_window=within_window,
        metadata={
            "design": "adaptive_baseline",
            "min_samples": cfg.min_samples,
            "iqr_eps": cfg.iqr_eps,
            "prefer_historical": cfg.prefer_historical,
            "enable_pre_fault": cfg.enable_pre_fault,
            "enable_cross_sectional": cfg.enable_cross_sectional,
            "enable_within_window": cfg.enable_within_window,
        },
    )


def _build_pair_store(
    df: pd.DataFrame,
    *,
    source: str,
    min_samples: int,
    iqr_eps: float,
) -> BaselineStore:
    stats = []
    if df is None or df.empty:
        return BaselineStore([], metadata={"source": source})
    for (cmdb_id, kpi_name), group in df.groupby(["cmdb_id", "kpi_name"], sort=True):
        stats.append(_stats_from_values(
            cmdb_id=str(cmdb_id),
            kpi_name=str(kpi_name),
            values=group["value"].to_numpy(dtype=float),
            min_samples=min_samples,
            iqr_eps=iqr_eps,
        ))
    return BaselineStore(stats, metadata={"source": source})


def _build_cross_sectional_store(
    df: pd.DataFrame,
    *,
    min_samples: int,
    iqr_eps: float,
) -> BaselineStore:
    stats = []
    if df is None or df.empty:
        return BaselineStore([], metadata={"source": "cross_sectional"})
    components = sorted(df["cmdb_id"].dropna().astype(str).unique())
    by_kpi = {
        str(kpi): group["value"].to_numpy(dtype=float)
        for kpi, group in df.groupby("kpi_name", sort=True)
    }
    for component in components:
        for kpi_name, values in by_kpi.items():
            stats.append(_stats_from_values(
                cmdb_id=component,
                kpi_name=kpi_name,
                values=values,
                min_samples=min_samples,
                iqr_eps=iqr_eps,
            ))
    return BaselineStore(stats, metadata={"source": "cross_sectional"})


def _stats_from_values(
    *,
    cmdb_id: str,
    kpi_name: str,
    values: np.ndarray,
    min_samples: int,
    iqr_eps: float,
) -> BaselineStats:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = int(values.size)
    if n == 0:
        return BaselineStats(cmdb_id, kpi_name, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False, "empty")
    p01, p05, q25, median, q75, p95, p99 = np.percentile(values, [1, 5, 25, 50, 75, 95, 99])
    iqr = float(q75 - q25)
    if n < int(min_samples):
        eligible, reason = False, "insufficient_samples"
    elif iqr < float(iqr_eps):
        eligible, reason = False, "near_constant_iqr"
    else:
        eligible, reason = True, "ok"
    return BaselineStats(
        cmdb_id=str(cmdb_id),
        kpi_name=str(kpi_name),
        n=n,
        median=float(median),
        iqr=iqr,
        p01=float(p01),
        p05=float(p05),
        p95=float(p95),
        p99=float(p99),
        eligible=eligible,
        reason=reason,
    )


def _clean_metric_frame(metric_df: pd.DataFrame | None) -> pd.DataFrame:
    if metric_df is None or metric_df.empty:
        return pd.DataFrame(columns=["timestamp", "cmdb_id", "kpi_name", "value"])
    required = {"timestamp", "cmdb_id", "kpi_name", "value"}
    missing = required - set(metric_df.columns)
    if missing:
        raise ValueError(f"metric_df missing required columns: {sorted(missing)}")
    df = metric_df.loc[:, ["timestamp", "cmdb_id", "kpi_name", "value"]].copy()
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df["cmdb_id"] = df["cmdb_id"].astype(str)
    df["kpi_name"] = df["kpi_name"].astype(str)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.dropna(subset=["timestamp", "cmdb_id", "kpi_name", "value"])


def _usable_result(result: AnomalyResult) -> bool:
    stats = getattr(result, "stats", None)
    if stats is None:
        return False
    return bool(getattr(stats, "eligible", False))


def _with_source(result: AnomalyResult, source: str) -> AnomalyResult:
    if source == "historical":
        return result
    return AnomalyResult(
        is_anomalous=bool(result.is_anomalous),
        deviation=float(result.deviation),
        direction=str(result.direction),
        threshold=result.threshold,
        reason=f"{source}:{result.reason}",
        stats=result.stats,
    )


def _metric_keys(metric_df: pd.DataFrame | None) -> set[tuple[str, str]]:
    if metric_df is None or metric_df.empty or not {"cmdb_id", "kpi_name"} <= set(metric_df.columns):
        return set()
    return {
        (str(row.cmdb_id), str(row.kpi_name))
        for row in metric_df[["cmdb_id", "kpi_name"]].dropna().itertuples(index=False)
    }
