"""
baseline_distributions.py -- Phase 1 / B-L1.2

Learn percentile-based baseline distributions for (cmdb_id, kpi_name) pairs.

The purpose is to replace fragile robust-z scoring on tiny historical windows.
When a KPI has too few samples or nearly zero IQR, the safe default is to report
"not anomalous" with zero deviation instead of emitting an explosive score.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, Iterable, Iterator, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from refute.src.node_container_split import classify_kpi
except ImportError:  # pragma: no cover - allows direct script execution from refute/src
    from node_container_split import classify_kpi


DEFAULT_MIN_SAMPLES = 8
DEFAULT_IQR_EPS = 1e-9
BASELINE_VERSION = 1


@dataclass(frozen=True)
class BaselineStats:
    cmdb_id: str
    kpi_name: str
    n: int
    median: float
    iqr: float
    p01: float
    p05: float
    p95: float
    p99: float
    eligible: bool
    reason: str = "ok"

    def to_dict(self) -> dict:
        return {
            "cmdb_id": self.cmdb_id,
            "kpi_name": self.kpi_name,
            "n": self.n,
            "median": self.median,
            "iqr": self.iqr,
            "p01": self.p01,
            "p05": self.p05,
            "p95": self.p95,
            "p99": self.p99,
            "eligible": self.eligible,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "BaselineStats":
        return cls(
            cmdb_id=str(data["cmdb_id"]),
            kpi_name=str(data["kpi_name"]),
            n=int(data["n"]),
            median=float(data["median"]),
            iqr=float(data["iqr"]),
            p01=float(data["p01"]),
            p05=float(data["p05"]),
            p95=float(data["p95"]),
            p99=float(data["p99"]),
            eligible=bool(data["eligible"]),
            reason=str(data.get("reason", "ok")),
        )


@dataclass(frozen=True)
class AnomalyResult:
    is_anomalous: bool
    deviation: float
    direction: str
    threshold: Optional[float]
    reason: str
    stats: Optional[BaselineStats] = None

    def to_dict(self) -> dict:
        return {
            "is_anomalous": self.is_anomalous,
            "deviation": self.deviation,
            "direction": self.direction,
            "threshold": self.threshold,
            "reason": self.reason,
            "stats": self.stats.to_dict() if self.stats else None,
        }


class BaselineStore:
    def __init__(self, stats: Iterable[BaselineStats], metadata: Optional[dict] = None):
        self.stats: Dict[Tuple[str, str], BaselineStats] = {
            (s.cmdb_id, s.kpi_name): s for s in stats
        }
        self.metadata = dict(metadata or {})

    def __len__(self) -> int:
        return len(self.stats)

    def get(self, cmdb_id: str, kpi_name: str) -> Optional[BaselineStats]:
        return self.stats.get((str(cmdb_id), str(kpi_name)))

    def is_anomalous(
        self,
        cmdb_id: str,
        kpi_name: str,
        value: float,
        threshold: str = "p99",
        two_sided: bool = True,
    ) -> AnomalyResult:
        stats = self.get(cmdb_id, kpi_name)
        if stats is None:
            return AnomalyResult(False, 0.0, "none", None, "missing_baseline", None)
        if not stats.eligible:
            return AnomalyResult(False, 0.0, "none", None, stats.reason, stats)

        try:
            x = float(value)
        except (TypeError, ValueError):
            return AnomalyResult(False, 0.0, "none", None, "non_numeric_value", stats)
        if not np.isfinite(x):
            return AnomalyResult(False, 0.0, "none", None, "non_finite_value", stats)

        deviation = (x - stats.median) / stats.iqr
        if threshold == "p95":
            high, low = stats.p95, stats.p05
        elif threshold == "p99":
            high, low = stats.p99, stats.p01
        else:
            raise ValueError("threshold must be 'p95' or 'p99'")

        if x > high:
            return AnomalyResult(True, float(deviation), "high", high, "above_threshold", stats)
        if two_sided and x < low:
            return AnomalyResult(True, float(deviation), "low", low, "below_threshold", stats)
        return AnomalyResult(False, float(deviation), "none", high, "within_threshold", stats)

    def to_dict(self) -> dict:
        metadata = {
            "version": BASELINE_VERSION,
            **self.metadata,
            "n_stats": len(self.stats),
        }
        rows = [s.to_dict() for s in sorted(self.stats.values(), key=lambda s: (s.cmdb_id, s.kpi_name))]
        return {"metadata": metadata, "stats": rows}

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "BaselineStore":
        rows = data.get("stats", [])
        return cls([BaselineStats.from_dict(row) for row in rows], metadata=dict(data.get("metadata", {})))

    def save_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2, sort_keys=True)

    @classmethod
    def load_json(cls, path: str | Path) -> "BaselineStore":
        with Path(path).open("r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


def _metric_read_spec(path: str | Path):
    header = pd.read_csv(path, nrows=0)
    cols = set(header.columns)
    if {"timestamp", "cmdb_id", "kpi_name", "value"} <= cols:
        return ["timestamp", "cmdb_id", "kpi_name", "value"], {}
    if {"timestamp", "cmdb_id", "name", "value"} <= cols:
        return ["timestamp", "cmdb_id", "name", "value"], {"name": "kpi_name"}
    return None


def _iter_metric_frames(csv_paths: Iterable[str | Path], chunksize: int) -> Iterator[pd.DataFrame]:
    for csv_path in csv_paths:
        spec = _metric_read_spec(csv_path)
        if spec is None:
            continue
        usecols, rename = spec
        for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=chunksize):
            if rename:
                chunk = chunk.rename(columns=rename)
            chunk["cmdb_id"] = chunk["cmdb_id"].astype(str).str.replace(r"^node-[^.]+[.]", "", regex=True)
            chunk["value"] = pd.to_numeric(chunk["value"], errors="coerce")
            yield chunk.dropna(subset=["cmdb_id", "kpi_name", "value"])

def _filter_kpi_level(df: pd.DataFrame, dataset: str, include_node_level: bool) -> pd.DataFrame:
    if include_node_level:
        return df
    keep = df["kpi_name"].map(lambda k: classify_kpi(str(k), dataset=dataset) == "container")
    return df[keep]


def build_baseline_distributions(
    df: pd.DataFrame,
    dataset: str = "Bank",
    include_node_level: bool = False,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    iqr_eps: float = DEFAULT_IQR_EPS,
    metadata: Optional[dict] = None,
) -> BaselineStore:
    required = {"cmdb_id", "kpi_name", "value"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")

    clean = df.dropna(subset=["cmdb_id", "kpi_name", "value"]).copy()
    clean["cmdb_id"] = clean["cmdb_id"].astype(str)
    clean["kpi_name"] = clean["kpi_name"].astype(str)
    clean["value"] = pd.to_numeric(clean["value"], errors="coerce")
    clean = clean.dropna(subset=["value"])
    clean = _filter_kpi_level(clean, dataset=dataset, include_node_level=include_node_level)

    stats = []
    for (cmdb_id, kpi_name), group in clean.groupby(["cmdb_id", "kpi_name"], sort=True):
        values = group["value"].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        n = int(values.size)
        if n == 0:
            continue
        p01, p05, q25, median, q75, p95, p99 = np.percentile(values, [1, 5, 25, 50, 75, 95, 99])
        iqr = float(q75 - q25)
        if n < min_samples:
            eligible, reason = False, "insufficient_samples"
        elif iqr < iqr_eps:
            eligible, reason = False, "near_constant_iqr"
        else:
            eligible, reason = True, "ok"
        stats.append(BaselineStats(
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
        ))

    meta = {
        "dataset": dataset,
        "include_node_level": include_node_level,
        "min_samples": min_samples,
        "iqr_eps": iqr_eps,
        **(metadata or {}),
    }
    return BaselineStore(stats, metadata=meta)


def build_baseline_from_metric_csvs(
    csv_paths: Iterable[str | Path],
    dataset: str = "Bank",
    include_node_level: bool = False,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    iqr_eps: float = DEFAULT_IQR_EPS,
    chunksize: int = 300_000,
    metadata: Optional[dict] = None,
) -> BaselineStore:
    frames = [
        _filter_kpi_level(frame, dataset=dataset, include_node_level=include_node_level)
        for frame in _iter_metric_frames(csv_paths, chunksize=chunksize)
    ]
    if not frames:
        return BaselineStore([], metadata=metadata)
    df = pd.concat(frames, ignore_index=True)
    store = build_baseline_distributions(
        df,
        dataset=dataset,
        include_node_level=True,
        min_samples=min_samples,
        iqr_eps=iqr_eps,
        metadata=metadata,
    )
    store.metadata["include_node_level"] = include_node_level
    return store


def summarize_store(store: BaselineStore) -> dict:
    total = len(store)
    eligible = sum(1 for s in store.stats.values() if s.eligible)
    reasons: Dict[str, int] = {}
    for s in store.stats.values():
        reasons[s.reason] = reasons.get(s.reason, 0) + 1
    return {
        "total": total,
        "eligible": eligible,
        "ineligible": total - eligible,
        "reasons": reasons,
        "metadata": store.metadata,
    }
