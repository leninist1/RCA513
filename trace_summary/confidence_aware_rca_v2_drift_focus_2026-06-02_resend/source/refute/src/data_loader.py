"""
data_loader.py -- shared Bank CSV loading helpers for refute v2.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class BankDataPaths:
    data_root: Path

    @classmethod
    def from_root(cls, root: str | Path) -> "BankDataPaths":
        return cls(Path(root))

    def record_csv(self) -> Path:
        return self.data_root / "record.csv"

    def telemetry_dir(self, date_key: str) -> Path:
        return self.data_root / "telemetry" / date_key

    def metric_container_csv(self, date_key: str) -> Path:
        return self.telemetry_dir(date_key) / "metric" / "metric_container.csv"

    def log_service_csv(self, date_key: str) -> Path:
        return self.telemetry_dir(date_key) / "log" / "log_service.csv"


def load_records(data_root: str | Path) -> pd.DataFrame:
    records = pd.read_csv(BankDataPaths.from_root(data_root).record_csv())
    records["timestamp"] = records["timestamp"].astype(int)
    records["date_key"] = pd.to_datetime(records["datetime"]).dt.strftime("%Y_%m_%d")
    return records


def _metric_frame_from_csv(path: Path) -> pd.DataFrame | None:
    header = pd.read_csv(path, nrows=0)
    cols = set(header.columns)
    if {"timestamp", "cmdb_id", "kpi_name", "value"} <= cols:
        frame = pd.read_csv(path, usecols=["timestamp", "cmdb_id", "kpi_name", "value"])
    elif {"timestamp", "cmdb_id", "name", "value"} <= cols:
        frame = pd.read_csv(path, usecols=["timestamp", "cmdb_id", "name", "value"]).rename(columns={"name": "kpi_name"})
    else:
        return None
    frame["timestamp"] = pd.to_numeric(frame["timestamp"], errors="coerce")
    if frame["timestamp"].max() and frame["timestamp"].max() > 10_000_000_000:
        frame["timestamp"] = (frame["timestamp"] // 1000).astype("Int64")
    frame["cmdb_id"] = frame["cmdb_id"].astype(str).str.replace(r"^node-[^.]+[.]", "", regex=True)
    return frame[["timestamp", "cmdb_id", "kpi_name", "value"]]


def load_metric_day(paths: BankDataPaths, date_key: str) -> pd.DataFrame:
    metric_dir = paths.telemetry_dir(date_key) / "metric"
    csvs = sorted(metric_dir.glob("metric_*.csv")) or [paths.metric_container_csv(date_key)]
    frames = []
    for csv_path in csvs:
        if csv_path.exists() and csv_path.stat().st_size:
            frame = _metric_frame_from_csv(csv_path)
            if frame is not None:
                frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["timestamp", "cmdb_id", "kpi_name", "value"])

def load_log_day(paths: BankDataPaths, date_key: str) -> pd.DataFrame:
    log_dir = paths.telemetry_dir(date_key) / "log"
    if not log_dir.exists():
        return pd.DataFrame(columns=["timestamp", "cmdb_id", "log_name", "value"])
    frames = []
    for csv_path in [log_dir / "log_service.csv", log_dir / "log_proxy.csv"]:
        if not csv_path.exists() or not csv_path.stat().st_size:
            continue
        header = pd.read_csv(csv_path, nrows=0)
        cols = [col for col in ["timestamp", "cmdb_id", "log_name", "value"] if col in header.columns]
        if not {"timestamp", "cmdb_id", "value"} <= set(cols):
            continue
        frame = pd.read_csv(csv_path, usecols=cols)
        if "log_name" not in frame.columns:
            frame["log_name"] = csv_path.stem
        frames.append(frame[["timestamp", "cmdb_id", "log_name", "value"]])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["timestamp", "cmdb_id", "log_name", "value"])

def load_trace_window(paths: BankDataPaths, date_key: str, timestamp: int,
                      window: int, chunksize: int = 500_000,
                      max_rows: int = 500_000) -> pd.DataFrame:
    path = paths.telemetry_dir(date_key) / "trace" / "trace_span.csv"
    cols = ["timestamp", "cmdb_id", "parent_id", "span_id", "trace_id", "duration"]
    if not path.exists():
        return pd.DataFrame(columns=cols)
    lo_ms = timestamp * 1000
    hi_ms = (timestamp + window) * 1000
    parts = []
    total = 0
    for chunk in pd.read_csv(path, usecols=cols, chunksize=chunksize,
                             dtype={"parent_id": str, "span_id": str, "trace_id": str}):
        sel = chunk[(chunk["timestamp"] >= lo_ms) & (chunk["timestamp"] < hi_ms)]
        if sel.empty:
            continue
        parts.append(sel)
        total += len(sel)
        if total >= max_rows:
            break
    if not parts:
        return pd.DataFrame(columns=cols)
    return pd.concat(parts, ignore_index=True)


def load_trace_windows_for_cases(paths: BankDataPaths, date_key: str,
                                 timestamps: list[int], window: int,
                                 chunksize: int = 500_000,
                                 max_rows_per_case: int = 500_000) -> dict[int, pd.DataFrame]:
    """Load several trace windows with one chunked pass over the daily trace file."""
    path = paths.telemetry_dir(date_key) / "trace" / "trace_span.csv"
    cols = ["timestamp", "cmdb_id", "parent_id", "span_id", "trace_id", "duration"]
    unique_ts = sorted({int(ts) for ts in timestamps})
    buckets = {ts: [] for ts in unique_ts}
    counts = {ts: 0 for ts in unique_ts}
    if not unique_ts or not path.exists():
        return {ts: pd.DataFrame(columns=cols) for ts in unique_ts}

    windows_ms = [(ts, ts * 1000, (ts + window) * 1000) for ts in unique_ts]
    global_lo = min(lo for _, lo, _ in windows_ms)
    global_hi = max(hi for _, _, hi in windows_ms)

    for chunk in pd.read_csv(path, usecols=cols, chunksize=chunksize,
                             dtype={"parent_id": str, "span_id": str, "trace_id": str}):
        chunk = chunk[(chunk["timestamp"] >= global_lo) & (chunk["timestamp"] <= global_hi)]
        if chunk.empty:
            continue
        for ts, lo, hi in windows_ms:
            if counts[ts] >= max_rows_per_case:
                continue
            sel = chunk[(chunk["timestamp"] >= lo) & (chunk["timestamp"] < hi)]
            if sel.empty:
                continue
            remaining = max_rows_per_case - counts[ts]
            if len(sel) > remaining:
                sel = sel.head(remaining)
            buckets[ts].append(sel)
            counts[ts] += len(sel)

    out = {}
    for ts in unique_ts:
        if buckets[ts]:
            out[ts] = pd.concat(buckets[ts], ignore_index=True)
        else:
            out[ts] = pd.DataFrame(columns=cols)
    return out


def filter_window(df: pd.DataFrame, timestamp: int, window: int) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    return df[(df["timestamp"] >= timestamp) & (df["timestamp"] <= timestamp + window)].copy()
