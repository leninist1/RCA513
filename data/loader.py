"""Unified data loader for OpenRCA Bank, Telecom, and Market systems."""

import os
import re
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

from ..config import (
    QueryCase, GroundTruth, ScoringPoint, UnifiedTelemetry,
    SYSTEM_PATHS, OPENRCA_ROOT,
)
from .schema import SYSTEM_SCHEMAS


class OpenRCALoader:
    def __init__(self, system_name: str):
        if system_name not in SYSTEM_PATHS:
            raise ValueError(f"Unknown system: {system_name}. Choose from {list(SYSTEM_PATHS)}")
        self.system_name = system_name
        self.paths = SYSTEM_PATHS[system_name]
        self.schema = SYSTEM_SCHEMAS[system_name]
        self._queries_cache: Dict[str, pd.DataFrame] = {}
        self._records_cache: Dict[str, pd.DataFrame] = {}

    @property
    def root(self) -> str:
        return self.paths["root"]

    @property
    def sub_systems(self) -> List[str]:
        return self.paths["sub_systems"]

    @property
    def has_logs(self) -> bool:
        return self.paths["has_logs"]

    @property
    def has_traces(self) -> bool:
        return self.paths["has_traces"]

    # ---------- Query loading ----------

    def _get_query_path(self, sub_system: str = "") -> str:
        base = self.root if not sub_system else os.path.join(self.root, sub_system)
        return os.path.join(base, "query.csv")

    def _get_record_path(self, sub_system: str = "") -> str:
        base = self.root if not sub_system else os.path.join(self.root, sub_system)
        return os.path.join(base, "record.csv")

    def load_queries(self, sub_system: str = "") -> List[QueryCase]:
        path = self._get_query_path(sub_system)
        if not os.path.exists(path):
            return []
        df = pd.read_csv(path)
        queries = []
        for _, row in df.iterrows():
            task_index = str(row["task_index"]).strip()
            instruction = str(row["instruction"])
            scoring_text = str(row.get("scoring_points", ""))
            time_window = self._parse_time_window(instruction)
            scoring_points = parse_scoring_points(scoring_text)
            queries.append(QueryCase(
                task_index=task_index,
                system=self.system_name,
                sub_system=sub_system,
                instruction=instruction,
                time_window=time_window,
                scoring_points=scoring_points,
            ))
        return queries

    def load_records(self, sub_system: str = "") -> pd.DataFrame:
        path = self._get_record_path(sub_system)
        if not os.path.exists(path):
            return pd.DataFrame()
        df = pd.read_csv(path)
        df = _normalize_record_columns(df, self.system_name)
        return df

    def load_all_queries(self) -> List[QueryCase]:
        all_queries = []
        for sub in self.sub_systems:
            all_queries.extend(self.load_queries(sub))
        return all_queries

    # ---------- Record matching ----------

    def match_query_to_records(self, query: QueryCase) -> Tuple[List[GroundTruth], Optional[float]]:
        """Match a query to its ground truth record(s).

        For single-fault: returns ([GroundTruth], inject_time)
        For multi-fault: returns ([GroundTruth, ...], first_inject_time)
        """
        records = self.load_records(query.sub_system)
        if records.empty:
            return [], None

        t_start, t_end = self._parse_time_range_epoch(query.time_window)
        if t_start is None or t_end is None:
            return [], None

        window_records = records[
            (records["timestamp"] >= t_start) & (records["timestamp"] <= t_end)
        ].copy()

        if window_records.empty:
            return [], None

        window_records = window_records.sort_values("timestamp")

        gts = []
        for _, rec in window_records.iterrows():
            gts.append(GroundTruth(
                component=str(rec.get("component", "")),
                reason=str(rec.get("reason", "")),
                timestamp=float(rec["timestamp"]),
                datetime_str=str(rec.get("datetime", "")),
            ))

        inject_time = float(window_records.iloc[0]["timestamp"])
        return gts, inject_time

    def _parse_time_range_epoch(self, time_window: Tuple[str, str]) -> Tuple[Optional[float], Optional[float]]:
        """Convert datetime string window to epoch timestamps."""
        try:
            t_start = datetime.strptime(time_window[0], "%Y-%m-%d %H:%M:%S").timestamp()
            t_end = datetime.strptime(time_window[1], "%Y-%m-%d %H:%M:%S").timestamp()
            return t_start, t_end
        except (ValueError, IndexError):
            return None, None

    def resolve_telemetry_date(self, query: QueryCase) -> Optional[str]:
        """Map query time window to telemetry date folder (YYYY_MM_DD format)."""
        try:
            dt = datetime.strptime(query.time_window[0][:10], "%Y-%m-%d")
            folder_name = dt.strftime("%Y_%m_%d")
            telemetry_root = os.path.join(self.root, query.sub_system, "telemetry") if query.sub_system else os.path.join(self.root, "telemetry")
            if os.path.isdir(os.path.join(telemetry_root, folder_name)):
                return folder_name
            # Fallback: find closest available date
            if os.path.isdir(telemetry_root):
                available = sorted(os.listdir(telemetry_root))
                if available:
                    return available[0]
            return None
        except (ValueError, IndexError):
            return None

    # ---------- Telemetry loading ----------

    _telemetry_cache: Dict[str, Any] = {}

    def load_telemetry(self, date_str: str, sub_system: str = "") -> UnifiedTelemetry:
        """Load all telemetry for a date, with caching."""
        cache_key = f"{self.system_name}:{sub_system}:{date_str}"
        if cache_key in self._telemetry_cache:
            return self._telemetry_cache[cache_key]

        telemetry_root = os.path.join(self.root, sub_system, "telemetry", date_str) if sub_system else os.path.join(self.root, "telemetry", date_str)

        metrics_dfs = self._load_all_metrics(telemetry_root)
        logs_df = self._load_all_logs(telemetry_root) if self.has_logs else None
        traces_df = self._load_all_traces(telemetry_root) if self.has_traces else None

        entities = set()
        entity_types = {}
        for mdf in metrics_dfs:
            if mdf is not None and "entity" in mdf.columns:
                for e in mdf["entity"].unique():
                    entities.add(str(e))
                    entity_types[str(e)] = "service"

        if logs_df is not None and "entity" in logs_df.columns:
            for e in logs_df["entity"].unique():
                entities.add(str(e))

        if traces_df is not None and "entity" in traces_df.columns:
            for e in traces_df["entity"].unique():
                entities.add(str(e))

        unified_metrics = pd.concat([m for m in metrics_dfs if m is not None], ignore_index=True) if metrics_dfs else pd.DataFrame()

        telemetry = UnifiedTelemetry(
            metrics=unified_metrics if not unified_metrics.empty else None,
            logs=logs_df,
            traces=traces_df,
            entities=sorted(entities),
            entity_types=entity_types,
            system=self.system_name,
        )
        # Limit cache size to avoid OOM
        if len(self._telemetry_cache) >= 3:
            oldest = next(iter(self._telemetry_cache))
            del self._telemetry_cache[oldest]
        self._telemetry_cache[cache_key] = telemetry
        return telemetry

    def _load_all_metrics(self, telemetry_root: str) -> List[pd.DataFrame]:
        metric_dir = os.path.join(telemetry_root, "metric")
        if not os.path.isdir(metric_dir):
            return []
        dfs = []
        for fname in sorted(os.listdir(metric_dir)):
            if not fname.endswith(".csv"):
                continue
            fpath = os.path.join(metric_dir, fname)
            schema_key = fname.replace(".csv", "")
            if schema_key not in self.schema:
                continue
            try:
                df = pd.read_csv(fpath)
                df = _unify_metric_dataframe(df, self.schema[schema_key], self.system_name)
                if df is not None and not df.empty:
                    dfs.append(df)
            except Exception:
                continue
        return dfs

    def _load_all_logs(self, telemetry_root: str) -> Optional[pd.DataFrame]:
        log_dir = os.path.join(telemetry_root, "log")
        if not os.path.isdir(log_dir):
            return None
        dfs = []
        for fname in sorted(os.listdir(log_dir)):
            if not fname.endswith(".csv"):
                continue
            fpath = os.path.join(log_dir, fname)
            schema_key = fname.replace(".csv", "")
            if schema_key not in self.schema:
                continue
            try:
                df = pd.read_csv(fpath, on_bad_lines='skip')
                df = _unify_log_dataframe(df, self.schema[schema_key])
                if df is not None and not df.empty:
                    dfs.append(df)
            except Exception:
                continue
        return pd.concat(dfs, ignore_index=True) if dfs else None

    def _load_all_traces(self, telemetry_root: str) -> Optional[pd.DataFrame]:
        trace_dir = os.path.join(telemetry_root, "trace")
        if not os.path.isdir(trace_dir):
            return None
        dfs = []
        for fname in sorted(os.listdir(trace_dir)):
            if not fname.endswith(".csv"):
                continue
            fpath = os.path.join(trace_dir, fname)
            schema_key = fname.replace(".csv", "")
            if schema_key not in self.schema:
                continue
            try:
                df = pd.read_csv(fpath)
                df = _unify_trace_dataframe(df, self.schema[schema_key])
                if df is not None and not df.empty:
                    dfs.append(df)
            except Exception:
                continue
        return pd.concat(dfs, ignore_index=True) if dfs else None

    # ---------- Time window parsing ----------

    @staticmethod
    def _parse_time_window(instruction: str) -> Tuple[str, str]:
        """Extract time window from instruction text.

        Handles: "between X and Y", "from X to Y", "within the time range of X to Y"
        """
        date_match = re.search(
            r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),?\s+(\d{4})',
            instruction
        )
        if not date_match:
            return ("", "")

        # Multiple time formats
        # 1. "between HH:MM and HH:MM"
        time_match = re.search(r'between\s+(\d{2}:\d{2})\s+and\s+(\d{2}:\d{2})', instruction)
        # 2. "from HH:MM to HH:MM" or "of HH:MM to HH:MM"
        if not time_match:
            time_match = re.search(r'(?:from|of)\s+(\d{2}:\d{2})\s+to\s+(\d{2}:\d{2})', instruction)
        # 3. Simple "HH:MM to HH:MM"
        if not time_match:
            time_match = re.search(r'(\d{2}:\d{2})\s+to\s+(\d{2}:\d{2})', instruction)
        # 4. Cross-date: "from HH:MM to Month DD, YYYY, at HH:MM"
        if not time_match:
            time_match = re.search(r'from\s+(\d{2}:\d{2})\s+to\s+', instruction)
            if time_match:
                end_match = re.search(
                    r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),?\s+(\d{4}),?\s+at\s+(\d{2}:\d{2})',
                    instruction
                )
                if end_match:
                    month_map = OpenRCALoader._month_map()
                    end_month = month_map[end_match.group(1)]
                    end_day = int(end_match.group(2))
                    end_year = int(end_match.group(3))
                    end_time = end_match.group(4)
                    start_time = time_match.group(1)
                    start_month = month_map[date_match.group(1)]
                    start_day = int(date_match.group(2))
                    start_year = int(date_match.group(3))
                    return (
                        f"{start_year}-{start_month:02d}-{start_day:02d} {start_time}:00",
                        f"{end_year}-{end_month:02d}-{end_day:02d} {end_time}:00",
                    )

        if not time_match:
            return ("", "")

        if time_match.lastindex and time_match.lastindex >= 2:
            month_str, day, year = date_match.group(1), date_match.group(2), date_match.group(3)
            month = OpenRCALoader._month_map()[month_str]
            start_time = time_match.group(1)
            end_time = time_match.group(2)
            return (
                f"{year}-{month:02d}-{int(day):02d} {start_time}:00",
                f"{year}-{month:02d}-{int(day):02d} {end_time}:00",
            )

        return ("", "")

    @staticmethod
    def _month_map():
        return {
            "January": 1, "February": 2, "March": 3, "April": 4,
            "May": 5, "June": 6, "July": 7, "August": 8,
            "September": 9, "October": 10, "November": 11, "December": 12,
        }


# ======== Scoring Points Parser ========

def parse_scoring_points(text: str) -> List[ScoringPoint]:
    """Parse scoring_points field from query.csv."""
    if not isinstance(text, str) or not text.strip():
        return []
    points = []
    lines = text.strip().split("\n")
    for line in lines:
        line = line.strip()
        if not line:
            continue
        rank_match = re.search(r'(?:The\s+)?(\d+)-th', line)
        rank = int(rank_match.group(1)) - 1 if rank_match else 0

        if "occurrence time" in line:
            dt_match = re.search(r'(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})', line)
            if dt_match:
                points.append(ScoringPoint("time", rank, dt_match.group(1)))
        elif "root cause component" in line:
            comp_match = re.search(r'component is\s+(.+)', line)
            if comp_match:
                points.append(ScoringPoint("component", rank, comp_match.group(1).strip()))
        elif "root cause reason" in line:
            reason_match = re.search(r'reason is\s+(.+)', line)
            if reason_match:
                points.append(ScoringPoint("reason", rank, reason_match.group(1).strip()))
    return points


# ======== Internal Helpers ========

def _normalize_record_columns(df: pd.DataFrame, system: str) -> pd.DataFrame:
    """Normalize record.csv columns to standard names."""
    rename_map = {}
    for col in df.columns:
        col_lower = col.strip().lower()
        if col_lower in ("component",):
            rename_map[col] = "component"
        elif col_lower in ("reason",):
            rename_map[col] = "reason"
        elif col_lower in ("timestamp",):
            rename_map[col] = "timestamp"
        elif col_lower in ("datetime",):
            rename_map[col] = "datetime"
        elif col_lower in ("level",):
            rename_map[col] = "level"
    if rename_map:
        df = df.rename(columns=rename_map)

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    if "datetime" not in df.columns and "timestamp" in df.columns:
        df["datetime"] = df["timestamp"].apply(
            lambda ts: datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S") if pd.notna(ts) else ""
        )
    return df


def _unify_metric_dataframe(df: pd.DataFrame, schema: dict, system: str) -> Optional[pd.DataFrame]:
    """Transform a raw metric CSV into unified format: timestamp, entity, metric_name, value."""
    time_col = schema["time_col"]
    entity_col = schema["entity_col"]

    if time_col not in df.columns or entity_col not in df.columns:
        return None

    df = df.copy()
    # Handle time
    if schema.get("time_unit") == "millis":
        df["timestamp"] = pd.to_numeric(df[time_col], errors="coerce") / 1000.0
    else:
        df["timestamp"] = pd.to_numeric(df[time_col], errors="coerce")

    df["entity"] = df[entity_col].astype(str)

    if "value_cols" in schema:
        # Wide format: each value column becomes a metric_name
        rows = []
        for vcol in schema["value_cols"]:
            if vcol in df.columns:
                subset = df[["timestamp", "entity"]].copy()
                subset["metric_name"] = vcol
                subset["value"] = pd.to_numeric(df[vcol], errors="coerce")
                rows.append(subset)
        if rows:
            result = pd.concat(rows, ignore_index=True)
            result = result.dropna(subset=["value"])
            result = result.replace([np.inf, -np.inf], np.nan).dropna(subset=["value"])
            return result
    elif "value_col" in schema and "metric_col" in schema:
        # Key-value format: metric_col holds metric name, value_col holds value
        if schema["metric_col"] in df.columns and schema["value_col"] in df.columns:
            df["metric_name"] = df[schema["metric_col"]].astype(str)
            df["value"] = pd.to_numeric(df[schema["value_col"]], errors="coerce")
            result = df[["timestamp", "entity", "metric_name", "value"]].copy()
            result = result.dropna(subset=["value"])
            result = result.replace([np.inf, -np.inf], np.nan).dropna(subset=["value"])
            return result
    elif "value_col" in schema:
        df["metric_name"] = "value"
        df["value"] = pd.to_numeric(df[schema["value_col"]], errors="coerce")
        result = df[["timestamp", "entity", "metric_name", "value"]].copy()
        result = result.dropna(subset=["value"]).replace([np.inf, -np.inf], np.nan).dropna(subset=["value"])
        return result

    return None


def _unify_log_dataframe(df: pd.DataFrame, schema: dict) -> Optional[pd.DataFrame]:
    """Transform raw log CSV into unified format: timestamp, entity, message."""
    time_col = schema["time_col"]
    entity_col = schema["entity_col"]
    message_col = schema["message_col"]

    if time_col not in df.columns:
        return None

    df = df.copy()
    if schema.get("time_unit") == "millis":
        df["timestamp"] = pd.to_numeric(df[time_col], errors="coerce") / 1000.0
    else:
        df["timestamp"] = pd.to_numeric(df[time_col], errors="coerce")

    df["entity"] = df[entity_col].astype(str) if entity_col in df.columns else "unknown"
    df["message"] = df[message_col].astype(str) if message_col in df.columns else ""

    result = df[["timestamp", "entity", "message"]].dropna(subset=["timestamp"])
    return result


def _unify_trace_dataframe(df: pd.DataFrame, schema: dict) -> Optional[pd.DataFrame]:
    """Transform raw trace CSV into unified format."""
    time_col = schema["time_col"]
    entity_col = schema["entity_col"]

    if time_col not in df.columns:
        return None

    df = df.copy()
    if schema.get("time_unit") == "millis":
        df["timestamp"] = pd.to_numeric(df[time_col], errors="coerce") / 1000.0
    else:
        df["timestamp"] = pd.to_numeric(df[time_col], errors="coerce")

    df["entity"] = df[entity_col].astype(str) if entity_col in df.columns else "unknown"

    trace_id_col = schema.get("trace_id_col")
    span_id_col = schema.get("span_id_col")
    parent_id_col = schema.get("parent_id_col")
    duration_col = schema.get("duration_col")

    df["trace_id"] = df[trace_id_col].astype(str) if trace_id_col and trace_id_col in df.columns else ""
    df["span_id"] = df[span_id_col].astype(str) if span_id_col and span_id_col in df.columns else ""
    df["parent_id"] = df[parent_id_col].astype(str) if parent_id_col and parent_id_col in df.columns else ""
    df["duration"] = pd.to_numeric(df[duration_col], errors="coerce") if duration_col and duration_col in df.columns else 0
    df["status_code"] = df[schema["status_code_col"]].astype(str) if "status_code_col" in schema and schema["status_code_col"] in df.columns else ""

    keep_cols = ["timestamp", "entity", "trace_id", "span_id", "parent_id", "duration", "status_code"]
    result = df[[c for c in keep_cols if c in df.columns]].dropna(subset=["timestamp"])
    return result
