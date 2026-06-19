"""Build portable log/trace/topology inputs for non-OpenRCA datasets.

The output contract matches ``eval/run_portable_d32.py``:

* logs.csv: timestamp,cmdb_id,value
* trace_summaries/<case_id>.json: D32 trace summary schema
* topology.json: lightweight component inventory
* summary.json: audit counters

This script only converts observed telemetry.  It does not use root-cause
labels for candidate generation, trace summaries, logs, or topology.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone, timedelta
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Mapping

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
for path in (PROJECT_ROOT, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


TRACE_COLUMNS = ("timestamp", "cmdb_id", "trace_id", "span_id", "parent_id", "duration")
LOG_COLUMNS = ("timestamp", "cmdb_id", "value")
_LOCAL_TZ = timezone(timedelta(hours=8))
_EADRO_LOG_TS = re.compile(r"^\[(?P<ts>\d{4}-[A-Za-z]{3}-\d{2} \d{2}:\d{2}:\d{2}(?:\.\d+)?)\]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["aiops2021", "eadro"], required=True)
    parser.add_argument("--cases-csv", required=True)
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--metrics-csv", default=None,
                        help="Optional normalized metric CSV used only for topology component inventory.")
    parser.add_argument("--case-filter-column", default=None)
    parser.add_argument("--case-filter-value", default=None)
    parser.add_argument("--case-exclude-value", default=None)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--chunksize", type=int, default=500_000)
    parser.add_argument("--trace-slow-ratio-threshold", type=float, default=1.5)
    parser.add_argument("--trace-top-k-edges", type=int, default=20)
    parser.add_argument("--eadro-trace-time-shift-sec", type=int, default=-28_800,
                        help="Shift Eadro span timestamps onto the existing portable case/metric time axis.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    trace_dir = out_dir / "trace_summaries"
    out_dir.mkdir(parents=True, exist_ok=True)
    trace_dir.mkdir(parents=True, exist_ok=True)

    cases = _load_cases(args)
    if args.dataset == "aiops2021":
        result = build_aiops2021_inputs(
            cases=cases,
            raw_root=Path(args.raw_root),
            out_dir=out_dir,
            chunksize=int(args.chunksize),
            slow_ratio_threshold=float(args.trace_slow_ratio_threshold),
            top_k_edges=int(args.trace_top_k_edges),
            metrics_csv=Path(args.metrics_csv) if args.metrics_csv else None,
        )
    else:
        result = build_eadro_inputs(
            cases=cases,
            raw_root=Path(args.raw_root),
            out_dir=out_dir,
            slow_ratio_threshold=float(args.trace_slow_ratio_threshold),
            top_k_edges=int(args.trace_top_k_edges),
            trace_time_shift_sec=int(args.eadro_trace_time_shift_sec),
            metrics_csv=Path(args.metrics_csv) if args.metrics_csv else None,
        )
    _write_json(out_dir / "summary.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def build_aiops2021_inputs(
    *,
    cases: pd.DataFrame,
    raw_root: Path,
    out_dir: Path,
    chunksize: int = 500_000,
    slow_ratio_threshold: float = 1.5,
    top_k_edges: int = 20,
    metrics_csv: Path | None = None,
) -> dict[str, Any]:
    trace_dir = out_dir / "trace_summaries"
    day_cases = _cases_by_local_day(cases)
    logs_by_case: dict[str, list[pd.DataFrame]] = defaultdict(list)
    trace_by_case: dict[str, list[pd.DataFrame]] = defaultdict(list)
    source_audit: dict[str, Any] = {}

    for day, rows in sorted(day_cases.items()):
        day_dir = raw_root / day
        if not day_dir.exists():
            source_audit[day] = {"status": "missing_day_dir", "cases": len(rows)}
            continue
        source_audit[day] = {"status": "present", "cases": len(rows)}
        _collect_aiops_logs(day_dir / "logs", rows, logs_by_case, chunksize=chunksize)
        trace_path = day_dir / "trace" / f"trace_{day}.csv"
        if trace_path.exists():
            _collect_aiops_trace(trace_path, rows, trace_by_case, chunksize=chunksize)
        else:
            source_audit[day]["trace_status"] = "missing_trace_file"

    log_rows = _write_logs_csv(out_dir / "logs.csv", logs_by_case)
    summaries = _write_trace_summaries(
        cases,
        trace_by_case,
        trace_dir,
        slow_ratio_threshold=slow_ratio_threshold,
        top_k_edges=top_k_edges,
        dataset="aiops2021",
    )
    topology = _build_topology(
        dataset="aiops2021",
        metrics_csv=metrics_csv,
        logs_by_case=logs_by_case,
        trace_by_case=trace_by_case,
    )
    _write_json(out_dir / "topology.json", topology)
    return {
        "dataset": "aiops2021",
        "n_cases": int(len(cases)),
        "log_rows": int(log_rows),
        "trace_summaries": summaries,
        "topology_components": len(topology.get("containers", {})),
        "sources": source_audit,
        "outputs": _output_manifest(out_dir),
    }


def build_eadro_inputs(
    *,
    cases: pd.DataFrame,
    raw_root: Path,
    out_dir: Path,
    slow_ratio_threshold: float = 1.5,
    top_k_edges: int = 20,
    trace_time_shift_sec: int = -28_800,
    metrics_csv: Path | None = None,
) -> dict[str, Any]:
    trace_dir = out_dir / "trace_summaries"
    root = _eadro_extracted_root(raw_root)
    logs_by_case: dict[str, list[pd.DataFrame]] = defaultdict(list)
    trace_by_case: dict[str, list[pd.DataFrame]] = defaultdict(list)
    source_audit: dict[str, Any] = {}

    for source_file, group in cases.groupby("source_file", sort=True):
        source = str(source_file)
        telemetry_dir = _eadro_telemetry_dir(root, source)
        if telemetry_dir is None:
            source_audit[source] = {"status": "missing_telemetry_dir", "cases": int(len(group))}
            continue
        source_audit[source] = {"status": "present", "cases": int(len(group)), "telemetry_dir": str(telemetry_dir)}
        spans_path = telemetry_dir / "spans.json"
        logs_path = telemetry_dir / "logs.json"
        if spans_path.exists():
            span_df = _load_eadro_spans(spans_path, trace_time_shift_sec=trace_time_shift_sec)
            _slice_frame_to_cases(span_df, list(_case_windows(group)), trace_by_case)
        else:
            source_audit[source]["trace_status"] = "missing_spans_json"
        if logs_path.exists():
            log_df = _load_eadro_logs(logs_path)
            _slice_frame_to_cases(log_df, list(_case_windows(group)), logs_by_case)
        else:
            source_audit[source]["log_status"] = "missing_logs_json"

    log_rows = _write_logs_csv(out_dir / "logs.csv", logs_by_case)
    summaries = _write_trace_summaries(
        cases,
        trace_by_case,
        trace_dir,
        slow_ratio_threshold=slow_ratio_threshold,
        top_k_edges=top_k_edges,
        dataset="eadro",
    )
    topology = _build_topology(
        dataset="eadro",
        metrics_csv=metrics_csv,
        logs_by_case=logs_by_case,
        trace_by_case=trace_by_case,
    )
    _write_json(out_dir / "topology.json", topology)
    return {
        "dataset": "eadro",
        "n_cases": int(len(cases)),
        "log_rows": int(log_rows),
        "trace_summaries": summaries,
        "topology_components": len(topology.get("containers", {})),
        "eadro_trace_time_shift_sec": int(trace_time_shift_sec),
        "sources": source_audit,
        "outputs": _output_manifest(out_dir),
    }


def build_trace_summary_from_spans(
    spans: pd.DataFrame,
    *,
    case_id: str,
    start_ts: int,
    end_ts: int,
    slow_ratio_threshold: float = 1.5,
    top_k_edges: int = 20,
    dataset: str = "portable",
) -> dict[str, Any]:
    if spans is None or spans.empty:
        return {
            "case_id": str(case_id),
            "trace_status": "empty_window",
            "window": {"start_ts": int(start_ts), "end_ts": int(end_ts)},
            "service_stats": {},
            "edge_stats": [],
            "events": {"first_anomalous_service": None, "slow_edges": [], "dropped_edges": []},
            "metadata": {"dataset": dataset, "builder": "portable_multimodal_inputs"},
        }
    df = spans.loc[:, [col for col in TRACE_COLUMNS if col in spans.columns]].copy()
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df["duration"] = pd.to_numeric(df["duration"], errors="coerce")
    df = df.dropna(subset=["timestamp", "cmdb_id", "duration"])
    if df.empty:
        return build_trace_summary_from_spans(
            pd.DataFrame(columns=TRACE_COLUMNS),
            case_id=case_id,
            start_ts=start_ts,
            end_ts=end_ts,
            slow_ratio_threshold=slow_ratio_threshold,
            top_k_edges=top_k_edges,
            dataset=dataset,
        )
    df["timestamp"] = df["timestamp"].astype("int64")
    df["cmdb_id"] = df["cmdb_id"].astype(str)
    df["trace_id"] = df.get("trace_id", "").astype(str)
    df["span_id"] = df.get("span_id", "").astype(str)
    df["parent_id"] = df.get("parent_id", "").astype(str)

    service_stats = _service_stats(df)
    edge_stats = _edge_stats(df)
    slow_edges = _slow_edges(edge_stats, threshold=slow_ratio_threshold, top_k=top_k_edges)
    first = _first_anomalous_service(df, service_stats, slow_edges, threshold=slow_ratio_threshold)
    return {
        "case_id": str(case_id),
        "trace_status": "present",
        "window": {"start_ts": int(start_ts), "end_ts": int(end_ts)},
        "service_stats": service_stats,
        "edge_stats": edge_stats,
        "events": {
            "first_anomalous_service": first,
            "slow_edges": slow_edges,
            "dropped_edges": [],
        },
        "metadata": {
            "dataset": dataset,
            "builder": "portable_multimodal_inputs",
            "span_rows": int(len(df)),
            "slow_ratio_threshold": float(slow_ratio_threshold),
        },
    }


def _load_cases(args: argparse.Namespace) -> pd.DataFrame:
    cases = pd.read_csv(args.cases_csv)
    for col in ("case_id", "start_ts", "end_ts"):
        if col not in cases.columns:
            raise ValueError(f"cases CSV missing required column: {col}")
    if args.case_filter_column:
        if args.case_filter_column not in cases.columns:
            raise ValueError(f"case filter column not found: {args.case_filter_column}")
        if args.case_filter_value is not None:
            cases = cases[cases[args.case_filter_column].astype(str) == str(args.case_filter_value)].copy()
        if args.case_exclude_value is not None:
            cases = cases[cases[args.case_filter_column].astype(str) != str(args.case_exclude_value)].copy()
    if args.max_cases is not None:
        cases = cases.head(max(0, int(args.max_cases))).copy()
    cases["case_id"] = cases["case_id"].astype(str)
    cases["start_ts"] = pd.to_numeric(cases["start_ts"], errors="raise").astype("int64")
    cases["end_ts"] = pd.to_numeric(cases["end_ts"], errors="raise").astype("int64")
    return cases.reset_index(drop=True)


def _case_windows(cases: pd.DataFrame) -> Iterable[tuple[str, int, int]]:
    for row in cases.itertuples(index=False):
        yield str(getattr(row, "case_id")), int(getattr(row, "start_ts")), int(getattr(row, "end_ts"))


def _cases_by_local_day(cases: pd.DataFrame) -> dict[str, list[tuple[str, int, int]]]:
    out: dict[str, list[tuple[str, int, int]]] = defaultdict(list)
    for case_id, start, end in _case_windows(cases):
        day = datetime.fromtimestamp(start, tz=_LOCAL_TZ).strftime("%m%d")
        out[day].append((case_id, start, end))
    return dict(out)


def _collect_aiops_logs(
    logs_dir: Path,
    windows: list[tuple[str, int, int]],
    logs_by_case: dict[str, list[pd.DataFrame]],
    *,
    chunksize: int,
) -> None:
    if not logs_dir.exists():
        return
    for path in sorted(logs_dir.glob("*.csv")):
        for chunk in pd.read_csv(path, chunksize=chunksize):
            if chunk.empty or not {"timestamp", "cmdb_id", "value"}.issubset(chunk.columns):
                continue
            ts = _normalize_epoch_seconds(chunk["timestamp"])
            chunk = chunk.assign(timestamp=ts)
            chunk = chunk.dropna(subset=["timestamp", "cmdb_id", "value"])
            if chunk.empty:
                continue
            log_name = chunk["log_name"].astype(str) if "log_name" in chunk.columns else path.stem
            frame = pd.DataFrame({
                "timestamp": chunk["timestamp"].astype("int64"),
                "cmdb_id": chunk["cmdb_id"].astype(str),
                "value": "[" + log_name.astype(str) + "] " + chunk["value"].astype(str),
            })
            _slice_frame_to_cases(frame, windows, logs_by_case)


def _collect_aiops_trace(
    trace_path: Path,
    windows: list[tuple[str, int, int]],
    trace_by_case: dict[str, list[pd.DataFrame]],
    *,
    chunksize: int,
) -> None:
    usecols = ["timestamp", "cmdb_id", "parent_id", "span_id", "trace_id", "duration"]
    for chunk in pd.read_csv(trace_path, usecols=usecols, chunksize=chunksize):
        if chunk.empty:
            continue
        chunk["timestamp"] = _normalize_epoch_seconds(chunk["timestamp"])
        chunk["duration"] = pd.to_numeric(chunk["duration"], errors="coerce")
        chunk = chunk.dropna(subset=["timestamp", "cmdb_id", "duration"])
        if chunk.empty:
            continue
        frame = chunk.rename(columns={col: col for col in usecols}).loc[:, list(TRACE_COLUMNS)].copy()
        frame["timestamp"] = frame["timestamp"].astype("int64")
        frame["duration"] = frame["duration"].astype(float)
        _slice_frame_to_cases(frame, windows, trace_by_case)


def _slice_frame_to_cases(
    frame: pd.DataFrame,
    windows: list[tuple[str, int, int]],
    out: dict[str, list[pd.DataFrame]],
) -> None:
    if frame is None or frame.empty or not windows:
        return
    min_start = min(start for _, start, _ in windows)
    max_end = max(end for _, _, end in windows)
    frame = frame[(frame["timestamp"] >= min_start) & (frame["timestamp"] < max_end)]
    if frame.empty:
        return
    for case_id, start, end in windows:
        rows = frame[(frame["timestamp"] >= int(start)) & (frame["timestamp"] < int(end))]
        if not rows.empty:
            out[str(case_id)].append(rows.copy())


def _load_eadro_spans(spans_path: Path, *, trace_time_shift_sec: int) -> pd.DataFrame:
    with spans_path.open("r", encoding="utf-8") as f:
        traces = json.load(f)
    rows: list[dict[str, Any]] = []
    for trace in traces if isinstance(traces, list) else []:
        processes = trace.get("processes", {}) or {}
        proc_to_service = {
            str(pid): str((proc or {}).get("serviceName") or pid)
            for pid, proc in processes.items()
        }
        trace_id = str(trace.get("traceID", ""))
        for span in trace.get("spans", []) or []:
            start = span.get("startTime")
            if start is None:
                continue
            parent_id = ""
            refs = span.get("references") or []
            if refs:
                parent_id = str((refs[0] or {}).get("spanID") or "")
            rows.append({
                "timestamp": int(float(start) / 1_000_000.0) + int(trace_time_shift_sec),
                "cmdb_id": proc_to_service.get(str(span.get("processID")), str(span.get("processID"))),
                "trace_id": str(span.get("traceID") or trace_id),
                "span_id": str(span.get("spanID") or ""),
                "parent_id": parent_id,
                "duration": float(span.get("duration") or 0.0) / 1000.0,
            })
    return pd.DataFrame(rows, columns=TRACE_COLUMNS)


def _load_eadro_logs(logs_path: Path) -> pd.DataFrame:
    with logs_path.open("r", encoding="utf-8") as f:
        logs = json.load(f)
    rows: list[dict[str, Any]] = []
    for service, messages in (logs or {}).items():
        for message in messages or []:
            ts = _parse_eadro_log_ts(str(message))
            if ts is None:
                continue
            rows.append({"timestamp": ts, "cmdb_id": str(service), "value": str(message)})
    return pd.DataFrame(rows, columns=LOG_COLUMNS)


def _parse_eadro_log_ts(message: str) -> int | None:
    match = _EADRO_LOG_TS.match(message)
    if not match:
        return None
    raw = match.group("ts")
    for fmt in ("%Y-%b-%d %H:%M:%S.%f", "%Y-%b-%d %H:%M:%S"):
        try:
            dt = datetime.strptime(raw, fmt).replace(tzinfo=_LOCAL_TZ)
            return int(dt.timestamp())
        except ValueError:
            continue
    return None


def _eadro_extracted_root(raw_root: Path) -> Path:
    if (raw_root / "extracted_sn").exists() or (raw_root / "extracted_tt").exists():
        return raw_root
    candidate = raw_root / ".adapter_work_v22"
    if (candidate / "extracted_sn").exists() or (candidate / "extracted_tt").exists():
        return candidate
    raise FileNotFoundError(f"cannot find Eadro extracted root under {raw_root}")


def _eadro_telemetry_dir(root: Path, source_file: str) -> Path | None:
    prefix = "extracted_sn" if source_file.startswith("SN.") else "extracted_tt"
    telemetry_name = source_file.replace(".fault-", ".", 1)
    if telemetry_name.endswith(".json"):
        telemetry_name = telemetry_name[:-5]
    for base in (root / prefix / "data", root / prefix):
        path = base / telemetry_name
        if path.exists():
            return path
    return None


def _write_logs_csv(path: Path, logs_by_case: Mapping[str, list[pd.DataFrame]]) -> int:
    frames = [frame for frames in logs_by_case.values() for frame in frames if frame is not None and not frame.empty]
    if frames:
        out = pd.concat(frames, ignore_index=True)
        out = out.loc[:, list(LOG_COLUMNS)].drop_duplicates().sort_values(["timestamp", "cmdb_id"], kind="mergesort")
    else:
        out = pd.DataFrame(columns=LOG_COLUMNS)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return int(len(out))


def _write_trace_summaries(
    cases: pd.DataFrame,
    trace_by_case: Mapping[str, list[pd.DataFrame]],
    trace_dir: Path,
    *,
    slow_ratio_threshold: float,
    top_k_edges: int,
    dataset: str,
) -> dict[str, int]:
    trace_dir.mkdir(parents=True, exist_ok=True)
    present = 0
    empty = 0
    for row in cases.itertuples(index=False):
        case_id = str(getattr(row, "case_id"))
        frames = list(trace_by_case.get(case_id, []))
        spans = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=TRACE_COLUMNS)
        summary = build_trace_summary_from_spans(
            spans,
            case_id=case_id,
            start_ts=int(getattr(row, "start_ts")),
            end_ts=int(getattr(row, "end_ts")),
            slow_ratio_threshold=slow_ratio_threshold,
            top_k_edges=top_k_edges,
            dataset=dataset,
        )
        if summary["trace_status"] == "present":
            present += 1
        else:
            empty += 1
        _write_json(trace_dir / f"{case_id}.json", summary)
    return {"present": int(present), "empty_window": int(empty)}


def _service_stats(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for service, group in df.groupby("cmdb_id", sort=True):
        durations = pd.to_numeric(group["duration"], errors="coerce").dropna()
        if durations.empty:
            continue
        out[str(service)] = {
            "span_count": int(len(durations)),
            "duration_p50": _quantile(durations, 0.50),
            "duration_p95": _quantile(durations, 0.95),
            "duration_p99": _quantile(durations, 0.99),
            "first_timestamp": int(group["timestamp"].min()),
        }
    return out


def _edge_stats(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df.empty or not {"trace_id", "span_id", "parent_id"}.issubset(df.columns):
        return []
    lookup = (
        df[["trace_id", "span_id", "cmdb_id"]]
        .dropna()
        .drop_duplicates(["trace_id", "span_id"])
        .rename(columns={"span_id": "parent_id", "cmdb_id": "src"})
    )
    lookup = lookup[lookup["parent_id"].astype(str) != ""]
    children = df[df["parent_id"].astype(str) != ""]
    if lookup.empty or children.empty:
        return []
    edges = children.merge(lookup, on=["trace_id", "parent_id"], how="left")
    edges = edges.dropna(subset=["src"])
    if edges.empty:
        return []
    edges = edges[edges["src"].astype(str) != edges["cmdb_id"].astype(str)]
    if edges.empty:
        return []
    out: list[dict[str, Any]] = []
    for (src, dst), group in edges.groupby(["src", "cmdb_id"], sort=True):
        durations = pd.to_numeric(group["duration"], errors="coerce").dropna()
        if durations.empty:
            continue
        out.append({
            "src": str(src),
            "dst": str(dst),
            "span_count": int(len(durations)),
            "duration_p50": _quantile(durations, 0.50),
            "duration_p95": _quantile(durations, 0.95),
            "duration_p99": _quantile(durations, 0.99),
            "first_timestamp": int(group["timestamp"].min()),
        })
    if not out:
        return []
    baseline = _median([item["duration_p95"] for item in out if item["duration_p95"] > 0.0])
    for item in out:
        item["slow_ratio"] = _safe_ratio(float(item["duration_p95"]), baseline)
    return sorted(out, key=lambda item: (-float(item["slow_ratio"]), -float(item["duration_p95"]), item["src"], item["dst"]))


def _slow_edges(edge_stats: list[dict[str, Any]], *, threshold: float, top_k: int) -> list[dict[str, Any]]:
    rows = [
        {
            "src": item["src"],
            "dst": item["dst"],
            "slow_ratio": float(item.get("slow_ratio", 0.0) or 0.0),
            "duration_p95": float(item.get("duration_p95", 0.0) or 0.0),
            "span_count": int(item.get("span_count", 0) or 0),
            "first_timestamp": item.get("first_timestamp"),
            "event": "slow_edges",
        }
        for item in edge_stats
        if float(item.get("slow_ratio", 0.0) or 0.0) >= float(threshold)
    ]
    return rows[: max(0, int(top_k))]


def _first_anomalous_service(
    df: pd.DataFrame,
    service_stats: Mapping[str, Mapping[str, Any]],
    slow_edges: list[Mapping[str, Any]],
    *,
    threshold: float,
) -> str | None:
    if slow_edges:
        first_edge = min(slow_edges, key=lambda item: int(item.get("first_timestamp") or 2**62))
        return str(first_edge.get("src") or first_edge.get("dst"))
    p95_values = [float(item.get("duration_p95", 0.0) or 0.0) for item in service_stats.values()]
    baseline = _median([value for value in p95_values if value > 0.0])
    if baseline <= 0.0:
        return None
    best_service = None
    best_ratio = 0.0
    for service, stats in service_stats.items():
        ratio = _safe_ratio(float(stats.get("duration_p95", 0.0) or 0.0), baseline)
        if ratio > best_ratio:
            best_service = str(service)
            best_ratio = ratio
    if best_service and best_ratio >= float(threshold):
        rows = df[df["cmdb_id"].astype(str) == best_service]
        if not rows.empty:
            return best_service
    return None


def _build_topology(
    *,
    dataset: str,
    metrics_csv: Path | None,
    logs_by_case: Mapping[str, list[pd.DataFrame]],
    trace_by_case: Mapping[str, list[pd.DataFrame]],
) -> dict[str, Any]:
    components: set[str] = set()
    if metrics_csv and metrics_csv.exists():
        for chunk in pd.read_csv(metrics_csv, usecols=["cmdb_id"], chunksize=500_000):
            components.update(str(item) for item in chunk["cmdb_id"].dropna().astype(str).unique())
    for frames_by_case in (logs_by_case, trace_by_case):
        for frames in frames_by_case.values():
            for frame in frames:
                if frame is not None and not frame.empty and "cmdb_id" in frame.columns:
                    components.update(str(item) for item in frame["cmdb_id"].dropna().astype(str).unique())
    containers = {
        component: {
            "node_proxy": component,
            "role": _component_role(component),
            "type": _component_role(component),
        }
        for component in sorted(components)
    }
    return {"dataset": dataset, "containers": containers, "nodes": {}, "edges": []}


def _component_role(component: str) -> str:
    text = str(component).lower()
    if "mysql" in text or "mongo" in text or "redis" in text or "postgres" in text:
        return "database"
    if "apache" in text or "nginx" in text or text.startswith("ig"):
        return "gateway"
    if "tomcat" in text or "-service" in text or text.startswith("ts-"):
        return "application"
    if text.startswith("mg"):
        return "middleware"
    if "docker" in text or "node" in text or "os" in text:
        return "host"
    return "service"


def _normalize_epoch_seconds(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return numeric.where(numeric.abs() <= 10_000_000_000, numeric / 1000.0).round()


def _quantile(values: pd.Series, q: float) -> float:
    value = float(values.quantile(q))
    return value if math.isfinite(value) else 0.0


def _median(values: Iterable[float]) -> float:
    rows = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not rows:
        return 0.0
    mid = len(rows) // 2
    if len(rows) % 2:
        return rows[mid]
    return (rows[mid - 1] + rows[mid]) / 2.0


def _safe_ratio(value: float, baseline: float) -> float:
    if baseline <= 0.0:
        return 0.0
    return float(value) / float(baseline)


def _output_manifest(out_dir: Path) -> dict[str, str]:
    return {
        "logs_csv": str(out_dir / "logs.csv"),
        "trace_summary_dir": str(out_dir / "trace_summaries"),
        "topology_json": str(out_dir / "topology.json"),
        "summary_json": str(out_dir / "summary.json"),
    }


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)


if __name__ == "__main__":
    raise SystemExit(main())
