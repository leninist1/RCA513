#!/usr/bin/env bash
set -euo pipefail

BASE=/home/dell2/RCA513/ysj/trace_summary/confidence_aware_rca_v2_drift_focus_2026-06-02_resend/source

cd "$BASE"
export PYTHONPATH=".:..:${PYTHONPATH:-}"

python3 - <<'PY'
from pathlib import Path
import re


def write_if_changed(path: Path, text: str) -> None:
    old = path.read_text(encoding="utf-8")
    if old != text:
        path.write_text(text, encoding="utf-8")
        print(f"patched {path}")
    else:
        print(f"unchanged {path}")


# query parser: support Telecom dates such as April/May 2020.
path = Path("refute_b_v2/query_windows.py")
text = path.read_text(encoding="utf-8")
if "MONTH_NAMES =" not in text:
    text = text.replace(
        "\ndef parse_query_window(instruction: str) -> QueryWindow:\n",
        '''\nMONTH_NAMES = {
    "January": 1, "February": 2, "March": 3, "April": 4,
    "May": 5, "June": 6, "July": 7, "August": 8,
    "September": 9, "October": 10, "November": 11, "December": 12,
}\n\n\ndef parse_query_window(instruction: str) -> QueryWindow:\n''',
    )

new_parse_query_window = '''def parse_query_window(instruction: str) -> QueryWindow:
    date_matches = re.findall(r"(January|February|March|April|May|June|July|August|September|October|November|December) (\\d{1,2}), (\\d{4})", instruction)
    times = re.findall(r"(\\d{1,2}:\\d{2})", instruction)
    if not date_matches or len(times) < 2:
        raise ValueError(f"cannot parse query window: {instruction}")
    m1, d1, y1 = date_matches[0]
    m2, d2, y2 = date_matches[1] if len(date_matches) > 1 else date_matches[0]
    h1, n1 = (int(x) for x in times[0].split(":"))
    h2, n2 = (int(x) for x in times[1].split(":"))
    start = datetime(int(y1), MONTH_NAMES[m1], int(d1), h1, n1)
    end = datetime(int(y2), MONTH_NAMES[m2], int(d2), h2, n2)
    if end <= start:
        end = datetime(int(y1), MONTH_NAMES[m1], int(d1), h2, n2) + timedelta(days=1)
    return QueryWindow(start, end, parse_failure_count(instruction))
'''
text = re.sub(
    r"def parse_query_window\(instruction: str\) -> QueryWindow:\n.*\Z",
    lambda _: new_parse_query_window,
    text,
    flags=re.S,
)
write_if_changed(path, text)


# data loader: support Telecom metric name->kpi_name and millisecond timestamps.
path = Path("refute/src/data_loader.py")
text = path.read_text(encoding="utf-8")
metric_start = text.find("def _metric_frame_from_csv")
if metric_start == -1:
    metric_start = text.index("def load_metric_day")
metric_end = text.index("\ndef load_log_day", metric_start)
new_metric_loader = '''def _metric_frame_from_csv(path: Path) -> pd.DataFrame | None:
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
'''
text = text[:metric_start] + new_metric_loader + text[metric_end:]
log_start = text.index("def load_log_day")
log_end = text.index("\ndef load_trace_window", log_start)
new_log_loader = '''def load_log_day(paths: BankDataPaths, date_key: str) -> pd.DataFrame:
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
'''
text = text[:log_start] + new_log_loader + text[log_end:]
write_if_changed(path, text)


# baseline reader: support Telecom name->kpi_name and skip incompatible metric_app-like tables.
path = Path("refute/src/baseline_distributions.py")
text = path.read_text(encoding="utf-8")
iter_start = text.find("def _metric_read_spec")
if iter_start == -1:
    iter_start = text.index("def _iter_metric_frames")
iter_end = text.index("\ndef _filter_kpi_level", iter_start)
new_iter = '''def _metric_read_spec(path: str | Path):
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
'''
text = text[:iter_start] + new_iter + text[iter_end:]
write_if_changed(path, text)


# trace summary builder: support Bank, Market, and Telecom trace schemas.
path = Path("eval/build_query_trace_summaries.py")
text = path.read_text(encoding="utf-8")
compat_func = '''TRACE_COLS = ["timestamp", "cmdb_id", "parent_id", "span_id", "trace_id", "duration"]


def read_trace_chunks_openrca_compat(trace_path: Path, chunksize: int):
    header = pd.read_csv(trace_path, nrows=0)
    cols = set(header.columns)
    if {"timestamp", "cmdb_id", "parent_id", "span_id", "trace_id", "duration"} <= cols:
        usecols = TRACE_COLS
        rename = {}
        dtype = {"parent_id": str, "span_id": str, "trace_id": str}
    elif {"timestamp", "cmdb_id", "parent_span", "span_id", "trace_id", "duration"} <= cols:
        usecols = ["timestamp", "cmdb_id", "parent_span", "span_id", "trace_id", "duration"]
        rename = {"parent_span": "parent_id"}
        dtype = {"parent_span": str, "span_id": str, "trace_id": str}
    elif {"startTime", "cmdb_id", "pid", "id", "traceId", "elapsedTime"} <= cols:
        usecols = ["startTime", "cmdb_id", "pid", "id", "traceId", "elapsedTime"]
        rename = {
            "startTime": "timestamp",
            "pid": "parent_id",
            "id": "span_id",
            "traceId": "trace_id",
            "elapsedTime": "duration",
        }
        dtype = {"pid": str, "id": str, "traceId": str}
    else:
        raise ValueError(f"unsupported trace columns in {trace_path}: {sorted(cols)}")
    for chunk in pd.read_csv(trace_path, usecols=usecols, chunksize=chunksize, dtype=dtype):
        if rename:
            chunk = chunk.rename(columns=rename)
        yield chunk[TRACE_COLS]
'''
text = re.sub(
    r'TRACE_COLS = \["timestamp", "cmdb_id", "parent_id", "span_id", "trace_id", "duration"\]\n(?:\n\ndef read_trace_chunks_[\s\S]*?yield chunk\[TRACE_COLS\]\n)?',
    lambda _: compat_func,
    text,
    count=1,
)
text = text.replace(
    "read_trace_chunks_market_compat(trace_path, args.chunksize)",
    "read_trace_chunks_openrca_compat(trace_path, args.chunksize)",
)
text = re.sub(
    r'''for chunk in pd\.read_csv\(
\s*trace_path,
\s*usecols=TRACE_COLS,
\s*chunksize=args\.chunksize,
\s*dtype=\{"parent_id": str, "span_id": str, "trace_id": str\},
\s*\):''',
    "for chunk in read_trace_chunks_openrca_compat(trace_path, args.chunksize):",
    text,
)
write_if_changed(path, text)


# Telecom reason bucket mappings.
path = Path("refute_b_v2_d32/schema.py")
text = path.read_text(encoding="utf-8")
if '"cpu fault": "cpu"' not in text:
    text = text.replace(
        '    "network packet loss": "network_packet_loss",\n',
        '''    "network packet loss": "network_packet_loss",
    "cpu fault": "cpu",
    "network delay": "network_latency",
    "network loss": "network_packet_loss",
    "db connection limit": "db_connection",
    "db close": "db_connection",
''',
    )
write_if_changed(path, text)


# Telecom KPI token matching: lower-case metric names are common.
path = Path("refute_b_v2_d32/signature.py")
text = path.read_text(encoding="utf-8")
text = text.replace(
    'CPU_TOKENS = ("CPU", "Cpu", "CPULoad", "SingleCpu")',
    'CPU_TOKENS = ("CPU", "Cpu", "cpu", "CPULoad", "SingleCpu")',
)
text = text.replace(
    'NET_TOKENS = ("Network", "TCP", "Packet", "rejected", "Aborted", "NET")',
    'NET_TOKENS = ("Network", "network", "traffic", "TCP", "Packet", "packet", "rejected", "Aborted", "NET")',
)
write_if_changed(path, text)

print("Telecom compat patch complete")
PY

python3 - <<'PY'
from refute_b_v2.query_windows import parse_query_window
from refute_b_v2_d32.schema import reason_bucket

w = parse_query_window("During the specified time range of April 11, 2020, from 00:00 to 00:30, there was one failure reported.")
assert w.start.year == 2020 and w.start.month == 4
assert reason_bucket("CPU fault") == "cpu"
assert reason_bucket("network delay") == "network_latency"
print("Telecom compat self-check ok")
PY
