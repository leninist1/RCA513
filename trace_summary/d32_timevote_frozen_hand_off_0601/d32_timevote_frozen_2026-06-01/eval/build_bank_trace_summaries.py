"""Build bounded trace summaries for Bank record cases.

The output is one small JSON per case. This avoids repeated runtime reads of
large trace_span.csv files and provides edge-count/drop evidence for network
rules.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
import time

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = PROJECT_ROOT.parent
for path in (PROJECT_ROOT, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from refute.src.data_loader import BankDataPaths, load_records  # noqa: E402
from refute_b_v2.trace_summary import summarize_trace_window  # noqa: E402


TRACE_COLS = ["timestamp", "cmdb_id", "parent_id", "span_id", "trace_id", "duration"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="/home/yan/workspace/data/openrca/Bank")
    parser.add_argument("--out-dir", default="trace_summaries/bank_records")
    parser.add_argument("--fault-window", type=int, default=600)
    parser.add_argument("--baseline-window", type=int, default=3600)
    parser.add_argument("--chunksize", type=int, default=500_000)
    parser.add_argument("--max-rows-per-part", type=int, default=120_000)
    parser.add_argument("--date", default="")
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def _empty_trace() -> pd.DataFrame:
    return pd.DataFrame(columns=TRACE_COLS)


def _append_limited(bucket: list[pd.DataFrame], counts: dict[str, int], key: str,
                    rows: pd.DataFrame, max_rows: int) -> None:
    if rows.empty or counts[key] >= max_rows:
        return
    remaining = max_rows - counts[key]
    if len(rows) > remaining:
        rows = rows.head(remaining)
    bucket.append(rows)
    counts[key] += len(rows)


def build_for_day(paths: BankDataPaths, date_key: str, day_records: pd.DataFrame,
                  out_dir: Path, fault_window: int, baseline_window: int,
                  chunksize: int, max_rows_per_part: int) -> list[dict]:
    trace_path = paths.telemetry_dir(date_key) / "trace" / "trace_span.csv"
    rows = []
    if not trace_path.exists() or trace_path.stat().st_size == 0:
        for rec in day_records.itertuples(index=False):
            out_path = out_dir / f"{date_key}__{int(rec.timestamp)}__{rec.component}.json"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            summary = {
                "date": date_key,
                "timestamp": int(rec.timestamp),
                "component": str(rec.component),
                "reason": str(rec.reason),
                "trace_status": "missing",
                **summarize_trace_window(_empty_trace(), _empty_trace()),
            }
            out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
            rows.append({"path": str(out_path), "trace_status": "missing", "fault_rows": 0, "baseline_rows": 0})
        return rows

    cases = []
    for rec in day_records.itertuples(index=False):
        ts = int(rec.timestamp)
        cases.append({
            "timestamp": ts,
            "component": str(rec.component),
            "reason": str(rec.reason),
            "baseline_lo": (ts - baseline_window) * 1000,
            "baseline_hi": ts * 1000,
            "fault_lo": ts * 1000,
            "fault_hi": (ts + fault_window) * 1000,
            "fault_parts": [],
            "baseline_parts": [],
            "counts": {"fault": 0, "baseline": 0},
        })
    global_lo = min(c["baseline_lo"] for c in cases)
    global_hi = max(c["fault_hi"] for c in cases)

    for chunk in pd.read_csv(
        trace_path,
        usecols=TRACE_COLS,
        chunksize=chunksize,
        dtype={"parent_id": str, "span_id": str, "trace_id": str},
    ):
        chunk = chunk[(chunk["timestamp"] >= global_lo) & (chunk["timestamp"] < global_hi)]
        if chunk.empty:
            continue
        for case in cases:
            fault = chunk[(chunk["timestamp"] >= case["fault_lo"]) & (chunk["timestamp"] < case["fault_hi"])]
            baseline = chunk[(chunk["timestamp"] >= case["baseline_lo"]) & (chunk["timestamp"] < case["baseline_hi"])]
            _append_limited(case["fault_parts"], case["counts"], "fault", fault, max_rows_per_part)
            _append_limited(case["baseline_parts"], case["counts"], "baseline", baseline, max_rows_per_part)
        del chunk
        gc.collect()

    for case in cases:
        fault_df = pd.concat(case["fault_parts"], ignore_index=True) if case["fault_parts"] else _empty_trace()
        baseline_df = pd.concat(case["baseline_parts"], ignore_index=True) if case["baseline_parts"] else _empty_trace()
        summary = {
            "date": date_key,
            "timestamp": case["timestamp"],
            "component": case["component"],
            "reason": case["reason"],
            "trace_status": "present" if not fault_df.empty else "empty_window",
            "fault_rows": int(len(fault_df)),
            "baseline_rows": int(len(baseline_df)),
            **summarize_trace_window(fault_df, baseline_df),
        }
        out_path = out_dir / f"{date_key}__{case['timestamp']}__{case['component']}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        rows.append({
            "path": str(out_path),
            "trace_status": summary["trace_status"],
            "fault_rows": summary["fault_rows"],
            "baseline_rows": summary["baseline_rows"],
        })
        del fault_df, baseline_df
    gc.collect()
    return rows


def main() -> int:
    args = parse_args()
    root = Path(args.data_root)
    paths = BankDataPaths.from_root(root)
    records = load_records(root)
    if args.date:
        records = records[records["date_key"] == args.date]
    if args.limit:
        records = records.head(args.limit)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    start = time.time()
    for date_key, day_records in records.groupby("date_key", sort=True):
        day_start = time.time()
        rows = build_for_day(
            paths,
            date_key,
            day_records,
            out_dir,
            args.fault_window,
            args.baseline_window,
            args.chunksize,
            args.max_rows_per_part,
        )
        manifest.extend({"date": date_key, **row} for row in rows)
        print(json.dumps({
            "date": date_key,
            "cases": len(rows),
            "seconds": round(time.time() - day_start, 2),
            "present": sum(1 for row in rows if row["trace_status"] == "present"),
            "missing_or_empty": sum(1 for row in rows if row["trace_status"] != "present"),
        }, ensure_ascii=False), flush=True)

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps({
        "n": len(manifest),
        "seconds": round(time.time() - start, 2),
        "rows": manifest,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
