"""Build trace summaries for official OpenRCA query windows."""
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

from refute.src.data_loader import BankDataPaths  # noqa: E402
from refute_b_v2.query_windows import parse_query_window  # noqa: E402
from refute_b_v2.trace_summary import summarize_trace_window  # noqa: E402


TRACE_COLS = ["timestamp", "cmdb_id", "parent_id", "span_id", "trace_id", "duration"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="/home/yan/workspace/data/openrca/Bank")
    parser.add_argument("--query-csv", default="/home/yan/workspace/data/openrca/Bank/query.csv")
    parser.add_argument("--out-dir", default="trace_summaries/openrca_queries")
    parser.add_argument("--baseline-window", type=int, default=3600)
    parser.add_argument("--chunksize", type=int, default=500_000)
    parser.add_argument("--max-rows-per-part", type=int, default=120_000)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def empty_trace() -> pd.DataFrame:
    return pd.DataFrame(columns=TRACE_COLS)


def append_limited(parts: list[pd.DataFrame], counts: dict[str, int], key: str,
                   rows: pd.DataFrame, max_rows: int) -> None:
    if rows.empty or counts[key] >= max_rows:
        return
    remaining = max_rows - counts[key]
    if len(rows) > remaining:
        rows = rows.head(remaining)
    parts.append(rows)
    counts[key] += len(rows)


def main() -> int:
    args = parse_args()
    paths = BankDataPaths.from_root(args.data_root)
    query_df = pd.read_csv(args.query_csv)
    if args.limit:
        query_df = query_df.head(args.limit)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    query_rows = []
    for idx, row in query_df.iterrows():
        window = parse_query_window(row["instruction"])
        query_rows.append({
            "row_id": int(idx),
            "window": window,
            "baseline_lo": (window.start_ts - args.baseline_window) * 1000,
            "baseline_hi": window.start_ts * 1000,
            "fault_lo": window.start_ts * 1000,
            "fault_hi": window.end_ts * 1000,
            "fault_parts": [],
            "baseline_parts": [],
            "counts": {"fault": 0, "baseline": 0},
        })

    start = time.time()
    manifest = []
    for date_key in sorted({dk for q in query_rows for dk in q["window"].date_keys}):
        date_queries = [q for q in query_rows if date_key in q["window"].date_keys]
        trace_path = paths.telemetry_dir(date_key) / "trace" / "trace_span.csv"
        if not trace_path.exists() or trace_path.stat().st_size == 0:
            continue
        global_lo = min(q["baseline_lo"] for q in date_queries)
        global_hi = max(q["fault_hi"] for q in date_queries)
        day_start = time.time()
        for chunk in pd.read_csv(
            trace_path,
            usecols=TRACE_COLS,
            chunksize=args.chunksize,
            dtype={"parent_id": str, "span_id": str, "trace_id": str},
        ):
            chunk = chunk[(chunk["timestamp"] >= global_lo) & (chunk["timestamp"] < global_hi)]
            if chunk.empty:
                continue
            for q in date_queries:
                baseline = chunk[(chunk["timestamp"] >= q["baseline_lo"]) & (chunk["timestamp"] < q["baseline_hi"])]
                fault = chunk[(chunk["timestamp"] >= q["fault_lo"]) & (chunk["timestamp"] < q["fault_hi"])]
                append_limited(q["baseline_parts"], q["counts"], "baseline", baseline, args.max_rows_per_part)
                append_limited(q["fault_parts"], q["counts"], "fault", fault, args.max_rows_per_part)
            del chunk
            gc.collect()
        print(json.dumps({"date": date_key, "seconds": round(time.time() - day_start, 2), "queries": len(date_queries)}), flush=True)

    for q in query_rows:
        fault_df = pd.concat(q["fault_parts"], ignore_index=True) if q["fault_parts"] else empty_trace()
        baseline_df = pd.concat(q["baseline_parts"], ignore_index=True) if q["baseline_parts"] else empty_trace()
        summary = {
            "row_id": q["row_id"],
            "window": {
                "start": q["window"].start.strftime("%Y-%m-%d %H:%M:%S"),
                "end": q["window"].end.strftime("%Y-%m-%d %H:%M:%S"),
                "failure_count": q["window"].failure_count,
            },
            "trace_status": "present" if not fault_df.empty else "empty_window",
            "fault_rows": int(len(fault_df)),
            "baseline_rows": int(len(baseline_df)),
            **summarize_trace_window(fault_df, baseline_df),
        }
        path = out_dir / f"query_{q['row_id']:03d}.json"
        path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        manifest.append({"row_id": q["row_id"], "path": str(path), "trace_status": summary["trace_status"], "fault_rows": summary["fault_rows"], "baseline_rows": summary["baseline_rows"]})
        del fault_df, baseline_df
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps({"n": len(manifest), "seconds": round(time.time() - start, 2), "rows": manifest}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
