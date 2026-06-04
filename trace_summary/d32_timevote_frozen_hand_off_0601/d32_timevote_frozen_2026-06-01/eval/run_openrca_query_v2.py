"""Query-driven OpenRCA Bank evaluation for Scheme B v2.

Prediction uses only query instructions plus telemetry. record.csv labels and
query scoring_points are not used during prediction.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = PROJECT_ROOT.parent
for path in (PROJECT_ROOT, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from refute.src.baseline_distributions import BaselineStore  # noqa: E402
from refute.src.data_loader import BankDataPaths, filter_window, load_log_day, load_metric_day  # noqa: E402
from refute.src.evidence_query import EvidenceQuery, kpi_in_bucket  # noqa: E402
from refute_b_v2.confidence_calibration import RuntimeConfidenceCalibrator  # noqa: E402
from refute_b_v2.default_rules import default_rule_set  # noqa: E402
from refute_b_v2.evidence_adapter import SummaryBackedEvidence  # noqa: E402
from refute_b_v2.joint_answer_selector import JointAnswerSelector, JointSelectorConfig  # noqa: E402
from refute_b_v2.llm_arbitration import run_llm_arbitration  # noqa: E402
from refute_b_v2.llm_clients import LLMApiError, llm_client_from_env  # noqa: E402
from refute_b_v2.query_windows import QueryWindow, parse_query_window  # noqa: E402
from refute_b_v2.rule_engine import RuleEngine  # noqa: E402
from refute_b_v2.rules import Candidate, RuleSet  # noqa: E402


@dataclass(frozen=True)
class EventSeed:
    timestamp: int
    component: str
    reason: str
    score: float
    source: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="/home/yan/workspace/data/openrca/Bank")
    parser.add_argument("--query-csv", default="/home/yan/workspace/data/openrca/Bank/query.csv")
    parser.add_argument("--baseline", default="../refute/knowledge/baseline_distributions_all_metric_dates.json")
    parser.add_argument("--node-graph", default="../refute/knowledge/node_container_graph_2021_03_10.json")
    parser.add_argument("--rules", default="knowledge/refutation_rules_v2.json")
    parser.add_argument("--out", default="logs/openrca_query_v2_predictions.csv")
    parser.add_argument("--debug-json", default="logs/openrca_query_v2_debug.json")
    parser.add_argument("--fault-window", type=int, default=600)
    parser.add_argument("--max-events-per-query", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--query-trace-summary-dir", default="")
    parser.add_argument(
        "--raw-trace-full-window",
        action="store_true",
        help="Load raw trace_span.csv rows for the full query instruction window; does not use trace summaries.",
    )
    parser.add_argument(
        "--max-trace-rows-per-query",
        type=int,
        default=0,
        help="Safety cap for raw trace rows per query; 0 means no cap.",
    )
    parser.add_argument(
        "--modalities",
        default="metric,log,trace",
        help="Comma-separated evidence modalities to use: metric,log,trace. Example: --modalities trace",
    )
    parser.add_argument("--use-llm", action="store_true", help="call real Claude/DeepSeek client for ambiguous evidence matrices")
    parser.add_argument("--force-llm", action="store_true", help="call LLM for every seed when --use-llm is set")
    parser.add_argument("--llm-provider", default="", help="claude/shqbb/deepseek; defaults to SCHEME_B_LLM_PROVIDER")
    parser.add_argument("--llm-max-calls", type=int, default=8)
    parser.add_argument("--background-baseline-window", type=int, default=3600)
    parser.add_argument("--confidence-calibration", default="", help="Optional calibration_report.json for runtime confidence labels")
    parser.add_argument("--checkpoint-jsonl", default="", help="Append one completed query per line for crash-safe resume.")
    parser.add_argument("--resume-checkpoint", action="store_true", help="Skip row_ids already present in --checkpoint-jsonl.")
    parser.add_argument("--llm-error-policy", choices=("raise", "fallback"), default="fallback", help="On LLM API failure, either abort or rerun that row without LLM.")
    return parser.parse_args()


class DayCache:
    def __init__(self, paths: BankDataPaths):
        self.paths = paths
        self.metric: dict[str, pd.DataFrame] = {}
        self.log: dict[str, pd.DataFrame] = {}

    def metric_day(self, date_key: str) -> pd.DataFrame:
        if date_key not in self.metric:
            self.metric[date_key] = load_metric_day(self.paths, date_key)
        return self.metric[date_key]

    def log_day(self, date_key: str) -> pd.DataFrame:
        if date_key not in self.log:
            self.log[date_key] = load_log_day(self.paths, date_key)
        return self.log[date_key]


def filter_between(df: pd.DataFrame, start_ts: int, end_ts: int) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    return df[(df["timestamp"] >= start_ts) & (df["timestamp"] < end_ts)].copy()


def load_raw_trace_full_window(paths: BankDataPaths, window: QueryWindow, max_rows: int = 0,
                               chunksize: int = 500_000) -> pd.DataFrame:
    cols = ["timestamp", "cmdb_id", "parent_id", "span_id", "trace_id", "duration"]
    lo_ms = int(window.start_ts) * 1000
    hi_ms = int(window.end_ts) * 1000
    parts = []
    total = 0
    for date_key in window.date_keys:
        path = paths.telemetry_dir(date_key) / "trace" / "trace_span.csv"
        if not path.exists() or path.stat().st_size == 0:
            continue
        for chunk in pd.read_csv(path, usecols=cols, chunksize=chunksize,
                                 dtype={"parent_id": str, "span_id": str, "trace_id": str}):
            sel = chunk[(chunk["timestamp"] >= lo_ms) & (chunk["timestamp"] <= hi_ms)]
            if sel.empty:
                continue
            if max_rows and max_rows > 0:
                remaining = int(max_rows) - total
                if remaining <= 0:
                    break
                if len(sel) > remaining:
                    sel = sel.head(remaining)
            parts.append(sel)
            total += len(sel)
        if max_rows and max_rows > 0 and total >= int(max_rows):
            break
    if not parts:
        return pd.DataFrame(columns=cols)
    return pd.concat(parts, ignore_index=True)


def parse_modalities(value: str) -> set[str]:
    allowed = {"metric", "log", "trace"}
    modalities = {part.strip().lower() for part in str(value).split(",") if part.strip()}
    unknown = modalities - allowed
    if unknown:
        raise ValueError(f"unknown modalities: {sorted(unknown)}; allowed={sorted(allowed)}")
    return modalities


def empty_metric_df() -> pd.DataFrame:
    return pd.DataFrame(columns=["timestamp", "cmdb_id", "kpi_name", "value"])


def empty_log_df() -> pd.DataFrame:
    return pd.DataFrame(columns=["timestamp", "cmdb_id", "value"])


def gather_window(cache: DayCache, window: QueryWindow, modalities: set[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = []
    logs = []
    for date_key in window.date_keys:
        if "metric" in modalities:
            metrics.append(filter_between(cache.metric_day(date_key), window.start_ts, window.end_ts))
        if "log" in modalities:
            logs.append(filter_between(cache.log_day(date_key), window.start_ts, window.end_ts))
    metric_df = pd.concat(metrics, ignore_index=True) if metrics else empty_metric_df()
    log_df = pd.concat(logs, ignore_index=True) if logs else empty_log_df()
    return metric_df, log_df


def date_keys_between(start_ts: int, end_ts: int) -> list[str]:
    start = datetime.fromtimestamp(start_ts).date()
    end = datetime.fromtimestamp(max(start_ts, end_ts - 1)).date()
    keys = []
    day = start
    while day <= end:
        keys.append(day.strftime("%Y_%m_%d"))
        day += timedelta(days=1)
    return keys


def gather_metric_range(cache: DayCache, start_ts: int, end_ts: int, modalities: set[str]) -> pd.DataFrame:
    if "metric" not in modalities or end_ts <= start_ts:
        return empty_metric_df()
    frames = []
    for date_key in date_keys_between(start_ts, end_ts):
        try:
            frames.append(filter_between(cache.metric_day(date_key), start_ts, end_ts))
        except FileNotFoundError:
            continue
    return pd.concat(frames, ignore_index=True) if frames else empty_metric_df()


def reason_for_metric(kpi_name: str) -> str | None:
    kpi = str(kpi_name)
    if kpi_in_bucket(kpi, "cpu"):
        return "high JVM CPU load" if "JVM" in kpi or "Tomcat" in kpi else "high CPU usage"
    if kpi_in_bucket(kpi, "memory"):
        if any(token in kpi for token in ("JVM", "Heap", "Tomcat-MEMORY")):
            return "JVM Out of Memory (OOM) Heap"
        return "high memory usage"
    if kpi_in_bucket(kpi, "filesystem"):
        return "high disk space usage"
    if kpi_in_bucket(kpi, "disk"):
        return "high disk I/O read usage"
    if kpi_in_bucket(kpi, "network"):
        packet_tokens = ("Packet", "Packets", "Err", "rejected", "Aborted", "TCP")
        return "network packet loss" if any(token in kpi for token in packet_tokens) else "network latency"
    return None


def reason_for_log(text: str) -> tuple[str, float] | None:
    s = str(text)
    low = s.lower()
    if any(token in s for token in ("OutOfMemoryError", "Full GC", "SIGKILL")) or "oom" in low:
        return "JVM Out of Memory (OOM) Heap", 8.0
    if any(token in low for token in ("reset by peer", "broken pipe", "connection reset", "retry")):
        return "network packet loss", 5.0
    if any(token in low for token in ("timeout", "timed out", "connection refused")):
        return "network latency", 5.0
    if "no space" in low or "disk" in low:
        return "high disk space usage", 4.0
    return None


def metric_seeds(metric_df: pd.DataFrame, baseline: BaselineStore) -> list[EventSeed]:
    out = []
    if metric_df.empty:
        return out
    for row in metric_df.itertuples(index=False):
        reason = reason_for_metric(row.kpi_name)
        if reason is None:
            continue
        result = baseline.is_anomalous(row.cmdb_id, row.kpi_name, row.value, threshold="p99")
        if result.is_anomalous:
            out.append(EventSeed(
                int(row.timestamp // 60 * 60),
                str(row.cmdb_id),
                reason,
                float(abs(result.deviation)),
                "metric",
            ))
    return out


def log_seeds(log_df: pd.DataFrame) -> list[EventSeed]:
    out = []
    if log_df.empty or "value" not in log_df.columns:
        return out
    for row in log_df.itertuples(index=False):
        hit = reason_for_log(getattr(row, "value", ""))
        if hit:
            reason, score = hit
            out.append(EventSeed(int(row.timestamp // 60 * 60), str(row.cmdb_id), reason, score, "log"))
    return out


def _summary_ts_to_seconds(value) -> int | None:
    if value is None:
        return None
    ts = int(value)
    if ts > 10_000_000_000:
        ts = ts // 1000
    return int(ts // 60 * 60)


def trace_seeds(summary: dict | None) -> list[EventSeed]:
    if not summary or summary.get("trace_status") != "present":
        return []
    out = []
    for edge in summary.get("events", {}).get("slow_edges", []):
        ts = _summary_ts_to_seconds(edge.get("first_timestamp"))
        if ts is None:
            continue
        score = float(edge.get("slow_ratio", 0.0) or 0.0)
        for component in {str(edge.get("src")), str(edge.get("dst"))}:
            if component and component != "None":
                out.append(EventSeed(ts, component, "network latency", score, "trace_slow_edge"))
    for edge in summary.get("events", {}).get("dropped_edges", []):
        ts = _summary_ts_to_seconds(edge.get("first_timestamp"))
        if ts is None:
            continue
        score = float(edge.get("count_drop_ratio", 0.0) or 0.0) * 10.0
        for component in {str(edge.get("src")), str(edge.get("dst"))}:
            if component and component != "None":
                out.append(EventSeed(ts, component, "network packet loss", score, "trace_edge_drop"))
    first = summary.get("events", {}).get("first_anomalous_service")
    if first:
        window = summary.get("window", {})
        try:
            ts = int(datetime.strptime(window["start"], "%Y-%m-%d %H:%M:%S").timestamp() // 60 * 60)
            out.append(EventSeed(ts, str(first), "network latency", 2.0, "trace_first_anomaly"))
        except (KeyError, ValueError):
            pass
    return out


def aggregate_seeds(seeds: list[EventSeed], limit: int) -> list[EventSeed]:
    merged: dict[tuple[int, str, str], EventSeed] = {}
    for seed in seeds:
        key = (seed.timestamp, seed.component, seed.reason)
        old = merged.get(key)
        if old is None:
            merged[key] = seed
        else:
            merged[key] = EventSeed(seed.timestamp, seed.component, seed.reason, old.score + seed.score, old.source + "+" + seed.source)
    return sorted(merged.values(), key=lambda s: (-s.score, s.timestamp, s.component, s.reason))[:limit]


def file_status(path: Path) -> str:
    if not path.exists() or path.stat().st_size == 0:
        return "missing"
    return "present"


def known_services(node_graph: dict) -> list[str]:
    services = sorted(str(s) for s in node_graph.get("containers", {}).keys())
    if services:
        return services
    return ["IG01", "IG02", "MG01", "MG02", "Mysql01", "Mysql02", "Redis01", "Redis02", "Tomcat01", "Tomcat02", "Tomcat03", "Tomcat04", "apache01", "apache02"]


def refine(seed: EventSeed, engine: RuleEngine, cache: DayCache, paths: BankDataPaths,
           baseline: BaselineStore, node_graph: dict, services: list[str], fault_window: int,
           trace_summary: dict | None = None, modalities: set[str] | None = None,
           llm_client=None, force_llm: bool = False, llm_max_calls: int = 8) -> tuple[dict, float, dict | None]:
    modalities = modalities or {"metric", "log", "trace"}
    date_key = datetime.fromtimestamp(seed.timestamp).strftime("%Y_%m_%d")
    if "metric" in modalities:
        metric_df = filter_window(cache.metric_day(date_key), seed.timestamp, fault_window)
        metric_status = file_status(paths.metric_container_csv(date_key)) if metric_df.empty else "present"
    else:
        metric_df = empty_metric_df()
        metric_status = "disabled"
    if "log" in modalities:
        log_df = filter_window(cache.log_day(date_key), seed.timestamp, fault_window)
        log_status = file_status(paths.log_service_csv(date_key)) if log_df.empty else "present"
    else:
        log_df = empty_log_df()
        log_status = "disabled"
    modal_status = {
        "metric": metric_status,
        "log": log_status,
        "trace": (trace_summary or {}).get("trace_status", "unloaded") if "trace" in modalities else "disabled",
    }
    base_evidence = EvidenceQuery(metric_df, baseline, node_graph=node_graph, log_df=log_df, trace_df=None, modal_status=modal_status)
    evidence = SummaryBackedEvidence(base_evidence, trace_summary if "trace" in modalities else None)
    ranked = engine.rank_candidates([Candidate(service, seed.reason) for service in services], evidence)
    component = ranked[0].candidate.service if ranked else seed.component
    support = ranked[0].support_strength if ranked else 0.0
    pred = {
        "root cause occurrence datetime": datetime.fromtimestamp(seed.timestamp).strftime("%Y-%m-%d %H:%M:%S"),
        "root cause component": component,
        "root cause reason": seed.reason,
    }
    llm_arbitration = None
    if llm_client is not None:
        llm_arbitration = run_llm_arbitration(
            f"{date_key}__{seed.component}__{seed.timestamp}",
            ranked[:5],
            llm_client,
            context={
                "modal_status": modal_status,
                "trace_summary": trace_summary,
                "seed": seed.__dict__,
            },
            force=force_llm,
            max_calls=llm_max_calls,
        )
    return pred, seed.score + support, llm_arbitration


def fallback(window: QueryWindow, services: list[str]) -> EventSeed:
    return EventSeed(int(window.start_ts // 60 * 60), services[0] if services else "", "network latency", 0.0, "fallback")


def load_query_trace_summary(base_dir: str, row_id: int) -> dict | None:
    if not base_dir:
        return None
    path = Path(base_dir) / f"query_{int(row_id):03d}.json"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def default_checkpoint_path(out_path: str) -> Path:
    path = Path(out_path)
    return path.with_suffix(path.suffix + ".checkpoint.jsonl")


def load_checkpoint(path: Path) -> dict[int, dict]:
    rows: dict[int, dict] = {}
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"bad checkpoint JSON at {path}:{line_no}") from exc
            rows[int(row["row_id"])] = row
    return rows


def append_checkpoint(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def write_final_outputs(completed: dict[int, dict], out_path: Path, debug_path: Path) -> None:
    ordered = [completed[row_id] for row_id in sorted(completed)]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([
        {
            "row_id": int(row["row_id"]),
            "prediction": json.dumps(row["prediction"], ensure_ascii=False),
        }
        for row in ordered
    ]).to_csv(out_path, index=False)
    debug_path.write_text(json.dumps({
        "n": len(ordered),
        "debug": [row["debug"] for row in ordered],
    }, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    modalities = parse_modalities(args.modalities)
    paths = BankDataPaths.from_root(args.data_root)
    cache = DayCache(paths)
    baseline = BaselineStore.load_json(args.baseline)
    with Path(args.node_graph).open("r", encoding="utf-8") as f:
        node_graph = json.load(f)
    services = known_services(node_graph)
    rules_path = Path(args.rules)
    engine = RuleEngine(RuleSet.load_json(rules_path) if rules_path.exists() else default_rule_set())
    llm_client = llm_client_from_env(args.llm_provider or None) if args.use_llm else None
    confidence_calibrator = RuntimeConfidenceCalibrator.from_json(args.confidence_calibration) if args.confidence_calibration else None
    selector = JointAnswerSelector(
        engine,
        baseline,
        node_graph,
        services,
        llm_client=llm_client,
        confidence_calibrator=confidence_calibrator,
        config=JointSelectorConfig(
            llm_enabled=args.use_llm,
            force_llm=args.force_llm,
            max_llm_calls=args.llm_max_calls,
        ),
    )
    fallback_selector = JointAnswerSelector(
        engine,
        baseline,
        node_graph,
        services,
        llm_client=None,
        confidence_calibrator=confidence_calibrator,
        config=JointSelectorConfig(llm_enabled=False),
    )
    query_df = pd.read_csv(args.query_csv)
    if args.limit:
        query_df = query_df.head(args.limit)
    checkpoint_path = Path(args.checkpoint_jsonl) if args.checkpoint_jsonl else default_checkpoint_path(args.out)
    if checkpoint_path.exists() and not args.resume_checkpoint:
        checkpoint_path.unlink()
    completed = load_checkpoint(checkpoint_path) if args.resume_checkpoint else {}
    total = len(query_df)
    for idx, row in query_df.iterrows():
        row_id = int(idx)
        if row_id in completed:
            if (row_id + 1) % 20 == 0 or row_id + 1 == total:
                print(f"skipped {row_id + 1}/{total} from checkpoint", flush=True)
            continue
        window = parse_query_window(row["instruction"])
        trace_summary = load_query_trace_summary(args.query_trace_summary_dir, int(idx))
        trace_summary_for_run = trace_summary if "trace" in modalities else None
        metric_df, log_df = gather_window(cache, window, modalities)
        trace_df = None
        if args.raw_trace_full_window and "trace" in modalities:
            trace_df = load_raw_trace_full_window(paths, window, max_rows=args.max_trace_rows_per_query)
            trace_summary_for_run = None
        background_metric_df = gather_metric_range(
            cache,
            window.start_ts - args.background_baseline_window,
            window.start_ts,
            modalities,
        )
        modal_status = {
            "metric": "present" if "metric" in modalities and not metric_df.empty else ("empty_window" if "metric" in modalities else "disabled"),
            "log": "present" if "log" in modalities and not log_df.empty else ("empty_window" if "log" in modalities else "disabled"),
            "trace": (
                "present" if trace_df is not None and not trace_df.empty
                else (trace_summary_for_run or {}).get("trace_status", "unloaded")
            ) if "trace" in modalities else "disabled",
        }
        llm_error = None
        try:
            selection = selector.select(
                case_id=f"query_{int(idx):03d}",
                metric_df=metric_df,
                log_df=log_df,
                trace_summary=trace_summary_for_run,
                modal_status=modal_status,
                trace_df=trace_df,
                failure_count=window.failure_count,
                window_start_ts=window.start_ts,
                background_metric_df=background_metric_df,
            )
        except LLMApiError as exc:
            if args.llm_error_policy == "raise":
                raise
            llm_error = str(exc)
            selection = fallback_selector.select(
                case_id=f"query_{int(idx):03d}",
                metric_df=metric_df,
                log_df=log_df,
                trace_summary=trace_summary_for_run,
                modal_status=modal_status,
                trace_df=trace_df,
                failure_count=window.failure_count,
                window_start_ts=window.start_ts,
                background_metric_df=background_metric_df,
            )
        prediction = selection.prediction
        debug_row = {
            "row_id": row_id,
            "window": {"start": window.start.strftime("%Y-%m-%d %H:%M:%S"), "end": window.end.strftime("%Y-%m-%d %H:%M:%S"), "failure_count": window.failure_count},
            "modalities": sorted(modalities),
            "selection": selection.to_dict(),
            "raw_trace_rows": int(len(trace_df)) if trace_df is not None else 0,
        }
        if llm_error:
            debug_row["llm_error"] = llm_error
            debug_row["llm_error_policy"] = args.llm_error_policy
        checkpoint_row = {
            "row_id": row_id,
            "prediction": prediction,
            "debug": debug_row,
        }
        completed[row_id] = checkpoint_row
        append_checkpoint(checkpoint_path, checkpoint_row)
        if (row_id + 1) % 20 == 0 or len(completed) == total:
            print(f"processed {len(completed)}/{total}", flush=True)
    out = Path(args.out)
    debug_path = Path(args.debug_json)
    if len(completed) != total:
        missing = sorted(set(int(i) for i in query_df.index) - set(completed))
        raise RuntimeError(f"incomplete run: completed={len(completed)} total={total} missing_head={missing[:10]}")
    write_final_outputs(completed, out, debug_path)
    print(f"wrote {out}")
    print(f"wrote {debug_path}")
    print(f"checkpoint {checkpoint_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
