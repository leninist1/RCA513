"""Audit executable Scheme B rules on Bank records.

This script intentionally reports rule behavior before reporting RCA accuracy.
The first safety question is whether a rule wrongly refutes the true root.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = PROJECT_ROOT.parent
for path in (PROJECT_ROOT, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from refute.src.baseline_distributions import BaselineStore  # noqa: E402
from refute.src.data_loader import (  # noqa: E402
    BankDataPaths,
    filter_window,
    load_log_day,
    load_metric_day,
    load_records,
    load_trace_windows_for_cases,
)
from refute.src.evidence_query import EvidenceQuery  # noqa: E402
from refute_b_v2.audit import summarize_rule_results  # noqa: E402
from refute_b_v2.default_rules import default_rule_set  # noqa: E402
from refute_b_v2.rule_engine import RuleEngine  # noqa: E402
from refute_b_v2.rules import Candidate, RuleSet  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="/home/yan/workspace/data/openrca/Bank")
    parser.add_argument("--baseline", default="../refute/knowledge/baseline_distributions_all_metric_dates.json")
    parser.add_argument("--node-graph", default="../refute/knowledge/node_container_graph_2021_03_10.json")
    parser.add_argument("--rules", default="knowledge/refutation_rules_v2.json")
    parser.add_argument("--out", default="logs/rule_audit_bank136.json")
    parser.add_argument("--fault-window", type=int, default=600)
    parser.add_argument("--use-trace", action="store_true")
    parser.add_argument("--trace-chunksize", type=int, default=500_000)
    parser.add_argument("--trace-max-rows", type=int, default=120_000)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def file_status(path: Path) -> str:
    if not path.exists() or path.stat().st_size == 0:
        return "missing"
    return "present"


def merge_rule_summaries(target: dict, source: dict) -> None:
    for rule_id, values in source.items():
        row = target.setdefault(rule_id, defaultdict(int))
        for key, value in values.items():
            row[key] += int(value)


def known_services_from_graph(node_graph: dict) -> list[str]:
    services = sorted(str(s) for s in node_graph.get("containers", {}).keys())
    if services:
        return services
    return [
        "IG01", "IG02", "MG01", "MG02",
        "Mysql01", "Mysql02", "Redis01", "Redis02",
        "Tomcat01", "Tomcat02", "Tomcat03", "Tomcat04",
        "apache01", "apache02",
    ]


def main() -> int:
    args = parse_args()
    root = Path(args.data_root)
    paths = BankDataPaths.from_root(root)
    records = load_records(root)
    if args.limit:
        records = records.head(args.limit)

    baseline = BaselineStore.load_json(args.baseline)
    with Path(args.node_graph).open("r", encoding="utf-8") as f:
        node_graph = json.load(f)
    services = known_services_from_graph(node_graph)
    rules_path = Path(args.rules)
    rules = RuleSet.load_json(rules_path) if rules_path.exists() else default_rule_set()
    engine = RuleEngine(rules)

    audit = {}
    by_reason = {}
    case_rows = []

    for date_key, day_records in records.groupby("date_key", sort=True):
        metric_day = load_metric_day(paths, date_key)
        log_day = load_log_day(paths, date_key)
        metric_file_status = file_status(paths.metric_container_csv(date_key))
        log_file_status = file_status(paths.log_service_csv(date_key))
        trace_file = paths.telemetry_dir(date_key) / "trace" / "trace_span.csv"
        trace_file_status = file_status(trace_file)

        trace_windows = {}
        if args.use_trace and trace_file_status == "present":
            timestamps = [int(row.timestamp) for row in day_records.itertuples(index=False)]
            trace_windows = load_trace_windows_for_cases(
                paths,
                date_key,
                timestamps,
                args.fault_window,
                chunksize=args.trace_chunksize,
                max_rows_per_case=args.trace_max_rows,
            )

        for rec in day_records.itertuples(index=False):
            ts = int(rec.timestamp)
            metric_df = filter_window(metric_day, ts, args.fault_window)
            log_df = filter_window(log_day, ts, args.fault_window)
            trace_df = trace_windows.get(ts) if args.use_trace else None
            trace_status = (
                "unloaded" if not args.use_trace else
                trace_file_status if trace_file_status != "present" else
                ("present" if trace_df is not None and not trace_df.empty else "empty_window")
            )
            modal_status = {
                "metric": metric_file_status if metric_file_status != "present" else ("present" if not metric_df.empty else "empty_window"),
                "log": log_file_status if log_file_status != "present" else ("present" if not log_df.empty else "empty_window"),
                "trace": trace_status,
            }
            evidence = EvidenceQuery(
                metric_df,
                baseline,
                node_graph=node_graph,
                log_df=log_df,
                trace_df=trace_df,
                modal_status=modal_status,
            )
            candidates = [Candidate(service, rec.reason) for service in services]
            results = [engine.run_candidate(candidate, evidence) for candidate in candidates]
            case_audit = summarize_rule_results(results, true_service=str(rec.component))
            merge_rule_summaries(audit, case_audit)
            reason_audit = by_reason.setdefault(str(rec.reason), {})
            merge_rule_summaries(reason_audit, case_audit)
            ranked = sorted(results, key=lambda r: (-r.support_strength, r.hard_refute, r.blind_count, r.candidate.service))
            top5 = [row.candidate.service for row in ranked[:5]]
            case_rows.append({
                "date": date_key,
                "datetime": rec.datetime,
                "true_component": rec.component,
                "reason": rec.reason,
                "modal_status": modal_status,
                "top5_by_rule_support": top5,
                "top1_hit": bool(top5 and top5[0] == rec.component),
                "top3_hit": rec.component in top5[:3],
            })

    total = len(case_rows)
    summary = {
        "n": total,
        "top1": sum(row["top1_hit"] for row in case_rows),
        "top3": sum(row["top3_hit"] for row in case_rows),
        "rule_audit": {k: dict(v) for k, v in sorted(audit.items())},
        "by_reason": {
            reason: {rule_id: dict(values) for rule_id, values in sorted(rows.items())}
            for reason, rows in sorted(by_reason.items())
        },
        "cases": case_rows,
    }
    summary["top1_rate"] = summary["top1"] / total if total else 0.0
    summary["top3_rate"] = summary["top3"] / total if total else 0.0

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: summary[k] for k in ["n", "top1", "top3", "top1_rate", "top3_rate"]}, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
