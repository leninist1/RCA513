"""Run Scheme B v2 executable rules on Bank cases and emit signatures."""
from __future__ import annotations

import argparse
import json
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
)
from refute.src.evidence_query import EvidenceQuery  # noqa: E402
from refute_b_v2.default_rules import default_rule_set  # noqa: E402
from refute_b_v2.evidence_adapter import SummaryBackedEvidence  # noqa: E402
from refute_b_v2.evidence_signature import case_signature_from_results  # noqa: E402
from refute_b_v2.llm_arbitration import run_llm_arbitration  # noqa: E402
from refute_b_v2.llm_clients import llm_client_from_env  # noqa: E402
from refute_b_v2.rule_engine import RuleEngine  # noqa: E402
from refute_b_v2.rules import Candidate, RuleSet  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="/home/yan/workspace/data/openrca/Bank")
    parser.add_argument("--baseline", default="../refute/knowledge/baseline_distributions_all_metric_dates.json")
    parser.add_argument("--node-graph", default="../refute/knowledge/node_container_graph_2021_03_10.json")
    parser.add_argument("--rules", default="knowledge/refutation_rules_v2.json")
    parser.add_argument("--out", default="logs/bank_cases_v2.json")
    parser.add_argument("--fault-window", type=int, default=600)
    parser.add_argument("--date", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--include-cards", action="store_true")
    parser.add_argument("--trace-summary-dir", default="")
    parser.add_argument("--use-llm", action="store_true", help="call real Claude/DeepSeek client for ambiguous evidence matrices")
    parser.add_argument("--force-llm", action="store_true", help="call LLM for every case when --use-llm is set")
    parser.add_argument("--llm-provider", default="", help="claude/shqbb/deepseek; defaults to SCHEME_B_LLM_PROVIDER")
    parser.add_argument("--llm-max-calls", type=int, default=8)
    return parser.parse_args()


def file_status(path: Path) -> str:
    if not path.exists() or path.stat().st_size == 0:
        return "missing"
    return "present"


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


def confidence_features(ranked) -> dict:
    if not ranked:
        return {
            "support_strength": 0.0,
            "top1_top2_gap": 0.0,
            "hard_support_count": 0,
            "hard_refute_count": 0,
            "blind_count": 0,
        }
    top = ranked[0]
    second = ranked[1] if len(ranked) > 1 else None
    return {
        "support_strength": top.support_strength,
        "top1_top2_gap": top.support_strength - (second.support_strength if second else 0.0),
        "hard_support_count": top.hard_support,
        "hard_refute_count": top.hard_refute,
        "blind_count": top.blind_count,
    }


def trace_summary_path(base_dir: str, date_key: str, timestamp: int, component: str) -> Path | None:
    if not base_dir:
        return None
    return Path(base_dir) / f"{date_key}__{int(timestamp)}__{component}.json"


def main() -> int:
    args = parse_args()
    root = Path(args.data_root)
    paths = BankDataPaths.from_root(root)
    records = load_records(root)
    if args.date:
        records = records[records["date_key"] == args.date]
    if args.limit:
        records = records.head(args.limit)

    baseline = BaselineStore.load_json(args.baseline)
    with Path(args.node_graph).open("r", encoding="utf-8") as f:
        node_graph = json.load(f)
    services = known_services_from_graph(node_graph)
    rules_path = Path(args.rules)
    rules = RuleSet.load_json(rules_path) if rules_path.exists() else default_rule_set()
    engine = RuleEngine(rules)
    llm_client = llm_client_from_env(args.llm_provider or None) if args.use_llm else None

    cases = []
    for date_key, day_records in records.groupby("date_key", sort=True):
        metric_day = load_metric_day(paths, date_key)
        log_day = load_log_day(paths, date_key)
        metric_file_status = file_status(paths.metric_container_csv(date_key))
        log_file_status = file_status(paths.log_service_csv(date_key))
        for rec in day_records.itertuples(index=False):
            ts = int(rec.timestamp)
            metric_df = filter_window(metric_day, ts, args.fault_window)
            log_df = filter_window(log_day, ts, args.fault_window)
            summary_path = trace_summary_path(args.trace_summary_dir, date_key, ts, str(rec.component))
            trace_status = "unloaded"
            trace_summary = None
            if summary_path and summary_path.exists():
                try:
                    with summary_path.open("r", encoding="utf-8") as f:
                        trace_summary = json.load(f)
                        trace_status = trace_summary.get("trace_status", "missing")
                except json.JSONDecodeError:
                    trace_status = "missing"
            modal_status = {
                "metric": metric_file_status if metric_file_status != "present" else ("present" if not metric_df.empty else "empty_window"),
                "log": log_file_status if log_file_status != "present" else ("present" if not log_df.empty else "empty_window"),
                "trace": trace_status,
            }
            base_evidence = EvidenceQuery(
                metric_df,
                baseline,
                node_graph=node_graph,
                log_df=log_df,
                trace_df=None,
                modal_status=modal_status,
            )
            evidence = SummaryBackedEvidence.with_summary_path(base_evidence, summary_path)
            candidates = [Candidate(service, rec.reason) for service in services]
            ranked = engine.rank_candidates(candidates, evidence)
            top5 = [row.candidate.service for row in ranked[:5]]
            signature = case_signature_from_results(
                f"{date_key}__{rec.component}__{ts}",
                modal_status,
                ranked[:5],
            )
            row = {
                "date": date_key,
                "timestamp": ts,
                "datetime": rec.datetime,
                "true_component": rec.component,
                "reason": rec.reason,
                "top5": top5,
                "top1_hit": bool(top5 and top5[0] == rec.component),
                "top3_hit": rec.component in top5[:3],
                "modal_status": modal_status,
                "confidence_features": confidence_features(ranked),
                "signature": signature.to_dict(),
            }
            if args.include_cards:
                row["ranked"] = [item.to_dict() for item in ranked[:5]]
            if llm_client is not None:
                row["llm_arbitration"] = run_llm_arbitration(
                    f"{date_key}__{rec.component}__{ts}",
                    ranked[:5],
                    llm_client,
                    context={"modal_status": modal_status, "trace_summary": trace_summary},
                    force=args.force_llm,
                    max_calls=args.llm_max_calls,
                )
            cases.append(row)

    n = len(cases)
    summary = {
        "n": n,
        "top1": sum(c["top1_hit"] for c in cases),
        "top3": sum(c["top3_hit"] for c in cases),
        "cases": cases,
    }
    summary["top1_rate"] = summary["top1"] / n if n else 0.0
    summary["top3_rate"] = summary["top3"] / n if n else 0.0

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: summary[k] for k in ["n", "top1", "top3", "top1_rate", "top3_rate"]}, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
