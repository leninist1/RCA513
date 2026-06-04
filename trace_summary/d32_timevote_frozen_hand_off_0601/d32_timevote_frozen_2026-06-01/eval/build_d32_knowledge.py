"""Build d32 Layer-1 knowledge artifacts from OpenRCA Bank cases."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
for path in (PROJECT_ROOT, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from refute.src.baseline_distributions import BaselineStore  # noqa: E402
from refute.src.data_loader import BankDataPaths, load_log_day, load_metric_day  # noqa: E402
from refute_b_v2_d32.layer1 import KnowledgeBuildConfig, build_knowledge, case_rows_from_openrca, save_knowledge  # noqa: E402


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="/home/yan/workspace/data/openrca/Bank")
    parser.add_argument("--query-csv", default="/home/yan/workspace/data/openrca/Bank/query.csv")
    parser.add_argument("--record-csv", default="/home/yan/workspace/data/openrca/Bank/record.csv")
    parser.add_argument("--baseline", default="../refute/knowledge/baseline_distributions_all_metric_dates.json")
    parser.add_argument("--trace-summary-dir", default="trace_summaries/openrca_queries_full_batch18")
    parser.add_argument("--modalities", default="metric,log,trace")
    parser.add_argument("--cluster-threshold", type=float, default=0.48)
    parser.add_argument("--min-rule-support", type=int, default=3)
    parser.add_argument("--min-rule-confidence", type=float, default=0.62)
    parser.add_argument("--mode", choices=["lodo", "split", "all-train"], default="lodo")
    parser.add_argument("--split-last-dates", type=int, default=1)
    parser.add_argument("--knowledge-dir", default="knowledge/d32_lodo_default")
    parser.add_argument("--out", default="knowledge/d32_layer1_knowledge.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    modalities = {part.strip() for part in args.modalities.split(",") if part.strip()}
    paths = BankDataPaths.from_root(args.data_root)
    cache = DayCache(paths)
    baseline = BaselineStore.load_json(args.baseline)
    case_rows = case_rows_from_openrca(args.query_csv, args.record_csv)
    config = KnowledgeBuildConfig(
        cluster_threshold=args.cluster_threshold,
        min_rule_support=args.min_rule_support,
        min_rule_confidence=args.min_rule_confidence,
    )
    if args.mode == "all-train":
        knowledge = build_knowledge(
            case_rows=case_rows,
            cache=cache,
            baseline=baseline,
            trace_summary_dir=args.trace_summary_dir,
            modalities=modalities,
            config=config,
        )
        save_knowledge(knowledge, args.out)
        print({
            "mode": "all-train",
            "warning": "all-train is for inspection only; do not report it as the main result.",
            "out": args.out,
            "n_cases": len(knowledge["cases"]),
            "n_clusters": len(knowledge["clusters"]),
            "n_mined_rules": len(knowledge["mined_rules"]),
        })
        return 0

    dates = sorted({row["date_key"] for row in case_rows})
    out_dir = Path(args.knowledge_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"mode": args.mode, "artifacts": []}
    if args.mode == "lodo":
        folds = [(date, [row for row in case_rows if row["date_key"] != date]) for date in dates]
    else:
        heldout = set(dates[-max(1, int(args.split_last_dates)):])
        folds = [("split_train", [row for row in case_rows if row["date_key"] not in heldout])]
        manifest["heldout_dates"] = sorted(heldout)
    for fold_name, train_rows in folds:
        knowledge = build_knowledge(
            case_rows=train_rows,
            cache=cache,
            baseline=baseline,
            trace_summary_dir=args.trace_summary_dir,
            modalities=modalities,
            config=config,
        )
        path = out_dir / f"d32_knowledge_{fold_name}.json"
        save_knowledge(knowledge, path)
        manifest["artifacts"].append({
            "fold": fold_name,
            "path": str(path),
            "n_cases": len(knowledge["cases"]),
            "n_clusters": len(knowledge["clusters"]),
            "n_mined_rules": len(knowledge["mined_rules"]),
        })
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print({"mode": args.mode, "manifest": str(manifest_path), "n_artifacts": len(manifest["artifacts"])})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
