"""Leave-one-date-out d32 OpenRCA simulation."""
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
from refute.src.data_loader import BankDataPaths  # noqa: E402
from refute.src.data_loader import load_log_day, load_metric_day  # noqa: E402
from refute_b_v2.query_windows import parse_query_window  # noqa: E402
from refute_b_v2_d32.evidence_summarizer import EvidenceSummaryLimits, build_summary_cards  # noqa: E402
from refute_b_v2_d32.layer1 import D32Knowledge, KnowledgeBuildConfig, build_knowledge, case_rows_from_openrca, load_trace_summary  # noqa: E402
from refute_b_v2_d32.layer2 import D32PipelineConfig, D32RefutationPipeline  # noqa: E402
from refute_b_v2_d32.llm_gated_rerank import GatedRerankConfig, apply_gated_rerank, apply_pairwise_gated_rerank, load_judgment_jsonl  # noqa: E402
from refute_b_v2_d32.llm_candidate_judge import LLMJudgeConfig, write_shadow_judgments_jsonl  # noqa: E402
from refute_b_v2_d32.llm_pairwise_judge import build_pairwise_requests, write_pairwise_judgments_jsonl  # noqa: E402
from refute_b_v2_d32.reason_classifier import contiguous_case_folds  # noqa: E402


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


def known_services(node_graph: dict) -> list[str]:
    services = sorted(str(s) for s in node_graph.get("containers", {}).keys())
    return services or ["IG01", "IG02", "MG01", "MG02", "Mysql01", "Mysql02", "Redis01", "Redis02", "Tomcat01", "Tomcat02", "Tomcat03", "Tomcat04", "apache01", "apache02"]


def load_rules(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="/home/yan/workspace/data/openrca/Bank")
    parser.add_argument("--query-csv", default="/home/yan/workspace/data/openrca/Bank/query.csv")
    parser.add_argument("--record-csv", default="/home/yan/workspace/data/openrca/Bank/record.csv")
    parser.add_argument("--baseline", default="../refute/knowledge/baseline_distributions_all_metric_dates.json")
    parser.add_argument("--node-graph", default="../refute/knowledge/node_container_graph_2021_03_10.json")
    parser.add_argument("--rules", default="knowledge/refutation_rules_v2.json")
    parser.add_argument("--trace-summary-dir", default="trace_summaries/openrca_queries_full_batch18")
    parser.add_argument("--modalities", default="metric,log,trace")
    parser.add_argument("--knowledge-dir", default="knowledge/d32_lodo")
    parser.add_argument("--out", default="logs/d32_lodo_trace_predictions.csv")
    parser.add_argument("--debug-json", default="logs/d32_lodo_trace_debug.json")
    parser.add_argument("--checkpoint-jsonl", default="logs/d32_lodo_trace_checkpoint.jsonl")
    parser.add_argument("--resume-checkpoint", action="store_true")
    parser.add_argument("--llm-mode", choices=["off", "shadow", "gated-rerank"], default="off")
    parser.add_argument("--llm-provider", choices=["openai_compatible"], default="openai_compatible")
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--llm-base-url", default=None)
    parser.add_argument("--llm-api-key-env", default="RCA_LLM_API_KEY")
    parser.add_argument("--llm-output-jsonl", default=None)
    parser.add_argument("--llm-rerank-output-jsonl", default=None)
    parser.add_argument("--llm-pairwise-output-jsonl", default=None)
    parser.add_argument("--llm-timeout-sec", type=float, default=60.0)
    parser.add_argument("--llm-temperature", type=float, default=0.0)
    parser.add_argument("--llm-max-retries", type=int, default=2)
    parser.add_argument("--llm-concurrency", type=int, default=1)
    parser.add_argument("--llm-refute-threshold", type=float, default=0.80)
    parser.add_argument("--llm-support-threshold", type=float, default=0.75)
    parser.add_argument("--llm-pairwise-margin-threshold", type=float, default=0.25)
    parser.add_argument("--llm-pairwise-top1-max-support", type=float, default=0.50)
    parser.add_argument("--llm-allow-pairwise-low-top1-support", action="store_true")
    parser.add_argument("--llm-min-alt-rank", type=int, default=2)
    parser.add_argument("--llm-max-alt-rank", type=int, default=None)
    parser.add_argument("--llm-rerank-policy", choices=["conservative"], default="conservative")
    parser.add_argument("--llm-rerank-judge", choices=["candidate", "pairwise"], default="candidate")
    parser.add_argument("--llm-summary-jsonl", default=None)
    parser.add_argument("--llm-top-k", type=int, default=5)
    parser.add_argument("--summary-max-metric-patterns", type=int, default=5)
    parser.add_argument("--summary-max-log-patterns", type=int, default=5)
    parser.add_argument("--summary-max-trace-edges", type=int, default=5)
    parser.add_argument("--summary-max-counter-evidence", type=int, default=5)
    parser.add_argument("--dataset-name", default="bank")
    parser.add_argument("--casefold-classifier-dir", default="knowledge/casefold_classifiers")
    parser.add_argument("--disable-family-filter", action="store_true",
                        help="Disable reason→component family filter (keeps all families per reason)")
    parser.add_argument("--disable-joint-candidates", action="store_true")
    parser.add_argument("--joint-beam-per-reason", type=int, default=8)
    parser.add_argument("--joint-prior-scale", type=float, default=0.15)
    parser.add_argument("--joint-prior-offset", type=float, default=0.05)
    parser.add_argument("--max-joint-prior", type=float, default=1.8)
    parser.add_argument("--legacy-component-evidence-scale-with-joint", type=float, default=1.0)
    parser.add_argument("--disable-window-reason-scores", action="store_true")
    parser.add_argument("--window-reason-score-credit", type=float, default=0.65)
    parser.add_argument("--window-reason-top-k", type=int, default=3)
    parser.add_argument("--window-reason-weak-penalty", type=float, default=0.8)
    parser.add_argument("--window-reason-type-bonus", type=float, default=0.6)
    parser.add_argument("--window-reason-type-penalty", type=float, default=0.4)
    parser.add_argument("--symptom-reason-score-credit", type=float, default=1.0)
    parser.add_argument("--symptom-reason-top-k", type=int, default=8)
    parser.add_argument("--time-anchor-early-bonus", type=float, default=0.25,
                        help="Early vote bonus for time anchor (higher = earlier onset)")
    parser.add_argument("--time-anchor-sustained-bonus", type=float, default=2.0,
                        help="Sustained onset bonus for time anchor (higher = earlier onset)")
    parser.add_argument("--onset-first", dest="onset_first", action="store_true", default=True,
                        help="Enable onset-first time anchor voting (first anomaly per KPI)")
    parser.add_argument("--no-onset-first", dest="onset_first", action="store_false",
                        help="Disable onset-first time anchor voting (use legacy all-anomaly voting)")
    parser.add_argument("--onset-first-weight", type=float, default=4.0,
                        help="Weight multiplier for onset votes vs peak votes")
    parser.add_argument("--onset-decay-seconds", type=float, default=300.0,
                        help="Exponential decay window for onset-first voting (seconds)")
    parser.add_argument("--component-onset-bonus", type=float, default=5.0,
                        help="Bonus for the very first anomaly of the blamed component")
    parser.add_argument("--no-lodo", dest="lodo", action="store_false", default=True,
                        help="Disable leave-one-date-out: build knowledge from all data (in-sample, for baseline comparison)")
    return parser.parse_args()


def gather_window(cache: DayCache, window, modalities: set[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = []
    logs = []
    for date_key in window.date_keys:
        if "metric" in modalities:
            metrics.append(filter_between(cache.metric_day(date_key), window.start_ts, window.end_ts))
        if "log" in modalities:
            logs.append(filter_between(cache.log_day(date_key), window.start_ts, window.end_ts))
    metric_df = pd.concat(metrics, ignore_index=True) if metrics else pd.DataFrame(columns=["timestamp", "cmdb_id", "kpi_name", "value"])
    log_df = pd.concat(logs, ignore_index=True) if logs else pd.DataFrame(columns=["timestamp", "cmdb_id", "value"])
    return metric_df, log_df


def load_checkpoint(path: Path) -> dict[int, dict]:
    out = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            out[int(row["row_id"])] = row
    return out


def append_checkpoint(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def write_predictions_csv(completed: dict[int, dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ordered = [completed[row_id] for row_id in sorted(completed)]
    pd.DataFrame([
        {"row_id": row["row_id"], "prediction": json.dumps(row["prediction"], ensure_ascii=False)}
        for row in ordered
    ]).to_csv(out_path, index=False)


def write_outputs(completed: dict[int, dict], out_path: Path, debug_path: Path) -> None:
    write_predictions_csv(completed, out_path)
    debug_path.parent.mkdir(parents=True, exist_ok=True)
    ordered = [completed[row_id] for row_id in sorted(completed)]
    debug_path.write_text(json.dumps({"n": len(ordered), "debug": [row["debug"] for row in ordered]}, ensure_ascii=False, indent=2), encoding="utf-8")


def collect_llm_summary_cards(
    *,
    completed: dict[int, dict],
    query_df: pd.DataFrame,
    cache: DayCache,
    baseline,
    trace_summary_dir: str,
    modalities: set[str],
    top_k: int,
    limits: EvidenceSummaryLimits,
) -> list[dict]:
    cards_out: list[dict] = []
    for row_id in sorted(completed):
        qrow = query_df.iloc[row_id]
        window = parse_query_window(qrow["instruction"])
        metric_df, log_df = gather_window(cache, window, modalities)
        trace_summary = load_trace_summary(trace_summary_dir, row_id) if "trace" in modalities else None
        modal_status = {
            "metric": "present" if "metric" in modalities and not metric_df.empty else ("empty_window" if "metric" in modalities else "disabled"),
            "log": "present" if "log" in modalities and not log_df.empty else ("empty_window" if "log" in modalities else "disabled"),
            "trace": (trace_summary or {}).get("trace_status", "unloaded") if "trace" in modalities else "disabled",
        }
        d32_result = (completed[row_id].get("debug", {}) or {}).get("d32_result", {}) or {}
        d32_debug = d32_result.get("debug", {}) or {}
        cards_out.extend(build_summary_cards(
            case_id=f"query_{row_id:03d}",
            metric_df=metric_df,
            log_df=log_df,
            trace_summary=trace_summary,
            baseline=baseline,
            modal_status=modal_status,
            d32_debug=d32_debug,
            window_start_ts=window.start_ts,
            top_k=top_k,
            limits=limits,
        ))
    return cards_out


def write_llm_summary_jsonl(
    *,
    completed: dict[int, dict],
    out_path: Path,
    query_df: pd.DataFrame,
    cache: DayCache,
    baseline,
    trace_summary_dir: str,
    modalities: set[str],
    top_k: int,
    limits: EvidenceSummaryLimits,
) -> int:
    cards = collect_llm_summary_cards(
        completed=completed,
        query_df=query_df,
        cache=cache,
        baseline=baseline,
        trace_summary_dir=trace_summary_dir,
        modalities=modalities,
        top_k=top_k,
        limits=limits,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for card in cards:
            f.write(json.dumps(card, ensure_ascii=False, sort_keys=True) + "\n")
    return len(cards)


def default_d32_baseline_path(out_path: Path) -> Path:
    suffix = out_path.suffix or ".csv"
    return out_path.with_name(f"{out_path.stem}.d32_baseline{suffix}")


def default_rerank_summary_path(trace_path: Path) -> Path:
    return trace_path.with_name(f"{trace_path.stem}.summary.json")


def default_pairwise_output_path(trace_path: Path) -> Path:
    return trace_path.with_name(f"{trace_path.stem}.pairwise_judge.jsonl")


def main() -> int:
    args = parse_args()
    modalities = {part.strip() for part in args.modalities.split(",") if part.strip()}
    paths = BankDataPaths.from_root(args.data_root)
    cache = DayCache(paths)
    baseline = BaselineStore.load_json(args.baseline)
    node_graph = json.load(open(args.node_graph, encoding="utf-8"))
    rules = load_rules(args.rules)
    services = known_services(node_graph)
    case_rows = case_rows_from_openrca(args.query_csv, args.record_csv)
    row_fold: dict[int, int] = {}
    for fold_idx, fold_indices in enumerate(contiguous_case_folds(len(case_rows), 2)):
        for case_pos in fold_indices:
            row_fold[int(case_pos)] = int(fold_idx)
    record_df = pd.read_csv(args.record_csv)
    dates = sorted(pd.to_datetime(record_df["datetime"]).dt.strftime("%Y_%m_%d").unique())
    row_date = {idx: pd.to_datetime(record_df.iloc[idx]["datetime"]).strftime("%Y_%m_%d") for idx in range(len(record_df))}
    checkpoint_path = Path(args.checkpoint_jsonl)
    if checkpoint_path.exists() and not args.resume_checkpoint:
        checkpoint_path.unlink()
    completed = load_checkpoint(checkpoint_path) if args.resume_checkpoint else {}
    knowledge_dir = Path(args.knowledge_dir)
    knowledge_dir.mkdir(parents=True, exist_ok=True)

    query_df = pd.read_csv(args.query_csv)
    family_kwargs = {}
    if args.disable_family_filter:
        family_kwargs = {"disable_family_filter": True}

    def _build_pipelines(knowledge: D32Knowledge) -> dict[int, D32RefutationPipeline]:
        pipes: dict[int, D32RefutationPipeline] = {}
        for fold_idx in sorted(set(row_fold.values())):
            pipeline_config = D32PipelineConfig(
                dataset_name=args.dataset_name,
                reason_classifier_fold=fold_idx,
                casefold_classifier_dir=args.casefold_classifier_dir,
                enable_joint_candidates=not args.disable_joint_candidates,
                joint_beam_per_reason=args.joint_beam_per_reason,
                joint_prior_scale=args.joint_prior_scale,
                joint_prior_offset=args.joint_prior_offset,
                max_joint_prior=args.max_joint_prior,
                legacy_component_evidence_scale_with_joint=args.legacy_component_evidence_scale_with_joint,
                enable_window_reason_scores=not args.disable_window_reason_scores,
                window_reason_score_credit=args.window_reason_score_credit,
                window_reason_top_k=args.window_reason_top_k,
                window_reason_weak_penalty=args.window_reason_weak_penalty,
                window_reason_type_bonus=args.window_reason_type_bonus,
                window_reason_type_penalty=args.window_reason_type_penalty,
                symptom_reason_score_credit=args.symptom_reason_score_credit,
                symptom_reason_top_k=args.symptom_reason_top_k,
                time_anchor_early_bonus=args.time_anchor_early_bonus,
                time_anchor_sustained_bonus=args.time_anchor_sustained_bonus,
                onset_first=args.onset_first,
                onset_first_weight=args.onset_first_weight,
                onset_decay_seconds=args.onset_decay_seconds,
                component_onset_bonus=args.component_onset_bonus,
                **family_kwargs,
            )
            pipes[fold_idx] = D32RefutationPipeline(knowledge, rules, baseline, node_graph, services, config=pipeline_config)
        return pipes

    def _process_row(row_id: int, pipelines: dict[int, D32RefutationPipeline], mode_label: str) -> None:
        if row_id in completed:
            return
        qrow = query_df.iloc[row_id]
        window = parse_query_window(qrow["instruction"])
        metric_df, log_df = gather_window(cache, window, modalities)
        trace_summary = load_trace_summary(args.trace_summary_dir, row_id) if "trace" in modalities else None
        modal_status = {
            "metric": "present" if "metric" in modalities and not metric_df.empty else ("empty_window" if "metric" in modalities else "disabled"),
            "log": "present" if "log" in modalities and not log_df.empty else ("empty_window" if "log" in modalities else "disabled"),
            "trace": (trace_summary or {}).get("trace_status", "unloaded") if "trace" in modalities else "disabled",
        }
        fold_idx = row_fold[int(row_id)]
        result = pipelines[fold_idx].select(
            case_id=f"query_{row_id:03d}",
            metric_df=metric_df,
            log_df=log_df,
            trace_summary=trace_summary,
            modal_status=modal_status,
            failure_count=window.failure_count,
            window_start_ts=window.start_ts,
        )
        checkpoint_row = {
            "row_id": row_id,
            "mode": mode_label,
            "reason_classifier_fold": fold_idx,
            "prediction": result.prediction,
            "debug": {
                "row_id": row_id,
                "mode": mode_label,
                "window": {"start": window.start.strftime("%Y-%m-%d %H:%M:%S"), "end": window.end.strftime("%Y-%m-%d %H:%M:%S"), "failure_count": window.failure_count},
                "modal_status": modal_status,
                "d32_result": result.to_dict(),
            },
        }
        if mode_label.startswith("heldout="):
            checkpoint_row["heldout_date"] = mode_label.split("=", 1)[1]
        append_checkpoint(checkpoint_path, checkpoint_row)
        completed[row_id] = checkpoint_row
        print(f"processed {row_id + 1}/{len(case_rows)} {mode_label} fold={fold_idx}", flush=True)

    if args.lodo:
        for heldout in dates:
            train_rows = [row for row in case_rows if row_date[int(row["row_id"])] != heldout]
            test_ids = [idx for idx, date in row_date.items() if date == heldout]
            knowledge_path = knowledge_dir / f"d32_knowledge_without_{heldout}.json"
            if knowledge_path.exists():
                knowledge = D32Knowledge.load_json(knowledge_path)
            else:
                data = build_knowledge(
                    case_rows=train_rows,
                    cache=cache,
                    baseline=baseline,
                    trace_summary_dir=args.trace_summary_dir,
                    modalities=modalities,
                    config=KnowledgeBuildConfig(),
                )
                knowledge_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
                knowledge = D32Knowledge(data)
            pipelines = _build_pipelines(knowledge)
            print(f"heldout {heldout}: train={len(train_rows)} test={len(test_ids)} clusters={len(knowledge.clusters)} rules={len(knowledge.mined_rules)}", flush=True)
            for row_id in test_ids:
                _process_row(row_id, pipelines, f"heldout={heldout}")
    else:
        # Non-LODO (all-train): build knowledge from all cases, predict all (in-sample)
        knowledge_path = knowledge_dir / "d32_knowledge_all_train.json"
        if knowledge_path.exists():
            knowledge = D32Knowledge.load_json(knowledge_path)
        else:
            data = build_knowledge(
                case_rows=case_rows,
                cache=cache,
                baseline=baseline,
                trace_summary_dir=args.trace_summary_dir,
                modalities=modalities,
                config=KnowledgeBuildConfig(),
            )
            knowledge_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            knowledge = D32Knowledge(data)
        pipelines = _build_pipelines(knowledge)
        print(f"all-train: n={len(case_rows)} clusters={len(knowledge.clusters)} rules={len(knowledge.mined_rules)}", flush=True)
        for row_id in range(len(case_rows)):
            _process_row(row_id, pipelines, "all-train")
    out_path = Path(args.out)
    debug_path = Path(args.debug_json)
    if args.llm_mode == "gated-rerank":
        baseline_path = default_d32_baseline_path(out_path)
        write_predictions_csv(completed, baseline_path)
        print(f"wrote {baseline_path} d32_baseline")
    else:
        write_outputs(completed, out_path, debug_path)
        print(f"wrote {args.out}")
        print(f"wrote {args.debug_json}")
    if args.llm_summary_jsonl:
        summary_limits = EvidenceSummaryLimits(
            max_metric_patterns=args.summary_max_metric_patterns,
            max_log_patterns=args.summary_max_log_patterns,
            max_trace_edges=args.summary_max_trace_edges,
            max_counter_evidence=args.summary_max_counter_evidence,
        )
        summary_count = write_llm_summary_jsonl(
            completed=completed,
            out_path=Path(args.llm_summary_jsonl),
            query_df=query_df,
            cache=cache,
            baseline=baseline,
            trace_summary_dir=args.trace_summary_dir,
            modalities=modalities,
            top_k=args.llm_top_k,
            limits=summary_limits,
        )
        print(f"wrote {args.llm_summary_jsonl} cards={summary_count}")
    if args.llm_mode in {"shadow", "gated-rerank"}:
        llm_output_jsonl = Path(args.llm_output_jsonl or "logs/llm_judge.jsonl")
        summary_limits = EvidenceSummaryLimits(
            max_metric_patterns=args.summary_max_metric_patterns,
            max_log_patterns=args.summary_max_log_patterns,
            max_trace_edges=args.summary_max_trace_edges,
            max_counter_evidence=args.summary_max_counter_evidence,
        )
        cards = collect_llm_summary_cards(
            completed=completed,
            query_df=query_df,
            cache=cache,
            baseline=baseline,
            trace_summary_dir=args.trace_summary_dir,
            modalities=modalities,
            top_k=args.llm_top_k,
            limits=summary_limits,
        )
        llm_config = LLMJudgeConfig(
            provider=args.llm_provider,
            model=args.llm_model,
            base_url=args.llm_base_url,
            api_key_env=args.llm_api_key_env,
            timeout_sec=args.llm_timeout_sec,
            temperature=args.llm_temperature,
            max_retries=args.llm_max_retries,
            concurrency=args.llm_concurrency,
        )
        if args.llm_mode == "shadow" or args.llm_rerank_judge == "candidate":
            shadow_count = write_shadow_judgments_jsonl(
                cards=cards,
                out_path=llm_output_jsonl,
                config=llm_config,
            )
            print(f"wrote {llm_output_jsonl} llm_shadow_judgments={shadow_count}")
        if args.llm_mode == "gated-rerank":
            rerank_trace_path = Path(args.llm_rerank_output_jsonl or "logs/llm_gated_rerank_trace.jsonl")
            max_alt_rank = args.llm_max_alt_rank if args.llm_max_alt_rank is not None else args.llm_top_k
            reason_name_map = D32PipelineConfig(dataset_name=args.dataset_name).REASON_NAME_MAPS.get(args.dataset_name, {})
            rerank_config = GatedRerankConfig(
                refute_threshold=args.llm_refute_threshold,
                support_threshold=args.llm_support_threshold,
                min_alt_rank=args.llm_min_alt_rank,
                max_alt_rank=max_alt_rank,
                policy=args.llm_rerank_policy,
                pairwise_margin_threshold=args.llm_pairwise_margin_threshold,
                pairwise_top1_max_support=args.llm_pairwise_top1_max_support,
                allow_pairwise_low_top1_support=args.llm_allow_pairwise_low_top1_support,
            )
            if args.llm_rerank_judge == "candidate":
                reranked_completed, rerank_summary = apply_gated_rerank(
                    completed=completed,
                    judgment_rows=load_judgment_jsonl(llm_output_jsonl),
                    trace_path=rerank_trace_path,
                    summary_path=default_rerank_summary_path(rerank_trace_path),
                    config=rerank_config,
                    reason_name_map=reason_name_map,
                )
            else:
                pairwise_output_jsonl = Path(args.llm_pairwise_output_jsonl or default_pairwise_output_path(rerank_trace_path))
                pairwise_requests = build_pairwise_requests(
                    cards,
                    min_alt_rank=args.llm_min_alt_rank,
                    max_alt_rank=max_alt_rank,
                )
                pairwise_count = write_pairwise_judgments_jsonl(
                    requests=pairwise_requests,
                    out_path=pairwise_output_jsonl,
                    config=llm_config,
                )
                print(f"wrote {pairwise_output_jsonl} llm_pairwise_judgments={pairwise_count}")
                reranked_completed, rerank_summary = apply_pairwise_gated_rerank(
                    completed=completed,
                    pairwise_rows=load_judgment_jsonl(pairwise_output_jsonl),
                    trace_path=rerank_trace_path,
                    summary_path=default_rerank_summary_path(rerank_trace_path),
                    config=rerank_config,
                    reason_name_map=reason_name_map,
                )
            write_outputs(reranked_completed, out_path, debug_path)
            print(f"wrote {args.out} llm_gated_predictions")
            print(f"wrote {args.debug_json}")
            print(f"wrote {rerank_trace_path} llm_gated_rerank_trace")
            print(f"wrote {default_rerank_summary_path(rerank_trace_path)} llm_gated_rerank_summary")
            print(json.dumps(rerank_summary, ensure_ascii=False, sort_keys=True))
    print(f"checkpoint {checkpoint_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
