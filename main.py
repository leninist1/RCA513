#!/usr/bin/env python3
"""Main entry point for OpenRCA Meta-Controller evaluation.

Usage:
    python -m openrca_meta_controller.main --option B --max-queries 5
    python -m openrca_meta_controller.main --option C --systems Bank
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openrca_meta_controller.config import (
    SYSTEM_PATHS, ResourceBudget, ResourceCost, MAX_ITERATIONS,
    QueryCase, EvalResult,
)
from openrca_meta_controller.data.loader import OpenRCALoader
from openrca_meta_controller.counterfactual.engine import CounterfactualEngine
from openrca_meta_controller.strategies.strategy_a import StrategyA
from openrca_meta_controller.strategies.strategy_b import StrategyB
from openrca_meta_controller.strategies.strategy_c import StrategyC
from openrca_meta_controller.strategies.strategy_d import StrategyD
from openrca_meta_controller.meta_controller.base import ControllerState
from openrca_meta_controller.meta_controller.controller_b import EmotionMetaController
from openrca_meta_controller.meta_controller.controller_c import LLMMetaController
from openrca_meta_controller.utils.llm_client import LLMClient
from openrca_meta_controller.evaluation.scorer import evaluate_prediction, construct_answer
from openrca_meta_controller.evaluation.aggregator import EvalAggregator
from openrca_meta_controller.evaluation.resource_tracker import ResourceTracker


def build_strategy_pool(llm_client=None):
    return {
        "A_broad_shallow": StrategyA(),
        "B_deep_dive": StrategyB(llm_client=llm_client),
        "C_causal_trace": StrategyC(),
        "D_log_compare": StrategyD(llm_client=llm_client),
    }


def extract_case_features(query, telemetry) -> Dict:
    task_num = int(query.task_index.split("_")[1]) if "task_" in query.task_index else 6
    return {
        "task_type": task_num,
        "system": query.system,
        "sub_system": query.sub_system,
        "time_range": f"{query.time_window[0]} to {query.time_window[1]}",
        "has_traces": telemetry.traces is not None and not telemetry.traces.empty,
        "has_logs": telemetry.logs is not None and not telemetry.logs.empty,
        "num_entities": len(telemetry.entities),
        "num_services": len(telemetry.entities),
    }


def _process_query_worker(args):
    """Picklable worker for ProcessPoolExecutor. Loads telemetry and runs full pipeline."""
    (system_name, sub_system, date_str, inject_time, task_index,
     instruction, time_window_start, time_window_end,
     scoring_points_json, ground_truth_json, option, api_key, api_provider) = args

    import json as _json
    from openrca_meta_controller.config import QueryCase, GroundTruth, ScoringPoint
    from openrca_meta_controller.data.loader import OpenRCALoader
    from openrca_meta_controller.counterfactual.engine import CounterfactualEngine
    from openrca_meta_controller.strategies.strategy_a import StrategyA
    from openrca_meta_controller.strategies.strategy_b import StrategyB
    from openrca_meta_controller.strategies.strategy_c import StrategyC
    from openrca_meta_controller.strategies.strategy_d import StrategyD
    from openrca_meta_controller.meta_controller.base import ControllerState
    from openrca_meta_controller.meta_controller.controller_b import EmotionMetaController
    from openrca_meta_controller.meta_controller.controller_c import LLMMetaController
    from openrca_meta_controller.utils.llm_client import LLMClient
    from openrca_meta_controller.evaluation.scorer import evaluate_prediction, construct_answer
    from openrca_meta_controller.config import ResourceBudget, MAX_ITERATIONS

    query_id = f"{system_name}_{task_index}"
    result = {
        "query_id": query_id, "system": system_name, "task_index": task_index,
        "correct": False, "partial": False, "error": None,
        "iterations": 0, "strategies_used": [], "api_calls": 0, "tokens": 0,
    }

    try:
        # Reconstruct query
        scoring_points = _json.loads(scoring_points_json)
        gt = _json.loads(ground_truth_json) if ground_truth_json else None
        query = QueryCase(
            task_index=task_index, system=system_name, sub_system=sub_system,
            instruction=instruction,
            time_window=(time_window_start, time_window_end),
            scoring_points=[ScoringPoint(**sp) for sp in scoring_points],
            inject_time=inject_time,
            telemetry_date=date_str,
            ground_truth=GroundTruth(**gt) if gt else None,
        )

        # Load telemetry
        loader = OpenRCALoader(system_name)
        telemetry = loader.load_telemetry(date_str, sub_system)

        # Build strategy pool (strategies use heuristics only — no LLM to avoid API contention)
        strategy_pool = {
            "A_broad_shallow": StrategyA(),
            "B_deep_dive": StrategyB(llm_client=None),
            "C_causal_trace": StrategyC(),
            "D_log_compare": StrategyD(llm_client=None),
        }

        # LLM client only for controller
        llm_client = LLMClient(provider=api_provider, api_key=api_key) if (option == "C" and api_key) else None

        # Run initial analysis
        engine = CounterfactualEngine(system_name)
        candidates, graph, baseline_df, fault_df = engine.run_initial_analysis(
            telemetry, inject_time, telemetry.entities
        )
        if not candidates:
            result["error"] = "no_candidates"
            return result

        # Initialize controller state
        task_num = int(task_index.split("_")[1]) if "task_" in task_index else 6
        state = ControllerState(
            evidence_pool={c.entity: [] for c in candidates},
            candidate_scores={c.entity: c.recovery_score or c.anomaly_score for c in candidates},
            budget=ResourceBudget(),
            case_features={
                "task_type": task_num, "system": system_name,
                "has_traces": telemetry.traces is not None and not telemetry.traces.empty,
                "has_logs": telemetry.logs is not None and not telemetry.logs.empty,
                "num_entities": len(telemetry.entities),
            },
        )

        # Initialize controller
        if option == "B":
            controller = EmotionMetaController()
        else:
            controller = LLMMetaController(llm_client=llm_client) if llm_client else None
            if controller is None:
                result["error"] = "no_llm_client"
                return result

        # Controller loop (option C uses fewer iterations to limit API calls)
        max_iter = 3 if option == "C" else MAX_ITERATIONS
        for iteration in range(max_iter):
            action = controller.select_action(state)
            if action.stop:
                state.stop_reason = action.stop_reason
                break

            strategy = strategy_pool.get(action.strategy)
            if strategy is None:
                break

            strategy_result = strategy.execute(
                candidates=state.top_candidates(), telemetry=telemetry,
                baseline_df=baseline_df, fault_df=fault_df,
                budget=state.budget, engine=engine,
            )

            state.update(strategy_result)
            if option == "B":
                controller.update_emotion(strategy_result, state.budget)

        # Construct answer and evaluate
        prediction = construct_answer(state, query)
        eval_result = evaluate_prediction(prediction, query)

        result["correct"] = eval_result.correct
        result["partial"] = eval_result.partial
        result["iterations"] = state.iteration
        result["strategies_used"] = state.strategy_history
        result["api_calls"] = state.budget.api_calls_used
        result["tokens"] = state.budget.tokens_used
        result["prediction"] = prediction
        if query.ground_truth:
            result["ground_truth"] = {
                "component": query.ground_truth.component,
                "reason": query.ground_truth.reason,
                "datetime": query.ground_truth.datetime_str,
            }
        result["top_candidates"] = [
            {"entity": e, "score": round(s, 4)}
            for e, s in sorted(state.candidate_scores.items(), key=lambda x: x[1], reverse=True)[:5]
        ]
        result["stop_reason"] = state.stop_reason

    except Exception as e:
        import traceback
        result["error"] = str(e)
        result["traceback"] = traceback.format_exc()

    return result


def run_single_query(query, loader, strategy_pool, option, resource_tracker, llm_client=None, preloaded_telemetry=None):
    """Run one query through the full pipeline. Returns (result_dict, eval_result)."""
    query_id = f"{query.system}_{query.task_index}"
    result = {
        "query_id": query_id, "system": query.system, "task_index": query.task_index,
        "correct": False, "partial": False, "error": None,
        "iterations": 0, "strategies_used": [], "api_calls": 0, "tokens": 0,
    }

    try:
        inject_time = query.inject_time
        if inject_time is None:
            result["error"] = "no_inject_time"
            return result, None

        # Use preloaded telemetry if available
        if preloaded_telemetry is not None:
            telemetry = preloaded_telemetry
        else:
            date_str = loader.resolve_telemetry_date(query)
            if date_str is None:
                result["error"] = "no_telemetry_date"
                return result, None
            query.telemetry_date = date_str
            telemetry = loader.load_telemetry(date_str, query.sub_system)

        # 3. Run initial analysis
        engine = CounterfactualEngine(query.system)
        candidates, graph, baseline_df, fault_df = engine.run_initial_analysis(
            telemetry, inject_time, telemetry.entities
        )
        if not candidates:
            result["error"] = "no_candidates"
            return result, None

        # 4. Initialize controller state
        state = ControllerState(
            evidence_pool={c.entity: [] for c in candidates},
            candidate_scores={c.entity: c.recovery_score or c.anomaly_score for c in candidates},
            budget=ResourceBudget(),
            case_features=extract_case_features(query, telemetry),
        )

        # 5. Initialize controller
        if option.upper() == "B":
            controller = EmotionMetaController()
        else:
            controller = LLMMetaController(llm_client=llm_client) if llm_client else None
            if controller is None:
                result["error"] = "no_llm_client"
                return result, None

        # 6. Controller loop
        for iteration in range(MAX_ITERATIONS):
            action = controller.select_action(state)
            if action.stop:
                state.stop_reason = action.stop_reason
                break

            strategy = strategy_pool.get(action.strategy)
            if strategy is None:
                break

            strategy_result = strategy.execute(
                candidates=state.top_candidates(),
                telemetry=telemetry,
                baseline_df=baseline_df,
                fault_df=fault_df,
                budget=state.budget,
                engine=engine,
            )

            state.update(strategy_result)
            resource_tracker.record(
                query_id,
                api_calls=strategy_result.cost.api_calls,
                tokens=strategy_result.cost.tokens,
                iterations=1,
                strategy_name=action.strategy,
            )

            if option.upper() == "B":
                controller.update_emotion(strategy_result, state.budget)

        # 7. Construct answer and evaluate
        prediction = construct_answer(state, query)
        eval_result = evaluate_prediction(prediction, query)

        result["correct"] = eval_result.correct
        result["partial"] = eval_result.partial
        result["iterations"] = state.iteration
        result["strategies_used"] = state.strategy_history
        result["api_calls"] = state.budget.api_calls_used
        result["tokens"] = state.budget.tokens_used
        result["prediction"] = prediction
        result["ground_truth"] = {
            "component": query.ground_truth.component if query.ground_truth else "",
            "reason": query.ground_truth.reason if query.ground_truth else "",
            "datetime": query.ground_truth.datetime_str if query.ground_truth else "",
        }
        result["top_candidates"] = [
            {"entity": e, "score": round(s, 4)}
            for e, s in sorted(state.candidate_scores.items(), key=lambda x: x[1], reverse=True)[:5]
        ]
        result["stop_reason"] = state.stop_reason

        return result, eval_result

    except Exception as e:
        import traceback
        result["error"] = str(e)
        result["traceback"] = traceback.format_exc()
        return result, None


def main():
    parser = argparse.ArgumentParser(description="OpenRCA Meta-Controller Evaluation")
    parser.add_argument("--option", choices=["B", "C"], required=True)
    parser.add_argument("--systems", nargs="+", default=["Bank", "Telecom", "Market"])
    parser.add_argument("--output", default="results/")
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--provider", default="deepseek")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel workers (threads) per date group")
    args = parser.parse_args()

    results_dir = Path(args.output)
    results_dir.mkdir(parents=True, exist_ok=True)

    llm_client = None
    if args.option.upper() == "C":
        llm_client = LLMClient(provider=args.provider, api_key=args.api_key)
        if not llm_client.api_key:
            print("WARNING: No API key found. Option C will fail without LLM access.")
            print("  Set DEEPSEEK_API_KEY or ANTHROPIC_API_KEY, or use --api-key.")

    strategy_pool = build_strategy_pool(llm_client=llm_client)
    resource_tracker = ResourceTracker()
    eval_aggregator = EvalAggregator()
    all_results = []

    for system_name in args.systems:
        if system_name not in SYSTEM_PATHS:
            print(f"Unknown system: {system_name}, skipping")
            continue

        print(f"\n{'='*60}")
        print(f"SYSTEM: {system_name}")
        print(f"{'='*60}")

        loader = OpenRCALoader(system_name)
        sub_systems = SYSTEM_PATHS[system_name]["sub_systems"]

        for sub in sub_systems:
            queries = loader.load_queries(sub)
            if args.max_queries:
                queries = queries[:args.max_queries]

            # Match records, group queries by date
            date_groups = {}
            skipped = 0
            for q in queries:
                gts, t = loader.match_query_to_records(q)
                if t is None:
                    all_results.append({
                        "query_id": f"{q.system}_{q.task_index}",
                        "system": q.system, "task_index": q.task_index,
                        "correct": False, "partial": False, "error": "no_matching_record",
                    })
                    skipped += 1
                    continue
                q.ground_truth = gts[0] if gts else None
                q.inject_time = t
                d = loader.resolve_telemetry_date(q)
                q.telemetry_date = d
                if d:
                    date_groups.setdefault(d, []).append(q)
                else:
                    skipped += 1

            sub_label = sub or "default"
            print(f"  Sub-system: {sub_label} ({sum(len(v) for v in date_groups.values())} valid in {len(date_groups)} dates, {skipped} skipped)", flush=True)

            # Process date by date to limit memory
            total_processed = 0
            for date_str, date_queries in sorted(date_groups.items()):
                print(f"    Date {date_str}: loading telemetry...", flush=True, end=" ")
                t0 = time.time()
                telemetry = loader.load_telemetry(date_str, sub)
                print(f"({time.time()-t0:.1f}s, {len(date_queries)} queries)", flush=True)

                # Build worker args (serializable) for each query
                worker_args = []
                for q in date_queries:
                    sp_json = json.dumps([
                        {"field_type": sp.field_type, "rank": sp.rank, "expected_value": sp.expected_value}
                        for sp in q.scoring_points
                    ])
                    gt_json = ""
                    if q.ground_truth:
                        gt_json = json.dumps({
                            "component": q.ground_truth.component or "",
                            "reason": q.ground_truth.reason or "",
                            "timestamp": q.ground_truth.timestamp or 0,
                            "datetime_str": q.ground_truth.datetime_str or "",
                        })
                    worker_args.append((
                        system_name, sub, date_str, q.inject_time, q.task_index,
                        q.instruction, q.time_window[0], q.time_window[1],
                        sp_json, gt_json, args.option,
                        args.api_key or os.environ.get("DEEPSEEK_API_KEY", ""),
                        args.provider,
                    ))

                # Process queries in parallel within this date
                n_workers = min(args.workers, len(date_queries))
                with ProcessPoolExecutor(max_workers=n_workers) as executor:
                    futures = [executor.submit(_process_query_worker, wa) for wa in worker_args]
                    for future in as_completed(futures):
                        result = future.result()
                        total_processed += 1

                        all_results.append(result)
                        correct = result.get("correct", False)
                        partial = result.get("partial", False)
                        if not result.get("error"):
                            eval_aggregator.add(EvalResult(
                                correct=correct, partial=partial,
                                task_type=result.get("task_index", ""),
                                system=result.get("system", ""),
                                prediction=result.get("prediction", {}),
                            ))

                        status = "OK" if correct else ("PART" if partial else "ERR" if result.get("error") else "NO")
                        err_msg = f" err={result.get('error','')}" if result.get("error") else ""
                        print(f"    [{total_processed}] {status}{err_msg}", flush=True)

    # Intermediate save after each system
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    interim_file = results_dir / f"option_{args.option}_{timestamp}_partial.json"
    with open(interim_file, "w") as f:
        interim_summary = eval_aggregator.summary()
        json.dump({
            "config": {"option": args.option, "systems": args.systems, "max_queries": args.max_queries},
            "summary": interim_summary,
            "resources": resource_tracker.summary(),
            "per_query": all_results,
        }, f, indent=2, default=str)
    print(f"  [Saved partial results to {interim_file}]", flush=True)

    # Final summary
    print(f"\n{'='*60}")
    print(f"EVALUATION SUMMARY — Option {args.option}")
    print(f"{'='*60}")

    summary = eval_aggregator.summary()
    print(f"Total: {summary['total']}")
    print(f"Correct: {summary['correct']} ({summary['correct_pct']}%)")
    print(f"Partial: {summary['partial']} ({summary['partial_pct']}%)")
    print(f"Correct+Partial: {summary['correct_or_partial_pct']}%")

    if "by_system" in summary:
        print(f"\nPer-system:")
        for sys, stats in summary["by_system"].items():
            print(f"  {sys}: C={stats['correct']}/{stats['total']} ({stats['correct_pct']}%)  P={stats['partial']} ({stats['partial_pct']}%)")

    resource_summary = resource_tracker.summary()
    print(f"\nRESOURCES:")
    print(f"  API calls: {resource_summary['total_api_calls']}")
    print(f"  Tokens: {resource_summary['total_tokens']}")
    print(f"  Avg iter: {resource_summary['avg_iterations']}")
    if resource_summary.get("strategy_distribution"):
        print(f"  Strategies: {resource_summary['strategy_distribution']}")

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = results_dir / f"option_{args.option}_{timestamp}.json"
    with open(output_file, "w") as f:
        json.dump({
            "config": {"option": args.option, "systems": args.systems, "max_queries": args.max_queries},
            "summary": summary,
            "resources": resource_summary,
            "per_query": all_results,
        }, f, indent=2, default=str)

    print(f"\nSaved to: {output_file}")


if __name__ == "__main__":
    main()
