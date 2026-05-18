#!/usr/bin/env python3
"""Ablation: no meta-controller, just counterfactual engine top-1. Multi-core."""

import json, os, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))


def _ablation_worker(args):
    """Picklable worker: load telemetry, run initial analysis, return top-1."""
    (system_name, sub_system, date_str, inject_time, task_index,
     instruction, time_window_start, time_window_end,
     scoring_points_json, ground_truth_json) = args

    import json as _json
    from openrca_meta_controller.config import QueryCase, GroundTruth, ScoringPoint
    from openrca_meta_controller.data.loader import OpenRCALoader
    from openrca_meta_controller.counterfactual.engine import CounterfactualEngine
    from openrca_meta_controller.meta_controller.base import ControllerState
    from openrca_meta_controller.config import ResourceBudget
    from openrca_meta_controller.evaluation.scorer import evaluate_prediction, construct_answer

    result = {
        "query_id": f"{system_name}_{task_index}", "system": system_name,
        "task_index": task_index, "correct": False, "partial": False, "error": None,
        "iterations": 0, "strategies_used": [], "api_calls": 0, "tokens": 0,
    }

    try:
        scoring_points = _json.loads(scoring_points_json)
        gt = _json.loads(ground_truth_json) if ground_truth_json else None
        query = QueryCase(
            task_index=task_index, system=system_name, sub_system=sub_system,
            instruction=instruction,
            time_window=(time_window_start, time_window_end),
            scoring_points=[ScoringPoint(**sp) for sp in scoring_points],
            inject_time=inject_time, telemetry_date=date_str,
            ground_truth=GroundTruth(**gt) if gt else None,
        )

        loader = OpenRCALoader(system_name)
        telemetry = loader.load_telemetry(date_str, sub_system)

        # Initial analysis only — NO controller loop
        engine = CounterfactualEngine(system_name)
        candidates, graph, baseline_df, fault_df = engine.run_initial_analysis(
            telemetry, inject_time, telemetry.entities
        )
        if not candidates:
            result["error"] = "no_candidates"
            return result

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

        prediction = construct_answer(state, query)
        eval_result = evaluate_prediction(prediction, query)

        result["correct"] = eval_result.correct
        result["partial"] = eval_result.partial
        result["prediction"] = prediction
        if query.ground_truth:
            result["ground_truth"] = {
                "component": query.ground_truth.component,
                "reason": query.ground_truth.reason,
                "datetime": query.ground_truth.datetime_str,
            }
    except Exception as e:
        import traceback
        result["error"] = str(e)
        result["traceback"] = traceback.format_exc()

    return result


def main():
    from openrca_meta_controller.config import SYSTEM_PATHS, EvalResult
    from openrca_meta_controller.data.loader import OpenRCALoader
    from openrca_meta_controller.evaluation.aggregator import EvalAggregator

    output_dir = Path("rca513/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_aggregator = EvalAggregator()
    all_results = []
    n_workers = 4

    for system_name in ["Bank", "Telecom", "Market"]:
        print(f"\n{'='*60}")
        print(f"SYSTEM: {system_name} (ABLATION: no controller, {n_workers} workers)")
        print(f"{'='*60}", flush=True)

        loader = OpenRCALoader(system_name)
        sub_systems = SYSTEM_PATHS[system_name]["sub_systems"]
        tp = 0

        for sub in sub_systems:
            queries = loader.load_queries(sub)
            date_groups = {}
            for q in queries:
                gts, t = loader.match_query_to_records(q)
                if t is None: continue
                q.ground_truth = gts[0] if gts else None
                q.inject_time = t
                d = loader.resolve_telemetry_date(q)
                if d: date_groups.setdefault(d, []).append(q)

            n_valid = sum(len(v) for v in date_groups.values())
            print(f"  {sub or 'default'}: {n_valid} valid in {len(date_groups)} dates", flush=True)

            for date_str, date_queries in sorted(date_groups.items()):
                print(f"  Date {date_str}: {len(date_queries)} queries...", flush=True, end=" ")
                t0 = time.time()

                # Pre-load telemetry once per date
                telemetry = loader.load_telemetry(date_str, sub)

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
                        q.instruction, q.time_window[0] if q.time_window else "", q.time_window[1] if q.time_window else "",
                        sp_json, gt_json,
                    ))

                with ProcessPoolExecutor(max_workers=n_workers) as executor:
                    futures = [executor.submit(_ablation_worker, wa) for wa in worker_args]
                    for future in as_completed(futures):
                        result = future.result()
                        tp += 1
                        all_results.append(result)
                        if not result.get("error"):
                            eval_aggregator.add(EvalResult(
                                correct=result["correct"], partial=result["partial"],
                                task_type=result["task_index"], system=result["system"],
                            ))
                        status = "OK" if result.get("correct") else ("PART" if result.get("partial") else "ERR" if result.get("error") else "NO")
                        if tp % 20 == 0:
                            print(f"[{tp}]", flush=True, end=" ")

                print(f"({time.time()-t0:.0f}s)", flush=True)

    # Summary
    print(f"\n{'='*60}")
    print(f"ABLATION: no meta-controller")
    print(f"{'='*60}")
    s = eval_aggregator.summary()
    print(f"Total: {s['total']} | C: {s['correct']} ({s['correct_pct']}%) | P: {s['partial']} ({s['partial_pct']}%) | C+P: {s['correct_or_partial_pct']}%")
    for sys, st in (s.get("by_system") or {}).items():
        print(f"  {sys}: C={st['correct']}/{st['total']} ({st['correct_pct']}%) P={st['partial']} ({st['partial_pct']}%)")
    for task, st in sorted((s.get("by_task") or {}).items()):
        print(f"  {task}: C={st['correct']}/{st['total']} ({st['correct_pct']}%)")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    f = output_dir / f"ablation_no_controller_{ts}.json"
    with open(f, "w") as fh:
        json.dump({"config": "ablation_no_controller", "summary": s, "per_query": all_results}, fh, indent=2, default=str)
    print(f"\nSaved: {f}")


if __name__ == "__main__":
    main()
