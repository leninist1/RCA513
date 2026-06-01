#!/usr/bin/env python3
"""PRISM v2 Ablation Experiment Script — LOSO-compliant.

Data leakage prevention:
  By default, --loso is ENABLED. This means for each test system, the
  learned modules (classifier, likelihood, entity profiles, embeddings)
  are trained ONLY on the OTHER two systems.  The test system is NEVER
  seen during training.  This guarantees a fair comparison between
  v1 (no learned modules) and v2 (modules trained on independent data).

  Pass --no-loso to use the legacy ALL-systems training (LEAKY — for
  debugging only, not valid for publication).

Usage:
    # LOSO (safe, recommended)
    python prism_v2/run_ablation.py --system Bank --max-queries 20

    # All 3 systems LOSO
    python prism_v2/run_ablation.py --system all --max-queries 20 --loso

    # Debug mode (leaky — not for publication)
    python prism_v2/run_ablation.py --system Bank --no-loso --max-queries 5
"""

from __future__ import annotations

import argparse, json, os, subprocess, sys, time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_THIS_DIR)
sys.path.insert(0, _PARENT_DIR)

from prism_v2.prism import PRISMConfig, PRISMPipeline


def _resolve_pretrained_dir(test_system: str, loso: bool) -> str:
    if loso:
        return os.path.join(
            _THIS_DIR, "learned", "pretrained", f"loso_no_{test_system}"
        )
    return os.path.join(_THIS_DIR, "learned", "pretrained")


def _train_loso(test_system: str) -> str:
    """Train on all systems EXCEPT test_system. Returns the pretrained dir path."""
    pretrained_dir = _resolve_pretrained_dir(test_system, loso=True)
    train_script = os.path.join(_THIS_DIR, "learned", "train.py")

    print(f"[LOSO] Training on all systems EXCEPT {test_system} ...")
    result = subprocess.run(
        [sys.executable, train_script, "--exclude-system", test_system],
        capture_output=True,
        text=True,
        cwd=os.path.dirname(_THIS_DIR),
        timeout=120,
    )
    if result.returncode != 0:
        print("  ERROR: Training failed")
        print("  stderr:", result.stderr[-500:])
        return ""
    print(f"  OK — weights saved to {pretrained_dir}")
    return pretrained_dir


def build_configs(
    pretrained_dir: str, fast_mode: bool = False
) -> Dict[str, PRISMConfig]:
    """Build all ablation configurations, pointing to LOSO-safe weights.

    Args:
        pretrained_dir: Path to pretrained weight files
        fast_mode: If True, disable counterfactual profiles and final CF for speed
    """

    ep = os.path.join(pretrained_dir, "entity_profiles.json")
    lh = os.path.join(pretrained_dir, "likelihood.pkl")
    cl = os.path.join(pretrained_dir, "classifier.pkl")
    emb = os.path.join(pretrained_dir, "embeddings.pkl")

    has_pretrained = os.path.isdir(pretrained_dir)

    fast_kwargs = (
        dict(
            cf_profiles_enabled=False,
            final_counterfactual_enabled=False,
        )
        if fast_mode
        else {}
    )

    configs = {}

    # V1 baseline
    configs["v1_baseline"] = PRISMConfig(**fast_kwargs)

    if not has_pretrained:
        return configs

    # Direction E: Hierarchical Prior only (entity profiles from LOSO data)
    configs["v2_hier"] = PRISMConfig(
        use_hierarchical_prior=True,
        use_system_prototype=True,
        use_entity_profiles=True,
        entity_profile_path=ep,
        **fast_kwargs,
    )

    # Direction B: Learned only (classifier + likelihood from LOSO data)
    configs["v2_learned"] = PRISMConfig(
        use_learned_classifier=True,
        use_learned_likelihood=True,
        use_learned_embeddings=True,
        learned_embedding_path=emb,
        learned_likelihood_path=lh,
        learned_classifier_path=cl,
        **fast_kwargs,
    )

    # Hierarchical + Learned combined
    configs["v2_prior_learned"] = PRISMConfig(
        use_hierarchical_prior=True,
        use_system_prototype=True,
        use_entity_profiles=True,
        entity_profile_path=ep,
        use_learned_classifier=True,
        use_learned_likelihood=True,
        use_learned_embeddings=True,
        learned_embedding_path=emb,
        learned_likelihood_path=lh,
        learned_classifier_path=cl,
        **fast_kwargs,
    )

    # Direction A: Active Inference
    configs["v2_ai"] = PRISMConfig(
        use_active_inference=True,
        use_hierarchical_prior=True,
        use_system_prototype=True,
        use_entity_profiles=True,
        entity_profile_path=ep,
        use_learned_classifier=True,
        use_learned_likelihood=True,
        learned_likelihood_path=lh,
        learned_classifier_path=cl,
        **fast_kwargs,
    )

    # Direction C: MCTS
    configs["v2_mcts"] = PRISMConfig(
        use_mcts=True,
        mcts_n_simulations=30,
        mcts_rollout_depth=2,
        use_hierarchical_prior=True,
        use_system_prototype=True,
        use_entity_profiles=True,
        entity_profile_path=ep,
        use_learned_classifier=True,
        use_learned_likelihood=True,
        learned_likelihood_path=lh,
        learned_classifier_path=cl,
        **fast_kwargs,
    )

    # Direction D: Active Perception
    configs["v2_ap"] = PRISMConfig(
        use_active_perception=True,
        use_metric_rescan=True,
        use_log_hypothesis_search=True,
        use_hypothesis_crossval=True,
        use_hierarchical_prior=True,
        use_system_prototype=True,
        use_entity_profiles=True,
        entity_profile_path=ep,
        use_learned_classifier=True,
        use_learned_likelihood=True,
        learned_likelihood_path=lh,
        learned_classifier_path=cl,
        **fast_kwargs,
    )

    # ALL v2 features
    configs["v2_all"] = PRISMConfig(
        use_hierarchical_prior=True,
        use_system_prototype=True,
        use_entity_profiles=True,
        entity_profile_path=ep,
        use_learned_embeddings=True,
        use_learned_classifier=True,
        use_learned_likelihood=True,
        learned_embedding_path=emb,
        learned_likelihood_path=lh,
        learned_classifier_path=cl,
        use_active_inference=True,
        use_mcts=True,
        mcts_n_simulations=20,
        mcts_rollout_depth=2,
        use_active_perception=True,
        use_metric_rescan=True,
        use_log_hypothesis_search=True,
        use_hypothesis_crossval=True,
        **fast_kwargs,
    )

    return configs


def run_pipeline(
    system_name: str,
    config: PRISMConfig,
    query: Any,
    telemetry: Any,
    inject_time: float,
) -> Tuple[Dict[str, Any], float]:
    t0 = time.time()
    pipeline = PRISMPipeline(system_name, config=config)
    result = pipeline.run(telemetry=telemetry, query=query, inject_time=inject_time)
    elapsed = time.time() - t0
    return result, elapsed


def evaluate_result(
    prism_result: Dict[str, Any],
    query: Any,
) -> Dict[str, Any]:
    prediction = prism_result.get("prediction", {})
    pc = prediction.get("component", "")
    pred_component = pc[0] if isinstance(pc, list) else (str(pc) if pc else "")

    gt = query.ground_truth
    gt_component = gt.component if gt else ""

    correct = bool(
        pred_component
        and gt_component
        and pred_component.lower().strip() == gt_component.lower().strip()
    )

    return {
        "correct": correct,
        "pred_component": pred_component,
        "gt_component": gt_component,
        "stop_reason": prism_result.get("stop_reason", ""),
        "iterations": len(prism_result.get("trace", [])),
    }


def compute_metrics(results: List[Dict], n_queries: int) -> Dict[str, float]:
    if not results:
        return {"Correct": 0.0, "N": 0, "AvgIter": 0, "AvgTime": 0}
    correct = sum(1 for r in results if r.get("correct"))
    count = len(results)
    avg_iter = sum(r.get("iterations", 0) for r in results) / max(1, count)
    avg_time = sum(r.get("elapsed", 0) for r in results) / max(1, count)
    return {
        "Correct": round(correct / max(1, count) * 100, 1),
        "N": count,
        "AvgIter": round(avg_iter, 1),
        "AvgTime": round(avg_time, 1),
    }


def evaluate_system(
    system_name: str,
    configs: Dict[str, PRISMConfig],
    config_names: List[str],
    max_queries: int,
    loso: bool,
    fast_mode: bool = False,
):
    # LOSO: train on other systems first
    pretrained_dir = _resolve_pretrained_dir(system_name, loso)
    if loso:
        retrained = _train_loso(system_name)
        if not retrained:
            print(f"  SKIP {system_name}: LOSO training failed")
            return {}
        pretrained_dir = retrained

    system_configs = build_configs(pretrained_dir, fast_mode=fast_mode)
    if not system_configs and loso:
        print(f"  WARN: No pretrained weights for {system_name}")
        return {}

    # Load queries
    try:
        from prism_v2.data.loader import OpenRCALoader

        loader = OpenRCALoader(system_name)
    except Exception:
        print(f"  Cannot create loader for {system_name}")
        return {}

    queries = loader.load_queries()
    queries = queries[:max_queries]
    print(f"  Loaded {len(queries)} queries")

    all_metrics = {}
    for config_name in config_names:
        if config_name not in system_configs:
            continue
        cfg = system_configs[config_name]
        print(f"  [{config_name}] ", end="", flush=True)

        results = []
        for i, query in enumerate(queries):
            try:
                gt_records, inject_time = loader.match_query_to_records(query)
                if not gt_records:
                    continue
                query.ground_truth = gt_records[0]
                telemetry_dt = loader.resolve_telemetry_date(query)
                if not telemetry_dt:
                    continue
                telemetry = loader.load_telemetry(telemetry_dt, query.sub_system)
                if telemetry is None:
                    continue

                prism_result, elapsed = run_pipeline(
                    system_name,
                    config=cfg,
                    query=query,
                    telemetry=telemetry,
                    inject_time=inject_time,
                )
                eval_data = evaluate_result(prism_result, query)
                eval_data["elapsed"] = elapsed
                results.append(eval_data)
            except Exception as e:
                print(f"\n    ERROR {query.task_index}: {e}")
                continue

            if (i + 1) % 10 == 0:
                print(".", end="", flush=True)

        metrics = compute_metrics(results, min(max_queries, len(queries)))
        all_metrics[config_name] = metrics
        print(f" Correct={metrics['Correct']}% ({metrics['N']} cases)")
        metrics["_details"] = [
            {
                "query": r.get("gt_component", ""),
                "pred": r.get("pred_component", ""),
                "correct": r.get("correct"),
                "time": r.get("elapsed", 0),
            }
            for r in results
        ]

    return all_metrics


def main():
    parser = argparse.ArgumentParser(
        description="PRISM v2 Ablation Experiment (LOSO-safe by default)"
    )
    parser.add_argument(
        "--system",
        type=str,
        default="Bank",
        choices=["Bank", "Telecom", "Market", "all"],
        help="System to evaluate ('all' = all 3 systems LOSO)",
    )
    parser.add_argument("--max-queries", type=int, default=20)
    parser.add_argument("--output", type=str, default="prism_v2/ablation_results")
    parser.add_argument(
        "--configs",
        type=str,
        nargs="+",
        default=[
            "v1_baseline",
            "v2_hier",
            "v2_learned",
            "v2_prior_learned",
            "v2_ai",
            "v2_mcts",
            "v2_ap",
            "v2_all",
        ],
    )
    parser.add_argument("--v2-all", action="store_true", help="Only run v2_all config")
    parser.add_argument(
        "--loso",
        action="store_true",
        default=True,
        help="Leave-One-System-Out training (DEFAULT, prevents leakage)",
    )
    parser.add_argument(
        "--no-loso",
        dest="loso",
        action="store_false",
        help="Train on ALL systems (LEAKY — debugging only!)",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        default=True,
        help="Fast mode: skip counterfactual profiles (DEFAULT, ~25s/query)",
    )
    parser.add_argument(
        "--full",
        dest="fast",
        action="store_false",
        help="Full mode: enable counterfactual profiles (SLOW, ~90s/query)",
    )
    args = parser.parse_args()

    if args.v2_all:
        args.configs = ["v2_all"]

    loso = args.loso

    if args.system == "all":
        systems = ["Bank", "Telecom", "Market"]
    else:
        systems = [args.system]

    print(f"\n{'=' * 70}")
    print(f"  PRISM v2 Ablation Experiment")
    print(f"  Systems: {systems}")
    print(f"  Max queries/system: {args.max_queries}")
    print(
        f"  LOSO: {'ON (safe — no data leakage)' if loso else 'OFF (LEAKY — debug only)'}"
    )
    print(f"  Configs: {args.configs}")
    print(f"{'=' * 70}\n")

    all_system_metrics = {}

    for system_name in systems:
        print(f"\n{'—' * 50}")
        print(f"SYSTEM: {system_name}")
        print(f"{'—' * 50}")

        metrics = evaluate_system(
            system_name=system_name,
            configs={},
            config_names=args.configs,
            max_queries=args.max_queries,
            loso=loso,
            fast_mode=args.fast,
        )
        all_system_metrics[system_name] = metrics

    # Summary table
    if all_system_metrics:
        print(f"\n{'=' * 70}")
        print(f"{'Config':<25} ", end="")
        for s in systems:
            print(f"  {s:>9}", end="")
        print()
        print("-" * 70)
        for name in args.configs:
            print(f"{name:<25} ", end="")
            for s in systems:
                m = all_system_metrics.get(s, {}).get(name, {})
                print(f"  {m.get('Correct', 0):>6.1f}%", end="")
            print()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"ablation_loso_{loso}.json"
    with open(out_file, "w") as f:
        json.dump(all_system_metrics, f, indent=2, default=str)
    print(f"\nResults saved to {out_file}")


if __name__ == "__main__":
    main()
