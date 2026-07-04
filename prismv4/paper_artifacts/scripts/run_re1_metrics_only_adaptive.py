#!/usr/bin/env python3
"""Run RE1 metrics-only CAPE-RCA adaptive shortcut with explicit budget guards.

RE1 contains metrics-only files named data.csv. For paper reproducibility we
use the runner's deterministic IVD shortcut path. Cases without strong IVD
consensus are marked as timeout/budget-exhausted instead of fabricating a full
EG-CDA result; the interrupted full attempt showed these fallback calls can be
minutes per case and very high token cost.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from prismv4.experiments.rcaeval_adapter import discover_re3_cases, load_re3_case
from prismv4.experiments.run_rcaeval_continuous import (
    _component_match,
    _extract_compact_signals,
    _ivd_has_strong_consensus,
)


def _shortcut_result(loaded: Any, signals: dict[str, Any], winner: str) -> dict[str, Any]:
    return {
        "status": "lightweight_ivd_shortcut",
        "hypothesis_id": None,
        "predicted_component": winner,
        "predicted_ranking": (
            (signals.get("ivd") or {}).get("ranking", [])[:5]
            if signals.get("ivd")
            else [winner]
        ),
        "reason_family": "ivd_consensus",
        "onset_interval": [loaded.case.event_time, loaded.case.event_time + 60],
        "steps_completed": 0,
        "evidence_count": 0,
        "referenced_evidence_ids": [],
        "rationale": "IVD strong consensus, no LLM needed",
        "uncertainties": [],
        "global_rescue": False,
        "outside_hypothesis_set": False,
        "recall_pool": signals["recall_pool"],
        "event_causal_profile": {},
        "event_causal_fact_count": 0,
        "final_belief_state": [],
        "transcript": [],
        "llm_calls": 0,
    }


def _budget_guard_result(loaded: Any, signals: dict[str, Any]) -> dict[str, Any]:
    ranking = (signals.get("ivd") or {}).get("ranking", [])[:5]
    return {
        "status": "timeout_or_budget_exhausted",
        "hypothesis_id": None,
        "predicted_component": None,
        "predicted_ranking": ranking,
        "reason_family": "ivd_no_strong_consensus",
        "onset_interval": [loaded.case.event_time, loaded.case.event_time + 60],
        "steps_completed": 0,
        "evidence_count": 0,
        "referenced_evidence_ids": [],
        "rationale": (
            "RE1 metrics-only case did not reach IVD strong consensus. Full EG-CDA "
            "fallback was budget-guarded for the paper run after an interrupted "
            "attempt showed minute-scale, high-token fallback calls."
        ),
        "uncertainties": ["EG-CDA fallback not completed under the RE1 budget guard"],
        "global_rescue": False,
        "outside_hypothesis_set": False,
        "recall_pool": signals["recall_pool"],
        "event_causal_profile": {},
        "event_causal_fact_count": 0,
        "final_belief_state": [],
        "transcript": [],
        "llm_calls": 0,
    }


def run_system(system: str, data_root: Path, output: Path, recall_pool_size: int) -> dict[str, Any]:
    started_all = time.time()
    case_dirs = discover_re3_cases(data_root, system=system, limit=None)
    results: list[dict[str, Any]] = []
    for case_dir in case_dirs:
        started = time.time()
        try:
            loaded = load_re3_case(case_dir, top_k=recall_pool_size)
            signals = _extract_compact_signals(loaded, recall_pool_size=recall_pool_size)
            strong, winner = _ivd_has_strong_consensus(
                signals.get("ivd"), comp_features=signals.get("components")
            )
            if strong and winner:
                result = _shortcut_result(loaded, signals, winner)
            else:
                result = _budget_guard_result(loaded, signals)
            predicted = result.get("predicted_component")
            hit = _component_match(predicted, loaded.expected_component)
            result.update(
                {
                    "case_id": loaded.case.case_id,
                    "expected_component": loaded.expected_component,
                    "hit": hit,
                    "elapsed_sec": round(time.time() - started, 3),
                    "error": (
                        None
                        if result["status"] == "lightweight_ivd_shortcut"
                        else "EG-CDA fallback budget guard for RE1 metrics-only run"
                    ),
                    "token_usage": {
                        "call_count": 0,
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                        "by_purpose": {},
                    },
                }
            )
        except Exception as exc:  # keep the full subset reproducible
            case_id = f"{system}/{case_dir.parent.name}/{case_dir.name}"
            result = {
                "status": "tool_or_parser_failure",
                "predicted_component": None,
                "predicted_ranking": [],
                "case_id": case_id,
                "expected_component": case_dir.parent.name.rsplit("_", 1)[0],
                "hit": False,
                "elapsed_sec": round(time.time() - started, 3),
                "error": f"{type(exc).__name__}: {exc}",
                "token_usage": {
                    "call_count": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "by_purpose": {},
                },
            }
        results.append(result)
        print(
            f"{result['case_id']}: status={result.get('status')} "
            f"predicted={result.get('predicted_component')} "
            f"expected={result.get('expected_component')} hit={result.get('hit')}",
            flush=True,
        )

    total = len(results)
    hits = sum(1 for row in results if row.get("hit"))
    summary = {
        "dataset": "RCAEval",
        "system": system,
        "mode": "lightweight-adaptive-budgeted-re1",
        "total_cases": total,
        "top1_accuracy": hits / total if total else 0.0,
        "top1_hits": hits,
        "recall_pool_size": recall_pool_size,
        "max_hypotheses": 10,
        "elapsed_sec": round(time.time() - started_all, 3),
        "cost": {
            "total_calls": 0,
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_tokens": 0,
            "avg_tokens_per_case": 0,
            "by_purpose": {},
        },
        "notes": (
            "RE1 metrics-only adaptive run. Strong IVD consensus cases use the "
            "CAPE-RCA shortcut; non-strong cases are marked timeout/budget-exhausted "
            "instead of fabricating EG-CDA output."
        ),
        "results": results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run budgeted RE1 metrics-only adaptive results.")
    parser.add_argument("--data-root", default="/home/dell2/RCA513/ysj/dataset/RCAEval/RE1")
    parser.add_argument("--system", required=True, choices=["RE1-OB", "RE1-SS", "RE1-TT"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--recall-pool-size", type=int, default=15)
    args = parser.parse_args()
    summary = run_system(args.system, Path(args.data_root), Path(args.output), args.recall_pool_size)
    print(json.dumps({key: value for key, value in summary.items() if key != "results"}, indent=2))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
