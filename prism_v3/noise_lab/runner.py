"""Standalone runner for Noise Lab experiments."""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import QueryCase, SYSTEM_PATHS
from ..data.loader import OpenRCALoader
from ..mace.graph import build_object_graph
from ..utils.llm_client import LLMClient
from .beamformer import StructuralBeamformer
from .causal_reranker import SemanticCausalChainReranker
from .delay_localizer import DelayPatternLocalizer
from .noise_field import NoiseFieldScorer
from .reverb_mask import ReverbSuppressionMask
from .structural_encoder import StructuralObjectEncoder
from .subspace import SourceNoiseSubspaceDecomposer


def run_noise_lab(
    system_name: str,
    max_queries: Optional[int] = None,
    variant: str = "p21",
    use_llm_reranker: bool = False,
    llm_provider: str = "deepseek",
    llm_model: str = "deepseek-chat",
) -> Dict[str, Any]:
    loader = OpenRCALoader(system_name)
    use_global_prior = "globalprior" in variant
    noise_scorer = NoiseFieldScorer()
    structural_encoder = StructuralObjectEncoder()
    delay_localizer = DelayPatternLocalizer()
    beamformer = StructuralBeamformer()
    subspace_decomposer = SourceNoiseSubspaceDecomposer()
    reverb_mask = ReverbSuppressionMask()

    reranker: Optional[SemanticCausalChainReranker] = None
    if use_llm_reranker:
        api_key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("ANTHROPIC_API_KEY", "")
        if api_key:
            reranker = SemanticCausalChainReranker(
                llm_client=LLMClient(provider=llm_provider, api_key=api_key),
                model=llm_model,
            )

    queries = loader.load_all_queries()
    if max_queries is not None:
        queries = queries[:max_queries]

    per_query: List[Dict[str, Any]] = []
    summary = {
        "total": 0,
        "top1": 0,
        "top3": 0,
        "top5": 0,
        "avg_gt_rank": 0.0,
        "skipped": 0,
    }

    gt_rank_sum = 0.0
    for query in queries:
        gts, inject_time = loader.match_query_to_records(query)
        if not gts:
            per_query.append({
                "query_id": f"{system_name}_{query.task_index}",
                "task_index": query.task_index,
                "status": "skipped",
                "reason": "no_matching_record",
            })
            summary["skipped"] += 1
            continue
        query.ground_truth = gts[0]
        query.inject_time = inject_time
        query.telemetry_date = loader.resolve_telemetry_date(query)
        if not query.telemetry_date:
            per_query.append({
                "query_id": f"{system_name}_{query.task_index}",
                "task_index": query.task_index,
                "status": "skipped",
                "reason": "missing_telemetry_date",
            })
            summary["skipped"] += 1
            continue

        telemetry = loader.load_telemetry(query.telemetry_date, query.sub_system)
        if _should_mask_telecom_metrics(system_name, variant):
            telemetry = _mask_metric_timestamps(telemetry)
        object_graph, graph_debug = build_object_graph(telemetry, query, inject_time)
        noise_scores = noise_scorer.score(object_graph)
        structural_scores = structural_encoder.encode(object_graph)
        delay_scores = delay_localizer.score(object_graph)
        beam_scores = beamformer.score(object_graph)
        subspace_scores = subspace_decomposer.score(object_graph, beam_scores=beam_scores)
        mask_scores = reverb_mask.score(
            object_graph,
            noise_scores=noise_scores,
            structural_scores=structural_scores,
            delay_scores=delay_scores,
            beam_scores=beam_scores,
            subspace_scores=subspace_scores,
        )
        ranking = _rank_objects(
            object_graph,
            noise_scores,
            structural_scores,
            delay_scores,
            beam_scores,
            subspace_scores,
            mask_scores,
        )
        reranker_debug: Dict[str, Any] = {}
        if reranker is not None:
            ranking, reranker_debug = reranker.rerank(
                instruction=query.instruction,
                graph=object_graph,
                ranking=ranking,
                entity_types=telemetry.entity_types,
            )
        gt_component = str(query.ground_truth.component or "")
        gt_rank = _ground_truth_rank(ranking, gt_component)

        summary["total"] += 1
        if gt_rank == 1:
            summary["top1"] += 1
        if gt_rank is not None and gt_rank <= 3:
            summary["top3"] += 1
        if gt_rank is not None and gt_rank <= 5:
            summary["top5"] += 1
        gt_rank_sum += float(gt_rank or (len(ranking) + 1))

        query_payload = {
            "query_id": f"{system_name}_{query.task_index}",
            "task_index": query.task_index,
            "status": "ok",
            "ground_truth": {
                "component": query.ground_truth.component,
                "reason": query.ground_truth.reason,
                "datetime": query.ground_truth.datetime_str,
            },
            "gt_rank": gt_rank,
            "top_candidates": ranking[:5],
            "reranker_debug": reranker_debug,
            "noise_heatmap": _top_heatmap(noise_scores),
            "delay_heatmap": _top_delaymap(delay_scores),
            "beam_heatmap": _top_beammap(beam_scores),
            "subspace_heatmap": _top_subspacemap(subspace_scores),
            "reverb_heatmap": _top_reverbmap(mask_scores),
            "graph_debug": {
                "query": graph_debug.get("query"),
                "objects": graph_debug.get("objects", {}),
            },
        }
        if use_global_prior:
            query_payload["_ranking_full"] = ranking
            query_payload["_gt_component"] = gt_component
        per_query.append(query_payload)

    if use_global_prior:
        _apply_global_prior(per_query)
        summary = _summarize_per_query(per_query)
    elif summary["total"] > 0:
        summary["top1_pct"] = round(100.0 * summary["top1"] / summary["total"], 2)
        summary["top3_pct"] = round(100.0 * summary["top3"] / summary["total"], 2)
        summary["top5_pct"] = round(100.0 * summary["top5"] / summary["total"], 2)
        summary["avg_gt_rank"] = round(gt_rank_sum / summary["total"], 4)
    else:
        summary["top1_pct"] = 0.0
        summary["top3_pct"] = 0.0
        summary["top5_pct"] = 0.0

    payload = {
        "config": {
            "system": system_name,
            "max_queries": max_queries,
            "variant": variant,
            "candidate_systems": list(SYSTEM_PATHS),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
        "summary": summary,
        "per_query": per_query,
    }
    _write_result(system_name, payload, variant=variant)
    return payload


def _apply_global_prior(per_query: List[Dict[str, Any]], alpha: float = 0.85) -> None:
    """Penalize objects that repeatedly appear in top5 across unrelated queries.

    Uses leave-one-out frequency to avoid directly using the current query's
    own top5 as evidence. This targets stable hub/carrier artifacts such as
    Telecom os_021/os_022 and Market adservice.
    """
    ok_rows = [
        row for row in per_query
        if row.get("status") == "ok" and row.get("_ranking_full")
    ]
    n_rows = len(ok_rows)
    if n_rows <= 1:
        _strip_private_ranking_fields(per_query)
        return

    top5_counts: Dict[str, int] = {}
    for row in ok_rows:
        for candidate in row["_ranking_full"][:5]:
            object_id = str(candidate.get("object_id", ""))
            top5_counts[object_id] = top5_counts.get(object_id, 0) + 1

    for row in ok_rows:
        ranking = row.get("_ranking_full", [])
        own_top5 = {str(candidate.get("object_id", "")) for candidate in ranking[:5]}
        adjusted: List[Dict[str, Any]] = []
        for candidate in ranking:
            object_id = str(candidate.get("object_id", ""))
            loo_count = top5_counts.get(object_id, 0) - (1 if object_id in own_top5 else 0)
            frequency = loo_count / max(1, n_rows - 1)
            old_score = float(candidate.get("score", 0.0))
            new_score = old_score
            if old_score > 0.0 and frequency > 0.0:
                new_score = old_score * max(0.0, 1.0 - alpha * frequency)
            new_candidate = dict(candidate)
            new_candidate["pre_global_score"] = round(old_score, 6)
            new_candidate["global_prior_frequency"] = round(frequency, 4)
            new_candidate["score"] = round(new_score, 6)
            adjusted.append(new_candidate)

        adjusted.sort(key=lambda item: item["score"], reverse=True)
        gt_component = str(row.get("_gt_component") or row.get("ground_truth", {}).get("component") or "")
        row["gt_rank"] = _ground_truth_rank(adjusted, gt_component)
        row["top_candidates"] = adjusted[:5]
        row["_rank_fallback"] = len(adjusted) + 1
        row["global_prior_debug"] = {
            "alpha": alpha,
            "top_penalized": [
                {
                    "object_id": object_id,
                    "leave_one_out_top5_frequency": round(
                        (top5_counts.get(object_id, 0) - (1 if object_id in own_top5 else 0))
                        / max(1, n_rows - 1),
                        4,
                    ),
                }
                for object_id in sorted(
                    own_top5,
                    key=lambda oid: top5_counts.get(oid, 0),
                    reverse=True,
                )[:5]
            ],
        }


def _summarize_per_query(per_query: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary = {
        "total": 0,
        "top1": 0,
        "top3": 0,
        "top5": 0,
        "avg_gt_rank": 0.0,
        "skipped": 0,
    }
    gt_rank_sum = 0.0
    for row in per_query:
        if row.get("status") != "ok":
            summary["skipped"] += 1
            continue
        rank = row.get("gt_rank")
        summary["total"] += 1
        if rank == 1:
            summary["top1"] += 1
        if rank is not None and rank <= 3:
            summary["top3"] += 1
        if rank is not None and rank <= 5:
            summary["top5"] += 1
        gt_rank_sum += float(rank or row.get("_rank_fallback") or 99)

    if summary["total"] > 0:
        summary["top1_pct"] = round(100.0 * summary["top1"] / summary["total"], 2)
        summary["top3_pct"] = round(100.0 * summary["top3"] / summary["total"], 2)
        summary["top5_pct"] = round(100.0 * summary["top5"] / summary["total"], 2)
        summary["avg_gt_rank"] = round(gt_rank_sum / summary["total"], 4)
    else:
        summary["top1_pct"] = 0.0
        summary["top3_pct"] = 0.0
        summary["top5_pct"] = 0.0

    _strip_private_ranking_fields(per_query)
    return summary


def _should_mask_telecom_metrics(system_name: str, variant: str) -> bool:
    """Keep current Telecom trace-only baseline unless metric scoring is explicit.

    Telecom key-value metric timestamps are milliseconds, but enabling them with
    the generic scorer currently introduces many persistent symptom false
    positives. Variants containing metricms opt into that experimental path.
    """
    return system_name == "Telecom" and "metricms" not in variant


def _mask_metric_timestamps(telemetry):
    if telemetry.metrics is None or "timestamp" not in telemetry.metrics.columns:
        return telemetry
    metrics = telemetry.metrics.copy()
    metrics["timestamp"] = float("nan")
    return replace(telemetry, metrics=metrics)


def _strip_private_ranking_fields(per_query: List[Dict[str, Any]]) -> None:
    for row in per_query:
        for key in ("_ranking_full", "_gt_component", "_rank_fallback"):
            row.pop(key, None)


def _rank_objects(
    graph,
    noise_scores: Dict[str, Dict[str, Any]],
    structural_scores: Dict[str, Dict[str, Any]],
    delay_scores: Dict[str, Dict[str, Any]],
    beam_scores: Dict[str, Dict[str, Any]],
    subspace_scores: Dict[str, Dict[str, Any]],
    mask_scores: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    ranking = []
    for object_id, node in graph.nodes.items():
        noise = noise_scores.get(object_id, {})
        structure = structural_scores.get(object_id, {})
        delay = delay_scores.get(object_id, {})
        beam = beam_scores.get(object_id, {})
        subspace = subspace_scores.get(object_id, {})
        mask = mask_scores.get(object_id, {})
        final_score = (
            # noise-field: full-spectrum signals
            0.14 * float(noise.get("root_source_score", 0.0))
            + 0.04 * float(noise.get("source_signal", 0.0))
            + 0.06 * float(noise.get("exclusive_explanation", 0.0))
            + 0.06 * float(noise.get("multi_lead_consistency", 0.0))
            + 0.08 * float(noise.get("collapse_gain", 0.0))
            + 0.04 * float(noise.get("collapse_recovery_score", 0.0))
            # structural encoder: full-spectrum signals
            + 0.18 * float(structure.get("structural_score", 0.0))
            + 0.10 * float(structure.get("source_likelihood", 0.0))
            + 0.04 * float(structure.get("propagation_role_score", 0.0))
            + 0.06 * float(structure.get("multi_view_consistency", 0.0))
            + 0.06 * float(structure.get("topological_eccentricity", 0.0))
            # delay, beam, subspace, mask (unchanged weights)
            + 0.14 * float(delay.get("source_time_consistency", 0.0))
            + 0.07 * float(delay.get("average_delay_gain", 0.0))
            + 0.12 * float(beam.get("beamformed_explanation", 0.0))
            + 0.08 * float(beam.get("directional_focus", 0.0))
            + 0.05 * float(beam.get("mechanism_coherence", 0.0))
            + 0.16 * float(subspace.get("local_residual_source_energy", 0.0))
            + 0.08 * float(subspace.get("sector_source_ratio", 0.0))
            + 0.06 * float(subspace.get("residual_distinctiveness", 0.0))
            + 0.05 * float(mask.get("source_protection", 0.0))
            # penalties
            - 0.25 * float(noise.get("hotspot_bias", 0.0))
            - 0.08 * float(noise.get("reverb_mass", 0.0))
            - 0.12 * float(structure.get("symptom_likelihood", 0.0))
            - 0.10 * float(structure.get("hard_negative_resistance", 0.0))
            - 0.10 * float(delay.get("reverse_penalty", 0.0))
            - 0.08 * float(beam.get("offbeam_penalty", 0.0))
            - 0.08 * float(subspace.get("sector_reverb_ratio", 0.0))
            - 0.06 * float(subspace.get("local_common_mode_alignment", 0.0))
            - 0.05 * float(subspace.get("replaceability", 0.0))
            - 0.10 * float(mask.get("gated_reverb_penalty", 0.0))
            - 0.06 * float(mask.get("replaceability_penalty", 0.0))
        )
        ranking.append({
            "object_id": object_id,
            "entity": node.representative,
            "score": round(final_score, 6),
            "reason": node.best_reason(),
            "noise": noise,
            "structure": structure,
            "delay": delay,
            "beam": beam,
            "subspace": subspace,
            "reverb_mask": mask,
        })
    ranking.sort(key=lambda item: item["score"], reverse=True)
    return ranking


def _ground_truth_rank(ranking: List[Dict[str, Any]], ground_truth_component: str) -> Optional[int]:
    from ..mace.graph import _canonical_object_name

    gt_raw = str(ground_truth_component).strip().lower()
    # Instance-labelled GTs such as docker_003, db_007, os_018, and node-1
    # must not be collapsed to docker/db/os/node; otherwise the evaluation
    # credits the correct layer rather than the correct object instance.
    keep_instance_suffix = bool(re.search(r"(?:^|[._-])[a-z]+[-_]\d+(?:$|[._-])", gt_raw))
    gt_canonical = _canonical_object_name(gt_raw, keep_instance_suffix=keep_instance_suffix)
    for idx, candidate in enumerate(ranking, start=1):
        entity = str(candidate["entity"]).strip().lower()
        obj_id = str(candidate.get("object_id", "")).strip().lower()
        obj_canonical = _canonical_object_name(obj_id, keep_instance_suffix=keep_instance_suffix)
        entity_canonical = _canonical_object_name(entity, keep_instance_suffix=keep_instance_suffix)
        # Exact match (raw or canonical)
        if entity == gt_raw or obj_id == gt_raw:
            return idx
        if entity_canonical == gt_canonical or obj_canonical == gt_canonical:
            return idx
        # Prefix match: GT is a prefix segment of the node's canonical form.
        # When keep_instance_suffix is true, this matches node-1.* but not node-6.*.
        if obj_canonical.startswith(gt_canonical + ".") or obj_canonical.startswith(gt_canonical + "-"):
            return idx
    return None


def _top_heatmap(noise_scores: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for object_id, payload in noise_scores.items():
        rows.append({
            "object_id": object_id,
            "root_source_score": payload.get("root_source_score", 0.0),
            "source_signal": payload.get("source_signal", 0.0),
            "exclusive_explanation": payload.get("exclusive_explanation", 0.0),
            "multi_lead_consistency": payload.get("multi_lead_consistency", 0.0),
            "reverb_mass": payload.get("reverb_mass", 0.0),
            "collapse_gain": payload.get("collapse_gain", 0.0),
            "hotspot_bias": payload.get("hotspot_bias", 0.0),
            "influenced_objects": payload.get("influenced_objects", []),
            "noise_field_chain": payload.get("noise_field_chain", []),
        })
    rows.sort(key=lambda item: item["root_source_score"], reverse=True)
    return rows[:8]


def _top_delaymap(delay_scores: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for object_id, payload in delay_scores.items():
        rows.append({
            "object_id": object_id,
            "source_time_consistency": payload.get("source_time_consistency", 0.0),
            "average_delay_gain": payload.get("average_delay_gain", 0.0),
            "reverse_penalty": payload.get("reverse_penalty", 0.0),
            "lead_pairs": payload.get("lead_pairs", []),
        })
    rows.sort(key=lambda item: item["source_time_consistency"], reverse=True)
    return rows[:8]


def _top_beammap(beam_scores: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for object_id, payload in beam_scores.items():
        rows.append({
            "object_id": object_id,
            "beamformed_explanation": payload.get("beamformed_explanation", 0.0),
            "directional_focus": payload.get("directional_focus", 0.0),
            "offbeam_penalty": payload.get("offbeam_penalty", 0.0),
            "mechanism_coherence": payload.get("mechanism_coherence", 0.0),
            "beam_targets": payload.get("beam_targets", []),
        })
    rows.sort(key=lambda item: item["beamformed_explanation"], reverse=True)
    return rows[:8]


def _top_subspacemap(subspace_scores: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for object_id, payload in subspace_scores.items():
        rows.append({
            "object_id": object_id,
            "object_id": object_id,
            "local_residual_source_energy": payload.get("local_residual_source_energy", 0.0),
            "sector_source_ratio": payload.get("sector_source_ratio", 0.0),
            "sector_reverb_ratio": payload.get("sector_reverb_ratio", 0.0),
            "residual_distinctiveness": payload.get("residual_distinctiveness", 0.0),
            "local_common_mode_alignment": payload.get("local_common_mode_alignment", 0.0),
            "replaceability": payload.get("replaceability", 0.0),
        })
    rows.sort(key=lambda item: item["local_residual_source_energy"], reverse=True)
    return rows[:8]


def _top_reverbmap(mask_scores: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for object_id, payload in mask_scores.items():
        rows.append({
            "object_id": object_id,
            "gated_reverb_penalty": payload.get("gated_reverb_penalty", 0.0),
            "source_protection": payload.get("source_protection", 0.0),
            "replaceability_penalty": payload.get("replaceability_penalty", 0.0),
            "local_hub_pressure": payload.get("local_hub_pressure", 0.0),
        })
    rows.sort(key=lambda item: item["gated_reverb_penalty"], reverse=True)
    return rows[:8]


def _write_result(system_name: str, payload: Dict[str, Any], variant: str) -> None:
    result_dir = Path("/home/dell2/RCA513/yyx/rca513/results")
    result_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = result_dir / f"noise_lab_{variant}_{system_name}_{ts}.json"
    file_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone Noise Lab experiment")
    parser.add_argument("--system", required=True, choices=list(SYSTEM_PATHS))
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--variant", default="p21")
    parser.add_argument("--llm-rerank", action="store_true", help="Apply SemanticCausalChainReranker on top-K")
    parser.add_argument("--llm-provider", default="deepseek", choices=["deepseek", "anthropic"])
    parser.add_argument("--llm-model", default="deepseek-chat")
    args = parser.parse_args()
    payload = run_noise_lab(
        system_name=args.system,
        max_queries=args.max_queries,
        variant=args.variant,
        use_llm_reranker=args.llm_rerank,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
    )
    print(json.dumps(payload["summary"], ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
