"""Dependency-light NoiseLab candidate ranking."""

from __future__ import annotations

from typing import Any, Dict, List


def rank_objects(
    graph: Any,
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
            0.14 * float(noise.get("root_source_score", 0.0))
            + 0.04 * float(noise.get("source_signal", 0.0))
            + 0.06 * float(noise.get("exclusive_explanation", 0.0))
            + 0.06 * float(noise.get("multi_lead_consistency", 0.0))
            + 0.08 * float(noise.get("collapse_gain", 0.0))
            + 0.04 * float(noise.get("collapse_recovery_score", 0.0))
            + 0.18 * float(structure.get("structural_score", 0.0))
            + 0.10 * float(structure.get("source_likelihood", 0.0))
            + 0.04 * float(structure.get("propagation_role_score", 0.0))
            + 0.06 * float(structure.get("multi_view_consistency", 0.0))
            + 0.06 * float(structure.get("topological_eccentricity", 0.0))
            + 0.14 * float(delay.get("source_time_consistency", 0.0))
            + 0.07 * float(delay.get("average_delay_gain", 0.0))
            + 0.12 * float(beam.get("beamformed_explanation", 0.0))
            + 0.08 * float(beam.get("directional_focus", 0.0))
            + 0.05 * float(beam.get("mechanism_coherence", 0.0))
            + 0.16 * float(subspace.get("local_residual_source_energy", 0.0))
            + 0.08 * float(subspace.get("sector_source_ratio", 0.0))
            + 0.06 * float(subspace.get("residual_distinctiveness", 0.0))
            + 0.05 * float(mask.get("source_protection", 0.0))
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
        ranking.append(
            {
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
            }
        )
    ranking.sort(key=lambda item: item["score"], reverse=True)
    return ranking
