"""Protected reverberation suppression for Noise Lab P2.1."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict

from ..mace.graph import ObjectGraph

EPS = 1e-6


@dataclass
class ReverbMaskScore:
    object_id: str
    gated_reverb_penalty: float
    source_protection: float
    replaceability_penalty: float
    local_hub_pressure: float


class ReverbSuppressionMask:
    """Suppress reverberation only when source evidence is weak after local de-reverb."""

    def score(
        self,
        graph: ObjectGraph,
        noise_scores: Dict[str, Dict[str, Any]],
        structural_scores: Dict[str, Dict[str, Any]],
        delay_scores: Dict[str, Dict[str, Any]],
        beam_scores: Dict[str, Dict[str, Any]],
        subspace_scores: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        if not graph.nodes:
            return {}
        raw: Dict[str, ReverbMaskScore] = {}
        for object_id, node in graph.nodes.items():
            incoming = graph.incoming_mass(object_id)
            outgoing = graph.topological_mass(object_id)
            incoming_dom = incoming / max(EPS, incoming + outgoing)
            hotspot = float(noise_scores.get(object_id, {}).get("hotspot_bias", 0.0))
            symptom = float(structural_scores.get(object_id, {}).get("symptom_likelihood", 0.0))
            reverse_penalty = float(delay_scores.get(object_id, {}).get("reverse_penalty", 0.0))
            delay_source = float(delay_scores.get(object_id, {}).get("source_time_consistency", 0.0))
            observer_coverage = float(delay_scores.get(object_id, {}).get("observer_coverage", 0.0))
            directional_focus = float(beam_scores.get(object_id, {}).get("directional_focus", 0.0))
            offbeam_penalty = float(beam_scores.get(object_id, {}).get("offbeam_penalty", 0.0))
            local_residual = float(subspace_scores.get(object_id, {}).get("local_residual_source_energy", 0.0))
            sector_reverb = float(subspace_scores.get(object_id, {}).get("sector_reverb_ratio", 0.0))
            replaceability = float(subspace_scores.get(object_id, {}).get("replaceability", 0.0))
            residual_distinctiveness = float(subspace_scores.get(object_id, {}).get("residual_distinctiveness", 0.0))
            collapse_gain = float(noise_scores.get(object_id, {}).get("collapse_gain", 0.0))

            local_hub_pressure = (
                0.20 * incoming_dom
                + 0.16 * hotspot
                + 0.14 * symptom
                + 0.16 * sector_reverb
                + 0.10 * replaceability
                + 0.10 * offbeam_penalty
                + 0.08 * reverse_penalty
                + 0.06 * max(0.0, 1.0 - directional_focus)
            )
            source_protection = (
                0.28 * delay_source
                + 0.28 * local_residual
                + 0.18 * collapse_gain
                + 0.14 * residual_distinctiveness
                + 0.12 * observer_coverage
            )
            replaceability_penalty = 0.55 * replaceability + 0.45 * sector_reverb
            gated_reverb_penalty = max(
                0.0,
                (0.72 * local_hub_pressure + 0.28 * replaceability_penalty) * (1.0 - 0.75 * source_protection),
            )
            raw[object_id] = ReverbMaskScore(
                object_id=object_id,
                gated_reverb_penalty=gated_reverb_penalty,
                source_protection=source_protection,
                replaceability_penalty=replaceability_penalty,
                local_hub_pressure=local_hub_pressure,
            )
        normalized = self._normalize(raw)
        return {
            object_id: {
                "gated_reverb_penalty": round(item.gated_reverb_penalty, 6),
                "source_protection": round(item.source_protection, 6),
                "replaceability_penalty": round(item.replaceability_penalty, 6),
                "local_hub_pressure": round(item.local_hub_pressure, 6),
            }
            for object_id, item in normalized.items()
        }

    def _normalize(self, raw: Dict[str, ReverbMaskScore]) -> Dict[str, ReverbMaskScore]:
        fields = [
            "gated_reverb_penalty",
            "source_protection",
            "replaceability_penalty",
            "local_hub_pressure",
        ]
        normalized: Dict[str, Dict[str, float]] = {field: {} for field in fields}
        for field in fields:
            values = [getattr(item, field) for item in raw.values()]
            lo = min(values)
            hi = max(values)
            for object_id, item in raw.items():
                value = getattr(item, field)
                if math.isclose(lo, hi):
                    normalized[field][object_id] = 0.5
                else:
                    normalized[field][object_id] = (value - lo) / max(EPS, hi - lo)
        output: Dict[str, ReverbMaskScore] = {}
        for object_id, item in raw.items():
            output[object_id] = ReverbMaskScore(
                object_id=object_id,
                gated_reverb_penalty=normalized["gated_reverb_penalty"][object_id],
                source_protection=normalized["source_protection"][object_id],
                replaceability_penalty=normalized["replaceability_penalty"][object_id],
                local_hub_pressure=normalized["local_hub_pressure"][object_id],
            )
        return output
