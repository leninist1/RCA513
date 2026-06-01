"""Conditional layer transition attenuation matrices.

Unlike the static LAYER_TRANSITIONS in prism.py, these matrices adapt
cross-layer propagation attenuation based on the suspected fault category.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple
import numpy as np


TransitionKey = Tuple[str, str]

LAYER_TRANSITIONS_CONDITIONAL: Dict[str, Dict[TransitionKey, float]] = {
    "Resource": {
        ("pod", "pod"): 1.0,
        ("pod", "service"): 0.60,
        ("pod", "node"): 0.90,
        ("service", "pod"): 0.60,
        ("service", "service"): 1.0,
        ("service", "node"): 0.80,
        ("node", "pod"): 0.90,
        ("node", "service"): 0.80,
        ("node", "node"): 1.0,
    },
    "Network": {
        ("pod", "pod"): 1.0,
        ("pod", "service"): 0.95,
        ("pod", "node"): 0.80,
        ("service", "pod"): 0.95,
        ("service", "service"): 1.0,
        ("service", "node"): 0.85,
        ("node", "pod"): 0.80,
        ("node", "service"): 0.85,
        ("node", "node"): 1.0,
    },
    "Application": {
        ("pod", "pod"): 1.0,
        ("pod", "service"): 0.90,
        ("pod", "node"): 0.40,
        ("service", "pod"): 0.90,
        ("service", "service"): 1.0,
        ("service", "node"): 0.50,
        ("node", "pod"): 0.40,
        ("node", "service"): 0.50,
        ("node", "node"): 1.0,
    },
    "Database": {
        ("pod", "pod"): 1.0,
        ("pod", "service"): 0.75,
        ("pod", "node"): 0.65,
        ("service", "pod"): 0.75,
        ("service", "service"): 1.0,
        ("service", "node"): 0.70,
        ("node", "pod"): 0.65,
        ("node", "service"): 0.70,
        ("node", "node"): 1.0,
    },
    "generic": {
        ("pod", "pod"): 1.0,
        ("pod", "service"): 0.85,
        ("pod", "node"): 0.70,
        ("service", "pod"): 0.85,
        ("service", "service"): 1.0,
        ("service", "node"): 0.75,
        ("node", "pod"): 0.70,
        ("node", "service"): 0.75,
        ("node", "node"): 1.0,
    },
}


class LayerTransitionsConditional:
    """Conditional cross-layer propagation attenuation."""

    def __init__(self):
        self._transitions = dict(LAYER_TRANSITIONS_CONDITIONAL)

    def get_for_category(self, fault_category: str) -> Dict[TransitionKey, float]:
        cat_key = fault_category
        for known in ("Resource", "Network", "Application", "Database"):
            if known.lower() in cat_key.lower():
                cat_key = known
                break
        return self._transitions.get(cat_key, self._transitions["generic"])

    def blend_by_category_weights(
        self,
        category_weights: Dict[str, float],
    ) -> Dict[TransitionKey, float]:
        blended: Dict[TransitionKey, float] = {}
        weight_sum = 0.0
        for cat, weight in category_weights.items():
            transitions = self.get_for_category(cat)
            for key, val in transitions.items():
                blended[key] = blended.get(key, 0.0) + weight * val
            weight_sum += weight
        if weight_sum > 0:
            for key in blended:
                blended[key] /= weight_sum
        else:
            blended = dict(self._transitions["generic"])
        return blended

    def get_attenuation(
        self,
        from_layer: str,
        to_layer: str,
        fault_category: str = "generic",
    ) -> float:
        transitions = self.get_for_category(fault_category)
        key = (from_layer, to_layer)
        return transitions.get(key, transitions.get(("pod", "pod"), 1.0))

    def build_alpha_matrix(
        self,
        entities: list,
        entity_types: Dict[str, str],
        fault_category: str = "generic",
        alpha_base: float = 0.8,
    ) -> np.ndarray:
        n = len(entities)
        alpha = np.zeros((n, n), dtype=float)
        transitions = self.get_for_category(fault_category)
        for i, u in enumerate(entities):
            for j, v in enumerate(entities):
                from_layer = entity_types.get(u, "pod")
                to_layer = entity_types.get(v, "pod")
                key = (from_layer, to_layer)
                alpha[i, j] = alpha_base * transitions.get(key, 1.0)
        return alpha
