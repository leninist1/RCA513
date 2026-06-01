"""System-level prior: recognize the system archetype and apply base fault distribution.

Three prototype templates are defined:
  - Microservice with DB-backend (Bank-like): heavy App+DB focus
  - Telecom-style latency-critical: Network+DB+OS triplet
  - Microservice with Heterogeneous Infra (Market-like): mixed, broad

The system is profiled automatically from telemetry structure (entity count, layers,
metric cardinality, trace density) and matched to the closest prototype.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math

import numpy as np


@dataclass
class SystemPrototypeTemplate:
    name: str
    description: str
    app_prior: float
    db_prior: float
    resource_prior: float
    network_prior: float
    other_prior: float
    feature_profile: Dict[str, float]


SYSTEM_PROTOTYPES: Dict[str, SystemPrototypeTemplate] = {
    "Bank-like": SystemPrototypeTemplate(
        name="Bank-like",
        description="Microservice with single DB-backend, service-heavy",
        app_prior=0.50,
        db_prior=0.25,
        resource_prior=0.12,
        network_prior=0.08,
        other_prior=0.05,
        feature_profile={
            "entity_count_norm": 0.5,
            "service_ratio": 0.8,
            "node_ratio": 0.0,
            "trace_coverage": 0.9,
            "metric_cardinality_norm": 0.3,
            "has_logs": 1.0,
            "layer_depth": 1.0,
        },
    ),
    "Telecom-like": SystemPrototypeTemplate(
        name="Telecom-like",
        description="Telecom latency-critical, multi-layer, no-logs, network-heavy",
        app_prior=0.18,
        db_prior=0.28,
        resource_prior=0.24,
        network_prior=0.22,
        other_prior=0.08,
        feature_profile={
            "entity_count_norm": 0.4,
            "service_ratio": 0.3,
            "node_ratio": 0.3,
            "trace_coverage": 0.7,
            "metric_cardinality_norm": 0.8,
            "has_logs": 0.0,
            "layer_depth": 2.0,
        },
    ),
    "Market-like": SystemPrototypeTemplate(
        name="Market-like",
        description="Heterogeneous multi-cloudbed infra, dense traces, broad fault spread",
        app_prior=0.35,
        db_prior=0.20,
        resource_prior=0.22,
        network_prior=0.15,
        other_prior=0.08,
        feature_profile={
            "entity_count_norm": 1.0,
            "service_ratio": 0.4,
            "node_ratio": 0.3,
            "trace_coverage": 0.85,
            "metric_cardinality_norm": 1.0,
            "has_logs": 1.0,
            "layer_depth": 3.0,
        },
    ),
}

FEATURE_WEIGHTS = {
    "entity_count_norm": 0.15,
    "service_ratio": 0.20,
    "node_ratio": 0.15,
    "trace_coverage": 0.20,
    "metric_cardinality_norm": 0.10,
    "has_logs": 0.10,
    "layer_depth": 0.10,
}


class SystemPrototype:
    """Identify system archetype and provide category-level root cause prior."""

    def __init__(self, prototypes: Optional[Dict[str, SystemPrototypeTemplate]] = None):
        self.prototypes = prototypes or SYSTEM_PROTOTYPES
        self._last_similarities: Dict[str, float] = {}

    def profile_system(
        self,
        entities: List[str],
        has_logs: bool,
        trace_edges_estimated: int,
        total_traces: int,
        metric_cardinality: int,
    ) -> Dict[str, float]:
        n = len(entities)
        max_entities = 60

        service_count = sum(
            1 for e in entities if "service" in e.lower() or "svc" in e.lower()
        )
        node_count = sum(
            1 for e in entities if "node" in e.lower() or "host" in e.lower()
        )

        features = {
            "entity_count_norm": min(1.0, n / max_entities),
            "service_ratio": service_count / max(1, n),
            "node_ratio": node_count / max(1, n),
            "trace_coverage": min(1.0, trace_edges_estimated / max(1, n)),
            "metric_cardinality_norm": min(1.0, metric_cardinality / 200),
            "has_logs": 1.0 if has_logs else 0.0,
            "layer_depth": 1.0
            + float(node_count > 0)
            + float(any("cloudbed" in e.lower() for e in entities)),
        }
        return features

    def match(self, features: Dict[str, float]) -> Dict[str, float]:
        similarities = {}
        for name, proto in self.prototypes.items():
            sim = self._weighted_cosine(features, proto.feature_profile)
            similarities[name] = sim
        self._last_similarities = similarities
        return similarities

    def prototype_weights(self, features: Dict[str, float]) -> Dict[str, float]:
        similarities = self.match(features)
        total = sum(similarities.values()) or 1.0
        return {name: s / total for name, s in similarities.items()}

    def blend_cat_prior(self, prototype_weights: Dict[str, float]) -> np.ndarray:
        cats = ["app", "db", "resource", "network", "other"]
        cat_prior = np.zeros(5, dtype=float)
        for proto_name, weight in prototype_weights.items():
            proto = self.prototypes[proto_name]
            cat_prior[0] += weight * proto.app_prior
            cat_prior[1] += weight * proto.db_prior
            cat_prior[2] += weight * proto.resource_prior
            cat_prior[3] += weight * proto.network_prior
            cat_prior[4] += weight * proto.other_prior
        return cat_prior / cat_prior.sum()

    @staticmethod
    def _weighted_cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
        dot, norm_a, norm_b = 0.0, 0.0, 0.0
        for key, weight in FEATURE_WEIGHTS.items():
            va = a.get(key, 0.0) * weight
            vb = b.get(key, 0.0)
            dot += va * vb
            norm_a += va * va
            norm_b += vb * vb
        denom = math.sqrt(norm_a * norm_b)
        if denom < 1e-9:
            return 0.0
        return max(0.0, dot / denom)
