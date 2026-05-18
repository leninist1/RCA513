"""Strategy C: Causal propagation chain tracing.

Best for topological propagation failures.
Walks the dependency graph to find cascading failure paths.
Cost: 0 API calls (pure graph traversal + compute).
"""

import numpy as np
import pandas as pd
from collections import defaultdict
from typing import List, Dict, Optional

from .base import BaseStrategy
from ..config import Candidate, StrategyResult, ResourceBudget, ResourceCost, UnifiedTelemetry


class StrategyC(BaseStrategy):
    name = "C_causal_trace"
    cost_profile = {"api_calls": 0, "compute_seconds": 0.3}

    def execute(
        self, candidates: List[Candidate], telemetry: UnifiedTelemetry,
        baseline_df: pd.DataFrame, fault_df: pd.DataFrame,
        budget: ResourceBudget, engine=None,
    ) -> StrategyResult:
        """Trace causal propagation along dependency graph.

        Algorithm:
        1. Build full dependency graph from traces (or metric-inferred)
        2. For each candidate, compute propagation path:
           - Walk upstream to find potential root
           - Walk downstream to find blast radius
        3. Score by: upstream_score (possible root) + downstream_impact (blast radius)
        """
        result = StrategyResult(strategy_name=self.name)

        if engine is None or not candidates:
            return result

        # Build graph
        graph = {}
        if telemetry.traces is not None and not telemetry.traces.empty:
            from ..counterfactual.graph import build_graph_from_traces
            graph = build_graph_from_traces(telemetry.traces)
        if not graph and not baseline_df.empty:
            from ..counterfactual.graph import infer_graph_from_metrics
            graph = infer_graph_from_metrics(baseline_df, fault_df, telemetry.entities)

        if not graph:
            return result

        verified = []
        evidence_added = {}

        for c in candidates:
            upstream = self._find_upstream(c.entity, graph)
            downstream = self._find_downstream(c.entity, graph)

            # Score upstream (potential true root cause)
            upstream_scores = {}
            for up_entity in upstream:
                cf_df = engine.apply_counterfactual(fault_df, baseline_df, up_entity)
                rec = engine.compute_recovery(
                    fault_df, cf_df, baseline_df, up_entity,
                    entities=telemetry.entities, logs_df=telemetry.logs
                )
                upstream_scores[up_entity] = rec

            # Score downstream impact (blast radius)
            downstream_scores = {}
            for down_entity in downstream:
                entity_b = baseline_df[baseline_df["entity"] == down_entity]
                entity_f = fault_df[fault_df["entity"] == down_entity]
                if not entity_f.empty and not entity_b.empty:
                    deg = engine._entity_degradation(fault_df, baseline_df, down_entity, telemetry.logs)
                    downstream_scores[down_entity] = float(deg)

            # Combined causal score: best upstream recovery + downstream magnitude
            best_upstream = max(upstream_scores.values()) if upstream_scores else 0
            total_downstream = sum(downstream_scores.values()) if downstream_scores else 0
            normalized_downstream = np.log1p(total_downstream) / 100.0

            causal_score = best_upstream * 0.7 + min(1.0, normalized_downstream) * 0.3

            evidence = [
                {"type": "causal_trace", "upstream": upstream_scores, "downstream": downstream_scores},
                {"type": "propagation_path", "upstream_chain": upstream, "downstream_chain": downstream},
                {"type": "causal_score", "score": causal_score},
            ]

            causal_c = Candidate(
                entity=c.entity,
                anomaly_score=c.anomaly_score,
                recovery_score=max(c.recovery_score or 0, causal_score),
                verification_depth="medium",
                evidence=evidence,
            )
            verified.append(causal_c)
            evidence_added[c.entity] = evidence

        verified.sort(key=lambda x: x.recovery_score or 0, reverse=True)
        result.candidates_verified = verified
        result.evidence_added = evidence_added
        result.cost = ResourceCost(api_calls=0)
        return result

    def _find_upstream(self, entity: str, graph: Dict, max_depth: int = 5) -> List[str]:
        """Find all upstream services (potential true root causes)."""
        reverse_graph = defaultdict(set)
        for parent, children in graph.items():
            for child in children:
                reverse_graph[child].add(parent)

        upstream = []
        visited = set()
        queue = [entity]
        for depth in range(max_depth):
            if not queue:
                break
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            for parent in reverse_graph.get(current, set()):
                if parent not in visited:
                    upstream.append(parent)
                    queue.append(parent)
        return upstream

    def _find_downstream(self, entity: str, graph: Dict, max_depth: int = 5) -> List[str]:
        """Find all downstream services (blast radius)."""
        downstream = []
        visited = set()
        queue = list(graph.get(entity, {}).keys())
        for depth in range(max_depth):
            if not queue:
                break
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            downstream.append(current)
            for child in graph.get(current, {}).keys():
                if child not in visited:
                    queue.append(child)
        return downstream
