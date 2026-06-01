"""Structural object representation for standalone RCA experiments."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, List

from ..mace.graph import ObjectGraph

EPS = 1e-6

# Mechanism compatibility: which source reason types propagate to which target reasons.
_MECH_COMPAT: Dict[str, frozenset] = {
    "cpu":     frozenset({"cpu", "latency", "process"}),
    "memory":  frozenset({"memory", "process", "latency"}),
    "network": frozenset({"network", "latency", "db"}),
    "disk":    frozenset({"disk", "db", "process"}),
    "db":      frozenset({"db", "latency", "network"}),
    "process": frozenset({"process", "latency", "cpu", "memory"}),
    "change":  frozenset({"change", "cpu", "memory", "latency", "network", "db"}),
    "latency": frozenset({"latency", "network", "db"}),
}


@dataclass
class StructuralEncoding:
    object_id: str
    upstream_context: float
    downstream_context: float
    temporal_lead: float
    mechanism_focus: float
    propagation_signature: float
    source_likelihood: float
    symptom_likelihood: float
    # --- new full-spectrum fields ---
    structural_uniqueness: float     # how much this role differs from graph average
    propagation_role_score: float    # explicit source/prop/symptom continuum
    hard_negative_resistance: float  # penalty for hub/high-exposure anti-pattern
    multi_view_consistency: float    # agreement across structural/delay/mechanism views
    topological_eccentricity: float  # proximity to graph periphery (sources tend to be peripheral)
    # --- composite ---
    structural_score: float
    # --- embedding (12-dim) ---
    embedding: List[float]


class StructuralObjectEncoder:
    """Encode each object using structural role and relation-aware signals.

    Full-spectrum version adds: structural uniqueness, explicit propagation role
    modeling (source/propagation/symptom continuum), hard-negative resistance
    (anti-hub bias), multi-view consistency, and topological eccentricity.
    Embedding grows from 8 to 12 dimensions.
    """

    def encode(self, graph: ObjectGraph) -> Dict[str, Dict[str, Any]]:
        if not graph.nodes:
            return {}

        # Pre-compute graph-wide statistics for uniqueness and eccentricity
        graph_stats = self._graph_statistics(graph)

        raw: Dict[str, StructuralEncoding] = {}
        for object_id, node in graph.nodes.items():
            downstream_context = self._context_mass(graph, object_id, direction="down")
            upstream_context = self._context_mass(graph, object_id, direction="up")
            temporal_lead = self._temporal_lead(graph, object_id)
            mechanism_focus = self._mechanism_focus(node)
            propagation_signature = self._propagation_signature(graph, object_id)

            # Clip context values to [0, 1] for formula use to prevent unbounded penalties.
            # topological_mass / incoming_mass can exceed 1.0 for highly connected nodes.
            dc = min(1.0, downstream_context)
            uc = min(1.0, upstream_context)

            # P0 additions
            propagation_role_score = self._propagation_role(
                graph, object_id,
                temporal_lead, dc, uc,
                propagation_signature,
            )
            hard_negative_resistance = self._hard_negative_resistance(
                graph, object_id, uc, dc
            )
            multi_view_consistency = self._multi_view_consistency(
                graph, object_id,
                temporal_lead, propagation_signature, mechanism_focus,
            )
            topological_eccentricity = self._topological_eccentricity(
                graph, object_id, graph_stats
            )
            structural_uniqueness = self._structural_uniqueness(
                graph, object_id,
                downstream_context, upstream_context, temporal_lead,
                propagation_signature, graph_stats,
            )

            # Source likelihood: high for both hub GTs (collapse many callees) and
            # leaf GTs (self-originated anomaly). Uses forward-direction signals.
            source_likelihood = (
                0.24 * temporal_lead
                + 0.22 * propagation_signature
                + 0.20 * mechanism_focus
                + 0.12 * propagation_role_score
                + 0.10 * topological_eccentricity
                + 0.12 * multi_view_consistency
            )
            # Symptom likelihood: nodes that are explained by graph neighbors.
            # Both high upstream (callers already suffering) and high downstream (calling
            # many anomalous services) context indicates an intermediate node.
            symptom_likelihood = (
                0.30 * uc                           # callers already anomalous (propagation target)
                + 0.25 * dc                          # callees anomalous (victim of failing callees)
                + 0.20 * (1.0 - temporal_lead)
                + 0.15 * max(0.0, node.log_score - node.metric_score)
                + 0.10 * max(0.0, 1.0 - topological_eccentricity)
            )

            structural_score = (
                source_likelihood
                - 0.50 * symptom_likelihood
                + 0.20 * structural_uniqueness
                + 0.20 * multi_view_consistency
                - 0.25 * hard_negative_resistance
            )

            embedding = [
                downstream_context,
                upstream_context,
                temporal_lead,
                mechanism_focus,
                propagation_signature,
                source_likelihood,
                symptom_likelihood,
                structural_score,
                # four new dims
                propagation_role_score,
                hard_negative_resistance,
                multi_view_consistency,
                topological_eccentricity,
            ]
            raw[object_id] = StructuralEncoding(
                object_id=object_id,
                upstream_context=upstream_context,
                downstream_context=downstream_context,
                temporal_lead=temporal_lead,
                mechanism_focus=mechanism_focus,
                propagation_signature=propagation_signature,
                source_likelihood=source_likelihood,
                symptom_likelihood=symptom_likelihood,
                structural_uniqueness=structural_uniqueness,
                propagation_role_score=propagation_role_score,
                hard_negative_resistance=hard_negative_resistance,
                multi_view_consistency=multi_view_consistency,
                topological_eccentricity=topological_eccentricity,
                structural_score=structural_score,
                embedding=embedding,
            )

        normalized = self._normalize(raw)
        return {
            obj: {
                "upstream_context": round(item.upstream_context, 6),
                "downstream_context": round(item.downstream_context, 6),
                "temporal_lead": round(item.temporal_lead, 6),
                "mechanism_focus": round(item.mechanism_focus, 6),
                "propagation_signature": round(item.propagation_signature, 6),
                "source_likelihood": round(item.source_likelihood, 6),
                "symptom_likelihood": round(item.symptom_likelihood, 6),
                "structural_uniqueness": round(item.structural_uniqueness, 6),
                "propagation_role_score": round(item.propagation_role_score, 6),
                "hard_negative_resistance": round(item.hard_negative_resistance, 6),
                "multi_view_consistency": round(item.multi_view_consistency, 6),
                "topological_eccentricity": round(item.topological_eccentricity, 6),
                "structural_score": round(item.structural_score, 6),
                "embedding": [round(v, 6) for v in item.embedding],
            }
            for obj, item in normalized.items()
        }

    # ------------------------------------------------------------------
    # Propagation role: source / propagator / symptom continuum
    # ------------------------------------------------------------------

    def _propagation_role(
        self,
        graph: ObjectGraph,
        object_id: str,
        temporal_lead: float,
        downstream_context: float,
        upstream_context: float,
        propagation_signature: float,
    ) -> float:
        """Continuous role score: +1 = pure source, 0 = propagator, -1 = pure symptom.

        Context values here are clipped [0, 1] by caller.
        A source node shows high temporal lead and directly collapses callees.
        An intermediate/symptom node is surrounded by anomaly from all directions.
        """
        # Source: leads in time, propagates to callees, OR is peripheral (few ancestors)
        src_score = (
            0.40 * temporal_lead
            + 0.30 * downstream_context       # has anomalous callees = collapse candidate
            + 0.30 * propagation_signature    # actively propagating to callees
        )
        # Symptom: late onset, surrounded by anomalous neighbors
        sym_score = (
            0.40 * (1.0 - temporal_lead)
            + 0.30 * upstream_context         # callers already anomalous
            + 0.30 * (1.0 - propagation_signature)
        )
        return float(max(0.0, min(1.0, (src_score - sym_score + 1.0) / 2.0)))

    # ------------------------------------------------------------------
    # Hard-negative resistance
    # ------------------------------------------------------------------

    def _hard_negative_resistance(
        self,
        graph: ObjectGraph,
        object_id: str,
        upstream_context: float,
        downstream_context: float,
    ) -> float:
        """Penalty for objects that look anomalous only because neighbors are anomalous,
        or that are completely isolated from the call graph.

        Clipped [0,1] context values expected.
        """
        node = graph.nodes[object_id]
        degree_out = len(graph.adjacency.get(object_id, {}))
        degree_in = sum(1 for _, ch in graph.adjacency.items() if object_id in ch)
        # Surrounded by anomaly from all directions = hard negative
        context_saturation = max(upstream_context, downstream_context)
        # High member count without a unique structural role → aggregator
        member_exposure = math.tanh(0.3 * len(node.members))
        # Isolated: completely disconnected from the call graph — anomaly cannot cascade
        is_isolated = 1.0 if (degree_in == 0 and degree_out == 0) else 0.0
        return float(
            0.40 * is_isolated             # strongly penalize disconnected nodes
            + 0.35 * context_saturation
            + 0.15 * member_exposure
            + 0.10 * math.tanh(0.3 * degree_out) * (1.0 if downstream_context > 0.5 else 0.0)
        )

    # ------------------------------------------------------------------
    # Multi-view consistency
    # ------------------------------------------------------------------

    def _multi_view_consistency(
        self,
        graph: ObjectGraph,
        object_id: str,
        temporal_lead: float,
        propagation_signature: float,
        mechanism_focus: float,
    ) -> float:
        """Agreement score across three independent views: temporal, structural, mechanism.

        High consistency = all views agree this node is a source.
        Low consistency = views contradict each other (noisy intermediate node).

        Uses variance of the three source-signal proxies: low variance = consistent.
        """
        views = [temporal_lead, propagation_signature, mechanism_focus]
        mean_v = sum(views) / len(views)
        variance = sum((v - mean_v) ** 2 for v in views) / len(views)
        # consistency = mean * (1 - normalized_std)
        std = math.sqrt(variance)
        consistency = mean_v * max(0.0, 1.0 - std)
        return float(min(1.0, consistency))

    # ------------------------------------------------------------------
    # Topological eccentricity
    # ------------------------------------------------------------------

    def _graph_statistics(self, graph: ObjectGraph) -> Dict[str, Any]:
        """Pre-compute per-graph stats: BFS shortest paths and degree distribution."""
        # BFS from each node to compute eccentricity approximation
        nodes = list(graph.nodes.keys())
        n = len(nodes)
        # avg out-degree for uniqueness baseline
        avg_downstream = (
            sum(
                sum(
                    graph.nodes[ch].anomaly_score * w
                    for ch, w in graph.adjacency.get(obj, {}).items()
                    if ch in graph.nodes
                )
                for obj in nodes
            ) / max(1, n)
        )
        avg_upstream = (
            sum(graph.incoming_mass(obj) for obj in nodes) / max(1, n)
        )
        avg_temporal = 0.0  # filled below to avoid double compute
        return {
            "avg_downstream": avg_downstream,
            "avg_upstream": avg_upstream,
            "n_nodes": n,
        }

    def _topological_eccentricity(
        self, graph: ObjectGraph, object_id: str, graph_stats: Dict[str, Any]
    ) -> float:
        """Measure how 'peripheral' this node is within the connected graph.

        Root causes tend to sit at the periphery (no or few upstream ancestors).
        Aggregators/symptoms sit in the center (many ancestors).
        Isolated nodes (no edges at all) return 0.5 — neither peripheral nor central.
        """
        n = max(1, graph_stats["n_nodes"])
        degree_out = len(graph.adjacency.get(object_id, {}))
        degree_in = sum(
            1 for _, ch in graph.adjacency.items() if object_id in ch
        )
        # Isolated node: not connected to anything — penalize with neutral score
        if degree_in == 0 and degree_out == 0:
            return 0.0

        # Count ancestors reachable by following edges backwards
        visited: set = {object_id}
        frontier = {object_id}
        for _ in range(5):
            nf = set()
            for src in frontier:
                for parent, children in graph.adjacency.items():
                    if src in children and parent not in visited and parent in graph.nodes:
                        nf.add(parent)
                        visited.add(parent)
            frontier = nf
            if not frontier:
                break

        ancestor_count = len(visited) - 1  # exclude self
        # eccentricity = 1 when no ancestors (pure source), 0 when fully embedded
        return float(max(0.0, 1.0 - ancestor_count / max(1, n - 1)))

    # ------------------------------------------------------------------
    # Structural uniqueness
    # ------------------------------------------------------------------

    def _structural_uniqueness(
        self,
        graph: ObjectGraph,
        object_id: str,
        downstream_context: float,
        upstream_context: float,
        temporal_lead: float,
        propagation_signature: float,
        graph_stats: Dict[str, Any],
    ) -> float:
        """How much this node's structural profile deviates from graph-average (L2 norm)."""
        avg_ds = graph_stats["avg_downstream"]
        avg_us = graph_stats["avg_upstream"]
        n = max(1, graph_stats["n_nodes"])

        # Compute graph-mean temporal lead as simple mean
        all_tl = [self._temporal_lead(graph, obj) for obj in graph.nodes]
        avg_tl = sum(all_tl) / max(1, len(all_tl))

        diff = math.sqrt(
            (downstream_context - avg_ds) ** 2
            + (upstream_context - avg_us) ** 2
            + (temporal_lead - avg_tl) ** 2
        )
        return float(math.tanh(diff))

    # ------------------------------------------------------------------
    # Existing helpers (kept / lightly refined)
    # ------------------------------------------------------------------

    def _context_mass(
        self, graph: ObjectGraph, object_id: str, direction: str
    ) -> float:
        if direction == "down":
            # Average anomaly of direct callees (not sum), to avoid out-degree bias.
            # This makes single-callee nodes (Mysql02 calling apache01) comparable to
            # multi-callee hubs (IG01 calling 4 nodes).
            callees = graph.adjacency.get(object_id, {})
            if not callees:
                return 0.0
            total_anom = sum(
                graph.nodes[dst].anomaly_score
                for dst in callees
                if dst in graph.nodes
            )
            n_callees = sum(1 for dst in callees if dst in graph.nodes)
            return total_anom / max(1, n_callees)
        return graph.incoming_mass(object_id)

    def _temporal_lead(self, graph: ObjectGraph, object_id: str) -> float:
        node = graph.nodes[object_id]
        if node.earliest_timestamp is None:
            return 0.5
        comparisons = []
        for child, _ in graph.adjacency.get(object_id, {}).items():
            if child not in graph.nodes:
                continue
            child_ts = graph.nodes[child].earliest_timestamp
            if child_ts is None:
                comparisons.append(0.5)
            elif node.earliest_timestamp <= child_ts:
                comparisons.append(1.0)
            else:
                comparisons.append(0.0)
        for parent, children in graph.adjacency.items():
            if object_id not in children or parent not in graph.nodes:
                continue
            parent_ts = graph.nodes[parent].earliest_timestamp
            if parent_ts is None:
                comparisons.append(0.5)
            elif node.earliest_timestamp <= parent_ts:
                comparisons.append(1.0)
            else:
                comparisons.append(0.0)
        if not comparisons:
            return 0.5
        return sum(comparisons) / len(comparisons)

    def _mechanism_focus(self, node) -> float:
        votes = list(node.reason_votes.values())
        if not votes:
            return 0.4
        total = sum(votes)
        if total <= EPS:
            return 0.4
        peak = max(votes)
        entropy = 0.0
        for v in votes:
            p = v / total
            entropy -= p * math.log(max(EPS, p))
        max_entropy = math.log(max(2, len(votes)))
        concentration = peak / total
        entropy_term = 1.0 - entropy / max(EPS, max_entropy)
        return 0.55 * concentration + 0.45 * entropy_term

    def _propagation_signature(self, graph: ObjectGraph, object_id: str) -> float:
        node = graph.nodes[object_id]
        downstream = graph.adjacency.get(object_id, {})
        if not downstream:
            return 0.1 * node.trace_score + 0.2 * node.metric_score
        score = 0.0
        weight_sum = 0.0
        for child, weight in downstream.items():
            if child not in graph.nodes:
                continue
            child_node = graph.nodes[child]
            score += weight * (
                0.45 * child_node.trace_score
                + 0.35 * child_node.metric_score
                + 0.20 * child_node.log_score
            )
            weight_sum += weight
        base = score / max(EPS, weight_sum)
        return 0.6 * base + 0.25 * node.trace_score + 0.15 * node.metric_score

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    def _normalize(
        self, encodings: Dict[str, StructuralEncoding]
    ) -> Dict[str, StructuralEncoding]:
        fields = [
            "upstream_context",
            "downstream_context",
            "temporal_lead",
            "mechanism_focus",
            "propagation_signature",
            "source_likelihood",
            "symptom_likelihood",
            "structural_uniqueness",
            "propagation_role_score",
            "hard_negative_resistance",
            "multi_view_consistency",
            "topological_eccentricity",
            "structural_score",
        ]
        nv: Dict[str, Dict[str, float]] = {f: {} for f in fields}
        for f in fields:
            vals = [getattr(item, f) for item in encodings.values()]
            lo, hi = min(vals), max(vals)
            for obj, item in encodings.items():
                v = getattr(item, f)
                if math.isclose(lo, hi):
                    nv[f][obj] = 0.5
                else:
                    nv[f][obj] = (v - lo) / max(EPS, hi - lo)
        output: Dict[str, StructuralEncoding] = {}
        for obj, item in encodings.items():
            # 12-dim embedding from normalized fields
            embedding = [nv[f][obj] for f in fields[:12]]
            output[obj] = StructuralEncoding(
                object_id=obj,
                upstream_context=nv["upstream_context"][obj],
                downstream_context=nv["downstream_context"][obj],
                temporal_lead=nv["temporal_lead"][obj],
                mechanism_focus=nv["mechanism_focus"][obj],
                propagation_signature=nv["propagation_signature"][obj],
                source_likelihood=nv["source_likelihood"][obj],
                symptom_likelihood=nv["symptom_likelihood"][obj],
                structural_uniqueness=nv["structural_uniqueness"][obj],
                propagation_role_score=nv["propagation_role_score"][obj],
                hard_negative_resistance=nv["hard_negative_resistance"][obj],
                multi_view_consistency=nv["multi_view_consistency"][obj],
                topological_eccentricity=nv["topological_eccentricity"][obj],
                structural_score=nv["structural_score"][obj],
                embedding=embedding,
            )
        return output
