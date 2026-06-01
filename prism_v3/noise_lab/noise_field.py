"""Noise-field modeling for standalone root-source ranking experiments."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, List, Optional

from ..mace.graph import ObjectGraph
from ..mace.intervention import evaluate_local_intervention

EPS = 1e-6

# How far upstream to trace when stripping reverb entering a candidate.
_REVERB_STRIP_DEPTH = 2
# Decay per hop when estimating incoming reverb.
_REVERB_DECAY = 0.60
# Exponential half-life for temporal alignment (seconds).
_TIME_HALFLIFE = 1800.0


@dataclass
class NoiseFieldScore:
    object_id: str
    # --- core anomaly signal ---
    local_noise: float
    resonance_mass: float
    # --- source/reverb decomposition ---
    source_signal: float          # direct causal energy pushed downstream
    reverb_mass: float            # incoming reverb absorbed from upstream sources
    unexplained_residual: float   # self-anomaly not explained by any upstream neighbor
    # --- intervention-based ---
    collapse_gain: float
    collapse_recovery_score: float  # fraction of downstream anomaly deflated on removal
    # --- uniqueness / exclusivity ---
    exclusive_explanation: float  # mass not coverable by any single alternative candidate
    # --- temporal ---
    temporal_source_score: float
    multi_lead_consistency: float  # TDOA-style: weighted fraction of BFS observers this node leads
    # --- bias penalty ---
    hotspot_bias: float
    # --- composite ---
    root_source_score: float
    # --- interpretability ---
    influenced_objects: List[str]
    noise_field_chain: List[str]   # greedy propagation chain anchored at this node


class NoiseFieldScorer:
    """Estimate which object most plausibly generates the observed noise field.

    Full-spectrum version: source/reverb decomposition, exclusive explanation,
    TDOA-style multi-lead consistency, collapse recovery, stronger hotspot penalty.
    """

    def score(self, graph: ObjectGraph) -> Dict[str, Dict[str, Any]]:
        if not graph.nodes:
            return {}

        # Forward interventions: used for collapse/resonance signals (helps hub GTs)
        interventions = {
            obj: evaluate_local_intervention(graph, obj)
            for obj in graph.nodes
        }
        all_covered = self._global_coverage_union(graph, interventions)

        # Reverse graph: for leaf-style root causes (DB/cache) that have no callees,
        # compute caller-side exclusive coverage as a supplementary signal.
        rev_graph = self._build_reverse_graph(graph)
        rev_interventions = {
            obj: evaluate_local_intervention(rev_graph, obj)
            for obj in graph.nodes
        }
        rev_covered = self._global_coverage_union(rev_graph, rev_interventions)

        raw: Dict[str, NoiseFieldScore] = {}
        for object_id, node in graph.nodes.items():
            local_noise = node.anomaly_score
            resonance_mass = self._resonance_mass(graph, object_id)
            source_signal, reverb_mass, unexplained_residual = self._source_reverb_decompose(
                graph, object_id
            )
            temporal_source_score = self._temporal_source_score(graph, object_id)
            multi_lead = self._multi_lead_consistency(graph, object_id)
            # Forward collapse: removing node deflates its anomalous callees (hub GT cases)
            collapse_gain = self._collapse_gain(
                graph, object_id, interventions[object_id]
            )
            collapse_recovery = self._collapse_recovery_score(
                graph, object_id, interventions[object_id]
            )
            # Exclusive explanation: max of forward (hub coverage) and reverse (leaf coverage)
            fwd_excl = self._exclusive_explanation(
                graph, object_id, interventions[object_id], all_covered
            )
            rev_excl = self._exclusive_explanation(
                rev_graph, object_id, rev_interventions[object_id], rev_covered
            )
            exclusive_explanation = max(fwd_excl, rev_excl)
            hotspot_bias = self._hotspot_bias(graph, object_id)

            # Connectivity gating: isolated nodes (no edges) cannot cause cascades.
            # Gate propagation-based signals to prevent isolation artifacts from dominating.
            degree_out = len(graph.adjacency.get(object_id, {}))
            degree_in = sum(1 for _, ch in graph.adjacency.items() if object_id in ch)
            connectivity_gate = 0.15 if (degree_in == 0 and degree_out == 0) else 1.0

            impacted = interventions[object_id].get("impacted", {})
            influenced_objects = [
                dst
                for dst, w in sorted(
                    impacted.items(), key=lambda kv: kv[1], reverse=True
                )
                if dst != object_id
            ][:6]
            noise_field_chain = self._build_field_chain(graph, object_id)

            root_source_score = (
                0.18 * local_noise
                + 0.10 * resonance_mass * connectivity_gate
                + 0.14 * source_signal
                + 0.12 * exclusive_explanation * connectivity_gate
                + 0.14 * multi_lead
                + 0.12 * collapse_gain * connectivity_gate
                + 0.08 * collapse_recovery * connectivity_gate
                + 0.12 * unexplained_residual * connectivity_gate
                - 0.24 * hotspot_bias
                - 0.10 * reverb_mass
            )
            raw[object_id] = NoiseFieldScore(
                object_id=object_id,
                local_noise=local_noise,
                resonance_mass=resonance_mass,
                source_signal=source_signal,
                reverb_mass=reverb_mass,
                unexplained_residual=unexplained_residual,
                collapse_gain=collapse_gain,
                collapse_recovery_score=collapse_recovery,
                exclusive_explanation=exclusive_explanation,
                temporal_source_score=temporal_source_score,
                multi_lead_consistency=multi_lead,
                hotspot_bias=hotspot_bias,
                root_source_score=root_source_score,
                influenced_objects=influenced_objects,
                noise_field_chain=noise_field_chain,
            )

        normalized = self._normalize(raw)
        return {
            obj: {
                "local_noise": round(item.local_noise, 6),
                "resonance_mass": round(item.resonance_mass, 6),
                "source_signal": round(item.source_signal, 6),
                "reverb_mass": round(item.reverb_mass, 6),
                "unexplained_residual": round(item.unexplained_residual, 6),
                "collapse_gain": round(item.collapse_gain, 6),
                "collapse_recovery_score": round(item.collapse_recovery_score, 6),
                "exclusive_explanation": round(item.exclusive_explanation, 6),
                "temporal_source_score": round(item.temporal_source_score, 6),
                "multi_lead_consistency": round(item.multi_lead_consistency, 6),
                "hotspot_bias": round(item.hotspot_bias, 6),
                "root_source_score": round(item.root_source_score, 6),
                "influenced_objects": item.influenced_objects,
                "noise_field_chain": item.noise_field_chain,
            }
            for obj, item in normalized.items()
        }

    # ------------------------------------------------------------------
    # Source / reverb decomposition
    # ------------------------------------------------------------------

    def _source_reverb_decompose(
        self, graph: ObjectGraph, object_id: str
    ) -> tuple[float, float, float]:
        """Return (source_signal, reverb_mass, unexplained_residual).

        In call-direction graphs (caller→callee):
        - Fault propagates UPSTREAM: callee fails → callers slow down.
        - Reverb = fraction of this node's own anomaly that can be attributed to
          anomalous CALLEES (downstream). High reverb = symptom (explained by callees).
          Low reverb = root cause (self-originated anomaly).
        - source_signal = self-anomaly fraction NOT explained by failing callees.
        - unexplained_residual = max(0, 1 - callee_reverb_fraction).
        """
        node = graph.nodes[object_id]
        self_anomaly = max(node.anomaly_score, EPS)

        # Reverb: sum of weighted callee anomaly, as a FRACTION of this node's own anomaly.
        # This makes reverb comparable regardless of how many callees there are.
        out_edges = graph.adjacency.get(object_id, {})
        n_direct_callees = max(1, len(out_edges))
        reverb_sum = 0.0
        frontier: Dict[str, float] = dict(out_edges)  # direct callees
        self_ts = node.earliest_timestamp

        visited = set(frontier.keys()) | {object_id}
        for depth in range(_REVERB_STRIP_DEPTH):
            decay = _REVERB_DECAY ** depth
            next_f: Dict[str, float] = {}
            for dst, w in frontier.items():
                if dst not in graph.nodes:
                    continue
                callee_node = graph.nodes[dst]
                # Temporal discount: if callee appeared AFTER this node, it's a
                # downstream effect (we caused it), not an upstream cause. Discount
                # reverb from callees that appeared after us.
                if self_ts is not None and callee_node.earliest_timestamp is not None:
                    if callee_node.earliest_timestamp > self_ts:
                        # Callee appeared later → we likely caused it → discount reverb
                        temporal_discount = 0.2
                    elif callee_node.earliest_timestamp == self_ts:
                        temporal_discount = 0.7  # simultaneous → partial discount
                    else:
                        temporal_discount = 1.0  # callee appeared before us → full reverb
                else:
                    temporal_discount = 0.7  # unknown → partial
                reverb_sum += (w / n_direct_callees) * callee_node.anomaly_score * decay * temporal_discount
                for grandchild, gw in graph.adjacency.get(dst, {}).items():
                    if grandchild not in visited and grandchild in graph.nodes:
                        prop = w * min(1.0, gw) * _REVERB_DECAY
                        if prop > 1e-4:
                            next_f[grandchild] = max(next_f.get(grandchild, 0.0), prop)
                            visited.add(grandchild)
            frontier = next_f
            if not frontier:
                break

        # Clamp reverb_fraction to [0, 1]: how much of THIS node's anomaly is explained
        reverb_fraction = min(1.0, reverb_sum / self_anomaly)
        unexplained_residual = max(0.0, 1.0 - reverb_fraction)

        # Source signal: unexplained residual, weighted by how many callers depend on this node
        n_callers = sum(1 for _, ch in graph.adjacency.items() if object_id in ch)
        caller_weight = min(1.0, math.log1p(n_callers) / math.log1p(4))
        source_signal = unexplained_residual * (0.3 + 0.7 * caller_weight)

        return (
            float(math.tanh(source_signal)),
            float(reverb_fraction),
            float(unexplained_residual),
        )

    # ------------------------------------------------------------------
    # Reverse graph helper
    # ------------------------------------------------------------------

    def _build_reverse_graph(self, graph: ObjectGraph) -> ObjectGraph:
        """Return an ObjectGraph with all edge directions reversed.

        Enables computing caller-side exclusivity for callee-style root causes
        (DB faults, caches) that have no forward (callee) exclusive reach.
        """
        rev_adj: Dict[str, Dict[str, float]] = {}
        for src, children in graph.adjacency.items():
            for dst, w in children.items():
                if dst not in rev_adj:
                    rev_adj[dst] = {}
                rev_adj[dst][src] = max(rev_adj[dst].get(src, 0.0), w)
        return ObjectGraph(nodes=graph.nodes, adjacency=rev_adj)

    # ------------------------------------------------------------------
    # Exclusive explanation
    # ------------------------------------------------------------------

    def _global_coverage_union(
        self, graph: ObjectGraph, interventions: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, float]]:
        """For each (candidate, dst) pair, the best influence any *other* candidate exerts on dst.

        Returns a nested dict: best_other[candidate][dst] = max influence from any obj != candidate.
        This avoids the bug where a candidate's own influence masks its uniqueness.
        """
        # First pass: per-dst, collect all candidate influences
        dst_influences: Dict[str, Dict[str, float]] = {}  # dst -> {obj: influence}
        for obj, iv in interventions.items():
            for dst, w in iv.get("impacted", {}).items():
                if dst == obj:
                    continue
                if dst not in dst_influences:
                    dst_influences[dst] = {}
                dst_influences[dst][obj] = float(w)

        # Second pass: for each candidate, best_other[dst] = max influence from any obj != this candidate
        best_other_per_cand: Dict[str, Dict[str, float]] = {}
        for obj in interventions:
            best_other_per_cand[obj] = {}
            for dst, obj_influences in dst_influences.items():
                competitors = [w for cand, w in obj_influences.items() if cand != obj]
                if competitors:
                    best_other_per_cand[obj][dst] = max(competitors)
        return best_other_per_cand

    def _exclusive_explanation(
        self,
        graph: ObjectGraph,
        object_id: str,
        intervention: Dict[str, Any],
        all_covered: Dict[str, Dict[str, float]],
    ) -> float:
        """Anomaly mass explained exclusively by this candidate (not replaceable by another)."""
        best_other = all_covered.get(object_id, {})
        exclusive = 0.0
        for dst, my_w in intervention.get("impacted", {}).items():
            if dst == object_id or dst not in graph.nodes:
                continue
            competitor_w = best_other.get(dst, 0.0)
            uniqueness = max(0.0, float(my_w) - competitor_w)
            exclusive += uniqueness * graph.nodes[dst].anomaly_score
        return float(math.tanh(exclusive))

    # ------------------------------------------------------------------
    # Multi-lead consistency (TDOA-inspired)
    # ------------------------------------------------------------------

    def _multi_lead_consistency(self, graph: ObjectGraph, object_id: str) -> float:
        """Weighted fraction of BFS-reachable CALLERS that this node leads in time.

        In call-direction graphs (caller→callee), fault propagates upstream.
        A root-cause node should appear anomalous BEFORE the nodes that call it
        (its callers experience the fault with delay). We therefore follow edges
        *backward* (toward callers) and check whether this node is earlier.
        """
        node = graph.nodes[object_id]
        if node.earliest_timestamp is None:
            return 0.5

        # BFS on reverse edges: collect callers reachable within 3 hops
        reachable: Dict[str, float] = {}
        # Frontier: immediate callers with their edge weights
        frontier: Dict[str, float] = {}
        for parent, children in graph.adjacency.items():
            if object_id in children and parent in graph.nodes and parent != object_id:
                frontier[parent] = children[object_id]
        reachable.update(frontier)

        for _ in range(2):  # 3 total hops (1 already done above)
            nf: Dict[str, float] = {}
            for src, w in frontier.items():
                for grandparent, gchildren in graph.adjacency.items():
                    if src in gchildren and grandparent not in reachable and grandparent in graph.nodes and grandparent != object_id:
                        p = w * min(1.0, gchildren[src])
                        if p > 1e-3:
                            nf[grandparent] = max(nf.get(grandparent, 0.0), p)
                            reachable[grandparent] = max(reachable.get(grandparent, 0.0), p)
            frontier = nf
            if not frontier:
                break

        if not reachable:
            return self._temporal_source_score(graph, object_id)

        score = 0.0
        weight_sum = 0.0
        for dst, path_w in reachable.items():
            dst_ts = graph.nodes[dst].earliest_timestamp
            confidence = self._lead_confidence(node.earliest_timestamp, dst_ts)
            score += path_w * confidence
            weight_sum += path_w

        return float(score / max(EPS, weight_sum))

    def _lead_confidence(
        self, src_ts: Optional[float], dst_ts: Optional[float]
    ) -> float:
        if src_ts is None or dst_ts is None:
            return 0.5
        delta = dst_ts - src_ts  # positive = src led
        if delta == 0:
            return 0.5           # simultaneous: neutral evidence
        if delta > 0:
            # src before dst — confidence grows with gap, caps near 1
            return 0.5 + 0.5 * (1.0 - math.exp(-delta / _TIME_HALFLIFE))
        # dst before src — penalty, caps near 0
        return 0.5 * math.exp(delta / (_TIME_HALFLIFE * 0.3))

    # ------------------------------------------------------------------
    # Collapse recovery
    # ------------------------------------------------------------------

    def _collapse_recovery_score(
        self, graph: ObjectGraph, object_id: str, intervention: Dict[str, Any]
    ) -> float:
        """Fraction of downstream anomaly that would be deflated by removing this node."""
        impacted = intervention.get("impacted", {})
        recoverable = 0.0
        potential = 0.0
        for dst, w in impacted.items():
            if dst == object_id or dst not in graph.nodes:
                continue
            dst_anom = graph.nodes[dst].anomaly_score
            potential += dst_anom
            recoverable += float(w) * dst_anom
        if potential <= EPS:
            return 0.0
        return float(min(1.0, recoverable / potential))

    # ------------------------------------------------------------------
    # Existing helpers (kept / lightly refined)
    # ------------------------------------------------------------------

    def _resonance_mass(self, graph: ObjectGraph, object_id: str) -> float:
        """Anomaly mass in callees, temporally correlated with this node's failure.

        A node with many anomalous callees may be propagating failure downstream,
        or may be a hub that suffers because its callees fail. Either way, this
        measures forward-direction correlated anomaly.
        """
        node = graph.nodes[object_id]
        downstream = graph.adjacency.get(object_id, {})
        mass = 0.0
        for child, weight in downstream.items():
            if child not in graph.nodes:
                continue
            child_node = graph.nodes[child]
            tf = self._temporal_alignment(
                node.earliest_timestamp, child_node.earliest_timestamp
            )
            mass += weight * child_node.anomaly_score * tf
        return mass

    def _temporal_source_score(self, graph: ObjectGraph, object_id: str) -> float:
        node = graph.nodes[object_id]
        if node.earliest_timestamp is None:
            return 0.5
        ahead = 0.0
        total = 0.0
        for child, weight in graph.adjacency.get(object_id, {}).items():
            if child not in graph.nodes:
                continue
            total += weight
            child_ts = graph.nodes[child].earliest_timestamp
            if child_ts is None:
                ahead += 0.5 * weight
            elif node.earliest_timestamp <= child_ts:
                ahead += weight
        for parent, children in graph.adjacency.items():
            if object_id not in children or parent not in graph.nodes:
                continue
            total += children[object_id]
            parent_ts = graph.nodes[parent].earliest_timestamp
            if parent_ts is None:
                ahead += 0.5 * children[object_id]
            elif node.earliest_timestamp <= parent_ts:
                ahead += children[object_id]
        if total <= EPS:
            return 0.5
        return ahead / total

    def _collapse_gain(
        self, graph: ObjectGraph, object_id: str, intervention: Dict[str, Any]
    ) -> float:
        removed_mass = float(intervention.get("removed_mass", 0.0))
        downstream_removed = float(intervention.get("downstream_removed_mass", 0.0))
        impacted = intervention.get("impacted", {})
        distributed_collapse = 0.0
        for dst, weight in impacted.items():
            if dst == object_id or dst not in graph.nodes:
                continue
            distributed_collapse += weight * graph.nodes[dst].anomaly_score
        coverage = float(intervention.get("coverage", 0.0))
        exclusivity = float(intervention.get("exclusivity", 0.0))
        return (
            0.35 * removed_mass
            + 0.30 * downstream_removed
            + 0.20 * distributed_collapse
            + 0.10 * coverage
            + 0.05 * exclusivity
        )

    def _hotspot_bias(self, graph: ObjectGraph, object_id: str) -> float:
        """Penalty for nodes whose anomaly is not causally connected to the system.

        A node is biased if:
        - It is completely isolated (no edges) — its anomaly can't cause cascade
        - It has many anomalous callees but is a known hub (high out-degree)
        - It has high modality exposure suggesting it's a busy intermediate node
        """
        node = graph.nodes[object_id]
        degree_out = len(graph.adjacency.get(object_id, {}))
        degree_in = sum(
            1 for _, children in graph.adjacency.items() if object_id in children
        )
        modality_exposure = sum(
            1.0
            for v in (node.metric_score, node.log_score, node.trace_score, node.change_score)
            if v > 0.05
        )
        member_exposure = math.log1p(len(node.members))

        # Isolated node: no edges at all → anomaly doesn't propagate → likely noise
        # Use strong penalty so isolated nodes don't dominate rankings
        is_isolated = 1.0 if (degree_in == 0 and degree_out == 0) else 0.0

        # Hub with many anomalous callees: might be victim not cause
        anomalous_callees = sum(
            1 for dst in graph.adjacency.get(object_id, {})
            if dst in graph.nodes and graph.nodes[dst].anomaly_score > 0.3
        )
        callee_anomaly_ratio = anomalous_callees / max(1, degree_out) if degree_out > 0 else 0.0
        dense_hub_signal = math.tanh(0.3 * degree_out) * callee_anomaly_ratio

        return (
            0.45 * is_isolated             # dominant penalty for isolated nodes
            + 0.25 * dense_hub_signal
            + 0.15 * member_exposure
            + 0.10 * modality_exposure
            + 0.05 * math.tanh(0.2 * degree_out)
        )

    def _temporal_alignment(
        self, src_ts: Optional[float], dst_ts: Optional[float]
    ) -> float:
        if src_ts is None or dst_ts is None:
            return 0.7
        delta = dst_ts - src_ts
        if delta >= 0:
            return math.exp(-delta / _TIME_HALFLIFE)
        return 0.2 * math.exp(delta / _TIME_HALFLIFE)

    # ------------------------------------------------------------------
    # Field chain builder
    # ------------------------------------------------------------------

    def _build_field_chain(self, graph: ObjectGraph, object_id: str) -> List[str]:
        """Greedy BFS propagation chain anchored at object_id."""
        chain = [object_id]
        visited = {object_id}
        current = object_id
        for _ in range(4):
            children = graph.adjacency.get(current, {})
            if not children:
                break
            ranked = sorted(
                (
                    (child, w * graph.nodes[child].anomaly_score)
                    for child, w in children.items()
                    if child in graph.nodes and child not in visited
                ),
                key=lambda kv: kv[1],
                reverse=True,
            )
            if not ranked:
                break
            nxt, _ = ranked[0]
            chain.append(nxt)
            visited.add(nxt)
            current = nxt
        return chain

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    def _normalize(
        self, scores: Dict[str, NoiseFieldScore]
    ) -> Dict[str, NoiseFieldScore]:
        fields = [
            "local_noise",
            "resonance_mass",
            "source_signal",
            "reverb_mass",
            "unexplained_residual",
            "collapse_gain",
            "collapse_recovery_score",
            "exclusive_explanation",
            "temporal_source_score",
            "multi_lead_consistency",
            "hotspot_bias",
            "root_source_score",
        ]
        nv: Dict[str, Dict[str, float]] = {f: {} for f in fields}
        for f in fields:
            vals = [getattr(item, f) for item in scores.values()]
            lo, hi = min(vals), max(vals)
            for obj, item in scores.items():
                v = getattr(item, f)
                if math.isclose(lo, hi):
                    nv[f][obj] = 0.5
                else:
                    nv[f][obj] = (v - lo) / max(EPS, hi - lo)
        output: Dict[str, NoiseFieldScore] = {}
        for obj, item in scores.items():
            output[obj] = NoiseFieldScore(
                object_id=obj,
                local_noise=nv["local_noise"][obj],
                resonance_mass=nv["resonance_mass"][obj],
                source_signal=nv["source_signal"][obj],
                reverb_mass=nv["reverb_mass"][obj],
                unexplained_residual=nv["unexplained_residual"][obj],
                collapse_gain=nv["collapse_gain"][obj],
                collapse_recovery_score=nv["collapse_recovery_score"][obj],
                exclusive_explanation=nv["exclusive_explanation"][obj],
                temporal_source_score=nv["temporal_source_score"][obj],
                multi_lead_consistency=nv["multi_lead_consistency"][obj],
                hotspot_bias=nv["hotspot_bias"][obj],
                root_source_score=nv["root_source_score"][obj],
                influenced_objects=item.influenced_objects,
                noise_field_chain=item.noise_field_chain,
            )
        return output
