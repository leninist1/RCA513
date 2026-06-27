"""Cross-candidate graph features computed per-candidate via onset + trace graph analysis.

These capture relative information (onset ordering, trace propagation structure)
that the self-attention Set Transformer would need 1000s of samples to learn.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from refute_b_v2.trace_propagation import propagation_roles


def add_cross_candidate_features(
    features: np.ndarray,
    candidate_space: list[dict[str, Any]],
    trace_summary: dict[str, Any] | None,
    onset_ranks: np.ndarray,
) -> np.ndarray:
    """Augment per-candidate feature matrix with cross-candidate graph features.

    Args:
        features: (N, 33) existing feature matrix
        candidate_space: list of {component, reason_bucket, ...}
        trace_summary: trace summary dict
        onset_ranks: (N,) onset rank per candidate

    Returns:
        (N, 39) augmented feature matrix with 6 extra cross-candidate features.
    """
    N = features.shape[0]
    if N == 0:
        return features

    # 1. onset_rank_normalized (0..1)
    max_rank = max(onset_ranks.max(), 1.0)
    onset_norm = onset_ranks.astype(np.float32) / max_rank

    # 2. Build trace adjacency for cross-candidate analysis
    adj_down = {}  # comp -> set of downstream components
    adj_up = {}    # comp -> set of upstream components
    if trace_summary:
        for edge in trace_summary.get("edge_stats", []):
            src, dst = str(edge.get("src")), str(edge.get("dst"))
            adj_down.setdefault(src, set()).add(dst)
            adj_up.setdefault(dst, set()).add(src)

    # 3. Compute component → onset rank mapping
    comp_rank = {}
    for i in range(N):
        comp = str(candidate_space[i].get("component", ""))
        if comp not in comp_rank or onset_ranks[i] < comp_rank[comp]:
            comp_rank[comp] = int(onset_ranks[i])

    # 4. Compute per-candidate cross-candidate features
    upstream_count = np.zeros(N, dtype=np.float32)
    downstream_count = np.zeros(N, dtype=np.float32)
    centrality_norm = np.zeros(N, dtype=np.float32)
    onset_consistency = np.zeros(N, dtype=np.float32)
    rank_advantage = np.zeros(N, dtype=np.float32)
    trace_cascade_score = np.zeros(N, dtype=np.float32)

    # Build sets of anomalous components
    anomalous_comps = set(comp_rank.keys())
    # Max centrality for normalization
    max_deg = max((len(adj_down.get(c, set())) + len(adj_up.get(c, set()))) for c in anomalous_comps) or 1

    for i in range(N):
        comp = str(candidate_space[i].get("component", ""))
        rank = int(onset_ranks[i])

        # Upstream anomalous count
        up = adj_up.get(comp, set()) & anomalous_comps
        upstream_count[i] = math.log1p(len(up))

        # Downstream anomalous count
        down = adj_down.get(comp, set()) & anomalous_comps
        downstream_count[i] = math.log1p(len(down))

        # Centrality
        deg = len(up) + len(down)
        centrality_norm[i] = deg / max(max_deg, 1)

        # Onset consistency: upstream components should have earlier onset
        up_ranks = [comp_rank.get(c, 999) for c in up]
        if up_ranks:
            # Late onset + upstream neighbors that are anomalous → inconsistent
            # (real upstream source should have earlier onset than neighbors)
            neighbor_rank = min(up_ranks)
            onset_consistency[i] = 1.0 if rank <= neighbor_rank else 0.3
        else:
            onset_consistency[i] = 0.5

        # Rank advantage: how many other candidates have higher rank (later onset)
        rank_advantage[i] = (max_rank - rank) / max(max_rank, 1)

        # Trace cascade score: BFS depth from this component along anomalous edges
        visited = set()
        queue = [comp]
        depth = 0
        while queue and depth < 4:
            next_q = []
            for node in queue:
                if node in visited:
                    continue
                visited.add(node)
                for nb in adj_down.get(node, set()):
                    if nb in anomalous_comps and nb not in visited:
                        next_q.append(nb)
            if next_q:
                depth += 1
            queue = next_q
        trace_cascade_score[i] = depth / 4.0

    # Stack extra features
    extra = np.stack([
        onset_norm,
        upstream_count,
        downstream_count,
        centrality_norm,
        onset_consistency,
        rank_advantage,
        trace_cascade_score,
    ], axis=-1).astype(np.float32)

    return np.concatenate([features, extra], axis=-1)
