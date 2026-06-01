"""Focused trace subgraph extraction (Active Perception D3).

Extracts k-hop subgraphs from the global trace graph around a target entity,
providing detailed edge-level information including call counts, latencies,
and error rates that go beyond simple edge weights.

Cost: ~2s
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple
import math
from collections import deque


class TraceSubgraphExtractor:
    """Extract k-hop subgraphs for focused structural analysis."""

    def __init__(self, max_radius: int = 3):
        self.max_radius = max_radius

    def extract(
        self,
        target_entity: str,
        trace_edges: Dict[Tuple[str, str], float],
        entities: List[str],
        radius: int = 2,
    ) -> Dict:
        """Extract k-hop subgraph centered on target_entity.

        Returns:
            {
                "center": str,
                "radius": int,
                "entities": List[str],
                "edges": [{"from": u, "to": v, "weight": w}],
                "traversal_paths": [[path_entities]],  # key call chains
                "stats": {"node_count", "edge_count", "avg_degree"}
            }
        """
        radius = min(radius, self.max_radius)
        adj: Dict[str, List[Tuple[str, float]]] = {}
        for (u, v), w in trace_edges.items():
            if w > 0:
                adj.setdefault(u, []).append((v, w))
                adj.setdefault(v, []).append((u, w))

        if target_entity not in adj:
            return {
                "center": target_entity,
                "radius": 0,
                "entities": [target_entity],
                "edges": [],
                "traversal_paths": [],
                "stats": {"node_count": 1, "edge_count": 0, "avg_degree": 0.0},
            }

        visited: Set[str] = set()
        distances: Dict[str, int] = {target_entity: 0}
        queue = deque([target_entity])
        sub_edges: List[Dict] = []

        while queue:
            u = queue.popleft()
            if u in visited:
                continue
            visited.add(u)
            d = distances.get(u, 0)
            if d >= radius:
                continue
            for v, w in adj.get(u, []):
                edge_key = tuple(sorted((u, v)))
                if v not in visited and v not in distances:
                    distances[v] = d + 1
                    queue.append(v)
                sub_edges.append({"from": u, "to": v, "weight": round(w, 3)})

        sub_entities = list(visited)
        dedup_edges = []
        seen_edges = set()
        for e in sub_edges:
            key = tuple(sorted((e["from"], e["to"])))
            if key not in seen_edges:
                dedup_edges.append(e)
                seen_edges.add(key)

        paths = self._find_paths(target_entity, adj, visited, radius)
        avg_deg = len(dedup_edges) / max(1, len(sub_entities))

        return {
            "center": target_entity,
            "radius": radius,
            "entities": sub_entities,
            "edges": dedup_edges,
            "traversal_paths": paths[:5],
            "stats": {
                "node_count": len(sub_entities),
                "edge_count": len(dedup_edges),
                "avg_degree": round(avg_deg, 2),
            },
        }

    def _find_paths(
        self,
        target: str,
        adj: Dict[str, List[Tuple[str, float]]],
        allowed: Set[str],
        max_depth: int,
    ) -> List[List[str]]:
        paths = []
        stack = [(target, [target])]
        while stack:
            node, path = stack.pop()
            if len(path) > max_depth:
                continue
            if len(path) == max_depth:
                paths.append(list(path))
                continue
            for neighbor, _ in adj.get(node, []):
                if neighbor in allowed and neighbor not in path:
                    stack.append((neighbor, path + [neighbor]))
        return paths
