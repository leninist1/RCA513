"""Trace propagation reasoning over trace summaries."""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass


@dataclass(frozen=True)
class TraceRole:
    service: str
    role: str
    edge_count: int
    earliest_timestamp: int | None
    details: dict

    def to_dict(self) -> dict:
        return {
            "service": self.service,
            "role": self.role,
            "edge_count": self.edge_count,
            "earliest_timestamp": self.earliest_timestamp,
            "details": self.details,
        }


def propagation_roles(summary: dict) -> dict[str, TraceRole]:
    edges = list(summary.get("events", {}).get("slow_edges", [])) + list(summary.get("events", {}).get("dropped_edges", []))
    if not edges:
        return {}
    incoming = defaultdict(int)
    outgoing = defaultdict(int)
    first_ts: dict[str, int] = {}
    details = defaultdict(lambda: {"slow_edges": 0, "dropped_edges": 0})
    for edge in edges:
        src, dst = str(edge.get("src")), str(edge.get("dst"))
        outgoing[src] += 1
        incoming[dst] += 1
        event = edge.get("event") or ("dropped_edges" if edge.get("count_drop_ratio", 0.0) else "slow_edges")
        details[src][event] = details[src].get(event, 0) + 1
        details[dst][event] = details[dst].get(event, 0) + 1
        ts = edge.get("first_timestamp")
        if ts is not None:
            ts = int(ts)
            first_ts[src] = min(first_ts.get(src, ts), ts)
            first_ts[dst] = min(first_ts.get(dst, ts), ts)
    roles = {}
    for service in sorted(set(incoming) | set(outgoing)):
        if outgoing[service] and not incoming[service]:
            role = "upstream_source"
        elif incoming[service] and outgoing[service]:
            role = "middle_propagator"
        elif incoming[service]:
            role = "downstream_sink"
        else:
            role = "isolated"
        roles[service] = TraceRole(service, role, incoming[service] + outgoing[service], first_ts.get(service), dict(details[service]))
    return roles


def propagation_path(summary: dict, start: str, max_depth: int = 4) -> list[str]:
    graph = defaultdict(list)
    for edge in summary.get("edge_stats", []):
        if edge.get("slow_ratio", 0.0) > 0 or edge.get("count_drop_ratio", 0.0) > 0:
            graph[str(edge.get("src"))].append(str(edge.get("dst")))
    seen = {str(start)}
    queue = deque([(str(start), [str(start)])])
    best = [str(start)]
    while queue:
        node, path = queue.popleft()
        if len(path) > len(best):
            best = path
        if len(path) >= max_depth:
            continue
        for nxt in sorted(graph.get(node, [])):
            if nxt in seen:
                continue
            seen.add(nxt)
            queue.append((nxt, path + [nxt]))
    return best


def trace_evidence_for_service(summary: dict, service: str) -> dict:
    roles = propagation_roles(summary)
    role = roles.get(str(service))
    return {
        "role": role.to_dict() if role else None,
        "path": propagation_path(summary, str(service)) if role else [],
        "first_anomalous_service": summary.get("events", {}).get("first_anomalous_service"),
    }
