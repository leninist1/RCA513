"""
hetero_topology.py -- dataset-portable topology schema.

The graph supports arbitrary entity/edge types. Bank's existing
node_container_graph is represented as a two-layer projection, while Market or
Telecom can add service/node/network-element layers without changing L2/L3.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional


@dataclass(frozen=True)
class Entity:
    entity_id: str
    entity_type: str
    attrs: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"id": self.entity_id, "type": self.entity_type, "attrs": self.attrs}

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "Entity":
        return cls(str(data["id"]), str(data["type"]), dict(data.get("attrs", {})))


@dataclass(frozen=True)
class Edge:
    src: str
    dst: str
    edge_type: str
    attrs: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"src": self.src, "dst": self.dst, "type": self.edge_type, "attrs": self.attrs}

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "Edge":
        return cls(str(data["src"]), str(data["dst"]), str(data["type"]), dict(data.get("attrs", {})))


class HeteroTopology:
    def __init__(self, entities: Iterable[Entity] = (), edges: Iterable[Edge] = (), dataset: str = "Bank"):
        self.dataset = dataset
        self.entities: Dict[str, Entity] = {e.entity_id: e for e in entities}
        self.edges: List[Edge] = list(edges)

    def add_entity(self, entity_id: str, entity_type: str, **attrs) -> None:
        self.entities[str(entity_id)] = Entity(str(entity_id), str(entity_type), attrs)

    def add_edge(self, src: str, dst: str, edge_type: str, **attrs) -> None:
        self.edges.append(Edge(str(src), str(dst), str(edge_type), attrs))

    def neighbors(self, entity_id: str, edge_type: Optional[str] = None, direction: str = "both") -> List[str]:
        out = []
        for edge in self.edges:
            if edge_type and edge.edge_type != edge_type:
                continue
            if direction in {"both", "out"} and edge.src == entity_id:
                out.append(edge.dst)
            if direction in {"both", "in"} and edge.dst == entity_id:
                out.append(edge.src)
        return list(dict.fromkeys(out))

    def entities_by_type(self, entity_type: str) -> List[str]:
        return [eid for eid, ent in self.entities.items() if ent.entity_type == entity_type]

    def to_dict(self) -> dict:
        return {
            "version": 1,
            "dataset": self.dataset,
            "entities": {eid: ent.to_dict() for eid, ent in sorted(self.entities.items())},
            "edges": [edge.to_dict() for edge in self.edges],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "HeteroTopology":
        entities = [Entity.from_dict(v) for v in dict(data.get("entities", {})).values()]
        edges = [Edge.from_dict(v) for v in list(data.get("edges", []))]
        return cls(entities, edges, dataset=str(data.get("dataset", "unknown")))


def from_bank_node_container_graph(graph: Mapping[str, object]) -> HeteroTopology:
    topo = HeteroTopology(dataset=str(graph.get("dataset", "Bank")))
    for node_id, node in dict(graph.get("nodes", {})).items():
        topo.add_entity(node_id, "node", n_hosted=node.get("n_hosted"), node_kpis=node.get("node_kpis", []))
    for cmdb_id, container in dict(graph.get("containers", {})).items():
        topo.add_entity(cmdb_id, "pod", container_kpis=container.get("container_kpis", []), node_kpis=container.get("node_kpis", []))
        node_proxy = container.get("node_proxy") or cmdb_id
        if node_proxy not in topo.entities:
            topo.add_entity(node_proxy, "node")
        topo.add_edge(node_proxy, cmdb_id, "hosts")
    return topo
