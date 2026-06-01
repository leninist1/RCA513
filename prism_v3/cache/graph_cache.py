"""L2 ObjectGraph cache serialization."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd

from .key import anchor_id, cache_config_hash, query_id as make_query_id, safe_segment, stable_json, telemetry_sha256
from .manifest import CacheManifest
from .store import CacheMiss, CacheStore, CacheValidationError


GRAPH_BUILDER_VERSION = "object_graph.builder.v1"
TRACE_GRAPH_CONFIG_VERSION = "trace_graph.config.v1"
METRIC_GRAPH_CONFIG_VERSION = "metric_graph.config.v1"
CANONICALIZATION_VERSION = "object_canonicalization.v1"


class ObjectGraphCache:
    def __init__(self, store: CacheStore) -> None:
        self.store = store

    def ids(self, query: Any, anchor_timestamp: float, anchor_source: str) -> Tuple[str, str, str]:
        qid = make_query_id(
            getattr(query, "system", ""),
            getattr(query, "sub_system", ""),
            getattr(query, "telemetry_date", ""),
            getattr(query, "task_index", ""),
            getattr(query, "query_index", None),
        )
        aid = anchor_id(anchor_timestamp, anchor_source, {"graph_builder": GRAPH_BUILDER_VERSION})
        cfg_hash = cache_config_hash(
            {
                "graph_builder": GRAPH_BUILDER_VERSION,
                "trace_graph_config": TRACE_GRAPH_CONFIG_VERSION,
                "metric_graph_config": METRIC_GRAPH_CONFIG_VERSION,
                "canonicalization": CANONICALIZATION_VERSION,
            }
        )
        return qid, aid, cfg_hash

    def cache_dir(self, query: Any, qid: str, aid: str, cfg_hash: str) -> Path:
        return (
            self.store.layer_dir("graphs")
            / safe_segment(getattr(query, "system", ""))
            / safe_segment(getattr(query, "sub_system", "") or "default")
            / safe_segment(qid)
            / aid
            / cfg_hash
        )

    def load(
        self,
        *,
        telemetry: Any,
        query: Any,
        anchor_timestamp: float,
        anchor_source: str,
    ) -> Tuple[Any, Dict[str, Any], Dict[str, Any]]:
        tsha = telemetry_sha256(telemetry)
        qid, aid, cfg_hash = self.ids(query, anchor_timestamp, anchor_source)
        cache_dir = self.cache_dir(query, qid, aid, cfg_hash)
        self.store.read_manifest(
            cache_dir,
            cache_type="object_graph",
            query_id=qid,
            anchor_source=anchor_source,
            telemetry_sha256=tsha,
            cache_config_hash=cfg_hash,
        )
        graph = self._read_graph(cache_dir)
        graph_debug = dict(self.store.read_json(cache_dir / "graph_debug.json"))
        debug = {
            "cache_layer": "object_graph",
            "cache_hit": True,
            "cache_dir": str(cache_dir),
            "node_count": len(getattr(graph, "nodes", {}) or {}),
            "edge_count": sum(len(v) for v in (getattr(graph, "adjacency", {}) or {}).values()),
        }
        return graph, graph_debug, debug

    def write(
        self,
        *,
        telemetry: Any,
        query: Any,
        anchor_timestamp: float,
        anchor_source: str,
        graph: Any,
        graph_debug: Dict[str, Any],
    ) -> Tuple[Path, Dict[str, Any]]:
        tsha = telemetry_sha256(telemetry)
        qid, aid, cfg_hash = self.ids(query, anchor_timestamp, anchor_source)
        cache_dir = self.cache_dir(query, qid, aid, cfg_hash)
        cache_dir.mkdir(parents=True, exist_ok=True)
        self._write_graph(cache_dir, graph)
        self.store.write_json(cache_dir / "graph_debug.json", graph_debug or {})
        self.store.write_manifest(
            cache_dir,
            CacheManifest(
                cache_type="object_graph",
                query_id=qid,
                anchor_timestamp=float(anchor_timestamp),
                anchor_source=anchor_source,
                telemetry_sha256=tsha,
                cache_config_hash=cfg_hash,
                metadata={
                    "anchor_id": aid,
                    "graph_builder_version": GRAPH_BUILDER_VERSION,
                    "trace_graph_config_version": TRACE_GRAPH_CONFIG_VERSION,
                    "metric_graph_config_version": METRIC_GRAPH_CONFIG_VERSION,
                    "canonicalization_version": CANONICALIZATION_VERSION,
                    "node_count": len(getattr(graph, "nodes", {}) or {}),
                    "edge_count": sum(len(v) for v in (getattr(graph, "adjacency", {}) or {}).values()),
                },
            ),
        )
        return cache_dir, {
            "cache_layer": "object_graph",
            "cache_hit": False,
            "cache_write": True,
            "cache_dir": str(cache_dir),
        }

    def load_or_build(
        self,
        *,
        telemetry: Any,
        query: Any,
        anchor_timestamp: float,
        anchor_source: str,
        builder: Any,
    ) -> Tuple[Any, Dict[str, Any], Dict[str, Any]]:
        try:
            return self.load(
                telemetry=telemetry,
                query=query,
                anchor_timestamp=anchor_timestamp,
                anchor_source=anchor_source,
            )
        except CacheMiss:
            pass
        except CacheValidationError:
            if self.store.strict:
                raise
        graph, graph_debug = builder()
        _cache_dir, debug = self.write(
            telemetry=telemetry,
            query=query,
            anchor_timestamp=anchor_timestamp,
            anchor_source=anchor_source,
            graph=graph,
            graph_debug=graph_debug,
        )
        return graph, graph_debug, debug

    def _write_graph(self, cache_dir: Path, graph: Any) -> None:
        node_rows = []
        for object_id, node in (getattr(graph, "nodes", {}) or {}).items():
            node_rows.append(
                {
                    "object_id": str(object_id),
                    "members": stable_json(list(getattr(node, "members", []) or [])),
                    "representative": str(getattr(node, "representative", "")),
                    "anomaly_score": float(getattr(node, "anomaly_score", 0.0) or 0.0),
                    "earliest_timestamp": getattr(node, "earliest_timestamp", None),
                    "metric_score": float(getattr(node, "metric_score", 0.0) or 0.0),
                    "log_score": float(getattr(node, "log_score", 0.0) or 0.0),
                    "trace_score": float(getattr(node, "trace_score", 0.0) or 0.0),
                    "change_score": float(getattr(node, "change_score", 0.0) or 0.0),
                    "reason_votes": stable_json(dict(getattr(node, "reason_votes", {}) or {})),
                    "evidence": stable_json([_to_plain(item) for item in (getattr(node, "evidence", []) or [])]),
                }
            )
        edge_rows = []
        for src, children in (getattr(graph, "adjacency", {}) or {}).items():
            for dst, weight in (children or {}).items():
                edge_rows.append({"src": str(src), "dst": str(dst), "weight": float(weight)})
        pd.DataFrame(
            node_rows,
            columns=[
                "object_id", "members", "representative", "anomaly_score",
                "earliest_timestamp", "metric_score", "log_score", "trace_score",
                "change_score", "reason_votes", "evidence",
            ],
        ).to_parquet(cache_dir / "graph_nodes.parquet", index=False)
        pd.DataFrame(edge_rows, columns=["src", "dst", "weight"]).to_parquet(
            cache_dir / "graph_edges.parquet", index=False
        )

    def _read_graph(self, cache_dir: Path) -> Any:
        from ..mace.graph import EvidenceRecord, ObjectGraph, ObjectNode
        import json

        nodes: Dict[str, Any] = {}
        node_df = pd.read_parquet(cache_dir / "graph_nodes.parquet")
        for row in node_df.to_dict(orient="records"):
            evidence_payload = json.loads(row.get("evidence") or "[]")
            evidence = [
                EvidenceRecord(
                    kind=str(item.get("kind", "")),
                    source=str(item.get("source", "")),
                    content=str(item.get("content", "")),
                    confidence=float(item.get("confidence", 0.0) or 0.0),
                    timestamp=item.get("timestamp"),
                )
                for item in evidence_payload
                if isinstance(item, dict)
            ]
            object_id = str(row.get("object_id", ""))
            nodes[object_id] = ObjectNode(
                object_id=object_id,
                members=list(json.loads(row.get("members") or "[]")),
                representative=str(row.get("representative", "")),
                anomaly_score=float(row.get("anomaly_score", 0.0) or 0.0),
                earliest_timestamp=row.get("earliest_timestamp"),
                metric_score=float(row.get("metric_score", 0.0) or 0.0),
                log_score=float(row.get("log_score", 0.0) or 0.0),
                trace_score=float(row.get("trace_score", 0.0) or 0.0),
                change_score=float(row.get("change_score", 0.0) or 0.0),
                reason_votes=dict(json.loads(row.get("reason_votes") or "{}")),
                evidence=evidence,
            )
        adjacency: Dict[str, Dict[str, float]] = {}
        edge_path = cache_dir / "graph_edges.parquet"
        if edge_path.exists():
            edge_df = pd.read_parquet(edge_path)
            for row in edge_df.to_dict(orient="records"):
                adjacency.setdefault(str(row.get("src", "")), {})[str(row.get("dst", ""))] = float(row.get("weight", 0.0) or 0.0)
        return ObjectGraph(nodes=nodes, adjacency=adjacency)


def _to_plain(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return {str(k): _to_plain(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_plain(item) for item in value]
    return value
