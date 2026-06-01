"""Entity embeddings: learned representations for services, containers, nodes.

Each entity gets a learned embedding vector that captures:
  - Entity type (pod/service/node/container/middleware)
  - Metric profile patterns (quantile summaries)
  - Relational position in the call graph
  - Historical co-occurrence patterns (trained via contrastive learning)

The embeddings unify entities across different systems, enabling
knowledge transfer and better generalization.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import math

import numpy as np


class EntityEmbedding:
    """Trainable entity embedding table with optional metric-profile encoder.

    Supports two pretrained formats:
      - Legacy tokens dict: {"tokens": {name: vec, ...}}
      - New structured dict:   {"vectors": {name: vec, ...},
                                 "affinities": {name: {subcat: score, ...}},
                                 "dim": 64}
      - Flat dict (compat):     {name: vec, ...}

    When no training data is available, falls back to deterministic
    positional encoding so the pipeline still works.
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        token_dim: int = 64,
        type_dim: int = 16,
        metric_dim: int = 32,
        position_dim: int = 16,
        load_pretrained: Optional[str] = None,
    ):
        self.embedding_dim = embedding_dim
        self.token_dim = token_dim
        self.type_dim = type_dim
        self.metric_dim = metric_dim
        self.position_dim = position_dim
        self._token_embeddings: Dict[str, np.ndarray] = {}
        self._entity_affinities: Dict[str, Dict[str, float]] = {}
        self._is_trained = False
        if load_pretrained:
            self._load(load_pretrained)

    def encode_entity_name(self, name: str) -> np.ndarray:
        if name in self._token_embeddings:
            return self._token_embeddings[name].copy()
        return self._positional_encode(name)

    def _positional_encode(self, name: str) -> np.ndarray:
        vec = np.zeros(self.token_dim, dtype=float)
        chars = list(name.lower())
        for i, ch in enumerate(chars):
            val = ord(ch) / 128.0
            for d in range(self.token_dim):
                if d % 2 == 0:
                    vec[d] += val * math.sin(i / (10000 ** (2 * d / self.token_dim)))
                else:
                    vec[d] += val * math.cos(i / (10000 ** (2 * d / self.token_dim)))
        norm = float(np.sqrt(np.sum(vec * vec)))
        if norm > 1e-9:
            vec /= norm
        return vec

    def encode_entity_type(self, entity_type: str) -> np.ndarray:
        type_map = {
            "pod": 0,
            "container": 0,
            "service": 1,
            "svc": 1,
            "node": 2,
            "host": 2,
            "middleware": 3,
        }
        idx = type_map.get(entity_type.lower(), 0)
        vec = np.zeros(self.type_dim, dtype=float)
        if idx < self.type_dim:
            vec[idx] = 1.0
            spread = self.type_dim // 4
            for offset in range(1, spread + 1):
                left = (idx - offset) % self.type_dim
                right = (idx + offset) % self.type_dim
                weight = 1.0 / (offset + 1)
                if left != idx:
                    vec[left] = weight
                if right != idx:
                    vec[right] = weight
        return vec

    def encode_metric_profile(
        self,
        metric_detail: Dict[str, float],
        metric_map: Optional[Dict[str, int]] = None,
    ) -> np.ndarray:
        if not metric_map:
            sorted_metrics = sorted(
                [(name, score) for name, score in metric_detail.items() if score > 0],
                key=lambda x: x[1],
                reverse=True,
            )[:10]
            vec = np.zeros(self.metric_dim, dtype=float)
            step = self.metric_dim / max(1, len(sorted_metrics))
            for i, (name, score) in enumerate(sorted_metrics):
                pos = int(i * step)
                if pos < self.metric_dim:
                    vec[pos] = score
        else:
            vec = np.zeros(self.metric_dim, dtype=float)
            for name, score in metric_detail.items():
                if name in metric_map and score > 0:
                    vec[metric_map[name] % self.metric_dim] = score
        norm = float(np.sqrt(np.sum(vec * vec)))
        if norm > 1e-9:
            vec /= norm
        return vec

    def encode_graph_position(
        self,
        in_degree: float,
        out_degree: float,
        page_rank: float = 0.0,
    ) -> np.ndarray:
        vec = np.zeros(self.position_dim, dtype=float)
        vec[0] = min(1.0, in_degree / 10.0)
        vec[1] = min(1.0, out_degree / 10.0)
        vec[2] = min(1.0, page_rank)
        return vec

    def embed_entity(
        self,
        name: str,
        entity_type: str,
        metric_detail: Optional[Dict[str, float]] = None,
        in_degree: float = 0.0,
        out_degree: float = 0.0,
        page_rank: float = 0.0,
        metric_map: Optional[Dict[str, int]] = None,
    ) -> np.ndarray:
        e_token = self.encode_entity_name(name)
        e_type = self.encode_entity_type(entity_type)
        e_metric = self.encode_metric_profile(metric_detail or {}, metric_map)
        e_position = self.encode_graph_position(in_degree, out_degree, page_rank)

        cat = np.concatenate([e_token, e_type, e_metric, e_position])
        if len(cat) > self.embedding_dim:
            cat = cat[: self.embedding_dim]
        elif len(cat) < self.embedding_dim:
            padded = np.zeros(self.embedding_dim, dtype=float)
            padded[: len(cat)] = cat
            cat = padded
        norm = float(np.sqrt(np.sum(cat * cat)))
        if norm > 1e-9:
            cat /= norm
        return cat

    def similarity(self, emb_a: np.ndarray, emb_b: np.ndarray) -> float:
        return float(np.dot(emb_a, emb_b))

    def _load(self, path: str):
        import pickle as _pickle

        try:
            with open(path, "rb") as f:
                data = _pickle.load(f)
            if isinstance(data, dict):
                if "vectors" in data:
                    self._token_embeddings = data["vectors"]
                    self._entity_affinities = data.get("affinities", {})
                elif "tokens" in data:
                    self._token_embeddings = data["tokens"]
                    self._entity_affinities = {}
                else:
                    self._token_embeddings = data
                    self._entity_affinities = {}
                self._is_trained = bool(self._token_embeddings)
        except (FileNotFoundError, Exception):
            self._is_trained = False

    def get_affinity(self, entity: str) -> Dict[str, float]:
        return self._entity_affinities.get(entity, {}).copy()

    def save(self, path: str):
        import pickle as _pickle

        data = {
            "tokens": self._token_embeddings,
            "affinities": self._entity_affinities,
        }
        with open(path, "wb") as f:
            _pickle.dump(data, f)


class EntityEmbedder:
    """High-level interface that embeds all entities in a case."""

    def __init__(
        self,
        embedding: Optional[EntityEmbedding] = None,
        embedding_dim: int = 128,
    ):
        self.embedding = embedding or EntityEmbedding(embedding_dim=embedding_dim)

    def embed_all(
        self,
        entities: List[str],
        entity_types: Dict[str, str],
        metric_detail: Dict[str, Dict[str, float]],
        graph_degrees: Optional[Dict[str, Tuple[float, float]]] = None,
        page_ranks: Optional[Dict[str, float]] = None,
    ) -> Dict[str, np.ndarray]:
        results = {}
        for entity in entities:
            etype = entity_types.get(entity, "pod")
            metrics = metric_detail.get(entity, {})
            in_deg, out_deg = 0.0, 0.0
            pr = page_ranks.get(entity, 0.0) if page_ranks else 0.0
            if graph_degrees:
                in_deg, out_deg = graph_degrees.get(entity, (0.0, 0.0))
            results[entity] = self.embedding.embed_entity(
                name=entity,
                entity_type=etype,
                metric_detail=metrics,
                in_degree=in_deg,
                out_degree=out_deg,
                page_rank=pr,
            )
        return results

    def compute_pairwise_similarity(
        self,
        embeddings: Dict[str, np.ndarray],
    ) -> np.ndarray:
        entities = list(embeddings.keys())
        n = len(entities)
        sim = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(i, n):
                s = self.embedding.similarity(
                    embeddings[entities[i]], embeddings[entities[j]]
                )
                sim[i, j] = s
                sim[j, i] = s
        return sim
