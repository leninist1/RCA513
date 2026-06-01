"""Sector-conditioned source/noise decomposition for Noise Lab P2.1."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, List, Tuple

import numpy as np

from ..mace.graph import ObjectGraph

EPS = 1e-6


@dataclass
class SubspaceScore:
    object_id: str
    local_residual_source_energy: float
    sector_source_ratio: float
    sector_reverb_ratio: float
    residual_distinctiveness: float
    local_common_mode_alignment: float
    replaceability: float
    sector_size: float


class SourceNoiseSubspaceDecomposer:
    """Estimate direct-source residuals inside each candidate's beamformed sector."""

    def __init__(self, max_sector_targets: int = 6):
        self.max_sector_targets = max_sector_targets

    def score(
        self,
        graph: ObjectGraph,
        beam_scores: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        if not graph.nodes:
            return {}
        raw: Dict[str, SubspaceScore] = {}
        for object_id in graph.nodes:
            sector = self._sector(graph, object_id, beam_scores)
            source_projection, common_mode, residual = self._sector_decompose(graph, object_id, sector)
            total_energy = max(EPS, float(np.dot(source_projection, source_projection)))
            common_energy = float(np.dot(common_mode, common_mode))
            residual_energy = float(np.dot(residual, residual))
            sector_reverb_ratio = min(1.0, common_energy / total_energy)
            sector_source_ratio = min(1.0, residual_energy / total_energy)
            local_residual_source_energy = sector_source_ratio * self._sector_weight(sector)
            residual_distinctiveness = self._residual_distinctiveness(residual)
            common_mode_alignment = self._common_mode_alignment(source_projection, common_mode)
            replaceability = self._replaceability(graph, object_id, sector, residual)
            raw[object_id] = SubspaceScore(
                object_id=object_id,
                local_residual_source_energy=local_residual_source_energy,
                sector_source_ratio=sector_source_ratio,
                sector_reverb_ratio=sector_reverb_ratio,
                residual_distinctiveness=residual_distinctiveness,
                local_common_mode_alignment=common_mode_alignment,
                replaceability=replaceability,
                sector_size=min(1.0, len(sector) / max(1.0, self.max_sector_targets + 1)),
            )
        normalized = self._normalize(raw)
        return {
            object_id: {
                "local_residual_source_energy": round(item.local_residual_source_energy, 6),
                "sector_source_ratio": round(item.sector_source_ratio, 6),
                "sector_reverb_ratio": round(item.sector_reverb_ratio, 6),
                "residual_distinctiveness": round(item.residual_distinctiveness, 6),
                "local_common_mode_alignment": round(item.local_common_mode_alignment, 6),
                "replaceability": round(item.replaceability, 6),
                "sector_size": round(item.sector_size, 6),
            }
            for object_id, item in normalized.items()
        }

    def _sector(
        self,
        graph: ObjectGraph,
        object_id: str,
        beam_scores: Dict[str, Dict[str, Any]],
    ) -> List[Tuple[str, float]]:
        payload = beam_scores.get(object_id, {})
        rows = [(object_id, 1.0)]
        for entry in payload.get("beam_targets", [])[: self.max_sector_targets]:
            target = entry.get("object_id")
            score = float(entry.get("score", 0.0))
            if target in graph.nodes and score > 0.0:
                rows.append((target, score))
        return rows

    def _sector_decompose(
        self,
        graph: ObjectGraph,
        object_id: str,
        sector: List[Tuple[str, float]],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        source_projection = np.array(self._feature_vector(graph, object_id), dtype=float)
        if len(sector) <= 1:
            common_mode = np.zeros_like(source_projection)
            return source_projection, common_mode, source_projection.copy()
        vectors = []
        weights = []
        for target, weight in sector:
            if target == object_id:
                continue
            vectors.append(np.array(self._feature_vector(graph, target), dtype=float))
            weights.append(weight)
        if not vectors:
            common_mode = np.zeros_like(source_projection)
            return source_projection, common_mode, source_projection.copy()
        weight_arr = np.array(weights, dtype=float)
        weight_arr = weight_arr / max(EPS, float(weight_arr.sum()))
        common_mode = np.average(np.stack(vectors, axis=0), axis=0, weights=weight_arr)
        residual = source_projection - common_mode
        return source_projection, common_mode, residual

    def _feature_vector(self, graph: ObjectGraph, object_id: str) -> List[float]:
        node = graph.nodes[object_id]
        incoming = graph.incoming_mass(object_id)
        outgoing = graph.topological_mass(object_id)
        earliest = node.earliest_timestamp or 0.0
        reason_votes = list(node.reason_votes.values())[:4]
        if len(reason_votes) < 4:
            reason_votes = reason_votes + [0.0] * (4 - len(reason_votes))
        return [
            node.anomaly_score,
            node.metric_score,
            node.log_score,
            node.trace_score,
            node.change_score,
            incoming,
            outgoing,
            math.log1p(len(node.members)),
            earliest,
            *reason_votes,
        ]

    def _sector_weight(self, sector: List[Tuple[str, float]]) -> float:
        if not sector:
            return 0.0
        total = sum(weight for _, weight in sector[1:])
        return min(1.0, total / max(1.0, 0.8 * self.max_sector_targets))

    def _residual_distinctiveness(self, residual: np.ndarray) -> float:
        l1 = float(np.sum(np.abs(residual)))
        l2 = float(np.linalg.norm(residual))
        if l1 <= EPS:
            return 0.0
        return l2 / l1

    def _common_mode_alignment(self, row: np.ndarray, common_mode: np.ndarray) -> float:
        denom = float(np.linalg.norm(row) * np.linalg.norm(common_mode))
        if denom <= EPS:
            return 0.0
        return abs(float(np.dot(row, common_mode)) / denom)

    def _replaceability(
        self,
        graph: ObjectGraph,
        object_id: str,
        sector: List[Tuple[str, float]],
        residual: np.ndarray,
    ) -> float:
        if len(sector) <= 1:
            return 0.0
        source_norm = float(np.linalg.norm(residual))
        if source_norm <= EPS:
            return 1.0
        similarities = []
        for target, _ in sector:
            if target == object_id:
                continue
            target_vec = np.array(self._feature_vector(graph, target), dtype=float)
            denom = source_norm * float(np.linalg.norm(target_vec))
            if denom <= EPS:
                continue
            similarities.append(abs(float(np.dot(residual, target_vec)) / denom))
        if not similarities:
            return 0.0
        similarities.sort(reverse=True)
        top = similarities[:2]
        return sum(top) / len(top)

    def _normalize(self, raw: Dict[str, SubspaceScore]) -> Dict[str, SubspaceScore]:
        fields = [
            "local_residual_source_energy",
            "sector_source_ratio",
            "sector_reverb_ratio",
            "residual_distinctiveness",
            "local_common_mode_alignment",
            "replaceability",
            "sector_size",
        ]
        normalized: Dict[str, Dict[str, float]] = {field: {} for field in fields}
        for field in fields:
            values = [getattr(item, field) for item in raw.values()]
            lo = min(values)
            hi = max(values)
            for object_id, item in raw.items():
                value = getattr(item, field)
                if math.isclose(lo, hi):
                    normalized[field][object_id] = 0.5
                else:
                    normalized[field][object_id] = (value - lo) / max(EPS, hi - lo)
        output: Dict[str, SubspaceScore] = {}
        for object_id, item in raw.items():
            output[object_id] = SubspaceScore(
                object_id=object_id,
                local_residual_source_energy=normalized["local_residual_source_energy"][object_id],
                sector_source_ratio=normalized["sector_source_ratio"][object_id],
                sector_reverb_ratio=normalized["sector_reverb_ratio"][object_id],
                residual_distinctiveness=normalized["residual_distinctiveness"][object_id],
                local_common_mode_alignment=normalized["local_common_mode_alignment"][object_id],
                replaceability=normalized["replaceability"][object_id],
                sector_size=normalized["sector_size"][object_id],
            )
        return output
