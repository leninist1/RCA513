"""Adapters from NoiseLab outputs into structured EvidenceFrame objects."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..config import QueryCase, UnifiedTelemetry
from .evidence_frame import (
    CandidateFrame,
    EvidenceFrame,
    canonical_entity_name,
    rank_to_prior_mass,
    softmax,
)


_SCORES_CACHE: Dict[str, pd.DataFrame] = {}


class NoiseLabEvidenceAdapter:
    def __init__(self, strategy: str = "ltr_full", temperature: float = 0.45) -> None:
        self.strategy = str(strategy or "ltr_full")
        self.temperature = max(float(temperature), 1e-9)

    def from_scores_csv(
        self,
        path: str,
        query: QueryCase,
        entities: Sequence[str],
    ) -> EvidenceFrame:
        query_index = getattr(query, "query_index", None)
        if query_index is None:
            return EvidenceFrame(
                source="scores_csv",
                applied=False,
                strategy=self.strategy,
                task_index=str(query.task_index),
                reason="missing_query_index",
            )
        try:
            table = _SCORES_CACHE.get(path)
            if table is None:
                table = pd.read_csv(path)
                _SCORES_CACHE[path] = table
            rows = table[
                (table["query_index"].astype(int) == int(query_index))
                & (table["task_index"].astype(str) == str(query.task_index))
            ].copy()
            if rows.empty:
                return EvidenceFrame(
                    source="scores_csv",
                    applied=False,
                    strategy=self.strategy,
                    query_index=int(query_index),
                    task_index=str(query.task_index),
                    reason="no_matching_rows",
                )
            score_col = self.score_column(rows)
            if score_col not in rows.columns:
                return EvidenceFrame(
                    source="scores_csv",
                    applied=False,
                    strategy=self.strategy,
                    score_column=score_col,
                    query_index=int(query_index),
                    task_index=str(query.task_index),
                    reason=f"missing_score_column:{score_col}",
                )
            rows = rows.copy()
            values = (
                rows[score_col]
                .astype(float)
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0.0)
                .to_numpy()
            )
            probs = softmax(values / self.temperature)
            entity_keys = {canonical_entity_name(entity) for entity in entities}
            candidates: List[CandidateFrame] = []
            for rank, (row, prob) in enumerate(
                zip(rows.itertuples(index=False), probs), start=1
            ):
                object_id = str(getattr(row, "object_id", ""))
                if canonical_entity_name(object_id) not in entity_keys:
                    continue
                candidates.append(
                    self._candidate_from_score_row(
                        row=row,
                        rank=rank,
                        score_col=score_col,
                        prior_mass=float(prob),
                    )
                )
            if not candidates:
                return EvidenceFrame(
                    source="scores_csv",
                    applied=False,
                    strategy=self.strategy,
                    score_column=score_col,
                    query_index=int(query_index),
                    task_index=str(query.task_index),
                    reason="no_entity_overlap",
                )
            return EvidenceFrame(
                source="scores_csv",
                applied=True,
                strategy=self.strategy,
                score_column=score_col,
                query_index=int(query_index),
                task_index=str(query.task_index),
                candidates=candidates,
                metadata={"path": path},
            )
        except Exception as exc:
            return EvidenceFrame(
                source="scores_csv",
                applied=False,
                strategy=self.strategy,
                query_index=int(query_index),
                task_index=str(query.task_index),
                error=str(exc),
            )

    def from_runtime_scorer(
        self,
        telemetry: UnifiedTelemetry,
        query: QueryCase,
        inject_time: float,
        entities: Sequence[str],
    ) -> EvidenceFrame:
        try:
            from ..mace.graph import build_object_graph
            from ..noise_lab.beamformer import StructuralBeamformer
            from ..noise_lab.delay_localizer import DelayPatternLocalizer
            from ..noise_lab.noise_field import NoiseFieldScorer
            from ..noise_lab.reverb_mask import ReverbSuppressionMask
            from ..noise_lab.runner import _rank_objects
            from ..noise_lab.structural_encoder import StructuralObjectEncoder
            from ..noise_lab.subspace import SourceNoiseSubspaceDecomposer

            object_graph, graph_debug = build_object_graph(telemetry, query, inject_time)
            noise_scores = NoiseFieldScorer().score(object_graph)
            structural_scores = StructuralObjectEncoder().encode(object_graph)
            delay_scores = DelayPatternLocalizer().score(object_graph)
            beam_scores = StructuralBeamformer().score(object_graph)
            subspace_scores = SourceNoiseSubspaceDecomposer().score(
                object_graph, beam_scores=beam_scores
            )
            mask_scores = ReverbSuppressionMask().score(
                object_graph,
                noise_scores=noise_scores,
                structural_scores=structural_scores,
                delay_scores=delay_scores,
                beam_scores=beam_scores,
                subspace_scores=subspace_scores,
            )
            ranking = _rank_objects(
                object_graph,
                noise_scores,
                structural_scores,
                delay_scores,
                beam_scores,
                subspace_scores,
                mask_scores,
            )
            scores = [float(item.get("score", 0.0)) for item in ranking]
            lo = min(scores) if scores else 0.0
            hi = max(scores) if scores else 0.0
            entity_keys = {canonical_entity_name(entity) for entity in entities}
            candidates: List[CandidateFrame] = []
            for rank, item in enumerate(ranking, start=1):
                object_id = str(item.get("object_id", ""))
                if canonical_entity_name(object_id) not in entity_keys:
                    continue
                score = float(item.get("score", 0.0))
                candidates.append(
                    CandidateFrame(
                        candidate_id=f"runtime:{rank}:{object_id}",
                        component_id=object_id,
                        object_id=object_id,
                        noise_score=score,
                        calibrated_logit=score,
                        metric_evidence={
                            "anomaly_score": self._node_attr(object_graph, object_id, "anomaly_score"),
                            "metric_score": self._node_attr(object_graph, object_id, "metric_score"),
                            "change_score": self._node_attr(object_graph, object_id, "change_score"),
                        },
                        log_evidence={
                            "log_score": self._node_attr(object_graph, object_id, "log_score"),
                        },
                        trace_evidence={
                            "trace_score": self._node_attr(object_graph, object_id, "trace_score"),
                            "degree_in": item.get("degree_in", 0.0),
                            "degree_out": item.get("degree_out", 0.0),
                        },
                        time_candidates=self._runtime_time_candidates(
                            object_graph, object_id
                        ),
                        reason_candidates=self._reason_candidates_from_mapping(item),
                        structural_features={
                            "noise": noise_scores.get(object_id, {}),
                            "structure": structural_scores.get(object_id, {}),
                            "delay": delay_scores.get(object_id, {}),
                            "beam": beam_scores.get(object_id, {}),
                            "subspace": subspace_scores.get(object_id, {}),
                            "reverb": mask_scores.get(object_id, {}),
                        },
                        symptomness=float(
                            structural_scores.get(object_id, {}).get("symptom_likelihood", 0.0)
                        ),
                        source_likelihood=float(
                            structural_scores.get(object_id, {}).get("source_likelihood", 0.0)
                        ),
                        rank=rank,
                        prior_mass=rank_to_prior_mass(score, rank, lo, hi),
                    )
                )
            if not candidates:
                return EvidenceFrame(
                    source="runtime_scorer",
                    applied=False,
                    strategy=self.strategy,
                    task_index=str(query.task_index),
                    reason="no_entity_overlap",
                )
            return EvidenceFrame(
                source="runtime_scorer",
                applied=True,
                strategy=self.strategy,
                task_index=str(query.task_index),
                candidates=candidates,
                metadata={
                    "graph_query": graph_debug.get("query"),
                    "object_count": len(getattr(object_graph, "nodes", {})),
                },
            )
        except Exception as exc:
            return EvidenceFrame(
                source="runtime_scorer",
                applied=False,
                strategy=self.strategy,
                task_index=str(query.task_index),
                error=str(exc),
            )

    def score_column(self, rows: pd.DataFrame) -> str:
        if self.strategy in {"ltr_full", "ltr", "xgbrank"}:
            return "ltr_score"
        if self.strategy == "base":
            return "base_score"
        if self.strategy in rows.columns:
            return self.strategy
        return self.strategy

    def _candidate_from_score_row(
        self,
        row: Any,
        rank: int,
        score_col: str,
        prior_mass: float,
    ) -> CandidateFrame:
        object_id = str(getattr(row, "object_id", ""))
        score = float(getattr(row, score_col, 0.0))
        return CandidateFrame(
            candidate_id=f"csv:{getattr(row, 'query_index', '')}:{rank}:{object_id}",
            component_id=object_id,
            object_id=object_id,
            noise_score=float(getattr(row, "base_score", score)),
            calibrated_logit=score,
            metric_evidence={
                "anomaly_score": self._row_float(row, "anomaly_score"),
                "metric_score": self._row_float(row, "metric_score"),
                "change_score": self._row_float(row, "change_score"),
            },
            log_evidence={"log_score": self._row_float(row, "log_score")},
            trace_evidence={
                "trace_score": self._row_float(row, "trace_score"),
                "degree_in": self._row_float(row, "degree_in"),
                "degree_out": self._row_float(row, "degree_out"),
            },
            time_candidates=self._time_candidates_from_row(row),
            reason_candidates=self._reason_candidates_from_mapping(row),
            structural_features=self._prefixed_features(
                row,
                prefixes=("noise_", "structure_", "delay_", "beam_", "subspace_", "mask_"),
            ),
            symptomness=self._row_float(row, "structure_symptom_likelihood"),
            source_likelihood=self._row_float(row, "structure_source_likelihood"),
            rank=rank,
            prior_mass=prior_mass,
        )

    def _time_candidates_from_row(self, row: Any) -> List[Dict[str, Any]]:
        has_earliest = self._row_float(row, "has_earliest")
        if has_earliest <= 0:
            return []
        return [
            {
                "kind": "earliest_offset",
                "offset_seconds": self._row_float(row, "earliest_offset"),
                "support": has_earliest,
            }
        ]

    def _runtime_time_candidates(self, graph: Any, object_id: str) -> List[Dict[str, Any]]:
        node = getattr(graph, "nodes", {}).get(object_id)
        earliest_ts = getattr(node, "earliest_ts", None)
        if earliest_ts is None:
            return []
        return [{"kind": "earliest_ts", "timestamp": float(earliest_ts), "support": 1.0}]

    def _reason_candidates_from_mapping(self, obj: Any) -> List[Dict[str, Any]]:
        reasons: List[Dict[str, Any]] = []
        for name in ("cpu", "memory", "disk", "db", "network", "process"):
            value = self._mapping_float(obj, f"reason_{name}")
            if value > 0:
                reasons.append({"reason": name, "score": value})
        reasons.sort(key=lambda item: item["score"], reverse=True)
        return reasons

    def _prefixed_features(self, row: Any, prefixes: Sequence[str]) -> Dict[str, Any]:
        fields = getattr(row, "_fields", ())
        result: Dict[str, Any] = {}
        for field in fields:
            if any(str(field).startswith(prefix) for prefix in prefixes):
                result[field] = self._row_float(row, field)
        return result

    def _node_attr(self, graph: Any, object_id: str, attr: str) -> float:
        node = getattr(graph, "nodes", {}).get(object_id)
        if node is None:
            return 0.0
        return float(getattr(node, attr, 0.0) or 0.0)

    def _row_float(self, row: Any, name: str) -> float:
        return self._mapping_float(row, name)

    def _mapping_float(self, obj: Any, name: str) -> float:
        value = 0.0
        if isinstance(obj, dict):
            value = obj.get(name, 0.0)
        else:
            value = getattr(obj, name, 0.0)
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0
