"""Structured NoiseLab perception frames.

The frame layer keeps NoiseLab observations as candidate-level evidence instead
of collapsing them immediately into a scalar prior. PRISM can still project the
frame into the legacy prior vector while newer agent state consumes the richer
candidate ledger.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence
import math
import re

import numpy as np


EPS = 1e-9


def canonical_entity_name(name: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name or "").lower())


def normalize(values: np.ndarray) -> np.ndarray:
    total = float(np.sum(values))
    if total <= EPS:
        if values.size == 0:
            return values
        return np.full(values.shape, 1.0 / values.size, dtype=float)
    return values / total


def softmax(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    shifted = values - np.max(values)
    exp_v = np.exp(shifted)
    return exp_v / max(float(np.sum(exp_v)), EPS)


@dataclass
class CandidateFrame:
    candidate_id: str
    component_id: str
    object_id: str
    noise_score: float = 0.0
    calibrated_logit: float = 0.0
    metric_evidence: Dict[str, Any] = field(default_factory=dict)
    log_evidence: Dict[str, Any] = field(default_factory=dict)
    trace_evidence: Dict[str, Any] = field(default_factory=dict)
    time_candidates: List[Dict[str, Any]] = field(default_factory=list)
    reason_candidates: List[Dict[str, Any]] = field(default_factory=list)
    structural_features: Dict[str, Any] = field(default_factory=dict)
    symptomness: float = 0.0
    source_likelihood: float = 0.0
    rank: Optional[int] = None
    prior_mass: float = 0.0

    @property
    def canonical_component(self) -> str:
        return canonical_entity_name(self.component_id or self.object_id)

    def to_debug(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "component_id": self.component_id,
            "object_id": self.object_id,
            "noise_score": round(float(self.noise_score), 6),
            "calibrated_logit": round(float(self.calibrated_logit), 6),
            "prior_mass": round(float(self.prior_mass), 6),
            "rank": self.rank,
            "metric_evidence": self.metric_evidence,
            "log_evidence": self.log_evidence,
            "trace_evidence": self.trace_evidence,
            "time_candidates": self.time_candidates,
            "reason_candidates": self.reason_candidates,
            "structural_features": self.structural_features,
            "symptomness": round(float(self.symptomness), 6),
            "source_likelihood": round(float(self.source_likelihood), 6),
        }


@dataclass
class EvidenceFrame:
    source: str
    applied: bool
    strategy: str = ""
    score_column: str = ""
    query_index: Optional[int] = None
    task_index: str = ""
    candidates: List[CandidateFrame] = field(default_factory=list)
    reason: str = ""
    error: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_prior_vector(self, entities: Sequence[str]) -> Optional[np.ndarray]:
        if not self.applied or not self.candidates:
            return None
        raw = np.zeros(len(entities), dtype=float)
        entity_index = {
            canonical_entity_name(entity): idx for idx, entity in enumerate(entities)
        }
        for candidate in self.candidates:
            idx = entity_index.get(candidate.canonical_component)
            if idx is not None:
                raw[idx] = max(raw[idx], float(candidate.prior_mass))
        if float(np.sum(raw)) <= EPS:
            return None
        return normalize(raw)

    def top_candidates(self, limit: int = 5) -> List[CandidateFrame]:
        return sorted(
            self.candidates,
            key=lambda item: (float(item.prior_mass), float(item.noise_score)),
            reverse=True,
        )[:limit]

    def to_debug(self, limit: int = 5) -> Dict[str, Any]:
        debug = {
            "enabled": True,
            "source": self.source,
            "applied": bool(self.applied),
            "strategy": self.strategy,
            "score_column": self.score_column,
            "query_index": self.query_index,
            "task_index": self.task_index,
            "candidate_count": len(self.candidates),
            "frame_schema": "noise_native.v1",
            "top": [
                {
                    "entity": candidate.component_id,
                    "score": round(float(candidate.prior_mass), 4),
                }
                for candidate in self.top_candidates(limit)
                if candidate.prior_mass > 0
            ],
            "candidate_frame_top": [
                candidate.to_debug() for candidate in self.top_candidates(limit)
            ],
        }
        if self.reason:
            debug["reason"] = self.reason
        if self.error:
            debug["error"] = self.error
        if self.metadata:
            debug["metadata"] = self.metadata
        return debug


def rank_to_prior_mass(score: float, rank: int, lo: float, hi: float) -> float:
    span = max(float(hi) - float(lo), EPS)
    score_norm = (float(score) - float(lo)) / span
    rank_score = 1.0 / math.log2(float(rank) + 1.0)
    return max(0.0, 0.60 * score_norm + 0.40 * rank_score)
