"""L2 NoiseLab atomic feature cache."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import pandas as pd

from ..noise_native.evidence_frame import CandidateFrame, EvidenceFrame, canonical_entity_name, rank_to_prior_mass
from .graph_cache import (
    CANONICALIZATION_VERSION,
    GRAPH_BUILDER_VERSION,
    METRIC_GRAPH_CONFIG_VERSION,
    TRACE_GRAPH_CONFIG_VERSION,
    ObjectGraphCache,
)
from .key import anchor_id, cache_config_hash, query_id as make_query_id, safe_segment, stable_json, telemetry_sha256
from .manifest import CacheManifest
from .store import CacheMiss, CacheStore, CacheValidationError


NOISE_FIELD_VERSION = "noise_field.v1"
STRUCTURAL_ENCODER_VERSION = "structural_encoder.v1"
DELAY_LOCALIZER_VERSION = "delay_localizer.v1"
BEAMFORMER_VERSION = "beamformer.v1"
SUBSPACE_VERSION = "subspace.v1"
REVERB_MASK_VERSION = "reverb_mask.v1"


class NoiseLabFeatureCache:
    def __init__(self, store: CacheStore) -> None:
        self.store = store
        self.graph_cache = ObjectGraphCache(store)

    def ids(self, query: Any, anchor_timestamp: float, anchor_source: str, strategy: str, temperature: float) -> Tuple[str, str, str]:
        qid = make_query_id(
            getattr(query, "system", ""),
            getattr(query, "sub_system", ""),
            getattr(query, "telemetry_date", ""),
            getattr(query, "task_index", ""),
            getattr(query, "query_index", None),
        )
        aid = anchor_id(anchor_timestamp, anchor_source, {"feature_pipeline": "noiselab_atomic.v1"})
        cfg_hash = cache_config_hash(
            {
                "graph_builder": GRAPH_BUILDER_VERSION,
                "trace_graph_config": TRACE_GRAPH_CONFIG_VERSION,
                "metric_graph_config": METRIC_GRAPH_CONFIG_VERSION,
                "canonicalization": CANONICALIZATION_VERSION,
                "noise_field": NOISE_FIELD_VERSION,
                "structural_encoder": STRUCTURAL_ENCODER_VERSION,
                "delay_localizer": DELAY_LOCALIZER_VERSION,
                "beamformer": BEAMFORMER_VERSION,
                "subspace": SUBSPACE_VERSION,
                "reverb_mask": REVERB_MASK_VERSION,
                "strategy": str(strategy or "ltr_full"),
                "temperature": round(float(temperature), 8),
            }
        )
        return qid, aid, cfg_hash

    def cache_dir(self, query: Any, qid: str, aid: str, cfg_hash: str) -> Path:
        return (
            self.store.layer_dir("features")
            / safe_segment(getattr(query, "system", ""))
            / safe_segment(getattr(query, "sub_system", "") or "default")
            / safe_segment(qid)
            / aid
            / cfg_hash
        )

    def load_or_build(
        self,
        *,
        telemetry: Any,
        query: Any,
        anchor_timestamp: float,
        anchor_source: str,
        entities: Sequence[str],
        strategy: str,
        temperature: float,
    ) -> Tuple[EvidenceFrame, Dict[str, Any]]:
        tsha = telemetry_sha256(telemetry)
        qid, aid, cfg_hash = self.ids(query, anchor_timestamp, anchor_source, strategy, temperature)
        cache_dir = self.cache_dir(query, qid, aid, cfg_hash)
        try:
            self.store.read_manifest(
                cache_dir,
                cache_type="noiselab_features",
                query_id=qid,
                anchor_source=anchor_source,
                telemetry_sha256=tsha,
                cache_config_hash=cfg_hash,
            )
            frame = self._read_frame(cache_dir, query, strategy)
            debug = frame.to_debug(limit=5)
            debug.update(
                {
                    "cache_layer": "noiselab_features",
                    "cache_hit": True,
                    "cache_dir": str(cache_dir),
                }
            )
            return frame, debug
        except CacheMiss:
            pass
        except CacheValidationError:
            if self.store.strict:
                raise

        frame, build_debug = self._build_frame(
            telemetry=telemetry,
            query=query,
            anchor_timestamp=anchor_timestamp,
            anchor_source=anchor_source,
            entities=entities,
            strategy=strategy,
        )
        self._write(
            cache_dir=cache_dir,
            frame=frame,
            telemetry_sha=tsha,
            qid=qid,
            aid=aid,
            anchor_timestamp=anchor_timestamp,
            anchor_source=anchor_source,
            config_hash=cfg_hash,
            build_debug=build_debug,
        )
        debug = frame.to_debug(limit=5)
        debug.update(
            {
                "cache_layer": "noiselab_features",
                "cache_hit": False,
                "cache_write": True,
                "cache_dir": str(cache_dir),
                "build": build_debug,
            }
        )
        return frame, debug

    def _build_frame(
        self,
        *,
        telemetry: Any,
        query: Any,
        anchor_timestamp: float,
        anchor_source: str,
        entities: Sequence[str],
        strategy: str,
    ) -> Tuple[EvidenceFrame, Dict[str, Any]]:
        from ..mace.graph import build_object_graph
        from ..noise_lab.beamformer import StructuralBeamformer
        from ..noise_lab.delay_localizer import DelayPatternLocalizer
        from ..noise_lab.noise_field import NoiseFieldScorer
        from ..noise_lab.reverb_mask import ReverbSuppressionMask
        from ..noise_lab.ranking import rank_objects
        from ..noise_lab.structural_encoder import StructuralObjectEncoder
        from ..noise_lab.subspace import SourceNoiseSubspaceDecomposer

        graph, graph_debug, graph_cache_debug = self.graph_cache.load_or_build(
            telemetry=telemetry,
            query=query,
            anchor_timestamp=anchor_timestamp,
            anchor_source=anchor_source,
            builder=lambda: build_object_graph(telemetry, query, anchor_timestamp),
        )
        noise_scores = NoiseFieldScorer().score(graph)
        structural_scores = StructuralObjectEncoder().encode(graph)
        delay_scores = DelayPatternLocalizer().score(graph)
        beam_scores = StructuralBeamformer().score(graph)
        subspace_scores = SourceNoiseSubspaceDecomposer().score(graph, beam_scores=beam_scores)
        mask_scores = ReverbSuppressionMask().score(
            graph,
            noise_scores=noise_scores,
            structural_scores=structural_scores,
            delay_scores=delay_scores,
            beam_scores=beam_scores,
            subspace_scores=subspace_scores,
        )
        ranking = rank_objects(
            graph,
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
            node = getattr(graph, "nodes", {}).get(object_id)
            score = float(item.get("score", 0.0))
            structure = structural_scores.get(object_id, {})
            candidates.append(
                CandidateFrame(
                    candidate_id=f"cache:{getattr(query, 'task_index', '')}:{rank}:{object_id}",
                    component_id=object_id,
                    object_id=object_id,
                    noise_score=score,
                    calibrated_logit=score,
                    metric_evidence={
                        "anomaly_score": _node_attr(node, "anomaly_score"),
                        "metric_score": _node_attr(node, "metric_score"),
                        "change_score": _node_attr(node, "change_score"),
                    },
                    log_evidence={"log_score": _node_attr(node, "log_score")},
                    trace_evidence={
                        "trace_score": _node_attr(node, "trace_score"),
                        "degree_in": item.get("degree_in", 0.0),
                        "degree_out": item.get("degree_out", 0.0),
                    },
                    time_candidates=_time_candidates(node),
                    reason_candidates=_reason_candidates(item),
                    structural_features={
                        "noise": noise_scores.get(object_id, {}),
                        "structure": structure,
                        "delay": delay_scores.get(object_id, {}),
                        "beam": beam_scores.get(object_id, {}),
                        "subspace": subspace_scores.get(object_id, {}),
                        "reverb": mask_scores.get(object_id, {}),
                    },
                    symptomness=float(structure.get("symptom_likelihood", 0.0) or 0.0),
                    source_likelihood=float(structure.get("source_likelihood", 0.0) or 0.0),
                    rank=rank,
                    prior_mass=rank_to_prior_mass(score, rank, lo, hi),
                )
            )
        frame = EvidenceFrame(
            source="runtime_scorer",
            applied=bool(candidates),
            strategy=strategy,
            task_index=str(getattr(query, "task_index", "")),
            candidates=candidates,
            reason="" if candidates else "no_entity_overlap",
            metadata={
                "graph_query": graph_debug.get("query"),
                "object_count": len(getattr(graph, "nodes", {}) or {}),
            },
        )
        return frame, {
            "object_graph": graph_cache_debug,
            "object_count": len(getattr(graph, "nodes", {}) or {}),
            "ranking_count": len(ranking),
            "candidate_count": len(candidates),
        }

    def _write(
        self,
        *,
        cache_dir: Path,
        frame: EvidenceFrame,
        telemetry_sha: str,
        qid: str,
        aid: str,
        anchor_timestamp: float,
        anchor_source: str,
        config_hash: str,
        build_debug: Dict[str, Any],
    ) -> None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        rows = [_candidate_to_row(candidate, frame, qid, aid, anchor_timestamp, anchor_source) for candidate in frame.candidates]
        columns = [
            "query_id", "anchor_id", "anchor_timestamp", "anchor_source", "anchor_confidence",
            "task_index", "object_id", "component_id", "candidate_id", "rank",
            "base_score", "calibrated_logit", "prior_mass", "metric_anomaly_score",
            "metric_metric_score", "metric_change_score", "log_log_score",
            "trace_trace_score", "trace_degree_in", "trace_degree_out", "symptomness",
            "source_likelihood", "time_candidates", "reason_candidates", "noise_features",
            "structure_features", "delay_features", "beam_features", "subspace_features",
            "mask_features", "structural_features",
        ]
        pd.DataFrame(rows, columns=columns).to_parquet(cache_dir / "candidate_features.parquet", index=False)
        self.store.write_json(cache_dir / "feature_debug.json", build_debug)
        self.store.write_manifest(
            cache_dir,
            CacheManifest(
                cache_type="noiselab_features",
                query_id=qid,
                anchor_timestamp=float(anchor_timestamp),
                anchor_source=anchor_source,
                telemetry_sha256=telemetry_sha,
                cache_config_hash=config_hash,
                metadata={
                    "anchor_id": aid,
                    "candidate_count": len(frame.candidates),
                    "feature_versions": {
                        "noise_field": NOISE_FIELD_VERSION,
                        "structural_encoder": STRUCTURAL_ENCODER_VERSION,
                        "delay_localizer": DELAY_LOCALIZER_VERSION,
                        "beamformer": BEAMFORMER_VERSION,
                        "subspace": SUBSPACE_VERSION,
                        "reverb_mask": REVERB_MASK_VERSION,
                    },
                },
            ),
        )

    def _read_frame(self, cache_dir: Path, query: Any, strategy: str) -> EvidenceFrame:
        path = cache_dir / "candidate_features.parquet"
        if not path.exists():
            return EvidenceFrame(
                source="runtime_scorer",
                applied=False,
                strategy=strategy,
                task_index=str(getattr(query, "task_index", "")),
                reason="missing_candidate_features",
            )
        table = pd.read_parquet(path)
        candidates = [_row_to_candidate(row) for row in table.to_dict(orient="records")]
        return EvidenceFrame(
            source="runtime_scorer",
            applied=bool(candidates),
            strategy=strategy,
            task_index=str(getattr(query, "task_index", "")),
            candidates=candidates,
            reason="" if candidates else "no_entity_overlap",
            metadata={"cache_source": str(cache_dir)},
        )


def _node_attr(node: Any, attr: str) -> float:
    if node is None:
        return 0.0
    try:
        return float(getattr(node, attr, 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _time_candidates(node: Any) -> List[Dict[str, Any]]:
    timestamp = getattr(node, "earliest_timestamp", None) if node is not None else None
    if timestamp is None:
        return []
    return [{"kind": "earliest_ts", "timestamp": float(timestamp), "support": 1.0}]


def _reason_candidates(item: Dict[str, Any]) -> List[Dict[str, Any]]:
    reason = str(item.get("reason", "") or "")
    if not reason:
        return []
    return [{"reason": reason, "score": 1.0}]


def _candidate_to_row(
    candidate: CandidateFrame,
    frame: EvidenceFrame,
    qid: str,
    aid: str,
    anchor_timestamp: float,
    anchor_source: str,
) -> Dict[str, Any]:
    structural = dict(candidate.structural_features or {})
    return {
        "query_id": qid,
        "anchor_id": aid,
        "anchor_timestamp": float(anchor_timestamp),
        "anchor_source": anchor_source,
        "anchor_confidence": 1.0,
        "task_index": frame.task_index,
        "object_id": candidate.object_id,
        "component_id": candidate.component_id,
        "candidate_id": candidate.candidate_id,
        "rank": int(candidate.rank or 0),
        "base_score": float(candidate.noise_score),
        "calibrated_logit": float(candidate.calibrated_logit),
        "prior_mass": float(candidate.prior_mass),
        "metric_anomaly_score": float(candidate.metric_evidence.get("anomaly_score", 0.0) or 0.0),
        "metric_metric_score": float(candidate.metric_evidence.get("metric_score", 0.0) or 0.0),
        "metric_change_score": float(candidate.metric_evidence.get("change_score", 0.0) or 0.0),
        "log_log_score": float(candidate.log_evidence.get("log_score", 0.0) or 0.0),
        "trace_trace_score": float(candidate.trace_evidence.get("trace_score", 0.0) or 0.0),
        "trace_degree_in": float(candidate.trace_evidence.get("degree_in", 0.0) or 0.0),
        "trace_degree_out": float(candidate.trace_evidence.get("degree_out", 0.0) or 0.0),
        "symptomness": float(candidate.symptomness),
        "source_likelihood": float(candidate.source_likelihood),
        "time_candidates": stable_json(candidate.time_candidates),
        "reason_candidates": stable_json(candidate.reason_candidates),
        "noise_features": stable_json(structural.get("noise", {})),
        "structure_features": stable_json(structural.get("structure", {})),
        "delay_features": stable_json(structural.get("delay", {})),
        "beam_features": stable_json(structural.get("beam", {})),
        "subspace_features": stable_json(structural.get("subspace", {})),
        "mask_features": stable_json(structural.get("reverb", structural.get("mask", {}))),
        "structural_features": stable_json(structural),
    }


def _row_to_candidate(row: Dict[str, Any]) -> CandidateFrame:
    structural = {
        "noise": _json(row.get("noise_features"), {}),
        "structure": _json(row.get("structure_features"), {}),
        "delay": _json(row.get("delay_features"), {}),
        "beam": _json(row.get("beam_features"), {}),
        "subspace": _json(row.get("subspace_features"), {}),
        "reverb": _json(row.get("mask_features"), {}),
    }
    return CandidateFrame(
        candidate_id=str(row.get("candidate_id", "")),
        component_id=str(row.get("component_id", "") or row.get("object_id", "")),
        object_id=str(row.get("object_id", "")),
        noise_score=float(row.get("base_score", 0.0) or 0.0),
        calibrated_logit=float(row.get("calibrated_logit", 0.0) or 0.0),
        metric_evidence={
            "anomaly_score": float(row.get("metric_anomaly_score", 0.0) or 0.0),
            "metric_score": float(row.get("metric_metric_score", 0.0) or 0.0),
            "change_score": float(row.get("metric_change_score", 0.0) or 0.0),
        },
        log_evidence={"log_score": float(row.get("log_log_score", 0.0) or 0.0)},
        trace_evidence={
            "trace_score": float(row.get("trace_trace_score", 0.0) or 0.0),
            "degree_in": float(row.get("trace_degree_in", 0.0) or 0.0),
            "degree_out": float(row.get("trace_degree_out", 0.0) or 0.0),
        },
        time_candidates=list(_json(row.get("time_candidates"), [])),
        reason_candidates=list(_json(row.get("reason_candidates"), [])),
        structural_features=structural,
        symptomness=float(row.get("symptomness", 0.0) or 0.0),
        source_likelihood=float(row.get("source_likelihood", 0.0) or 0.0),
        rank=int(row.get("rank", 0) or 0),
        prior_mass=float(row.get("prior_mass", 0.0) or 0.0),
    )


def _json(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value
    text = str(value)
    if not text:
        return default
    try:
        return json.loads(text)
    except Exception:
        return default
