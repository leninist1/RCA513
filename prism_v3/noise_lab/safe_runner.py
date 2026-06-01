"""Label-free NoiseLab feature generation for certified LTR artifacts.

Unlike ``noise_lab.runner``, this module never calls ``match_query_to_records``
or reads ``record.csv`` while generating features.  Ground-truth labels may be
joined later by a separate training-only process.
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from ..config import QueryCase, SYSTEM_PATHS
from ..data.loader import OpenRCALoader
from ..leakage_guard import (
    AnchorSource,
    InferenceAnchor,
    assert_inference_query_safe,
    build_query_id,
)
from ..mace.graph import build_object_graph
from .beamformer import StructuralBeamformer
from .delay_localizer import DelayPatternLocalizer
from .noise_field import NoiseFieldScorer
from .reverb_mask import ReverbSuppressionMask
from .runner import _rank_objects
from .structural_encoder import StructuralObjectEncoder
from .subspace import SourceNoiseSubspaceDecomposer


FEATURE_PIPELINE_VERSION = "noise_lab_no_gt_v1"


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _public_anchor(loader: OpenRCALoader, query: QueryCase) -> Optional[InferenceAnchor]:
    timestamp = loader.infer_query_anchor_time(query)
    if timestamp is None:
        return None
    return InferenceAnchor(
        timestamp=float(timestamp),
        source=AnchorSource.PUBLIC_QUERY_WINDOW,
        confidence=1.0,
        evidence_ids=("query.time_window.start",),
    )


def _clean_query(query: QueryCase, query_index: int) -> QueryCase:
    """Strip evaluation-only fields before feature extraction."""
    query.query_index = int(query_index)
    query.ground_truth = None
    query.scoring_points = []
    assert_inference_query_safe(query)
    return query


def _flatten(prefix: str, payload: Dict[str, Any], out: Dict[str, Any]) -> None:
    for key, value in payload.items():
        name = f"{prefix}_{key}"
        if isinstance(value, (str, int, float, bool)) or value is None:
            out[name] = value


def _feature_rows_for_query(
    loader: OpenRCALoader,
    query: QueryCase,
    query_index: int,
) -> List[Dict[str, Any]]:
    query = _clean_query(query, query_index)
    anchor = _public_anchor(loader, query)
    if anchor is None:
        return []
    query.inject_time = anchor.timestamp
    query.telemetry_date = loader.resolve_telemetry_date(query)
    if not query.telemetry_date:
        return []
    telemetry = loader.load_telemetry(query.telemetry_date, query.sub_system)
    object_graph, graph_debug = build_object_graph(telemetry, query, anchor.timestamp)
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
    rows: List[Dict[str, Any]] = []
    query_id = build_query_id(query)
    for rank, candidate in enumerate(ranking, start=1):
        object_id = str(candidate.get("object_id", ""))
        node = object_graph.nodes.get(object_id)
        row: Dict[str, Any] = {
            "query_id": query_id,
            "query_index": int(query_index),
            "task_index": str(query.task_index),
            "system": str(query.system),
            "sub_system": str(query.sub_system),
            "telemetry_date": str(query.telemetry_date),
            "anchor_timestamp": float(anchor.timestamp),
            "anchor_source": anchor.source.value,
            "anchor_confidence": float(anchor.confidence),
            "object_id": object_id,
            "rank": int(rank),
            "base_score": float(candidate.get("score", 0.0)),
            "entity": str(candidate.get("entity", "")),
            "reason": str(candidate.get("reason", "")),
            "anomaly_score": float(getattr(node, "anomaly_score", 0.0) or 0.0),
            "metric_score": float(getattr(node, "metric_score", 0.0) or 0.0),
            "log_score": float(getattr(node, "log_score", 0.0) or 0.0),
            "trace_score": float(getattr(node, "trace_score", 0.0) or 0.0),
            "change_score": float(getattr(node, "change_score", 0.0) or 0.0),
            "earliest_timestamp": getattr(node, "earliest_timestamp", None),
            "feature_pipeline_version": FEATURE_PIPELINE_VERSION,
            "generated_without_gt": True,
            "graph_query": str(graph_debug.get("query", "")),
        }
        _flatten("noise", dict(candidate.get("noise", {}) or {}), row)
        _flatten("structure", dict(candidate.get("structure", {}) or {}), row)
        _flatten("delay", dict(candidate.get("delay", {}) or {}), row)
        _flatten("beam", dict(candidate.get("beam", {}) or {}), row)
        _flatten("subspace", dict(candidate.get("subspace", {}) or {}), row)
        _flatten("mask", dict(candidate.get("reverb_mask", {}) or {}), row)
        rows.append(row)
    return rows


def generate_no_gt_feature_rows(
    system_name: str,
    max_queries: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Generate label-free candidate rows for one system.

    This function intentionally has no path to ``load_records`` or
    ``match_query_to_records``.  It remains runnable when ``record.csv`` is
    hidden, which is enforced by canary tests.
    """
    loader = OpenRCALoader(system_name)
    rows: List[Dict[str, Any]] = []
    query_counter = 0
    for sub_system in loader.sub_systems:
        queries = loader.load_queries(sub_system)
        for query in queries:
            if max_queries is not None and query_counter >= max_queries:
                return rows
            rows.extend(_feature_rows_for_query(loader, query, query_counter))
            query_counter += 1
    return rows


def write_feature_csv(rows: Sequence[Dict[str, Any]], output: str) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def write_manifest(path: Path, *, system_name: str, row_count: int) -> Path:
    manifest_path = Path(f"{path}.manifest.json")
    payload = {
        "artifact_type": "no_leak_noiselab_scores",
        "feature_pipeline_version": FEATURE_PIPELINE_VERSION,
        "generated_without_gt": True,
        "allowed_anchor_sources": [AnchorSource.PUBLIC_QUERY_WINDOW.value],
        "fit_query_ids": [],
        "fit_dates": [],
        "fit_systems": [],
        "git_commit": "",
        "artifact_sha256": _sha256(path),
        "metadata": {
            "source_system": system_name,
            "row_count": int(row_count),
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "contains_labels": False,
            "safe_for_feature_generation": True,
        },
    }
    manifest_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate label-free NoiseLab features")
    parser.add_argument("--system", required=True, choices=list(SYSTEM_PATHS))
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-queries", type=int, default=None)
    args = parser.parse_args()
    rows = generate_no_gt_feature_rows(args.system, max_queries=args.max_queries)
    output = write_feature_csv(rows, args.output)
    manifest = write_manifest(output, system_name=args.system, row_count=len(rows))
    print(json.dumps({"rows": len(rows), "output": str(output), "manifest": str(manifest)}, indent=2))


if __name__ == "__main__":
    main()
