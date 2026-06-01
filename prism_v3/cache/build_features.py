"""Offline cache builder for PRISM v3 cache artifacts.

This CLI intentionally loads only public queries and telemetry. It never calls
record matching, never sets ground truth, and strips scoring points before any
cache key or feature builder sees the query.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from ..config import SYSTEM_PATHS
from ..data.loader import OpenRCALoader
from ..noise_native.cmi import build_cmi_profiles
from ..prism import PRISMConfig, PRISMPipeline
from .feature_store import FeatureCacheStore
from .key import telemetry_sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build PRISM v3 no-leakage cache artifacts")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--systems", nargs="+", default=["Bank"])
    parser.add_argument("--date", default="")
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--strict", action="store_true", help="Fail if an existing cache artifact is invalid")
    parser.add_argument("--noise-lab-strategy", default="ltr_full")
    parser.add_argument("--noise-lab-temperature", type=float, default=0.45)
    parser.add_argument("--build-cmi", action="store_true", help="Also prebuild CMI profiles for cached NoiseLab top candidates")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    store = FeatureCacheStore(args.cache_dir, strict=bool(args.strict))
    summary: Dict[str, Any] = {
        "cache_dir": str(store.root),
        "systems": list(args.systems),
        "total_queries": 0,
        "built": [],
        "skipped": [],
        "errors": [],
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    for system_name in args.systems:
        if system_name not in SYSTEM_PATHS:
            summary["errors"].append({"system": system_name, "error": "unknown_system"})
            continue
        loader = OpenRCALoader(system_name)
        for sub_system in SYSTEM_PATHS[system_name]["sub_systems"]:
            queries = loader.load_queries(sub_system)
            if args.max_queries is not None:
                queries = queries[: args.max_queries]
            for query_index, query in enumerate(queries):
                query.query_index = query_index
                query.ground_truth = None
                query.scoring_points = []
                inference_time = loader.infer_query_anchor_time(query)
                if inference_time is None:
                    summary["skipped"].append({"system": system_name, "task_index": query.task_index, "reason": "bad_query_window"})
                    continue
                date_str = loader.resolve_telemetry_date(query)
                if not date_str or (args.date and date_str != args.date):
                    continue
                query.inject_time = inference_time
                query.telemetry_date = date_str
                query.anchor_source = "public_query_window"
                try:
                    telemetry, telemetry_debug = store.load_or_build_telemetry(loader, date_str, sub_system)
                    cfg = PRISMConfig(
                        noise_lab_enabled=True,
                        noise_lab_strategy=args.noise_lab_strategy,
                        noise_lab_temperature=float(args.noise_lab_temperature),
                        feature_cache_dir=str(store.root),
                        feature_cache_strict=bool(args.strict),
                    )
                    prism = PRISMPipeline(system_name, config=cfg)
                    induced, object_debug = prism._induce_object_telemetry_cached(telemetry, query, inference_time)
                    entities = prism._collect_entities(induced, query)
                    baseline_df, fault_df, window_debug = store.load_or_build_window(
                        telemetry=induced,
                        query=query,
                        anchor_timestamp=inference_time,
                        anchor_source=query.anchor_source,
                        baseline_window=int(cfg.baseline_window),
                        fault_window=int(cfg.fault_window),
                        builder=lambda: prism._split_temporal_public_query_window(induced, query, inference_time),
                    )
                    frame, feature_debug = store.load_or_build_noiselab_features(
                        telemetry=induced,
                        query=query,
                        anchor_timestamp=inference_time,
                        anchor_source=query.anchor_source,
                        entities=entities,
                        strategy=args.noise_lab_strategy,
                        temperature=float(args.noise_lab_temperature),
                    )
                    cmi_debug: Dict[str, Any] = {"enabled": False}
                    if args.build_cmi and entities and frame.candidates and not baseline_df.empty and not fault_df.empty:
                        trace_edges = prism._trace_edges(induced.traces, inference_time, entities)
                        metric_signal, anomaly_times, _metric_detail = prism._metric_anomaly_scores(
                            baseline_df, fault_df, entities, trace_edges
                        )
                        metric_edges = prism._metric_edges(anomaly_times, entities)
                        W0, _W_frozen = prism._fuse_edges(trace_edges, metric_edges, entities, induced, metric_signal)
                        candidate_entities = _candidate_entities(frame, entities, limit=int(cfg.noise_native_max_events))
                        _cmi_profiles, cmi_debug = store.load_or_build_cmi_profiles(
                            telemetry_sha256=telemetry_sha256(induced),
                            query=query,
                            anchor_timestamp=inference_time,
                            anchor_source=query.anchor_source,
                            entities=entities,
                            baseline_df=baseline_df,
                            fault_df=fault_df,
                            graph=W0,
                            candidate_entities=candidate_entities,
                            max_conditioners=int(cfg.noise_native_cmi_max_conditioners),
                            max_effect_scope=int(cfg.noise_native_cmi_max_effect_scope),
                            builder=lambda: build_cmi_profiles(
                                entities=entities,
                                baseline_df=baseline_df,
                                fault_df=fault_df,
                                graph=W0,
                                candidate_entities=candidate_entities,
                                max_conditioners=int(cfg.noise_native_cmi_max_conditioners),
                                max_effect_scope=int(cfg.noise_native_cmi_max_effect_scope),
                            ),
                        )
                    summary["total_queries"] += 1
                    summary["built"].append(
                        {
                            "system": system_name,
                            "sub_system": sub_system or "default",
                            "date": date_str,
                            "task_index": query.task_index,
                            "query_index": query_index,
                            "telemetry_cache": telemetry_debug,
                            "window_cache_hit": bool(window_debug.get("cache_hit", False)),
                            "feature_cache_hit": bool(feature_debug.get("cache_hit", False)),
                            "candidate_count": len(frame.candidates),
                            "cmi_cache": cmi_debug,
                        }
                    )
                except Exception as exc:
                    summary["errors"].append(
                        {
                            "system": system_name,
                            "sub_system": sub_system or "default",
                            "task_index": query.task_index,
                            "query_index": query_index,
                            "error": str(exc),
                        }
                    )
    summary["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    out_path = Path(args.cache_dir) / "build_features_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=True, indent=2, sort_keys=True)
        f.write("\n")
    print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))


def _candidate_entities(frame: Any, entities: List[str], limit: int) -> List[str]:
    canonical = {
        "".join(ch for ch in str(entity).lower() if ch.isalnum()): entity
        for entity in entities
    }
    selected: List[str] = []
    for candidate in frame.top_candidates(max(1, int(limit))):
        key = getattr(candidate, "canonical_component", "")
        entity = canonical.get(key, str(getattr(candidate, "component_id", "")))
        if entity in entities and entity not in selected:
            selected.append(entity)
    return selected


if __name__ == "__main__":
    main()
