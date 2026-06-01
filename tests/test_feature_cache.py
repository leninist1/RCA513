from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from prism_v3.cache.feature_store import FeatureCacheStore, FeaturePipelineResult
from prism_v3.cache.manifest import CacheManifestError
from prism_v3.config import QueryCase, UnifiedTelemetry
from prism_v3.leakage_guard import AnchorSource, InferenceAnchor
from prism_v3.mace.graph import ObjectGraph, ObjectNode


def _query() -> QueryCase:
    query = QueryCase(
        task_index="task_1",
        system="Bank",
        sub_system="",
        instruction="between 2026-01-01 00:00:00 and 2026-01-01 00:10:00",
        time_window=("2026-01-01 00:00:00", "2026-01-01 00:10:00"),
    )
    query.query_index = 0
    query.telemetry_date = "2026_01_01"
    return query


def _telemetry() -> UnifiedTelemetry:
    metrics = pd.DataFrame(
        {
            "timestamp": [1.0, 2.0],
            "entity": ["a", "a"],
            "metric_name": ["cpu", "cpu"],
            "value": [1.0, 8.0],
        }
    )
    return UnifiedTelemetry(metrics=metrics, logs=None, traces=None, entities=["a"], system="Bank")


def _result() -> FeaturePipelineResult:
    graph = ObjectGraph(
        nodes={"a": ObjectNode(object_id="a", members=["a"], representative="a", anomaly_score=1.0)},
        adjacency={},
    )
    return FeaturePipelineResult(
        object_graph=graph,
        graph_debug={"query": "task_1"},
        rows=[{"query_id": "Bank::task_1:0", "object_id": "a", "base_score": 1.0}],
    )


def test_feature_cache_round_trip_and_checksum(tmp_path: Path) -> None:
    query = _query()
    telemetry = _telemetry()
    anchor = InferenceAnchor(1.0, AnchorSource.PUBLIC_QUERY_WINDOW)
    store = FeatureCacheStore(tmp_path, feature_pipeline_version="noise_lab_no_gt_v1")
    hit = store.write(query, anchor, telemetry, _result())
    assert hit.rows[0]["object_id"] == "a"
    loaded = store.load(query, anchor, telemetry)
    assert loaded.object_graph.nodes["a"].representative == "a"

    features = loaded.cache_dir / "candidate_features.parquet"
    pd.DataFrame([{"tampered": 1}]).to_parquet(features, index=False)
    with pytest.raises(CacheManifestError, match="payload sha256 mismatch"):
        store.load(query, anchor, telemetry)


def test_feature_cache_rejects_telemetry_mismatch(tmp_path: Path) -> None:
    query = _query()
    anchor = InferenceAnchor(1.0, AnchorSource.PUBLIC_QUERY_WINDOW)
    store = FeatureCacheStore(tmp_path, feature_pipeline_version="noise_lab_no_gt_v1")
    store.write(query, anchor, _telemetry(), _result())
    changed = _telemetry()
    changed.metrics.loc[0, "value"] = 99.0
    with pytest.raises(CacheManifestError, match="telemetry_sha256 mismatch"):
        store.load(query, anchor, changed)


def test_cmi_cache_key_ignores_window_index_round_trip(tmp_path: Path) -> None:
    query = _query()
    store = FeatureCacheStore(tmp_path)
    baseline = pd.DataFrame(
        {"timestamp": [1.0, 2.0], "entity": ["a", "a"], "value": [1.0, 2.0]},
        index=[10, 11],
    )
    fault = pd.DataFrame(
        {"timestamp": [3.0, 4.0], "entity": ["a", "a"], "value": [3.0, 4.0]},
        index=[20, 21],
    )
    calls = {"count": 0}

    def build() -> dict:
        calls["count"] += 1
        return {"a": {"cmi_score": 0.5}}

    first, first_debug = store.load_or_build_cmi_profiles(
        telemetry_sha256="telemetry-sha",
        query=query,
        anchor_timestamp=1.0,
        anchor_source=AnchorSource.PUBLIC_QUERY_WINDOW.value,
        entities=["a"],
        baseline_df=baseline,
        fault_df=fault,
        graph=[[0.0]],
        candidate_entities=["a"],
        max_conditioners=1,
        max_effect_scope=1,
        builder=build,
    )
    second, second_debug = store.load_or_build_cmi_profiles(
        telemetry_sha256="telemetry-sha",
        query=query,
        anchor_timestamp=1.0,
        anchor_source=AnchorSource.PUBLIC_QUERY_WINDOW.value,
        entities=["a"],
        baseline_df=baseline.reset_index(drop=True),
        fault_df=fault.reset_index(drop=True),
        graph=[[0.0]],
        candidate_entities=["a"],
        max_conditioners=1,
        max_effect_scope=1,
        builder=build,
    )

    assert first == second == {"a": {"cmi_score": 0.5}}
    assert first_debug["cache_hit"] is False
    assert second_debug["cache_hit"] is True
    assert calls["count"] == 1
