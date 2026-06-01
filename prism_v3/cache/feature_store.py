"""Facade for the full PRISM feature cache stack."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple

import pandas as pd

from .causal_cache import CausalProfileCache
from .feature_cache import NoiseLabFeatureCache
from .graph_cache import ObjectGraphCache
from .key import anchor_id, cache_config_hash, query_id as make_query_id, safe_segment, telemetry_sha256
from .manifest import CacheManifest
from .store import CacheError as FeatureCacheError
from .store import CacheMiss, CacheStore, CacheValidationError, PIPELINE_VERSION
from .telemetry_cache import TelemetryCache
from .window_cache import WindowCache


@dataclass
class FeaturePipelineResult:
    """Compatibility result for label-free NoiseLab feature builders."""

    object_graph: Any
    graph_debug: Dict[str, Any]
    rows: List[Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)
    cache_dir: Path = Path()


class FeatureCacheStore:
    """High-level entry point used by offline builders and online PRISM."""

    def __init__(
        self,
        root: str | Path,
        *,
        strict: bool = True,
        pipeline_version: str = PIPELINE_VERSION,
        feature_pipeline_version: str = "",
    ) -> None:
        self.store = CacheStore(root, strict=strict, pipeline_version=pipeline_version)
        self.telemetry = TelemetryCache(self.store)
        self.windows = WindowCache(self.store)
        self.features = NoiseLabFeatureCache(self.store)
        self.causal = CausalProfileCache(self.store)
        self.feature_pipeline_version = str(feature_pipeline_version or "feature_pipeline_compat.v1")

    @property
    def root(self) -> Path:
        return self.store.root

    @property
    def strict(self) -> bool:
        return self.store.strict

    def load_or_build_telemetry(self, loader: Any, date_str: str, sub_system: str = ""):
        return self.telemetry.load_or_build(loader, date_str, sub_system)

    def load_or_build_window(
        self,
        *,
        telemetry: Any,
        query: Any,
        anchor_timestamp: float,
        anchor_source: str,
        baseline_window: int,
        fault_window: int,
        builder: Callable[[], Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]],
    ):
        return self.windows.load_or_build(
            telemetry=telemetry,
            query=query,
            anchor_timestamp=anchor_timestamp,
            anchor_source=anchor_source,
            baseline_window=baseline_window,
            fault_window=fault_window,
            builder=builder,
        )

    def load_or_build_noiselab_features(
        self,
        *,
        telemetry: Any,
        query: Any,
        anchor_timestamp: float,
        anchor_source: str,
        entities: Sequence[str],
        strategy: str,
        temperature: float,
    ):
        return self.features.load_or_build(
            telemetry=telemetry,
            query=query,
            anchor_timestamp=anchor_timestamp,
            anchor_source=anchor_source,
            entities=entities,
            strategy=strategy,
            temperature=temperature,
        )

    def load_or_build_cmi_profiles(self, **kwargs: Any):
        return self.causal.load_or_build_cmi(**kwargs)

    def load_cf_profile(self, **kwargs: Any):
        return self.causal.load_cf_profile(**kwargs)

    def write_cf_profile(self, **kwargs: Any):
        return self.causal.write_cf_profile(**kwargs)

    def config_hash(self, payload: Dict[str, Any]) -> str:
        return cache_config_hash(payload)

    def get_or_compute(
        self,
        query: Any,
        anchor: Any,
        telemetry: Any,
        compute: Callable[[], FeaturePipelineResult],
        *,
        config_hash: str = "",
        allow_recompute_on_error: bool = False,
    ) -> FeaturePipelineResult:
        try:
            return self.load(query, anchor, telemetry, config_hash=config_hash)
        except CacheMiss:
            pass
        except CacheValidationError:
            if self.strict and not bool(allow_recompute_on_error):
                raise
        result = compute()
        return self.write(query, anchor, telemetry, result, config_hash=config_hash)

    def load(
        self,
        query: Any,
        anchor: Any,
        telemetry: Any,
        *,
        config_hash: str = "",
    ) -> FeaturePipelineResult:
        qid, aid, cfg_hash = self._compat_ids(query, anchor, config_hash)
        anchor_source = self._anchor_source(anchor)
        tsha = telemetry_sha256(telemetry)
        cache_dir = self._compat_cache_dir(query, qid, aid, cfg_hash)
        self.store.read_manifest(
            cache_dir,
            cache_type="feature_pipeline_result",
            query_id=qid,
            anchor_source=anchor_source,
            telemetry_sha256=tsha,
            cache_config_hash=cfg_hash,
        )
        graph_cache = ObjectGraphCache(self.store)
        graph = graph_cache._read_graph(cache_dir)
        graph_debug = dict(self.store.read_json(cache_dir / "graph_debug.json"))
        rows = pd.read_parquet(cache_dir / "candidate_features.parquet").to_dict(orient="records")
        metadata = dict(self.store.read_json(cache_dir / "metadata.json"))
        metadata["cache_hit"] = True
        return FeaturePipelineResult(
            object_graph=graph,
            graph_debug=graph_debug,
            rows=rows,
            metadata=metadata,
            cache_dir=cache_dir,
        )

    def write(
        self,
        query: Any,
        anchor: Any,
        telemetry: Any,
        result: FeaturePipelineResult,
        *,
        config_hash: str = "",
    ) -> FeaturePipelineResult:
        qid, aid, cfg_hash = self._compat_ids(query, anchor, config_hash)
        anchor_source = self._anchor_source(anchor)
        tsha = telemetry_sha256(telemetry)
        cache_dir = self._compat_cache_dir(query, qid, aid, cfg_hash)
        cache_dir.mkdir(parents=True, exist_ok=True)
        graph_cache = ObjectGraphCache(self.store)
        graph_cache._write_graph(cache_dir, result.object_graph)
        self.store.write_json(cache_dir / "graph_debug.json", dict(result.graph_debug or {}))
        pd.DataFrame(list(result.rows or [])).to_parquet(
            cache_dir / "candidate_features.parquet",
            index=False,
        )
        metadata = dict(result.metadata or {})
        metadata.update(
            {
                "cache_hit": False,
                "cache_write": True,
                "feature_pipeline_version": self.feature_pipeline_version,
            }
        )
        self.store.write_json(cache_dir / "metadata.json", metadata)
        self.store.write_manifest(
            cache_dir,
            CacheManifest(
                cache_type="feature_pipeline_result",
                query_id=qid,
                anchor_timestamp=float(getattr(anchor, "timestamp", 0.0) or 0.0),
                anchor_source=anchor_source,
                telemetry_sha256=tsha,
                cache_config_hash=cfg_hash,
                metadata={
                    "anchor_id": aid,
                    "feature_pipeline_version": self.feature_pipeline_version,
                    "row_count": len(result.rows or []),
                },
            ),
        )
        return FeaturePipelineResult(
            object_graph=result.object_graph,
            graph_debug=dict(result.graph_debug or {}),
            rows=list(result.rows or []),
            metadata=metadata,
            cache_dir=cache_dir,
        )

    def _compat_ids(self, query: Any, anchor: Any, config_hash: str = "") -> Tuple[str, str, str]:
        qid = make_query_id(
            getattr(query, "system", ""),
            getattr(query, "sub_system", ""),
            getattr(query, "telemetry_date", ""),
            getattr(query, "task_index", ""),
            getattr(query, "query_index", None),
        )
        source = self._anchor_source(anchor)
        aid = anchor_id(
            float(getattr(anchor, "timestamp", 0.0) or 0.0),
            source,
            {"feature_pipeline": self.feature_pipeline_version, "compat": True},
        )
        cfg_hash = cache_config_hash(
            {
                "feature_pipeline": self.feature_pipeline_version,
                "compat": True,
                "config_hash": str(config_hash or ""),
            }
        )
        return qid, aid, cfg_hash

    def _compat_cache_dir(self, query: Any, qid: str, aid: str, cfg_hash: str) -> Path:
        return (
            self.store.layer_dir("features")
            / safe_segment(getattr(query, "system", ""))
            / safe_segment(getattr(query, "sub_system", "") or "default")
            / safe_segment(qid)
            / aid
            / "compat"
            / cfg_hash
        )

    def _anchor_source(self, anchor: Any) -> str:
        source = getattr(anchor, "source", "")
        return str(getattr(source, "value", source) or "")
