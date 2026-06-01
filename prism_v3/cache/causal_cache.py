"""L3 causal profile caches for CMI and counterfactual base profiles."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Callable, Dict, Sequence, Tuple

import numpy as np

from .key import anchor_id, cache_config_hash, dataframe_content_sha256, query_id as make_query_id, safe_segment, stable_hash
from .manifest import CacheManifest
from .store import CacheMiss, CacheStore, CacheValidationError


CMI_BUILDER_VERSION = "cmi_profiles.v1"
CF_BUILDER_VERSION = "cf_profiles.v1"


class CausalProfileCache:
    def __init__(self, store: CacheStore) -> None:
        self.store = store

    def _ids(self, query: Any, anchor_timestamp: float, anchor_source: str) -> Tuple[str, str]:
        qid = make_query_id(
            getattr(query, "system", ""),
            getattr(query, "sub_system", ""),
            getattr(query, "telemetry_date", ""),
            getattr(query, "task_index", ""),
            getattr(query, "query_index", None),
        )
        aid = anchor_id(anchor_timestamp, anchor_source, {"causal_cache": "v1"})
        return qid, aid

    def _base_dir(self, query: Any, qid: str, aid: str) -> Path:
        return (
            self.store.layer_dir("causal")
            / safe_segment(getattr(query, "system", ""))
            / safe_segment(getattr(query, "sub_system", "") or "default")
            / safe_segment(qid)
            / aid
        )

    def load_or_build_cmi(
        self,
        *,
        telemetry_sha256: str,
        query: Any,
        anchor_timestamp: float,
        anchor_source: str,
        entities: Sequence[str],
        baseline_df: Any,
        fault_df: Any,
        graph: Any,
        candidate_entities: Sequence[str],
        max_conditioners: int,
        max_effect_scope: int,
        builder: Callable[[], Dict[str, Dict[str, Any]]],
    ) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
        qid, aid = self._ids(query, anchor_timestamp, anchor_source)
        pool_hash = stable_hash({"candidate_entities": list(candidate_entities)})
        cfg_hash = cache_config_hash(
            {
                "builder": CMI_BUILDER_VERSION,
                "entities": list(entities),
                "baseline": dataframe_content_sha256(baseline_df),
                "fault": dataframe_content_sha256(fault_df),
                "graph": _array_hash(graph),
                "max_conditioners": int(max_conditioners),
                "max_effect_scope": int(max_effect_scope),
                "pool_hash": pool_hash,
            }
        )
        cache_dir = self._base_dir(query, qid, aid) / "cmi" / pool_hash[:16] / cfg_hash
        try:
            self.store.read_manifest(
                cache_dir,
                cache_type="cmi_profiles",
                query_id=qid,
                anchor_source=anchor_source,
                telemetry_sha256=telemetry_sha256,
                cache_config_hash=cfg_hash,
            )
            payload = dict(self.store.read_json(cache_dir / "cmi_profiles.json") or {})
            return payload, {
                "cache_layer": "cmi_profiles",
                "cache_hit": True,
                "cache_dir": str(cache_dir),
                "profile_count": len(payload),
            }
        except CacheMiss:
            pass
        except CacheValidationError:
            if self.store.strict:
                raise
        payload = builder()
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.store.write_json(cache_dir / "cmi_profiles.json", payload)
        self.store.write_manifest(
            cache_dir,
            CacheManifest(
                cache_type="cmi_profiles",
                query_id=qid,
                anchor_timestamp=float(anchor_timestamp),
                anchor_source=anchor_source,
                telemetry_sha256=telemetry_sha256,
                cache_config_hash=cfg_hash,
                metadata={
                    "anchor_id": aid,
                    "pool_hash": pool_hash,
                    "candidate_entities": list(candidate_entities),
                    "builder_version": CMI_BUILDER_VERSION,
                    "profile_count": len(payload),
                },
            ),
        )
        return payload, {
            "cache_layer": "cmi_profiles",
            "cache_hit": False,
            "cache_write": True,
            "cache_dir": str(cache_dir),
            "profile_count": len(payload),
        }

    def load_cf_profile(
        self,
        *,
        telemetry_sha256: str,
        query: Any,
        anchor_timestamp: float,
        anchor_source: str,
        entity: str,
        degradation_scope: Sequence[str],
        graph: Any,
        config_payload: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        qid, aid = self._ids(query, anchor_timestamp, anchor_source)
        cfg_hash = self.cf_config_hash(entity, degradation_scope, graph, config_payload)
        cache_dir = self._base_dir(query, qid, aid) / "cf" / safe_segment(entity) / cfg_hash
        self.store.read_manifest(
            cache_dir,
            cache_type="cf_profile",
            query_id=qid,
            anchor_source=anchor_source,
            telemetry_sha256=telemetry_sha256,
            cache_config_hash=cfg_hash,
        )
        payload = dict(self.store.read_json(cache_dir / "cf_profile.json") or {})
        return payload, {
            "cache_layer": "cf_profile",
            "cache_hit": True,
            "cache_dir": str(cache_dir),
            "entity": entity,
        }

    def write_cf_profile(
        self,
        *,
        telemetry_sha256: str,
        query: Any,
        anchor_timestamp: float,
        anchor_source: str,
        entity: str,
        degradation_scope: Sequence[str],
        graph: Any,
        config_payload: Dict[str, Any],
        profile: Dict[str, Any],
    ) -> Dict[str, Any]:
        qid, aid = self._ids(query, anchor_timestamp, anchor_source)
        cfg_hash = self.cf_config_hash(entity, degradation_scope, graph, config_payload)
        cache_dir = self._base_dir(query, qid, aid) / "cf" / safe_segment(entity) / cfg_hash
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.store.write_json(cache_dir / "cf_profile.json", profile)
        self.store.write_manifest(
            cache_dir,
            CacheManifest(
                cache_type="cf_profile",
                query_id=qid,
                anchor_timestamp=float(anchor_timestamp),
                anchor_source=anchor_source,
                telemetry_sha256=telemetry_sha256,
                cache_config_hash=cfg_hash,
                metadata={
                    "anchor_id": aid,
                    "entity": entity,
                    "degradation_scope": list(degradation_scope),
                    "builder_version": CF_BUILDER_VERSION,
                },
            ),
        )
        return {
            "cache_layer": "cf_profile",
            "cache_hit": False,
            "cache_write": True,
            "cache_dir": str(cache_dir),
            "entity": entity,
        }

    def cf_config_hash(
        self,
        entity: str,
        degradation_scope: Sequence[str],
        graph: Any,
        config_payload: Dict[str, Any],
    ) -> str:
        return cache_config_hash(
            {
                "builder": CF_BUILDER_VERSION,
                "entity": entity,
                "degradation_scope": list(degradation_scope),
                "graph": _graph_hash(graph),
                "config": config_payload,
            }
        )


def _array_hash(value: Any) -> str:
    try:
        arr = np.asarray(value, dtype=float)
        hasher = hashlib.sha256()
        hasher.update(str(arr.shape).encode("ascii"))
        hasher.update(np.round(arr, 8).tobytes())
        return hasher.hexdigest()
    except Exception:
        return stable_hash(str(value))


def _graph_hash(graph: Any) -> str:
    if isinstance(graph, dict):
        payload = {
            str(src): {str(dst): round(float(weight), 8) for dst, weight in sorted((children or {}).items())}
            for src, children in sorted(graph.items())
        }
        return stable_hash(payload)
    return _array_hash(graph)
