"""L0 normalized telemetry cache."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd

from ..config import UnifiedTelemetry
from .key import file_tree_sha256, safe_segment, telemetry_sha256
from .manifest import CacheManifest
from .store import CacheMiss, CacheStore, CacheValidationError


TELEMETRY_NORMALIZER_VERSION = "telemetry_normalizer.v1"


class TelemetryCache:
    def __init__(self, store: CacheStore) -> None:
        self.store = store

    def cache_dir(self, system: str, sub_system: str, date_str: str, raw_sha256: str) -> Path:
        return (
            self.store.layer_dir("telemetry")
            / safe_segment(system)
            / safe_segment(sub_system or "default")
            / safe_segment(date_str)
            / raw_sha256[:24]
        )

    def telemetry_root(self, loader: Any, date_str: str, sub_system: str = "") -> Path:
        if sub_system:
            return Path(loader.root) / sub_system / "telemetry" / date_str
        return Path(loader.root) / "telemetry" / date_str

    def raw_sha256(self, loader: Any, date_str: str, sub_system: str = "") -> str:
        return file_tree_sha256(self.telemetry_root(loader, date_str, sub_system), suffixes=(".csv",))

    def load_or_build(self, loader: Any, date_str: str, sub_system: str = "") -> Tuple[UnifiedTelemetry, Dict[str, Any]]:
        raw_sha = self.raw_sha256(loader, date_str, sub_system)
        cache_dir = self.cache_dir(loader.system_name, sub_system, date_str, raw_sha)
        config_hash = self.store.metadata_hash(
            {
                "normalizer": TELEMETRY_NORMALIZER_VERSION,
                "system": loader.system_name,
                "sub_system": sub_system or "default",
                "date": date_str,
            }
        )
        try:
            self.store.read_manifest(
                cache_dir,
                cache_type="telemetry",
                telemetry_sha256=raw_sha,
                cache_config_hash=config_hash,
            )
            telemetry = self._read(cache_dir)
            normalized_sha = str((self.store.read_manifest(cache_dir, cache_type="telemetry").metadata or {}).get("normalized_telemetry_sha256", ""))
            if normalized_sha:
                setattr(telemetry, "_cache_normalized_sha256", normalized_sha)
            debug = {
                "cache_layer": "telemetry",
                "cache_hit": True,
                "cache_dir": str(cache_dir),
                "telemetry_sha256": raw_sha,
                "normalized_telemetry_sha256": normalized_sha or telemetry_sha256(telemetry),
            }
            return telemetry, debug
        except CacheMiss:
            pass
        except CacheValidationError:
            if self.store.strict:
                raise

        telemetry = loader.load_telemetry(date_str, sub_system)
        normalized_sha = telemetry_sha256(telemetry)
        setattr(telemetry, "_cache_normalized_sha256", normalized_sha)
        self._write(
            cache_dir=cache_dir,
            telemetry=telemetry,
            raw_sha256=raw_sha,
            config_hash=config_hash,
            metadata={
                "system": loader.system_name,
                "sub_system": sub_system or "default",
                "telemetry_date": date_str,
                "normalizer_version": TELEMETRY_NORMALIZER_VERSION,
                "normalized_telemetry_sha256": normalized_sha,
            },
        )
        return telemetry, {
            "cache_layer": "telemetry",
            "cache_hit": False,
            "cache_write": True,
            "cache_dir": str(cache_dir),
            "telemetry_sha256": raw_sha,
            "normalized_telemetry_sha256": normalized_sha,
        }

    def _write(
        self,
        *,
        cache_dir: Path,
        telemetry: UnifiedTelemetry,
        raw_sha256: str,
        config_hash: str,
        metadata: Dict[str, Any],
    ) -> None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        self._write_frame(cache_dir / "metrics.parquet", telemetry.metrics)
        self._write_frame(cache_dir / "logs.parquet", telemetry.logs)
        self._write_frame(cache_dir / "traces.parquet", telemetry.traces)
        self.store.write_json(cache_dir / "entities.json", list(telemetry.entities or []))
        self.store.write_json(cache_dir / "entity_types.json", dict(telemetry.entity_types or {}))
        self.store.write_json(
            cache_dir / "frames.json",
            {
                "has_metrics": telemetry.metrics is not None,
                "has_logs": telemetry.logs is not None,
                "has_traces": telemetry.traces is not None,
            },
        )
        self.store.write_manifest(
            cache_dir,
            CacheManifest(
                cache_type="telemetry",
                telemetry_sha256=raw_sha256,
                cache_config_hash=config_hash,
                metadata=metadata,
            ),
        )

    def _read(self, cache_dir: Path) -> UnifiedTelemetry:
        frames = self.store.read_json(cache_dir / "frames.json")
        metrics = self._read_frame(cache_dir / "metrics.parquet") if frames.get("has_metrics") else None
        logs = self._read_frame(cache_dir / "logs.parquet") if frames.get("has_logs") else None
        traces = self._read_frame(cache_dir / "traces.parquet") if frames.get("has_traces") else None
        entities = list(self.store.read_json(cache_dir / "entities.json") or [])
        entity_types = dict(self.store.read_json(cache_dir / "entity_types.json") or {})
        system = str((self.store.read_manifest(cache_dir, cache_type="telemetry").metadata or {}).get("system", ""))
        return UnifiedTelemetry(
            metrics=metrics,
            logs=logs,
            traces=traces,
            entities=entities,
            entity_types=entity_types,
            system=system,
        )

    def _write_frame(self, path: Path, frame: Any) -> None:
        if frame is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)

    def _read_frame(self, path: Path) -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame()
        return pd.read_parquet(path)
