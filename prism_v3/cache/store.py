"""Filesystem cache store with manifest and checksum validation."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from .key import file_sha256, stable_hash, stable_json
from .manifest import CacheManifest, CacheManifestError


PIPELINE_VERSION = "prism_v3_cache.full_cache.v1"


class CacheError(RuntimeError):
    pass


class CacheMiss(CacheError):
    pass


class CacheValidationError(CacheError, CacheManifestError):
    pass


def normalize_cache_root(path: str | Path) -> Path:
    root = Path(path).expanduser()
    if root.name in {"telemetry", "windows", "graphs", "features", "causal"}:
        return root.parent
    return root


class CacheStore:
    def __init__(
        self,
        root: str | Path,
        *,
        strict: bool = True,
        pipeline_version: str = PIPELINE_VERSION,
    ) -> None:
        self.root = normalize_cache_root(root)
        self.strict = bool(strict)
        self.pipeline_version = str(pipeline_version or PIPELINE_VERSION)

    def layer_dir(self, layer: str) -> Path:
        return self.root / layer

    def ensure_layer_dir(self, layer: str) -> Path:
        path = self.layer_dir(layer)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def manifest_path(self, cache_dir: Path) -> Path:
        return cache_dir / "manifest.json"

    def payload_sha256(self, cache_dir: Path, *, exclude: Iterable[str] = ("manifest.json",)) -> str:
        excluded = set(exclude)
        if not cache_dir.exists():
            return ""
        files = [
            path
            for path in cache_dir.rglob("*")
            if path.is_file() and path.relative_to(cache_dir).as_posix() not in excluded
        ]
        payload: Dict[str, str] = {}
        for path in sorted(files, key=lambda item: item.relative_to(cache_dir).as_posix()):
            payload[path.relative_to(cache_dir).as_posix()] = file_sha256(path)
        return stable_hash(payload)

    def read_manifest(
        self,
        cache_dir: Path,
        *,
        cache_type: str,
        query_id: str = "",
        anchor_source: str = "",
        telemetry_sha256: str = "",
        cache_config_hash: str = "",
    ) -> CacheManifest:
        path = self.manifest_path(cache_dir)
        if not path.exists():
            raise CacheMiss(f"manifest missing: {path}")
        try:
            with open(path, "r", encoding="utf-8") as f:
                manifest = CacheManifest.from_dict(json.load(f))
        except Exception as exc:
            raise CacheValidationError(f"manifest unreadable: {path}: {exc}") from exc
        actual_sha = self.payload_sha256(cache_dir)
        errors = manifest.validate(
            cache_type=cache_type,
            payload_sha256=actual_sha,
            pipeline_version=self.pipeline_version,
            query_id=query_id,
            anchor_source=anchor_source,
            telemetry_sha256=telemetry_sha256,
            cache_config_hash=cache_config_hash,
        )
        if errors:
            raise CacheValidationError("; ".join(errors))
        return manifest

    def write_manifest(self, cache_dir: Path, manifest: CacheManifest) -> None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        manifest.payload_sha256 = self.payload_sha256(cache_dir)
        manifest.pipeline_version = self.pipeline_version
        payload = manifest.to_dict()
        path = self.manifest_path(cache_dir)
        tmp = path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp, path)

    def write_json(self, path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True, indent=2, sort_keys=True, default=str)
            f.write("\n")
        os.replace(tmp, path)

    def read_json(self, path: Path) -> Any:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def cache_status(self, cache_dir: Path, cache_type: str, **kwargs: Any) -> Dict[str, Any]:
        if not cache_dir.exists():
            return {"hit": False, "reason": "missing"}
        try:
            manifest = self.read_manifest(cache_dir, cache_type=cache_type, **kwargs)
            return {"hit": True, "manifest": manifest.to_dict()}
        except CacheMiss as exc:
            return {"hit": False, "reason": str(exc)}
        except CacheValidationError as exc:
            if self.strict:
                raise
            return {"hit": False, "reason": "invalid", "error": str(exc)}

    def metadata_hash(self, payload: Dict[str, Any]) -> str:
        return stable_hash(payload)
