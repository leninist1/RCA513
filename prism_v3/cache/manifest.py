"""Cache manifests and strict validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


ALLOWED_ANCHOR_SOURCES = {
    "public_query_window",
    "telemetry_unsupervised_onset",
    "fallback_query_window_start",
    "query_window_start_fallback",
}


class CacheManifestError(RuntimeError):
    pass


@dataclass
class CacheManifest:
    cache_type: str
    query_id: str = ""
    anchor_timestamp: Optional[float] = None
    anchor_source: str = ""
    telemetry_sha256: str = ""
    pipeline_version: str = ""
    cache_config_hash: str = ""
    payload_sha256: str = ""
    generated_without_gt: bool = True
    allowed_anchor_sources: List[str] = field(default_factory=lambda: sorted(ALLOWED_ANCHOR_SOURCES))
    git_commit: str = ""
    created_at: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        created_at = self.created_at or datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        return {
            "cache_type": self.cache_type,
            "generated_without_gt": bool(self.generated_without_gt),
            "allowed_anchor_sources": list(self.allowed_anchor_sources),
            "query_id": self.query_id,
            "anchor_timestamp": self.anchor_timestamp,
            "anchor_source": self.anchor_source,
            "telemetry_sha256": self.telemetry_sha256,
            "pipeline_version": self.pipeline_version,
            "cache_config_hash": self.cache_config_hash,
            "git_commit": self.git_commit,
            "created_at": created_at,
            "payload_sha256": self.payload_sha256,
            "metadata": dict(self.metadata or {}),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "CacheManifest":
        return cls(
            cache_type=str(payload.get("cache_type", "")),
            query_id=str(payload.get("query_id", "")),
            anchor_timestamp=payload.get("anchor_timestamp"),
            anchor_source=str(payload.get("anchor_source", "")),
            telemetry_sha256=str(payload.get("telemetry_sha256", "")),
            pipeline_version=str(payload.get("pipeline_version", "")),
            cache_config_hash=str(payload.get("cache_config_hash", "")),
            payload_sha256=str(payload.get("payload_sha256", "")),
            generated_without_gt=bool(payload.get("generated_without_gt", False)),
            allowed_anchor_sources=list(payload.get("allowed_anchor_sources", [])),
            git_commit=str(payload.get("git_commit", "")),
            created_at=str(payload.get("created_at", "")),
            metadata=dict(payload.get("metadata", {}) or {}),
        )

    def validate(
        self,
        *,
        cache_type: str,
        payload_sha256: str,
        pipeline_version: str,
        query_id: str = "",
        anchor_source: str = "",
        telemetry_sha256: str = "",
        cache_config_hash: str = "",
    ) -> List[str]:
        errors: List[str] = []
        if self.cache_type != cache_type:
            errors.append(f"cache_type mismatch: {self.cache_type} != {cache_type}")
        if not self.generated_without_gt:
            errors.append("generated_without_gt is false")
        allowed = set(self.allowed_anchor_sources or [])
        if self.anchor_source and self.anchor_source not in allowed:
            errors.append(f"anchor_source not allowed by manifest: {self.anchor_source}")
        if anchor_source and self.anchor_source and self.anchor_source != anchor_source:
            errors.append(f"anchor_source mismatch: {self.anchor_source} != {anchor_source}")
        if self.anchor_source and self.anchor_source not in ALLOWED_ANCHOR_SOURCES:
            errors.append(f"anchor_source is not globally allowed: {self.anchor_source}")
        if payload_sha256 and self.payload_sha256 != payload_sha256:
            errors.append("payload sha256 mismatch")
        if pipeline_version and self.pipeline_version != pipeline_version:
            errors.append(f"pipeline_version mismatch: {self.pipeline_version} != {pipeline_version}")
        if query_id and self.query_id and self.query_id != query_id:
            errors.append(f"query_id mismatch: {self.query_id} != {query_id}")
        if telemetry_sha256 and self.telemetry_sha256 and self.telemetry_sha256 != telemetry_sha256:
            errors.append("telemetry_sha256 mismatch")
        if cache_config_hash and self.cache_config_hash and self.cache_config_hash != cache_config_hash:
            errors.append("cache_config_hash mismatch")
        return errors
