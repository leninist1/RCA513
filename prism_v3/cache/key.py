"""Stable cache key helpers."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import pandas as pd


_TELEMETRY_SHA_CACHE: Dict[Any, str] = {}


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), default=str)


def stable_hash(value: Any) -> str:
    payload = stable_json(value).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def short_hash(value: Any, length: int = 16) -> str:
    return stable_hash(value)[: max(8, int(length))]


def safe_segment(value: Any, fallback: str = "default") -> str:
    text = str(value or "").strip()
    if not text:
        text = fallback
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    text = text.strip("._-")
    return text or fallback


def query_id(system: str, sub_system: str, telemetry_date: str, task_index: str, query_index: Any = None) -> str:
    idx = "" if query_index is None else f":q{query_index}"
    sub = sub_system or "default"
    return f"{system}:{sub}:{telemetry_date}:{task_index}{idx}"


def anchor_id(anchor_timestamp: float, anchor_source: str, extra: Optional[Dict[str, Any]] = None) -> str:
    payload: Dict[str, Any] = {
        "anchor_timestamp": round(float(anchor_timestamp), 6),
        "anchor_source": str(anchor_source or ""),
    }
    if extra:
        payload.update(extra)
    return short_hash(payload)


def cache_config_hash(payload: Dict[str, Any]) -> str:
    return short_hash(payload, length=24)


def dataframe_sha256(frame: Optional[pd.DataFrame]) -> str:
    if frame is None:
        return "none"
    if frame.empty:
        return stable_hash({"columns": list(frame.columns), "shape": list(frame.shape), "empty": True})
    normalized = frame.copy()
    normalized = normalized.reindex(sorted(normalized.columns), axis=1)
    try:
        row_hashes = pd.util.hash_pandas_object(normalized, index=True).to_numpy(dtype="uint64")
        hasher = hashlib.sha256()
        hasher.update(stable_json({"columns": list(normalized.columns), "shape": list(normalized.shape)}).encode("utf-8"))
        hasher.update(row_hashes.tobytes())
        return hasher.hexdigest()
    except Exception:
        return stable_hash(
            {
                "columns": list(normalized.columns),
                "shape": list(normalized.shape),
                "records": normalized.astype(str).to_dict(orient="records"),
            }
        )


def dataframe_content_sha256(frame: Optional[pd.DataFrame]) -> str:
    if frame is None:
        return "none"
    return dataframe_sha256(frame.reset_index(drop=True))


def telemetry_sha256(telemetry: Any) -> str:
    cached = getattr(telemetry, "_cache_normalized_sha256", None)
    if cached:
        return str(cached)
    key = (
        id(getattr(telemetry, "metrics", None)),
        id(getattr(telemetry, "logs", None)),
        id(getattr(telemetry, "traces", None)),
        tuple(getattr(getattr(telemetry, "metrics", None), "shape", ()) or ()),
        tuple(getattr(getattr(telemetry, "logs", None), "shape", ()) or ()),
        tuple(getattr(getattr(telemetry, "traces", None), "shape", ()) or ()),
    )
    if key in _TELEMETRY_SHA_CACHE:
        return _TELEMETRY_SHA_CACHE[key]
    payload = {
        "system": getattr(telemetry, "system", ""),
        "entities": list(getattr(telemetry, "entities", []) or []),
        "entity_types": dict(getattr(telemetry, "entity_types", {}) or {}),
        "metrics": dataframe_sha256(getattr(telemetry, "metrics", None)),
        "logs": dataframe_sha256(getattr(telemetry, "logs", None)),
        "traces": dataframe_sha256(getattr(telemetry, "traces", None)),
    }
    digest = stable_hash(payload)
    _TELEMETRY_SHA_CACHE[key] = digest
    try:
        setattr(telemetry, "_cache_normalized_sha256", digest)
    except Exception:
        pass
    return digest


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def file_tree_sha256(root: Path, suffixes: Iterable[str] = (".csv", ".json", ".parquet", ".npy")) -> str:
    if not root.exists():
        return stable_hash({"missing": str(root)})
    suffix_set = {item.lower() for item in suffixes}
    hasher = hashlib.sha256()
    files = [
        path
        for path in root.rglob("*")
        if path.is_file() and (not suffix_set or path.suffix.lower() in suffix_set)
    ]
    for path in sorted(files, key=lambda item: item.relative_to(root).as_posix()):
        rel = path.relative_to(root).as_posix()
        hasher.update(rel.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(file_sha256(path).encode("ascii"))
        hasher.update(b"\0")
    return hasher.hexdigest()
