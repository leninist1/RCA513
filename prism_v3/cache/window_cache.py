"""L1 query window and signal matrix cache."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .key import anchor_id, cache_config_hash, query_id as make_query_id, safe_segment, telemetry_sha256
from .manifest import CacheManifest
from .store import CacheMiss, CacheStore, CacheValidationError


WINDOW_BUILDER_VERSION = "window_builder.public_query_window.v1"
SIGNAL_BUILDER_VERSION = "signal_matrix.dominant_metric_z.v1"


class WindowCache:
    def __init__(self, store: CacheStore) -> None:
        self.store = store

    def ids(
        self,
        query: Any,
        anchor_timestamp: float,
        anchor_source: str,
        baseline_window: int,
        fault_window: int,
    ) -> Tuple[str, str, str]:
        qid = make_query_id(
            getattr(query, "system", ""),
            getattr(query, "sub_system", ""),
            getattr(query, "telemetry_date", ""),
            getattr(query, "task_index", ""),
            getattr(query, "query_index", None),
        )
        aid = anchor_id(
            anchor_timestamp,
            anchor_source,
            {
                "baseline_window": int(baseline_window),
                "fault_window": int(fault_window),
                "window_builder": WINDOW_BUILDER_VERSION,
            },
        )
        cfg_hash = cache_config_hash(
            {
                "baseline_window": int(baseline_window),
                "fault_window": int(fault_window),
                "window_builder": WINDOW_BUILDER_VERSION,
                "signal_builder": SIGNAL_BUILDER_VERSION,
            }
        )
        return qid, aid, cfg_hash

    def cache_dir(self, query: Any, qid: str, aid: str, cfg_hash: str) -> Path:
        return (
            self.store.layer_dir("windows")
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
        baseline_window: int,
        fault_window: int,
        builder: Callable[[], Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]],
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        tsha = telemetry_sha256(telemetry)
        qid, aid, cfg_hash = self.ids(query, anchor_timestamp, anchor_source, baseline_window, fault_window)
        cache_dir = self.cache_dir(query, qid, aid, cfg_hash)
        try:
            self.store.read_manifest(
                cache_dir,
                cache_type="window_signal",
                query_id=qid,
                anchor_source=anchor_source,
                telemetry_sha256=tsha,
                cache_config_hash=cfg_hash,
            )
            baseline = self._read_frame(cache_dir / "baseline_metrics.parquet")
            fault = self._read_frame(cache_dir / "fault_metrics.parquet")
            debug = dict(self.store.read_json(cache_dir / "split_debug.json"))
            debug.update(
                {
                    "cache_layer": "window_signal",
                    "cache_hit": True,
                    "cache_dir": str(cache_dir),
                }
            )
            return baseline, fault, debug
        except CacheMiss:
            pass
        except CacheValidationError:
            if self.store.strict:
                raise

        baseline, fault, debug = builder()
        self.write(
            cache_dir=cache_dir,
            baseline_df=baseline,
            fault_df=fault,
            telemetry=telemetry,
            query=query,
            qid=qid,
            aid=aid,
            anchor_timestamp=anchor_timestamp,
            anchor_source=anchor_source,
            telemetry_sha=tsha,
            config_hash=cfg_hash,
            split_debug=debug,
        )
        out_debug = dict(debug)
        out_debug.update(
            {
                "cache_layer": "window_signal",
                "cache_hit": False,
                "cache_write": True,
                "cache_dir": str(cache_dir),
            }
        )
        return baseline, fault, out_debug

    def write(
        self,
        *,
        cache_dir: Path,
        baseline_df: pd.DataFrame,
        fault_df: pd.DataFrame,
        telemetry: Any,
        query: Any,
        qid: str,
        aid: str,
        anchor_timestamp: float,
        anchor_source: str,
        telemetry_sha: str,
        config_hash: str,
        split_debug: Dict[str, Any],
    ) -> None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        self._write_frame(cache_dir / "baseline_metrics.parquet", baseline_df)
        self._write_frame(cache_dir / "fault_metrics.parquet", fault_df)
        self._write_window_frames(cache_dir, telemetry, anchor_timestamp)
        self.store.write_json(cache_dir / "split_debug.json", dict(split_debug or {}))
        self._write_signal_matrix(cache_dir, baseline_df, fault_df, list(getattr(telemetry, "entities", []) or []))
        self.store.write_manifest(
            cache_dir,
            CacheManifest(
                cache_type="window_signal",
                query_id=qid,
                anchor_timestamp=float(anchor_timestamp),
                anchor_source=anchor_source,
                telemetry_sha256=telemetry_sha,
                cache_config_hash=config_hash,
                metadata={
                    "anchor_id": aid,
                    "task_index": getattr(query, "task_index", ""),
                    "window_builder_version": WINDOW_BUILDER_VERSION,
                    "signal_builder_version": SIGNAL_BUILDER_VERSION,
                    "baseline_rows": int(len(baseline_df)),
                    "fault_rows": int(len(fault_df)),
                },
            ),
        )

    def _write_window_frames(self, cache_dir: Path, telemetry: Any, anchor_timestamp: float) -> None:
        start = float(anchor_timestamp)
        baseline_start = start - 300.0
        fault_end = start + 300.0
        for name in ("logs", "traces"):
            frame = getattr(telemetry, name, None)
            if frame is None or getattr(frame, "empty", True) or "timestamp" not in frame.columns:
                continue
            baseline = frame[(frame["timestamp"] >= baseline_start) & (frame["timestamp"] < start)]
            fault = frame[(frame["timestamp"] >= start) & (frame["timestamp"] <= fault_end)]
            self._write_frame(cache_dir / f"baseline_{name}.parquet", baseline)
            self._write_frame(cache_dir / f"fault_{name}.parquet", fault)

    def _write_signal_matrix(
        self,
        cache_dir: Path,
        baseline_df: pd.DataFrame,
        fault_df: pd.DataFrame,
        entities: Sequence[str],
    ) -> None:
        try:
            from ..noise_native.cmi import _build_signal

            signal, baseline_mask, fault_mask = _build_signal(baseline_df, fault_df, entities)
        except Exception:
            signal = pd.DataFrame()
            baseline_mask = np.array([], dtype=bool)
            fault_mask = np.array([], dtype=bool)
        self._write_frame(cache_dir / "signal_matrix.parquet", signal)
        np.save(cache_dir / "baseline_mask.npy", baseline_mask)
        np.save(cache_dir / "fault_mask.npy", fault_mask)
        self.store.write_json(cache_dir / "entity_order.json", list(entities))

    def _write_frame(self, path: Path, frame: Optional[pd.DataFrame]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        out = frame if frame is not None else pd.DataFrame()
        out.to_parquet(path, index=False)

    def _read_frame(self, path: Path) -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame()
        return pd.read_parquet(path)
