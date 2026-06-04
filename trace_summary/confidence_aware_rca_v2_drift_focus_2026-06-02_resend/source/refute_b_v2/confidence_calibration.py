"""Runtime confidence calibration helpers.

The calibration scripts produce reports; this module is the runtime counterpart
that can load those reports and assign labels during answer selection.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping

from refute_b_v2.rules import normalize_reason_bucket


def evidence_state(features: Mapping) -> str:
    support = float(features.get("support_strength", 0.0) or 0.0)
    blind = int(features.get("blind_count", 0) or 0)
    if support <= 0:
        return "no_evidence"
    if blind > 0:
        return "partial_evidence"
    return "evidence"


def gap_bin(features: Mapping) -> str:
    gap = float(features.get("top1_top2_gap", 0.0) or 0.0)
    if gap >= 10:
        return "gap_high"
    if gap >= 3:
        return "gap_mid"
    if gap > 0:
        return "gap_low"
    return "gap_none"


def confidence_bucket_key(reason: str, features: Mapping) -> str:
    return "|".join([normalize_reason_bucket(reason), evidence_state(features), gap_bin(features)])


@dataclass(frozen=True)
class RuntimeConfidence:
    label: str
    source: str
    bucket_key: str
    training_bucket: Mapping | None = None

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "source": self.source,
            "bucket_key": self.bucket_key,
            "training_bucket": dict(self.training_bucket) if self.training_bucket else None,
        }


@dataclass
class RuntimeConfidenceCalibrator:
    table: dict[str, Mapping]
    min_n: int = 5
    source: str = "runtime_table"

    @classmethod
    def from_json(cls, path: str | Path) -> "RuntimeConfidenceCalibrator":
        data = json.load(open(path, "r", encoding="utf-8"))
        if "buckets" in data:
            return cls(dict(data.get("buckets", {})), int(data.get("min_n", 5)), "calibration_report")
        if "overall_lodo" in data:
            # LODO reports aggregate by label, not bucket. They are useful for
            # reporting but not for per-case runtime lookup.
            return cls({}, int(data.get("min_n", 5)), "lodo_no_bucket_table")
        return cls({}, int(data.get("min_n", 5)), "unknown_report")

    def label_for(self, reason: str, features: Mapping, fallback_label: str = "MEDIUM") -> RuntimeConfidence:
        key = confidence_bucket_key(reason, features)
        row = self.table.get(key)
        if not row:
            return RuntimeConfidence(fallback_label.upper(), "fallback_semantic", key, None)
        if int(row.get("n", 0) or 0) < self.min_n:
            return RuntimeConfidence("UNKNOWN", self.source, key, row)
        label = str(row.get("confidence_label") or fallback_label).upper()
        return RuntimeConfidence(label, self.source, key, row)
