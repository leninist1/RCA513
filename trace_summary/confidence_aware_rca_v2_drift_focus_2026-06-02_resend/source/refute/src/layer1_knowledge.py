"""
layer1_knowledge.py -- Phase 4 knowledge builders.

This module builds lightweight, portable knowledge artifacts:
- refutation rules: evidence requirements per reason bucket
- fault clusters: historical reason -> likely component priors

The implementation deliberately avoids dataset-specific hardcoding beyond the
reason text normalization already centralized in evidence_query.reason_to_bucket.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping

import pandas as pd

from refute.src.evidence_query import reason_to_bucket


@dataclass(frozen=True)
class RefutationRule:
    reason_bucket: str
    required_evidence: str
    confidence: float
    description: str

    def to_dict(self) -> dict:
        return {
            "reason_bucket": self.reason_bucket,
            "required_evidence": self.required_evidence,
            "confidence": self.confidence,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "RefutationRule":
        return cls(
            reason_bucket=str(data["reason_bucket"]),
            required_evidence=str(data["required_evidence"]),
            confidence=float(data["confidence"]),
            description=str(data["description"]),
        )


DEFAULT_RULES = [
    RefutationRule("cpu", "container_kpi", 1.0, "CPU roots require local container CPU evidence."),
    RefutationRule("memory", "container_kpi_or_log", 1.0, "Memory/OOM roots require memory KPI or fatal log evidence."),
    RefutationRule("network", "container_or_node_kpi", 1.0, "Network roots can be container network or node network evidence."),
    RefutationRule("disk", "node_kpi", 1.0, "Disk I/O roots are often node-level OSLinux disk evidence in Bank."),
    RefutationRule("filesystem", "node_kpi", 1.0, "Disk space roots are filesystem/node evidence."),
]


def build_refutation_rules() -> dict:
    return {
        "version": 1,
        "rules": [rule.to_dict() for rule in DEFAULT_RULES],
    }


def load_refutation_rules(path: str | Path) -> List[RefutationRule]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    return [RefutationRule.from_dict(row) for row in data.get("rules", [])]


def save_json(data: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)


def build_fault_clusters_from_records(
    records: pd.DataFrame,
    known_services: Iterable[str],
    top_k: int = 5,
) -> dict:
    required = {"component", "reason"}
    missing = required - set(records.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")

    known = set(str(s) for s in known_services)
    clusters: Dict[str, dict] = {}
    for reason, group in records.groupby("reason", sort=True):
        bucket = reason_to_bucket(str(reason))
        counts = (
            group["component"].astype(str)
            .loc[lambda s: s.isin(known)]
            .value_counts()
        )
        candidates = counts.head(top_k).index.tolist()
        clusters[str(reason)] = {
            "reason": str(reason),
            "reason_bucket": bucket,
            "n_cases": int(len(group)),
            "component_counts": {str(k): int(v) for k, v in counts.items()},
            "candidate_services": candidates,
        }
    return {"version": 1, "clusters": clusters}


def candidate_services_for_reason(
    fault_clusters: Mapping[str, object],
    reason: str,
    fallback_services: Iterable[str],
    min_candidates: int = 3,
) -> List[str]:
    fallback = list(dict.fromkeys(str(s) for s in fallback_services))
    cluster = fault_clusters.get("clusters", {}).get(str(reason), {})
    candidates = list(cluster.get("candidate_services", []))
    merged = list(dict.fromkeys(candidates + fallback))
    if len(merged) < min_candidates:
        return fallback
    return merged
