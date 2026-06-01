"""Hierarchical Bayesian prior: fault category → sub-category → entity.

Replaces the flat REASON_TAXONOMY with a three-level conditional structure:
  Level 0: System prototype → category blend
  Level 1: Fault category (Resource/Network/Application/Database)
  Level 2: Fault sub-category (CPU/Memory/Disk/Latency/...)
  Level 3: Entity profile (per-entity historical failure tendency)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple
import math

import numpy as np


class FaultCategory(Enum):
    RESOURCE = "Resource"
    NETWORK = "Network"
    APPLICATION = "Application"
    DATABASE = "Database"
    OTHER = "Other"


class FaultSubCategory(Enum):
    CPU = "cpu"
    MEMORY = "memory"
    DISK_IO = "disk_io"
    DISK_SPACE = "disk_space"
    NETWORK_LATENCY = "network_latency"
    NETWORK_PACKET_LOSS = "network_packet_loss"
    NETWORK_FAULT = "network_fault"
    JVM_OOM = "jvm_oom"
    PROCESS_KILL = "process_kill"
    HIGH_MEMORY = "high_memory"
    DB_FAULT = "db_fault"
    UNKNOWN = "unknown"


CATEGORY_TO_SUBCATEGORIES: Dict[FaultCategory, List[FaultSubCategory]] = {
    FaultCategory.RESOURCE: [
        FaultSubCategory.CPU,
        FaultSubCategory.MEMORY,
        FaultSubCategory.DISK_IO,
        FaultSubCategory.DISK_SPACE,
    ],
    FaultCategory.NETWORK: [
        FaultSubCategory.NETWORK_LATENCY,
        FaultSubCategory.NETWORK_PACKET_LOSS,
        FaultSubCategory.NETWORK_FAULT,
    ],
    FaultCategory.APPLICATION: [
        FaultSubCategory.JVM_OOM,
        FaultSubCategory.PROCESS_KILL,
        FaultSubCategory.HIGH_MEMORY,
    ],
    FaultCategory.DATABASE: [
        FaultSubCategory.DB_FAULT,
    ],
    FaultCategory.OTHER: [
        FaultSubCategory.UNKNOWN,
    ],
}

CATEGORY_SUBCAT_PRIOR: Dict[FaultCategory, Dict[FaultSubCategory, float]] = {
    FaultCategory.RESOURCE: {
        FaultSubCategory.CPU: 0.30,
        FaultSubCategory.MEMORY: 0.35,
        FaultSubCategory.DISK_IO: 0.20,
        FaultSubCategory.DISK_SPACE: 0.15,
    },
    FaultCategory.NETWORK: {
        FaultSubCategory.NETWORK_LATENCY: 0.35,
        FaultSubCategory.NETWORK_PACKET_LOSS: 0.25,
        FaultSubCategory.NETWORK_FAULT: 0.40,
    },
    FaultCategory.APPLICATION: {
        FaultSubCategory.JVM_OOM: 0.45,
        FaultSubCategory.PROCESS_KILL: 0.30,
        FaultSubCategory.HIGH_MEMORY: 0.25,
    },
    FaultCategory.DATABASE: {
        FaultSubCategory.DB_FAULT: 1.0,
    },
    FaultCategory.OTHER: {
        FaultSubCategory.UNKNOWN: 1.0,
    },
}

SUBCAT_TO_METRIC_FAMILIES: Dict[FaultSubCategory, List[str]] = {
    FaultSubCategory.CPU: ["cpu"],
    FaultSubCategory.MEMORY: ["memory"],
    FaultSubCategory.DISK_IO: ["disk_io"],
    FaultSubCategory.DISK_SPACE: ["disk_space"],
    FaultSubCategory.NETWORK_LATENCY: ["network"],
    FaultSubCategory.NETWORK_PACKET_LOSS: ["network"],
    FaultSubCategory.NETWORK_FAULT: ["network"],
    FaultSubCategory.JVM_OOM: ["jvm"],
    FaultSubCategory.PROCESS_KILL: ["process"],
    FaultSubCategory.HIGH_MEMORY: ["memory"],
    FaultSubCategory.DB_FAULT: ["db"],
    FaultSubCategory.UNKNOWN: [],
}

SUBCAT_TO_REASON_LABEL: Dict[FaultSubCategory, str] = {
    FaultSubCategory.CPU: "CPU fault",
    FaultSubCategory.MEMORY: "high memory usage",
    FaultSubCategory.DISK_IO: "disk I/O consumption",
    FaultSubCategory.DISK_SPACE: "disk I/O consumption",
    FaultSubCategory.NETWORK_LATENCY: "network latency",
    FaultSubCategory.NETWORK_PACKET_LOSS: "network packet loss",
    FaultSubCategory.NETWORK_FAULT: "network fault",
    FaultSubCategory.JVM_OOM: "JVM Out of Memory",
    FaultSubCategory.PROCESS_KILL: "process termination",
    FaultSubCategory.HIGH_MEMORY: "high memory usage",
    FaultSubCategory.DB_FAULT: "db fault",
    FaultSubCategory.UNKNOWN: "unknown",
}

SUBCAT_TO_KEYWORDS: Dict[FaultSubCategory, List[str]] = {
    FaultSubCategory.CPU: ["cpu", "load", "throttle", "utilization", "busy"],
    FaultSubCategory.MEMORY: ["memory", "mem", "rss", "resident", "swap"],
    FaultSubCategory.DISK_IO: ["disk", "iops", "iowait", "filesystem", "storage", "io"],
    FaultSubCategory.DISK_SPACE: ["disk", "pct_usage", "free", "used", "fscapacity"],
    FaultSubCategory.NETWORK_LATENCY: ["latency", "timeout", "slow", "delay", "rtt"],
    FaultSubCategory.NETWORK_PACKET_LOSS: [
        "packet",
        "loss",
        "drop",
        "dropped",
        "retransmit",
    ],
    FaultSubCategory.NETWORK_FAULT: [
        "network",
        "connection",
        "refused",
        "unreachable",
        "reset",
    ],
    FaultSubCategory.JVM_OOM: [
        "oom",
        "outofmemory",
        "heap",
        "memoryerror",
        "killed",
        "evicted",
    ],
    FaultSubCategory.PROCESS_KILL: ["sigkill", "killed", "terminated", "crash", "exit"],
    FaultSubCategory.HIGH_MEMORY: ["memory", "mem", "rss", "resident", "swap"],
    FaultSubCategory.DB_FAULT: [
        "db",
        "jdbc",
        "mysql",
        "postgres",
        "redis",
        "sql",
        "database",
    ],
    FaultSubCategory.UNKNOWN: [],
}

EPS = 1e-9


@dataclass
class HierarchicalPriorState:
    cat_probs: np.ndarray
    subcat_probs: Dict[FaultCategory, Dict[FaultSubCategory, float]]
    entity_prior_boosts: Dict[str, float]


class HierarchicalPrior:
    """Three-level hierarchical prior for root cause reasoning."""

    def __init__(self):
        self._cat_probs: np.ndarray = np.ones(len(FaultCategory), dtype=float) / len(
            FaultCategory
        )

    def set_system_category_prior(self, cat_prior: np.ndarray):
        self._cat_probs = np.clip(cat_prior, EPS, None)
        self._cat_probs /= self._cat_probs.sum()

    def infer_subcategory_from_metrics(
        self,
        metric_detail: Dict[str, Dict[str, float]],
        entities: List[str],
        top_k: int = 8,
    ) -> Dict[FaultCategory, Dict[FaultSubCategory, float]]:
        subcat_scores: Dict[FaultSubCategory, float] = {
            sc: 0.0 for sc in FaultSubCategory
        }

        all_metric_names: List[str] = []
        for entity in entities:
            all_metric_names.extend(metric_detail.get(entity, {}).keys())

        metric_text = " ".join(all_metric_names).lower()
        for sc, keywords in SUBCAT_TO_KEYWORDS.items():
            hits = sum(1 for kw in keywords if kw in metric_text)
            if keywords:
                subcat_scores[sc] = hits / len(keywords)

        cat_subcats: Dict[FaultCategory, Dict[FaultSubCategory, float]] = {}
        for cat in FaultCategory:
            cat_subcats[cat] = {}
            total = 0.0
            for sc in CATEGORY_TO_SUBCATEGORIES.get(cat, []):
                base_prior = CATEGORY_SUBCAT_PRIOR.get(cat, {}).get(sc, 0.1)
                evidence_score = subcat_scores.get(sc, 0.0)
                blended = 0.5 * base_prior + 0.5 * evidence_score
                cat_subcats[cat][sc] = blended
                total += blended
            if total > EPS:
                for sc in cat_subcats[cat]:
                    cat_subcats[cat][sc] /= total
        return cat_subcats

    def infer_subcategory_from_logs(
        self,
        log_detail: Dict[str, Dict[str, any]],
        entities: List[str],
    ) -> Dict[FaultCategory, Dict[FaultSubCategory, float]]:
        subcat_scores: Dict[FaultSubCategory, float] = {
            sc: 0.0 for sc in FaultSubCategory
        }

        log_text_parts = []
        for entity in entities:
            info = log_detail.get(entity, {})
            if info:
                log_text_parts.append(str(info.get("text", ""))[:300])
        log_text = " ".join(log_text_parts).lower()

        for sc, keywords in SUBCAT_TO_KEYWORDS.items():
            hits = sum(1 for kw in keywords if kw in log_text)
            if keywords:
                subcat_scores[sc] = hits / len(keywords)

        cat_subcats: Dict[FaultCategory, Dict[FaultSubCategory, float]] = {}
        for cat in FaultCategory:
            cat_subcats[cat] = {}
            total = 0.0
            for sc in CATEGORY_TO_SUBCATEGORIES.get(cat, []):
                base_prior = CATEGORY_SUBCAT_PRIOR.get(cat, {}).get(sc, 0.1)
                evidence_score = subcat_scores.get(sc, 0.0)
                blended = 0.5 * base_prior + 0.5 * evidence_score
                cat_subcats[cat][sc] = blended
                total += blended
            if total > EPS:
                for sc in cat_subcats[cat]:
                    cat_subcats[cat][sc] /= total
        return cat_subcats

    def merge_subcategory_posteriors(
        self,
        metric_inferred: Dict[FaultCategory, Dict[FaultSubCategory, float]],
        log_inferred: Dict[FaultCategory, Dict[FaultSubCategory, float]],
        has_logs: bool = True,
    ) -> Dict[FaultCategory, Dict[FaultSubCategory, float]]:
        w_metric = 0.6 if has_logs else 1.0
        w_log = 0.4 if has_logs else 0.0
        merged: Dict[FaultCategory, Dict[FaultSubCategory, float]] = {}
        for cat in FaultCategory:
            merged[cat] = {}
            for sc in CATEGORY_TO_SUBCATEGORIES.get(cat, []):
                m_val = metric_inferred.get(cat, {}).get(sc, 0.0)
                l_val = log_inferred.get(cat, {}).get(sc, 0.0)
                merged[cat][sc] = w_metric * m_val + w_log * l_val
            total = sum(merged[cat].values())
            if total > EPS:
                for sc in merged[cat]:
                    merged[cat][sc] /= total
        return merged

    def compute_entity_category_affinity(
        self,
        entity: str,
        subcat_posterior: Dict[FaultCategory, Dict[FaultSubCategory, float]],
        entity_profile: Dict[FaultSubCategory, int],
    ) -> float:
        affinity = 0.0
        total_weight = 0.0
        for cat in FaultCategory:
            cat_weight = self._cat_probs[list(FaultCategory).index(cat)]
            for sc, prob in subcat_posterior.get(cat, {}).items():
                profile_count = entity_profile.get(sc, 0)
                profile_boost = (
                    1.0
                    + 0.3
                    * min(profile_count / max(1, sum(entity_profile.values())), 1.0)
                    if sum(entity_profile.values()) >= 3
                    else 1.0
                )
                affinity += cat_weight * prob * profile_boost
                total_weight += cat_weight * prob
        if total_weight < EPS:
            return 1.0
        return affinity / total_weight

    def compute_entity_prior_boosts(
        self,
        entities: List[str],
        subcat_posterior: Dict[FaultCategory, Dict[FaultSubCategory, float]],
        entity_profile_store: "EntityProfileStore",
    ) -> Dict[str, float]:
        boosts = {}
        for entity in entities:
            profile = entity_profile_store.get_profile(entity)
            profile_map = {
                FaultSubCategory(sc_name): cnt for sc_name, cnt in profile.items()
            }
            boost = self.compute_entity_category_affinity(
                entity, subcat_posterior, profile_map
            )
            boosts[entity] = boost
        return boosts
