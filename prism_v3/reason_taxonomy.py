"""System-aware reason taxonomy normalization for PRISM v3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple
import re


CANONICAL_REASONS = {
    "cpu": "CPU fault",
    "memory": "high memory usage",
    "disk": "disk I/O consumption",
    "disk_space": "disk space consumption",
    "network": "network fault",
    "latency": "network latency",
    "db": "db fault",
    "process": "process termination",
    "jvm": "high memory usage",
    "change": "configuration change",
    "unknown": "high memory usage",
}

KEYWORDS = {
    "cpu": ("cpu", "load", "throttle", "util", "proc_user"),
    "memory": ("memory", "mem", "oom", "heap", "gc", "allocation failure", "nocachemem"),
    "disk": ("disk", "io", "i/o", "await", "iops", "read", "write"),
    "disk_space": ("space", "capacity", "usage", "filesystem", "fsused"),
    "network": ("network", "packet", "drop", "retrans", "connection refused", "unavailable"),
    "latency": ("latency", "timeout", "slow", "duration", "elapsed"),
    "db": ("db", "mysql", "redis", "sql", "jdbc", "session", "tnsping"),
    "process": ("killed", "restart", "terminated", "sigkill", "process"),
    "change": ("deploy", "release", "config", "configuration", "change"),
}


@dataclass(frozen=True)
class ReasonCandidate:
    reason: str
    family: str
    score: float
    evidence: str = ""


def normalize_reason(reason: str, *, system: str = "") -> str:
    family, _score = reason_family(reason, system=system)
    return CANONICAL_REASONS.get(family, CANONICAL_REASONS["unknown"])


def reason_family(text: str, *, system: str = "") -> Tuple[str, float]:
    lower = str(text or "").lower()
    best_family = "unknown"
    best_score = 0.0
    for family, keywords in KEYWORDS.items():
        hits = sum(1 for keyword in keywords if keyword in lower)
        if hits:
            score = min(1.0, 0.35 * hits + 0.15)
            if score > best_score:
                best_family = family
                best_score = score
    return best_family, best_score


def reason_candidates_from_votes(votes: Dict[str, float], *, limit: int = 5) -> List[ReasonCandidate]:
    candidates = [
        ReasonCandidate(
            reason=CANONICAL_REASONS.get(str(family), CANONICAL_REASONS["unknown"]),
            family=str(family),
            score=float(score),
            evidence="reason_votes",
        )
        for family, score in votes.items()
        if float(score) > 0
    ]
    candidates.sort(key=lambda item: item.score, reverse=True)
    return candidates[: max(1, int(limit))]


def reason_candidates_from_texts(texts: Iterable[str], *, limit: int = 5) -> List[ReasonCandidate]:
    totals: Dict[str, float] = {}
    for text in texts:
        family, score = reason_family(text)
        totals[family] = max(totals.get(family, 0.0), score)
    return reason_candidates_from_votes(totals, limit=limit)
