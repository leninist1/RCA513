"""Shared d32 schemas."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


REASON_BUCKETS = {
    "high cpu usage": "cpu",
    "high jvm cpu load": "cpu",
    "high memory usage": "memory",
    "jvm out of memory (oom) heap": "jvm_oom",
    "high disk i/o read usage": "disk_io",
    "high disk io read usage": "disk_io",
    "high disk space usage": "filesystem",
    "network latency": "network_latency",
    "network packet loss": "network_packet_loss",
}


def reason_bucket(reason: str) -> str:
    return REASON_BUCKETS.get(str(reason).strip().lower(), str(reason).strip().lower().replace(" ", "_"))


@dataclass(frozen=True)
class RootCandidate:
    component: str
    reason: str
    prior: float = 0.0
    source: str = "unknown"
    details: Mapping[str, Any] = field(default_factory=dict)

    @property
    def reason_bucket(self) -> str:
        return reason_bucket(self.reason)

    def key(self) -> tuple[str, str]:
        return (self.component, self.reason)

    def to_dict(self) -> dict[str, Any]:
        return {
            "component": self.component,
            "reason": self.reason,
            "reason_bucket": self.reason_bucket,
            "prior": self.prior,
            "source": self.source,
            "details": dict(self.details),
        }


@dataclass(frozen=True)
class EvidenceCard:
    rule_id: str
    polarity: str
    strength: float
    text: str
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "polarity": self.polarity,
            "strength": self.strength,
            "text": self.text,
            "details": dict(self.details),
        }


@dataclass(frozen=True)
class RefutationDecision:
    candidate: RootCandidate
    rebuttal_score: float
    support_strength: float
    refute_strength: float
    blind_count: int
    cards: tuple[EvidenceCard, ...]
    confidence: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate.to_dict(),
            "rebuttal_score": self.rebuttal_score,
            "support_strength": self.support_strength,
            "refute_strength": self.refute_strength,
            "blind_count": self.blind_count,
            "confidence": self.confidence,
            "cards": [card.to_dict() for card in self.cards],
        }


@dataclass(frozen=True)
class D32Result:
    prediction: dict[str, dict[str, str]]
    high_suspicion: tuple[RefutationDecision, ...]
    low_suspicion: tuple[RefutationDecision, ...]
    data_blind_spots: tuple[dict[str, Any], ...]
    debug: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "prediction": self.prediction,
            "high_suspicion": [row.to_dict() for row in self.high_suspicion],
            "low_suspicion": [row.to_dict() for row in self.low_suspicion],
            "data_blind_spots": [dict(row) for row in self.data_blind_spots],
            "debug": dict(self.debug),
        }

