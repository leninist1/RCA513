"""Executable rule schema for Scheme B."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Mapping


def normalize_reason_bucket(reason: str) -> str:
    key = str(reason).strip().lower()
    mapping = {
        "high cpu usage": "cpu",
        "high jvm cpu load": "cpu",
        "cpu": "cpu",
        "high memory usage": "memory",
        "jvm out of memory (oom) heap": "jvm_oom",
        "oom": "jvm_oom",
        "memory": "memory",
        "high disk i/o read usage": "disk_io",
        "high disk io read usage": "disk_io",
        "disk": "disk_io",
        "high disk space usage": "filesystem",
        "filesystem": "filesystem",
        "file system": "filesystem",
        "network latency": "network_latency",
        "latency": "network_latency",
        "network packet loss": "network_packet_loss",
        "packet loss": "network_packet_loss",
        "network": "network",
    }
    return mapping.get(key, key.replace(" ", "_"))


@dataclass(frozen=True)
class Candidate:
    service: str
    reason: str

    @property
    def reason_bucket(self) -> str:
        return normalize_reason_bucket(self.reason)

    def to_dict(self) -> dict[str, str]:
        return {"service": self.service, "reason": self.reason, "reason_bucket": self.reason_bucket}


@dataclass(frozen=True)
class RefutationRule:
    id: str
    reason_buckets: tuple[str, ...]
    evidence_query: str
    query_args: Mapping[str, Any] = field(default_factory=dict)
    enabled: bool = True
    reason_names: tuple[str, ...] = ()
    candidate_scope: str = "service"
    rule_type: str = "soft"
    support_if: str = "matched"
    refute_if: str = ""
    missing_policy: str = "blind"
    confidence: float = 1.0
    group: str = "round_1_cheap_metric"
    description: str = ""

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RefutationRule":
        return cls(
            id=str(data["id"]),
            enabled=bool(data.get("enabled", True)),
            reason_buckets=tuple(str(x) for x in data.get("reason_buckets", [])),
            reason_names=tuple(str(x) for x in data.get("reason_names", [])),
            candidate_scope=str(data.get("candidate_scope", "service")),
            evidence_query=str(data["evidence_query"]),
            query_args=dict(data.get("query_args", {})),
            rule_type=str(data.get("rule_type", "soft")),
            support_if=str(data.get("support_if", "matched")),
            refute_if=str(data.get("refute_if", "")),
            missing_policy=str(data.get("missing_policy", "blind")),
            confidence=float(data.get("confidence", 1.0)),
            group=str(data.get("group", "round_1_cheap_metric")),
            description=str(data.get("description", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "enabled": self.enabled,
            "reason_buckets": list(self.reason_buckets),
            "reason_names": list(self.reason_names),
            "candidate_scope": self.candidate_scope,
            "evidence_query": self.evidence_query,
            "query_args": dict(self.query_args),
            "rule_type": self.rule_type,
            "support_if": self.support_if,
            "refute_if": self.refute_if,
            "missing_policy": self.missing_policy,
            "confidence": self.confidence,
            "group": self.group,
            "description": self.description,
        }

    def applies_to(self, candidate: Candidate) -> bool:
        if not self.enabled:
            return False
        if self.reason_names and candidate.reason not in self.reason_names:
            return False
        return candidate.reason_bucket in self.reason_buckets


@dataclass(frozen=True)
class RuleSet:
    rules: tuple[RefutationRule, ...]
    version: int = 2

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RuleSet":
        return cls(
            version=int(data.get("version", 2)),
            rules=tuple(RefutationRule.from_dict(row) for row in data.get("rules", [])),
        )

    @classmethod
    def load_json(cls, path: str | Path) -> "RuleSet":
        with Path(path).open("r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def to_dict(self) -> dict[str, Any]:
        return {"version": self.version, "rules": [rule.to_dict() for rule in self.rules]}

    def save_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2, sort_keys=True)

    def applicable(self, candidate: Candidate, groups: set[str] | None = None) -> list[RefutationRule]:
        out = []
        for rule in self.rules:
            if groups is not None and rule.group not in groups:
                continue
            if rule.applies_to(candidate):
                out.append(rule)
        return out
