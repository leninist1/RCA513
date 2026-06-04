"""Abstract evidence signatures for Scheme B."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from refute_b_v2.rule_engine import CandidateRuleResult, RuleEvidenceCard


def intensity_from_strength(strength: float) -> int:
    value = abs(float(strength or 0.0))
    if value <= 0:
        return 0
    if value < 3:
        return 1
    if value < 10:
        return 2
    return 3


@dataclass(frozen=True)
class SignatureEvidence:
    state: str
    intensity: int = 0
    scope: str = ""
    role: str = ""
    reason: str = ""
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out = {
            "state": self.state,
            "intensity": self.intensity,
        }
        if self.scope:
            out["scope"] = self.scope
        if self.role:
            out["role"] = self.role
        if self.reason:
            out["reason"] = self.reason
        if self.details:
            out["details"] = dict(self.details)
        return out


@dataclass(frozen=True)
class ServiceSignature:
    service: str
    metric: Mapping[str, SignatureEvidence] = field(default_factory=dict)
    log: Mapping[str, SignatureEvidence] = field(default_factory=dict)
    trace: Mapping[str, SignatureEvidence] = field(default_factory=dict)
    topology: Mapping[str, SignatureEvidence] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "service": self.service,
            "metric": {k: v.to_dict() for k, v in self.metric.items()},
            "log": {k: v.to_dict() for k, v in self.log.items()},
            "trace": {k: v.to_dict() for k, v in self.trace.items()},
            "topology": {k: v.to_dict() for k, v in self.topology.items()},
        }


@dataclass(frozen=True)
class CaseSignature:
    case_id: str
    modal_coverage: Mapping[str, str]
    services: tuple[ServiceSignature, ...]
    anomaly_distribution: Mapping[str, Any] = field(default_factory=dict)
    dominant_evidence_types: tuple[str, ...] = ()
    blind_spots: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "modal_coverage": dict(self.modal_coverage),
            "services": [svc.to_dict() for svc in self.services],
            "anomaly_distribution": dict(self.anomaly_distribution),
            "dominant_evidence_types": list(self.dominant_evidence_types),
            "blind_spots": list(self.blind_spots),
        }


def card_to_signature_evidence(card: RuleEvidenceCard) -> SignatureEvidence:
    state = card.polarity
    return SignatureEvidence(
        state=state,
        intensity=intensity_from_strength(card.strength),
        scope=str(card.details.get("scope", "")) if card.details else "",
        role=str(card.details.get("role", "")) if card.details else "",
        reason=str(card.details.get("reason", "")) if card.details else "",
        details=card.details,
    )


def service_signature_from_rule_result(result: CandidateRuleResult) -> ServiceSignature:
    metric: dict[str, SignatureEvidence] = {}
    log: dict[str, SignatureEvidence] = {}
    trace: dict[str, SignatureEvidence] = {}
    topology: dict[str, SignatureEvidence] = {}
    buckets = {"metric": metric, "log": log, "trace": trace, "topology": topology}
    for card in result.cards:
        modality = card.modality
        target = buckets.get(modality)
        if target is None:
            if modality.startswith("network"):
                target = trace
            elif modality.startswith("memory") or modality in {"cpu", "disk", "filesystem"}:
                target = metric
            else:
                target = topology
        target[card.kind] = card_to_signature_evidence(card)
    return ServiceSignature(result.candidate.service, metric=metric, log=log, trace=trace, topology=topology)


def case_signature_from_results(case_id: str, modal_coverage: Mapping[str, str],
                                results: Iterable[CandidateRuleResult]) -> CaseSignature:
    services = tuple(service_signature_from_rule_result(result) for result in results)
    active_services = sum(1 for svc in services if _service_has_support(svc))
    support_counts = {
        svc.service: _service_support_intensity(svc)
        for svc in services
    }
    dominant_service = max(support_counts, key=support_counts.get) if support_counts else None
    total_support = sum(support_counts.values())
    concentration = (support_counts.get(dominant_service, 0) / total_support) if total_support else 0.0
    blind_spots = tuple(name for name, status in modal_coverage.items() if status != "present")
    dominant_types = tuple(_dominant_types(services))
    return CaseSignature(
        case_id=case_id,
        modal_coverage=modal_coverage,
        services=services,
        anomaly_distribution={
            "active_services": active_services,
            "dominant_service": dominant_service,
            "concentration": concentration,
        },
        dominant_evidence_types=dominant_types,
        blind_spots=blind_spots,
    )


def _service_has_support(service: ServiceSignature) -> bool:
    for group in (service.metric, service.log, service.trace, service.topology):
        if any(item.state == "support" for item in group.values()):
            return True
    return False


def _service_support_intensity(service: ServiceSignature) -> int:
    total = 0
    for group in (service.metric, service.log, service.trace, service.topology):
        total += sum(item.intensity for item in group.values() if item.state == "support")
    return total


def _dominant_types(services: tuple[ServiceSignature, ...]) -> list[str]:
    counts: dict[str, int] = {}
    for svc in services:
        for group in (svc.metric, svc.log, svc.trace, svc.topology):
            for key, item in group.items():
                if item.state == "support":
                    counts[key] = counts.get(key, 0) + item.intensity
    return [key for key, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:5]]
