"""Executable refutation rule engine."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from refute_b_v2.rules import Candidate, RefutationRule, RuleSet


@dataclass(frozen=True)
class QueryResult:
    matched: bool
    evidence: str
    strength: float = 0.0
    details: Mapping[str, Any] | None = None
    unavailable: bool = False


@dataclass(frozen=True)
class RuleEvidenceCard:
    rule_id: str
    modality: str
    kind: str
    polarity: str
    strength: float
    text: str
    details: Mapping[str, Any] = field(default_factory=dict)
    rule_type: str = "soft"
    group: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "modality": self.modality,
            "kind": self.kind,
            "polarity": self.polarity,
            "strength": self.strength,
            "text": self.text,
            "details": dict(self.details),
            "rule_type": self.rule_type,
            "group": self.group,
        }


@dataclass
class CandidateRuleResult:
    candidate: Candidate
    cards: list[RuleEvidenceCard] = field(default_factory=list)

    @property
    def hard_support(self) -> int:
        return sum(1 for card in self.cards if card.polarity == "support" and card.rule_type == "hard")

    @property
    def hard_refute(self) -> int:
        return sum(1 for card in self.cards if card.polarity == "refute" and card.rule_type == "hard")

    @property
    def blind_count(self) -> int:
        return sum(1 for card in self.cards if card.polarity == "blind")

    @property
    def support_strength(self) -> float:
        return sum(card.strength for card in self.cards if card.polarity == "support")

    @property
    def refute_strength(self) -> float:
        return sum(card.strength for card in self.cards if card.polarity == "refute")

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate.to_dict(),
            "evidence_vector": {
                "hard_support": self.hard_support,
                "hard_refute": self.hard_refute,
                "blind_count": self.blind_count,
                "support_strength": self.support_strength,
                "refute_strength": self.refute_strength,
            },
            "evidence_cards": [card.to_dict() for card in self.cards],
        }


QueryFn = Callable[[Any, Candidate, Mapping[str, Any]], QueryResult]


class QueryRegistry:
    def __init__(self):
        self._fns: dict[str, QueryFn] = {}

    def register(self, name: str, fn: QueryFn) -> None:
        self._fns[name] = fn

    def call(self, name: str, evidence: Any, candidate: Candidate, args: Mapping[str, Any]) -> QueryResult:
        if name not in self._fns:
            return QueryResult(False, f"evidence query not registered: {name}", 0.0, {"query": name}, unavailable=True)
        return self._fns[name](evidence, candidate, args)


def _coerce_result(result: Any) -> QueryResult:
    if isinstance(result, QueryResult):
        return result
    return QueryResult(
        matched=bool(getattr(result, "matched", False)),
        evidence=str(getattr(result, "evidence", "")),
        strength=float(getattr(result, "strength", 0.0) or 0.0),
        details=getattr(result, "details", None) or {},
        unavailable=bool(getattr(result, "unavailable", False)),
    )


def build_default_registry() -> QueryRegistry:
    registry = QueryRegistry()

    def container_kpi(evidence: Any, candidate: Candidate, args: Mapping[str, Any]) -> QueryResult:
        return _coerce_result(evidence.is_container_kpi_anomalous(candidate.service, str(args["kpi_bucket"])))

    def node_kpi(evidence: Any, candidate: Candidate, args: Mapping[str, Any]) -> QueryResult:
        return _coerce_result(evidence.is_node_kpi_anomalous(candidate.service, str(args["kpi_bucket"])))

    def log_keyword(evidence: Any, candidate: Candidate, args: Mapping[str, Any]) -> QueryResult:
        keywords = tuple(args.get("keywords", ()))
        return _coerce_result(evidence.does_match_log_keyword(candidate.service, keywords))

    def trace_slow_edge(evidence: Any, candidate: Candidate, args: Mapping[str, Any]) -> QueryResult:
        return _coerce_result(evidence.has_slow_trace_edge(candidate.service))

    def service_trace(evidence: Any, candidate: Candidate, args: Mapping[str, Any]) -> QueryResult:
        return _coerce_result(evidence.is_service_trace_anomalous(candidate.service))

    def trace_edge_count_drop(evidence: Any, candidate: Candidate, args: Mapping[str, Any]) -> QueryResult:
        if not hasattr(evidence, "has_trace_edge_count_drop"):
            return QueryResult(False, "trace edge count drop query unavailable", 0.0, unavailable=True)
        return _coerce_result(evidence.has_trace_edge_count_drop(candidate.service))

    def trace_first_anomalous(evidence: Any, candidate: Candidate, args: Mapping[str, Any]) -> QueryResult:
        if not hasattr(evidence, "is_trace_first_anomalous_service"):
            return QueryResult(False, "trace first-anomalous query unavailable", 0.0, unavailable=True)
        return _coerce_result(evidence.is_trace_first_anomalous_service(candidate.service))

    def trace_propagation_role(evidence: Any, candidate: Candidate, args: Mapping[str, Any]) -> QueryResult:
        if not hasattr(evidence, "has_trace_propagation_role"):
            return QueryResult(False, "trace propagation role query unavailable", 0.0, unavailable=True)
        return _coerce_result(evidence.has_trace_propagation_role(candidate.service, tuple(args.get("roles", ()))))

    def specialty_kpi(evidence: Any, candidate: Candidate, args: Mapping[str, Any]) -> QueryResult:
        if not hasattr(evidence, "is_specialty_kpi_anomalous"):
            return QueryResult(False, "specialty KPI query unavailable", 0.0, unavailable=True)
        return _coerce_result(evidence.is_specialty_kpi_anomalous(candidate.service, str(args["kpi_bucket"])))

    registry.register("container_kpi_anomalous", container_kpi)
    registry.register("node_kpi_anomalous", node_kpi)
    registry.register("log_keyword_match", log_keyword)
    registry.register("trace_slow_edge", trace_slow_edge)
    registry.register("service_trace_anomalous", service_trace)
    registry.register("trace_edge_count_drop", trace_edge_count_drop)
    registry.register("trace_first_anomalous_service", trace_first_anomalous)
    registry.register("trace_propagation_role", trace_propagation_role)
    registry.register("specialty_kpi_anomalous", specialty_kpi)
    return registry


class RuleEngine:
    def __init__(self, rules: RuleSet, registry: QueryRegistry | None = None):
        self.rules = rules
        self.registry = registry or build_default_registry()

    def run_candidate(self, candidate: Candidate, evidence: Any, groups: set[str] | None = None) -> CandidateRuleResult:
        result = CandidateRuleResult(candidate)
        for rule in self.rules.applicable(candidate, groups=groups):
            query_result = self._call_query(rule, evidence, candidate)
            card = self._card_for(rule, query_result)
            if card is not None:
                result.cards.append(card)
        return result

    def rank_candidates(self, candidates: list[Candidate], evidence: Any, groups: set[str] | None = None) -> list[CandidateRuleResult]:
        rows = [self.run_candidate(candidate, evidence, groups=groups) for candidate in candidates]
        return sorted(rows, key=lambda r: (-r.support_strength, r.hard_refute, r.blind_count, r.candidate.service, r.candidate.reason))

    def _call_query(self, rule: RefutationRule, evidence: Any, candidate: Candidate) -> QueryResult:
        if rule.evidence_query == "all_of":
            parts = []
            total_strength = 0.0
            details = {}
            for spec in rule.query_args.get("queries", []):
                name = str(spec["name"])
                args = dict(spec.get("args", {}))
                part = self.registry.call(name, evidence, candidate, args)
                parts.append(part)
                total_strength += part.strength
                details[name] = {"matched": part.matched, "unavailable": part.unavailable, "details": part.details or {}}
            if any(part.unavailable for part in parts):
                return QueryResult(False, "one or more composite evidence queries unavailable", total_strength, details, unavailable=True)
            matched = all(part.matched for part in parts)
            text = "; ".join(part.evidence for part in parts)
            return QueryResult(matched, text, total_strength, details)
        return self.registry.call(rule.evidence_query, evidence, candidate, rule.query_args)

    @staticmethod
    def _card_for(rule: RefutationRule, query_result: QueryResult) -> RuleEvidenceCard | None:
        modality, _, kind = rule.id.partition(".")
        if query_result.unavailable:
            if rule.missing_policy != "blind":
                return None
            return RuleEvidenceCard(
                rule_id=rule.id,
                modality=modality,
                kind=kind or rule.evidence_query,
                polarity="blind",
                strength=0.0,
                text=query_result.evidence,
                details=query_result.details or {},
                rule_type=rule.rule_type,
                group=rule.group,
            )
        if query_result.matched and rule.support_if == "matched":
            return RuleEvidenceCard(
                rule_id=rule.id,
                modality=modality,
                kind=kind or rule.evidence_query,
                polarity="support",
                strength=max(0.0, query_result.strength) * rule.confidence,
                text=query_result.evidence,
                details=query_result.details or {},
                rule_type=rule.rule_type,
                group=rule.group,
            )
        if (not query_result.matched) and rule.refute_if == "not_matched":
            return RuleEvidenceCard(
                rule_id=rule.id,
                modality=modality,
                kind=kind or rule.evidence_query,
                polarity="refute",
                strength=max(1.0, abs(query_result.strength)) * rule.confidence,
                text=query_result.evidence,
                details=query_result.details or {},
                rule_type=rule.rule_type,
                group=rule.group,
            )
        return None
