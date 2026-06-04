"""Layer 2: cluster-first refutation pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from refute_b_v2_d32.evidence import D32EvidenceQuery, EvidenceResult
from refute_b_v2_d32.layer1 import D32Knowledge
from refute_b_v2_d32.schema import D32Result, EvidenceCard, RefutationDecision, RootCandidate, reason_bucket
from refute_b_v2_d32.signature import build_case_signature, reason_for_kpi, reason_for_log
from refute_b_v2_d32.time_anchor import CandidateTimeAnchorer


@dataclass(frozen=True)
class D32PipelineConfig:
    max_cluster_candidates: int = 8
    max_mined_reason_candidates: int = 8
    fallback_top_events: int = 10
    high_suspicion_max_rebuttal: float = 2.0
    blind_penalty: float = 0.35
    support_credit: float = 0.18
    prior_credit: float = 0.9
    component_evidence_prior_scale: float = 0.08
    max_component_evidence_prior: float = 2.5


class D32RefutationPipeline:
    def __init__(self, knowledge: D32Knowledge, rules: Mapping[str, Any], baseline, node_graph: Mapping[str, Any],
                 services: list[str], config: D32PipelineConfig | None = None):
        self.knowledge = knowledge
        self.rules = list(rules.get("rules", []))
        self.baseline = baseline
        self.node_graph = dict(node_graph)
        self.services = [str(s) for s in services]
        self.config = config or D32PipelineConfig()

    def select(
        self,
        *,
        case_id: str,
        metric_df: pd.DataFrame,
        log_df: pd.DataFrame,
        trace_summary: Mapping[str, Any] | None,
        modal_status: Mapping[str, str],
        failure_count: int,
        window_start_ts: int,
    ) -> D32Result:
        signature = build_case_signature(case_id, metric_df, log_df, trace_summary, self.baseline, modal_status)
        clusters = self.knowledge.match_clusters(signature, case_id=case_id, k=3)
        mined_rules = self.knowledge.mined_reason_priors(signature)
        candidates = self._candidate_space(clusters, mined_rules, signature, metric_df, log_df)
        evidence = D32EvidenceQuery(metric_df, log_df, trace_summary, self.baseline, self.node_graph, modal_status)
        decisions = [self._evaluate(candidate, evidence) for candidate in candidates]
        decisions = sorted(decisions, key=lambda row: (row.rebuttal_score, -row.support_strength, row.candidate.component, row.candidate.reason))
        if not decisions:
            decisions = [self._empty_decision(RootCandidate(self.services[0] if self.services else "", "network latency"), "no_candidates")]
        selected = decisions[: max(1, int(failure_count))]
        anchorer = CandidateTimeAnchorer(self.baseline, self.node_graph)
        time_anchors = {
            row.candidate.key(): anchorer.anchor(row.candidate, metric_df, log_df, trace_summary, window_start_ts)
            for row in selected
        }
        prediction = {
            str(i + 1): {
                "root cause occurrence datetime": self._time_anchor(time_anchors[row.candidate.key()]),
                "root cause component": row.candidate.component,
                "root cause reason": row.candidate.reason,
            }
            for i, row in enumerate(selected)
        }
        high = tuple(row for row in decisions if row.rebuttal_score <= self.config.high_suspicion_max_rebuttal)[:10]
        low = tuple(row for row in decisions if row.rebuttal_score > self.config.high_suspicion_max_rebuttal)[:10]
        blind = tuple({"area": item, "reason": "modality unavailable or empty"} for item in signature.get("blind_spots", []))
        return D32Result(
            prediction=prediction,
            high_suspicion=high,
            low_suspicion=low,
            data_blind_spots=blind,
            debug={
                "case_id": case_id,
                "signature": signature,
                "matched_clusters": clusters,
                "mined_rules_matched": mined_rules,
                "candidate_space": [candidate.to_dict() for candidate in candidates],
                "selected_time_anchors": [
                    {
                        "candidate": row.candidate.to_dict(),
                        "time_anchor": time_anchors[row.candidate.key()].to_dict(),
                    }
                    for row in selected
                ],
                "all_decisions": [row.to_dict() for row in decisions[:30]],
            },
        )

    def _candidate_space(
        self,
        clusters: list[Mapping[str, Any]],
        mined_rules: list[Mapping[str, Any]],
        signature: Mapping[str, Any],
        metric_df: pd.DataFrame,
        log_df: pd.DataFrame,
    ) -> list[RootCandidate]:
        out: dict[tuple[str, str], RootCandidate] = {}
        evidence_priors = self._component_evidence_priors(signature)
        services_by_bucket = self._services_by_reason_bucket(signature, evidence_priors)
        for cluster in clusters:
            sim = float(cluster.get("similarity", 0.0) or 0.0)
            for row in cluster.get("reason_prior", [])[: self.config.max_cluster_candidates]:
                reason = str(row["reason"])
                bucket = reason_bucket(reason)
                for component in services_by_bucket.get(bucket, services_by_bucket.get("*", []))[:8]:
                    evidence_prior = evidence_priors.get((component, bucket), 0.0)
                    candidate = RootCandidate(
                        component,
                        reason,
                        prior=sim * float(row.get("weight", 0.0) or 0.0) + evidence_prior,
                        source="layer1_fault_cluster",
                        details={
                            "cluster_id": cluster.get("cluster_id"),
                            "similarity": sim,
                            "component_evidence_prior": evidence_prior,
                            "reason_prior_only": True,
                            **dict(row),
                        },
                    )
                    self._put_best(out, candidate)
        services = _signature_services(signature) or self.services
        for rule in mined_rules[: self.config.max_mined_reason_candidates]:
            reason = str(rule.get("target_reason", ""))
            if not reason:
                continue
            for service in services[:8]:
                evidence_prior = evidence_priors.get((str(service), reason_bucket(reason)), 0.0)
                candidate = RootCandidate(
                    str(service),
                    reason,
                    prior=float(rule.get("confidence", 0.0) or 0.0) + evidence_prior,
                    source="layer1_mined_rule",
                    details={**dict(rule), "component_evidence_prior": evidence_prior},
                )
                self._put_best(out, candidate)
        for candidate in self._fallback_event_candidates(metric_df, log_df):
            self._put_best(out, candidate)
        return sorted(out.values(), key=lambda row: (-row.prior, row.component, row.reason))[:80]

    def _services_by_reason_bucket(
        self,
        signature: Mapping[str, Any],
        evidence_priors: Mapping[tuple[str, str], float],
    ) -> dict[str, list[str]]:
        scored: dict[str, list[tuple[str, float]]] = {}
        active_services = _signature_services(signature) or self.services
        for (component, bucket), prior in evidence_priors.items():
            scored.setdefault(bucket, []).append((component, float(prior)))
            scored.setdefault("*", []).append((component, float(prior)))
        out = {
            bucket: [component for component, _ in sorted(rows, key=lambda item: (-item[1], item[0]))]
            for bucket, rows in scored.items()
        }
        out.setdefault("*", list(active_services))
        for bucket in ("cpu", "memory", "jvm_oom", "disk_io", "filesystem", "network_latency", "network_packet_loss"):
            out.setdefault(bucket, out["*"])
        return out

    def _component_evidence_priors(self, signature: Mapping[str, Any]) -> dict[tuple[str, str], float]:
        priors: dict[tuple[str, str], float] = {}
        for service in signature.get("services", []):
            component = str(service.get("service", ""))
            for modality in ("metric", "log", "trace", "topology"):
                for bucket, item in service.get(modality, {}).items():
                    if str(item.get("state", "")) != "support":
                        continue
                    strength = abs(float(item.get("strength", 0.0) or 0.0))
                    intensity = float(item.get("intensity", 0.0) or 0.0)
                    prior = min(
                        self.config.max_component_evidence_prior,
                        strength * self.config.component_evidence_prior_scale + intensity * 0.15,
                    )
                    key = (component, str(bucket))
                    priors[key] = max(priors.get(key, 0.0), prior)
        return priors

    @staticmethod
    def _put_best(out: dict[tuple[str, str], RootCandidate], candidate: RootCandidate) -> None:
        old = out.get(candidate.key())
        if old is None or candidate.prior > old.prior:
            out[candidate.key()] = candidate

    def _fallback_event_candidates(self, metric_df: pd.DataFrame, log_df: pd.DataFrame) -> list[RootCandidate]:
        rows = []
        if metric_df is not None and not metric_df.empty:
            for row in metric_df.itertuples(index=False):
                reason = reason_for_kpi(getattr(row, "kpi_name", ""))
                if reason:
                    rows.append(RootCandidate(str(row.cmdb_id), reason, 0.05, "event_fallback", {"kpi": str(row.kpi_name)}))
                if len(rows) >= self.config.fallback_top_events:
                    break
        if log_df is not None and not log_df.empty and "value" in log_df.columns:
            for row in log_df.itertuples(index=False):
                reason = reason_for_log(getattr(row, "value", ""))
                if reason:
                    rows.append(RootCandidate(str(row.cmdb_id), reason, 0.1, "log_fallback", {}))
        if not rows:
            for service in self.services[:3]:
                rows.append(RootCandidate(service, "network latency", 0.0, "empty_fallback", {}))
        return rows

    def _evaluate(self, candidate: RootCandidate, evidence: D32EvidenceQuery) -> RefutationDecision:
        cards = []
        for rule in self.rules:
            if not rule.get("enabled", True):
                continue
            buckets = {str(item) for item in rule.get("reason_buckets", [])}
            names = {str(item) for item in rule.get("reason_names", [])}
            if names and candidate.reason not in names:
                continue
            if buckets and candidate.reason_bucket not in buckets:
                continue
            result = self._call_rule(rule, candidate, evidence)
            card = self._card_for(rule, result)
            if card:
                cards.append(card)
        support = sum(card.strength for card in cards if card.polarity == "support")
        refute = sum(card.strength for card in cards if card.polarity == "refute")
        blind = sum(1 for card in cards if card.polarity == "blind")
        rebuttal = refute + blind * self.config.blind_penalty - support * self.config.support_credit - candidate.prior * self.config.prior_credit
        confidence = "HIGH" if rebuttal <= 0 and support >= 2 else ("MEDIUM" if rebuttal <= 2 else "LOW")
        return RefutationDecision(candidate, float(rebuttal), float(support), float(refute), int(blind), tuple(cards), confidence)

    def _call_rule(self, rule: Mapping[str, Any], candidate: RootCandidate, evidence: D32EvidenceQuery) -> EvidenceResult:
        name = str(rule.get("evidence_query", ""))
        args = dict(rule.get("query_args", {}))
        if name == "container_kpi_anomalous":
            return evidence.container_kpi(candidate.component, _mapped_bucket(args.get("kpi_bucket", candidate.reason_bucket)))
        if name == "node_kpi_anomalous":
            return evidence.node_kpi(candidate.component, _mapped_bucket(args.get("kpi_bucket", candidate.reason_bucket)))
        if name == "log_keyword_match":
            return evidence.log_keyword(candidate.component, tuple(args.get("keywords", ())))
        if name == "trace_slow_edge":
            return evidence.trace_slow_edge(candidate.component)
        if name == "service_trace_anomalous":
            return evidence.service_trace(candidate.component)
        if name == "trace_edge_count_drop":
            return evidence.trace_edge_count_drop(candidate.component)
        if name == "trace_first_anomalous_service":
            return evidence.trace_first_anomalous(candidate.component)
        if name == "trace_propagation_role":
            return evidence.trace_propagation_role(candidate.component, tuple(args.get("roles", ())))
        if name == "specialty_kpi_anomalous":
            return evidence.container_kpi(candidate.component, _mapped_bucket(args.get("kpi_bucket", candidate.reason_bucket)))
        if name == "all_of":
            parts = [self._call_rule({"evidence_query": spec["name"], "query_args": spec.get("args", {})}, candidate, evidence) for spec in args.get("queries", [])]
            if any(part.unavailable for part in parts):
                return EvidenceResult(False, "composite evidence unavailable", sum(part.strength for part in parts), unavailable=True)
            matched = all(part.matched for part in parts)
            return EvidenceResult(matched, "; ".join(part.evidence for part in parts), sum(part.strength for part in parts), {"parts": [part.details for part in parts]})
        return EvidenceResult(False, f"unknown evidence query {name}", unavailable=True)

    @staticmethod
    def _card_for(rule: Mapping[str, Any], result: EvidenceResult) -> EvidenceCard | None:
        rule_id = str(rule.get("id", rule.get("evidence_query", "")))
        confidence = float(rule.get("confidence", 1.0) or 1.0)
        if result.unavailable:
            return EvidenceCard(rule_id, "blind", 0.0, result.evidence, result.details or {}) if rule.get("missing_policy", "blind") == "blind" else None
        if result.matched and rule.get("support_if", "matched") == "matched":
            return EvidenceCard(rule_id, "support", max(0.0, result.strength) * confidence, result.evidence, result.details or {})
        if (not result.matched) and rule.get("refute_if", "") == "not_matched":
            return EvidenceCard(rule_id, "refute", max(1.0, abs(result.strength)) * confidence, result.evidence, result.details or {})
        return None

    @staticmethod
    def _time_anchor(anchor) -> str:
        return datetime.fromtimestamp(int(anchor.timestamp)).strftime("%Y-%m-%d %H:%M:%S")

    @staticmethod
    def _empty_decision(candidate: RootCandidate, reason: str) -> RefutationDecision:
        return RefutationDecision(candidate, 999.0, 0.0, 0.0, 1, (EvidenceCard("pipeline.empty", "blind", 0.0, reason),), "LOW")


def _signature_services(signature: Mapping[str, Any]) -> list[str]:
    return [str(row.get("service")) for row in signature.get("services", []) if row.get("service")]


def _mapped_bucket(bucket: Any) -> str:
    text = str(bucket)
    return {
        "disk": "disk_io",
        "network": "network",
        "jvm_heap": "jvm_oom",
        "redis_memory": "memory",
        "mysql_memory": "memory",
    }.get(text, text)


def load_rules(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)
