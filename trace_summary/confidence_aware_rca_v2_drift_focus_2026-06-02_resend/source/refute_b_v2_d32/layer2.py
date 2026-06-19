"""Layer 2: cluster-first refutation pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from refute_b_v2_d32.evidence import D32EvidenceQuery, EvidenceResult
from refute_b_v2_d32.joint_candidates import component_family, generate_joint_root_candidates, primary_bucket_for_reason
from refute_b_v2_d32.layer1 import D32Knowledge
from refute_b_v2_d32.reason_classifier import ReasonClassifier, extract_features
from refute_b_v2_d32.schema import D32Result, EvidenceCard, RefutationDecision, RootCandidate, reason_bucket
from refute_b_v2_d32.signature import build_case_signature, reason_for_kpi, reason_for_log
from refute_b_v2_d32.bucket_resolver import row_reason
from refute_b_v2_d32.time_anchor import CandidateTimeAnchorer, CandidateTimeAnchorPolicy


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
    enable_two_stage_selector: bool = True
    two_stage_top_reasons: int = 5
    two_stage_reason_confidence_bonus: float = 0.15
    # Evidence scoring parameters (used by 2-stage selector)
    two_stage_support_credit: float = 0.70
    two_stage_refute_penalty: float = 0.90
    two_stage_joint_signal_credit: float = 0.70
    two_stage_type_bonus: float = 0.45
    two_stage_type_penalty: float = 0.55
    two_stage_background_os_cpu_penalty: float = 0.85
    two_stage_weak_joint_penalty: float = 0.45
    two_stage_direct_support_bonus: float = 0.90
    two_stage_prior_tiebreak_credit: float = 0.05
    # Window reason scoring
    enable_window_reason_scores: bool = True
    window_reason_score_credit: float = 0.65
    window_reason_top_k: int = 3
    window_reason_weak_penalty: float = 0.8
    window_reason_type_bonus: float = 0.6
    window_reason_type_penalty: float = 0.4
    symptom_reason_score_credit: float = 1.0
    symptom_reason_top_k: int = 8
    # Dataset/fold identify the same-dataset cross-fit classifier.
    dataset_name: str = ""
    reason_classifier_fold: int | None = None
    casefold_classifier_dir: str = "knowledge/casefold_classifiers"
    strict_reason_classifier: bool = True
    reason_classifier_path: str = "knowledge/reason_classifier.json"
    # Reason → expected component family (system knowledge, NOT GT-derived)
    # Based on the type-compatibility rules already in joint_candidates.py:
    #   docker → CPU fault, os → network delay/loss, db → db connection/close
    reason_family_map: dict[str, str] = field(default_factory=lambda: {
        "cpu": "docker",
        "network_latency": "os",
        "network_packet_loss": "os",
        "db_connection": "db",
    })
    # Per-dataset overrides: if dataset_name matches, use this map instead
    # Bank: cpu/os/db/proxy faults all map to cpu bucket, no single family works
    # Market: cpu faults are node-level (os), not container-level (docker)
    DATASET_FAMILY_MAPS: dict[str, dict[str, str]] = field(default_factory=lambda: {
        "bank": {},
        "market_cb1": {},
        "market_cb2": {},
    })
    # Per-dataset reason name mappings: pipeline reason string → scoring_points canonical name
    REASON_NAME_MAPS: dict[str, dict[str, str]] = field(default_factory=lambda: {
        "bank": {
            "CPU fault": "high CPU usage",
            "network delay": "network latency",
            "network loss": "network packet loss",
        },
        "market_cb1": {
            "CPU fault": "container CPU load",
            "network delay": "container network latency",
            "network loss": "container network packet retransmission",
        },
        "market_cb2": {
            "CPU fault": "container CPU load",
            "network delay": "container network latency",
            "network loss": "container network packet retransmission",
        },
    })
    enable_joint_candidates: bool = True
    joint_beam_per_reason: int = 8
    joint_prior_scale: float = 0.15
    joint_prior_offset: float = 0.05
    max_joint_prior: float = 1.8
    legacy_component_evidence_scale_with_joint: float = 1.0
    time_anchor_early_bonus: float = 0.25
    time_anchor_sustained_bonus: float = 2.0
    onset_first: bool = True
    onset_first_weight: float = 4.0
    onset_decay_seconds: float = 300.0
    component_onset_bonus: float = 5.0
    disable_family_filter: bool = False
    portable_ontology_enabled: bool = False


class D32RefutationPipeline:
    def __init__(self, knowledge: D32Knowledge, rules: Mapping[str, Any], baseline, node_graph: Mapping[str, Any],
                 services: list[str], config: D32PipelineConfig | None = None):
        self.knowledge = knowledge
        self.rules = list(rules.get("rules", []))
        self.baseline = baseline
        self.node_graph = dict(node_graph)
        self.services = [str(s) for s in services]
        self.config = config or D32PipelineConfig()
        self._reason_classifier: ReasonClassifier | None = None
        if self.config.enable_two_stage_selector:
            self._load_reason_classifier()

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
        candidates = self._candidate_space(clusters, mined_rules, signature, metric_df, log_df, trace_summary, window_start_ts)
        window_reason_scores = self._window_reason_scores(candidates, signature)
        evidence = D32EvidenceQuery(metric_df, log_df, trace_summary, self.baseline, self.node_graph, modal_status)
        reason_posterior: dict[str, float] | None = None
        # Only 2-stage selector is supported (evidence-first and legacy removed)
        reason_posterior = dict(self._compute_reason_posterior(candidates, evidence, signature))
        decisions = self._evaluate_two_stage(candidates, evidence, reason_posterior)
        decisions = sorted(decisions, key=lambda row: (
            -reason_posterior.get(row.candidate.reason_bucket, 0.0),
            row.rebuttal_score,
            -row.support_strength,
            row.candidate.component,
            row.candidate.reason,
        ))
        if not decisions:
            decisions = [self._empty_decision(RootCandidate(self.services[0] if self.services else "", "network latency"), "no_candidates")]
        selected = decisions[: max(1, int(failure_count))]
        anchorer = CandidateTimeAnchorer(
            self.baseline, self.node_graph,
            policy=CandidateTimeAnchorPolicy(
                early_vote_bonus=self.config.time_anchor_early_bonus,
                metric_sustained_bonus=self.config.time_anchor_sustained_bonus,
                onset_first=self.config.onset_first,
                onset_first_weight=self.config.onset_first_weight,
                onset_decay_seconds=self.config.onset_decay_seconds,
                component_onset_bonus=self.config.component_onset_bonus,
            ),
        )
        time_anchors = {
            row.candidate.key(): anchorer.anchor(row.candidate, metric_df, log_df, trace_summary, window_start_ts)
            for row in selected
        }
        prediction = {
            str(i + 1): {
                "root cause occurrence datetime": self._time_anchor(time_anchors[row.candidate.key()]),
                "root cause component": row.candidate.component,
                "root cause reason": self._map_reason_name(row.candidate.reason),
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
                "window_reason_scores": window_reason_scores,
                "reason_posterior": reason_posterior,
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
        trace_summary: Mapping[str, Any] | None,
        window_start_ts: int,
    ) -> list[RootCandidate]:
        out: dict[tuple[str, str], RootCandidate] = {}
        evidence_priors = self._component_evidence_priors(signature)
        services_by_bucket = self._services_by_reason_bucket(signature, evidence_priors)

        if self.config.enable_joint_candidates:
            for candidate in generate_joint_root_candidates(
                metric_df=metric_df,
                log_df=log_df,
                trace_summary=trace_summary,
                baseline=self.baseline,
                joint_prior=self.knowledge.joint_prior,
                window_start_ts=window_start_ts,
                beam_per_reason=self.config.joint_beam_per_reason,
                prior_scale=self.config.joint_prior_scale,
                prior_offset=self.config.joint_prior_offset,
                max_prior=self.config.max_joint_prior,
            ):
                self._put_best(out, candidate)

        for cluster in clusters:
            sim = float(cluster.get("similarity", 0.0) or 0.0)
            for row in cluster.get("reason_prior", [])[: self.config.max_cluster_candidates]:
                reason = str(row["reason"])
                bucket = reason_bucket(reason)
                for component in services_by_bucket.get(bucket, services_by_bucket.get("*", []))[:8]:
                    raw_evidence_prior = evidence_priors.get((component, bucket), 0.0)
                    evidence_prior = self._legacy_component_evidence_prior(raw_evidence_prior)
                    candidate = RootCandidate(
                        component,
                        reason,
                        prior=sim * float(row.get("weight", 0.0) or 0.0) + evidence_prior,
                        source="layer1_fault_cluster",
                        details={
                            "cluster_id": cluster.get("cluster_id"),
                            "similarity": sim,
                            "component_evidence_prior": raw_evidence_prior,
                            "effective_component_evidence_prior": evidence_prior,
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
                raw_evidence_prior = evidence_priors.get((str(service), reason_bucket(reason)), 0.0)
                evidence_prior = self._legacy_component_evidence_prior(raw_evidence_prior)
                candidate = RootCandidate(
                    str(service),
                    reason,
                    prior=float(rule.get("confidence", 0.0) or 0.0) + evidence_prior,
                    source="layer1_mined_rule",
                    details={
                        **dict(rule),
                        "component_evidence_prior": raw_evidence_prior,
                        "effective_component_evidence_prior": evidence_prior,
                    },
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

    def _legacy_component_evidence_prior(self, raw_prior: float) -> float:
        if not self.config.enable_joint_candidates:
            return float(raw_prior)
        return float(raw_prior) * float(self.config.legacy_component_evidence_scale_with_joint)

    @staticmethod
    def _put_best(out: dict[tuple[str, str], RootCandidate], candidate: RootCandidate) -> None:
        old = out.get(candidate.key())
        if old is None or candidate.prior > old.prior:
            out[candidate.key()] = candidate

    def _fallback_event_candidates(self, metric_df: pd.DataFrame, log_df: pd.DataFrame) -> list[RootCandidate]:
        rows = []
        if metric_df is not None and not metric_df.empty:
            for row in metric_df.itertuples(index=False):
                reason = row_reason(row, getattr(row, "kpi_name", ""))
                if reason:
                    rows.append(RootCandidate(str(row.cmdb_id), reason, 0.05, "event_fallback", {"kpi": str(row.kpi_name)}))
                if len(rows) >= self.config.fallback_top_events:
                    break
        if log_df is not None and not log_df.empty and "value" in log_df.columns:
            for row in log_df.itertuples(index=False):
                reason = reason_for_log(getattr(row, "value", ""))
                if reason:
                    rows.append(RootCandidate(str(row.cmdb_id), reason, 0.1, "log_fallback", {}))
        # Ensure node names from node_graph are also candidate components
        seen_components = {c.component for c in rows}
        node_names = [
            str(n) for n in self.node_graph.get("nodes", {}).keys()
            if not str(n).startswith("node::")
        ]
        for node_name in sorted(node_names):
            if node_name not in seen_components:
                rows.append(RootCandidate(str(node_name), "node CPU load", 0.02, "node_fallback", {}))
        if not rows:
            for service in self.services[:3]:
                rows.append(RootCandidate(service, "network latency", 0.0, "empty_fallback", {}))
        return rows

    def _load_reason_classifier(self) -> None:
        from pathlib import Path

        if self.config.dataset_name and self.config.reason_classifier_fold is not None:
            clf_name = f"reason_classifier_{self.config.dataset_name}_fold{int(self.config.reason_classifier_fold)}.json"
            fold_path = Path(self.config.casefold_classifier_dir) / clf_name
            if not fold_path.exists():
                fold_path = Path(__file__).parent.parent / self.config.casefold_classifier_dir / clf_name
            if fold_path.exists():
                self._reason_classifier = ReasonClassifier.load(str(fold_path))
                return
            if self.config.strict_reason_classifier:
                raise FileNotFoundError(f"missing strict case-fold reason classifier: {fold_path}")

        if self.config.strict_reason_classifier and self.config.dataset_name:
            raise ValueError("strict same-dataset case-fold mode requires reason_classifier_fold")

        path = Path(self.config.reason_classifier_path)
        if not path.exists():
            path = Path(__file__).parent.parent / self.config.reason_classifier_path
        if path.exists():
            self._reason_classifier = ReasonClassifier.load(str(path))

    def _compute_reason_posterior(
        self,
        candidates: list[RootCandidate],
        evidence: D32EvidenceQuery,
        signature: Mapping[str, Any],
    ) -> dict[str, float]:
        if self._reason_classifier is not None:
            return self._learned_reason_posterior(candidates, evidence)

        import math
        from collections import defaultdict
        # ... (保留了原来的 hand-tuned 实现作为 fallback)
        config = self.config
        raw: dict[str, list[float]] = defaultdict(list)
        for candidate in candidates:
            if candidate.source != "joint_generator":
                continue
            details = dict(candidate.details)
            signal_strength = max(0.0, float(details.get("signal_strength", 0.0) or 0.0))
            signal_count = max(0.0, float(details.get("signal_count", 0.0) or 0.0))
            if bool(details.get("weak_family_presence")):
                s = -config.window_reason_weak_penalty
            else:
                s = math.log1p(min(signal_strength, 100.0)) + min(signal_count, 20.0) / 5.0
            family = component_family(candidate.component)
            bucket = primary_bucket_for_reason(candidate.reason)
            if _type_compatible(family, candidate.reason, bucket):
                s += config.window_reason_type_bonus
            else:
                s -= config.window_reason_type_penalty
            sources = {str(item) for item in details.get("joint_sources", [])}
            if any("trace" in item for item in sources):
                s += 0.8
            if any("log" in item for item in sources):
                s += 1.0
            raw[candidate.reason].append(float(s))

        posterior: dict[str, float] = {}
        top_k = max(1, int(config.window_reason_top_k))
        for reason, scores in raw.items():
            top = sorted(scores, reverse=True)[:top_k]
            posterior[reason] = float(sum(top) / max(1, len(top)))

        symptom_scores = self.knowledge.symptom_reason_scores(
            signature,
            top_k=config.symptom_reason_top_k,
            extra_features=_joint_reason_features_from_candidates(candidates),
        )
        for reason, val in symptom_scores.items():
            posterior[reason] = posterior.get(reason, 0.0) + config.symptom_reason_score_credit * float(val)

        evidence_weight = float(getattr(config, 'two_stage_reason_evidence_weight', 0.5))
        if evidence.trace_summary and evidence.trace_summary.get("trace_status") == "present":
            events = evidence.trace_summary.get("events", {}) or {}
            slow_edges = events.get("slow_edges", []) or []
            dropped_edges = events.get("dropped_edges", []) or []
            if slow_edges:
                posterior["network delay"] = posterior.get("network delay", 0.0) + evidence_weight * 1.5
            if dropped_edges:
                posterior["network loss"] = posterior.get("network loss", 0.0) + evidence_weight * 1.5
            if events.get("first_anomalous_service"):
                posterior["network delay"] = posterior.get("network delay", 0.0) + evidence_weight * 0.5

        if evidence.log_df is not None and not evidence.log_df.empty and "value" in evidence.log_df.columns:
            log_texts = " ".join(evidence.log_df["value"].dropna().astype(str))
            log_low = log_texts.lower()
            if any(kw in log_low for kw in ("outofmemory", "oom", "heap space", "java heap")):
                posterior["JVM Out of Memory (OOM) Heap"] = posterior.get("JVM Out of Memory (OOM) Heap", 0.0) + evidence_weight * 2.0
            if any(kw in log_low for kw in ("connection", "timeout", "session", "too many")):
                posterior["db connection limit"] = posterior.get("db connection limit", 0.0) + evidence_weight * 1.5

        if evidence.metric_df is not None and not evidence.metric_df.empty:
            reason_bucket_map = {
                "CPU fault": "cpu",
                "network delay": "network_latency",
                "network loss": "network_packet_loss",
                "db connection limit": "db_connection",
                "db close": "db_connection",
            }
            bucket_aggregate: dict[str, float] = defaultdict(float)
            for row in evidence.metric_df.itertuples(index=False):
                value = getattr(row, "value", None)
                if value is None:
                    continue
                result = evidence.baseline.is_anomalous(
                    str(row.cmdb_id), str(row.kpi_name), value, threshold="p99",
                )
                if not result.is_anomalous:
                    continue
                strength = abs(float(result.deviation or 0.0))
                kpi_low = str(row.kpi_name).lower()
                if any(t in kpi_low for t in ("cpu", "load")):
                    bucket_aggregate["cpu"] += strength
                if any(t in kpi_low for t in ("mem", "memory", "heap", "swap", "cache")):
                    bucket_aggregate["memory"] += strength
                if any(t in kpi_low for t in ("drop", "loss", "retrans", "error", "err", "reject", "abort", "reset")):
                    bucket_aggregate["network_packet_loss"] += strength
                elif any(t in kpi_low for t in ("net", "traffic", "tcp", "ping", "packet", "recv", "send", "queue")):
                    bucket_aggregate["network_latency"] += strength
            for reason, bucket in reason_bucket_map.items():
                agg = bucket_aggregate.get(bucket, 0.0)
                if agg > 0.0:
                    posterior[reason] = posterior.get(reason, 0.0) + evidence_weight * math.log1p(agg)

            mem_agg = bucket_aggregate.get("memory", 0.0)
            if mem_agg > 0.0:
                posterior["high memory usage"] = posterior.get("high memory usage", 0.0) + evidence_weight * math.log1p(mem_agg)

        return posterior

    def _learned_reason_posterior(
        self,
        candidates: list[RootCandidate],
        evidence: D32EvidenceQuery,
    ) -> dict[str, float]:
        feats = extract_features(
            metric_df=evidence.metric_df,
            log_df=evidence.log_df,
            trace_summary=evidence.trace_summary,
            baseline=evidence.baseline,
            joint_candidates=candidates,
        )
        if self._reason_classifier is None:
            return {}
        proba = self._reason_classifier.predict_proba(feats)
        # Ensure every reason bucket present in candidates gets a minimal probability
        # (handles LODO case where some classes were never in training data)
        seen_buckets = {c.reason_bucket for c in candidates}
        min_prob = 0.005
        for bucket in sorted(seen_buckets):
            if bucket not in proba:
                proba[bucket] = min_prob
        return proba

    def _component_earliness_strength(
        self,
        component: str,
        reason_bucket: str,
        evidence: D32EvidenceQuery,
    ) -> tuple[float, float, float]:
        """Compute (earliness, max_deviation, anomaly_count) for a component + reason bucket.

        Earliness is 1.0 if first anomaly is at window start, decaying to 0.
        """
        import math
        from refute_b_v2_d32.bucket_resolver import row_in_bucket
        from refute_b_v2_d32.portable_schema import BUCKETS_COLUMN

        df = evidence.metric_df
        if df is None or df.empty or "kpi_name" not in df.columns:
            return (0.0, 0.0, 0.0)

        comp_rows = df[df["cmdb_id"].astype(str) == str(component)]
        if comp_rows.empty:
            return (0.0, 0.0, 0.0)

        # When a precomputed buckets column is present (portable datasets), use
        # ALL anomalous KPIs on this component for earliness/strength, not just
        # those matching the candidate reason bucket.  This lets cross-fault
        # propagation symptoms (e.g. Tomcat error/rejected-session counters,
        # which are excluded from reason buckets to avoid density pollution)
        # still contribute to component-localisation evidence.  OpenRCA-native
        # frames have no buckets column and keep the original reason-bucket
        # filtering, staying byte-identical.
        has_buckets_col = BUCKETS_COLUMN in comp_rows.columns
        if has_buckets_col:
            def _evidence_match(row, kpi):
                return True
        else:
            def _evidence_match(row, kpi):
                return row_in_bucket(row, reason_bucket, kpi)

        first_ts = None
        max_dev = 0.0
        anom_count = 0

        window_start = float(comp_rows["timestamp"].min())
        window_end = float(comp_rows["timestamp"].max())
        window_duration = max(window_end - window_start, 1.0)

        for row in comp_rows.itertuples(index=False):
            kpi = str(row.kpi_name)
            if not _evidence_match(row, kpi):
                continue
            value = getattr(row, "value", None)
            if value is None or not isinstance(value, (int, float)) or not (isinstance(value, float) or isinstance(value, int)):
                try:
                    value = float(value)
                except (ValueError, TypeError):
                    continue
            if not math.isfinite(float(value)):
                continue
            try:
                result = evidence.baseline.is_anomalous(
                    str(row.cmdb_id), kpi, float(value), threshold="p99",
                )
            except Exception:
                continue
            if not result.is_anomalous:
                continue

            dev = abs(float(result.deviation or 0.0))
            anom_count += 1
            if dev > max_dev:
                max_dev = dev
            ts = float(row.timestamp)
            if first_ts is None or ts < first_ts:
                first_ts = ts

        if first_ts is None:
            return (0.0, 0.0, 0.0)

        earliness = 1.0 - min(1.0, (first_ts - window_start) / window_duration)
        return (earliness, math.log1p(max_dev), math.log1p(anom_count))

    def _evaluate_two_stage(
        self,
        candidates: list[RootCandidate],
        evidence: D32EvidenceQuery,
        reason_posterior: Mapping[str, float],
    ) -> list[RefutationDecision]:
        """Stage 2 v2: reason→family filter + component-level earliness/strength ranking."""
        import math

        top_k = max(1, int(self.config.two_stage_top_reasons))
        sorted_reasons = sorted(reason_posterior.items(), key=lambda x: -x[1])
        selected_reasons = {reason for reason, _ in sorted_reasons[:top_k]}

        # Phase 1: Family filter — for each selected reason, keep only matching family
        family_map = self.config.reason_family_map if self.config.disable_family_filter else \
            self.config.DATASET_FAMILY_MAPS.get(self.config.dataset_name, self.config.reason_family_map)
        family_candidates: dict[str, list[RootCandidate]] = {}
        for c in candidates:
            c_bucket = c.reason_bucket
            c_family = component_family(c.component)
            expected_family = family_map.get(c_bucket)
            if c_bucket in selected_reasons or c.reason in selected_reasons:
                if expected_family is None or c_family == expected_family:
                    family_candidates.setdefault(c_bucket, []).append(c)

        if not family_candidates:
            return [self._empty_decision(
                candidates[0] if candidates else RootCandidate(self.services[0] if self.services else "", "network latency"),
                "no_family_matching_candidates",
            )]

        decisions: list[RefutationDecision] = []
        for reason in [r for r, _ in sorted_reasons[:top_k]]:
            reason_cands = family_candidates.get(reason, [])
            if not reason_cands:
                continue

            # Phase 2: Compute component-level (earliness, strength, density) for each candidate
            scored_cands: list[tuple[float, RootCandidate, tuple[float, float, float]]] = []
            for candidate in reason_cands:
                e, s, d = self._component_earliness_strength(
                    candidate.component, reason, evidence,
                )
                joint_sig = 0.0
                if candidate.source == "joint_generator":
                    details = dict(candidate.details)
                    js = float(details.get("signal_strength", 0.0) or 0.0)
                    jc = float(details.get("signal_count", 0.0) or 0.0)
                    joint_sig = math.log1p(js) + min(jc, 12.0) / 8.0

                # Combined score: earliness (0.6) + strength (0.25) + density (0.15) + joint signal bonus
                component_score = 0.60 * e + 0.25 * s + 0.15 * d + 0.10 * joint_sig
                scored_cands.append((component_score, candidate, (e, s, d)))

            scored_cands.sort(key=lambda x: -x[0])

            rc = float(max(0.0, reason_posterior.get(reason, 0.0) or 0.0))
            for rank, (comp_score, candidate, (e, s, d)) in enumerate(scored_cands):
                # Lightweight evidence evaluation (rules are still run for debug)
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
                    card = self._card_for(rule, self._call_rule(rule, candidate, evidence))
                    if card is not None:
                        cards.append(card)

                support = sum(_squash(card.strength) for card in cards if card.polarity == "support")
                refute = sum(_squash(card.strength) for card in cards if card.polarity == "refute")
                blind = sum(1 for card in cards if card.polarity == "blind")

                # Score: reason_confidence * component_score dominates
                base = rc * comp_score
                score = (
                    1.00 * base
                    + 0.05 * self.config.two_stage_support_credit * support
                    - 0.03 * self.config.two_stage_refute_penalty * refute
                    - 0.02 * float(blind)
                )
                cards.append(EvidenceCard(
                    "selector.two_stage_v2",
                    "selector",
                    score,
                    f"component-level score reason={candidate.reason} comp={candidate.component} e={e:.3f} s={s:.3f} d={d:.3f} rank={rank}",
                    {"component_score": float(comp_score), "earliness": float(e), "strength": float(s), "density": float(d),
                     "rank": rank, "reason_confidence": float(rc)},
                ))
                confidence = "HIGH" if rank == 0 and score >= 1.0 else ("MEDIUM" if rank <= 2 else "LOW")
                decisions.append(RefutationDecision(
                    candidate, float(-score), float(support), float(refute), int(blind), tuple(cards), confidence,
                ))

        return decisions

    def _window_reason_scores(self, candidates: list[RootCandidate], signature: Mapping[str, Any]) -> dict[str, float]:
        if not self.config.enable_window_reason_scores:
            return {}
        raw: dict[str, list[float]] = {}
        for candidate in candidates:
            if candidate.source != "joint_generator":
                continue
            details = dict(candidate.details)
            signal_strength = max(0.0, float(details.get("signal_strength", 0.0) or 0.0))
            signal_count = max(0.0, float(details.get("signal_count", 0.0) or 0.0))
            if bool(details.get("weak_family_presence")):
                score = -self.config.window_reason_weak_penalty
            else:
                import math
                score = math.log1p(min(signal_strength, 100.0)) + min(signal_count, 20.0) / 5.0
            family = component_family(candidate.component)
            bucket = primary_bucket_for_reason(candidate.reason)
            if _type_compatible(family, candidate.reason, bucket):
                score += self.config.window_reason_type_bonus
            else:
                score -= self.config.window_reason_type_penalty
            sources = {str(item) for item in details.get("joint_sources", [])}
            if any("trace" in item for item in sources):
                score += 0.8
            if any("log" in item for item in sources):
                score += 1.0
            raw.setdefault(candidate.reason, []).append(float(score))
        joint_margins: dict[str, float] = {}
        top_k = max(1, int(self.config.window_reason_top_k))
        if raw:
            absolute = {
                reason: sum(sorted(scores, reverse=True)[:top_k]) / min(top_k, len(scores))
                for reason, scores in raw.items()
            }
            joint_margins = _margin_scores(absolute)
        symptom_scores = self.knowledge.symptom_reason_scores(
            signature,
            top_k=self.config.symptom_reason_top_k,
            extra_features=_joint_reason_features_from_candidates(candidates),
        )
        symptom_margins = _margin_scores(symptom_scores)
        reasons = set(joint_margins) | set(symptom_margins)
        return {
            reason: float(joint_margins.get(reason, 0.0) + self.config.symptom_reason_score_credit * symptom_margins.get(reason, 0.0))
            for reason in sorted(reasons)
        }

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

    def _map_reason_name(self, reason: str) -> str:
        name_map = self.config.REASON_NAME_MAPS.get(self.config.dataset_name, {})
        return name_map.get(reason, reason)


def _signature_services(signature: Mapping[str, Any]) -> list[str]:
    return [str(row.get("service")) for row in signature.get("services", []) if row.get("service")]


def _squash(strength: float) -> float:
    import math

    return math.log(1.0 + min(abs(float(strength)), 10.0))


def _type_compatible(family: str, reason: str, bucket: str) -> bool:
    if reason == "CPU fault" and family == "docker":
        return True
    if bucket in {"network_latency", "network_packet_loss"} and family == "os":
        return True
    if bucket == "db_connection" and family == "db":
        return True
    return False


def _margin_scores(scores: Mapping[str, float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in scores.items():
        others = [float(other) for other_key, other in scores.items() if other_key != key]
        other_best = max(others) if others else 0.0
        out[str(key)] = float(value) - other_best
    return out


def _joint_reason_features_from_candidates(candidates: list[RootCandidate]) -> list[str]:
    features: set[str] = set()
    for candidate in candidates:
        if candidate.source != "joint_generator":
            continue
        details = dict(candidate.details)
        if bool(details.get("weak_family_presence")):
            continue
        signal_count = float(details.get("signal_count", 0.0) or 0.0)
        if signal_count <= 0.0:
            continue
        signal_strength = float(details.get("signal_strength", 0.0) or 0.0)
        family = component_family(candidate.component)
        bucket = str(details.get("primary_bucket") or primary_bucket_for_reason(candidate.reason))
        features.add(f"joint:family:{family}:bucket:{bucket}")
        if signal_count >= 5:
            features.add(f"joint:family:{family}:bucket:{bucket}:multi")
        if signal_strength >= 20.0:
            features.add(f"joint:family:{family}:bucket:{bucket}:strong")
        elif signal_strength >= 5.0:
            features.add(f"joint:family:{family}:bucket:{bucket}:medium")
    return sorted(features)


def _compute_time_ranks(
    metric_df: pd.DataFrame,
    baseline,
    bucket: str,
) -> dict[str, float]:
    """Compute time-based priority for each component in a bucket.

    For each component, find the earliest timestamp where any KPI in this bucket
    exceeds the baseline threshold. Components with earlier anomalies get higher ranks.

    Returns {component: 1/(rank+1)} where rank=1 means earliest anomaly.
    """
    from refute_b_v2_d32.bucket_resolver import row_in_bucket
    if metric_df is None or metric_df.empty or "kpi_name" not in metric_df.columns:
        return {}

    rows = metric_df[metric_df.apply(
        lambda r: row_in_bucket(r, bucket, str(r["kpi_name"])), axis=1
    )]

    if rows.empty:
        return {}

    first_anomaly: dict[str, int] = {}
    for (cmdb_id, kpi), grp in rows.groupby(["cmdb_id", "kpi_name"]):
        grp_sorted = grp.sort_values("timestamp")
        found_ts = None
        for row in grp_sorted.itertuples(index=False):
            result = baseline.is_anomalous(row.cmdb_id, row.kpi_name, row.value, threshold="p99")
            if result.is_anomalous:
                found_ts = int(row.timestamp)
                break
        if found_ts is not None:
            comp = str(cmdb_id)
            if comp not in first_anomaly or found_ts < first_anomaly[comp]:
                first_anomaly[comp] = found_ts

    if not first_anomaly:
        return {}

    # Sort by timestamp ascending (earliest first), rank starts at 1
    sorted_comps = sorted(first_anomaly.items(), key=lambda x: x[1])
    ranks = {}
    for rank, (comp, _) in enumerate(sorted_comps, start=1):
        ranks[comp] = 1.0 / (rank + 1.0)  # rank 1 → 0.5, rank 2 → 0.33, etc.

    return ranks


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
