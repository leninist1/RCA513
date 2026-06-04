"""Textbook: reason-symptom and reason-component priors learned from training cases.

Design:
  1. symptom_pattern[reason][bucket][state] = P(state | reason)
     Learned from training case signatures.
  2. primary_bucket[reason] = most discriminative bucket for this reason
     Currently: reason_bucket(reason) direct mapping.
  3. component_prior[reason][component] = P(component | reason)
     Count-based prior from training cases.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Iterable, Mapping
import math


class Textbook:
    """Learned priors from training cases for reason-symptom and reason-component matching."""

    def __init__(self, data: Mapping[str, Any] | None = None):
        if data is not None:
            self.symptom_pattern: dict[str, dict[str, dict[str, float]]] = data.get("symptom_pattern", {})
            self.component_prior: dict[str, list[tuple[str, float]]] = data.get("component_prior", {})
            self.primary_bucket: dict[str, str] = data.get("primary_bucket", {})
            self.metadata: dict[str, Any] = data.get("metadata", {})
        else:
            self.symptom_pattern = {}
            self.component_prior = {}
            self.primary_bucket = {}
            self.metadata = {}

    @classmethod
    def build(
        cls,
        case_rows: Iterable[Mapping[str, Any]],
        all_buckets: Iterable[str] = (
            "cpu", "memory", "disk_io", "filesystem",
            "network_latency", "network_packet_loss",
        ),
    ) -> "Textbook":
        """Build textbook from training case rows (each must have 'reason', 'component', 'signature')."""
        buckets = list(all_buckets)

        # --- 1. symptom_pattern: P(state | reason) per bucket ---
        # reason → bucket → state → count
        symptom_counts: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
        reason_total: Counter = Counter()

        for row in case_rows:
            reason = str(row.get("reason", "")).strip()
            if not reason:
                continue
            reason_total[reason] += 1

            signature = row.get("signature", {})
            # aggregate system-level state: "support" if any service has this bucket support
            sys_states: dict[str, str] = {}
            for svc in signature.get("services", []):
                metric = svc.get("metric", {})
                for bucket in buckets:
                    info = metric.get(bucket, {})
                    state = str(info.get("state", "normal"))
                    if state == "support":
                        sys_states[bucket] = "support"
                    elif bucket not in sys_states:
                        sys_states[bucket] = state

            for bucket in buckets:
                state = sys_states.get(bucket, "missing")
                symptom_counts[reason][bucket][state] += 1

        symptom_pattern: dict[str, dict[str, dict[str, float]]] = {}
        for reason, bucket_dict in symptom_counts.items():
            total = reason_total[reason]
            symptom_pattern[reason] = {}
            for bucket, state_counts in bucket_dict.items():
                symptom_pattern[reason][bucket] = {
                    state: cnt / total for state, cnt in state_counts.items()
                }

        # --- 2. primary_bucket: direct mapping from reason_bucket ---
        from refute_b_v2_d32.schema import reason_bucket as _rb
        primary_bucket: dict[str, str] = {}
        for reason in symptom_pattern:
            primary_bucket[reason] = _rb(reason)

        # --- 3. component_prior: P(component | reason), smoothed ---
        comp_counts: dict[str, Counter] = defaultdict(Counter)
        for row in case_rows:
            reason = str(row.get("reason", "")).strip()
            component = str(row.get("component", "")).strip()
            if reason and component:
                comp_counts[reason][component] += 1

        component_prior: dict[str, list[tuple[str, float]]] = {}
        for reason, counts in comp_counts.items():
            total = sum(counts.values())
            # Add +1 smoothing to avoid zero prior for unseen components
            smoothed_total = total + 1
            entries = []
            for comp, cnt in counts.items():
                entries.append((comp, (cnt + 1) / smoothed_total))
            # Sort by prior descending
            entries.sort(key=lambda x: -x[1])
            component_prior[reason] = entries

        return cls({
            "symptom_pattern": symptom_pattern,
            "component_prior": component_prior,
            "primary_bucket": primary_bucket,
            "metadata": {
                "n_cases": sum(reason_total.values()),
                "n_reasons": len(reason_total),
                "reasons": sorted(reason_total.keys()),
                "buckets": buckets,
            },
        })

    def score_reasons(self, symptoms: dict[str, str], top_k: int = 3) -> list[tuple[str, float]]:
        """Score all reasons given current symptoms.

        symptoms: {bucket: state}  e.g. {"cpu": "support", "memory": "normal", ...}

        Returns list of (reason, score) sorted by score descending.
        Uses log-likelihood: Σ log P(state_b | reason) for each bucket.
        Missing states are treated as P=0.5 (uninformative).
        """
        scored = []
        for reason, bucket_states in self.symptom_pattern.items():
            log_score = 0.0
            for bucket, state in symptoms.items():
                probs = bucket_states.get(bucket, {})
                p = probs.get(state, 0.5)  # unseen state → weakly informative
                if p > 0:
                    log_score += math.log(p)
                else:
                    log_score += math.log(1e-6)  # floor
            scored.append((reason, log_score))
        scored.sort(key=lambda x: -x[1])
        return scored[:top_k]

    def top_components(
        self,
        reason: str,
        evidence_strengths: dict[str, float],
        log_hits: dict[str, int],
        time_ranks: dict[str, float],
        k: int = 3,
        lambda_evidence: float = 0.3,
        lambda_log: float = 3.0,
        lambda_time: float = 2.0,
        lambda_prior: float = 2.0,
    ) -> list[str]:
        """Rank components given a specific reason.

        evidence_strengths: {component: max_deviation_in_primary_bucket}
        log_hits:           {component: 1 if log matches reason else 0}
        time_ranks:         {component: 1/rank (higher = earlier anomaly)}

        Scoring balances:
          - Historical prior (strong signal: which components caused this before)
          - Log match (direct evidence: error logs)
          - Time priority (earliest anomaly → more likely root cause)
          - Evidence strength (weak modifier: anomaly magnitude, normalized)
        """
        primary = self.primary_bucket.get(reason, reason)
        priors = dict(self.component_prior.get(reason, []))

        # Normalize evidence strengths
        max_e = max(evidence_strengths.values()) if evidence_strengths else 1.0
        norm_evidence = {c: min(s / max(max_e, 1.0), 1.0) for c, s in evidence_strengths.items()}

        scored = []
        all_components = set(priors) | set(evidence_strengths) | set(log_hits) | set(time_ranks)
        for comp in all_components:
            e = norm_evidence.get(comp, 0.0)
            l = log_hits.get(comp, 0)
            t = time_ranks.get(comp, 0.0)
            p = priors.get(comp, 0.005)  # small default prior for unseen components

            score = (
                lambda_prior * p +
                lambda_log * l +
                lambda_time * t +
                lambda_evidence * e
            )
            scored.append((comp, score))

        scored.sort(key=lambda x: -x[1])
        return [comp for comp, _ in scored[:k]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "symptom_pattern": self.symptom_pattern,
            "component_prior": self.component_prior,
            "primary_bucket": self.primary_bucket,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Textbook":
        return cls(data)
