"""Factorized posterior updates for noise-native FaultEvents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

from .fault_event import FaultEvent, PosteriorAttribution


@dataclass
class PosteriorWeights:
    noise: float = 1.00
    metric: float = 0.85
    log: float = 0.65
    trace: float = 0.45
    counterfactual: float = 0.55
    pairwise: float = 0.22
    symptom: float = 0.80
    broad: float = 0.55
    structural: float = 0.25
    isolation: float = 0.75
    hotspot: float = 0.95
    collapse: float = 1.10
    mechanism: float = 1.20
    parent_refutation: float = 0.80
    intervention: float = 0.55


FACTOR_TO_WEIGHT = {
    "NoiseLab_logit": "noise",
    "metric_likelihood": "metric",
    "log_likelihood": "log",
    "trace_direction_likelihood": "trace",
    "counterfactual_likelihood": "counterfactual",
    "residual_collapse": "collapse",
    "pairwise_verdict": "pairwise",
    "symptomness": "symptom",
    "hotspot_symptom": "hotspot",
    "broad_explainer": "broad",
    "source_likelihood": "structural",
    "source_isolation": "isolation",
    "mechanism_break_likelihood": "mechanism",
    "mechanism_parent_refutation": "parent_refutation",
    "intervention_uniqueness": "intervention",
    "split_event": "structural",
    "merge_duplicate": "structural",
}


def score_events(
    events: Sequence[FaultEvent],
    weights: PosteriorWeights,
) -> List[PosteriorAttribution]:
    if not events:
        return []
    old = {event.event_id: float(event.posterior) for event in events}
    logits = np.array([_event_logit(event, weights) for event in events], dtype=float)
    posterior = _softmax(logits)
    attributions: List[PosteriorAttribution] = []
    for event, new_value in zip(events, posterior):
        previous = old.get(event.event_id, 0.0)
        event.posterior = float(new_value)
        _update_status_distribution(event, weights)
        for factor, delta in sorted(event.factors.items()):
            factor_state = event.factor_states.get(factor)
            attributions.append(
                PosteriorAttribution(
                    event_id=event.event_id,
                    old_posterior=previous,
                    new_posterior=float(new_value),
                    factor_name=factor,
                    factor_delta=float(delta),
                    evidence_ids=list(factor_state.evidence_ids) if factor_state else [],
                )
            )
    return attributions


def posterior_vector(events: Sequence[FaultEvent], entities: Sequence[str]) -> np.ndarray:
    raw = np.zeros(len(entities), dtype=float)
    index = {entity: idx for idx, entity in enumerate(entities)}
    for event in events:
        if event.status in {"merged", "duplicate"}:
            continue
        idx = index.get(event.component)
        if idx is None:
            continue
        root_prob = max(float(event.status_probs.get("root", 0.0)), 0.05)
        unresolved = float(event.status_probs.get("unresolved", 0.0))
        role = root_role_features(event)
        source_margin = max(0.0, role["source_margin"])
        anti_hotspot = max(0.0, role["source_isolation"] - role["hotspot_symptom"])
        status_gate = max(root_prob, 0.25 * unresolved)
        status_gate += 0.20 * source_margin + 0.25 * anti_hotspot
        status_gate += 0.18 * max(0.0, role["mechanism_break_likelihood"])
        status_gate += 0.12 * max(0.0, role["mechanism_parent_refutation"])
        status_gate += 0.10 * max(0.0, role["intervention_uniqueness"])
        status_gate -= 0.18 * max(0.0, -role["mechanism_parent_refutation"])
        status_gate -= 0.25 * role["hotspot_symptom"]
        status_gate = max(0.0, status_gate)
        raw[idx] = max(raw[idx], float(event.posterior) * status_gate)
    if float(np.sum(raw)) <= 1e-9:
        return np.full(len(entities), 1.0 / max(1, len(entities)), dtype=float)
    return raw / float(np.sum(raw))


def _event_logit(event: FaultEvent, weights: PosteriorWeights) -> float:
    score = 0.0
    symptom = max(0.0, float(event.factors.get("symptomness", 0.0)))
    source = max(0.0, float(event.factors.get("source_likelihood", 0.0)))
    isolation = max(0.0, float(event.factors.get("source_isolation", 0.0)))
    hotspot = max(0.0, float(event.factors.get("hotspot_symptom", 0.0)))
    symptom_conflict = max(0.0, symptom - source)
    for factor, value in event.factors.items():
        key = FACTOR_TO_WEIGHT.get(factor)
        if not key:
            weight = 0.10
        else:
            weight = float(getattr(weights, key))
        if factor == "symptomness":
            score -= weight * symptom_conflict
        elif factor in {"hotspot_symptom", "broad_explainer", "merge_duplicate"}:
            score -= weight * abs(float(value))
        else:
            score += weight * float(value)
    score += 0.35 * max(0.0, isolation - hotspot)
    score += 0.20 * max(0.0, source - symptom)
    return float(score)


def _softmax(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    shifted = values - np.max(values)
    exp_values = np.exp(shifted)
    total = float(np.sum(exp_values))
    if total <= 1e-9:
        return np.full(values.shape, 1.0 / values.size, dtype=float)
    return exp_values / total


def _update_status_distribution(event: FaultEvent, weights: PosteriorWeights) -> None:
    role = root_role_features(event)
    symptom = role["symptomness"]
    source = role["source_likelihood"]
    isolation = role["source_isolation"]
    hotspot = role["hotspot_symptom"]
    source_margin = role["source_margin"]
    symptom_margin = role["symptom_margin"]
    metric = max(0.0, float(event.factors.get("metric_likelihood", 0.0)))
    log = max(0.0, float(event.factors.get("log_likelihood", 0.0)))
    trace = float(event.factors.get("trace_direction_likelihood", 0.0))
    cf = float(event.factors.get("counterfactual_likelihood", 0.0))
    collapse = float(event.factors.get("residual_collapse", 0.0))
    mechanism = float(event.factors.get("mechanism_break_likelihood", 0.0))
    parent_refute = float(event.factors.get("mechanism_parent_refutation", 0.0))
    intervention = float(event.factors.get("intervention_uniqueness", 0.0))
    pair = float(event.factors.get("pairwise_verdict", 0.0))
    broad = max(0.0, float(event.factors.get("broad_explainer", 0.0)))
    merge = float(event.factors.get("merge_duplicate", 0.0))
    observed = {
        tool
        for tool in event.observed_tools
        if tool not in {"observe_noiselab", "split_event", "merge_events"}
    }
    coverage = min(1.0, len(observed) / 2.0)
    symptom_conflict = max(0.0, symptom - source)

    root_logit = (
        1.10 * source
        + 1.15 * isolation
        + 0.45 * max(0.0, source_margin)
        + 0.30 * metric
        + 0.18 * log
        + 0.35 * max(0.0, trace)
        + 0.35 * max(0.0, cf)
        + 0.65 * max(0.0, collapse)
        + 0.80 * max(0.0, mechanism)
        + 0.45 * max(0.0, parent_refute)
        + 0.35 * max(0.0, intervention)
        + 0.20 * max(0.0, pair)
        - 0.85 * symptom_conflict
        - 0.50 * max(0.0, -collapse)
        - 0.70 * max(0.0, -parent_refute)
        - 1.05 * hotspot
        - 0.55 * broad
    )
    symptom_logit = (
        1.20 * symptom
        + 1.10 * hotspot
        + 0.25 * max(0.0, metric)
        + 0.20 * max(0.0, log)
        + 0.45 * max(0.0, symptom_margin)
        + 0.45 * max(0.0, -collapse)
        + 0.60 * max(0.0, -parent_refute)
        - 0.80 * source
        - 0.65 * isolation
        - 0.45 * max(0.0, mechanism)
        - 0.20 * max(0.0, trace)
    )
    broad_logit = 1.20 * broad + 0.35 * hotspot + 0.25 * max(0.0, log) - 0.35 * source
    duplicate_logit = 2.0 * max(0.0, -merge)
    unresolved_logit = (
        1.15 * (1.0 - coverage)
        + 0.35 * event.status_entropy()
        + 0.25 * max(0.0, 0.20 - abs(source_margin))
    )
    logits = np.array(
        [root_logit, symptom_logit, broad_logit, duplicate_logit, unresolved_logit],
        dtype=float,
    )
    probs = _softmax(logits)
    labels = ["root", "symptom", "broad_explainer", "duplicate", "unresolved"]
    event.status_probs = {label: float(prob) for label, prob in zip(labels, probs)}
    if merge < -0.1 and event.status_probs["duplicate"] >= 0.45:
        event.status = "duplicate"
    elif coverage < 1.0 and event.status_probs["root"] < 0.55:
        event.status = "unresolved"
    else:
        best_label = max(event.status_probs, key=lambda label: event.status_probs[label])
        if best_label == "root":
            event.status = "active" if event.posterior >= 0.08 else "reserve"
        elif best_label == "duplicate":
            event.status = "merged"
        else:
            event.status = best_label


def root_role_features(event: FaultEvent) -> Dict[str, float]:
    source = max(0.0, float(event.factors.get("source_likelihood", 0.0)))
    symptom = max(0.0, float(event.factors.get("symptomness", 0.0)))
    isolation = max(0.0, float(event.factors.get("source_isolation", 0.0)))
    hotspot = max(0.0, float(event.factors.get("hotspot_symptom", 0.0)))
    mechanism = float(event.factors.get("mechanism_break_likelihood", 0.0))
    parent_refute = float(event.factors.get("mechanism_parent_refutation", 0.0))
    intervention = float(event.factors.get("intervention_uniqueness", 0.0))
    return {
        "source_likelihood": source,
        "symptomness": symptom,
        "source_isolation": isolation,
        "hotspot_symptom": hotspot,
        "mechanism_break_likelihood": mechanism,
        "mechanism_parent_refutation": parent_refute,
        "intervention_uniqueness": intervention,
        "source_margin": source + isolation - symptom - hotspot,
        "symptom_margin": symptom + hotspot - source - isolation,
    }


def root_selection_score(event: FaultEvent) -> float:
    role = root_role_features(event)
    root = float(event.status_probs.get("root", 0.0))
    unresolved = float(event.status_probs.get("unresolved", 0.0))
    symptom = float(event.status_probs.get("symptom", 0.0))
    broad = float(event.status_probs.get("broad_explainer", 0.0))
    trace = max(0.0, float(event.factors.get("trace_direction_likelihood", 0.0)))
    cf = max(0.0, float(event.factors.get("counterfactual_likelihood", 0.0)))
    collapse = max(0.0, float(event.factors.get("residual_collapse", 0.0)))
    mechanism = max(0.0, role["mechanism_break_likelihood"])
    parent_support = max(0.0, role["mechanism_parent_refutation"])
    parent_explained = max(0.0, -role["mechanism_parent_refutation"])
    intervention = max(0.0, role["intervention_uniqueness"])
    source_margin = max(0.0, role["source_margin"])
    anti_hotspot = max(0.0, role["source_isolation"] - role["hotspot_symptom"])
    root_role = (
        root
        + 0.45 * role["source_isolation"]
        + 0.25 * source_margin
        + 0.15 * trace
        + 0.18 * cf
        + 0.32 * collapse
        + 0.38 * mechanism
        + 0.22 * parent_support
        + 0.18 * intervention
        + 0.15 * max(0.0, role["source_likelihood"] - role["symptomness"])
    )
    penalty = (
        0.55 * role["hotspot_symptom"]
        + 0.25 * broad
        + 0.20 * max(0.0, symptom - root)
        + 0.10 * max(0.0, role["symptomness"] - role["source_likelihood"])
        + 0.25 * parent_explained
    )
    posterior_term = float(event.posterior) * max(0.0, root_role - penalty)
    role_floor = (
        0.08 * anti_hotspot
        + 0.04 * source_margin
        + 0.06 * collapse
        + 0.06 * mechanism
        + 0.03 * intervention
        + 0.02 * unresolved
    )
    return max(0.0, posterior_term + role_floor)
