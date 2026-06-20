"""Regression tests for PRISM-CHT candidate recall, TT entry inference,
final-decision global_rescue, and the structural fixes for the
candidate-recall cliff documented in PRISM_CHT_HANDOFF.md.

These tests use synthetic telemetry DataFrames (no LLM, no provider env)
to keep them deterministic and fast.
"""

from __future__ import annotations

import pandas as pd
import pytest

from prismv4.experiments.rcaeval_adapter import (
    RCAEvalTelemetryStore,
    _infer_entry_components,
)
from prismv4.prism_cht.case_types import GenericRCACase, ObservedComponent
from prismv4.prism_cht.diagnostic_policy import build_hypotheses_from_case
from prismv4.prism_cht.evidence_graph import EvidenceGraph


def _make_metrics(components_signals, *, n_pre=50, n_post=50, event_time=100.0):
    """Build a metrics DataFrame like RCAEval simple_metrics.csv.

    components_signals: {component: {signal: (base, spike)}}
    """
    rows = []
    times = list(range(-n_pre, n_post))
    for t in times:
        row = {"time": float(event_time + t)}
        for comp, signals in components_signals.items():
            for sig, (base, spike) in signals.items():
                col = f"{comp}_{sig}"
                val = base
                if t >= 0:
                    val = spike
                row[col] = float(val)
        rows.append(row)
    return pd.DataFrame(rows)


def _make_traces(edges, *, event_time=100.0):
    """Build a traces DataFrame like RCAEval traces.csv.

    edges: list of (trace_id, span_id, parent_id, service, t_offset)
    """
    rows = []
    for trace_id, span_id, parent_id, service, t_offset in edges:
        rows.append(
            {
                "traceID": trace_id,
                "spanID": span_id,
                "parentSpanID": parent_id,
                "serviceName": service,
                "startTimeMillis": int((event_time + t_offset) * 1000),
                "duration": 100,
                "statusCode": "200",
            }
        )
    return pd.DataFrame(rows)


def _make_store(metrics, traces=None, logs=None, event_time=100.0):
    return RCAEvalTelemetryStore(
        metrics=metrics,
        logs=logs,
        traces=traces,
        event_time=event_time,
    )


# ---------------------------------------------------------------------------
# Candidate recall pool
# ---------------------------------------------------------------------------


def test_tiered_recall_pool_includes_low_magnitude_source():
    """The true root has smaller local magnitude than a noisy downstream
    service, but the tiered pool must still include it (the original
    failure mode from the handoff)."""
    metrics = _make_metrics(
        {
            "source-svc": {"cpu": (10.0, 25.0)},  # small magnitude source
            "downstream-svc": {
                "latency-90": (50.0, 5000.0),  # huge downstream symptom
                "request-total": (100.0, 50000.0),  # workload dominates
            },
            "other-svc": {"memory": (40.0, 45.0)},
        }
    )
    store = _make_store(metrics)
    pool = store.build_tiered_recall_pool(pool_size=15)
    assert "source-svc" in pool, "tiered pool dropped the low-magnitude source"
    assert "downstream-svc" in pool


def test_tiered_recall_pool_down_weights_workload():
    """Workload/request-total should not dominate pool ranking."""
    metrics = _make_metrics(
        {
            "real-root": {"cpu": (10.0, 30.0), "error": (0.0, 10.0)},
            "workload-noise": {"request-total": (100.0, 500000.0)},
        }
    )
    store = _make_store(metrics)
    pool = store.build_tiered_recall_pool(pool_size=15)
    assert "real-root" in pool
    assert "workload-noise" in pool
    # real-root (mechanism signals) should rank before workload-noise
    assert pool.index("real-root") < pool.index("workload-noise")


def test_tiered_recall_pool_size_cap():
    metrics = _make_metrics(
        {f"svc-{i}": {"cpu": (10.0, 10.0 + i)} for i in range(20)}
    )
    store = _make_store(metrics)
    pool = store.build_tiered_recall_pool(pool_size=5)
    assert len(pool) <= 5


def test_tiered_recall_pool_dedup():
    metrics = _make_metrics(
        {"svc-a": {"cpu": (10.0, 50.0)}, "svc-b": {"memory": (40.0, 80.0)}}
    )
    store = _make_store(metrics)
    pool = store.build_tiered_recall_pool(pool_size=10)
    assert len(pool) == len(set(pool)), "pool contains duplicates"


def test_build_observations_caches_results():
    """build_observations should cache and not recompute on repeat calls."""
    metrics = _make_metrics(
        {"svc-a": {"cpu": (10.0, 50.0)}, "svc-b": {"memory": (40.0, 80.0)}}
    )
    store = _make_store(metrics)
    obs1 = store.build_observations(top_k=2)
    obs2 = store.build_observations(top_k=2)
    assert obs1 == obs2
    # cache should make a smaller top_k cheap
    obs_small = store.build_observations(top_k=1)
    assert len(obs_small) == 1


# ---------------------------------------------------------------------------
# build_hypotheses_from_case with explicit candidates
# ---------------------------------------------------------------------------


def _make_case(components, observations=None, entry_components=None):
    if observations is None:
        observations = tuple(
            ObservedComponent(
                component=c,
                reason_family="unspecified anomaly",
                first_seen=100.0,
                magnitude=10.0,
                signals=("cpu",),
                symptoms=("anomaly",),
            )
            for c in components
        )
    if entry_components is None:
        entry_components = (components[0],)
    return GenericRCACase(
        case_id="synth/case/1",
        dataset_name="synth",
        system_name="synth",
        event_time=100.0,
        components=tuple(components),
        entry_components=entry_components,
        observations=observations,
    )


def test_build_hypotheses_uses_explicit_candidates_order():
    """When candidates are given, hypotheses follow the candidate order,
    not magnitude top-k."""
    components = ["svc-a", "svc-b", "svc-c", "svc-d", "svc-e", "svc-f"]
    observations = tuple(
        ObservedComponent(
            component=c,
            reason_family="anomaly",
            first_seen=100.0,
            magnitude=10.0 * (ord(c[-1]) - ord("a")),
            signals=("cpu",),
            symptoms=("anomaly",),
        )
        for c in components
    )
    case = _make_case(components, observations=observations)
    # candidates reverse the magnitude order
    candidates = ("svc-f", "svc-a", "svc-e", "svc-b", "svc-d", "svc-c")
    bundle = build_hypotheses_from_case(case, max_hypotheses=3, candidates=candidates)
    roots = [h.root_component for h in bundle.hypotheses]
    assert roots == ["svc-f", "svc-a", "svc-e"], roots


def test_build_hypotheses_candidates_without_observation_get_placeholder():
    """A candidate with no observation should still become a hypothesis
    (recall must not be lost due to missing telemetry)."""
    components = ["svc-a", "svc-b", "svc-c"]
    observations = (
        ObservedComponent(
            component="svc-a",
            reason_family="anomaly",
            first_seen=100.0,
            magnitude=50.0,
            signals=("cpu",),
            symptoms=("anomaly",),
        ),
    )
    case = _make_case(components, observations=observations)
    bundle = build_hypotheses_from_case(
        case, max_hypotheses=3, candidates=("svc-a", "svc-b", "svc-c")
    )
    roots = {h.root_component for h in bundle.hypotheses}
    assert roots == {"svc-a", "svc-b", "svc-c"}


def test_build_hypotheses_backward_compatible_without_candidates():
    """Without candidates, legacy magnitude top-k behaviour is preserved."""
    components = ["svc-a", "svc-b", "svc-c"]
    observations = tuple(
        ObservedComponent(
            component=c,
            reason_family="anomaly",
            first_seen=100.0,
            magnitude=10.0 * (ord(c[-1]) - ord("a") + 1),
            signals=("cpu",),
            symptoms=("anomaly",),
        )
        for c in components
    )
    case = _make_case(components, observations=observations)
    bundle = build_hypotheses_from_case(case, max_hypotheses=2)
    roots = [h.root_component for h in bundle.hypotheses]
    assert roots == ["svc-c", "svc-b"], roots


# ---------------------------------------------------------------------------
# TT entry-component inference
# ---------------------------------------------------------------------------


def test_entry_inference_prefers_named_entry():
    metrics = _make_metrics(
        {"frontend-svc": {"cpu": (10.0, 50.0)}, "svc-a": {"cpu": (10.0, 50.0)}}
    )
    traces = _make_traces(
        [("t1", "s1", "", "frontend-svc", 1.0), ("t1", "s2", "s1", "svc-a", 1.0)]
    )
    store = _make_store(metrics, traces=traces)
    entries = _infer_entry_components(
        components=("frontend-svc", "svc-a"), store=store, event_time=100.0
    )
    assert entries == ("frontend-svc",)


def test_entry_inference_uses_trace_topology_when_no_named_entry():
    """When no frontend/gateway name exists, entry inference should use
    trace topology roots (callers with no callers but with callees),
    not fall back to the first sorted component."""
    metrics = _make_metrics(
        {
            "ts-admin-basic-info-service": {"cpu": (10.0, 50.0)},
            "ts-preserve-service": {"cpu": (10.0, 50.0)},
            "ts-food-service": {"cpu": (10.0, 50.0)},
        }
    )
    # ts-preserve-service is a trace root (no parent), calls ts-food-service.
    # ts-admin-basic-info-service is isolated (no edges).
    traces = _make_traces(
        [
            ("t1", "s1", "", "ts-preserve-service", 1.0),
            ("t1", "s2", "s1", "ts-food-service", 1.0),
        ]
    )
    store = _make_store(metrics, traces=traces)
    components = (
        "ts-admin-basic-info-service",
        "ts-food-service",
        "ts-preserve-service",
    )
    entries = _infer_entry_components(
        components=components, store=store, event_time=100.0
    )
    # Should pick the trace root, NOT the first sorted component
    assert "ts-preserve-service" in entries
    assert "ts-admin-basic-info-service" not in entries or entries[0] != "ts-admin-basic-info-service"


def test_entry_inference_falls_back_to_first_sorted_when_no_traces():
    metrics = _make_metrics({"svc-a": {"cpu": (10.0, 50.0)}})
    store = _make_store(metrics, traces=None)
    entries = _infer_entry_components(
        components=("svc-a",), store=store, event_time=100.0
    )
    assert entries == ("svc-a",)


# ---------------------------------------------------------------------------
# global_rescue / final-decision consistency
# ---------------------------------------------------------------------------


def _make_hypothesis_bundle(components, max_hypotheses=3):
    case = _make_case(components)
    return build_hypotheses_from_case(case, max_hypotheses=max_hypotheses)


def test_global_rescue_flags_out_of_set_nomination():
    from prismv4.experiments.run_rcaeval_continuous import _normalize_final_decision

    components = ["svc-a", "svc-b", "svc-c", "svc-d"]
    case = _make_case(components)
    bundle = _make_hypothesis_bundle(components, max_hypotheses=2)
    active = {h.root_component for h in bundle.hypotheses}
    outside = [c for c in components if c not in active][0]
    graph = EvidenceGraph()
    decision = _normalize_final_decision(
        {"root_component": outside, "rationale": "jumped outside"},
        case=case,
        hypotheses=bundle.hypotheses,
        graph=graph,
        context_state={},
        recall_pool=tuple(components),
    )
    assert decision["global_rescue"] is True
    assert decision["outside_hypothesis_set"] is True
    assert "global_rescue" in decision["rationale"]


def test_in_set_nomination_not_flagged_as_rescue():
    from prismv4.experiments.run_rcaeval_continuous import _normalize_final_decision

    components = ["svc-a", "svc-b", "svc-c"]
    case = _make_case(components)
    bundle = _make_hypothesis_bundle(components)
    active = [h.root_component for h in bundle.hypotheses]
    inside = active[0]
    graph = EvidenceGraph()
    decision = _normalize_final_decision(
        {"root_component": inside, "rationale": "normal pick"},
        case=case,
        hypotheses=bundle.hypotheses,
        graph=graph,
        context_state={},
        recall_pool=tuple(components),
    )
    assert decision.get("global_rescue") is not True
    assert "global_rescue" not in decision["rationale"]


def test_invalid_component_falls_back_to_leading_component():
    from prismv4.experiments.run_rcaeval_continuous import _normalize_final_decision

    components = ["svc-a", "svc-b"]
    case = _make_case(components)
    bundle = _make_hypothesis_bundle(components)
    graph = EvidenceGraph()
    decision = _normalize_final_decision(
        {"root_component": "does-not-exist", "rationale": ""},
        case=case,
        hypotheses=bundle.hypotheses,
        graph=graph,
        context_state={},
        recall_pool=tuple(components),
    )
    # falls back to a valid component in the case
    assert decision["root_component"] in components


# ---------------------------------------------------------------------------
# Leakage guard (no label leakage in recall pool / entry inference)
# ---------------------------------------------------------------------------


def test_recall_pool_does_not_embed_expected_labels():
    """The recall pool is a list of component names only; it must never
    contain fault names, case ids, or expected-label fields."""
    metrics = _make_metrics(
        {"ts-auth-service": {"cpu": (10.0, 50.0)}, "ts-contacts-service": {"cpu": (10.0, 50.0)}}
    )
    store = _make_store(metrics)
    pool = store.build_tiered_recall_pool(pool_size=10)
    for item in pool:
        assert "_f" not in item, f"fault-name fragment leaked into pool: {item}"
        assert "RE3" not in item
        assert "/" not in item


def test_entry_inference_does_not_embed_fault_names():
    metrics = _make_metrics({"svc-a": {"cpu": (10.0, 50.0)}})
    store = _make_store(metrics)
    entries = _infer_entry_components(
        components=("svc-a",), store=store, event_time=100.0
    )
    for item in entries:
        assert "_f" not in item
        assert "/" not in item


# ---------------------------------------------------------------------------
# CostWindow / token monitoring
# ---------------------------------------------------------------------------


def test_cost_window_aggregates_usage():
    from prismv4.prism_cht.llm_audit import CostWindow

    cw = CostWindow()
    cw.record(
        purpose="event_causalizer",
        usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
    )
    cw.record(
        purpose="agent_state_update",
        usage={"prompt_tokens": 200, "completion_tokens": 80, "total_tokens": 280},
    )
    snap = cw.snapshot()
    assert snap["call_count"] == 2
    assert snap["prompt_tokens"] == 300
    assert snap["completion_tokens"] == 130
    assert snap["total_tokens"] == 430
    assert "event_causalizer" in snap["by_purpose"]
    assert snap["by_purpose"]["event_causalizer"]["total_tokens"] == 150


def test_cost_window_handles_missing_total():
    from prismv4.prism_cht.llm_audit import CostWindow

    cw = CostWindow()
    cw.record(
        purpose="test",
        usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": None},
    )
    snap = cw.snapshot()
    assert snap["total_tokens"] == 150  # derived from prompt + completion


def test_cost_window_handles_none_usage():
    from prismv4.prism_cht.llm_audit import CostWindow

    cw = CostWindow()
    cw.record(purpose="test", usage=None)
    snap = cw.snapshot()
    assert snap["call_count"] == 1
    assert snap["total_tokens"] == 0


def test_model_response_usage_backward_compatible():
    from prismv4.prism_cht.llm_types import ModelResponse

    r = ModelResponse(content="hello")
    assert r.usage is None
    r2 = ModelResponse(content="hello", usage={"prompt_tokens": 10})
    assert r2.usage is not None
    assert r2.usage["prompt_tokens"] == 10
