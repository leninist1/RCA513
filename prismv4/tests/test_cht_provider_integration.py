"""Integration test: full pipeline with provider client + FakeTransport.

Runs the complete Lead → Challenger → FinalVerifier pipeline using
``OpenAICompatibleChatModelClient`` with ``FakeTransport``, verifying
the full integration of the provider adapter with structured policies.
"""

import json
import copy

import pytest

from prismv4.prism_cht.action_gate import ActionGate
from prismv4.prism_cht.assessment_gate import EvidenceAssessmentGate
from prismv4.prism_cht.challenge_gate import ChallengeReviewGate
from prismv4.prism_cht.challenger_controller import ChallengerController
from prismv4.prism_cht.evidence_graph import EvidenceGraph
from prismv4.prism_cht.executor import InvestigationExecutor
from prismv4.prism_cht.hypothesis import CausalHypothesis, HypothesisStatus
from prismv4.prism_cht.lead_controller import LeadTournamentController
from prismv4.prism_cht.llm_policy import (
    StructuredLLMLeadPolicy,
    StructuredLLMChallengerPolicy,
)
from prismv4.prism_cht.final_verifier import FinalVerifier
from prismv4.prism_cht.final_types import FinalRCAResult
from prismv4.prism_cht.telemetry_store import (
    MockTelemetryStore,
    OnsetObservation,
    TraceHop,
    TracePath,
)
from prismv4.prism_cht.tool_registry import build_default_tool_registry
from prismv4.prism_cht.action_schema import DiscriminativeAction
from prismv4.prism_cht.provider_config import OpenAICompatibleChatConfig
from prismv4.prism_cht.provider_types import (
    ProviderHTTPError,
    ProviderResponseError,
)
from prismv4.prism_cht.http_transport import HttpResponse
from prismv4.prism_cht.openai_compatible_client import (
    OpenAICompatibleChatModelClient,
)
from prismv4.prism_cht.llm_types import (
    ModelClient,
    ModelRequest,
    ModelResponse,
    ModelMessage,
    StructuredOutputError,
)
from prismv4.prism_cht.llm_json import (
    parse_lead_policy_decision,
    parse_evidence_assessment,
    parse_challenge_proposal,
    parse_challenge_resolution,
)
from prismv4.prism_cht.llm_prompts import (
    build_lead_decision_request,
    build_lead_assessment_request,
    build_challenge_proposal_request,
    build_challenge_resolution_request,
)


# ===========================================================================
# FakeTransport (fifo, deterministic)
# ===========================================================================


class FakeTransport:
    def __init__(self, responses):
        import collections
        self._queue = collections.deque(responses)
        self._requests = []

    def send(self, *, request, max_response_bytes):
        self._requests.append(request)
        if not self._queue:
            raise RuntimeError("FakeTransport: no more responses queued")
        item = self._queue.popleft()
        if isinstance(item, Exception):
            raise item
        return item

    @property
    def requests(self):
        return tuple(self._requests)


# ===========================================================================
# Demo hypotheses and store
# ===========================================================================


def _build_hypotheses():
    h1 = CausalHypothesis(
        hypothesis_id="H-os_009-cpu",
        root_component="os_009",
        reason_family="high cpu usage",
        onset_interval=(1000.0, 2000.0),
        local_trigger="os_009 CPU utilization exceeds 95% threshold",
        propagation_path=["os_009", "app_003"],
        explained_symptoms=[
            "app_003 latency spike",
            "app_003 error rate increase",
        ],
        predicted_observations=[
            "os_009 CPU peak is the earliest anomaly in the system",
            "app_003 symptoms are downstream of os_009",
        ],
        falsifiers=[
            "A different component (e.g. db_002) has an earlier anomaly onset",
            "No trace path exists from os_009 to app_003",
        ],
    )
    h2 = CausalHypothesis(
        hypothesis_id="H-db_002-pool",
        root_component="db_002",
        reason_family="connection pool exhaustion",
        onset_interval=(1000.0, 2000.0),
        local_trigger="db_002 connection pool exhausted, queued requests > 100",
        propagation_path=["db_002", "app_003"],
        explained_symptoms=[
            "app_003 latency spike",
            "app_003 error rate increase",
        ],
        predicted_observations=[
            "db_002 connection pool anomaly onset precedes os_009 CPU spike",
            "Explicit trace path exists from db_002 to app_003",
        ],
        falsifiers=[
            "os_009 CPU anomaly precedes db_002 pool exhaustion",
            "No trace path exists from db_002 to app_003",
        ],
    )
    return h1, h2


def _build_store():
    onset_observations = [
        OnsetObservation("db_002", "connection_pool_queued", 1100.0, "prom_db"),
        OnsetObservation("db_002", "connection_pool_errors", 1110.0, "prom_db"),
        OnsetObservation("os_009", "cpu", 1250.0, "prom"),
        OnsetObservation("os_009", "load_average", 1260.0, "prom"),
        OnsetObservation("app_003", "latency_p99", 1300.0, "prom_app"),
    ]
    trace_paths = [
        TracePath(
            hops=(
                TraceHop(
                    source_component="db_002",
                    target_component="app_003",
                    timestamp=1120.0,
                    latency_ms=45.0,
                    status="error",
                ),
            )
        )
    ]
    raw_records = [
        {
            "modality": "metrics",
            "component": "db_002",
            "timestamp": 1100.0,
            "payload": {
                "connection_pool_queued": 150,
                "connection_pool_active": 0,
                "connection_pool_errors": 12,
                "status": "exhausted",
                "duration_seconds": 300,
            },
        },
        {
            "modality": "metrics",
            "component": "db_002",
            "timestamp": 1200.0,
            "payload": {
                "connection_pool_queued": 200,
                "connection_pool_active": 0,
                "connection_pool_errors": 25,
                "status": "exhausted",
                "duration_seconds": 200,
            },
        },
    ]
    return MockTelemetryStore(
        onset_observations=onset_observations,
        trace_paths=trace_paths,
        raw_records=raw_records,
    )


def _build_infrastructure(store):
    gate = ActionGate()
    registry = build_default_tool_registry()
    graph = EvidenceGraph()
    assessment_gate = EvidenceAssessmentGate()
    executor = InvestigationExecutor(
        gate=gate, registry=registry, graph=graph, store=store
    )
    return gate, registry, graph, assessment_gate, executor


# ===========================================================================
# Pre-compute evidence IDs
# ===========================================================================


def _precompute_evidence_ids():
    onset_sig_r1 = DiscriminativeAction(
        action_id="R1-compare-onset",
        action_type="run_discriminative_test",
        target_hypothesis_ids=("H-os_009-cpu", "H-db_002-pool"),
        question="onset order question",
        tool_name="compare_onset_order",
        args={
            "component_scope": ["db_002", "os_009", "app_003"],
            "signal_scope": [
                "connection_pool_queued", "connection_pool_errors",
                "cpu", "load_average", "latency_p99",
            ],
            "time_window": [1000.0, 2000.0],
        },
        expected_outcomes={"H-os_009-cpu": "a", "H-db_002-pool": "b"},
        why_discriminative="onset order",
    ).query_signature()
    eid_r1 = "evidence:" + onset_sig_r1

    trace_sig_r2 = DiscriminativeAction(
        action_id="R2-inspect-trace",
        action_type="run_discriminative_test",
        target_hypothesis_ids=("H-os_009-cpu", "H-db_002-pool"),
        question="trace question",
        tool_name="inspect_trace_path",
        args={
            "source_component": "db_002",
            "target_component": "app_003",
            "time_window": [1000.0, 2000.0],
            "max_hops": 4,
            "max_paths": 10,
        },
        expected_outcomes={"H-os_009-cpu": "a", "H-db_002-pool": "b"},
        why_discriminative="trace path",
    ).query_signature()
    eid_r2 = "evidence:" + trace_sig_r2

    raw_sig_c1 = DiscriminativeAction(
        action_id="C1-retrieve-raw",
        action_type="run_discriminative_test",
        target_hypothesis_ids=("H-db_002-pool", "H-os_009-cpu"),
        question="raw records question",
        tool_name="retrieve_raw_evidence",
        args={
            "modality": "metrics",
            "component_scope": ["db_002"],
            "time_window": [1000.0, 2000.0],
            "limit": 20,
        },
        expected_outcomes={"H-db_002-pool": "a", "H-os_009-cpu": "b"},
        why_discriminative="raw records",
    ).query_signature()
    eid_c1 = "evidence:" + raw_sig_c1

    return eid_r1, eid_r2, eid_c1


# ===========================================================================
# Build fake JSON responses
# ===========================================================================


def _build_fake_responses(eid_r1, eid_r2, eid_c1):
    responses = []

    # 1. Lead Round 1 decide_next → compare_onset_order action
    responses.append(json.dumps({
        "kind": "action",
        "payload": {
            "action_id": "R1-compare-onset",
            "action_type": "run_discriminative_test",
            "target_hypothesis_ids": ["H-os_009-cpu", "H-db_002-pool"],
            "question": "Which root-cause candidate exhibits the earliest observable anomaly onset?",
            "tool_name": "compare_onset_order",
            "args": {
                "component_scope": ["db_002", "os_009", "app_003"],
                "signal_scope": [
                    "connection_pool_queued", "connection_pool_errors",
                    "cpu", "load_average", "latency_p99",
                ],
                "time_window": [1000.0, 2000.0],
            },
            "expected_outcomes": {
                "H-os_009-cpu": "os_009 CPU spike is the earliest anomaly",
                "H-db_002-pool": "db_002 connection pool exhaustion is the earliest anomaly",
            },
            "why_discriminative": "onset order is discriminative between the two hypotheses",
        },
    }))

    # 2. Lead Round 1 assess_evidence
    responses.append(json.dumps({
        "action_id": "R1-compare-onset",
        "evidence_id": eid_r1,
        "outcome": "informative",
        "links": [
            {
                "hypothesis_id": "H-db_002-pool",
                "evidence_id": eid_r1,
                "relation": "supports",
                "rationale": "db_002 connection pool onset precedes os_009 CPU onset",
            },
            {
                "hypothesis_id": "H-os_009-cpu",
                "evidence_id": eid_r1,
                "relation": "contradicts",
                "rationale": "os_009 predicted earliest anomaly but db_002 onset is earlier",
            },
        ],
        "status_updates": [
            {
                "hypothesis_id": "H-db_002-pool",
                "new_status": "supported",
                "rationale": "db_002 connection pool exhaustion is the earliest observable anomaly",
            },
            {
                "hypothesis_id": "H-os_009-cpu",
                "new_status": "weakened",
                "rationale": "os_009 predicted earliest anomaly is contradicted",
            },
        ],
        "rationale": "Onset order comparison reveals db_002 is earlier",
    }))

    # 3. Lead Round 2 decide_next → inspect_trace_path action
    responses.append(json.dumps({
        "kind": "action",
        "payload": {
            "action_id": "R2-inspect-trace",
            "action_type": "run_discriminative_test",
            "target_hypothesis_ids": ["H-os_009-cpu", "H-db_002-pool"],
            "question": "Is there a trace path from db_002 to app_003?",
            "tool_name": "inspect_trace_path",
            "args": {
                "source_component": "db_002",
                "target_component": "app_003",
                "time_window": [1000.0, 2000.0],
                "max_hops": 4,
                "max_paths": 10,
            },
            "expected_outcomes": {
                "H-os_009-cpu": "No trace path from db_002 to app_003",
                "H-db_002-pool": "Trace path exists from db_002 to app_003",
            },
            "why_discriminative": "db_002 hypothesis requires explicit trace path",
        },
    }))

    # 4. Lead Round 2 assess_evidence
    responses.append(json.dumps({
        "action_id": "R2-inspect-trace",
        "evidence_id": eid_r2,
        "outcome": "informative",
        "links": [
            {
                "hypothesis_id": "H-db_002-pool",
                "evidence_id": eid_r2,
                "relation": "supports",
                "rationale": "Explicit trace path db_002 -> app_003 found",
            },
        ],
        "status_updates": [],
        "rationale": "Trace inspection confirms direct path from db_002 to app_003",
    }))

    # 5. Lead decide_next → H2 nomination with triplet grounding
    responses.append(json.dumps({
        "kind": "nomination",
        "payload": {
            "hypothesis_id": "H-db_002-pool",
            "supporting_evidence_ids": [eid_r1, eid_r2],
            "addressed_competitor_ids": ["H-os_009-cpu"],
            "triplet_grounding": {
                "component_evidence_ids": [eid_r2],
                "reason_evidence_ids": [eid_r1],
                "onset_evidence_ids": [eid_r1],
            },
            "rationale": "db_002 connection pool exhaustion is nominated as root cause",
        },
    }))

    # 6. Challenge propose_challenge
    responses.append(json.dumps({
        "challenge_id": "C1",
        "nominated_hypothesis_id": "H-db_002-pool",
        "competitor_hypothesis_ids": ["H-os_009-cpu"],
        "challenge_claim": "db_002 connection pool may be transient",
        "falsification_target": "Show db_002 pool exhaustion was sustained",
        "action": {
            "action_id": "C1-retrieve-raw",
            "action_type": "run_discriminative_test",
            "target_hypothesis_ids": ["H-db_002-pool", "H-os_009-cpu"],
            "question": "Do raw records show sustained pool exhaustion?",
            "tool_name": "retrieve_raw_evidence",
            "args": {
                "modality": "metrics",
                "component_scope": ["db_002"],
                "time_window": [1000.0, 2000.0],
                "limit": 20,
            },
            "expected_outcomes": {
                "H-db_002-pool": "Raw records confirm sustained pool exhaustion",
                "H-os_009-cpu": "Raw records show pool was not exhausted",
            },
            "why_discriminative": "Raw records reveal whether pool exhaustion is sustained",
        },
        "rationale": "Raw telemetry records for db_002 will reveal whether pool exhaustion was sustained",
    }))

    # 7. Challenge assess_challenge → nomination_survived
    responses.append(json.dumps({
        "challenge_id": "C1",
        "action_id": "C1-retrieve-raw",
        "evidence_id": eid_c1,
        "verdict": "nomination_survived",
        "assessment": {
            "action_id": "C1-retrieve-raw",
            "evidence_id": eid_c1,
            "outcome": "informative",
            "links": [
                {
                    "hypothesis_id": "H-db_002-pool",
                    "evidence_id": eid_c1,
                    "relation": "supports",
                    "rationale": "Raw records show sustained pool exhaustion",
                },
            ],
            "status_updates": [
                {
                    "hypothesis_id": "H-db_002-pool",
                    "new_status": "survived",
                    "rationale": "Challenge attempted to falsify but raw records confirm sustained exhaustion",
                },
            ],
            "rationale": "Raw records confirm sustained connection pool exhaustion",
        },
        "rationale": "Evidence confirms sustained pool exhaustion",
    }))

    return responses


# ===========================================================================
# Build provider-compatible HTTP responses from JSON strings
# ===========================================================================


def _json_to_http_response(json_str):
    body = json.dumps({
        "choices": [{"message": {"content": json_str}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }).encode("utf-8")
    return HttpResponse(
        status_code=200,
        headers={"Content-Type": "application/json"},
        body=body,
    )


def _make_provider_config():
    return OpenAICompatibleChatConfig(
        base_url="https://api.example.com",
        model="deepseek-v4-pro",
        api_key="test-secret-not-real",
        timeout_seconds=30.0,
        max_tokens=4096,
        max_response_bytes=2_000_000,
        json_mode=True,
    )


# ===========================================================================
# Run full pipeline
# ===========================================================================


def _run_full_pipeline_provider():
    h1, h2 = _build_hypotheses()
    store = _build_store()
    gate, registry, graph, assessment_gate, executor = _build_infrastructure(store)

    lead_ctrl = LeadTournamentController(
        executor=executor,
        assessment_gate=assessment_gate,
        graph=graph,
        max_rounds=4,
    )

    eid_r1, eid_r2, eid_c1 = _precompute_evidence_ids()
    response_strs = _build_fake_responses(eid_r1, eid_r2, eid_c1)
    http_responses = [_json_to_http_response(s) for s in response_strs]

    config = _make_provider_config()

    # Lead client: first 5 responses
    lead_transport = FakeTransport(http_responses[:5])
    lead_client = OpenAICompatibleChatModelClient(config=config, transport=lead_transport)
    lead_policy = StructuredLLMLeadPolicy(client=lead_client, graph=graph)

    lead_result = lead_ctrl.run(
        initial_hypotheses=(h1, h2),
        policy=lead_policy,
    )

    # Challenger: last 2 responses
    challenge_transport = FakeTransport(http_responses[5:])
    chall_client = OpenAICompatibleChatModelClient(config=config, transport=challenge_transport)
    chall_policy = StructuredLLMChallengerPolicy(client=chall_client, graph=graph)

    challenge_gate = ChallengeReviewGate()
    challenge_executor = InvestigationExecutor(
        gate=gate, registry=registry, graph=graph, store=store
    )
    challenge_ctrl = ChallengerController(
        executor=challenge_executor,
        assessment_gate=assessment_gate,
        challenge_gate=challenge_gate,
        graph=graph,
    )

    challenge_result = challenge_ctrl.run(
        lead_result=lead_result,
        hypotheses={"H-os_009-cpu": h1, "H-db_002-pool": h2},
        policy=chall_policy,
    )

    verifier = FinalVerifier()
    final_result = verifier.finalize(
        lead_result=lead_result,
        challenge_result=challenge_result,
        hypotheses={"H-os_009-cpu": h1, "H-db_002-pool": h2},
        graph=graph,
    )

    return (
        lead_result, challenge_result, final_result, graph,
        lead_client, chall_client, lead_transport, challenge_transport,
    )


# ===========================================================================
# Integration tests
# ===========================================================================


class TestFullProviderIntegration:
    def test_full_pipeline_outputs_final_verified(self):
        _, _, final_result, *_ = _run_full_pipeline_provider()
        assert final_result.status == "final_verified"

    def test_root_component_is_db_002(self):
        _, _, final_result, *_ = _run_full_pipeline_provider()
        assert final_result.root_component == "db_002"

    def test_reason_family_is_connection_pool_exhaustion(self):
        _, _, final_result, *_ = _run_full_pipeline_provider()
        assert final_result.reason_family == "connection pool exhaustion"

    def test_h2_final_status_is_final(self):
        _, _, _, graph, *_ = _run_full_pipeline_provider()
        h2 = graph.hypotheses_by_id["H-db_002-pool"]
        assert h2.status == HypothesisStatus.FINAL

    def test_provider_client_call_count_is_7(self):
        _, _, _, _, lead_client, chall_client, *_ = _run_full_pipeline_provider()
        assert len(lead_client.audit_records) == 5
        assert len(chall_client.audit_records) == 2
        total = len(lead_client.audit_records) + len(chall_client.audit_records)
        assert total == 7

    def test_fake_transport_request_count_is_7(self):
        _, _, _, _, _, _, lead_t, chall_t = _run_full_pipeline_provider()
        assert len(lead_t.requests) == 5
        assert len(chall_t.requests) == 2
        assert len(lead_t.requests) + len(chall_t.requests) == 7

    def test_audit_count_is_7(self):
        _, _, _, _, lead_client, chall_client, *_ = _run_full_pipeline_provider()
        total = len(lead_client.audit_records) + len(chall_client.audit_records)
        assert total == 7

    def test_audit_outcome_all_success(self):
        _, _, _, _, lead_client, chall_client, *_ = _run_full_pipeline_provider()
        for audit in lead_client.audit_records:
            assert audit.outcome == "success"
        for audit in chall_client.audit_records:
            assert audit.outcome == "success"

    def test_lead_request_count_is_5(self):
        _, _, _, _, lead_client, _, *_ = _run_full_pipeline_provider()
        assert len(lead_client.audit_records) == 5

    def test_challenger_request_count_is_2(self):
        _, _, _, _, _, chall_client, *_ = _run_full_pipeline_provider()
        assert len(chall_client.audit_records) == 2

    def test_tool_call_count_is_3(self):
        # Lead R1 + R2 + Challenge C1 = 3 tool calls
        _, _, final_result, *_ = _run_full_pipeline_provider()
        assert len(final_result.referenced_evidence_ids) >= 3

    def test_evidence_atom_count_is_3(self):
        _, _, _, graph, *_ = _run_full_pipeline_provider()
        assert graph.evidence_count() >= 3

    def test_triplet_grounding_preserved(self):
        _, _, final_result, *_ = _run_full_pipeline_provider()
        assert final_result.triplet_grounding is not None
        assert len(final_result.triplet_grounding.component_evidence_ids) >= 1
        assert len(final_result.triplet_grounding.reason_evidence_ids) >= 1
        assert len(final_result.triplet_grounding.onset_evidence_ids) >= 1

    def test_all_evidence_ids_traceable(self):
        _, _, final_result, graph, *_ = _run_full_pipeline_provider()
        for eid in final_result.referenced_evidence_ids:
            assert eid in graph.evidence_by_id

    def test_all_requests_use_stream_false(self):
        _, _, _, _, _, _, lead_t, chall_t = _run_full_pipeline_provider()
        for req in lead_t.requests + chall_t.requests:
            body = json.loads(req.body.decode())
            assert body["stream"] is False

    def test_all_requests_use_response_format_json_object(self):
        _, _, _, _, _, _, lead_t, chall_t = _run_full_pipeline_provider()
        for req in lead_t.requests + chall_t.requests:
            body = json.loads(req.body.decode())
            assert body["response_format"] == {"type": "json_object"}

    def test_all_requests_same_model(self):
        _, _, _, _, _, _, lead_t, chall_t = _run_full_pipeline_provider()
        model = None
        for req in lead_t.requests + chall_t.requests:
            body = json.loads(req.body.decode())
            if model is None:
                model = body["model"]
            else:
                assert body["model"] == model

    def test_all_requests_same_endpoint(self):
        _, _, _, _, _, _, lead_t, chall_t = _run_full_pipeline_provider()
        endpoint = None
        for req in lead_t.requests + chall_t.requests:
            if endpoint is None:
                endpoint = req.url
            else:
                assert req.url == endpoint

    def test_all_requests_exclude_ground_truth(self):
        """Verify requests do not contain ground truth data in body."""
        _, _, _, _, _, _, lead_t, chall_t = _run_full_pipeline_provider()
        for req in lead_t.requests + chall_t.requests:
            body_text = req.body.decode().lower()
            assert "ground_truth" not in body_text
            assert "inject_time" not in body_text

    def test_all_audit_excludes_prompt(self):
        _, _, _, _, lead_client, chall_client, *_ = _run_full_pipeline_provider()
        for audit in lead_client.audit_records + chall_client.audit_records:
            assert not hasattr(audit, "messages")
            assert not hasattr(audit, "prompt")
            assert not hasattr(audit, "request")

    def test_all_audit_excludes_api_key(self):
        _, _, _, _, lead_client, chall_client, *_ = _run_full_pipeline_provider()
        for audit in lead_client.audit_records + chall_client.audit_records:
            assert not hasattr(audit, "api_key")

    def test_two_independent_runs_same_final_result(self):
        r1 = _run_full_pipeline_provider()
        r2 = _run_full_pipeline_provider()

        f1 = r1[2]
        f2 = r2[2]
        assert f1.status == f2.status
        assert f1.hypothesis_id == f2.hypothesis_id
        assert f1.root_component == f2.root_component
        assert f1.reason_family == f2.reason_family
        assert f1.onset_interval == f2.onset_interval

    def test_two_independent_runs_independent_clients(self):
        r1 = _run_full_pipeline_provider()
        r2 = _run_full_pipeline_provider()
        # Different audit objects
        assert r1[4].audit_records is not r2[4].audit_records

    def test_two_independent_runs_independent_audit(self):
        r1 = _run_full_pipeline_provider()
        r2 = _run_full_pipeline_provider()
        assert len(r1[4].audit_records) == len(r2[4].audit_records)
        # Different client instances
        assert r1[4] is not r2[4]

    def test_no_network_access(self):
        """Verified by construction — all responses from FakeTransport."""
        _, _, _, _, _, _, lead_t, chall_t = _run_full_pipeline_provider()
        assert len(lead_t.requests) + len(chall_t.requests) == 7


# ===========================================================================
# Error path tests
# ===========================================================================


class TestErrorPaths:
    def test_invalid_json_then_valid_retry_success(self):
        """First response invalid JSON, second valid → structured retry succeeds."""
        eid_r1, eid_r2, eid_c1 = _precompute_evidence_ids()
        good_json = _build_fake_responses(eid_r1, eid_r2, eid_c1)[0]

        # Bad response first, then good response
        bad_json = json.dumps({"kind": "action", "payload": {"action_id": "R1-compare-onset"}})

        config = _make_provider_config()
        transport = FakeTransport([
            _json_to_http_response(bad_json),
            _json_to_http_response(good_json),
        ])
        client = OpenAICompatibleChatModelClient(config=config, transport=transport)

        from prismv4.prism_cht.llm_policy import StructuredModelRequester
        requester = StructuredModelRequester(client=client, max_attempts=2)

        def builder(*, attempt_index, repair_error=None, **kwargs):
            msgs = [ModelMessage(role="system", content="prompt"), ModelMessage(role="user", content="ctx")]
            if repair_error:
                msgs.append(ModelMessage(role="user", content=f"Repair: {repair_error}"))
            return ModelRequest(purpose="test", messages=tuple(msgs), attempt_index=attempt_index)

        result = requester.request_parsed(
            request_builder=builder,
            parser=parse_lead_policy_decision,
            builder_kwargs={},
        )
        assert isinstance(result, DiscriminativeAction)
        # Two provider calls were made
        assert len(client.audit_records) == 2

    def test_structured_retry_increases_audit_count(self):
        """Structured retry on bad JSON causes more provider calls."""
        eid_r1, eid_r2, eid_c1 = _precompute_evidence_ids()
        good_json = _build_fake_responses(eid_r1, eid_r2, eid_c1)[0]
        bad_json = json.dumps({"kind": "action"})  # Missing payload

        config = _make_provider_config()
        transport = FakeTransport([
            _json_to_http_response(bad_json),
            _json_to_http_response(good_json),
        ])
        client = OpenAICompatibleChatModelClient(config=config, transport=transport)

        from prismv4.prism_cht.llm_policy import StructuredModelRequester
        requester = StructuredModelRequester(client=client, max_attempts=2)

        def builder(*, attempt_index, repair_error=None, **kwargs):
            msgs = [ModelMessage(role="system", content="prompt"), ModelMessage(role="user", content="ctx")]
            if repair_error:
                msgs.append(ModelMessage(role="user", content=f"Repair: {repair_error}"))
            return ModelRequest(purpose="test", messages=tuple(msgs), attempt_index=attempt_index)

        requester.request_parsed(
            request_builder=builder,
            parser=parse_lead_policy_decision,
            builder_kwargs={},
        )
        assert len(client.audit_records) == 2

    def test_http_429_not_auto_retried(self):
        transport = FakeTransport([
            HttpResponse(status_code=429, headers={}, body=b'{"error":"rate limited"}'),
            _json_to_http_response(json.dumps({"kind": "action", "payload": {"action_id": "x"}})),
        ])
        config = _make_provider_config()
        client = OpenAICompatibleChatModelClient(config=config, transport=transport)

        req = ModelRequest(
            purpose="test",
            messages=(ModelMessage(role="system", content="sys"),),
            attempt_index=0,
        )
        with pytest.raises(ProviderHTTPError):
            client.complete(request=req)
        # Only 1 call was made — no auto retry
        assert len(client.audit_records) == 1

    def test_http_429_propagates_upward(self):
        transport = FakeTransport([
            HttpResponse(status_code=429, headers={}, body=b"{}"),
        ])
        config = _make_provider_config()
        client = OpenAICompatibleChatModelClient(config=config, transport=transport)

        req = ModelRequest(
            purpose="test",
            messages=(ModelMessage(role="system", content="sys"),),
            attempt_index=0,
        )
        with pytest.raises(ProviderHTTPError) as exc:
            client.complete(request=req)
        assert exc.value.status_code == 429

    def test_finish_reason_length_error_propagates(self):
        body = json.dumps({
            "choices": [
                {"message": {"content": "truncated"}, "finish_reason": "length"}
            ]
        }).encode()
        transport = FakeTransport([HttpResponse(status_code=200, headers={}, body=body)])
        config = _make_provider_config()
        client = OpenAICompatibleChatModelClient(config=config, transport=transport)

        req = ModelRequest(
            purpose="test",
            messages=(ModelMessage(role="system", content="sys"),),
            attempt_index=0,
        )
        with pytest.raises(ProviderResponseError, match="truncated"):
            client.complete(request=req)

    def test_empty_content_error_propagates(self):
        body = json.dumps({
            "choices": [{"message": {"content": "  "}, "finish_reason": "stop"}]
        }).encode()
        transport = FakeTransport([HttpResponse(status_code=200, headers={}, body=body)])
        config = _make_provider_config()
        client = OpenAICompatibleChatModelClient(config=config, transport=transport)

        req = ModelRequest(
            purpose="test",
            messages=(ModelMessage(role="system", content="sys"),),
            attempt_index=0,
        )
        with pytest.raises(ProviderResponseError, match="non-empty"):
            client.complete(request=req)

    def test_api_key_not_in_exception_string(self):
        transport = FakeTransport([
            HttpResponse(status_code=500, headers={}, body=b"{}"),
        ])
        config = _make_provider_config()
        client = OpenAICompatibleChatModelClient(config=config, transport=transport)

        req = ModelRequest(
            purpose="test",
            messages=(ModelMessage(role="system", content="sys"),),
            attempt_index=0,
        )
        try:
            client.complete(request=req)
        except ProviderHTTPError as e:
            msg = str(e)
            assert "test-secret-not-real" not in msg

    def test_prompt_not_in_exception_string(self):
        transport = FakeTransport([
            HttpResponse(status_code=400, headers={}, body=b"{}"),
        ])
        config = _make_provider_config()
        client = OpenAICompatibleChatModelClient(config=config, transport=transport)

        req = ModelRequest(
            purpose="test",
            messages=(ModelMessage(role="system", content="sensitive prompt data"),),
            attempt_index=0,
        )
        try:
            client.complete(request=req)
        except ProviderHTTPError as e:
            msg = str(e)
            assert "sensitive prompt data" not in msg

    def test_no_silent_fallback(self):
        """Verify provider client does not silently fall back to another provider."""
        transport = FakeTransport([
            HttpResponse(status_code=500, headers={}, body=b"{}"),
        ])
        config = _make_provider_config()
        client = OpenAICompatibleChatModelClient(config=config, transport=transport)

        req = ModelRequest(
            purpose="test",
            messages=(ModelMessage(role="system", content="sys"),),
            attempt_index=0,
        )
        with pytest.raises(ProviderHTTPError):
            client.complete(request=req)

    def test_no_default_provider(self):
        """Provider client requires explicit config and transport."""
        # Cannot construct without config
        with pytest.raises(TypeError):
            OpenAICompatibleChatModelClient(transport=None)  # type: ignore
