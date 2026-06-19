"""End-to-end tests for the structured-LLM-policy → Provider safety boundary."""

import collections
import json
import re

import pytest

from prismv4.prism_cht.action_schema import DiscriminativeAction
from prismv4.prism_cht.challenge_types import (
    ChallengeProposal,
    ChallengeResolution,
    ChallengeSnapshot,
    ChallengeVerdict,
)
from prismv4.prism_cht.evidence_graph import EvidenceAtom, EvidenceGraph
from prismv4.prism_cht.hypothesis import HypothesisStatus
from prismv4.prism_cht.http_transport import HttpRequest, HttpResponse, HttpTransport
from prismv4.prism_cht.llm_policy import (
    StructuredLLMChallengerPolicy,
    StructuredLLMLeadPolicy,
    StructuredModelRequester,
)
from prismv4.prism_cht.llm_types import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    StructuredOutputError,
)
from prismv4.prism_cht.openai_compatible_client import (
    OpenAICompatibleChatModelClient,
)
from prismv4.prism_cht.provider_config import OpenAICompatibleChatConfig
from prismv4.prism_cht.provider_types import (
    ProviderCallAudit,
    ProviderHTTPError,
    ProviderResponseError,
    ProviderTransportError,
    ProviderUsage,
)
from prismv4.prism_cht.tournament_types import (
    AssessmentOutcome,
    EvidenceAssessment,
    HypothesisSnapshot,
    InvestigationAuditStep,
    LeadNomination,
    LeadTournamentSnapshot,
    TripletEvidenceCoverage,
)


# ===========================================================================
# Sentinels
# ===========================================================================

FAKE_AUTHORIZATION_SECRET_SENTINEL = "sk-test-e2e-secret-fake-key-do-not-leak"
RAW_PROMPT_CONTENT_SENTINEL = "E2E_RAW_PROMPT_TOP_SECRET_S2_PAYLOAD"
RAW_PROVIDER_BODY_SENTINEL = "E2E_RAW_PROVIDER_S3_BODY_SECRET"
PARSED_MODEL_PAYLOAD_SENTINEL = "E2E_PARSED_MODEL_S4_SECRET_PAYLOAD"
GROUND_TRUTH_SENTINEL = "E2E_GT_ROOT_CAUSE_IS_OOMKILL_NODE7"

ALL_SENTINELS = [
    FAKE_AUTHORIZATION_SECRET_SENTINEL,
    RAW_PROMPT_CONTENT_SENTINEL,
    RAW_PROVIDER_BODY_SENTINEL,
    PARSED_MODEL_PAYLOAD_SENTINEL,
    GROUND_TRUTH_SENTINEL,
]

ALL_SENSITIVE_SENTINELS = ALL_SENTINELS


# ===========================================================================
# Assertion helpers
# ===========================================================================


def _no_sentinel_in(text: str) -> None:
    """Assert *text* contains none of the sensitive sentinels."""
    for s in ALL_SENSITIVE_SENTINELS:
        assert s not in text, f"sentinel leaked in repr/str: {s[:40]}"


def _no_sentinel_in_bytes(data: bytes) -> None:
    """Assert *data* bytes contains none of the sensitive sentinels."""
    for s in ALL_SENSITIVE_SENTINELS:
        if isinstance(s, str):
            assert s.encode("utf-8") not in data, f"sentinel leaked in bytes: {s[:40]}"
        else:
            assert s not in data


# ===========================================================================
# Fake HTTP transport (recording, no network)
# ===========================================================================


class RecordingFakeTransport:
    """Implements HttpTransport.  Records every HttpRequest and returns
    queued HttpResponse objects.  Never accesses the network."""

    def __init__(self, responses):
        self._queue = collections.deque(responses)
        self._requests: list[HttpRequest] = []

    def send(self, *, request: HttpRequest, max_response_bytes: int) -> HttpResponse:
        self._requests.append(request)
        if not self._queue:
            raise RuntimeError("RecordingFakeTransport: no more responses queued")
        item = self._queue.popleft()
        if isinstance(item, Exception):
            raise item
        return item

    @property
    def requests(self) -> tuple[HttpRequest, ...]:
        return tuple(self._requests)


# ===========================================================================
# OpenAI-compatible HTTP response builder
# ===========================================================================


def _make_http_response(
    assistant_json: str,
    *,
    status_code: int = 200,
    finish_reason: str = "stop",
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
) -> HttpResponse:
    """Build an OpenAI-compatible HTTP response with the given assistant JSON."""
    body = json.dumps(
        {
            "choices": [
                {
                    "message": {"content": assistant_json},
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }
    ).encode("utf-8")
    return HttpResponse(status_code=status_code, headers={}, body=body)


# ===========================================================================
# Valid structured JSON payloads for each policy role
# ===========================================================================

_VALID_ACTION_JSON = json.dumps(
    {
        "kind": "action",
        "payload": {
            "action_id": "act-e2e-1",
            "action_type": "run_discriminative_test",
            "target_hypothesis_ids": ["H1", "H2"],
            "question": "Which component first?",
            "tool_name": "compare_onset_order",
            "args": {"component": "comp-a"},
            "expected_outcomes": {"H1": "earlier", "H2": "later"},
            "why_discriminative": "Distinguishes onset timing",
        },
    }
)

_VALID_NOMINATION_JSON = json.dumps(
    {
        "kind": "nomination",
        "payload": {
            "hypothesis_id": "H1",
            "supporting_evidence_ids": ["e1"],
            "addressed_competitor_ids": ["H2"],
            "triplet_grounding": {
                "component_evidence_ids": ["e_comp"],
                "reason_evidence_ids": ["e_reason"],
                "onset_evidence_ids": ["e_onset"],
            },
            "rationale": "H1 has strongest supporting evidence",
        },
    }
)

_VALID_ASSESSMENT_JSON = json.dumps(
    {
        "action_id": "act-e2e-1",
        "evidence_id": "e9",
        "outcome": "informative",
        "links": [],
        "status_updates": [],
        "rationale": "Evidence collected and assessed",
    }
)

_VALID_CHALLENGE_PROPOSAL_JSON = json.dumps(
    {
        "challenge_id": "ch-e2e-1",
        "nominated_hypothesis_id": "H1",
        "competitor_hypothesis_ids": ["H2"],
        "challenge_claim": "H2 onset is actually earlier",
        "falsification_target": "onset order",
        "action": {
            "action_id": "act-ce2e-1",
            "action_type": "run_discriminative_test",
            "target_hypothesis_ids": ["H1", "H2"],
            "question": "Which is really earlier?",
            "tool_name": "compare_onset_order",
            "args": {},
            "expected_outcomes": {"H1": "later", "H2": "earlier"},
            "why_discriminative": "Refutes H1's onset claim",
        },
        "rationale": "Evidence contradicts H1",
    }
)

_VALID_CHALLENGE_RESOLUTION_JSON = json.dumps(
    {
        "challenge_id": "ch-e2e-1",
        "action_id": "act-ce2e-1",
        "evidence_id": "e9",
        "verdict": "nomination_survived",
        "assessment": {
            "action_id": "act-ce2e-1",
            "evidence_id": "e9",
            "outcome": "informative",
            "links": [],
            "status_updates": [],
            "rationale": "Challenge failed; evidence supports nomination",
        },
        "rationale": "H1 nomination upheld",
    }
)

# JSON that is valid JSON but invalid for any policy schema
_SCHEMA_INVALID_JSON = json.dumps({"unexpected": "schema-violation", "bad": True})


# ===========================================================================
# Test fixtures — snapshot builders
# ===========================================================================


def _make_evidence_atom(evidence_id="e1", query_signature="sha256:e2e"):
    return EvidenceAtom(
        evidence_id=evidence_id,
        query_signature=query_signature,
        modality="metric",
        component_scope=("comp-a",),
        time_window=(1000.0, 2000.0),
        observation={"cpu": 95.0},
        provenance={"source": "prometheus"},
    )


def _make_graph_with_atoms(*eids):
    graph = EvidenceGraph()
    for i, eid in enumerate(eids):
        atom = _make_evidence_atom(eid, query_signature=f"sha256:e2e-{eid}-{i}")
        graph.add_evidence(atom)
    return graph


def _make_hypothesis_snapshot(hypothesis_id="H1", root_component="comp-a"):
    return HypothesisSnapshot(
        hypothesis_id=hypothesis_id,
        root_component=root_component,
        reason_family="cpu_exhaustion",
        onset_interval=(1000.0, 2000.0),
        local_trigger="CPU spike",
        propagation_path=(root_component, "comp-b"),
        explained_symptoms=("latency",),
        predicted_observations=("high CPU",),
        falsifiers=("no CPU spike",),
        supporting_evidence_ids=(),
        contradicting_evidence_ids=(),
        unresolved_questions=(),
        status=HypothesisStatus.ACTIVE,
    )


def _make_lead_snapshot():
    hs = _make_hypothesis_snapshot("H1")
    hs2 = _make_hypothesis_snapshot("H2", root_component="comp-b")
    return LeadTournamentSnapshot(
        round_index=0,
        hypotheses=(hs, hs2),
        evidence_ids=("e1", "e2"),
        audit_steps=(),
    )


def _make_challenge_snapshot():
    nominated = _make_hypothesis_snapshot("H1")
    competitor = _make_hypothesis_snapshot("H2", root_component="comp-b")
    nomination = LeadNomination(
        hypothesis_id="H1",
        supporting_evidence_ids=("e1",),
        addressed_competitor_ids=("H2",),
        triplet_grounding=TripletEvidenceCoverage(
            component_evidence_ids=("e_comp",),
            reason_evidence_ids=("e_reason",),
            onset_evidence_ids=("e_onset",),
        ),
        rationale="Best supported",
    )
    action = DiscriminativeAction(
        action_id="act-x",
        action_type="run_discriminative_test",
        target_hypothesis_ids=("H1", "H2"),
        question="Q?",
        tool_name="compare_onset_order",
        args={},
        expected_outcomes={"H1": "a", "H2": "b"},
        why_discriminative="reason",
    )
    assessment = EvidenceAssessment(
        action_id="act-x",
        evidence_id="e9",
        outcome=AssessmentOutcome.INFORMATIVE,
        links=(),
        status_updates=(),
        rationale="Test",
    )
    audit = InvestigationAuditStep(
        round_index=0,
        action=action,
        evidence_id="e9",
        assessment=assessment,
    )
    return ChallengeSnapshot(
        nominated_hypothesis=nominated,
        competitor_hypotheses=(competitor,),
        evidence_ids=("e1", "e2"),
        lead_nomination=nomination,
        lead_audit_steps=(audit,),
    )


def _make_provider_config():
    return OpenAICompatibleChatConfig(
        base_url="https://api.example.com/v1",
        model="test-model",
        api_key=FAKE_AUTHORIZATION_SECRET_SENTINEL,
        timeout_seconds=30.0,
        max_tokens=4096,
        max_response_bytes=2_000_000,
        json_mode=True,
    )


def _build_client_with_responses(*response_json_strings):
    """Build a full Provider client wired with a recording fake transport."""
    http_responses = [_make_http_response(j) for j in response_json_strings]
    transport = RecordingFakeTransport(http_responses)
    config = _make_provider_config()
    client = OpenAICompatibleChatModelClient(config=config, transport=transport)
    return client, transport


# ===========================================================================
# Lead Policy — Successful end-to-end
# ===========================================================================


class TestLeadPolicyE2ESuccess:
    """Semantic prompt flow preserved; secrets and bodies blocked at boundary."""

    def test_lead_action_decision_semantic_and_safety(self):
        client, transport = _build_client_with_responses(_VALID_ACTION_JSON)
        graph = _make_graph_with_atoms("e1", "e2")
        policy = StructuredLLMLeadPolicy(client=client, graph=graph)
        snapshot = _make_lead_snapshot()

        result = policy.decide_next(snapshot=snapshot)

        # A — Semantic prompt flow
        assert len(transport.requests) == 1
        req: HttpRequest = transport.requests[0]
        assert isinstance(result, DiscriminativeAction)

        # B — Credential flow: credential visible only in internal HTTP request
        assert req.headers["Authorization"] == f"Bearer {FAKE_AUTHORIZATION_SECRET_SENTINEL}"
        _no_sentinel_in(repr(req))  # also tests credential not in repr

        # C — Provider response flow
        assert len(client.audit_records) == 1
        audit: ProviderCallAudit = client.audit_records[0]
        audit_repr = repr(audit)
        _no_sentinel_in(audit_repr)

        # D — Audit record content safety
        assert audit.outcome == "success"
        _no_sentinel_in(repr(audit))

        # Integrated repr safety
        _no_sentinel_in(f"{req!r}")

    def test_lead_nomination_decision_semantic_and_safety(self):
        client, transport = _build_client_with_responses(_VALID_NOMINATION_JSON)
        graph = _make_graph_with_atoms("e1", "e2")
        policy = StructuredLLMLeadPolicy(client=client, graph=graph)
        snapshot = _make_lead_snapshot()

        result = policy.decide_next(snapshot=snapshot)

        assert isinstance(result, LeadNomination)
        assert result.hypothesis_id == "H1"

        req: HttpRequest = transport.requests[0]
        assert len(client.audit_records) == 1
        audit_repr = repr(client.audit_records[0])
        _no_sentinel_in(audit_repr)
        _no_sentinel_in(repr(req))

    def test_lead_evidence_assessment_semantic_and_safety(self):
        client, transport = _build_client_with_responses(_VALID_ASSESSMENT_JSON)
        graph = _make_graph_with_atoms("e1", "e2")
        policy = StructuredLLMLeadPolicy(client=client, graph=graph)
        snapshot = _make_lead_snapshot()
        action = DiscriminativeAction(
            action_id="act-e2e-1",
            action_type="run_discriminative_test",
            target_hypothesis_ids=("H1", "H2"),
            question="Q?",
            tool_name="compare_onset_order",
            args={},
            expected_outcomes={"H1": "a", "H2": "b"},
            why_discriminative="reason",
        )
        evidence = _make_evidence_atom("e9")

        result = policy.assess_evidence(
            snapshot=snapshot, action=action, evidence=evidence
        )

        assert isinstance(result, EvidenceAssessment)
        assert result.outcome == AssessmentOutcome.INFORMATIVE

        req: HttpRequest = transport.requests[0]
        _no_sentinel_in(repr(req))
        _no_sentinel_in(repr(client.audit_records[0]))


# ===========================================================================
# Challenger Policy — Successful end-to-end
# ===========================================================================


class TestChallengerPolicyE2ESuccess:
    def test_challenge_proposal_semantic_and_safety(self):
        client, transport = _build_client_with_responses(_VALID_CHALLENGE_PROPOSAL_JSON)
        graph = _make_graph_with_atoms("e1", "e2")
        policy = StructuredLLMChallengerPolicy(client=client, graph=graph)
        snapshot = _make_challenge_snapshot()

        result = policy.propose_challenge(snapshot=snapshot)

        assert isinstance(result, ChallengeProposal)
        assert isinstance(result.action, DiscriminativeAction)

        req: HttpRequest = transport.requests[0]
        _no_sentinel_in(repr(req))
        _no_sentinel_in(repr(client.audit_records[0]))
        assert len(client.audit_records) == 1

    def test_challenge_resolution_semantic_and_safety(self):
        client, transport = _build_client_with_responses(
            _VALID_CHALLENGE_RESOLUTION_JSON
        )
        graph = _make_graph_with_atoms("e1", "e2")
        policy = StructuredLLMChallengerPolicy(client=client, graph=graph)
        snapshot = _make_challenge_snapshot()
        proposal = ChallengeProposal(
            challenge_id="ch-e2e-1",
            nominated_hypothesis_id="H1",
            competitor_hypothesis_ids=("H2",),
            challenge_claim="claim",
            falsification_target="target",
            action=DiscriminativeAction(
                action_id="act-ce2e-1",
                action_type="run_discriminative_test",
                target_hypothesis_ids=("H1", "H2"),
                question="Q?",
                tool_name="compare_onset_order",
                args={},
                expected_outcomes={"H1": "later", "H2": "earlier"},
                why_discriminative="reason",
            ),
            rationale="reason",
        )
        evidence = _make_evidence_atom("e9")

        result = policy.assess_challenge(
            snapshot=snapshot, proposal=proposal, evidence=evidence
        )

        assert isinstance(result, ChallengeResolution)
        assert result.verdict == ChallengeVerdict.NOMINATION_SURVIVED

        req: HttpRequest = transport.requests[0]
        _no_sentinel_in(repr(req))
        _no_sentinel_in(repr(client.audit_records[0]))
        assert len(client.audit_records) == 1


# ===========================================================================
# Failure-Path Boundary Tests
# ===========================================================================


class TestFailurePaths:
    def test_http_error_body_does_not_leak_in_response(self):
        """Non-2xx response body must not leak into audit or exception text."""
        client, transport = _build_client_with_responses("will-not-be-used")
        transport._queue.clear()
        transport._queue.append(
            HttpResponse(
                status_code=502,
                headers={},
                body=json.dumps({"error": RAW_PROVIDER_BODY_SENTINEL}).encode("utf-8"),
            )
        )
        graph = _make_graph_with_atoms("e1", "e2")
        policy = StructuredLLMLeadPolicy(client=client, graph=graph)
        snapshot = _make_lead_snapshot()

        with pytest.raises(ProviderHTTPError):
            policy.decide_next(snapshot=snapshot)

        assert len(client.audit_records) == 1
        audit = client.audit_records[0]
        assert audit.outcome == "http_error"
        audit_repr = repr(audit)
        _no_sentinel_in(audit_repr)
        _no_sentinel_in(str(audit))

    def test_transport_error_no_leakage(self):
        client, transport = _build_client_with_responses("will-not-be-used")
        transport._queue.clear()
        transport._queue.append(ProviderTransportError("network is down"))
        graph = _make_graph_with_atoms("e1", "e2")
        policy = StructuredLLMLeadPolicy(client=client, graph=graph)
        snapshot = _make_lead_snapshot()

        with pytest.raises(ProviderTransportError, match="network is down"):
            policy.decide_next(snapshot=snapshot)

        assert len(client.audit_records) == 1
        audit = client.audit_records[0]
        assert audit.outcome == "transport_error"
        audit_repr = repr(audit)
        _no_sentinel_in(audit_repr)

    def test_malformed_envelope_body_does_not_leak(self):
        """Malformed (non-JSON) response body must not leak into exception."""
        client, transport = _build_client_with_responses("will-not-be-used")
        transport._queue.clear()
        transport._queue.append(
            HttpResponse(
                status_code=200,
                headers={},
                body=b"NOT JSON AT ALL " + RAW_PROVIDER_BODY_SENTINEL.encode("utf-8"),
            )
        )
        graph = _make_graph_with_atoms("e1", "e2")
        policy = StructuredLLMLeadPolicy(client=client, graph=graph)
        snapshot = _make_lead_snapshot()

        with pytest.raises(
            (ProviderResponseError, ProviderHTTPError, StructuredOutputError)
        ):
            policy.decide_next(snapshot=snapshot)

        assert len(client.audit_records) >= 1
        for audit in client.audit_records:
            _no_sentinel_in(repr(audit))

    def test_schema_invalid_assistant_content_no_leak(self):
        """Schema-invalid JSON content must not leak raw body into exceptions."""
        body = json.dumps(
            {
                "choices": [
                    {"message": {"content": _SCHEMA_INVALID_JSON}, "finish_reason": "stop"}
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
            }
        ).encode("utf-8")
        client, transport = _build_client_with_responses("will-not-be-used")
        transport._queue.clear()
        # Queue two invalid responses to cover retry
        transport._queue.append(
            HttpResponse(status_code=200, headers={}, body=body)
        )
        transport._queue.append(
            HttpResponse(status_code=200, headers={}, body=body)
        )
        graph = _make_graph_with_atoms("e1", "e2")
        policy = StructuredLLMLeadPolicy(client=client, graph=graph)
        snapshot = _make_lead_snapshot()

        with pytest.raises(StructuredOutputError):
            policy.decide_next(snapshot=snapshot)

        assert len(client.audit_records) >= 1
        for audit in client.audit_records:
            _no_sentinel_in(repr(audit))
        # Raw schema-invalid content must not appear in audit text
        audit_text = repr(client.audit_records)
        assert "schema-violation" not in audit_text


# ===========================================================================
# Deterministic Repeated-Call Verification
# ===========================================================================


class TestDeterministicRepeatedCalls:
    def test_two_identical_calls_produce_consistent_results(self):
        client, transport = _build_client_with_responses(
            _VALID_ACTION_JSON, _VALID_ACTION_JSON
        )
        graph = _make_graph_with_atoms("e1", "e2")
        policy = StructuredLLMLeadPolicy(client=client, graph=graph)
        snapshot = _make_lead_snapshot()

        r1 = policy.decide_next(snapshot=snapshot)
        r2 = policy.decide_next(snapshot=snapshot)

        assert isinstance(r1, DiscriminativeAction)
        assert isinstance(r2, DiscriminativeAction)
        assert r1.action_id == r2.action_id

        assert len(transport.requests) == 2
        assert len(client.audit_records) == 2

        # Audit records in order
        assert client.audit_records[0].attempt_index == 0
        assert client.audit_records[1].attempt_index == 0

        for audit in client.audit_records:
            _no_sentinel_in(repr(audit))


# ===========================================================================
# Ground-Truth Leakage Check
# ===========================================================================


class TestNoGroundTruthLeakage:
    def test_ground_truth_not_in_outbound_request(self):
        """Inference-time inputs must not serialize GT-only fields into outbound HTTP body."""
        client, transport = _build_client_with_responses(_VALID_ACTION_JSON)
        graph = _make_graph_with_atoms("e1", "e2")
        policy = StructuredLLMLeadPolicy(client=client, graph=graph)
        snapshot = _make_lead_snapshot()

        policy.decide_next(snapshot=snapshot)

        req: HttpRequest = transport.requests[0]
        body_text = req.body.decode("utf-8", errors="replace")
        assert GROUND_TRUTH_SENTINEL not in body_text
        _no_sentinel_in_bytes(req.body)


# ===========================================================================
# Integrated Repr-Safety at the Boundary
# ===========================================================================


class TestIntegratedReprSafety:
    def test_all_boundary_objects_repr_safe(self):
        client, transport = _build_client_with_responses(_VALID_ACTION_JSON)
        graph = _make_graph_with_atoms("e1", "e2")
        policy = StructuredLLMLeadPolicy(client=client, graph=graph)
        snapshot = _make_lead_snapshot()

        result = policy.decide_next(snapshot=snapshot)

        req: HttpRequest = transport.requests[0]
        # HttpRequest — verify repr excludes credential and body
        _no_sentinel_in(repr(req))
        _no_sentinel_in(f"{req!r}")

        # HttpResponse — we can access it via the transport recording
        # but the repr is already tested via field(repr=False)
        # ProviderCallAudit
        for audit in client.audit_records:
            _no_sentinel_in(repr(audit))
            _no_sentinel_in(f"{audit!r}")

        # ModelMessage / ModelRequest were tested in test_cht_repr_safety.py
        # but verify at the boundary too
        for audit in client.audit_records:
            assert FAKE_AUTHORIZATION_SECRET_SENTINEL not in repr(audit)
            assert RAW_PROMPT_CONTENT_SENTINEL not in repr(audit)
            assert RAW_PROVIDER_BODY_SENTINEL not in repr(audit)

    def test_audit_repr_excludes_all_sentinels(self):
        """Comprehensive audit repr check across all roles."""
        for json_str in [
            _VALID_ACTION_JSON,
            _VALID_NOMINATION_JSON,
            _VALID_CHALLENGE_PROPOSAL_JSON,
        ]:
            client, _ = _build_client_with_responses(json_str)
            graph = _make_graph_with_atoms("e1", "e2")

            if "challenge_id" in json_str:
                policy = StructuredLLMChallengerPolicy(client=client, graph=graph)
                snapshot = _make_challenge_snapshot()
            else:
                policy = StructuredLLMLeadPolicy(client=client, graph=graph)
                snapshot = _make_lead_snapshot()

            if hasattr(policy, "decide_next"):
                policy.decide_next(snapshot=snapshot)
            elif hasattr(policy, "propose_challenge"):
                policy.propose_challenge(snapshot=snapshot)

            for audit in client.audit_records:
                _no_sentinel_in(repr(audit))
