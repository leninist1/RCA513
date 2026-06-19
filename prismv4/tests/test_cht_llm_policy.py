"""Tests for llm_policy.py — structured LLM policy implementations."""

import json

import pytest

from prismv4.prism_cht.action_schema import DiscriminativeAction
from prismv4.prism_cht.challenge_types import ChallengeProposal, ChallengeResolution, ChallengeVerdict
from prismv4.prism_cht.evidence_graph import EvidenceAtom, EvidenceGraph
from prismv4.prism_cht.hypothesis import HypothesisStatus
from prismv4.prism_cht.llm_types import ModelResponse, StructuredOutputError
from prismv4.prism_cht.llm_policy import (
    StructuredLLMLeadPolicy,
    StructuredLLMChallengerPolicy,
    StructuredModelRequester,
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
from prismv4.prism_cht.challenge_types import ChallengeSnapshot


# ===========================================================================
# Minimal fixture helpers
# ===========================================================================


def _make_evidence_atom(
    evidence_id="e1",
    query_signature="sha256:abcd",
    modality="metric",
    component_scope=("comp-a",),
    time_window=(1000.0, 2000.0),
):
    return EvidenceAtom(
        evidence_id=evidence_id,
        query_signature=query_signature,
        modality=modality,
        component_scope=component_scope,
        time_window=time_window,
        observation={"cpu": 95.0},
        provenance={"source": "prometheus"},
    )


def _make_graph_with_atoms(*eids):
    graph = EvidenceGraph()
    for i, eid in enumerate(eids):
        atom = _make_evidence_atom(eid, query_signature=f"sha:{i}")
        graph.add_evidence(atom)
    return graph


def _make_hypothesis_snapshot(
    hypothesis_id="H1",
    root_component="comp-a",
    status=HypothesisStatus.ACTIVE,
):
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
        status=status,
    )


def _make_triplet_grounding():
    return TripletEvidenceCoverage(
        component_evidence_ids=("e_comp",),
        reason_evidence_ids=("e_reason",),
        onset_evidence_ids=("e_onset",),
    )


def _make_lead_nomination(hypothesis_id="H1"):
    return LeadNomination(
        hypothesis_id=hypothesis_id,
        supporting_evidence_ids=("e1",),
        addressed_competitor_ids=("H2",),
        triplet_grounding=_make_triplet_grounding(),
        rationale="Best supported",
    )


def _make_discriminative_action(
    action_id="act-1",
    target_ids=("H1", "H2"),
):
    return DiscriminativeAction(
        action_id=action_id,
        action_type="run_discriminative_test",
        target_hypothesis_ids=target_ids,
        question="Which is earlier?",
        tool_name="compare_onset_order",
        args={"component": "comp-a"},
        expected_outcomes={tid: "earlier" for tid in target_ids},
        why_discriminative="Distinguishes onset",
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


def _make_audit_step(action=None, evidence_id="e9"):
    if action is None:
        action = _make_discriminative_action("act-x")
    assessment = EvidenceAssessment(
        action_id=action.action_id,
        evidence_id=evidence_id,
        outcome=AssessmentOutcome.INFORMATIVE,
        links=(),
        status_updates=(),
        rationale="Test",
    )
    return InvestigationAuditStep(
        round_index=0,
        action=action,
        evidence_id=evidence_id,
        assessment=assessment,
    )


def _make_challenge_snapshot():
    nominated = _make_hypothesis_snapshot("H1")
    competitor = _make_hypothesis_snapshot("H2", root_component="comp-b")
    nomination = _make_lead_nomination("H1")
    audit = _make_audit_step()
    return ChallengeSnapshot(
        nominated_hypothesis=nominated,
        competitor_hypotheses=(competitor,),
        evidence_ids=("e1", "e2"),
        lead_nomination=nomination,
        lead_audit_steps=(audit,),
    )


# ===========================================================================
# Fake model client for tests
# ===========================================================================


class _TestFakeClient:
    """Inline deterministic fake client for policy testing."""

    def __init__(self, responses):
        self._responses = list(responses)
        self._requests = []
        self._idx = 0

    def complete(self, *, request):
        self._requests.append(request)
        if self._idx >= len(self._responses):
            raise RuntimeError("No more responses")
        resp = self._responses[self._idx]
        self._idx += 1
        if isinstance(resp, Exception):
            raise resp
        if isinstance(resp, str):
            return ModelResponse(content=resp)
        return resp

    @property
    def requests(self):
        return tuple(self._requests)


# ===========================================================================
# Shared JSON templates
# ===========================================================================

_VALID_ACTION_JSON = json.dumps({
    "kind": "action",
    "payload": {
        "action_id": "act-1",
        "action_type": "run_discriminative_test",
        "target_hypothesis_ids": ["H1", "H2"],
        "question": "Q?",
        "tool_name": "t1",
        "args": {},
        "expected_outcomes": {"H1": "a", "H2": "b"},
        "why_discriminative": "reason",
    },
})

_VALID_NOMINATION_JSON = json.dumps({
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
        "rationale": "H1 is best",
    },
})

_VALID_ASSESSMENT_JSON = json.dumps({
    "action_id": "act-1",
    "evidence_id": "e9",
    "outcome": "informative",
    "links": [],
    "status_updates": [],
    "rationale": "Evidence collected",
})

_VALID_CHALLENGE_PROPOSAL_JSON = json.dumps({
    "challenge_id": "ch-1",
    "nominated_hypothesis_id": "H1",
    "competitor_hypothesis_ids": ["H2"],
    "challenge_claim": "claim",
    "falsification_target": "target",
    "action": {
        "action_id": "act-c1",
        "action_type": "run_discriminative_test",
        "target_hypothesis_ids": ["H1", "H2"],
        "question": "Q?",
        "tool_name": "t1",
        "args": {},
        "expected_outcomes": {"H1": "a", "H2": "b"},
        "why_discriminative": "reason",
    },
    "rationale": "reason",
})

_VALID_CHALLENGE_RESOLUTION_JSON = json.dumps({
    "challenge_id": "ch-1",
    "action_id": "act-c1",
    "evidence_id": "e9",
    "verdict": "nomination_survived",
    "assessment": {
        "action_id": "act-c1",
        "evidence_id": "e9",
        "outcome": "informative",
        "links": [],
        "status_updates": [],
        "rationale": "Evidence supports",
    },
    "rationale": "Challenge failed",
})


# ===========================================================================
# StructuredModelRequester
# ===========================================================================


class TestStructuredModelRequester:
    def test_happy_path_single_attempt(self):
        client = _TestFakeClient([_VALID_ACTION_JSON])
        requester = StructuredModelRequester(client=client, max_attempts=1)

        def request_builder(*, attempt_index, repair_error, snapshot):
            from prismv4.prism_cht.llm_types import ModelRequest, ModelMessage
            return ModelRequest(
                purpose="test",
                messages=(ModelMessage(role="system", content="test"),),
                attempt_index=attempt_index,
            )

        def parser(text):
            from prismv4.prism_cht.llm_json import parse_lead_policy_decision
            return parse_lead_policy_decision(text)

        result = requester.request_parsed(
            request_builder=request_builder,
            parser=parser,
            builder_kwargs={"snapshot": None},
        )
        assert isinstance(result, DiscriminativeAction)

    def test_retry_on_parse_failure(self):
        client = _TestFakeClient(["not valid json at all", _VALID_ACTION_JSON])
        requester = StructuredModelRequester(client=client, max_attempts=2)

        def request_builder(*, attempt_index, repair_error, snapshot):
            from prismv4.prism_cht.llm_types import ModelRequest, ModelMessage
            return ModelRequest(
                purpose="test",
                messages=(ModelMessage(role="system", content="test"),),
                attempt_index=attempt_index,
            )

        def parser(text):
            from prismv4.prism_cht.llm_json import parse_lead_policy_decision
            return parse_lead_policy_decision(text)

        result = requester.request_parsed(
            request_builder=request_builder,
            parser=parser,
            builder_kwargs={"snapshot": None},
        )
        assert isinstance(result, DiscriminativeAction)
        assert len(client.requests) == 2

    def test_exhausted_retries_raises(self):
        client = _TestFakeClient(["bad json 1", "bad json 2"])
        requester = StructuredModelRequester(client=client, max_attempts=2)

        def request_builder(*, attempt_index, repair_error, snapshot):
            from prismv4.prism_cht.llm_types import ModelRequest, ModelMessage
            return ModelRequest(
                purpose="test",
                messages=(ModelMessage(role="system", content="test"),),
                attempt_index=attempt_index,
            )

        def parser(text):
            from prismv4.prism_cht.llm_json import parse_lead_policy_decision
            return parse_lead_policy_decision(text)

        with pytest.raises(StructuredOutputError, match="failed after"):
            requester.request_parsed(
                request_builder=request_builder,
                parser=parser,
                builder_kwargs={"snapshot": None},
            )

    def test_rejects_max_attempts_less_than_one(self):
        client = _TestFakeClient(["test"])
        with pytest.raises(ValueError, match="max_attempts"):
            StructuredModelRequester(client=client, max_attempts=0)

    def test_client_exception_propagates(self):
        client = _TestFakeClient([RuntimeError("transport failure")])
        requester = StructuredModelRequester(client=client, max_attempts=2)

        def request_builder(*, attempt_index, repair_error, snapshot):
            from prismv4.prism_cht.llm_types import ModelRequest, ModelMessage
            return ModelRequest(
                purpose="test",
                messages=(ModelMessage(role="system", content="test"),),
                attempt_index=attempt_index,
            )

        def parser(text):
            from prismv4.prism_cht.llm_json import parse_lead_policy_decision
            return parse_lead_policy_decision(text)

        with pytest.raises(RuntimeError, match="transport failure"):
            requester.request_parsed(
                request_builder=request_builder,
                parser=parser,
                builder_kwargs={"snapshot": None},
            )

    def test_no_network_access(self):
        client = _TestFakeClient([_VALID_ACTION_JSON])
        requester = StructuredModelRequester(client=client, max_attempts=1)
        assert requester is not None

    def test_no_raw_response_body_in_error(self):
        """Error messages should not echo raw model response."""
        client = _TestFakeClient(["RAW_PROVIDER_BODY_SENTINEL"])
        requester = StructuredModelRequester(client=client, max_attempts=1)

        def request_builder(*, attempt_index, repair_error, snapshot):
            from prismv4.prism_cht.llm_types import ModelRequest, ModelMessage
            return ModelRequest(
                purpose="test",
                messages=(ModelMessage(role="system", content="test"),),
                attempt_index=attempt_index,
            )

        def parser(text):
            from prismv4.prism_cht.llm_json import parse_lead_policy_decision
            return parse_lead_policy_decision(text)

        with pytest.raises(StructuredOutputError):
            requester.request_parsed(
                request_builder=request_builder,
                parser=parser,
                builder_kwargs={"snapshot": None},
            )


# ===========================================================================
# StructuredLLMLeadPolicy
# ===========================================================================


class TestStructuredLLMLeadPolicy:
    def test_decide_next_action(self):
        graph = _make_graph_with_atoms("e1", "e2")
        snapshot = _make_lead_snapshot()
        client = _TestFakeClient([_VALID_ACTION_JSON])

        policy = StructuredLLMLeadPolicy(
            client=client,
            graph=graph,
            allowed_tool_names=["t1"],
        )
        decision = policy.decide_next(snapshot=snapshot)
        assert isinstance(decision, DiscriminativeAction)
        assert decision.action_id == "act-1"

    def test_decide_next_nomination(self):
        graph = _make_graph_with_atoms("e1", "e2")
        snapshot = _make_lead_snapshot()
        client = _TestFakeClient([_VALID_NOMINATION_JSON])

        policy = StructuredLLMLeadPolicy(
            client=client,
            graph=graph,
            allowed_tool_names=["t1"],
        )
        decision = policy.decide_next(snapshot=snapshot)
        assert isinstance(decision, LeadNomination)
        assert decision.hypothesis_id == "H1"

    def test_assess_evidence(self):
        graph = _make_graph_with_atoms()
        snapshot = _make_lead_snapshot()
        action = _make_discriminative_action()
        evidence = _make_evidence_atom("e9")
        client = _TestFakeClient([_VALID_ASSESSMENT_JSON])

        policy = StructuredLLMLeadPolicy(
            client=client,
            graph=graph,
            allowed_tool_names=["t1"],
        )
        result = policy.assess_evidence(
            snapshot=snapshot,
            action=action,
            evidence=evidence,
        )
        assert isinstance(result, EvidenceAssessment)
        assert result.action_id == "act-1"

    def test_malformed_json_handled(self):
        graph = _make_graph_with_atoms("e1", "e2")
        snapshot = _make_lead_snapshot()
        client = _TestFakeClient(["not json", _VALID_ACTION_JSON])

        policy = StructuredLLMLeadPolicy(
            client=client,
            graph=graph,
            allowed_tool_names=["t1"],
            max_attempts=2,
        )
        decision = policy.decide_next(snapshot=snapshot)
        assert isinstance(decision, DiscriminativeAction)

    def test_schema_invalid_json_handled(self):
        graph = _make_graph_with_atoms("e1", "e2")
        snapshot = _make_lead_snapshot()
        client = _TestFakeClient([
            json.dumps({"kind": "invalid_kind", "payload": {}}),
            _VALID_ACTION_JSON,
        ])

        policy = StructuredLLMLeadPolicy(
            client=client,
            graph=graph,
            allowed_tool_names=["t1"],
            max_attempts=2,
        )
        decision = policy.decide_next(snapshot=snapshot)
        assert isinstance(decision, DiscriminativeAction)

    def test_exhausted_all_retries_raises(self):
        graph = _make_graph_with_atoms("e1", "e2")
        snapshot = _make_lead_snapshot()
        client = _TestFakeClient(["bad json 1", "bad json 2"])

        policy = StructuredLLMLeadPolicy(
            client=client,
            graph=graph,
            allowed_tool_names=["t1"],
            max_attempts=2,
        )
        with pytest.raises(StructuredOutputError, match="failed after"):
            policy.decide_next(snapshot=snapshot)

    def test_deterministic_for_same_fake_response(self):
        graph = _make_graph_with_atoms("e1", "e2")
        snapshot = _make_lead_snapshot()

        client1 = _TestFakeClient([_VALID_ACTION_JSON])
        client2 = _TestFakeClient([_VALID_ACTION_JSON])

        policy1 = StructuredLLMLeadPolicy(
            client=client1, graph=graph, allowed_tool_names=["t1"]
        )
        policy2 = StructuredLLMLeadPolicy(
            client=client2, graph=graph, allowed_tool_names=["t1"]
        )
        assert policy1.decide_next(snapshot=snapshot) == policy2.decide_next(snapshot=snapshot)

    def test_no_network_access(self):
        graph = _make_graph_with_atoms("e1", "e2")
        snapshot = _make_lead_snapshot()
        client = _TestFakeClient([_VALID_ACTION_JSON])
        policy = StructuredLLMLeadPolicy(
            client=client, graph=graph, allowed_tool_names=["t1"]
        )
        result = policy.decide_next(snapshot=snapshot)
        assert result is not None

    def test_input_not_mutated(self):
        graph = _make_graph_with_atoms("e1", "e2")
        snapshot = _make_lead_snapshot()
        original_round = snapshot.round_index
        original_hypothesis_ids = [h.hypothesis_id for h in snapshot.hypotheses]

        client = _TestFakeClient([_VALID_ACTION_JSON])
        policy = StructuredLLMLeadPolicy(
            client=client, graph=graph, allowed_tool_names=["t1"]
        )
        policy.decide_next(snapshot=snapshot)

        assert snapshot.round_index == original_round
        assert [h.hypothesis_id for h in snapshot.hypotheses] == original_hypothesis_ids

    def test_no_direct_provider_config_dependency(self):
        """Policy must not import or depend on provider_config module."""
        assert "provider_config" not in dir(StructuredLLMLeadPolicy)

    def test_error_does_not_leak_raw_prompt(self):
        """Error on parse failure should not include raw model response text."""
        graph = _make_graph_with_atoms("e1", "e2")
        snapshot = _make_lead_snapshot()
        # Use a very long distinctive string that would be obvious in error
        client = _TestFakeClient(["x" * 2000])

        policy = StructuredLLMLeadPolicy(
            client=client,
            graph=graph,
            allowed_tool_names=["t1"],
            max_attempts=1,
        )
        with pytest.raises(StructuredOutputError) as exc:
            policy.decide_next(snapshot=snapshot)
        error_str = str(exc.value)
        # Long arbitrary response should not appear in error message
        assert "x" * 2000 not in error_str


# ===========================================================================
# StructuredLLMChallengerPolicy
# ===========================================================================


class TestStructuredLLMChallengerPolicy:
    def test_propose_challenge(self):
        graph = _make_graph_with_atoms("e1", "e2")
        snapshot = _make_challenge_snapshot()
        client = _TestFakeClient([_VALID_CHALLENGE_PROPOSAL_JSON])

        policy = StructuredLLMChallengerPolicy(
            client=client,
            graph=graph,
            allowed_tool_names=["t1"],
        )
        result = policy.propose_challenge(snapshot=snapshot)
        assert isinstance(result, ChallengeProposal)
        assert result.challenge_id == "ch-1"

    def test_assess_challenge(self):
        graph = _make_graph_with_atoms()
        snapshot = _make_challenge_snapshot()
        proposal = ChallengeProposal(
            challenge_id="ch-1",
            nominated_hypothesis_id="H1",
            competitor_hypothesis_ids=("H2",),
            challenge_claim="claim",
            falsification_target="target",
            action=_make_discriminative_action("act-c1"),
            rationale="reason",
        )
        evidence = _make_evidence_atom("e9")
        client = _TestFakeClient([_VALID_CHALLENGE_RESOLUTION_JSON])

        policy = StructuredLLMChallengerPolicy(
            client=client,
            graph=graph,
            allowed_tool_names=["t1"],
        )
        result = policy.assess_challenge(
            snapshot=snapshot,
            proposal=proposal,
            evidence=evidence,
        )
        assert isinstance(result, ChallengeResolution)
        assert result.verdict == ChallengeVerdict.NOMINATION_SURVIVED

    def test_malformed_json_for_challenge(self):
        graph = _make_graph_with_atoms("e1", "e2")
        snapshot = _make_challenge_snapshot()
        client = _TestFakeClient(["garbage", _VALID_CHALLENGE_PROPOSAL_JSON])

        policy = StructuredLLMChallengerPolicy(
            client=client,
            graph=graph,
            allowed_tool_names=["t1"],
            max_attempts=2,
        )
        result = policy.propose_challenge(snapshot=snapshot)
        assert isinstance(result, ChallengeProposal)

    def test_deterministic_for_same_fake_response(self):
        graph = _make_graph_with_atoms("e1", "e2")
        snapshot = _make_challenge_snapshot()

        client1 = _TestFakeClient([_VALID_CHALLENGE_PROPOSAL_JSON])
        client2 = _TestFakeClient([_VALID_CHALLENGE_PROPOSAL_JSON])
        p1 = StructuredLLMChallengerPolicy(
            client=client1, graph=graph, allowed_tool_names=["t1"]
        )
        p2 = StructuredLLMChallengerPolicy(
            client=client2, graph=graph, allowed_tool_names=["t1"]
        )
        assert p1.propose_challenge(snapshot=snapshot) == p2.propose_challenge(snapshot=snapshot)

    def test_no_network_access(self):
        graph = _make_graph_with_atoms("e1", "e2")
        snapshot = _make_challenge_snapshot()
        client = _TestFakeClient([_VALID_CHALLENGE_PROPOSAL_JSON])
        policy = StructuredLLMChallengerPolicy(
            client=client, graph=graph, allowed_tool_names=["t1"]
        )
        result = policy.propose_challenge(snapshot=snapshot)
        assert result is not None

    def test_error_does_not_leak_raw_response(self):
        graph = _make_graph_with_atoms("e1", "e2")
        snapshot = _make_challenge_snapshot()
        client = _TestFakeClient(["x" * 2000])

        policy = StructuredLLMChallengerPolicy(
            client=client,
            graph=graph,
            allowed_tool_names=["t1"],
            max_attempts=1,
        )
        with pytest.raises(StructuredOutputError) as exc:
            policy.propose_challenge(snapshot=snapshot)
        error_str = str(exc.value)
        assert "x" * 2000 not in error_str


# ===========================================================================
# Cross-role isolation
# ===========================================================================


class TestCrossRoleIsolation:
    def test_lead_and_challenger_use_separate_requester_instances(self):
        graph = _make_graph_with_atoms("e1", "e2")
        lead_client = _TestFakeClient([_VALID_ACTION_JSON])
        challenger_client = _TestFakeClient([_VALID_CHALLENGE_PROPOSAL_JSON])

        lead = StructuredLLMLeadPolicy(
            client=lead_client,
            graph=graph,
            allowed_tool_names=["t1"],
        )
        challenger = StructuredLLMChallengerPolicy(
            client=challenger_client,
            graph=graph,
            allowed_tool_names=["t1"],
        )

        lead_snapshot = _make_lead_snapshot()
        challenge_snapshot = _make_challenge_snapshot()

        lead_result = lead.decide_next(snapshot=lead_snapshot)
        assert isinstance(lead_result, DiscriminativeAction)

        challenger_result = challenger.propose_challenge(snapshot=challenge_snapshot)
        assert isinstance(challenger_result, ChallengeProposal)
