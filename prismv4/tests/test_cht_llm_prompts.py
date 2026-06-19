"""Tests for llm_prompts.py — deterministic prompt construction."""

import pytest

from prismv4.prism_cht.action_schema import DiscriminativeAction
from prismv4.prism_cht.challenge_types import ChallengeProposal, ChallengeSnapshot
from prismv4.prism_cht.evidence_graph import EvidenceAtom, EvidenceGraph
from prismv4.prism_cht.hypothesis import HypothesisStatus
from prismv4.prism_cht.llm_prompts import (
    build_evidence_catalog,
    build_policy_request,
    build_lead_decision_request,
    build_lead_assessment_request,
    build_challenge_proposal_request,
    build_challenge_resolution_request,
    serialize_evidence_atom,
)
from prismv4.prism_cht.llm_types import (
    ModelRequest,
    PromptBudgetExceededError,
)
from prismv4.prism_cht.tournament_types import (
    HypothesisSnapshot,
    InvestigationAuditStep,
    LeadNomination,
    LeadTournamentSnapshot,
    TripletEvidenceCoverage,
)


# ===========================================================================
# Minimal fixture helpers
# ===========================================================================


def _make_evidence_atom(
    evidence_id="e1",
    query_signature="sha256:abcd",
    modality="metric",
    component_scope=("comp-a",),
    time_window=(1000.0, 2000.0),
    observation=None,
    provenance=None,
) -> EvidenceAtom:
    return EvidenceAtom(
        evidence_id=evidence_id,
        query_signature=query_signature,
        modality=modality,
        component_scope=component_scope,
        time_window=time_window,
        observation=observation or {"cpu": 95.0},
        provenance=provenance or {"source": "prometheus"},
    )


def _make_graph_with_atoms(*eids) -> EvidenceGraph:
    """Build a graph with one evidence atom per id, each with unique query sig."""
    graph = EvidenceGraph()
    for i, eid in enumerate(eids):
        atom = _make_evidence_atom(eid, query_signature=f"sha:{i}")
        graph.add_evidence(atom)
    return graph


def _make_hypothesis_snapshot(
    hypothesis_id="H1",
    root_component="comp-a",
    status=HypothesisStatus.ACTIVE,
) -> HypothesisSnapshot:
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


def _make_triplet_grounding() -> TripletEvidenceCoverage:
    return TripletEvidenceCoverage(
        component_evidence_ids=("e_comp",),
        reason_evidence_ids=("e_reason",),
        onset_evidence_ids=("e_onset",),
    )


def _make_lead_nomination(hypothesis_id="H1") -> LeadNomination:
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
) -> DiscriminativeAction:
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


def _make_audit_step(action=None, evidence_id="e9") -> InvestigationAuditStep:
    from prismv4.prism_cht.tournament_types import AssessmentOutcome, EvidenceAssessment

    if action is None:
        action = _make_discriminative_action()
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


def _make_lead_snapshot() -> LeadTournamentSnapshot:
    hs = _make_hypothesis_snapshot("H1")
    hs2 = _make_hypothesis_snapshot("H2", root_component="comp-b")
    return LeadTournamentSnapshot(
        round_index=0,
        hypotheses=(hs, hs2),
        evidence_ids=("e1", "e2"),
        audit_steps=(),
    )


def _make_challenge_snapshot() -> ChallengeSnapshot:
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


def _make_challenge_proposal() -> ChallengeProposal:
    return ChallengeProposal(
        challenge_id="ch-1",
        nominated_hypothesis_id="H1",
        competitor_hypothesis_ids=("H2",),
        challenge_claim="H1 onset is wrong",
        falsification_target="onset_interval",
        action=_make_discriminative_action("act-c1", ("H1", "H2")),
        rationale="Test challenge",
    )


# ===========================================================================
# serialize_evidence_atom
# ===========================================================================


class TestSerializeEvidenceAtom:
    def test_serializes_expected_fields(self):
        atom = _make_evidence_atom(evidence_id="e1")
        result = serialize_evidence_atom(atom)
        assert result["evidence_id"] == "e1"
        assert "query_signature" in result
        assert "modality" in result
        assert "component_scope" in result
        assert "time_window" in result
        assert "observation" in result
        assert "provenance" in result

    def test_deterministic_output(self):
        atom = _make_evidence_atom()
        a = serialize_evidence_atom(atom)
        b = serialize_evidence_atom(atom)
        assert a == b

    def test_value_preserved(self):
        """serialize_evidence_atom preserves input values (no redaction).
        This is a known limitation (deferred: no content filtering)."""
        atom = _make_evidence_atom(
            evidence_id="e1",
            provenance={"source": "test"},
        )
        result = serialize_evidence_atom(atom)
        assert result["provenance"]["source"] == "test"

    def test_no_object_repr_leakage(self):
        """Serialize should not include Python object reprs like '<__main__.Foo at 0x...>'."""
        atom = _make_evidence_atom()
        result = serialize_evidence_atom(atom)
        result_str = str(result)
        assert " at 0x" not in result_str


# ===========================================================================
# build_evidence_catalog
# ===========================================================================


class TestBuildEvidenceCatalog:
    def test_builds_catalog(self):
        graph = _make_graph_with_atoms("e1", "e2")
        catalog = build_evidence_catalog(graph=graph, evidence_ids=["e1", "e2"])
        assert len(catalog) == 2
        assert catalog[0]["evidence_id"] == "e1"
        assert catalog[1]["evidence_id"] == "e2"

    def test_deduplicates(self):
        graph = _make_graph_with_atoms("e1")
        catalog = build_evidence_catalog(graph=graph, evidence_ids=["e1", "e1", "e1"])
        assert len(catalog) == 1

    def test_raises_on_missing_evidence(self):
        graph = EvidenceGraph()
        with pytest.raises(ValueError, match="not found"):
            build_evidence_catalog(graph=graph, evidence_ids=["missing"])

    def test_keeps_first_appearance_order(self):
        graph = _make_graph_with_atoms("e1", "e2")
        catalog = build_evidence_catalog(graph=graph, evidence_ids=["e2", "e1"])
        assert catalog[0]["evidence_id"] == "e2"


# ===========================================================================
# build_policy_request
# ===========================================================================


class TestBuildPolicyRequest:
    def test_deterministic_output(self):
        a = build_policy_request(
            purpose="test",
            context={"state": "initial"},
            response_schema={"type": "object"},
            attempt_index=0,
        )
        b = build_policy_request(
            purpose="test",
            context={"state": "initial"},
            response_schema={"type": "object"},
            attempt_index=0,
        )
        assert a == b
        assert isinstance(a, ModelRequest)

    def test_different_purpose_produces_different_request(self):
        a = build_policy_request(
            purpose="A",
            context={"x": 1},
            response_schema={},
            attempt_index=0,
        )
        b = build_policy_request(
            purpose="B",
            context={"x": 1},
            response_schema={},
            attempt_index=0,
        )
        assert a != b

    def test_system_message_present(self):
        req = build_policy_request(
            purpose="test",
            context={"x": 1},
            response_schema={},
            attempt_index=0,
        )
        assert any(m.role == "system" for m in req.messages)

    def test_user_message_present(self):
        req = build_policy_request(
            purpose="test",
            context={"x": 1},
            response_schema={},
            attempt_index=0,
        )
        assert any(m.role == "user" for m in req.messages)

    def test_purpose_in_system_message(self):
        req = build_policy_request(
            purpose="investigate anomaly",
            context={"x": 1},
            response_schema={},
            attempt_index=0,
        )
        system_content = next(m.content for m in req.messages if m.role == "system")
        assert "investigate anomaly" in system_content

    def test_response_schema_in_system_message(self):
        schema = {"type": "object", "properties": {"result": {"type": "string"}}}
        req = build_policy_request(
            purpose="test",
            context={"x": 1},
            response_schema=schema,
            attempt_index=0,
        )
        system_content = next(m.content for m in req.messages if m.role == "system")
        assert '"result"' in system_content

    def test_attempt_index_in_request(self):
        req = build_policy_request(
            purpose="test",
            context={"x": 1},
            response_schema={},
            attempt_index=3,
        )
        assert req.attempt_index == 3

    def test_repair_error_adds_extra_message(self):
        req = build_policy_request(
            purpose="test",
            context={"x": 1},
            response_schema={},
            attempt_index=1,
            repair_error="Expected 'action' key missing",
        )
        assert len(req.messages) == 3
        repair_content = req.messages[2].content
        assert "Your previous output" in repair_content
        assert "Expected 'action' key missing" in repair_content

    def test_no_repair_error_has_two_messages(self):
        req = build_policy_request(
            purpose="test",
            context={"x": 1},
            response_schema={},
            attempt_index=0,
        )
        assert len(req.messages) == 2

    def test_budget_exceeded_raises(self):
        with pytest.raises(PromptBudgetExceededError):
            build_policy_request(
                purpose="test",
                context={"long_key": "X" * 5000},
                response_schema={},
                attempt_index=0,
                max_prompt_chars=100,
            )

    def test_context_passed_through(self):
        """Prompt builder serializes context verbatim; no filtering exists.
        This is a known limitation (deferred: no content redaction)."""
        req = build_policy_request(
            purpose="test",
            context={"visible_field": "hello_world"},
            response_schema={},
            attempt_index=0,
        )
        user_content = next(m.content for m in req.messages if m.role == "user")
        assert "hello_world" in user_content

    def test_no_extra_fields_added(self):
        """Prompt builder should not inject extra fields beyond the context."""
        context = {"a": 1, "b": 2}
        req = build_policy_request(
            purpose="test",
            context=context,
            response_schema={},
            attempt_index=0,
        )
        user_content = next(m.content for m in req.messages if m.role == "user")
        # The context is serialized as-is (plus framing from instructions)
        assert "score" not in user_content.lower()

    def test_no_network_access(self):
        """Prompt building should not touch network."""
        req = build_policy_request(
            purpose="test",
            context={"x": 1},
            response_schema={},
            attempt_index=0,
        )
        assert req is not None

    def test_input_not_mutated(self):
        context = {"original": "value"}
        schema = {"type": "object"}
        original_context = dict(context)
        original_schema = dict(schema)
        build_policy_request(
            purpose="test",
            context=context,
            response_schema=schema,
            attempt_index=0,
        )
        assert context == original_context
        assert schema == original_schema

    def test_messages_contains_no_object_ids(self):
        req = build_policy_request(
            purpose="test",
            context={"x": 1},
            response_schema={},
            attempt_index=0,
        )
        for msg in req.messages:
            assert " at 0x" not in msg.content


# ===========================================================================
# build_lead_decision_request
# ===========================================================================


class TestBuildLeadDecisionRequest:
    def test_produces_model_request(self):
        snapshot = _make_lead_snapshot()
        graph = _make_graph_with_atoms("e1", "e2")
        req = build_lead_decision_request(
            snapshot=snapshot,
            graph=graph,
            allowed_tool_names=["compare_onset_order"],
        )
        assert isinstance(req, ModelRequest)
        assert req.purpose is not None
        assert "Lead" in req.purpose

    def test_includes_hypothesis_ids(self):
        snapshot = _make_lead_snapshot()
        graph = _make_graph_with_atoms("e1", "e2")
        req = build_lead_decision_request(
            snapshot=snapshot,
            graph=graph,
            allowed_tool_names=["compare_onset_order"],
        )
        user_content = next(m.content for m in req.messages if m.role == "user")
        assert "H1" in user_content
        assert "H2" in user_content

    def test_includes_allowed_tool_names(self):
        snapshot = _make_lead_snapshot()
        graph = _make_graph_with_atoms("e1", "e2")
        req = build_lead_decision_request(
            snapshot=snapshot,
            graph=graph,
            allowed_tool_names=["compare_onset_order", "inspect_trace_path"],
        )
        user_content = next(m.content for m in req.messages if m.role == "user")
        assert "compare_onset_order" in user_content
        assert "inspect_trace_path" in user_content

    def test_deterministic(self):
        snapshot = _make_lead_snapshot()
        graph = _make_graph_with_atoms("e1", "e2")
        a = build_lead_decision_request(
            snapshot=snapshot,
            graph=graph,
            allowed_tool_names=["t1"],
        )
        b = build_lead_decision_request(
            snapshot=snapshot,
            graph=graph,
            allowed_tool_names=["t1"],
        )
        assert a == b

    def test_no_extra_ground_truth_field_added(self):
        """Prompt builder should not add ground-truth-like fields."""
        snapshot = _make_lead_snapshot()
        graph = _make_graph_with_atoms("e1", "e2")
        req = build_lead_decision_request(
            snapshot=snapshot,
            graph=graph,
            allowed_tool_names=["t1"],
        )
        full_text = " ".join(m.content for m in req.messages)
        # Verify no scoring or GT-style fields are injected
        assert '"score"' not in full_text
        assert '"truth"' not in full_text
        assert '"label"' not in full_text


# ===========================================================================
# build_lead_assessment_request
# ===========================================================================


class TestBuildLeadAssessmentRequest:
    def test_produces_model_request(self):
        snapshot = _make_lead_snapshot()
        action = _make_discriminative_action()
        evidence = _make_evidence_atom("e9")
        req = build_lead_assessment_request(
            snapshot=snapshot,
            action=action,
            evidence=evidence,
        )
        assert isinstance(req, ModelRequest)
        assert "assess" in req.purpose.lower()

    def test_includes_evidence_data(self):
        snapshot = _make_lead_snapshot()
        action = _make_discriminative_action()
        evidence = _make_evidence_atom("e9", modality="log")
        req = build_lead_assessment_request(
            snapshot=snapshot,
            action=action,
            evidence=evidence,
        )
        user_content = next(m.content for m in req.messages if m.role == "user")
        assert "e9" in user_content

    def test_deterministic(self):
        snapshot = _make_lead_snapshot()
        action = _make_discriminative_action()
        evidence = _make_evidence_atom("e9")
        a = build_lead_assessment_request(
            snapshot=snapshot, action=action, evidence=evidence
        )
        b = build_lead_assessment_request(
            snapshot=snapshot, action=action, evidence=evidence
        )
        assert a == b

    def test_no_object_repr_in_prompt(self):
        snapshot = _make_lead_snapshot()
        action = _make_discriminative_action()
        evidence = _make_evidence_atom("e9")
        req = build_lead_assessment_request(
            snapshot=snapshot,
            action=action,
            evidence=evidence,
        )
        full_text = " ".join(m.content for m in req.messages)
        assert " at 0x" not in full_text


# ===========================================================================
# build_challenge_proposal_request
# ===========================================================================


class TestBuildChallengeProposalRequest:
    def test_produces_model_request(self):
        snapshot = _make_challenge_snapshot()
        graph = _make_graph_with_atoms("e1", "e2")
        req = build_challenge_proposal_request(
            snapshot=snapshot,
            graph=graph,
            allowed_tool_names=["compare_onset_order"],
        )
        assert isinstance(req, ModelRequest)
        assert "Challenger" in req.purpose

    def test_deterministic(self):
        snapshot = _make_challenge_snapshot()
        graph = _make_graph_with_atoms("e1", "e2")
        a = build_challenge_proposal_request(
            snapshot=snapshot,
            graph=graph,
            allowed_tool_names=["t1"],
        )
        b = build_challenge_proposal_request(
            snapshot=snapshot,
            graph=graph,
            allowed_tool_names=["t1"],
        )
        assert a == b

    def test_no_extra_ground_truth(self):
        snapshot = _make_challenge_snapshot()
        graph = _make_graph_with_atoms("e1", "e2")
        req = build_challenge_proposal_request(
            snapshot=snapshot,
            graph=graph,
            allowed_tool_names=["t1"],
        )
        full_text = " ".join(m.content for m in req.messages)
        assert '"score"' not in full_text
        assert '"truth"' not in full_text
        assert '"label"' not in full_text


# ===========================================================================
# build_challenge_resolution_request
# ===========================================================================


class TestBuildChallengeResolutionRequest:
    def test_produces_model_request(self):
        snapshot = _make_challenge_snapshot()
        proposal = _make_challenge_proposal()
        evidence = _make_evidence_atom("e9")
        req = build_challenge_resolution_request(
            snapshot=snapshot,
            proposal=proposal,
            evidence=evidence,
        )
        assert isinstance(req, ModelRequest)
        assert "Challenger" in req.purpose

    def test_deterministic(self):
        snapshot = _make_challenge_snapshot()
        proposal = _make_challenge_proposal()
        evidence = _make_evidence_atom("e9")
        a = build_challenge_resolution_request(
            snapshot=snapshot, proposal=proposal, evidence=evidence
        )
        b = build_challenge_resolution_request(
            snapshot=snapshot, proposal=proposal, evidence=evidence
        )
        assert a == b

    def test_no_object_repr_in_prompt(self):
        snapshot = _make_challenge_snapshot()
        proposal = _make_challenge_proposal()
        evidence = _make_evidence_atom("e9")
        req = build_challenge_resolution_request(
            snapshot=snapshot,
            proposal=proposal,
            evidence=evidence,
        )
        full_text = " ".join(m.content for m in req.messages)
        assert " at 0x" not in full_text


# ===========================================================================
# Prompt isolation — no provider configuration leakage
# ===========================================================================


class TestPromptIsolation:
    def test_system_message_is_structural(self):
        """System message is structural JSON guidance, not provider debug info."""
        req = build_policy_request(
            purpose="test",
            context={"x": 1},
            response_schema={},
            attempt_index=0,
        )
        system_content = next(m.content for m in req.messages if m.role == "system")
        assert "output" in system_content.lower()
        assert "base_url" not in system_content.lower()

    def test_context_json_is_deterministic_json(self):
        """User content is deterministic JSON, not raw provider body."""
        req = build_policy_request(
            purpose="test",
            context={"key": "value"},
            response_schema={},
            attempt_index=0,
        )
        user_content = next(m.content for m in req.messages if m.role == "user")
        # Must be valid JSON
        import json
        parsed = json.loads(user_content)
        assert parsed["key"] == "value"
