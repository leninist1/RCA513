"""Tests for LeadTournamentController: validation, budget, nomination guardrails."""

import pytest

from prismv4.prism_cht.action_gate import ActionGate
from prismv4.prism_cht.action_schema import DiscriminativeAction
from prismv4.prism_cht.assessment_gate import EvidenceAssessmentGate
from prismv4.prism_cht.evidence_graph import EvidenceAtom, EvidenceGraph
from prismv4.prism_cht.executor import InvestigationExecutor
from prismv4.prism_cht.hypothesis import CausalHypothesis, HypothesisStatus
from prismv4.prism_cht.lead_controller import (
    LeadTournamentController,
    NominationRejectedError,
    TournamentBudgetExhaustedError,
)
from prismv4.prism_cht.lead_policy import ScriptedInvestigationTurn, ScriptedLeadPolicy
from prismv4.prism_cht.telemetry_store import MockTelemetryStore, OnsetObservation
from prismv4.prism_cht.tool_registry import ToolRegistry
from prismv4.prism_cht.tools.compare_onset_order import CompareOnsetOrderTool
from prismv4.prism_cht.tournament_types import (
    AssessmentOutcome,
    EvidenceAssessment,
    EvidenceLinkProposal,
    EvidenceRelation,
    HypothesisStatusUpdate,
    LeadNomination,
    TripletEvidenceCoverage,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_hypothesis(hid="H1", component="comp-a", activate=True):
    h = CausalHypothesis(
        hypothesis_id=hid,
        root_component=component,
        reason_family="cpu_exhaustion",
        onset_interval=(1000.0, 2000.0),
        local_trigger="CPU spike",
        propagation_path=[component],
        explained_symptoms=["latency"],
        predicted_observations=["high CPU"],
        falsifiers=["no CPU spike"],
    )
    if activate:
        h.activate()
    return h


def _make_store():
    return MockTelemetryStore(
        onset_observations=[
            OnsetObservation("comp-a", "cpu", 1100.0, "prom"),
            OnsetObservation("comp-b", "cpu", 1200.0, "prom"),
        ]
    )


def _make_controller(max_rounds=4, store=None):
    gate = ActionGate()
    registry = ToolRegistry()
    registry.register(CompareOnsetOrderTool())
    graph = EvidenceGraph()
    assessment_gate = EvidenceAssessmentGate()
    executor = InvestigationExecutor(
        gate=gate,
        registry=registry,
        graph=graph,
        store=store or _make_store(),
    )
    return LeadTournamentController(
        executor=executor,
        assessment_gate=assessment_gate,
        graph=graph,
        max_rounds=max_rounds,
    )


def _make_action(**overrides):
    defaults = {
        "action_id": "A1",
        "action_type": "run_discriminative_test",
        "target_hypothesis_ids": ("H1", "H2"),
        "question": "Which fails first?",
        "tool_name": "compare_onset_order",
        "args": {
            "component_scope": ["comp-a", "comp-b"],
            "signal_scope": ["cpu"],
            "time_window": [1000.0, 2000.0],
        },
        "expected_outcomes": {
            "H1": "comp-a first",
            "H2": "comp-b first",
        },
        "why_discriminative": "Opposite onset order",
    }
    defaults.update(overrides)
    return DiscriminativeAction(**defaults)


def _setup_tournament_for_nomination_test(*, h1_status, h2_status, graph, eid="e1"):
    """Helper: register hypotheses with given statuses, add evidence, optionally link."""
    h1 = _make_hypothesis("H1", "comp-a", activate=False)
    h2 = _make_hypothesis("H2", "comp-b", activate=False)
    # Manually set statuses
    h1.status = h1_status
    h2.status = h2_status
    graph.register_hypothesis(h1)
    graph.register_hypothesis(h2)
    return h1, h2, graph


# ===========================================================================
# 1. Minimum two hypotheses
# ===========================================================================


class TestMinHypotheses:
    def test_requires_at_least_two(self):
        ctrl = _make_controller()
        h1 = _make_hypothesis("H1")
        # Policy that immediately nominates
        policy = ScriptedLeadPolicy(
            turns=[],
            nomination=LeadNomination("H1", ("evidence:x",), ("H2",), TripletEvidenceCoverage(("dummy",), ("dummy",), ("dummy",)), "rationale"),
        )
        with pytest.raises(ValueError, match="At least two"):
            ctrl.run(initial_hypotheses=[h1], policy=policy)


# ===========================================================================
# 2. DRAFT auto-activate
# ===========================================================================


class TestDraftAutoActivate:
    def test_draft_hypothesis_auto_activated(self):
        ctrl = _make_controller()
        h1 = _make_hypothesis("H1", activate=False)
        h2 = _make_hypothesis("H2", activate=False)
        assert h1.status == HypothesisStatus.DRAFT
        assert h2.status == HypothesisStatus.DRAFT

        # Policy that immediately nominates H1 (will fail on nomination validation)
        policy = ScriptedLeadPolicy(
            turns=[],
            nomination=LeadNomination(
                "H1", ("evidence:will-be-filled",), ("H2",), TripletEvidenceCoverage(("dummy",), ("dummy",), ("dummy",)), "rationale"
            ),
        )

        with pytest.raises(NominationRejectedError):
            ctrl.run(initial_hypotheses=[h1, h2], policy=policy)

        # After controller.run() processes them (even though nomination fails),
        # the hypotheses should have been activated
        assert h1.status == HypothesisStatus.ACTIVE
        assert h2.status == HypothesisStatus.ACTIVE


# ===========================================================================
# 3. ACTIVE can be initial input
# ===========================================================================


class TestActiveInitial:
    def test_active_hypotheses_accepted(self):
        ctrl = _make_controller(max_rounds=1)
        h1 = _make_hypothesis("H1")
        h2 = _make_hypothesis("H2")

        action = _make_action()
        sig = action.query_signature()
        eid = "evidence:" + sig

        # Support H1, contradict H2
        link_support = EvidenceLinkProposal("H1", eid, EvidenceRelation.SUPPORTS, "supports H1")
        link_contra = EvidenceLinkProposal("H2", eid, EvidenceRelation.CONTRADICTS, "contra H2")
        status_support = HypothesisStatusUpdate("H1", HypothesisStatus.SUPPORTED, "supported")
        status_weakened = HypothesisStatusUpdate("H2", HypothesisStatus.WEAKENED, "weakened")
        assessment = EvidenceAssessment(
            action_id="A1", evidence_id=eid,
            outcome=AssessmentOutcome.INFORMATIVE,
            links=(link_support, link_contra),
            status_updates=(status_support, status_weakened),
            rationale="onset supports H1, contradicts H2",
        )
        nomination = LeadNomination("H1", (eid,), ("H2",), TripletEvidenceCoverage((eid,), (eid,), (eid,)), "H1 wins")
        policy = ScriptedLeadPolicy(
            turns=[ScriptedInvestigationTurn(action=action, assessment=assessment)],
            nomination=nomination,
        )

        result = ctrl.run(initial_hypotheses=[h1, h2], policy=policy)
        assert result.status == "challenge_required"
        assert result.nominated_hypothesis_id == "H1"


# ===========================================================================
# 4. SUPPORTED cannot be initial input
# ===========================================================================


class TestSupportedNotInitial:
    def test_supported_rejected_as_initial(self):
        h1 = _make_hypothesis("H1")
        h1.transition_to(HypothesisStatus.SUPPORTED)
        h2 = _make_hypothesis("H2")

        ctrl = _make_controller()
        policy = ScriptedLeadPolicy(
            turns=[],
            nomination=LeadNomination("H1", ("x",), ("H2",), TripletEvidenceCoverage(("dummy",), ("dummy",), ("dummy",)), "..."),
        )
        with pytest.raises(ValueError, match="only DRAFT or ACTIVE"):
            ctrl.run(initial_hypotheses=[h1, h2], policy=policy)


# ===========================================================================
# 5. Register all hypotheses
# ===========================================================================


class TestRegistration:
    def test_all_hypotheses_registered(self):
        ctrl = _make_controller(max_rounds=1)
        h1 = _make_hypothesis("H1")
        h2 = _make_hypothesis("H2")

        action = _make_action()
        sig = action.query_signature()
        eid = "evidence:" + sig

        link_support = EvidenceLinkProposal("H1", eid, EvidenceRelation.SUPPORTS, "supports H1")
        link_contra = EvidenceLinkProposal("H2", eid, EvidenceRelation.CONTRADICTS, "contra H2")
        status_support = HypothesisStatusUpdate("H1", HypothesisStatus.SUPPORTED, "supported")
        status_weakened = HypothesisStatusUpdate("H2", HypothesisStatus.WEAKENED, "weakened")
        assessment = EvidenceAssessment(
            action_id="A1", evidence_id=eid,
            outcome=AssessmentOutcome.INFORMATIVE,
            links=(link_support, link_contra),
            status_updates=(status_support, status_weakened),
            rationale="onset supports H1, contradicts H2",
        )
        nomination = LeadNomination("H1", (eid,), ("H2",), TripletEvidenceCoverage((eid,), (eid,), (eid,)), "H1 wins")
        policy = ScriptedLeadPolicy(
            turns=[ScriptedInvestigationTurn(action=action, assessment=assessment)],
            nomination=nomination,
        )

        ctrl.run(initial_hypotheses=[h1, h2], policy=policy)
        assert "H1" in ctrl._graph.hypotheses_by_id
        assert "H2" in ctrl._graph.hypotheses_by_id


# ===========================================================================
# 6. Cannot run twice
# ===========================================================================


class TestRunOnce:
    def test_cannot_run_twice(self):
        ctrl = _make_controller(max_rounds=1)
        h1 = _make_hypothesis("H1")
        h2 = _make_hypothesis("H2")

        action = _make_action()
        sig = action.query_signature()
        eid = "evidence:" + sig

        link_support = EvidenceLinkProposal("H1", eid, EvidenceRelation.SUPPORTS, "supports H1")
        link_contra = EvidenceLinkProposal("H2", eid, EvidenceRelation.CONTRADICTS, "contra H2")
        status_support = HypothesisStatusUpdate("H1", HypothesisStatus.SUPPORTED, "supported")
        status_weakened = HypothesisStatusUpdate("H2", HypothesisStatus.WEAKENED, "weakened")
        assessment = EvidenceAssessment(
            action_id="A1", evidence_id=eid,
            outcome=AssessmentOutcome.INFORMATIVE,
            links=(link_support, link_contra),
            status_updates=(status_support, status_weakened),
            rationale="onset supports H1, contradicts H2",
        )
        nomination = LeadNomination("H1", (eid,), ("H2",), TripletEvidenceCoverage((eid,), (eid,), (eid,)), "H1 wins")
        policy = ScriptedLeadPolicy(
            turns=[ScriptedInvestigationTurn(action=action, assessment=assessment)],
            nomination=nomination,
        )

        ctrl.run(initial_hypotheses=[h1, h2], policy=policy)
        with pytest.raises(RuntimeError, match="only be called once"):
            ctrl.run(initial_hypotheses=[h1, h2], policy=policy)


# ===========================================================================
# 7. Polluted graph rejected
# ===========================================================================


class TestPollutedGraph:
    def test_polluted_graph_rejected(self):
        graph = EvidenceGraph()
        h_pre = _make_hypothesis("H-pre")
        graph.register_hypothesis(h_pre)

        gate = ActionGate()
        registry = ToolRegistry()
        registry.register(CompareOnsetOrderTool())
        executor = InvestigationExecutor(
            gate=gate, registry=registry, graph=graph, store=_make_store(),
        )
        ctrl = LeadTournamentController(
            executor=executor,
            assessment_gate=EvidenceAssessmentGate(),
            graph=graph,
            max_rounds=4,
        )

        h1 = _make_hypothesis("H1")
        h2 = _make_hypothesis("H2")
        policy = ScriptedLeadPolicy(
            turns=[],
            nomination=LeadNomination("H1", ("x",), ("H2",), TripletEvidenceCoverage(("dummy",), ("dummy",), ("dummy",)), "..."),
        )
        with pytest.raises(ValueError, match="must be empty of registered hypotheses"):
            ctrl.run(initial_hypotheses=[h1, h2], policy=policy)


# ===========================================================================
# 8. Nomination: SUPPORTED hypothesis
# ===========================================================================


class TestNominationSupported:
    def test_nomination_must_be_supported(self):
        ctrl = _make_controller(max_rounds=1)
        h1 = _make_hypothesis("H1")
        h2 = _make_hypothesis("H2")

        action = _make_action()
        sig = action.query_signature()
        eid = "evidence:" + sig

        # Support H1 but DON'T transition to SUPPORTED (no status update)
        link = EvidenceLinkProposal("H1", eid, EvidenceRelation.SUPPORTS, "supports H1")
        link_contra = EvidenceLinkProposal("H2", eid, EvidenceRelation.CONTRADICTS, "contra H2")
        status_weakened = HypothesisStatusUpdate("H2", HypothesisStatus.WEAKENED, "weakened")
        assessment = EvidenceAssessment(
            action_id="A1", evidence_id=eid,
            outcome=AssessmentOutcome.INFORMATIVE,
            links=(link, link_contra),
            status_updates=(status_weakened,),
            rationale="onset supports H1",
        )
        # H1 is still ACTIVE, nomination should fail
        nomination = LeadNomination("H1", (eid,), ("H2",), TripletEvidenceCoverage((eid,), (eid,), (eid,)), "H1 wins")
        policy = ScriptedLeadPolicy(
            turns=[ScriptedInvestigationTurn(action=action, assessment=assessment)],
            nomination=nomination,
        )

        with pytest.raises(NominationRejectedError, match="must be SUPPORTED"):
            ctrl.run(initial_hypotheses=[h1, h2], policy=policy)


# ===========================================================================
# 9. Nomination must reference at least one support evidence
# ===========================================================================


class TestNominationEvidence:
    def test_nomination_needs_support_evidence(self):
        ctrl = _make_controller(max_rounds=1)
        h1 = _make_hypothesis("H1")
        h2 = _make_hypothesis("H2")

        action = _make_action()
        sig = action.query_signature()
        eid = "evidence:" + sig

        link = EvidenceLinkProposal("H1", eid, EvidenceRelation.SUPPORTS, "supports H1")
        link_contra = EvidenceLinkProposal("H2", eid, EvidenceRelation.CONTRADICTS, "contra H2")
        status_support = HypothesisStatusUpdate("H1", HypothesisStatus.SUPPORTED, "supported")
        status_weakened = HypothesisStatusUpdate("H2", HypothesisStatus.WEAKENED, "weakened")
        assessment = EvidenceAssessment(
            action_id="A1", evidence_id=eid,
            outcome=AssessmentOutcome.INFORMATIVE,
            links=(link, link_contra),
            status_updates=(status_support, status_weakened),
            rationale="onset supports H1",
        )
        # Nomination has empty supporting evidence
        nomination = LeadNomination("H1", (), ("H2",), TripletEvidenceCoverage(("dummy",), ("dummy",), ("dummy",)), "H1 wins")
        policy = ScriptedLeadPolicy(
            turns=[ScriptedInvestigationTurn(action=action, assessment=assessment)],
            nomination=nomination,
        )

        with pytest.raises(NominationRejectedError, match="at least one supporting"):
            ctrl.run(initial_hypotheses=[h1, h2], policy=policy)

    def test_nomination_nonexistent_evidence_rejected(self):
        ctrl = _make_controller(max_rounds=1)
        h1 = _make_hypothesis("H1")
        h2 = _make_hypothesis("H2")

        action = _make_action()
        sig = action.query_signature()
        eid = "evidence:" + sig

        link = EvidenceLinkProposal("H1", eid, EvidenceRelation.SUPPORTS, "supports H1")
        link_contra = EvidenceLinkProposal("H2", eid, EvidenceRelation.CONTRADICTS, "contra H2")
        status_support = HypothesisStatusUpdate("H1", HypothesisStatus.SUPPORTED, "supported")
        status_weakened = HypothesisStatusUpdate("H2", HypothesisStatus.WEAKENED, "weakened")
        assessment = EvidenceAssessment(
            action_id="A1", evidence_id=eid,
            outcome=AssessmentOutcome.INFORMATIVE,
            links=(link, link_contra),
            status_updates=(status_support, status_weakened),
            rationale="onset supports H1",
        )
        # Reference nonexistent evidence
        nomination = LeadNomination("H1", ("nonexistent",), ("H2",), TripletEvidenceCoverage(("dummy",), ("dummy",), ("dummy",)), "H1 wins")
        policy = ScriptedLeadPolicy(
            turns=[ScriptedInvestigationTurn(action=action, assessment=assessment)],
            nomination=nomination,
        )

        with pytest.raises(NominationRejectedError, match="does not exist in graph"):
            ctrl.run(initial_hypotheses=[h1, h2], policy=policy)

    def test_nomination_unlinked_evidence_rejected(self):
        ctrl = _make_controller(max_rounds=1)
        h1 = _make_hypothesis("H1")
        h2 = _make_hypothesis("H2")

        action = _make_action()
        sig = action.query_signature()
        eid = "evidence:" + sig

        # Link H2 but NOT H1
        link_contra = EvidenceLinkProposal("H2", eid, EvidenceRelation.CONTRADICTS, "contra H2")
        status_weakened = HypothesisStatusUpdate("H2", HypothesisStatus.WEAKENED, "weakened")
        assessment = EvidenceAssessment(
            action_id="A1", evidence_id=eid,
            outcome=AssessmentOutcome.INFORMATIVE,
            links=(link_contra,),
            status_updates=(status_weakened,),
            rationale="onset contradicts H2",
        )
        # Nominate H1 with eid but eid is not linked to H1
        nomination = LeadNomination("H1", (eid,), ("H2",), TripletEvidenceCoverage((eid,), (eid,), (eid,)), "H1 wins")
        policy = ScriptedLeadPolicy(
            turns=[ScriptedInvestigationTurn(action=action, assessment=assessment)],
            nomination=nomination,
        )

        with pytest.raises(NominationRejectedError, match="must be SUPPORTED"):
            ctrl.run(initial_hypotheses=[h1, h2], policy=policy)


# ===========================================================================
# 12-13. Competitor validation
# ===========================================================================


class TestCompetitorValidation:
    def test_nomination_needs_competitor(self):
        ctrl = _make_controller(max_rounds=1)
        h1 = _make_hypothesis("H1")
        h2 = _make_hypothesis("H2")

        action = _make_action()
        sig = action.query_signature()
        eid = "evidence:" + sig

        link = EvidenceLinkProposal("H1", eid, EvidenceRelation.SUPPORTS, "supports H1")
        link_contra = EvidenceLinkProposal("H2", eid, EvidenceRelation.CONTRADICTS, "contra H2")
        status_support = HypothesisStatusUpdate("H1", HypothesisStatus.SUPPORTED, "supported")
        status_weakened = HypothesisStatusUpdate("H2", HypothesisStatus.WEAKENED, "weakened")
        assessment = EvidenceAssessment(
            action_id="A1", evidence_id=eid,
            outcome=AssessmentOutcome.INFORMATIVE,
            links=(link, link_contra),
            status_updates=(status_support, status_weakened),
            rationale="onset supports H1",
        )
        nomination = LeadNomination("H1", (eid,), (), TripletEvidenceCoverage(("dummy",), ("dummy",), ("dummy",)), "H1 wins")  # no competitors
        policy = ScriptedLeadPolicy(
            turns=[ScriptedInvestigationTurn(action=action, assessment=assessment)],
            nomination=nomination,
        )

        with pytest.raises(NominationRejectedError, match="at least one competitor"):
            ctrl.run(initial_hypotheses=[h1, h2], policy=policy)

    def test_competitor_cannot_be_nominated_self(self):
        ctrl = _make_controller(max_rounds=1)
        h1 = _make_hypothesis("H1")
        h2 = _make_hypothesis("H2")

        action = _make_action()
        sig = action.query_signature()
        eid = "evidence:" + sig

        link = EvidenceLinkProposal("H1", eid, EvidenceRelation.SUPPORTS, "supports H1")
        link_contra = EvidenceLinkProposal("H2", eid, EvidenceRelation.CONTRADICTS, "contra H2")
        status_support = HypothesisStatusUpdate("H1", HypothesisStatus.SUPPORTED, "supported")
        status_weakened = HypothesisStatusUpdate("H2", HypothesisStatus.WEAKENED, "weakened")
        assessment = EvidenceAssessment(
            action_id="A1", evidence_id=eid,
            outcome=AssessmentOutcome.INFORMATIVE,
            links=(link, link_contra),
            status_updates=(status_support, status_weakened),
            rationale="onset supports H1",
        )
        # H1 nominated and also listed as competitor
        nomination = LeadNomination("H1", (eid,), ("H1",), TripletEvidenceCoverage(("dummy",), ("dummy",), ("dummy",)), "H1 wins")
        policy = ScriptedLeadPolicy(
            turns=[ScriptedInvestigationTurn(action=action, assessment=assessment)],
            nomination=nomination,
        )

        with pytest.raises(NominationRejectedError, match="cannot be the nominated"):
            ctrl.run(initial_hypotheses=[h1, h2], policy=policy)

    def test_competitor_must_be_weakened_or_refuted(self):
        ctrl = _make_controller(max_rounds=1)
        h1 = _make_hypothesis("H1")
        h2 = _make_hypothesis("H2")

        action = _make_action()
        sig = action.query_signature()
        eid = "evidence:" + sig

        # Support H1, but DON'T weaken H2
        link = EvidenceLinkProposal("H1", eid, EvidenceRelation.SUPPORTS, "supports H1")
        status_support = HypothesisStatusUpdate("H1", HypothesisStatus.SUPPORTED, "supported")
        assessment = EvidenceAssessment(
            action_id="A1", evidence_id=eid,
            outcome=AssessmentOutcome.INFORMATIVE,
            links=(link,),
            status_updates=(status_support,),
            rationale="onset supports H1",
        )
        nomination = LeadNomination("H1", (eid,), ("H2",), TripletEvidenceCoverage((eid,), (eid,), (eid,)), "H1 wins")
        policy = ScriptedLeadPolicy(
            turns=[ScriptedInvestigationTurn(action=action, assessment=assessment)],
            nomination=nomination,
        )

        with pytest.raises(NominationRejectedError, match="must be WEAKENED or REFUTED"):
            ctrl.run(initial_hypotheses=[h1, h2], policy=policy)

    def test_competitor_must_have_contradiction(self):
        ctrl = _make_controller(max_rounds=1)
        h1 = _make_hypothesis("H1")
        h2 = _make_hypothesis("H2")

        action = _make_action()
        sig = action.query_signature()
        eid = "evidence:" + sig

        link = EvidenceLinkProposal("H1", eid, EvidenceRelation.SUPPORTS, "supports H1")
        status_support = HypothesisStatusUpdate("H1", HypothesisStatus.SUPPORTED, "supported")
        # Weaken H2 but no contradiction link for H2
        status_weakened = HypothesisStatusUpdate("H2", HypothesisStatus.WEAKENED, "weakened")
        assessment = EvidenceAssessment(
            action_id="A1", evidence_id=eid,
            outcome=AssessmentOutcome.INFORMATIVE,
            links=(link,),  # No contradiction link for H2
            status_updates=(status_support, status_weakened),
            rationale="onset supports H1",
        )
        nomination = LeadNomination("H1", (eid,), ("H2",), TripletEvidenceCoverage((eid,), (eid,), (eid,)), "H1 wins")
        policy = ScriptedLeadPolicy(
            turns=[ScriptedInvestigationTurn(action=action, assessment=assessment)],
            nomination=nomination,
        )

        with pytest.raises(NominationRejectedError, match="no contradiction evidence"):
            ctrl.run(initial_hypotheses=[h1, h2], policy=policy)


# ===========================================================================
# 16-17. Nomination not SURVIVED or FINAL
# ===========================================================================


class TestNoSurvivedOrFinal:
    def test_nomination_rejects_survived(self):
        """Directly test that _validate_nomination rejects SURVIVED."""
        ctrl = _make_controller()
        h1 = _make_hypothesis("H1")
        h2 = _make_hypothesis("H2")
        # Register in graph
        ctrl._graph.register_hypothesis(h1)
        ctrl._graph.register_hypothesis(h2)
        ctrl._hypotheses["H1"] = h1
        ctrl._hypotheses["H2"] = h2
        # Force H1 to SURVIVED
        h1.status = HypothesisStatus.SURVIVED

        nom = LeadNomination("H1", ("x",), ("H2",), TripletEvidenceCoverage(("dummy",), ("dummy",), ("dummy",)), "H1 wins")
        with pytest.raises(NominationRejectedError, match="SURVIVED"):
            ctrl._validate_nomination(nom)

    def test_nomination_rejects_final(self):
        """Directly test that _validate_nomination rejects FINAL."""
        ctrl = _make_controller()
        h1 = _make_hypothesis("H1")
        h2 = _make_hypothesis("H2")
        ctrl._graph.register_hypothesis(h1)
        ctrl._graph.register_hypothesis(h2)
        ctrl._hypotheses["H1"] = h1
        ctrl._hypotheses["H2"] = h2
        h1.status = HypothesisStatus.FINAL

        nom = LeadNomination("H1", ("x",), ("H2",), TripletEvidenceCoverage(("dummy",), ("dummy",), ("dummy",)), "H1 wins")
        with pytest.raises(NominationRejectedError, match="FINAL"):
            ctrl._validate_nomination(nom)


# ===========================================================================
# 18. Budget exhaustion
# ===========================================================================


class TestBudgetExhaustion:
    def test_max_rounds_exhausted_raises(self):
        ctrl = _make_controller(max_rounds=1)
        h1 = _make_hypothesis("H1")
        h2 = _make_hypothesis("H2")

        action = _make_action()
        sig = action.query_signature()
        eid = "evidence:" + sig

        link = EvidenceLinkProposal("H1", eid, EvidenceRelation.SUPPORTS, "supports H1")
        assessment = EvidenceAssessment(
            action_id="A1", evidence_id=eid,
            outcome=AssessmentOutcome.INFORMATIVE,
            links=(link,), status_updates=(),
            rationale="onset supports H1",
        )
        # Second action with different query signature
        action2 = _make_action(
            action_id="A2",
            args={
                "component_scope": ["comp-a", "comp-b"],
                "signal_scope": ["load"],
                "time_window": [1000.0, 2000.0],
            },
            expected_outcomes={"H1": "diff1", "H2": "diff2"},
        )
        assessment2 = EvidenceAssessment(
            action_id="A2", evidence_id="evidence:will-differ",
            outcome=AssessmentOutcome.INFORMATIVE,
            links=(link,), status_updates=(),
            rationale="second",
        )
        nomination = LeadNomination("H1", (eid,), ("H2",), TripletEvidenceCoverage((eid,), (eid,), (eid,)), "H1 wins")
        # Two turns but max_rounds=1
        policy = ScriptedLeadPolicy(
            turns=[
                ScriptedInvestigationTurn(action=action, assessment=assessment),
                ScriptedInvestigationTurn(action=action2, assessment=assessment2),
            ],
            nomination=nomination,
        )

        with pytest.raises(TournamentBudgetExhaustedError, match="max_rounds"):
            ctrl.run(initial_hypotheses=[h1, h2], policy=policy)


# ===========================================================================
# 19. Nomination does not consume a round
# ===========================================================================


class TestNominationRoundCount:
    def test_nomination_not_counted_as_round(self):
        ctrl = _make_controller(max_rounds=1)
        h1 = _make_hypothesis("H1")
        h2 = _make_hypothesis("H2")

        action = _make_action()
        sig = action.query_signature()
        eid = "evidence:" + sig

        link_support = EvidenceLinkProposal("H1", eid, EvidenceRelation.SUPPORTS, "supports H1")
        link_contra = EvidenceLinkProposal("H2", eid, EvidenceRelation.CONTRADICTS, "contra H2")
        status_support = HypothesisStatusUpdate("H1", HypothesisStatus.SUPPORTED, "supported")
        status_weakened = HypothesisStatusUpdate("H2", HypothesisStatus.WEAKENED, "weakened")
        assessment = EvidenceAssessment(
            action_id="A1", evidence_id=eid,
            outcome=AssessmentOutcome.INFORMATIVE,
            links=(link_support, link_contra),
            status_updates=(status_support, status_weakened),
            rationale="onset supports H1, contradicts H2",
        )
        nomination = LeadNomination("H1", (eid,), ("H2",), TripletEvidenceCoverage((eid,), (eid,), (eid,)), "H1 wins")
        policy = ScriptedLeadPolicy(
            turns=[ScriptedInvestigationTurn(action=action, assessment=assessment)],
            nomination=nomination,
        )

        result = ctrl.run(initial_hypotheses=[h1, h2], policy=policy)
        assert result.rounds_completed == 1


# ===========================================================================
# 20. Controller does not auto-link evidence
# ===========================================================================


class TestNoAutoLink:
    def test_controller_does_not_auto_link(self):
        ctrl = _make_controller(max_rounds=1)
        h1 = _make_hypothesis("H1")
        h2 = _make_hypothesis("H2")

        action = _make_action()
        sig = action.query_signature()
        eid = "evidence:" + sig

        # Assessment only supports H1, also weakens H2
        link = EvidenceLinkProposal("H1", eid, EvidenceRelation.SUPPORTS, "supports H1")
        link_contra = EvidenceLinkProposal("H2", eid, EvidenceRelation.CONTRADICTS, "contra H2")
        status = HypothesisStatusUpdate("H1", HypothesisStatus.SUPPORTED, "supported")
        status_weakened = HypothesisStatusUpdate("H2", HypothesisStatus.WEAKENED, "weakened")
        assessment = EvidenceAssessment(
            action_id="A1", evidence_id=eid,
            outcome=AssessmentOutcome.INFORMATIVE,
            links=(link, link_contra),
            status_updates=(status, status_weakened),
            rationale="supports H1, contradicts H2",
        )
        nomination = LeadNomination("H1", (eid,), ("H2",), TripletEvidenceCoverage((eid,), (eid,), (eid,)), "H1 wins")
        policy = ScriptedLeadPolicy(
            turns=[ScriptedInvestigationTurn(action=action, assessment=assessment)],
            nomination=nomination,
        )

        result = ctrl.run(initial_hypotheses=[h1, h2], policy=policy)

        # H2 should have no support edges added automatically
        assert eid not in ctrl._graph.support_edges.get("H2", set())
