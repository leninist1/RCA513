"""Tests for CHT-2.1: atomic evidence assessment application.

Covers:
- EvidenceGraph.relation_transaction() commit / rollback semantics
- EvidenceAssessmentGate.apply() atomicity guarantees
- Session boundary invariants for Lead tournaments
- Regression invariants for the demo scenario
"""

import pytest

from prismv4.prism_cht.action_gate import ActionGate
from prismv4.prism_cht.action_schema import DiscriminativeAction
from prismv4.prism_cht.assessment_gate import EvidenceAssessmentGate
from prismv4.prism_cht.demo_scenario import build_demo_lead_tournament
from prismv4.prism_cht.evidence_graph import EvidenceAtom, EvidenceGraph
from prismv4.prism_cht.executor import InvestigationExecutor
from prismv4.prism_cht.hypothesis import CausalHypothesis, HypothesisStatus
from prismv4.prism_cht.lead_controller import LeadTournamentController
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
)


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


def _make_evidence(evidence_id="e1", **overrides):
    defaults = {
        "evidence_id": evidence_id,
        "query_signature": "sig-abc123",
        "modality": "onset",
        "component_scope": ("comp-a", "comp-b"),
        "time_window": (1000.0, 2000.0),
        "observation": {"onsets": []},
        "provenance": {"source": "mock"},
    }
    defaults.update(overrides)
    return EvidenceAtom(**defaults)


def _make_graph_with_h2(h1_status=HypothesisStatus.ACTIVE, h2_status=HypothesisStatus.ACTIVE):
    h1 = _make_hypothesis("H1", "comp-a", activate=False)
    h2 = _make_hypothesis("H2", "comp-b", activate=False)
    h1.status = h1_status
    h2.status = h2_status
    graph = EvidenceGraph()
    graph.register_hypothesis(h1)
    graph.register_hypothesis(h2)
    evidence = _make_evidence("e1")
    graph.add_evidence(evidence)
    return graph, h1, h2, evidence


# ===========================================================================
# A. relation_transaction basic behaviour
# ===========================================================================


class TestRelationTransactionBasic:
    def test_empty_transaction_commits(self):
        graph, h1, h2, _ = _make_graph_with_h2()
        pre_count = graph.evidence_count()
        pre_h1_status = h1.status
        with graph.relation_transaction():
            pass
        assert graph.evidence_count() == pre_count
        assert h1.status == pre_h1_status
        graph.validate_consistency()

    def test_link_support_commits(self):
        graph, h1, h2, ev = _make_graph_with_h2()
        with graph.relation_transaction():
            graph.link_support("H1", "e1")
        assert "e1" in graph.support_edges["H1"]
        assert "e1" in h1.supporting_evidence_ids
        graph.validate_consistency()

    def test_link_contradiction_commits(self):
        graph, h1, h2, ev = _make_graph_with_h2()
        with graph.relation_transaction():
            graph.link_contradiction("H1", "e1")
        assert "e1" in graph.contradiction_edges["H1"]
        assert "e1" in h1.contradicting_evidence_ids
        graph.validate_consistency()

    def test_status_transition_commits(self):
        graph, h1, h2, _ = _make_graph_with_h2()
        with graph.relation_transaction():
            h1.transition_to(HypothesisStatus.SUPPORTED)
        assert h1.status == HypothesisStatus.SUPPORTED
        graph.validate_consistency()

    def test_consistency_passes_after_commit(self):
        graph, h1, h2, ev = _make_graph_with_h2()
        with graph.relation_transaction():
            graph.link_support("H1", "e1")
            h1.transition_to(HypothesisStatus.SUPPORTED)
        graph.validate_consistency()


# ===========================================================================
# B. Rollback behaviour
# ===========================================================================


class TestRelationTransactionRollback:
    def test_link_support_rollback_edges(self):
        graph, h1, h2, ev = _make_graph_with_h2()
        with pytest.raises(RuntimeError, match="injected failure"):
            with graph.relation_transaction():
                graph.link_support("H1", "e1")
                raise RuntimeError("injected failure")
        assert "e1" not in graph.support_edges.get("H1", set())
        assert len(graph.support_edges.get("H1", set())) == 0

    def test_link_support_rollback_hypothesis_ids(self):
        graph, h1, h2, ev = _make_graph_with_h2()
        with pytest.raises(RuntimeError, match="injected failure"):
            with graph.relation_transaction():
                graph.link_support("H1", "e1")
                raise RuntimeError("injected failure")
        assert "e1" not in h1.supporting_evidence_ids

    def test_link_contradiction_rollback_edges(self):
        graph, h1, h2, ev = _make_graph_with_h2()
        with pytest.raises(RuntimeError, match="injected failure"):
            with graph.relation_transaction():
                graph.link_contradiction("H1", "e1")
                raise RuntimeError("injected failure")
        assert "e1" not in graph.contradiction_edges.get("H1", set())

    def test_link_contradiction_rollback_hypothesis_ids(self):
        graph, h1, h2, ev = _make_graph_with_h2()
        with pytest.raises(RuntimeError, match="injected failure"):
            with graph.relation_transaction():
                graph.link_contradiction("H1", "e1")
                raise RuntimeError("injected failure")
        assert "e1" not in h1.contradicting_evidence_ids

    def test_status_rollback(self):
        graph, h1, h2, _ = _make_graph_with_h2()
        pre_status = h1.status
        with pytest.raises(RuntimeError, match="injected failure"):
            with graph.relation_transaction():
                h1.transition_to(HypothesisStatus.SUPPORTED)
                raise RuntimeError("injected failure")
        assert h1.status == pre_status

    def test_multi_edge_rollback(self):
        graph, h1, h2, ev = _make_graph_with_h2()
        evidence2 = _make_evidence("e2", query_signature="sig-other")
        graph.add_evidence(evidence2)
        with pytest.raises(RuntimeError, match="injected failure"):
            with graph.relation_transaction():
                graph.link_support("H1", "e1")
                graph.link_support("H1", "e2")
                graph.link_contradiction("H2", "e1")
                raise RuntimeError("injected failure")
        assert "e1" not in graph.support_edges.get("H1", set())
        assert "e2" not in graph.support_edges.get("H1", set())
        assert "e1" not in graph.contradiction_edges.get("H2", set())
        assert "e1" not in h1.supporting_evidence_ids
        assert "e2" not in h1.supporting_evidence_ids

    def test_multi_status_rollback(self):
        graph, h1, h2, _ = _make_graph_with_h2()
        pre_h1 = h1.status
        pre_h2 = h2.status
        with pytest.raises(RuntimeError, match="injected failure"):
            with graph.relation_transaction():
                h1.transition_to(HypothesisStatus.SUPPORTED)
                h2.transition_to(HypothesisStatus.WEAKENED)
                raise RuntimeError("injected failure")
        assert h1.status == pre_h1
        assert h2.status == pre_h2

    def test_rollback_preserves_evidence_atoms(self):
        graph, h1, h2, ev = _make_graph_with_h2()
        with pytest.raises(RuntimeError, match="injected failure"):
            with graph.relation_transaction():
                graph.link_support("H1", "e1")
                raise RuntimeError("injected failure")
        assert "e1" in graph.evidence_by_id

    def test_rollback_preserves_evidence_count(self):
        graph, h1, h2, ev = _make_graph_with_h2()
        pre_count = graph.evidence_count()
        with pytest.raises(RuntimeError, match="injected failure"):
            with graph.relation_transaction():
                graph.link_support("H1", "e1")
                raise RuntimeError("injected failure")
        assert graph.evidence_count() == pre_count

    def test_rollback_consistency_passes(self):
        graph, h1, h2, ev = _make_graph_with_h2()
        with pytest.raises(RuntimeError, match="injected failure"):
            with graph.relation_transaction():
                graph.link_support("H1", "e1")
                raise RuntimeError("injected failure")
        graph.validate_consistency()

    def test_original_exception_re_raised(self):
        graph, h1, h2, _ = _make_graph_with_h2()
        with pytest.raises(RuntimeError, match="injected failure"):
            with graph.relation_transaction():
                raise RuntimeError("injected failure")


# ===========================================================================
# C. AssessmentGate.apply atomicity
# ===========================================================================


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


class TestAssessmentGateAtomicity:
    def test_legal_assessment_apply(self):
        graph, h1, h2, ev = _make_graph_with_h2()
        action = _make_action()
        link = EvidenceLinkProposal(
            hypothesis_id="H1", evidence_id="e1",
            relation=EvidenceRelation.SUPPORTS, rationale="supports H1",
        )
        status_update = HypothesisStatusUpdate(
            hypothesis_id="H1", new_status=HypothesisStatus.SUPPORTED,
            rationale="supported by evidence",
        )
        assessment = EvidenceAssessment(
            action_id="A1", evidence_id="e1",
            outcome=AssessmentOutcome.INFORMATIVE,
            links=(link,), status_updates=(status_update,),
            rationale="onset supports H1",
        )
        gate = EvidenceAssessmentGate()
        gate.apply(assessment=assessment, action=action, evidence=ev,
                   hypotheses={"H1": h1, "H2": h2}, graph=graph)
        assert "e1" in graph.support_edges["H1"]
        assert h1.status == HypothesisStatus.SUPPORTED
        graph.validate_consistency()

    def test_illegal_assessment_rejected_before_mutate(self):
        graph, h1, h2, ev = _make_graph_with_h2()
        action = _make_action()
        assessment = EvidenceAssessment(
            action_id="WRONG", evidence_id="e1",
            outcome=AssessmentOutcome.INFORMATIVE,
            links=(), status_updates=(), rationale="...",
        )
        gate = EvidenceAssessmentGate()
        with pytest.raises(Exception):
            gate.apply(assessment=assessment, action=action, evidence=ev,
                       hypotheses={"H1": h1, "H2": h2}, graph=graph)
        assert "e1" not in graph.support_edges.get("H1", set())

    def test_second_link_failure_rolls_back_first(self, monkeypatch):
        graph, h1, h2, ev = _make_graph_with_h2()
        action = _make_action()

        original_link_support = graph.link_support
        call_count = [0]

        def failing_link_support(hid, eid):
            call_count[0] += 1
            original_link_support(hid, eid)
            if call_count[0] >= 2:
                raise RuntimeError("injected second link failure")

        monkeypatch.setattr(graph, "link_support", failing_link_support)

        link1 = EvidenceLinkProposal(
            hypothesis_id="H1", evidence_id="e1",
            relation=EvidenceRelation.SUPPORTS, rationale="supports H1",
        )
        link2 = EvidenceLinkProposal(
            hypothesis_id="H2", evidence_id="e1",
            relation=EvidenceRelation.SUPPORTS, rationale="supports H2",
        )
        assessment = EvidenceAssessment(
            action_id="A1", evidence_id="e1",
            outcome=AssessmentOutcome.INFORMATIVE,
            links=(link1, link2), status_updates=(), rationale="...",
        )
        gate = EvidenceAssessmentGate()
        with pytest.raises(RuntimeError, match="injected second link failure"):
            gate.apply(assessment=assessment, action=action, evidence=ev,
                       hypotheses={"H1": h1, "H2": h2}, graph=graph)
        assert "e1" not in graph.support_edges.get("H1", set())
        assert "e1" not in graph.support_edges.get("H2", set())
        assert "e1" not in h1.supporting_evidence_ids
        assert "e1" not in h2.supporting_evidence_ids
        assert h1.status == HypothesisStatus.ACTIVE
        assert h2.status == HypothesisStatus.ACTIVE

    def test_transition_to_failure_rolls_back_links(self, monkeypatch):
        graph, h1, h2, ev = _make_graph_with_h2()
        action = _make_action()

        def failing_transition_to(h, new_status):
            raise RuntimeError("injected transition failure")

        monkeypatch.setattr(
            CausalHypothesis, "transition_to",
            lambda self, new_status: failing_transition_to(self, new_status),
        )

        link = EvidenceLinkProposal(
            hypothesis_id="H1", evidence_id="e1",
            relation=EvidenceRelation.SUPPORTS, rationale="supports H1",
        )
        status_update = HypothesisStatusUpdate(
            hypothesis_id="H1", new_status=HypothesisStatus.SUPPORTED,
            rationale="supported",
        )
        assessment = EvidenceAssessment(
            action_id="A1", evidence_id="e1",
            outcome=AssessmentOutcome.INFORMATIVE,
            links=(link,), status_updates=(status_update,),
            rationale="onset supports H1",
        )
        gate = EvidenceAssessmentGate()
        with pytest.raises(RuntimeError, match="injected transition failure"):
            gate.apply(assessment=assessment, action=action, evidence=ev,
                       hypotheses={"H1": h1, "H2": h2}, graph=graph)
        assert "e1" not in graph.support_edges.get("H1", set())
        assert "e1" not in h1.supporting_evidence_ids
        assert h1.status == HypothesisStatus.ACTIVE

    def test_validate_consistency_failure_rolls_back(self, monkeypatch):
        graph, h1, h2, ev = _make_graph_with_h2()
        action = _make_action()

        def failing_validate(self):
            raise ValueError("injected consistency failure")

        monkeypatch.setattr(EvidenceGraph, "validate_consistency", failing_validate)

        link = EvidenceLinkProposal(
            hypothesis_id="H1", evidence_id="e1",
            relation=EvidenceRelation.SUPPORTS, rationale="supports H1",
        )
        assessment = EvidenceAssessment(
            action_id="A1", evidence_id="e1",
            outcome=AssessmentOutcome.INFORMATIVE,
            links=(link,), status_updates=(), rationale="...",
        )
        gate = EvidenceAssessmentGate()
        with pytest.raises(ValueError, match="injected consistency failure"):
            gate.apply(assessment=assessment, action=action, evidence=ev,
                       hypotheses={"H1": h1, "H2": h2}, graph=graph)
        assert "e1" not in graph.support_edges.get("H1", set())
        assert "e1" not in h1.supporting_evidence_ids


# ===========================================================================
# D. Session boundary
# ===========================================================================


def _make_store():
    return MockTelemetryStore(
        onset_observations=[
            OnsetObservation("comp-a", "cpu", 1100.0, "prom"),
            OnsetObservation("comp-b", "cpu", 1200.0, "prom"),
        ]
    )


def _make_controller(max_rounds=4):
    gate = ActionGate()
    registry = ToolRegistry()
    registry.register(CompareOnsetOrderTool())
    graph = EvidenceGraph()
    assessment_gate = EvidenceAssessmentGate()
    executor = InvestigationExecutor(
        gate=gate, registry=registry, graph=graph, store=_make_store(),
    )
    ctrl = LeadTournamentController(
        executor=executor, assessment_gate=assessment_gate,
        graph=graph, max_rounds=max_rounds,
    )
    return ctrl


class TestSessionBoundary:
    def test_single_case_multi_round_shares_action_gate(self):
        ctrl = _make_controller(max_rounds=2)
        h1 = _make_hypothesis("H1")
        h2 = _make_hypothesis("H2")

        action = DiscriminativeAction(
            action_id="A1",
            action_type="run_discriminative_test",
            target_hypothesis_ids=("H1", "H2"),
            question="Which fails first?",
            tool_name="compare_onset_order",
            args={
                "component_scope": ["comp-a", "comp-b"],
                "signal_scope": ["cpu"],
                "time_window": [1000.0, 2000.0],
            },
            expected_outcomes={"H1": "H1 first", "H2": "H2 first"},
            why_discriminative="Opposite onset order",
        )
        sig = action.query_signature()
        eid = "evidence:" + sig
        link_sup = EvidenceLinkProposal("H1", eid, EvidenceRelation.SUPPORTS, "supports")
        link_con = EvidenceLinkProposal("H2", eid, EvidenceRelation.CONTRADICTS, "contradicts")
        status_sup = HypothesisStatusUpdate("H1", HypothesisStatus.SUPPORTED, "supported")
        status_weakened = HypothesisStatusUpdate("H2", HypothesisStatus.WEAKENED, "weakened")
        assessment = EvidenceAssessment(
            action_id="A1", evidence_id=eid,
            outcome=AssessmentOutcome.INFORMATIVE,
            links=(link_sup, link_con),
            status_updates=(status_sup, status_weakened),
            rationale="onset supports H1, contradicts H2",
        )
        nomination = LeadNomination("H1", (eid,), ("H2",), "H1 wins")
        policy = ScriptedLeadPolicy(
            turns=[ScriptedInvestigationTurn(action=action, assessment=assessment)],
            nomination=nomination,
        )
        result = ctrl.run(initial_hypotheses=[h1, h2], policy=policy)
        assert result.status == "challenge_required"

    def test_single_case_same_action_not_re_executable(self):
        ctrl = _make_controller(max_rounds=4)
        h1 = _make_hypothesis("H1")
        h2 = _make_hypothesis("H2")
        action = DiscriminativeAction(
            action_id="A1",
            action_type="run_discriminative_test",
            target_hypothesis_ids=("H1", "H2"),
            question="Which fails first?",
            tool_name="compare_onset_order",
            args={
                "component_scope": ["comp-a", "comp-b"],
                "signal_scope": ["cpu"],
                "time_window": [1000.0, 2000.0],
            },
            expected_outcomes={"H1": "H1 first", "H2": "H2 first"},
            why_discriminative="Opposite onset order",
        )
        sig = action.query_signature()
        eid = "evidence:" + sig
        link_sup = EvidenceLinkProposal("H1", eid, EvidenceRelation.SUPPORTS, "supports")
        link_con = EvidenceLinkProposal("H2", eid, EvidenceRelation.CONTRADICTS, "contradicts")
        status_sup = HypothesisStatusUpdate("H1", HypothesisStatus.SUPPORTED, "supported")
        status_weakened = HypothesisStatusUpdate("H2", HypothesisStatus.WEAKENED, "weakened")
        assessment = EvidenceAssessment(
            action_id="A1", evidence_id=eid,
            outcome=AssessmentOutcome.INFORMATIVE,
            links=(link_sup, link_con),
            status_updates=(status_sup, status_weakened),
            rationale="onset supports H1, contradicts H2",
        )
        nomination = LeadNomination("H1", (eid,), ("H2",), "H1 wins")
        policy = ScriptedLeadPolicy(
            turns=[ScriptedInvestigationTurn(action=action, assessment=assessment)],
            nomination=nomination,
        )
        ctrl.run(initial_hypotheses=[h1, h2], policy=policy)
        # signature should now be in executed set
        assert sig in ctrl._executor._gate._executed_signatures

    def test_two_independent_demos_different_evidence_graph(self):
        ctrl1, h1, p1 = build_demo_lead_tournament()
        ctrl2, h2, p2 = build_demo_lead_tournament()
        assert ctrl1._graph is not ctrl2._graph

    def test_two_independent_demos_different_action_gate(self):
        ctrl1, h1, p1 = build_demo_lead_tournament()
        ctrl2, h2, p2 = build_demo_lead_tournament()
        assert ctrl1._executor._gate is not ctrl2._executor._gate

    def test_first_case_runs_second_case_still_runs(self):
        ctrl1, h1, p1 = build_demo_lead_tournament()
        result1 = ctrl1.run(initial_hypotheses=list(h1), policy=p1)
        assert result1.status == "challenge_required"

        ctrl2, h2, p2 = build_demo_lead_tournament()
        result2 = ctrl2.run(initial_hypotheses=list(h2), policy=p2)
        assert result2.status == "challenge_required"

    def test_first_case_signatures_do_not_pollute_second(self):
        ctrl1, h1, p1 = build_demo_lead_tournament()
        result1 = ctrl1.run(initial_hypotheses=list(h1), policy=p1)
        sigs1 = set(ctrl1._executor._gate._executed_signatures)

        ctrl2, h2, p2 = build_demo_lead_tournament()
        sigs2_before = set(ctrl2._executor._gate._executed_signatures)
        assert sigs2_before == set()
        # Verify the first case signatures are not in the second case
        for s in sigs1:
            assert s not in sigs2_before


# ===========================================================================
# E. Regression — demo invariants
# ===========================================================================


class TestDemoRegression:
    def test_demo_outputs_challenge_required(self):
        ctrl, hypotheses, policy = build_demo_lead_tournament()
        result = ctrl.run(initial_hypotheses=list(hypotheses), policy=policy)
        assert result.status == "challenge_required"

    def test_h2_stays_supported(self):
        ctrl, hypotheses, policy = build_demo_lead_tournament()
        result = ctrl.run(initial_hypotheses=list(hypotheses), policy=policy)
        h2 = ctrl._graph.hypotheses_by_id.get("H-db_002-pool")
        assert h2 is not None
        assert h2.status == HypothesisStatus.SUPPORTED

    def test_h2_not_survived(self):
        ctrl, hypotheses, policy = build_demo_lead_tournament()
        result = ctrl.run(initial_hypotheses=list(hypotheses), policy=policy)
        h2 = ctrl._graph.hypotheses_by_id.get("H-db_002-pool")
        assert h2 is not None
        assert h2.status != HypothesisStatus.SURVIVED

    def test_h2_not_final(self):
        ctrl, hypotheses, policy = build_demo_lead_tournament()
        result = ctrl.run(initial_hypotheses=list(hypotheses), policy=policy)
        h2 = ctrl._graph.hypotheses_by_id.get("H-db_002-pool")
        assert h2 is not None
        assert h2.status != HypothesisStatus.FINAL


# ===========================================================================
# F. Rollback evidence ID list order preservation
# ===========================================================================


class TestRollbackEvidenceIdOrder:
    def test_rollback_preserves_supporting_evidence_ids_order(self):
        graph, h1, h2, ev = _make_graph_with_h2()
        ev2 = _make_evidence("e2", query_signature="sig-2")
        ev3 = _make_evidence("e3", query_signature="sig-3")
        ev4 = _make_evidence("e4", query_signature="sig-4")
        graph.add_evidence(ev2)
        graph.add_evidence(ev3)
        graph.add_evidence(ev4)

        # Build initial order [E3, E1, E2] using graph's own methods
        with graph.relation_transaction():
            graph.link_support("H1", "e3")
            graph.link_support("H1", "e1")
            graph.link_support("H1", "e2")
        original_order = tuple(h1.supporting_evidence_ids)
        assert original_order == ("e3", "e1", "e2")

        with pytest.raises(RuntimeError, match="injected failure"):
            with graph.relation_transaction():
                graph.link_support("H1", "e4")
                raise RuntimeError("injected failure")

        restored_order = tuple(h1.supporting_evidence_ids)
        assert restored_order == original_order
        assert restored_order == ("e3", "e1", "e2")
        graph.validate_consistency()

    def test_rollback_preserves_contradicting_evidence_ids_order(self):
        graph, h1, h2, ev = _make_graph_with_h2()
        ev2 = _make_evidence("e2", query_signature="sig-2")
        ev3 = _make_evidence("e3", query_signature="sig-3")
        ev4 = _make_evidence("e4", query_signature="sig-4")
        graph.add_evidence(ev2)
        graph.add_evidence(ev3)
        graph.add_evidence(ev4)

        # Build initial order [E2, E3, E1] using graph's own methods
        with graph.relation_transaction():
            graph.link_contradiction("H1", "e2")
            graph.link_contradiction("H1", "e3")
            graph.link_contradiction("H1", "e1")
        original_order = tuple(h1.contradicting_evidence_ids)
        assert original_order == ("e2", "e3", "e1")

        with pytest.raises(RuntimeError, match="injected failure"):
            with graph.relation_transaction():
                graph.link_contradiction("H1", "e4")
                raise RuntimeError("injected failure")

        restored_order = tuple(h1.contradicting_evidence_ids)
        assert restored_order == original_order
        assert restored_order == ("e2", "e3", "e1")
        graph.validate_consistency()

    def test_rollback_supporting_not_just_set_equivalent_but_list_identical(self):
        graph, h1, h2, ev = _make_graph_with_h2()
        ev2 = _make_evidence("e2", query_signature="sig-2")
        ev3 = _make_evidence("e3", query_signature="sig-3")
        ev4 = _make_evidence("e4", query_signature="sig-4")
        graph.add_evidence(ev2)
        graph.add_evidence(ev3)
        graph.add_evidence(ev4)

        with graph.relation_transaction():
            graph.link_support("H1", "e3")
            graph.link_support("H1", "e1")
            graph.link_support("H1", "e2")
        assert h1.supporting_evidence_ids == ["e3", "e1", "e2"]

        with pytest.raises(RuntimeError, match="injected failure"):
            with graph.relation_transaction():
                graph.link_support("H1", "e4")
                raise RuntimeError("injected failure")

        assert h1.supporting_evidence_ids == ["e3", "e1", "e2"]
        assert set(h1.supporting_evidence_ids) == {"e3", "e1", "e2"}
        graph.validate_consistency()

    def test_rollback_contradicting_not_just_set_equivalent_but_list_identical(self):
        graph, h1, h2, ev = _make_graph_with_h2()
        ev2 = _make_evidence("e2", query_signature="sig-2")
        ev3 = _make_evidence("e3", query_signature="sig-3")
        ev4 = _make_evidence("e4", query_signature="sig-4")
        graph.add_evidence(ev2)
        graph.add_evidence(ev3)
        graph.add_evidence(ev4)

        with graph.relation_transaction():
            graph.link_contradiction("H1", "e2")
            graph.link_contradiction("H1", "e3")
            graph.link_contradiction("H1", "e1")
        assert h1.contradicting_evidence_ids == ["e2", "e3", "e1"]

        with pytest.raises(RuntimeError, match="injected failure"):
            with graph.relation_transaction():
                graph.link_contradiction("H1", "e4")
                raise RuntimeError("injected failure")

        assert h1.contradicting_evidence_ids == ["e2", "e3", "e1"]
        assert set(h1.contradicting_evidence_ids) == {"e2", "e3", "e1"}
        graph.validate_consistency()
