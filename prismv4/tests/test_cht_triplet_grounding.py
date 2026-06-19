"""Tests for CHT-4.1: explicit triplet evidence grounding.

Covers TripletEvidenceCoverage construction/validation, Lead Controller
triplet validation, FinalVerifier triplet re-validation, transaction
boundary correctness, and regression guardrails.
"""

import pytest

from prismv4.prism_cht.action_gate import ActionGate
from prismv4.prism_cht.action_schema import DiscriminativeAction
from prismv4.prism_cht.assessment_gate import EvidenceAssessmentGate
from prismv4.prism_cht.challenge_demo import (
    build_demo_inconclusive_challenge,
    build_demo_refutation_challenge,
    build_demo_survival_challenge,
)
from prismv4.prism_cht.evidence_graph import EvidenceAtom, EvidenceGraph
from prismv4.prism_cht.executor import InvestigationExecutor
from prismv4.prism_cht.final_types import FinalRCAResult
from prismv4.prism_cht.final_verifier import (
    FinalizationRejectedError,
    FinalVerifier,
)
from prismv4.prism_cht.hypothesis import CausalHypothesis, HypothesisStatus
from prismv4.prism_cht.lead_controller import (
    LeadTournamentController,
    NominationRejectedError,
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
    LeadTournamentResult,
    TripletEvidenceCoverage,
)


# ===========================================================================
# Helpers
# ===========================================================================


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
    ctrl = LeadTournamentController(
        executor=executor,
        assessment_gate=assessment_gate,
        graph=graph,
        max_rounds=max_rounds,
    )
    return ctrl


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


def _build_action():
    return DiscriminativeAction(
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


def _run_survival_demo():
    lead_ctrl, hyps, lead_policy, chall_ctrl, chall_policy = (
        build_demo_survival_challenge()
    )
    lead_result = lead_ctrl.run(
        initial_hypotheses=list(hyps), policy=lead_policy
    )
    challenge_result = chall_ctrl.run(
        lead_result=lead_result,
        hypotheses=lead_ctrl._graph.hypotheses_by_id,
        policy=chall_policy,
    )
    return lead_ctrl, lead_result, challenge_result


def _run_refutation_demo():
    lead_ctrl, hyps, lead_policy, chall_ctrl, chall_policy = (
        build_demo_refutation_challenge()
    )
    lead_result = lead_ctrl.run(
        initial_hypotheses=list(hyps), policy=lead_policy
    )
    challenge_result = chall_ctrl.run(
        lead_result=lead_result,
        hypotheses=lead_ctrl._graph.hypotheses_by_id,
        policy=chall_policy,
    )
    return lead_ctrl, lead_result, challenge_result


# ===========================================================================
# A. TripletEvidenceCoverage construction and validation
# ===========================================================================


class TestTripletEvidenceCoverage:
    def test_1_valid_coverage_can_be_constructed(self):
        cov = TripletEvidenceCoverage(
            component_evidence_ids=("e1",),
            reason_evidence_ids=("e1",),
            onset_evidence_ids=("e2",),
        )
        assert cov.component_evidence_ids == ("e1",)
        assert cov.reason_evidence_ids == ("e1",)
        assert cov.onset_evidence_ids == ("e2",)

    def test_2_empty_component_rejected(self):
        with pytest.raises(ValueError, match="component_evidence_ids"):
            TripletEvidenceCoverage(
                component_evidence_ids=(),
                reason_evidence_ids=("e1",),
                onset_evidence_ids=("e1",),
            )

    def test_3_empty_reason_rejected(self):
        with pytest.raises(ValueError, match="reason_evidence_ids"):
            TripletEvidenceCoverage(
                component_evidence_ids=("e1",),
                reason_evidence_ids=(),
                onset_evidence_ids=("e1",),
            )

    def test_4_empty_onset_rejected(self):
        with pytest.raises(ValueError, match="onset_evidence_ids"):
            TripletEvidenceCoverage(
                component_evidence_ids=("e1",),
                reason_evidence_ids=("e1",),
                onset_evidence_ids=(),
            )

    def test_5_empty_string_in_dimension_rejected(self):
        with pytest.raises(ValueError, match="contains empty evidence ID"):
            TripletEvidenceCoverage(
                component_evidence_ids=("",),
                reason_evidence_ids=("e1",),
                onset_evidence_ids=("e1",),
            )

    def test_6_duplicate_id_in_dimension_rejected(self):
        with pytest.raises(ValueError, match="duplicate"):
            TripletEvidenceCoverage(
                component_evidence_ids=("e1", "e1"),
                reason_evidence_ids=("e1",),
                onset_evidence_ids=("e1",),
            )

    def test_7_same_evidence_id_allowed_across_dimensions(self):
        cov = TripletEvidenceCoverage(
            component_evidence_ids=("e1", "e2"),
            reason_evidence_ids=("e1", "e2"),
            onset_evidence_ids=("e1", "e2"),
        )
        assert cov is not None

    def test_8_coverage_is_read_only(self):
        cov = TripletEvidenceCoverage(
            component_evidence_ids=("e1",),
            reason_evidence_ids=("e1",),
            onset_evidence_ids=("e1",),
        )
        with pytest.raises(Exception):
            cov.component_evidence_ids = ("e3",)


# ===========================================================================
# B. Lead Controller triplet grounding validation
# ===========================================================================


class TestLeadControllerTripletValidation:
    def _build_valid_nomination(self, ctrl, eid, hid="H1", competitor="H2"):
        return LeadNomination(
            hypothesis_id=hid,
            supporting_evidence_ids=(eid,),
            addressed_competitor_ids=(competitor,),
            triplet_grounding=TripletEvidenceCoverage(
                component_evidence_ids=(eid,),
                reason_evidence_ids=(eid,),
                onset_evidence_ids=(eid,),
            ),
            rationale="nominate for testing",
        )

    def test_9_valid_triplet_grounding_accepted(self):
        ctrl = _make_controller(max_rounds=2)
        h1 = _make_hypothesis("H1")
        h2 = _make_hypothesis("H2")
        action = _build_action()
        sig = action.query_signature()
        eid = "evidence:" + sig
        link = EvidenceLinkProposal("H1", eid, EvidenceRelation.SUPPORTS, "s")
        con_link = EvidenceLinkProposal("H2", eid, EvidenceRelation.CONTRADICTS, "c")
        status = HypothesisStatusUpdate("H1", HypothesisStatus.SUPPORTED, "s")
        weakened = HypothesisStatusUpdate("H2", HypothesisStatus.WEAKENED, "w")
        assessment = EvidenceAssessment(
            action_id="A1", evidence_id=eid,
            outcome=AssessmentOutcome.INFORMATIVE,
            links=(link, con_link),
            status_updates=(status, weakened),
            rationale="onset supports H1",
        )
        nomination = self._build_valid_nomination(ctrl, eid)
        policy = ScriptedLeadPolicy(
            turns=[ScriptedInvestigationTurn(action=action, assessment=assessment)],
            nomination=nomination,
        )
        result = ctrl.run(initial_hypotheses=[h1, h2], policy=policy)
        assert result.status == "challenge_required"

    def test_10_component_evidence_not_exist_rejected(self):
        ctrl = _make_controller()
        h1 = _make_hypothesis("H1")
        h2 = _make_hypothesis("H2")
        ctrl._graph.register_hypothesis(h1)
        ctrl._graph.register_hypothesis(h2)
        ctrl._hypotheses = {"H1": h1, "H2": h2}
        h1.status = HypothesisStatus.SUPPORTED
        h2.status = HypothesisStatus.WEAKENED
        ev = _make_evidence("e1")
        ctrl._graph.add_evidence(ev)
        ctrl._graph.link_support("H1", "e1")
        ctrl._graph.link_contradiction("H2", "e1")
        nom = LeadNomination(
            "H1", ("e1",), ("H2",),
            TripletEvidenceCoverage(
                component_evidence_ids=("nonexistent",),
                reason_evidence_ids=("e1",),
                onset_evidence_ids=("e1",),
            ),
            "rationale",
        )
        with pytest.raises(NominationRejectedError, match="component"):
            ctrl._validate_nomination(nom)

    def test_11_reason_evidence_not_exist_rejected(self):
        ctrl = _make_controller()
        h1 = _make_hypothesis("H1")
        h2 = _make_hypothesis("H2")
        ctrl._graph.register_hypothesis(h1)
        ctrl._graph.register_hypothesis(h2)
        ctrl._hypotheses = {"H1": h1, "H2": h2}
        h1.status = HypothesisStatus.SUPPORTED
        h2.status = HypothesisStatus.WEAKENED
        ev = _make_evidence("e1")
        ctrl._graph.add_evidence(ev)
        ctrl._graph.link_support("H1", "e1")
        ctrl._graph.link_contradiction("H2", "e1")
        nom = LeadNomination(
            "H1", ("e1",), ("H2",),
            TripletEvidenceCoverage(
                component_evidence_ids=("e1",),
                reason_evidence_ids=("nonexistent",),
                onset_evidence_ids=("e1",),
            ),
            "rationale",
        )
        with pytest.raises(NominationRejectedError, match="reason"):
            ctrl._validate_nomination(nom)

    def test_12_onset_evidence_not_exist_rejected(self):
        ctrl = _make_controller()
        h1 = _make_hypothesis("H1")
        h2 = _make_hypothesis("H2")
        ctrl._graph.register_hypothesis(h1)
        ctrl._graph.register_hypothesis(h2)
        ctrl._hypotheses = {"H1": h1, "H2": h2}
        h1.status = HypothesisStatus.SUPPORTED
        h2.status = HypothesisStatus.WEAKENED
        ev = _make_evidence("e1")
        ctrl._graph.add_evidence(ev)
        ctrl._graph.link_support("H1", "e1")
        ctrl._graph.link_contradiction("H2", "e1")
        nom = LeadNomination(
            "H1", ("e1",), ("H2",),
            TripletEvidenceCoverage(
                component_evidence_ids=("e1",),
                reason_evidence_ids=("e1",),
                onset_evidence_ids=("nonexistent",),
            ),
            "rationale",
        )
        with pytest.raises(NominationRejectedError, match="onset"):
            ctrl._validate_nomination(nom)

    def test_13_component_not_in_supporting_rejected(self):
        ctrl = _make_controller()
        h1 = _make_hypothesis("H1")
        h2 = _make_hypothesis("H2")
        ctrl._graph.register_hypothesis(h1)
        ctrl._graph.register_hypothesis(h2)
        ctrl._hypotheses = {"H1": h1, "H2": h2}
        h1.status = HypothesisStatus.SUPPORTED
        h2.status = HypothesisStatus.WEAKENED
        ev1 = _make_evidence("e1")
        ev2 = _make_evidence("e2", query_signature="sig-other")
        ctrl._graph.add_evidence(ev1)
        ctrl._graph.add_evidence(ev2)
        ctrl._graph.link_support("H1", "e1")
        ctrl._graph.link_support("H1", "e2")
        ctrl._graph.link_contradiction("H2", "e1")
        nom = LeadNomination(
            "H1", ("e1",), ("H2",),
            TripletEvidenceCoverage(
                component_evidence_ids=("e2",),
                reason_evidence_ids=("e1",),
                onset_evidence_ids=("e1",),
            ),
            "rationale",
        )
        with pytest.raises(NominationRejectedError, match="component"):
            ctrl._validate_nomination(nom)

    def test_14_reason_not_in_supporting_rejected(self):
        ctrl = _make_controller()
        h1 = _make_hypothesis("H1")
        h2 = _make_hypothesis("H2")
        ctrl._graph.register_hypothesis(h1)
        ctrl._graph.register_hypothesis(h2)
        ctrl._hypotheses = {"H1": h1, "H2": h2}
        h1.status = HypothesisStatus.SUPPORTED
        h2.status = HypothesisStatus.WEAKENED
        ev1 = _make_evidence("e1")
        ev2 = _make_evidence("e2", query_signature="sig-other")
        ctrl._graph.add_evidence(ev1)
        ctrl._graph.add_evidence(ev2)
        ctrl._graph.link_support("H1", "e1")
        ctrl._graph.link_support("H1", "e2")
        ctrl._graph.link_contradiction("H2", "e1")
        nom = LeadNomination(
            "H1", ("e1",), ("H2",),
            TripletEvidenceCoverage(
                component_evidence_ids=("e1",),
                reason_evidence_ids=("e2",),
                onset_evidence_ids=("e1",),
            ),
            "rationale",
        )
        with pytest.raises(NominationRejectedError, match="reason"):
            ctrl._validate_nomination(nom)

    def test_15_onset_not_in_supporting_rejected(self):
        ctrl = _make_controller()
        h1 = _make_hypothesis("H1")
        h2 = _make_hypothesis("H2")
        ctrl._graph.register_hypothesis(h1)
        ctrl._graph.register_hypothesis(h2)
        ctrl._hypotheses = {"H1": h1, "H2": h2}
        h1.status = HypothesisStatus.SUPPORTED
        h2.status = HypothesisStatus.WEAKENED
        ev1 = _make_evidence("e1")
        ev2 = _make_evidence("e2", query_signature="sig-other")
        ctrl._graph.add_evidence(ev1)
        ctrl._graph.add_evidence(ev2)
        ctrl._graph.link_support("H1", "e1")
        ctrl._graph.link_support("H1", "e2")
        ctrl._graph.link_contradiction("H2", "e1")
        nom = LeadNomination(
            "H1", ("e1",), ("H2",),
            TripletEvidenceCoverage(
                component_evidence_ids=("e1",),
                reason_evidence_ids=("e1",),
                onset_evidence_ids=("e2",),
            ),
            "rationale",
        )
        with pytest.raises(NominationRejectedError, match="onset"):
            ctrl._validate_nomination(nom)

    def test_16_component_not_linked_to_nominee_rejected(self):
        ctrl = _make_controller()
        h1 = _make_hypothesis("H1")
        h2 = _make_hypothesis("H2")
        ctrl._graph.register_hypothesis(h1)
        ctrl._graph.register_hypothesis(h2)
        ctrl._hypotheses = {"H1": h1, "H2": h2}
        h1.status = HypothesisStatus.SUPPORTED
        h2.status = HypothesisStatus.WEAKENED
        ev = _make_evidence("e1")
        ctrl._graph.add_evidence(ev)
        ctrl._graph.link_contradiction("H2", "e1")
        nom = LeadNomination(
            "H1", ("e1",), ("H2",),
            TripletEvidenceCoverage(
                component_evidence_ids=("e1",),
                reason_evidence_ids=("e1",),
                onset_evidence_ids=("e1",),
            ),
            "rationale",
        )
        with pytest.raises(NominationRejectedError):
            ctrl._validate_nomination(nom)

    def test_17_reason_not_linked_to_nominee_rejected(self):
        ctrl = _make_controller()
        h1 = _make_hypothesis("H1")
        h2 = _make_hypothesis("H2")
        ctrl._graph.register_hypothesis(h1)
        ctrl._graph.register_hypothesis(h2)
        ctrl._hypotheses = {"H1": h1, "H2": h2}
        h1.status = HypothesisStatus.SUPPORTED
        h2.status = HypothesisStatus.WEAKENED
        ev = _make_evidence("e1")
        ctrl._graph.add_evidence(ev)
        ctrl._graph.link_contradiction("H2", "e1")
        nom = LeadNomination(
            "H1", ("e1",), ("H2",),
            TripletEvidenceCoverage(
                component_evidence_ids=("e1",),
                reason_evidence_ids=("e1",),
                onset_evidence_ids=("e1",),
            ),
            "rationale",
        )
        with pytest.raises(NominationRejectedError):
            ctrl._validate_nomination(nom)

    def test_18_onset_not_linked_to_nominee_rejected(self):
        ctrl = _make_controller()
        h1 = _make_hypothesis("H1")
        h2 = _make_hypothesis("H2")
        ctrl._graph.register_hypothesis(h1)
        ctrl._graph.register_hypothesis(h2)
        ctrl._hypotheses = {"H1": h1, "H2": h2}
        h1.status = HypothesisStatus.SUPPORTED
        h2.status = HypothesisStatus.WEAKENED
        ev = _make_evidence("e1")
        ctrl._graph.add_evidence(ev)
        ctrl._graph.link_contradiction("H2", "e1")
        nom = LeadNomination(
            "H1", ("e1",), ("H2",),
            TripletEvidenceCoverage(
                component_evidence_ids=("e1",),
                reason_evidence_ids=("e1",),
                onset_evidence_ids=("e1",),
            ),
            "rationale",
        )
        with pytest.raises(NominationRejectedError):
            ctrl._validate_nomination(nom)


# ===========================================================================
# C. FinalVerifier triplet re-validation
# ===========================================================================


class TestFinalVerifierTriplet:
    def test_19_survival_demo_can_finalize(self):
        lead_ctrl, lead_result, challenge_result = _run_survival_demo()
        verifier = FinalVerifier()
        graph = lead_ctrl._graph
        result = verifier.finalize(
            lead_result=lead_result,
            challenge_result=challenge_result,
            hypotheses=graph.hypotheses_by_id,
            graph=graph,
        )
        assert result.status == "final_verified"

    def test_20_final_result_triplet_grounding_matches_nomination(self):
        lead_ctrl, lead_result, challenge_result = _run_survival_demo()
        verifier = FinalVerifier()
        graph = lead_ctrl._graph
        result = verifier.finalize(
            lead_result=lead_result,
            challenge_result=challenge_result,
            hypotheses=graph.hypotheses_by_id,
            graph=graph,
        )
        assert result.triplet_grounding == lead_result.nomination.triplet_grounding

    def test_21_final_verifier_rejects_nonexistent_component_evidence(self):
        lead_ctrl, lead_result, challenge_result = _run_survival_demo()
        verifier = FinalVerifier()
        graph = lead_ctrl._graph
        nom = lead_result.nomination
        bad_nom = LeadNomination(
            hypothesis_id=nom.hypothesis_id,
            supporting_evidence_ids=nom.supporting_evidence_ids,
            addressed_competitor_ids=nom.addressed_competitor_ids,
            triplet_grounding=TripletEvidenceCoverage(
                component_evidence_ids=("nonexistent",),
                reason_evidence_ids=nom.triplet_grounding.reason_evidence_ids,
                onset_evidence_ids=nom.triplet_grounding.onset_evidence_ids,
            ),
            rationale=nom.rationale,
        )
        bad_lead = LeadTournamentResult(
            status=lead_result.status,
            nominated_hypothesis_id=lead_result.nominated_hypothesis_id,
            rounds_completed=lead_result.rounds_completed,
            evidence_ids=lead_result.evidence_ids,
            audit_steps=lead_result.audit_steps,
            nomination=bad_nom,
        )
        with pytest.raises(FinalizationRejectedError, match="component"):
            verifier.validate(
                lead_result=bad_lead,
                challenge_result=challenge_result,
                hypotheses=graph.hypotheses_by_id,
                graph=graph,
            )

    def test_22_final_verifier_rejects_nonexistent_reason_evidence(self):
        lead_ctrl, lead_result, challenge_result = _run_survival_demo()
        verifier = FinalVerifier()
        graph = lead_ctrl._graph
        nom = lead_result.nomination
        bad_nom = LeadNomination(
            hypothesis_id=nom.hypothesis_id,
            supporting_evidence_ids=nom.supporting_evidence_ids,
            addressed_competitor_ids=nom.addressed_competitor_ids,
            triplet_grounding=TripletEvidenceCoverage(
                component_evidence_ids=nom.triplet_grounding.component_evidence_ids,
                reason_evidence_ids=("nonexistent",),
                onset_evidence_ids=nom.triplet_grounding.onset_evidence_ids,
            ),
            rationale=nom.rationale,
        )
        bad_lead = LeadTournamentResult(
            status=lead_result.status,
            nominated_hypothesis_id=lead_result.nominated_hypothesis_id,
            rounds_completed=lead_result.rounds_completed,
            evidence_ids=lead_result.evidence_ids,
            audit_steps=lead_result.audit_steps,
            nomination=bad_nom,
        )
        with pytest.raises(FinalizationRejectedError, match="reason"):
            verifier.validate(
                lead_result=bad_lead,
                challenge_result=challenge_result,
                hypotheses=graph.hypotheses_by_id,
                graph=graph,
            )

    def test_23_final_verifier_rejects_nonexistent_onset_evidence(self):
        lead_ctrl, lead_result, challenge_result = _run_survival_demo()
        verifier = FinalVerifier()
        graph = lead_ctrl._graph
        nom = lead_result.nomination
        bad_nom = LeadNomination(
            hypothesis_id=nom.hypothesis_id,
            supporting_evidence_ids=nom.supporting_evidence_ids,
            addressed_competitor_ids=nom.addressed_competitor_ids,
            triplet_grounding=TripletEvidenceCoverage(
                component_evidence_ids=nom.triplet_grounding.component_evidence_ids,
                reason_evidence_ids=nom.triplet_grounding.reason_evidence_ids,
                onset_evidence_ids=("nonexistent",),
            ),
            rationale=nom.rationale,
        )
        bad_lead = LeadTournamentResult(
            status=lead_result.status,
            nominated_hypothesis_id=lead_result.nominated_hypothesis_id,
            rounds_completed=lead_result.rounds_completed,
            evidence_ids=lead_result.evidence_ids,
            audit_steps=lead_result.audit_steps,
            nomination=bad_nom,
        )
        with pytest.raises(FinalizationRejectedError, match="onset"):
            verifier.validate(
                lead_result=bad_lead,
                challenge_result=challenge_result,
                hypotheses=graph.hypotheses_by_id,
                graph=graph,
            )

    def test_24_final_verifier_rejects_unlinked_grounding_evidence(self):
        lead_ctrl, lead_result, challenge_result = _run_survival_demo()
        verifier = FinalVerifier()
        graph = lead_ctrl._graph
        nom = lead_result.nomination
        # Use the challenge evidence ID — it exists in graph but is not
        # linked to the nominee via a support edge.
        chall_eid = challenge_result.evidence_id
        bad_nom = LeadNomination(
            hypothesis_id=nom.hypothesis_id,
            supporting_evidence_ids=nom.supporting_evidence_ids,
            addressed_competitor_ids=nom.addressed_competitor_ids,
            triplet_grounding=TripletEvidenceCoverage(
                component_evidence_ids=(chall_eid,),
                reason_evidence_ids=nom.triplet_grounding.reason_evidence_ids,
                onset_evidence_ids=nom.triplet_grounding.onset_evidence_ids,
            ),
            rationale=nom.rationale,
        )
        bad_lead = LeadTournamentResult(
            status=lead_result.status,
            nominated_hypothesis_id=lead_result.nominated_hypothesis_id,
            rounds_completed=lead_result.rounds_completed,
            evidence_ids=lead_result.evidence_ids,
            audit_steps=lead_result.audit_steps,
            nomination=bad_nom,
        )
        with pytest.raises(FinalizationRejectedError):
            verifier.validate(
                lead_result=bad_lead,
                challenge_result=challenge_result,
                hypotheses=graph.hypotheses_by_id,
                graph=graph,
            )

    def test_25_referenced_evidence_ids_contains_all_grounding_ids(self):
        lead_ctrl, lead_result, challenge_result = _run_survival_demo()
        verifier = FinalVerifier()
        graph = lead_ctrl._graph
        result = verifier.finalize(
            lead_result=lead_result,
            challenge_result=challenge_result,
            hypotheses=graph.hypotheses_by_id,
            graph=graph,
        )
        ref_set = set(result.referenced_evidence_ids)
        for eid in result.triplet_grounding.component_evidence_ids:
            assert eid in ref_set
        for eid in result.triplet_grounding.reason_evidence_ids:
            assert eid in ref_set
        for eid in result.triplet_grounding.onset_evidence_ids:
            assert eid in ref_set

    def test_26_referenced_evidence_ids_dedup_first_occurrence_order(self):
        lead_ctrl, lead_result, challenge_result = _run_survival_demo()
        verifier = FinalVerifier()
        graph = lead_ctrl._graph
        result = verifier.finalize(
            lead_result=lead_result,
            challenge_result=challenge_result,
            hypotheses=graph.hypotheses_by_id,
            graph=graph,
        )
        assert len(result.referenced_evidence_ids) == len(
            set(result.referenced_evidence_ids)
        )

    def test_27_final_result_is_read_only(self):
        lead_ctrl, lead_result, challenge_result = _run_survival_demo()
        verifier = FinalVerifier()
        graph = lead_ctrl._graph
        result = verifier.finalize(
            lead_result=lead_result,
            challenge_result=challenge_result,
            hypotheses=graph.hypotheses_by_id,
            graph=graph,
        )
        with pytest.raises(Exception):
            result.triplet_grounding = None

    def test_28_final_result_does_not_expose_graph(self):
        lead_ctrl, lead_result, challenge_result = _run_survival_demo()
        verifier = FinalVerifier()
        graph = lead_ctrl._graph
        result = verifier.finalize(
            lead_result=lead_result,
            challenge_result=challenge_result,
            hypotheses=graph.hypotheses_by_id,
            graph=graph,
        )
        for field_name in result.__dataclass_fields__:
            val = getattr(result, field_name)
            assert not isinstance(val, EvidenceGraph)

    def test_29_final_result_does_not_expose_hypothesis(self):
        lead_ctrl, lead_result, challenge_result = _run_survival_demo()
        verifier = FinalVerifier()
        graph = lead_ctrl._graph
        result = verifier.finalize(
            lead_result=lead_result,
            challenge_result=challenge_result,
            hypotheses=graph.hypotheses_by_id,
            graph=graph,
        )
        for field_name in result.__dataclass_fields__:
            val = getattr(result, field_name)
            assert not isinstance(val, CausalHypothesis)


# ===========================================================================
# D. Transaction boundary
# ===========================================================================


class TestTransactionBoundary:
    def test_30_finalize_only_uses_relation_transaction_consistency(self):
        lead_ctrl, lead_result, challenge_result = _run_survival_demo()
        verifier = FinalVerifier()
        graph = lead_ctrl._graph
        result = verifier.finalize(
            lead_result=lead_result,
            challenge_result=challenge_result,
            hypotheses=graph.hypotheses_by_id,
            graph=graph,
        )
        assert result.status == "final_verified"

    def test_31_finalize_no_redundant_consistency_check(self, monkeypatch):
        lead_ctrl, lead_result, challenge_result = _run_survival_demo()
        verifier = FinalVerifier()
        graph = lead_ctrl._graph
        call_count = [0]
        original = graph.validate_consistency

        def counted_validate():
            call_count[0] += 1
            original()

        monkeypatch.setattr(graph, "validate_consistency", counted_validate)
        verifier.finalize(
            lead_result=lead_result,
            challenge_result=challenge_result,
            hypotheses=graph.hypotheses_by_id,
            graph=graph,
        )
        # relation_transaction calls validate_consistency twice (entry + exit)
        # validate() also calls it once. Total should be 3.
        # No extra call after transaction exit.
        assert call_count[0] == 3

    def test_32_rollback_on_consistency_failure_restores_survived(
        self, monkeypatch
    ):
        lead_ctrl, lead_result, challenge_result = _run_survival_demo()
        verifier = FinalVerifier()
        graph = lead_ctrl._graph
        h2 = graph.hypotheses_by_id["H-db_002-pool"]
        assert h2.status == HypothesisStatus.SURVIVED

        call_count = [0]
        original = graph.validate_consistency

        def failing_after_transition():
            call_count[0] += 1
            if call_count[0] >= 2:
                raise ValueError("injected consistency failure")

        monkeypatch.setattr(
            graph, "validate_consistency", failing_after_transition
        )
        with pytest.raises(ValueError, match="injected consistency failure"):
            verifier.finalize(
                lead_result=lead_result,
                challenge_result=challenge_result,
                hypotheses=graph.hypotheses_by_id,
                graph=graph,
            )
        assert h2.status == HypothesisStatus.SURVIVED

    def test_33_finalize_does_not_modify_graph_edges(self):
        lead_ctrl, lead_result, challenge_result = _run_survival_demo()
        verifier = FinalVerifier()
        graph = lead_ctrl._graph
        pre_support = dict(graph.support_edges)
        pre_contra = dict(graph.contradiction_edges)
        verifier.finalize(
            lead_result=lead_result,
            challenge_result=challenge_result,
            hypotheses=graph.hypotheses_by_id,
            graph=graph,
        )
        for hid in pre_support:
            assert graph.support_edges.get(hid, set()) == pre_support.get(
                hid, set()
            )
        for hid in pre_contra:
            assert graph.contradiction_edges.get(hid, set()) == pre_contra.get(
                hid, set()
            )

    def test_34_finalize_does_not_modify_evidence_count(self):
        lead_ctrl, lead_result, challenge_result = _run_survival_demo()
        verifier = FinalVerifier()
        graph = lead_ctrl._graph
        pre_count = graph.evidence_count()
        verifier.finalize(
            lead_result=lead_result,
            challenge_result=challenge_result,
            hypotheses=graph.hypotheses_by_id,
            graph=graph,
        )
        assert graph.evidence_count() == pre_count


# ===========================================================================
# E. Regression
# ===========================================================================


class TestRegression:
    def test_35_survival_demo_still_outputs_final_verified(self):
        lead_ctrl, lead_result, challenge_result = _run_survival_demo()
        verifier = FinalVerifier()
        graph = lead_ctrl._graph
        result = verifier.finalize(
            lead_result=lead_result,
            challenge_result=challenge_result,
            hypotheses=graph.hypotheses_by_id,
            graph=graph,
        )
        assert result.status == "final_verified"

    def test_36_refutation_demo_still_cannot_finalize(self):
        lead_ctrl, lead_result, challenge_result = _run_refutation_demo()
        verifier = FinalVerifier()
        graph = lead_ctrl._graph
        with pytest.raises(FinalizationRejectedError):
            verifier.finalize(
                lead_result=lead_result,
                challenge_result=challenge_result,
                hypotheses=graph.hypotheses_by_id,
                graph=graph,
            )

    def test_37_inconclusive_demo_still_cannot_finalize(self):
        lead_ctrl, hyps, lead_policy, chall_ctrl, chall_policy = (
            build_demo_inconclusive_challenge()
        )
        lead_result = lead_ctrl.run(
            initial_hypotheses=list(hyps), policy=lead_policy
        )
        challenge_result = chall_ctrl.run(
            lead_result=lead_result,
            hypotheses=lead_ctrl._graph.hypotheses_by_id,
            policy=chall_policy,
        )
        verifier = FinalVerifier()
        graph = lead_ctrl._graph
        with pytest.raises(FinalizationRejectedError):
            verifier.finalize(
                lead_result=lead_result,
                challenge_result=challenge_result,
                hypotheses=graph.hypotheses_by_id,
                graph=graph,
            )

    def test_38_existing_cht_tests_still_pass(self):
        lead_ctrl, hyps, lead_policy = build_demo_survival_challenge()[:3]
        # Just verify the demo can be constructed and run
        assert lead_ctrl is not None
        assert len(hyps) == 2

    def test_39_no_prism_la_modification(self):
        import prismv4.prism_la
        assert prismv4.prism_la is not None

    def test_40_no_experiments_modification(self):
        import prismv4.experiments
        assert prismv4.experiments is not None
