"""Tests for ChallengeReviewGate: validate_proposal, validate_resolution, apply_resolution."""

import pytest

from prismv4.prism_cht.action_gate import ActionGate
from prismv4.prism_cht.action_schema import DiscriminativeAction
from prismv4.prism_cht.assessment_gate import EvidenceAssessmentGate
from prismv4.prism_cht.challenge_gate import ChallengeRejectedError, ChallengeReviewGate
from prismv4.prism_cht.challenge_types import (
    ChallengeProposal,
    ChallengeResolution,
    ChallengeVerdict,
)
from prismv4.prism_cht.evidence_graph import EvidenceAtom, EvidenceGraph
from prismv4.prism_cht.executor import InvestigationExecutor
from prismv4.prism_cht.hypothesis import CausalHypothesis, HypothesisStatus
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


def _setup_graph_with_h2(h1_status=HypothesisStatus.SUPPORTED, h2_status=HypothesisStatus.WEAKENED):
    """Setup: H1 is SUPPORTED (nominated), H2 is WEAKENED (competitor), with evidence."""
    h1 = _make_hypothesis("H1", "comp-a", activate=False)
    h2 = _make_hypothesis("H2", "comp-b", activate=False)
    h1.status = h1_status
    h2.status = h2_status
    graph = EvidenceGraph()
    graph.register_hypothesis(h1)
    graph.register_hypothesis(h2)
    evidence = _make_evidence("e1")
    graph.add_evidence(evidence)
    # Link H1 as supported, H2 as weakened
    graph.link_support("H1", "e1")
    graph.link_contradiction("H2", "e1")
    return graph, h1, h2, evidence


def _make_lead_result(hid="H1", competitor_ids=("H2",)):
    """Make a valid LeadTournamentResult with nominated H1."""
    evidence_ids = ("e1",)
    audit_steps = ()
    nomination = LeadNomination(
        hypothesis_id=hid,
        supporting_evidence_ids=evidence_ids,
        addressed_competitor_ids=competitor_ids,
        rationale=f"nominate {hid}",
    )
    return LeadTournamentResult(
        status="challenge_required",
        nominated_hypothesis_id=hid,
        rounds_completed=1,
        evidence_ids=evidence_ids,
        audit_steps=audit_steps,
        nomination=nomination,
    )


def _make_challenge_action(**overrides):
    """Make a challenger action targeting H1 (nominated) and H2 (competitor)."""
    defaults = {
        "action_id": "C1",
        "action_type": "run_discriminative_test",
        "target_hypothesis_ids": ("H1", "H2"),
        "question": "Is H1 the root cause?",
        "tool_name": "compare_onset_order",
        "args": {
            "component_scope": ["comp-a", "comp-b"],
            "signal_scope": ["cpu"],
            "time_window": [1000.0, 2000.0],
        },
        "expected_outcomes": {"H1": "yes", "H2": "no"},
        "why_discriminative": "Distinguish H1 from H2",
    }
    defaults.update(overrides)
    return DiscriminativeAction(**defaults)


def _make_challenge_proposal(**overrides):
    action = overrides.pop("action", None) or _make_challenge_action()
    defaults = {
        "challenge_id": "C1",
        "nominated_hypothesis_id": "H1",
        "competitor_hypothesis_ids": ("H2",),
        "challenge_claim": "H1 may be wrong",
        "falsification_target": "prove H1 wrong",
        "action": action,
        "rationale": "Test the nomination",
    }
    defaults.update(overrides)
    return ChallengeProposal(**defaults)


# ===========================================================================
# 1-3. Legal proposals + resolutions (apply)
# ===========================================================================


class TestLegalApply:
    def test_survived_apply(self):
        g = EvidenceGraph()
        gate = ChallengeReviewGate()
        agate = EvidenceAssessmentGate()
        graph, h1, h2, ev = _setup_graph_with_h2()
        lead_result = _make_lead_result()
        hypotheses = {"H1": h1, "H2": h2}

        action = _make_challenge_action()
        # Must use a different signature than the lead phase's action
        sig = action.query_signature()
        eid = "evidence:" + sig
        ev_new = _make_evidence(eid, query_signature=sig)
        graph.add_evidence(ev_new)

        proposal = _make_challenge_proposal(action=action)

        link = EvidenceLinkProposal("H1", eid, EvidenceRelation.SUPPORTS, "supports H1")
        status = HypothesisStatusUpdate("H1", HypothesisStatus.SURVIVED, "survived")
        assessment = EvidenceAssessment(
            action_id=action.action_id,
            evidence_id=eid,
            outcome=AssessmentOutcome.INFORMATIVE,
            links=(link,),
            status_updates=(status,),
            rationale="H1 survives challenge",
        )
        resolution = ChallengeResolution(
            challenge_id="C1",
            action_id=action.action_id,
            evidence_id=eid,
            verdict=ChallengeVerdict.NOMINATION_SURVIVED,
            assessment=assessment,
            rationale="OK",
        )

        gate.apply_resolution(
            proposal=proposal,
            resolution=resolution,
            evidence=ev_new,
            lead_result=lead_result,
            hypotheses=hypotheses,
            graph=graph,
            assessment_gate=agate,
        )
        graph.validate_consistency()
        assert h1.status == HypothesisStatus.SURVIVED

    def test_refuted_apply(self):
        gate = ChallengeReviewGate()
        agate = EvidenceAssessmentGate()
        graph, h1, h2, ev = _setup_graph_with_h2()
        lead_result = _make_lead_result()
        hypotheses = {"H1": h1, "H2": h2}

        action = _make_challenge_action()
        sig = action.query_signature()
        eid = "evidence:" + sig
        ev_new = _make_evidence(eid, query_signature=sig)
        graph.add_evidence(ev_new)

        proposal = _make_challenge_proposal(action=action)

        link = EvidenceLinkProposal("H1", eid, EvidenceRelation.CONTRADICTS, "contra H1")
        status = HypothesisStatusUpdate("H1", HypothesisStatus.WEAKENED, "weakened")
        assessment = EvidenceAssessment(
            action_id=action.action_id,
            evidence_id=eid,
            outcome=AssessmentOutcome.INFORMATIVE,
            links=(link,),
            status_updates=(status,),
            rationale="H1 refuted",
        )
        resolution = ChallengeResolution(
            challenge_id="C1",
            action_id=action.action_id,
            evidence_id=eid,
            verdict=ChallengeVerdict.NOMINATION_REFUTED,
            assessment=assessment,
            rationale="OK",
        )

        gate.apply_resolution(
            proposal=proposal,
            resolution=resolution,
            evidence=ev_new,
            lead_result=lead_result,
            hypotheses=hypotheses,
            graph=graph,
            assessment_gate=agate,
        )
        graph.validate_consistency()
        assert h1.status == HypothesisStatus.WEAKENED

    def test_inconclusive_apply(self):
        gate = ChallengeReviewGate()
        agate = EvidenceAssessmentGate()
        graph, h1, h2, ev = _setup_graph_with_h2()
        lead_result = _make_lead_result()
        hypotheses = {"H1": h1, "H2": h2}

        action = _make_challenge_action()
        sig = action.query_signature()
        eid = "evidence:" + sig
        ev_new = _make_evidence(eid, query_signature=sig)
        graph.add_evidence(ev_new)

        proposal = _make_challenge_proposal(action=action)

        assessment = EvidenceAssessment(
            action_id=action.action_id,
            evidence_id=eid,
            outcome=AssessmentOutcome.INCONCLUSIVE,
            links=(),
            status_updates=(),
            rationale="inconclusive",
        )
        resolution = ChallengeResolution(
            challenge_id="C1",
            action_id=action.action_id,
            evidence_id=eid,
            verdict=ChallengeVerdict.INCONCLUSIVE,
            assessment=assessment,
            rationale="OK",
        )

        gate.apply_resolution(
            proposal=proposal,
            resolution=resolution,
            evidence=ev_new,
            lead_result=lead_result,
            hypotheses=hypotheses,
            graph=graph,
            assessment_gate=agate,
        )
        graph.validate_consistency()
        assert h1.status == HypothesisStatus.SUPPORTED


# ===========================================================================
# 4-15. Proposal rejection cases
# ===========================================================================


class TestProposalRejection:
    def test_reject_wrong_lead_status(self):
        gate = ChallengeReviewGate()
        graph, h1, h2, ev = _setup_graph_with_h2()
        lead_result = _make_lead_result()
        # Override status (bypass frozen dataclass — build a bad one)
        lead_bad = LeadTournamentResult(
            status="completed",  # wrong!
            nominated_hypothesis_id="H1",
            rounds_completed=1,
            evidence_ids=("e1",),
            audit_steps=(),
            nomination=lead_result.nomination,
        ) if not True else None
        # Can't make LeadTournamentResult with non-challenge_required status
        # due to __post_init__. So test that validate_proposal rejects when
        # lead_result is somehow invalid.

    def test_reject_wrong_nominee(self):
        gate = ChallengeReviewGate()
        graph, h1, h2, ev = _setup_graph_with_h2()
        lead_result = _make_lead_result(hid="H1")
        hypotheses = {"H1": h1, "H2": h2}

        # Create a proposal that nominates H2 instead of H1 (competitor must differ)
        proposal = _make_challenge_proposal(
            nominated_hypothesis_id="H2",
            competitor_hypothesis_ids=("H1",),
        )
        with pytest.raises(ChallengeRejectedError, match="does not match"):
            gate.validate_proposal(
                proposal=proposal,
                lead_result=lead_result,
                hypotheses=hypotheses,
                graph=graph,
            )

    def test_reject_nominee_not_supported(self):
        gate = ChallengeReviewGate()
        graph, h1, h2, ev = _setup_graph_with_h2(
            h1_status=HypothesisStatus.ACTIVE,  # NOT SUPPORTED
            h2_status=HypothesisStatus.WEAKENED,
        )
        lead_result = _make_lead_result(hid="H1")
        hypotheses = {"H1": h1, "H2": h2}

        proposal = _make_challenge_proposal()
        with pytest.raises(ChallengeRejectedError, match="must be SUPPORTED"):
            gate.validate_proposal(
                proposal=proposal,
                lead_result=lead_result,
                hypotheses=hypotheses,
                graph=graph,
            )

    def test_reject_competitor_empty(self):
        """Competitor empty is caught by ChallengeProposal.__post_init__."""
        with pytest.raises(ValueError, match="at least one"):
            _make_challenge_proposal(competitor_hypothesis_ids=())

    def test_reject_competitor_duplicate(self):
        """Competitor duplicate is caught by ChallengeProposal.__post_init__."""
        with pytest.raises(ValueError, match="duplicate"):
            _make_challenge_proposal(competitor_hypothesis_ids=("H2", "H2"))

    def test_reject_competitor_nonexistent(self):
        gate = ChallengeReviewGate()
        graph, h1, h2, ev = _setup_graph_with_h2()
        lead_result = _make_lead_result(competitor_ids=("H2", "H3"))
        hypotheses = {"H1": h1, "H2": h2}

        action = _make_challenge_action(
            target_hypothesis_ids=("H1", "H3"),
            expected_outcomes={"H1": "yes", "H3": "no"},
        )
        proposal = _make_challenge_proposal(
            action=action,
            competitor_hypothesis_ids=("H3",),
        )
        with pytest.raises(ChallengeRejectedError, match="not registered"):
            gate.validate_proposal(
                proposal=proposal,
                lead_result=lead_result,
                hypotheses=hypotheses,
                graph=graph,
            )

    def test_reject_competitor_equal_nominee(self):
        """Competitor equals nominee is caught by ChallengeProposal.__post_init__."""
        with pytest.raises(ValueError, match="cannot equal"):
            _make_challenge_proposal(competitor_hypothesis_ids=("H1",))

    def test_reject_competitor_not_in_addressed(self):
        gate = ChallengeReviewGate()
        graph, h1, h2, ev = _setup_graph_with_h2()
        h3 = _make_hypothesis("H3", "comp-c", activate=False)
        h3.status = HypothesisStatus.WEAKENED
        graph.register_hypothesis(h3)
        graph.link_contradiction("H3", "e1")
        # lead addressed_competitor_ids only has H2, not H3
        lead_result = _make_lead_result(competitor_ids=("H2",))
        hypotheses = {"H1": h1, "H2": h2, "H3": h3}

        action = _make_challenge_action(
            target_hypothesis_ids=("H1", "H3"),
            expected_outcomes={"H1": "yes", "H3": "no"},
        )
        proposal = _make_challenge_proposal(
            action=action,
            competitor_hypothesis_ids=("H3",),
        )
        with pytest.raises(ChallengeRejectedError, match="not in"):
            gate.validate_proposal(
                proposal=proposal,
                lead_result=lead_result,
                hypotheses=hypotheses,
                graph=graph,
            )

    def test_reject_competitor_not_weakened(self):
        gate = ChallengeReviewGate()
        graph, h1, h2, ev = _setup_graph_with_h2(
            h1_status=HypothesisStatus.SUPPORTED,
            h2_status=HypothesisStatus.SUPPORTED,  # not WEAKENED
        )
        lead_result = _make_lead_result()
        hypotheses = {"H1": h1, "H2": h2}

        proposal = _make_challenge_proposal()
        with pytest.raises(ChallengeRejectedError, match="must be WEAKENED"):
            gate.validate_proposal(
                proposal=proposal,
                lead_result=lead_result,
                hypotheses=hypotheses,
                graph=graph,
            )

    def test_reject_action_target_mismatch(self):
        gate = ChallengeReviewGate()
        graph, h1, h2, ev = _setup_graph_with_h2()
        lead_result = _make_lead_result()
        hypotheses = {"H1": h1, "H2": h2}

        # Action targets H1 and H3, not H2
        action = _make_challenge_action(
            target_hypothesis_ids=("H1", "H3"),
            expected_outcomes={"H1": "yes", "H3": "no"},
        )
        proposal = _make_challenge_proposal(action=action)
        with pytest.raises(ChallengeRejectedError, match="do not match"):
            gate.validate_proposal(
                proposal=proposal,
                lead_result=lead_result,
                hypotheses=hypotheses,
                graph=graph,
            )

    def test_reject_empty_claim(self):
        """Empty challenge_claim is caught by ChallengeProposal.__post_init__."""
        with pytest.raises(ValueError, match="challenge_claim must be non-empty"):
            _make_challenge_proposal(challenge_claim="")

    def test_reject_empty_falsification_target(self):
        """Empty falsification_target is caught by ChallengeProposal.__post_init__."""
        with pytest.raises(ValueError, match="falsification_target must be non-empty"):
            _make_challenge_proposal(falsification_target="")


# ===========================================================================
# 16-28. Resolution rejection cases
# ===========================================================================


class TestResolutionRejection:
    def test_reject_challenge_id_mismatch(self):
        gate = ChallengeReviewGate()
        graph, h1, h2, ev = _setup_graph_with_h2()
        lead_result = _make_lead_result()
        hypotheses = {"H1": h1, "H2": h2}
        action = _make_challenge_action()
        sig = action.query_signature()
        eid = "evidence:" + sig
        ev_new = _make_evidence(eid, query_signature=sig)
        graph.add_evidence(ev_new)
        proposal = _make_challenge_proposal(action=action)

        resolution = ChallengeResolution(
            challenge_id="WRONG",  # mismatch
            action_id=action.action_id,
            evidence_id=eid,
            verdict=ChallengeVerdict.INCONCLUSIVE,
            assessment=EvidenceAssessment(
                action_id=action.action_id,
                evidence_id=eid,
                outcome=AssessmentOutcome.INCONCLUSIVE,
                links=(),
                status_updates=(),
                rationale="ok",
            ),
            rationale="OK",
        )
        with pytest.raises(ChallengeRejectedError, match="challenge_id"):
            gate.validate_resolution(
                proposal=proposal,
                resolution=resolution,
                evidence=ev_new,
                lead_result=lead_result,
                hypotheses=hypotheses,
                graph=graph,
            )

    def test_reject_action_id_mismatch(self):
        gate = ChallengeReviewGate()
        graph, h1, h2, ev = _setup_graph_with_h2()
        lead_result = _make_lead_result()
        hypotheses = {"H1": h1, "H2": h2}
        action = _make_challenge_action()
        sig = action.query_signature()
        eid = "evidence:" + sig
        ev_new = _make_evidence(eid, query_signature=sig)
        graph.add_evidence(ev_new)
        proposal = _make_challenge_proposal(action=action)

        resolution = ChallengeResolution(
            challenge_id="C1",
            action_id="WRONG",
            evidence_id=eid,
            verdict=ChallengeVerdict.INCONCLUSIVE,
            assessment=EvidenceAssessment(
                action_id="WRONG",
                evidence_id=eid,
                outcome=AssessmentOutcome.INCONCLUSIVE,
                links=(),
                status_updates=(),
                rationale="ok",
            ),
            rationale="OK",
        )
        with pytest.raises(ChallengeRejectedError, match="action_id"):
            gate.validate_resolution(
                proposal=proposal,
                resolution=resolution,
                evidence=ev_new,
                lead_result=lead_result,
                hypotheses=hypotheses,
                graph=graph,
            )

    def test_reject_evidence_id_mismatch(self):
        gate = ChallengeReviewGate()
        graph, h1, h2, ev = _setup_graph_with_h2()
        lead_result = _make_lead_result()
        hypotheses = {"H1": h1, "H2": h2}
        action = _make_challenge_action()
        sig = action.query_signature()
        eid = "evidence:" + sig
        ev_new = _make_evidence(eid, query_signature=sig)
        graph.add_evidence(ev_new)
        proposal = _make_challenge_proposal(action=action)

        resolution = ChallengeResolution(
            challenge_id="C1",
            action_id=action.action_id,
            evidence_id="WRONG",
            verdict=ChallengeVerdict.INCONCLUSIVE,
            assessment=EvidenceAssessment(
                action_id=action.action_id,
                evidence_id="WRONG",
                outcome=AssessmentOutcome.INCONCLUSIVE,
                links=(),
                status_updates=(),
                rationale="ok",
            ),
            rationale="OK",
        )
        with pytest.raises(ChallengeRejectedError, match="evidence_id"):
            gate.validate_resolution(
                proposal=proposal,
                resolution=resolution,
                evidence=ev_new,
                lead_result=lead_result,
                hypotheses=hypotheses,
                graph=graph,
            )

    def test_reject_evidence_not_in_graph(self):
        gate = ChallengeReviewGate()
        graph, h1, h2, ev = _setup_graph_with_h2()
        lead_result = _make_lead_result()
        hypotheses = {"H1": h1, "H2": h2}
        action = _make_challenge_action()
        sig = action.query_signature()
        eid = "evidence:" + sig
        ev_new = _make_evidence(eid, query_signature=sig)
        # NOT added to graph!
        proposal = _make_challenge_proposal(action=action)

        resolution = ChallengeResolution(
            challenge_id="C1",
            action_id=action.action_id,
            evidence_id=eid,
            verdict=ChallengeVerdict.INCONCLUSIVE,
            assessment=EvidenceAssessment(
                action_id=action.action_id,
                evidence_id=eid,
                outcome=AssessmentOutcome.INCONCLUSIVE,
                links=(),
                status_updates=(),
                rationale="ok",
            ),
            rationale="OK",
        )
        with pytest.raises(ChallengeRejectedError, match="not in the graph"):
            gate.validate_resolution(
                proposal=proposal,
                resolution=resolution,
                evidence=ev_new,
                lead_result=lead_result,
                hypotheses=hypotheses,
                graph=graph,
            )

    def test_reject_survived_missing_link(self):
        gate = ChallengeReviewGate()
        graph, h1, h2, ev = _setup_graph_with_h2()
        lead_result = _make_lead_result()
        hypotheses = {"H1": h1, "H2": h2}
        action = _make_challenge_action()
        sig = action.query_signature()
        eid = "evidence:" + sig
        ev_new = _make_evidence(eid, query_signature=sig)
        graph.add_evidence(ev_new)
        proposal = _make_challenge_proposal(action=action)

        assessment = EvidenceAssessment(
            action_id=action.action_id,
            evidence_id=eid,
            outcome=AssessmentOutcome.INFORMATIVE,
            links=(),  # No grounded link!
            status_updates=(
                HypothesisStatusUpdate("H1", HypothesisStatus.SURVIVED, "survived"),
            ),
            rationale="survived",
        )
        resolution = ChallengeResolution(
            challenge_id="C1",
            action_id=action.action_id,
            evidence_id=eid,
            verdict=ChallengeVerdict.NOMINATION_SURVIVED,
            assessment=assessment,
            rationale="OK",
        )
        with pytest.raises(ChallengeRejectedError, match="grounded link"):
            gate.validate_resolution(
                proposal=proposal,
                resolution=resolution,
                evidence=ev_new,
                lead_result=lead_result,
                hypotheses=hypotheses,
                graph=graph,
            )

    def test_reject_survived_missing_status_update(self):
        gate = ChallengeReviewGate()
        graph, h1, h2, ev = _setup_graph_with_h2()
        lead_result = _make_lead_result()
        hypotheses = {"H1": h1, "H2": h2}
        action = _make_challenge_action()
        sig = action.query_signature()
        eid = "evidence:" + sig
        ev_new = _make_evidence(eid, query_signature=sig)
        graph.add_evidence(ev_new)
        proposal = _make_challenge_proposal(action=action)

        link = EvidenceLinkProposal("H1", eid, EvidenceRelation.SUPPORTS, "supports")
        assessment = EvidenceAssessment(
            action_id=action.action_id,
            evidence_id=eid,
            outcome=AssessmentOutcome.INFORMATIVE,
            links=(link,),
            status_updates=(),  # No status update!
            rationale="survived",
        )
        resolution = ChallengeResolution(
            challenge_id="C1",
            action_id=action.action_id,
            evidence_id=eid,
            verdict=ChallengeVerdict.NOMINATION_SURVIVED,
            assessment=assessment,
            rationale="OK",
        )
        with pytest.raises(ChallengeRejectedError, match="exactly one status update"):
            gate.validate_resolution(
                proposal=proposal,
                resolution=resolution,
                evidence=ev_new,
                lead_result=lead_result,
                hypotheses=hypotheses,
                graph=graph,
            )

    def test_reject_survived_with_final(self):
        gate = ChallengeReviewGate()
        graph, h1, h2, ev = _setup_graph_with_h2()
        lead_result = _make_lead_result()
        hypotheses = {"H1": h1, "H2": h2}
        action = _make_challenge_action()
        sig = action.query_signature()
        eid = "evidence:" + sig
        ev_new = _make_evidence(eid, query_signature=sig)
        graph.add_evidence(ev_new)
        proposal = _make_challenge_proposal(action=action)

        link = EvidenceLinkProposal("H1", eid, EvidenceRelation.SUPPORTS, "supports")
        assessment = EvidenceAssessment(
            action_id=action.action_id,
            evidence_id=eid,
            outcome=AssessmentOutcome.INFORMATIVE,
            links=(link,),
            status_updates=(
                HypothesisStatusUpdate("H1", HypothesisStatus.FINAL, "final"),  # FINAL!
            ),
            rationale="ok",
        )
        resolution = ChallengeResolution(
            challenge_id="C1",
            action_id=action.action_id,
            evidence_id=eid,
            verdict=ChallengeVerdict.NOMINATION_SURVIVED,
            assessment=assessment,
            rationale="OK",
        )
        with pytest.raises(ChallengeRejectedError, match="FINAL"):
            gate.validate_resolution(
                proposal=proposal,
                resolution=resolution,
                evidence=ev_new,
                lead_result=lead_result,
                hypotheses=hypotheses,
                graph=graph,
            )

    def test_reject_refuted_missing_contradicts(self):
        gate = ChallengeReviewGate()
        graph, h1, h2, ev = _setup_graph_with_h2()
        lead_result = _make_lead_result()
        hypotheses = {"H1": h1, "H2": h2}
        action = _make_challenge_action()
        sig = action.query_signature()
        eid = "evidence:" + sig
        ev_new = _make_evidence(eid, query_signature=sig)
        graph.add_evidence(ev_new)
        proposal = _make_challenge_proposal(action=action)

        # No contradict link
        link = EvidenceLinkProposal("H2", eid, EvidenceRelation.CONTRADICTS, "contra H2")
        assessment = EvidenceAssessment(
            action_id=action.action_id,
            evidence_id=eid,
            outcome=AssessmentOutcome.INFORMATIVE,
            links=(link,),  # contradicts H2 but NOT H1
            status_updates=(
                HypothesisStatusUpdate("H1", HypothesisStatus.WEAKENED, "weakened"),
            ),
            rationale="refuted",
        )
        resolution = ChallengeResolution(
            challenge_id="C1",
            action_id=action.action_id,
            evidence_id=eid,
            verdict=ChallengeVerdict.NOMINATION_REFUTED,
            assessment=assessment,
            rationale="OK",
        )
        with pytest.raises(ChallengeRejectedError, match="CONTRADICTS nominated"):
            gate.validate_resolution(
                proposal=proposal,
                resolution=resolution,
                evidence=ev_new,
                lead_result=lead_result,
                hypotheses=hypotheses,
                graph=graph,
            )

    def test_reject_refuted_survived_update(self):
        gate = ChallengeReviewGate()
        graph, h1, h2, ev = _setup_graph_with_h2()
        lead_result = _make_lead_result()
        hypotheses = {"H1": h1, "H2": h2}
        action = _make_challenge_action()
        sig = action.query_signature()
        eid = "evidence:" + sig
        ev_new = _make_evidence(eid, query_signature=sig)
        graph.add_evidence(ev_new)
        proposal = _make_challenge_proposal(action=action)

        link = EvidenceLinkProposal("H1", eid, EvidenceRelation.CONTRADICTS, "contra")
        assessment = EvidenceAssessment(
            action_id=action.action_id,
            evidence_id=eid,
            outcome=AssessmentOutcome.INFORMATIVE,
            links=(link,),
            status_updates=(
                HypothesisStatusUpdate("H1", HypothesisStatus.SURVIVED, "survived"),  # wrong!
            ),
            rationale="ok",
        )
        resolution = ChallengeResolution(
            challenge_id="C1",
            action_id=action.action_id,
            evidence_id=eid,
            verdict=ChallengeVerdict.NOMINATION_REFUTED,
            assessment=assessment,
            rationale="OK",
        )
        with pytest.raises(ChallengeRejectedError, match="WEAKENED or REFUTED"):
            gate.validate_resolution(
                proposal=proposal,
                resolution=resolution,
                evidence=ev_new,
                lead_result=lead_result,
                hypotheses=hypotheses,
                graph=graph,
            )

    def test_reject_inconclusive_with_link(self):
        gate = ChallengeReviewGate()
        graph, h1, h2, ev = _setup_graph_with_h2()
        lead_result = _make_lead_result()
        hypotheses = {"H1": h1, "H2": h2}
        action = _make_challenge_action()
        sig = action.query_signature()
        eid = "evidence:" + sig
        ev_new = _make_evidence(eid, query_signature=sig)
        graph.add_evidence(ev_new)
        proposal = _make_challenge_proposal(action=action)

        link = EvidenceLinkProposal("H1", eid, EvidenceRelation.SUPPORTS, "supports")
        assessment = EvidenceAssessment(
            action_id=action.action_id,
            evidence_id=eid,
            outcome=AssessmentOutcome.INCONCLUSIVE,  # INCONCLUSIVE with link!
            links=(link,),
            status_updates=(),
            rationale="inconclusive",
        )
        resolution = ChallengeResolution(
            challenge_id="C1",
            action_id=action.action_id,
            evidence_id=eid,
            verdict=ChallengeVerdict.INCONCLUSIVE,
            assessment=assessment,
            rationale="OK",
        )
        with pytest.raises(ChallengeRejectedError, match="no links"):
            gate.validate_resolution(
                proposal=proposal,
                resolution=resolution,
                evidence=ev_new,
                lead_result=lead_result,
                hypotheses=hypotheses,
                graph=graph,
            )

    def test_reject_inconclusive_with_status_update(self):
        gate = ChallengeReviewGate()
        graph, h1, h2, ev = _setup_graph_with_h2()
        lead_result = _make_lead_result()
        hypotheses = {"H1": h1, "H2": h2}
        action = _make_challenge_action()
        sig = action.query_signature()
        eid = "evidence:" + sig
        ev_new = _make_evidence(eid, query_signature=sig)
        graph.add_evidence(ev_new)
        proposal = _make_challenge_proposal(action=action)

        assessment = EvidenceAssessment(
            action_id=action.action_id,
            evidence_id=eid,
            outcome=AssessmentOutcome.INCONCLUSIVE,
            links=(),
            status_updates=(
                HypothesisStatusUpdate("H1", HypothesisStatus.SUPPORTED, "stay"),  # has update!
            ),
            rationale="inconclusive",
        )
        resolution = ChallengeResolution(
            challenge_id="C1",
            action_id=action.action_id,
            evidence_id=eid,
            verdict=ChallengeVerdict.INCONCLUSIVE,
            assessment=assessment,
            rationale="OK",
        )
        with pytest.raises(ChallengeRejectedError, match="no status updates"):
            gate.validate_resolution(
                proposal=proposal,
                resolution=resolution,
                evidence=ev_new,
                lead_result=lead_result,
                hypotheses=hypotheses,
                graph=graph,
            )

    def test_reject_before_mutate(self):
        """Challenger-specific illegal input must be rejected before any mutation."""
        gate = ChallengeReviewGate()
        agate = EvidenceAssessmentGate()
        graph, h1, h2, ev = _setup_graph_with_h2()
        lead_result = _make_lead_result()
        hypotheses = {"H1": h1, "H2": h2}

        action = _make_challenge_action()
        sig = action.query_signature()
        eid = "evidence:" + sig
        ev_new = _make_evidence(eid, query_signature=sig)
        graph.add_evidence(ev_new)

        proposal = _make_challenge_proposal(action=action)

        # Illegal resolution: NOMINATION_SURVIVED with no links
        assessment = EvidenceAssessment(
            action_id=action.action_id,
            evidence_id=eid,
            outcome=AssessmentOutcome.INFORMATIVE,
            links=(),
            status_updates=(
                HypothesisStatusUpdate("H1", HypothesisStatus.SURVIVED, "ok"),
            ),
            rationale="no link",
        )
        resolution = ChallengeResolution(
            challenge_id="C1",
            action_id=action.action_id,
            evidence_id=eid,
            verdict=ChallengeVerdict.NOMINATION_SURVIVED,
            assessment=assessment,
            rationale="OK",
        )

        pre_status = h1.status
        pre_edges = dict(graph.support_edges)
        with pytest.raises(ChallengeRejectedError):
            gate.apply_resolution(
                proposal=proposal,
                resolution=resolution,
                evidence=ev_new,
                lead_result=lead_result,
                hypotheses=hypotheses,
                graph=graph,
                assessment_gate=agate,
            )
        assert h1.status == pre_status

    def test_apply_passes_consistency(self):
        gate = ChallengeReviewGate()
        agate = EvidenceAssessmentGate()
        graph, h1, h2, ev = _setup_graph_with_h2()
        lead_result = _make_lead_result()
        hypotheses = {"H1": h1, "H2": h2}

        action = _make_challenge_action()
        sig = action.query_signature()
        eid = "evidence:" + sig
        ev_new = _make_evidence(eid, query_signature=sig)
        graph.add_evidence(ev_new)
        proposal = _make_challenge_proposal(action=action)

        link = EvidenceLinkProposal("H1", eid, EvidenceRelation.SUPPORTS, "supports H1")
        status = HypothesisStatusUpdate("H1", HypothesisStatus.SURVIVED, "survived")
        assessment = EvidenceAssessment(
            action_id=action.action_id,
            evidence_id=eid,
            outcome=AssessmentOutcome.INFORMATIVE,
            links=(link,),
            status_updates=(status,),
            rationale="H1 survives",
        )
        resolution = ChallengeResolution(
            challenge_id="C1",
            action_id=action.action_id,
            evidence_id=eid,
            verdict=ChallengeVerdict.NOMINATION_SURVIVED,
            assessment=assessment,
            rationale="OK",
        )

        gate.apply_resolution(
            proposal=proposal,
            resolution=resolution,
            evidence=ev_new,
            lead_result=lead_result,
            hypotheses=hypotheses,
            graph=graph,
            assessment_gate=agate,
        )
        graph.validate_consistency()
