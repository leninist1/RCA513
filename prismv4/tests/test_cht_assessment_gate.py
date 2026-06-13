"""Tests for EvidenceAssessmentGate: validate, apply, and rejection cases."""

import pytest

from prismv4.prism_cht.action_gate import ActionGate
from prismv4.prism_cht.action_schema import DiscriminativeAction
from prismv4.prism_cht.assessment_gate import (
    AssessmentRejectedError,
    EvidenceAssessmentGate,
)
from prismv4.prism_cht.evidence_graph import EvidenceAtom, EvidenceGraph
from prismv4.prism_cht.hypothesis import CausalHypothesis, HypothesisStatus
from prismv4.prism_cht.tournament_types import (
    AssessmentOutcome,
    EvidenceAssessment,
    EvidenceLinkProposal,
    EvidenceRelation,
    HypothesisStatusUpdate,
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


# ---------------------------------------------------------------------------
# 1. Legal INFORMATIVE assessment writes support edge
# ---------------------------------------------------------------------------


class TestLegalInformativeSupport:
    def test_support_edge_written(self):
        h1 = _make_hypothesis("H1", "comp-a")
        h2 = _make_hypothesis("H2", "comp-b")
        graph = EvidenceGraph()
        graph.register_hypothesis(h1)
        graph.register_hypothesis(h2)
        evidence = _make_evidence("e1")
        graph.add_evidence(evidence)

        action = _make_action()

        link = EvidenceLinkProposal(
            hypothesis_id="H1",
            evidence_id="e1",
            relation=EvidenceRelation.SUPPORTS,
            rationale="Evidence supports H1",
        )
        status_update = HypothesisStatusUpdate(
            hypothesis_id="H1",
            new_status=HypothesisStatus.SUPPORTED,
            rationale="H1 supported by onset evidence",
        )
        assessment = EvidenceAssessment(
            action_id="A1",
            evidence_id="e1",
            outcome=AssessmentOutcome.INFORMATIVE,
            links=(link,),
            status_updates=(status_update,),
            rationale="Onset order supports H1",
        )

        gate_class = EvidenceAssessmentGate()
        gate_class.apply(
            assessment=assessment,
            action=action,
            evidence=evidence,
            hypotheses={"H1": h1, "H2": h2},
            graph=graph,
        )

        assert "e1" in graph.support_edges["H1"]
        assert h1.status == HypothesisStatus.SUPPORTED

    def test_contradiction_edge_written(self):
        h1 = _make_hypothesis("H1", "comp-a")
        h2 = _make_hypothesis("H2", "comp-b")
        graph = EvidenceGraph()
        graph.register_hypothesis(h1)
        graph.register_hypothesis(h2)
        evidence = _make_evidence("e1")
        graph.add_evidence(evidence)

        action = _make_action()

        link = EvidenceLinkProposal(
            hypothesis_id="H1",
            evidence_id="e1",
            relation=EvidenceRelation.CONTRADICTS,
            rationale="Evidence contradicts H1",
        )
        status_update = HypothesisStatusUpdate(
            hypothesis_id="H1",
            new_status=HypothesisStatus.WEAKENED,
            rationale="H1 weakened by onset evidence",
        )
        assessment = EvidenceAssessment(
            action_id="A1",
            evidence_id="e1",
            outcome=AssessmentOutcome.INFORMATIVE,
            links=(link,),
            status_updates=(status_update,),
            rationale="Onset order contradicts H1",
        )

        gate_class = EvidenceAssessmentGate()
        gate_class.apply(
            assessment=assessment,
            action=action,
            evidence=evidence,
            hypotheses={"H1": h1, "H2": h2},
            graph=graph,
        )

        assert "e1" in graph.contradiction_edges["H1"]
        assert h1.status == HypothesisStatus.WEAKENED

    def test_link_support_syncs_hypothesis(self):
        h1 = _make_hypothesis("H1", "comp-a")
        h2 = _make_hypothesis("H2", "comp-b")
        graph = EvidenceGraph()
        graph.register_hypothesis(h1)
        graph.register_hypothesis(h2)
        evidence = _make_evidence("e1")
        graph.add_evidence(evidence)

        action = _make_action()
        link = EvidenceLinkProposal(
            hypothesis_id="H1",
            evidence_id="e1",
            relation=EvidenceRelation.SUPPORTS,
            rationale="Evidence supports H1",
        )
        assessment = EvidenceAssessment(
            action_id="A1",
            evidence_id="e1",
            outcome=AssessmentOutcome.INFORMATIVE,
            links=(link,),
            status_updates=(),
            rationale="Onset order supports H1",
        )

        gate_class = EvidenceAssessmentGate()
        gate_class.apply(
            assessment=assessment,
            action=action,
            evidence=evidence,
            hypotheses={"H1": h1, "H2": h2},
            graph=graph,
        )

        assert "e1" in h1.supporting_evidence_ids

    def test_link_contradiction_syncs_hypothesis(self):
        h1 = _make_hypothesis("H1", "comp-a")
        h2 = _make_hypothesis("H2", "comp-b")
        graph = EvidenceGraph()
        graph.register_hypothesis(h1)
        graph.register_hypothesis(h2)
        evidence = _make_evidence("e1")
        graph.add_evidence(evidence)

        action = _make_action()
        link = EvidenceLinkProposal(
            hypothesis_id="H1",
            evidence_id="e1",
            relation=EvidenceRelation.CONTRADICTS,
            rationale="Evidence contradicts H1",
        )
        assessment = EvidenceAssessment(
            action_id="A1",
            evidence_id="e1",
            outcome=AssessmentOutcome.INFORMATIVE,
            links=(link,),
            status_updates=(),
            rationale="Onset order contradicts H1",
        )

        gate_class = EvidenceAssessmentGate()
        gate_class.apply(
            assessment=assessment,
            action=action,
            evidence=evidence,
            hypotheses={"H1": h1, "H2": h2},
            graph=graph,
        )

        assert "e1" in h1.contradicting_evidence_ids


# ---------------------------------------------------------------------------
# 5-6. Action ID / Evidence ID mismatch
# ---------------------------------------------------------------------------


class TestActionIdMismatch:
    def test_action_id_mismatch_rejected(self):
        h1 = _make_hypothesis("H1")
        h2 = _make_hypothesis("H2")
        graph = EvidenceGraph()
        graph.register_hypothesis(h1)
        graph.register_hypothesis(h2)
        evidence = _make_evidence("e1")
        graph.add_evidence(evidence)

        action = _make_action(action_id="A1")
        assessment = EvidenceAssessment(
            action_id="A2",  # wrong
            evidence_id="e1",
            outcome=AssessmentOutcome.INFORMATIVE,
            links=(),
            status_updates=(),
            rationale="...",
        )

        gate_class = EvidenceAssessmentGate()
        with pytest.raises(AssessmentRejectedError, match="action_id"):
            gate_class.validate(
                assessment=assessment,
                action=action,
                evidence=evidence,
                hypotheses={"H1": h1, "H2": h2},
                graph=graph,
            )

    def test_evidence_id_mismatch_rejected(self):
        h1 = _make_hypothesis("H1")
        h2 = _make_hypothesis("H2")
        graph = EvidenceGraph()
        graph.register_hypothesis(h1)
        graph.register_hypothesis(h2)
        evidence = _make_evidence("e1")
        graph.add_evidence(evidence)

        action = _make_action()
        assessment = EvidenceAssessment(
            action_id="A1",
            evidence_id="e2",  # wrong
            outcome=AssessmentOutcome.INFORMATIVE,
            links=(),
            status_updates=(),
            rationale="...",
        )

        gate_class = EvidenceAssessmentGate()
        with pytest.raises(AssessmentRejectedError, match="evidence_id"):
            gate_class.validate(
                assessment=assessment,
                action=action,
                evidence=evidence,
                hypotheses={"H1": h1, "H2": h2},
                graph=graph,
            )


# ---------------------------------------------------------------------------
# 7-8. Evidence existence checks
# ---------------------------------------------------------------------------


class TestEvidenceExistence:
    def test_nonexistent_evidence_rejected(self):
        h1 = _make_hypothesis("H1")
        h2 = _make_hypothesis("H2")
        graph = EvidenceGraph()
        graph.register_hypothesis(h1)
        graph.register_hypothesis(h2)
        evidence = _make_evidence("e1")
        # NOT added to graph

        action = _make_action()
        link = EvidenceLinkProposal(
            hypothesis_id="H1",
            evidence_id="e1",
            relation=EvidenceRelation.SUPPORTS,
            rationale="Evidence supports H1",
        )
        assessment = EvidenceAssessment(
            action_id="A1",
            evidence_id="e1",
            outcome=AssessmentOutcome.INFORMATIVE,
            links=(link,),
            status_updates=(),
            rationale="...",
        )

        gate_class = EvidenceAssessmentGate()
        with pytest.raises(AssessmentRejectedError, match="not in the graph"):
            gate_class.validate(
                assessment=assessment,
                action=action,
                evidence=evidence,
                hypotheses={"H1": h1, "H2": h2},
                graph=graph,
            )

    def test_link_not_this_rounds_evidence_rejected(self):
        h1 = _make_hypothesis("H1")
        h2 = _make_hypothesis("H2")
        graph = EvidenceGraph()
        graph.register_hypothesis(h1)
        graph.register_hypothesis(h2)
        evidence_r1 = _make_evidence("e1")
        evidence_r2 = _make_evidence("e2", query_signature="sig-other")
        graph.add_evidence(evidence_r1)
        graph.add_evidence(evidence_r2)

        action = _make_action()
        link = EvidenceLinkProposal(
            hypothesis_id="H1",
            evidence_id="e2",  # references e2, but current round is e1
            relation=EvidenceRelation.SUPPORTS,
            rationale="Evidence supports H1",
        )
        assessment = EvidenceAssessment(
            action_id="A1",
            evidence_id="e1",
            outcome=AssessmentOutcome.INFORMATIVE,
            links=(link,),
            status_updates=(),
            rationale="...",
        )

        gate_class = EvidenceAssessmentGate()
        with pytest.raises(AssessmentRejectedError, match="not the current round"):
            gate_class.validate(
                assessment=assessment,
                action=action,
                evidence=evidence_r1,
                hypotheses={"H1": h1, "H2": h2},
                graph=graph,
            )


# ---------------------------------------------------------------------------
# 9. Link targets non-action-target hypothesis
# ---------------------------------------------------------------------------


class TestLinkTargetScope:
    def test_link_outside_action_targets_rejected(self):
        h1 = _make_hypothesis("H1", "comp-a")
        h2 = _make_hypothesis("H2", "comp-b")
        h3 = _make_hypothesis("H3", "comp-c")
        graph = EvidenceGraph()
        for h in (h1, h2, h3):
            graph.register_hypothesis(h)
        evidence = _make_evidence("e1")
        graph.add_evidence(evidence)

        # Action targets only H1 and H2
        action = _make_action(target_hypothesis_ids=("H1", "H2"))
        link = EvidenceLinkProposal(
            hypothesis_id="H3",  # not in target set
            evidence_id="e1",
            relation=EvidenceRelation.SUPPORTS,
            rationale="Evidence supports H3",
        )
        assessment = EvidenceAssessment(
            action_id="A1",
            evidence_id="e1",
            outcome=AssessmentOutcome.INFORMATIVE,
            links=(link,),
            status_updates=(),
            rationale="...",
        )

        gate_class = EvidenceAssessmentGate()
        with pytest.raises(AssessmentRejectedError, match="not in action.target"):
            gate_class.validate(
                assessment=assessment,
                action=action,
                evidence=evidence,
                hypotheses={"H1": h1, "H2": h2, "H3": h3},
                graph=graph,
            )


# ---------------------------------------------------------------------------
# 10. Supports+Contradicts same hypothesis
# ---------------------------------------------------------------------------


class TestSupportsAndContradicts:
    def test_both_support_and_contradict_rejected(self):
        h1 = _make_hypothesis("H1")
        h2 = _make_hypothesis("H2")
        graph = EvidenceGraph()
        graph.register_hypothesis(h1)
        graph.register_hypothesis(h2)
        evidence = _make_evidence("e1")
        graph.add_evidence(evidence)

        action = _make_action()
        link1 = EvidenceLinkProposal(
            hypothesis_id="H1",
            evidence_id="e1",
            relation=EvidenceRelation.SUPPORTS,
            rationale="supports",
        )
        link2 = EvidenceLinkProposal(
            hypothesis_id="H1",
            evidence_id="e1",
            relation=EvidenceRelation.CONTRADICTS,
            rationale="contradicts",
        )
        assessment = EvidenceAssessment(
            action_id="A1",
            evidence_id="e1",
            outcome=AssessmentOutcome.INFORMATIVE,
            links=(link1, link2),
            status_updates=(),
            rationale="...",
        )

        gate_class = EvidenceAssessmentGate()
        with pytest.raises(AssessmentRejectedError, match="both SUPPORTS and CONTRADICTS"):
            gate_class.validate(
                assessment=assessment,
                action=action,
                evidence=evidence,
                hypotheses={"H1": h1, "H2": h2},
                graph=graph,
            )


# ---------------------------------------------------------------------------
# 11. Duplicate link (handled by EvidenceAssessment __post_init__)
# ---------------------------------------------------------------------------


class TestDuplicateLink:
    def test_duplicate_link_rejected_at_construction(self):
        link = EvidenceLinkProposal(
            hypothesis_id="H1",
            evidence_id="e1",
            relation=EvidenceRelation.SUPPORTS,
            rationale="supports",
        )
        with pytest.raises(ValueError, match="Duplicate link"):
            EvidenceAssessment(
                action_id="A1",
                evidence_id="e1",
                outcome=AssessmentOutcome.INFORMATIVE,
                links=(link, link),  # duplicate
                status_updates=(),
                rationale="...",
            )


# ---------------------------------------------------------------------------
# 12. Illegal state transition
# ---------------------------------------------------------------------------


class TestIllegalStateTransition:
    def test_illegal_transition_rejected(self):
        h1 = _make_hypothesis("H1")
        h2 = _make_hypothesis("H2")
        h1.transition_to(HypothesisStatus.SUPPORTED)
        h2.transition_to(HypothesisStatus.SUPPORTED)
        graph = EvidenceGraph()
        graph.register_hypothesis(h1)
        graph.register_hypothesis(h2)
        evidence = _make_evidence("e1")
        graph.add_evidence(evidence)

        action = _make_action()
        link = EvidenceLinkProposal(
            hypothesis_id="H1",
            evidence_id="e1",
            relation=EvidenceRelation.SUPPORTS,
            rationale="supports",
        )
        # Attempt transition from SUPPORTED -> SURVIVED (legal, but let's try illegal: DRAFT -> REFUTED)
        status_update = HypothesisStatusUpdate(
            hypothesis_id="H2",
            new_status=HypothesisStatus.DRAFT,  # cannot go from SUPPORTED to DRAFT
            rationale="back to draft",
        )
        assessment = EvidenceAssessment(
            action_id="A1",
            evidence_id="e1",
            outcome=AssessmentOutcome.INFORMATIVE,
            links=(link,),
            status_updates=(status_update,),
            rationale="...",
        )

        gate_class = EvidenceAssessmentGate()
        with pytest.raises(AssessmentRejectedError, match="Cannot transition"):
            gate_class.validate(
                assessment=assessment,
                action=action,
                evidence=evidence,
                hypotheses={"H1": h1, "H2": h2},
                graph=graph,
            )


# ---------------------------------------------------------------------------
# 13. Illegal assessment does not mutate before rejection
# ---------------------------------------------------------------------------


class TestNoPartialMutation:
    def test_rejected_assessment_does_not_mutate_graph(self):
        h1 = _make_hypothesis("H1")
        h2 = _make_hypothesis("H2")
        graph = EvidenceGraph()
        graph.register_hypothesis(h1)
        graph.register_hypothesis(h2)
        evidence = _make_evidence("e1")
        graph.add_evidence(evidence)

        action = _make_action()
        # Illegal: action_id mismatch
        assessment = EvidenceAssessment(
            action_id="WRONG",
            evidence_id="e1",
            outcome=AssessmentOutcome.INFORMATIVE,
            links=(
                EvidenceLinkProposal(
                    hypothesis_id="H1",
                    evidence_id="e1",
                    relation=EvidenceRelation.SUPPORTS,
                    rationale="supports",
                ),
            ),
            status_updates=(),
            rationale="...",
        )

        gate_class = EvidenceAssessmentGate()
        with pytest.raises(AssessmentRejectedError):
            gate_class.apply(
                assessment=assessment,
                action=action,
                evidence=evidence,
                hypotheses={"H1": h1, "H2": h2},
                graph=graph,
            )

        # Graph and hypothesis must be untouched
        assert "e1" not in graph.support_edges.get("H1", set())
        assert h1.status == HypothesisStatus.ACTIVE


# ---------------------------------------------------------------------------
# 14. INFORMATIVE with no links
# ---------------------------------------------------------------------------


class TestInformativeNoLinks:
    def test_informative_without_links_rejected(self):
        h1 = _make_hypothesis("H1")
        h2 = _make_hypothesis("H2")
        graph = EvidenceGraph()
        graph.register_hypothesis(h1)
        graph.register_hypothesis(h2)
        evidence = _make_evidence("e1")
        graph.add_evidence(evidence)

        action = _make_action()
        assessment = EvidenceAssessment(
            action_id="A1",
            evidence_id="e1",
            outcome=AssessmentOutcome.INFORMATIVE,
            links=(),  # empty
            status_updates=(),
            rationale="...",
        )

        gate_class = EvidenceAssessmentGate()
        with pytest.raises(AssessmentRejectedError, match="INFORMATIVE but links is empty"):
            gate_class.validate(
                assessment=assessment,
                action=action,
                evidence=evidence,
                hypotheses={"H1": h1, "H2": h2},
                graph=graph,
            )


# ---------------------------------------------------------------------------
# 15-16. INCONCLUSIVE with links or status updates
# ---------------------------------------------------------------------------


class TestInconclusiveConstraints:
    def test_inconclusive_with_links_rejected(self):
        h1 = _make_hypothesis("H1")
        h2 = _make_hypothesis("H2")
        graph = EvidenceGraph()
        graph.register_hypothesis(h1)
        graph.register_hypothesis(h2)
        evidence = _make_evidence("e1")
        graph.add_evidence(evidence)

        action = _make_action()
        link = EvidenceLinkProposal(
            hypothesis_id="H1",
            evidence_id="e1",
            relation=EvidenceRelation.SUPPORTS,
            rationale="supports",
        )
        assessment = EvidenceAssessment(
            action_id="A1",
            evidence_id="e1",
            outcome=AssessmentOutcome.INCONCLUSIVE,
            links=(link,),  # should be empty
            status_updates=(),
            rationale="...",
        )

        gate_class = EvidenceAssessmentGate()
        with pytest.raises(AssessmentRejectedError, match="INCONCLUSIVE but links is non-empty"):
            gate_class.validate(
                assessment=assessment,
                action=action,
                evidence=evidence,
                hypotheses={"H1": h1, "H2": h2},
                graph=graph,
            )

    def test_inconclusive_with_status_updates_rejected(self):
        h1 = _make_hypothesis("H1")
        h2 = _make_hypothesis("H2")
        graph = EvidenceGraph()
        graph.register_hypothesis(h1)
        graph.register_hypothesis(h2)
        evidence = _make_evidence("e1")
        graph.add_evidence(evidence)

        action = _make_action()
        status_update = HypothesisStatusUpdate(
            hypothesis_id="H1",
            new_status=HypothesisStatus.SUPPORTED,
            rationale="should be blocked",
        )
        assessment = EvidenceAssessment(
            action_id="A1",
            evidence_id="e1",
            outcome=AssessmentOutcome.INCONCLUSIVE,
            links=(),
            status_updates=(status_update,),  # should be empty
            rationale="...",
        )

        gate_class = EvidenceAssessmentGate()
        with pytest.raises(AssessmentRejectedError, match="INCONCLUSIVE but status_updates is non-empty"):
            gate_class.validate(
                assessment=assessment,
                action=action,
                evidence=evidence,
                hypotheses={"H1": h1, "H2": h2},
                graph=graph,
            )


# ---------------------------------------------------------------------------
# 17. Legal INCONCLUSIVE writes no edges
# ---------------------------------------------------------------------------


class TestLegalInconclusive:
    def test_legal_inconclusive_writes_no_edges(self):
        h1 = _make_hypothesis("H1")
        h2 = _make_hypothesis("H2")
        graph = EvidenceGraph()
        graph.register_hypothesis(h1)
        graph.register_hypothesis(h2)
        evidence = _make_evidence("e1")
        graph.add_evidence(evidence)

        action = _make_action()
        assessment = EvidenceAssessment(
            action_id="A1",
            evidence_id="e1",
            outcome=AssessmentOutcome.INCONCLUSIVE,
            links=(),
            status_updates=(),
            rationale="Could not determine",
        )

        gate_class = EvidenceAssessmentGate()
        gate_class.apply(
            assessment=assessment,
            action=action,
            evidence=evidence,
            hypotheses={"H1": h1, "H2": h2},
            graph=graph,
        )

        # No edges written
        assert "e1" not in graph.support_edges.get("H1", set())
        assert "e1" not in graph.contradiction_edges.get("H1", set())
        assert h1.status == HypothesisStatus.ACTIVE
        assert h2.status == HypothesisStatus.ACTIVE


# ---------------------------------------------------------------------------
# 18. validate_consistency passes after apply
# ---------------------------------------------------------------------------


class TestConsistency:
    def test_consistency_passes_after_apply(self):
        h1 = _make_hypothesis("H1")
        h2 = _make_hypothesis("H2")
        graph = EvidenceGraph()
        graph.register_hypothesis(h1)
        graph.register_hypothesis(h2)
        evidence = _make_evidence("e1")
        graph.add_evidence(evidence)

        action = _make_action()
        link = EvidenceLinkProposal(
            hypothesis_id="H1",
            evidence_id="e1",
            relation=EvidenceRelation.SUPPORTS,
            rationale="supports",
        )
        assessment = EvidenceAssessment(
            action_id="A1",
            evidence_id="e1",
            outcome=AssessmentOutcome.INFORMATIVE,
            links=(link,),
            status_updates=(),
            rationale="...",
        )

        gate_class = EvidenceAssessmentGate()
        gate_class.apply(
            assessment=assessment,
            action=action,
            evidence=evidence,
            hypotheses={"H1": h1, "H2": h2},
            graph=graph,
        )

        # Should not raise
        graph.validate_consistency()


# ---------------------------------------------------------------------------
# Additional: status_update outside action targets
# ---------------------------------------------------------------------------


class TestStatusUpdateOutsideTargets:
    def test_status_update_outside_targets_rejected(self):
        h1 = _make_hypothesis("H1")
        h2 = _make_hypothesis("H2")
        h3 = _make_hypothesis("H3")
        graph = EvidenceGraph()
        graph.register_hypothesis(h1)
        graph.register_hypothesis(h2)
        graph.register_hypothesis(h3)
        evidence = _make_evidence("e1")
        graph.add_evidence(evidence)

        action = _make_action(target_hypothesis_ids=("H1", "H2"))
        link = EvidenceLinkProposal(
            hypothesis_id="H1",
            evidence_id="e1",
            relation=EvidenceRelation.SUPPORTS,
            rationale="supports",
        )
        status_update = HypothesisStatusUpdate(
            hypothesis_id="H3",  # not in targets
            new_status=HypothesisStatus.SUPPORTED,
            rationale="should be rejected",
        )
        assessment = EvidenceAssessment(
            action_id="A1",
            evidence_id="e1",
            outcome=AssessmentOutcome.INFORMATIVE,
            links=(link,),
            status_updates=(status_update,),
            rationale="...",
        )

        gate_class = EvidenceAssessmentGate()
        with pytest.raises(AssessmentRejectedError, match="not in action.target"):
            gate_class.validate(
                assessment=assessment,
                action=action,
                evidence=evidence,
                hypotheses={"H1": h1, "H2": h2, "H3": h3},
                graph=graph,
            )
