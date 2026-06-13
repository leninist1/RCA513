"""Tests for CausalHypothesis schema: activation, state machine, evidence tracking."""

import pytest

from prismv4.prism_cht.hypothesis import CausalHypothesis, HypothesisStatus


def _valid_hypothesis(**overrides):
    defaults = {
        "hypothesis_id": "H1",
        "root_component": "payment-svc",
        "reason_family": "cpu_exhaustion",
        "onset_interval": (1000.0, 2000.0),
        "local_trigger": "CPU spike > 95% on payment-svc",
        "propagation_path": ["payment-svc", "order-svc", "gateway"],
        "explained_symptoms": ["latency increase on gateway"],
        "predicted_observations": ["payment-svc CPU > 90% at t0"],
        "falsifiers": ["No CPU anomaly on payment-svc at t0"],
    }
    defaults.update(overrides)
    return CausalHypothesis(**defaults)


class TestActivation:
    """Legal activation and mandatory falsifiers."""

    def test_valid_hypothesis_activates(self):
        h = _valid_hypothesis()
        h.activate()
        assert h.status == HypothesisStatus.ACTIVE

    def test_missing_falsifiers_fails(self):
        h = _valid_hypothesis(falsifiers=[])
        with pytest.raises(ValueError, match="falsifiers"):
            h.activate()

    def test_falsifiers_all_whitespace_fail(self):
        h = _valid_hypothesis(falsifiers=["   ", "\t", ""])
        with pytest.raises(ValueError, match="falsifiers"):
            h.activate()

    def test_onset_interval_reversed_fails(self):
        h = _valid_hypothesis(onset_interval=(2000.0, 1000.0))
        with pytest.raises(ValueError, match="must be <= end"):
            h.activate()

    def test_onset_interval_wrong_length(self):
        h = _valid_hypothesis(onset_interval=(1000.0,))
        with pytest.raises(ValueError, match="exactly 2"):
            h.activate()

    def test_missing_predicted_observations_fails(self):
        h = _valid_hypothesis(predicted_observations=[])
        with pytest.raises(ValueError, match="predicted_observations"):
            h.activate()

    def test_already_active_cannot_activate(self):
        h = _valid_hypothesis()
        h.activate()
        with pytest.raises(ValueError, match="DRAFT"):
            h.activate()

    def test_empty_hypothesis_id_fails(self):
        h = _valid_hypothesis(hypothesis_id="")
        with pytest.raises(ValueError, match="hypothesis_id"):
            h.activate()

    def test_empty_root_component_fails(self):
        h = _valid_hypothesis(root_component="")
        with pytest.raises(ValueError, match="root_component"):
            h.activate()

    def test_empty_reason_family_fails(self):
        h = _valid_hypothesis(reason_family="")
        with pytest.raises(ValueError, match="reason_family"):
            h.activate()

    def test_empty_local_trigger_fails(self):
        h = _valid_hypothesis(local_trigger="")
        with pytest.raises(ValueError, match="local_trigger"):
            h.activate()

    def test_empty_propagation_path_fails(self):
        h = _valid_hypothesis(propagation_path=[])
        with pytest.raises(ValueError, match="propagation_path"):
            h.activate()

    def test_propagation_path_with_empty_component_fails(self):
        h = _valid_hypothesis(propagation_path=["x", "", "y"])
        with pytest.raises(ValueError, match="propagation_path"):
            h.activate()


class TestStateMachine:
    """State transition rules."""

    def test_draft_to_active(self):
        h = _valid_hypothesis()
        h.activate()
        assert h.status == HypothesisStatus.ACTIVE

    def test_active_to_supported(self):
        h = _valid_hypothesis()
        h.activate()
        h.transition_to(HypothesisStatus.SUPPORTED)
        assert h.status == HypothesisStatus.SUPPORTED

    def test_active_to_weakened(self):
        h = _valid_hypothesis()
        h.activate()
        h.transition_to(HypothesisStatus.WEAKENED)
        assert h.status == HypothesisStatus.WEAKENED

    def test_active_to_refuted(self):
        h = _valid_hypothesis()
        h.activate()
        h.transition_to(HypothesisStatus.REFUTED)
        assert h.status == HypothesisStatus.REFUTED

    def test_supported_to_survived(self):
        h = _valid_hypothesis()
        h.activate()
        h.transition_to(HypothesisStatus.SUPPORTED)
        h.transition_to(HypothesisStatus.SURVIVED)
        assert h.status == HypothesisStatus.SURVIVED

    def test_survived_to_final(self):
        h = _valid_hypothesis()
        h.activate()
        h.transition_to(HypothesisStatus.SUPPORTED)
        h.transition_to(HypothesisStatus.SURVIVED)
        h.transition_to(HypothesisStatus.FINAL)
        assert h.status == HypothesisStatus.FINAL

    def test_refuted_cannot_transition(self):
        h = _valid_hypothesis()
        h.activate()
        h.transition_to(HypothesisStatus.REFUTED)
        with pytest.raises(ValueError, match="Cannot transition"):
            h.transition_to(HypothesisStatus.SUPPORTED)

    def test_final_cannot_transition(self):
        h = _valid_hypothesis()
        h.activate()
        h.transition_to(HypothesisStatus.SUPPORTED)
        h.transition_to(HypothesisStatus.SURVIVED)
        h.transition_to(HypothesisStatus.FINAL)
        with pytest.raises(ValueError, match="Cannot transition"):
            h.transition_to(HypothesisStatus.ACTIVE)

    def test_draft_cannot_jump_to_supported(self):
        h = _valid_hypothesis()
        with pytest.raises(ValueError, match="Cannot transition"):
            h.transition_to(HypothesisStatus.SUPPORTED)

    def test_active_cannot_go_to_draft(self):
        h = _valid_hypothesis()
        h.activate()
        with pytest.raises(ValueError, match="Cannot transition"):
            h.transition_to(HypothesisStatus.DRAFT)

    def test_invalid_status_type_raises(self):
        h = _valid_hypothesis()
        h.activate()
        with pytest.raises(ValueError, match="must be a HypothesisStatus"):
            h.transition_to("supported")

    def test_weakened_back_to_supported(self):
        h = _valid_hypothesis()
        h.activate()
        h.transition_to(HypothesisStatus.WEAKENED)
        h.transition_to(HypothesisStatus.SUPPORTED)
        assert h.status == HypothesisStatus.SUPPORTED


class TestEvidenceTracking:
    """Evidence attachment without duplicates."""

    def test_attach_support_no_duplicates(self):
        h = _valid_hypothesis()
        h.attach_support("e1")
        with pytest.raises(ValueError, match="Duplicate"):
            h.attach_support("e1")

    def test_attach_contradiction_no_duplicates(self):
        h = _valid_hypothesis()
        h.attach_contradiction("e1")
        with pytest.raises(ValueError, match="Duplicate"):
            h.attach_contradiction("e1")

    def test_attach_support_empty_id_raises(self):
        h = _valid_hypothesis()
        with pytest.raises(ValueError, match="evidence_id must be non-empty"):
            h.attach_support("")

    def test_attach_contradiction_empty_id_raises(self):
        h = _valid_hypothesis()
        with pytest.raises(ValueError, match="evidence_id must be non-empty"):
            h.attach_contradiction("")

    def test_attach_support_whitespace_only_raises(self):
        h = _valid_hypothesis()
        with pytest.raises(ValueError, match="evidence_id must be non-empty"):
            h.attach_support("   ")

    def test_attach_multiple_unique_supports(self):
        h = _valid_hypothesis()
        h.attach_support("e1")
        h.attach_support("e2")
        h.attach_support("e3")
        assert h.supporting_evidence_ids == ["e1", "e2", "e3"]

    def test_attach_multiple_unique_contradictions(self):
        h = _valid_hypothesis()
        h.attach_contradiction("e1")
        h.attach_contradiction("e2")
        assert h.contradicting_evidence_ids == ["e1", "e2"]
