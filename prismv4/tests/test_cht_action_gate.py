"""Tests for ActionGate: reject invalid or duplicate actions."""

import pytest

from prismv4.prism_cht.action_gate import ActionGate
from prismv4.prism_cht.action_schema import DiscriminativeAction
from prismv4.prism_cht.hypothesis import CausalHypothesis, HypothesisStatus


def _valid_action(**overrides):
    defaults = {
        "action_id": "A1",
        "action_type": "run_discriminative_test",
        "target_hypothesis_ids": ("H1", "H2"),
        "question": "Which component fails first?",
        "tool_name": "compare_onset_order",
        "args": {"component_a": "payment-svc", "component_b": "order-svc"},
        "expected_outcomes": {
            "H1": "payment-svc spikes before order-svc",
            "H2": "order-svc spikes before payment-svc",
        },
        "why_discriminative": "These two hypotheses predict opposite onset order",
    }
    defaults.update(overrides)
    return DiscriminativeAction(**defaults)


def _make_active_hypothesis(hid="H1", component="payment-svc"):
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
    h.activate()
    return h


class TestGateAccept:
    """Valid actions accepted."""

    def test_legal_action_accepted(self):
        gate = ActionGate()
        h1 = _make_active_hypothesis("H1")
        h2 = _make_active_hypothesis("H2")
        action = _valid_action()
        d = gate.evaluate(action, {"H1": h1, "H2": h2})
        assert d.accepted is True

    def test_legal_action_admitted(self):
        gate = ActionGate()
        h1 = _make_active_hypothesis("H1")
        h2 = _make_active_hypothesis("H2")
        action = _valid_action()
        d = gate.admit(action, {"H1": h1, "H2": h2})
        assert d.accepted is True


class TestGateReject:
    """Invalid actions rejected by the gate."""

    def _hmap(self):
        return {
            "H1": _make_active_hypothesis("H1", "payment-svc"),
            "H2": _make_active_hypothesis("H2", "order-svc"),
        }

    def test_identical_outcomes_rejected(self):
        gate = ActionGate()
        action = _valid_action(
            expected_outcomes={
                "H1": "Same outcome description",
                "H2": "Same outcome description",
            }
        )
        d = gate.evaluate(action, self._hmap())
        assert d.accepted is False
        assert any("identical" in r for r in d.reasons)

    def test_identical_outcomes_after_normalization_rejected(self):
        gate = ActionGate()
        action = _valid_action(
            expected_outcomes={
                "H1": "  Same   OUTCOME  description  ",
                "H2": "same outcome description",
            }
        )
        d = gate.evaluate(action, self._hmap())
        assert d.accepted is False
        assert any("identical" in r for r in d.reasons)

    def test_nonexistent_hypothesis_rejected(self):
        gate = ActionGate()
        action = _valid_action(
            target_hypothesis_ids=("H1", "H99"),
            expected_outcomes={
                "H1": "payment-svc fails first",
                "H99": "H99 fails first",
            },
        )
        d = gate.evaluate(action, {"H1": _make_active_hypothesis("H1")})
        assert d.accepted is False
        assert any("does not exist" in r for r in d.reasons)

    def test_refuted_hypothesis_rejected(self):
        gate = ActionGate()
        h1 = _make_active_hypothesis("H1")
        h1.transition_to(HypothesisStatus.REFUTED)
        h2 = _make_active_hypothesis("H2")
        action = _valid_action()
        d = gate.evaluate(action, {"H1": h1, "H2": h2})
        assert d.accepted is False
        assert any("refuted" in r.lower() for r in d.reasons)

    def test_final_hypothesis_rejected(self):
        gate = ActionGate()
        h1 = _make_active_hypothesis("H1")
        h1.transition_to(HypothesisStatus.SUPPORTED)
        h1.transition_to(HypothesisStatus.SURVIVED)
        h1.transition_to(HypothesisStatus.FINAL)
        h2 = _make_active_hypothesis("H2")
        action = _valid_action()
        d = gate.evaluate(action, {"H1": h1, "H2": h2})
        assert d.accepted is False
        assert any("final" in r.lower() for r in d.reasons)

    def test_unknown_tool_rejected(self):
        gate = ActionGate()
        action = _valid_action(tool_name="guess_root_cause")
        d = gate.evaluate(action, self._hmap())
        assert d.accepted is False
        assert any("allowlist" in r for r in d.reasons)


class TestActionConstructionRejection:
    """Invariants caught at DiscriminativeAction construction time."""

    def test_single_target_rejected_at_construction(self):
        with pytest.raises(ValueError, match="at least two"):
            _valid_action(target_hypothesis_ids=("H1",))

    def test_missing_outcome_coverage_rejected_at_construction(self):
        with pytest.raises(ValueError, match="expected_outcomes must cover"):
            _valid_action(expected_outcomes={"H1": "does X first"})

    def test_empty_question_rejected_at_construction(self):
        with pytest.raises(ValueError, match="question must be non-empty"):
            _valid_action(question="")

    def test_empty_why_discriminative_rejected_at_construction(self):
        with pytest.raises(ValueError, match="why_discriminative must be non-empty"):
            _valid_action(why_discriminative="")

    def test_duplicate_target_ids_rejected_at_construction(self):
        with pytest.raises(ValueError, match="duplicate"):
            _valid_action(
                target_hypothesis_ids=("H1", "H2", "H1"),
                expected_outcomes={
                    "H1": "outcome A",
                    "H2": "outcome B",
                },
            )

    def test_empty_expected_outcome_value_rejected_at_construction(self):
        with pytest.raises(ValueError, match="must be non-empty"):
            _valid_action(
                expected_outcomes={"H1": "outcome A", "H2": "   "},
            )


class TestSignatureDedup:
    """Query signature deduplication in ActionGate."""

    def test_same_signature_twice_rejected(self):
        gate = ActionGate()
        hm = {
            "H1": _make_active_hypothesis("H1", "payment-svc"),
            "H2": _make_active_hypothesis("H2", "order-svc"),
        }
        action = _valid_action()
        d1 = gate.admit(action, hm)
        assert d1.accepted is True
        d2 = gate.admit(action, hm)
        assert d2.accepted is False
        assert any("already been executed" in r for r in d2.reasons)

    def test_question_wording_does_not_affect_signature(self):
        hm = {
            "H1": _make_active_hypothesis("H1", "payment-svc"),
            "H2": _make_active_hypothesis("H2", "order-svc"),
        }
        a1 = _valid_action(
            question="Which one fails first?",
            action_id="A1",
        )
        a2 = _valid_action(
            question="Determine temporal order of failures",
            action_id="A2",
        )
        assert a1.query_signature() == a2.query_signature()

    def test_why_discriminative_does_not_affect_signature(self):
        a1 = _valid_action(why_discriminative="reason v1")
        a2 = _valid_action(why_discriminative="reason v2")
        assert a1.query_signature() == a2.query_signature()

    def test_evaluate_does_not_record_signature(self):
        gate = ActionGate()
        hm = {
            "H1": _make_active_hypothesis("H1", "payment-svc"),
            "H2": _make_active_hypothesis("H2", "order-svc"),
        }
        action = _valid_action()
        d = gate.evaluate(action, hm)
        assert d.accepted is True
        # admit should still work (signature not recorded)
        d2 = gate.admit(action, hm)
        assert d2.accepted is True

    def test_admit_records_signature(self):
        gate = ActionGate()
        hm = {
            "H1": _make_active_hypothesis("H1", "payment-svc"),
            "H2": _make_active_hypothesis("H2", "order-svc"),
        }
        action = _valid_action()
        d1 = gate.admit(action, hm)
        assert d1.accepted is True
        d2 = gate.admit(action, hm)
        assert d2.accepted is False


class TestGateNoScoring:
    """ActionGate must not perform root-cause scoring."""

    def test_no_scoring_attrs(self):
        gate = ActionGate()
        assert not hasattr(gate, "compute_score")
        assert not hasattr(gate, "root_score")
        assert not hasattr(gate, "posterior")
        assert not hasattr(gate, "rank_hypotheses")
        assert not hasattr(gate, "pick_winner")
