"""Tests for Phase CHT-0.1 invariant hardening.

Covers: deep immutability, unified signatures, EvidenceGraph consistency,
conflict detection, and ActionGate DRAFT rejection.
"""

import pytest

from prismv4.prism_cht.action_gate import ActionGate
from prismv4.prism_cht.action_schema import DiscriminativeAction
from prismv4.prism_cht.canonical import (
    build_tool_call_signature,
    canonicalize_json_value,
    deep_freeze,
)
from prismv4.prism_cht.evidence_graph import (
    EvidenceAtom,
    EvidenceGraph,
    build_query_signature,
)
from prismv4.prism_cht.hypothesis import CausalHypothesis, HypothesisStatus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_atom(evidence_id="e1", **overrides):
    defaults = {
        "evidence_id": evidence_id,
        "query_signature": "sig-abc",
        "modality": "metric",
        "component_scope": ("payment-svc",),
        "time_window": (1000.0, 2000.0),
        "observation": {"cpu_pct": 95.0},
        "provenance": {"source": "prometheus"},
    }
    defaults.update(overrides)
    return EvidenceAtom(**defaults)


def _make_hypothesis(hid="H1", component="payment-svc", activate=True):
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


# ===========================================================================
# A. Deep Immutability
# ===========================================================================

class TestDeepImmutabilityEvidenceAtom:
    """EvidenceAtom nested containers cannot be mutated externally or internally."""

    def test_external_observation_dict_mutation_no_effect(self):
        obs = {"cpu_pct": 95.0, "nested": {"a": 1}}
        prov = {"source": "prometheus"}
        atom = _make_atom(observation=obs, provenance=prov)

        # Mutate the original dict
        obs["cpu_pct"] = 0.0
        obs["new_key"] = "injected"
        obs["nested"]["a"] = 999

        assert atom.observation["cpu_pct"] == 95.0
        assert "new_key" not in atom.observation
        assert atom.observation["nested"]["a"] == 1

    def test_external_provenance_dict_mutation_no_effect(self):
        prov = {"source": "prometheus", "extra": {"b": 2}}
        atom = _make_atom(provenance=prov, query_signature="sig-deep-prov")

        prov["source"] = "hacked"
        prov["extra"]["b"] = 888

        assert atom.provenance["source"] == "prometheus"
        assert atom.provenance["extra"]["b"] == 2

    def test_direct_observation_write_fails(self):
        atom = _make_atom(observation={"x": 1})
        with pytest.raises(TypeError):
            atom.observation["x"] = 2

    def test_direct_provenance_write_fails(self):
        atom = _make_atom(provenance={"y": 1})
        with pytest.raises(TypeError):
            atom.provenance["y"] = 2

    def test_nested_list_in_observation_not_mutable(self):
        atom = _make_atom(observation={"items": [1, 2, 3]})
        assert isinstance(atom.observation["items"], tuple)

    def test_nested_dict_in_observation_not_mutable(self):
        atom = _make_atom(observation={"nested": {"k": "v"}})
        with pytest.raises(TypeError):
            atom.observation["nested"]["k"] = "changed"

    def test_deeply_nested_mutation_impossible(self):
        deep = {"level1": {"level2": {"level3": [1, {"deep_key": "deep_val"}]}}}
        atom = _make_atom(observation=deep, query_signature="sig-deep")
        with pytest.raises(TypeError):
            atom.observation["level1"]["level2"]["level3"][1]["deep_key"] = "hacked"


class TestDeepImmutabilityAction:
    """DiscriminativeAction args and expected_outcomes cannot be mutated."""

    def test_external_args_mutation_no_effect(self):
        raw_args = {"x": 1, "nested": {"y": 2}}
        action = _make_action(args=raw_args)
        raw_args["x"] = 999
        raw_args["nested"]["y"] = 888
        assert action.args["x"] == 1
        assert action.args["nested"]["y"] == 2

    def test_external_args_mutation_signature_unchanged(self):
        raw_args = {"x": 1, "y": 2}
        action = _make_action(args=raw_args)
        sig_before = action.query_signature()
        raw_args["x"] = 999
        assert action.query_signature() == sig_before

    def test_direct_args_write_fails(self):
        action = _make_action(args={"k": "v"})
        with pytest.raises(TypeError):
            action.args["k"] = "changed"

    def test_external_expected_outcomes_mutation_no_effect(self):
        outcomes = {"H1": "a", "H2": "b"}
        action = _make_action(expected_outcomes=outcomes)
        outcomes["H1"] = "injected"
        assert action.expected_outcomes["H1"] == "a"


# ===========================================================================
# B. Canonicalization
# ===========================================================================

class TestToolCallSignature:
    """build_tool_call_signature determinism and error handling."""

    def test_dict_order_independence(self):
        sig1 = build_tool_call_signature(
            tool_name="test",
            args={"a": 1, "b": 2},
        )
        sig2 = build_tool_call_signature(
            tool_name="test",
            args={"b": 2, "a": 1},
        )
        assert sig1 == sig2

    def test_nested_dict_order_independence(self):
        sig1 = build_tool_call_signature(
            tool_name="test",
            args={"outer": {"inner_a": 1, "inner_b": 2}},
        )
        sig2 = build_tool_call_signature(
            tool_name="test",
            args={"outer": {"inner_b": 2, "inner_a": 1}},
        )
        assert sig1 == sig2

    def test_set_order_independence(self):
        sig1 = build_tool_call_signature(
            tool_name="test",
            args={"items": frozenset({3, 1, 2})},
        )
        sig2 = build_tool_call_signature(
            tool_name="test",
            args={"items": frozenset({1, 3, 2})},
        )
        assert sig1 == sig2

    def test_unsupported_type_raises(self):
        class CustomObj:
            pass

        with pytest.raises(TypeError, match="Cannot canonicalize"):
            build_tool_call_signature(
                tool_name="test",
                args={"bad": CustomObj()},
            )

    def test_build_query_signature_delegates_to_unified_signature(self):
        sig1 = build_query_signature(
            tool_name="test_tool",
            component_scope=["A", "B", "A"],  # duplicates removed
            signal_scope=["x"],
            time_window=(0.0, 10.0),
            parameters={"depth": "1"},
        )
        sig2 = build_tool_call_signature(
            tool_name="test_tool",
            args={
                "component_scope": ["A", "B"],
                "signal_scope": ["x"],
                "time_window": (0.0, 10.0),
                "parameters": {"depth": "1"},
            },
        )
        assert sig1 == sig2

    def test_signature_is_64_char_hex(self):
        sig = build_tool_call_signature(
            tool_name="test",
            args={"x": 1},
        )
        assert len(sig) == 64
        assert all(c in "0123456789abcdef" for c in sig)

    def test_empty_tool_name_raises(self):
        with pytest.raises(ValueError, match="tool_name"):
            build_tool_call_signature(tool_name="", args={})

    def test_args_not_mapping_raises(self):
        with pytest.raises(ValueError, match="Mapping"):
            build_tool_call_signature(tool_name="test", args="not_a_map")

    def test_different_data_types(self):
        """Ensure bool, None, int, float, str are all properly handled."""
        sig1 = build_tool_call_signature(
            tool_name="test",
            args={
                "flag": True,
                "null_val": None,
                "int_val": 42,
                "float_val": 3.14,
                "str_val": "hello",
            },
        )
        sig2 = build_tool_call_signature(
            tool_name="test",
            args={
                "str_val": "hello",
                "float_val": 3.14,
                "null_val": None,
                "int_val": 42,
                "flag": True,
            },
        )
        assert sig1 == sig2


class TestActionQuerySignature:
    """DiscriminativeAction.query_signature uses unified implementation."""

    def test_question_change_same_signature(self):
        a1 = _make_action(question="Which one?")
        a2 = _make_action(question="Determine temporal order")
        assert a1.query_signature() == a2.query_signature()

    def test_args_order_independent(self):
        a1 = _make_action(args={"a": 1, "b": 2})
        a2 = _make_action(args={"b": 2, "a": 1})
        assert a1.query_signature() == a2.query_signature()


# ===========================================================================
# C. EvidenceGraph Consistency
# ===========================================================================

class TestGraphHypothesisSync:
    """link_support / link_contradiction sync graph edges to hypothesis."""

    def test_link_support_syncs_to_hypothesis(self):
        g = EvidenceGraph()
        h = _make_hypothesis("H1")
        g.register_hypothesis(h)
        g.add_evidence(_make_atom("e1", query_signature="sig-1"))

        g.link_support("H1", "e1")
        assert "e1" in h.supporting_evidence_ids
        assert "e1" in g.support_edges["H1"]

    def test_link_contradiction_syncs_to_hypothesis(self):
        g = EvidenceGraph()
        h = _make_hypothesis("H1")
        g.register_hypothesis(h)
        g.add_evidence(_make_atom("e1", query_signature="sig-1"))

        g.link_contradiction("H1", "e1")
        assert "e1" in h.contradicting_evidence_ids
        assert "e1" in g.contradiction_edges["H1"]

    def test_repeated_link_no_dupes_in_hypothesis(self):
        g = EvidenceGraph()
        h = _make_hypothesis("H1")
        g.register_hypothesis(h)
        g.add_evidence(_make_atom("e1", query_signature="sig-1"))

        g.link_support("H1", "e1")
        g.link_support("H1", "e1")  # duplicate
        assert len(h.supporting_evidence_ids) == 1

    def test_repeated_link_no_dupes_in_graph(self):
        g = EvidenceGraph()
        h = _make_hypothesis("H1")
        g.register_hypothesis(h)
        g.add_evidence(_make_atom("e1", query_signature="sig-1"))

        g.link_support("H1", "e1")
        g.link_support("H1", "e1")
        assert len(g.support_edges["H1"]) == 1

    def test_validate_consistency_passes_normal(self):
        g = EvidenceGraph()
        h = _make_hypothesis("H1")
        g.register_hypothesis(h)
        g.add_evidence(_make_atom("e1", query_signature="sig-1"))
        g.link_support("H1", "e1")
        g.validate_consistency()  # should not raise

    def test_validate_consistency_detects_missing_hypothesis_support(self):
        g = EvidenceGraph()
        h = _make_hypothesis("H1")
        g.register_hypothesis(h)
        g.add_evidence(_make_atom("e1", query_signature="sig-1"))
        g.link_support("H1", "e1")

        # Manually corrupt hypothesis (drift)
        h.supporting_evidence_ids.clear()

        with pytest.raises(ValueError, match="support edge"):
            g.validate_consistency()

    def test_validate_consistency_detects_missing_hypothesis_contradiction(self):
        g = EvidenceGraph()
        h = _make_hypothesis("H1")
        g.register_hypothesis(h)
        g.add_evidence(_make_atom("e1", query_signature="sig-1"))
        g.link_contradiction("H1", "e1")

        h.contradicting_evidence_ids.clear()

        with pytest.raises(ValueError, match="contradiction edge"):
            g.validate_consistency()

    def test_validate_consistency_detects_orphan_hypothesis_evidence(self):
        g = EvidenceGraph()
        h = _make_hypothesis("H1")
        g.register_hypothesis(h)
        g.add_evidence(_make_atom("e1", query_signature="sig-1"))

        # Manually add evidence_id to hypothesis without graph edge
        h.supporting_evidence_ids.append("e1")

        with pytest.raises(ValueError, match="no support edge"):
            g.validate_consistency()

    def test_validate_consistency_detects_evidence_not_in_graph(self):
        g = EvidenceGraph()
        h = _make_hypothesis("H1")
        g.register_hypothesis(h)

        # Manually add phantom evidence id
        h.supporting_evidence_ids.append("phantom")

        with pytest.raises(ValueError, match="not in graph"):
            g.validate_consistency()

    def test_evidence_supports_one_contradicts_another(self):
        g = EvidenceGraph()
        h1 = _make_hypothesis("H1")
        h2 = _make_hypothesis("H2")
        g.register_hypothesis(h1)
        g.register_hypothesis(h2)
        g.add_evidence(_make_atom("e1", query_signature="sig-1"))

        g.link_support("H1", "e1")
        g.link_contradiction("H2", "e1")

        assert "e1" in h1.supporting_evidence_ids
        assert "e1" in h2.contradicting_evidence_ids


# ===========================================================================
# C2. EvidenceGraph Conflict Detection
# ===========================================================================

class TestEvidenceConflictDetection:
    """Conflict detection for duplicate evidence_id or query_signature."""

    def test_same_evidence_id_different_payload_raises(self):
        g = EvidenceGraph()
        g.add_evidence(_make_atom("e1", query_signature="sig-1", observation={"a": 1}))
        with pytest.raises(ValueError, match="Conflicting evidence payload for evidence_id"):
            g.add_evidence(
                _make_atom("e1", query_signature="sig-2", observation={"b": 2})
            )

    def test_same_query_signature_different_payload_raises(self):
        g = EvidenceGraph()
        g.add_evidence(_make_atom("e1", query_signature="sig-conflict", observation={"a": 1}))
        with pytest.raises(ValueError, match="Conflicting evidence payload for query_signature"):
            g.add_evidence(
                _make_atom("e2", query_signature="sig-conflict", observation={"b": 2})
            )

    def test_same_query_signature_same_payload_returns_existing(self):
        g = EvidenceGraph()
        a1 = g.add_evidence(
            _make_atom("e1", query_signature="sig-same", observation={"a": 1})
        )
        a2 = g.add_evidence(
            _make_atom("e1", query_signature="sig-same", observation={"a": 1})
        )
        assert a1 is a2
        assert g.evidence_count() == 1

    def test_same_evidence_id_same_payload_no_error(self):
        g = EvidenceGraph()
        a1 = g.add_evidence(_make_atom("e1", query_signature="sig-1"))
        a2 = g.add_evidence(_make_atom("e1", query_signature="sig-1"))
        assert a1 is a2


# ===========================================================================
# D. ActionGate State Constraints
# ===========================================================================

class TestActionGateDraftRejection:
    """ActionGate must reject DRAFT hypotheses and allow valid states."""

    def _gate_and_action(self):
        gate = ActionGate()
        action = _make_action()
        return gate, action

    def test_draft_rejected(self):
        gate, action = self._gate_and_action()
        h1 = _make_hypothesis("H1", activate=False)  # DRAFT
        h2 = _make_hypothesis("H2")
        hm = {"H1": h1, "H2": h2}
        d = gate.evaluate(action, hm)
        assert d.accepted is False
        assert any("draft" in r.lower() for r in d.reasons)

    def test_active_accepted(self):
        gate, action = self._gate_and_action()
        h1 = _make_hypothesis("H1")  # ACTIVE
        h2 = _make_hypothesis("H2")  # ACTIVE
        d = gate.evaluate(action, {"H1": h1, "H2": h2})
        assert d.accepted is True

    def test_supported_accepted(self):
        gate, action = self._gate_and_action()
        h1 = _make_hypothesis("H1")
        h1.transition_to(HypothesisStatus.SUPPORTED)
        h2 = _make_hypothesis("H2")
        d = gate.evaluate(action, {"H1": h1, "H2": h2})
        assert d.accepted is True

    def test_weakened_accepted(self):
        gate, action = self._gate_and_action()
        h1 = _make_hypothesis("H1")
        h1.transition_to(HypothesisStatus.WEAKENED)
        h2 = _make_hypothesis("H2")
        d = gate.evaluate(action, {"H1": h1, "H2": h2})
        assert d.accepted is True

    def test_survived_accepted(self):
        gate, action = self._gate_and_action()
        h1 = _make_hypothesis("H1")
        h1.transition_to(HypothesisStatus.SUPPORTED)
        h1.transition_to(HypothesisStatus.SURVIVED)
        h2 = _make_hypothesis("H2")
        d = gate.evaluate(action, {"H1": h1, "H2": h2})
        assert d.accepted is True

    def test_refuted_rejected(self):
        gate, action = self._gate_and_action()
        h1 = _make_hypothesis("H1")
        h1.transition_to(HypothesisStatus.REFUTED)
        h2 = _make_hypothesis("H2")
        d = gate.evaluate(action, {"H1": h1, "H2": h2})
        assert d.accepted is False
        assert any("refuted" in r.lower() for r in d.reasons)

    def test_final_rejected(self):
        gate, action = self._gate_and_action()
        h1 = _make_hypothesis("H1")
        h1.transition_to(HypothesisStatus.SUPPORTED)
        h1.transition_to(HypothesisStatus.SURVIVED)
        h1.transition_to(HypothesisStatus.FINAL)
        h2 = _make_hypothesis("H2")
        d = gate.evaluate(action, {"H1": h1, "H2": h2})
        assert d.accepted is False
        assert any("final" in r.lower() for r in d.reasons)


# ===========================================================================
# E. Component Scope Normalization
# ===========================================================================

class TestComponentScopeNormalization:
    """EvidenceAtom normalizes component_scope (dedup, sort)."""

    def test_deduplicates_and_sorts(self):
        atom = _make_atom(
            component_scope=("C", "A", "B", "A", "C"),
            query_signature="sig-norm",
        )
        assert atom.component_scope == ("A", "B", "C")

    def test_missing_fields_deduplicates_and_sorts(self):
        atom = _make_atom(
            missing_fields=("z_field", "a_field", "z_field"),
            query_signature="sig-mf",
        )
        assert atom.missing_fields == ("a_field", "z_field")

    def test_observation_not_mapping_raises(self):
        with pytest.raises(ValueError, match="observation must be a Mapping"):
            _make_atom(observation="not_a_map")


# ===========================================================================
# F. Deep Freeze Standalone
# ===========================================================================

class TestDeepFreeze:
    """deep_freeze standalone behavior."""

    def test_scalars_pass_through(self):
        assert deep_freeze(42) == 42
        assert deep_freeze("hello") == "hello"
        assert deep_freeze(True) is True
        assert deep_freeze(None) is None
        assert deep_freeze(3.14) == 3.14

    def test_nested_mapping_is_readonly(self):
        frozen = deep_freeze({"a": {"b": 1}})
        with pytest.raises(TypeError):
            frozen["a"]["b"] = 2

    def test_list_becomes_tuple(self):
        result = deep_freeze([1, 2, 3])
        assert isinstance(result, tuple)
        assert result == (1, 2, 3)

    def test_set_becomes_sorted_tuple(self):
        result = deep_freeze({3, 1, 2})
        assert isinstance(result, tuple)
        assert result == (1, 2, 3)

    def test_frozenset_becomes_sorted_tuple(self):
        result = deep_freeze(frozenset({3, 1, 2}))
        assert isinstance(result, tuple)
        assert result == (1, 2, 3)

    def test_structurally_equal_inputs_equal_outputs(self):
        a = deep_freeze({"x": [1, {"y": {3, 2}}]})
        b = deep_freeze({"x": [1, {"y": {2, 3}}]})
        assert a == b

    def test_unsupported_type_raises(self):
        class Foo:
            pass

        with pytest.raises(TypeError, match="deep_freeze does not support"):
            deep_freeze(Foo())


class TestCanonicalizeJsonValue:
    """canonicalize_json_value helpers."""

    def test_set_sorted_by_canonical_json(self):
        result = canonicalize_json_value({"items": {3, 1, 2}})
        assert result["items"] == [1, 2, 3]

    def test_nested_dict_key_ordered(self):
        result = canonicalize_json_value({"b": 1, "a": 2})
        assert list(result.keys()) == ["a", "b"]

    def test_unsupported_type_raises(self):
        class Foo:
            pass

        with pytest.raises(TypeError, match="Cannot canonicalize"):
            canonicalize_json_value(Foo())

    def test_does_not_silently_stringify(self):
        class Bad:
            def __str__(self):
                return "bad"

        with pytest.raises(TypeError, match="Cannot canonicalize"):
            canonicalize_json_value({"key": Bad()})
