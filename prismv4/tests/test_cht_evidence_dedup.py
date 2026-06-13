"""Tests for EvidenceGraph: deduplication, canonical signatures, linkage."""

import pytest

from prismv4.prism_cht.evidence_graph import (
    EvidenceAtom,
    EvidenceGraph,
    build_query_signature,
)
from prismv4.prism_cht.hypothesis import CausalHypothesis


def _make_atom(evidence_id="e1", **overrides):
    defaults = {
        "evidence_id": evidence_id,
        "query_signature": "sig-abc",
        "modality": "metric",
        "component_scope": ("payment-svc",),
        "time_window": (1000.0, 2000.0),
        "observation": {"cpu_pct": 95.0},
        "provenance": {"source": "prometheus", "query_time": 1000.0},
    }
    defaults.update(overrides)
    return EvidenceAtom(**defaults)


def _make_hypothesis(hid="H1"):
    return CausalHypothesis(
        hypothesis_id=hid,
        root_component="payment-svc",
        reason_family="cpu_exhaustion",
        onset_interval=(1000.0, 2000.0),
        local_trigger="CPU spike",
        propagation_path=["payment-svc"],
        explained_symptoms=["latency"],
        predicted_observations=["high CPU"],
        falsifiers=["no CPU spike"],
    )


class TestBuildQuerySignature:
    """Canonical signature determinism."""

    def test_order_independence(self):
        sig1 = build_query_signature(
            tool_name="inspect_trace_path",
            component_scope=["B", "A", "C"],
            signal_scope=["latency", "errors"],
            time_window=(1000.0, 2000.0),
            parameters={"span_kind": "client"},
        )
        sig2 = build_query_signature(
            tool_name="inspect_trace_path",
            component_scope=["C", "A", "B"],
            signal_scope=["errors", "latency"],
            time_window=(1000.0, 2000.0),
            parameters={"span_kind": "client"},
        )
        assert sig1 == sig2

    def test_different_tool_produces_different_signature(self):
        sig1 = build_query_signature(
            tool_name="inspect_trace_path",
            component_scope=["A"],
            signal_scope=["x"],
            time_window=(1000.0, 2000.0),
            parameters={},
        )
        sig2 = build_query_signature(
            tool_name="check_propagation_consistency",
            component_scope=["A"],
            signal_scope=["x"],
            time_window=(1000.0, 2000.0),
            parameters={},
        )
        assert sig1 != sig2

    def test_different_args_produce_different_signature(self):
        sig1 = build_query_signature(
            tool_name="inspect_trace_path",
            component_scope=["A"],
            signal_scope=["x"],
            time_window=(1000.0, 2000.0),
            parameters={"depth": "1"},
        )
        sig2 = build_query_signature(
            tool_name="inspect_trace_path",
            component_scope=["A"],
            signal_scope=["x"],
            time_window=(1000.0, 2000.0),
            parameters={"depth": "2"},
        )
        assert sig1 != sig2

    def test_signature_is_hex_string(self):
        sig = build_query_signature(
            tool_name="inspect_trace_path",
            component_scope=["A"],
            signal_scope=["x"],
            time_window=(1000.0, 2000.0),
            parameters={},
        )
        assert len(sig) == 64
        assert all(c in "0123456789abcdef" for c in sig)


class TestEvidenceDedup:
    """Evidence Atom deduplication in EvidenceGraph."""

    def test_same_signature_added_once(self):
        g = EvidenceGraph()
        g.add_evidence(_make_atom("e1", query_signature="sig-x"))
        g.add_evidence(_make_atom("e1", query_signature="sig-x"))
        assert g.evidence_count() == 1

    def test_same_signature_returns_existing_atom(self):
        g = EvidenceGraph()
        a1 = g.add_evidence(_make_atom("e1", query_signature="sig-x"))
        a2 = g.add_evidence(_make_atom("e1", query_signature="sig-x"))
        assert a1 is a2

    def test_different_signatures_increase_count(self):
        g = EvidenceGraph()
        g.add_evidence(_make_atom("e1", query_signature="sig-1"))
        g.add_evidence(_make_atom("e2", query_signature="sig-2"))
        assert g.evidence_count() == 2

    def test_same_evidence_id_different_payload_raises(self):
        g = EvidenceGraph()
        g.add_evidence(_make_atom("e1", query_signature="sig-1", observation={"a": 1}))
        with pytest.raises(ValueError, match="evidence_id"):
            g.add_evidence(
                _make_atom("e1", query_signature="sig-2", observation={"b": 2})
            )

    def test_get_evidence_returns_correct_atom(self):
        g = EvidenceGraph()
        a = g.add_evidence(_make_atom("e1", query_signature="sig-1"))
        assert g.get_evidence("e1") is a

    def test_get_missing_evidence_raises(self):
        g = EvidenceGraph()
        with pytest.raises(KeyError, match="not found"):
            g.get_evidence("nonexistent")


class TestEvidenceLinkage:
    """Support/contradiction edges."""

    def test_repeated_link_support_no_duplicate(self):
        g = EvidenceGraph()
        g.register_hypothesis(_make_hypothesis("H1"))
        g.add_evidence(_make_atom("e1", query_signature="sig-1"))
        g.link_support("H1", "e1")
        g.link_support("H1", "e1")
        assert g.support_edges["H1"] == {"e1"}

    def test_evidence_supports_one_contradicts_another(self):
        g = EvidenceGraph()
        g.register_hypothesis(_make_hypothesis("H1"))
        g.register_hypothesis(_make_hypothesis("H2"))
        g.add_evidence(_make_atom("e1", query_signature="sig-1"))
        g.link_support("H1", "e1")
        g.link_contradiction("H2", "e1")
        assert g.support_edges["H1"] == {"e1"}
        assert g.contradiction_edges["H2"] == {"e1"}

    def test_link_with_missing_hypothesis_raises(self):
        g = EvidenceGraph()
        g.add_evidence(_make_atom("e1", query_signature="sig-1"))
        with pytest.raises(ValueError, match="not registered"):
            g.link_support("H1", "e1")

    def test_link_with_missing_evidence_raises(self):
        g = EvidenceGraph()
        g.register_hypothesis(_make_hypothesis("H1"))
        with pytest.raises(ValueError, match="not found"):
            g.link_support("H1", "e1")

    def test_no_scoring_logic(self):
        """EvidenceGraph must not contain scoring logic."""
        g = EvidenceGraph()
        assert not hasattr(g, "compute_score")
        assert not hasattr(g, "support_score")
        assert not hasattr(g, "against_score")
        assert not hasattr(g, "net_evidence")
        assert not hasattr(g, "posterior")
        assert not hasattr(g, "root_score")


class TestEvidenceAtomValidation:
    """EvidenceAtom invariants."""

    def test_empty_evidence_id_raises(self):
        with pytest.raises(ValueError, match="evidence_id"):
            _make_atom(evidence_id="")

    def test_empty_query_signature_raises(self):
        with pytest.raises(ValueError, match="query_signature"):
            _make_atom(query_signature="")

    def test_empty_modality_raises(self):
        with pytest.raises(ValueError, match="modality"):
            _make_atom(modality="")

    def test_empty_component_scope_raises(self):
        with pytest.raises(ValueError, match="component_scope"):
            _make_atom(component_scope=())

    def test_reversed_time_window_raises(self):
        with pytest.raises(ValueError, match="must be <= end"):
            _make_atom(time_window=(2000.0, 1000.0))

    def test_none_observation_raises(self):
        with pytest.raises(ValueError, match="observation"):
            _make_atom(observation=None)

    def test_none_provenance_raises(self):
        with pytest.raises(ValueError, match="provenance"):
            _make_atom(provenance=None)
