"""Tests for InvestigationExecutor transaction semantics."""

import pytest

from prismv4.prism_cht.action_gate import ActionGate
from prismv4.prism_cht.action_schema import DiscriminativeAction
from prismv4.prism_cht.canonical import deep_freeze
from prismv4.prism_cht.evidence_graph import EvidenceAtom, EvidenceGraph
from prismv4.prism_cht.executor import ActionRejectedError, InvestigationExecutor
from prismv4.prism_cht.hypothesis import CausalHypothesis
from prismv4.prism_cht.telemetry_store import MockTelemetryStore, OnsetObservation
from prismv4.prism_cht.tool_registry import ToolRegistry
from prismv4.prism_cht.tools.compare_onset_order import CompareOnsetOrderTool
from prismv4.prism_cht.tools.inspect_trace_path import InspectTracePathTool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
        "args": {
            "component_scope": ["payment-svc", "order-svc"],
            "signal_scope": ["cpu"],
            "time_window": [1000.0, 2000.0],
        },
        "expected_outcomes": {
            "H1": "payment-svc spikes before order-svc",
            "H2": "order-svc spikes before payment-svc",
        },
        "why_discriminative": "These predict opposite onset order",
    }
    defaults.update(overrides)
    return DiscriminativeAction(**defaults)


def _make_executor(gate=None, registry=None, graph=None, store=None):
    gate = gate or ActionGate()
    registry = registry or _make_registry()
    graph = graph or EvidenceGraph()
    store = store or MockTelemetryStore()
    return InvestigationExecutor(
        gate=gate, registry=registry, graph=graph, store=store
    )


def _make_registry(tool=None):
    r = ToolRegistry()
    r.register(tool or CompareOnsetOrderTool())
    return r


def _make_hypotheses():
    h1 = _make_hypothesis("H1", "payment-svc")
    h2 = _make_hypothesis("H2", "order-svc")
    return {"H1": h1, "H2": h2}


def _make_store_with_onset():
    return MockTelemetryStore(
        onset_observations=[
            OnsetObservation("payment-svc", "cpu", 1100.0, "prom"),
            OnsetObservation("order-svc", "cpu", 1200.0, "prom"),
        ]
    )


# ===========================================================================
# 1-3. Happy path
# ===========================================================================


class TestExecutorHappyPath:
    """Legal action completes full flow."""

    def test_successful_execution_returns_evidence_atom(self):
        executor = _make_executor(store=_make_store_with_onset())
        action = _make_action()
        hypotheses = _make_hypotheses()

        atom = executor.execute(action=action, hypotheses=hypotheses)

        assert isinstance(atom, EvidenceAtom)
        assert atom.modality == "onset"
        assert len(atom.observation["observed_onsets"]) == 2

    def test_evidence_atom_query_signature_matches_action(self):
        executor = _make_executor(store=_make_store_with_onset())
        action = _make_action()
        hypotheses = _make_hypotheses()

        atom = executor.execute(action=action, hypotheses=hypotheses)

        assert atom.query_signature == action.query_signature()

    def test_evidence_id_is_stable(self):
        executor = _make_executor(store=_make_store_with_onset())
        action = _make_action()
        hypotheses = _make_hypotheses()

        atom = executor.execute(action=action, hypotheses=hypotheses)

        expected_id = "evidence:" + action.query_signature()
        assert atom.evidence_id == expected_id
        # hex sha256 is 64 chars, so evidence_id = 9 + 64 = 73
        assert len(atom.evidence_id) == 73

    def test_evidence_atom_in_graph(self):
        executor = _make_executor(store=_make_store_with_onset())
        action = _make_action()
        hypotheses = _make_hypotheses()

        atom = executor.execute(action=action, hypotheses=hypotheses)

        assert executor._graph.get_evidence(atom.evidence_id) is atom
        assert executor._graph.evidence_count() == 1


# ===========================================================================
# 4-5. Duplicate rejection
# ===========================================================================


class TestExecutorDuplicateRejection:
    """Same action executed twice is rejected before tool call."""

    def test_second_execution_raises(self):
        executor = _make_executor(store=_make_store_with_onset())
        action = _make_action()
        hypotheses = _make_hypotheses()

        executor.execute(action=action, hypotheses=hypotheses)

        with pytest.raises(ActionRejectedError, match="rejected"):
            executor.execute(action=action, hypotheses=hypotheses)

    def test_second_execution_no_extra_store_call(self):
        store = _make_store_with_onset()
        executor = _make_executor(store=store)
        action = _make_action()
        hypotheses = _make_hypotheses()

        executor.execute(action=action, hypotheses=hypotheses)
        count_after_first = store.onset_call_count()

        with pytest.raises(ActionRejectedError):
            executor.execute(action=action, hypotheses=hypotheses)

        # Store should not have been called again
        assert store.onset_call_count() == count_after_first

    def test_graph_evidence_count_unchanged_on_reject(self):
        executor = _make_executor(store=_make_store_with_onset())
        action = _make_action()
        hypotheses = _make_hypotheses()

        executor.execute(action=action, hypotheses=hypotheses)
        count = executor._graph.evidence_count()

        with pytest.raises(ActionRejectedError):
            executor.execute(action=action, hypotheses=hypotheses)

        assert executor._graph.evidence_count() == count


# ===========================================================================
# 6-7. No auto-linking
# ===========================================================================


class TestExecutorNoAutoLinking:
    """Executor never auto-links evidence to hypotheses."""

    def test_no_support_edges_after_execution(self):
        executor = _make_executor(store=_make_store_with_onset())
        action = _make_action()
        hypotheses = _make_hypotheses()

        executor.execute(action=action, hypotheses=hypotheses)

        # No support edges should have been added automatically
        assert executor._graph.support_edges == {}

    def test_no_contradiction_edges_after_execution(self):
        executor = _make_executor(store=_make_store_with_onset())
        action = _make_action()
        hypotheses = _make_hypotheses()

        executor.execute(action=action, hypotheses=hypotheses)

        assert executor._graph.contradiction_edges == {}


# ===========================================================================
# 8-9. Dispatch arg isolation
# ===========================================================================


class TestExecutorDispatchArgsIsolation:
    """Tool receives plain mutable dict, modifying it does not affect action."""

    def test_tool_receives_plain_dict_not_mappingproxy(self):
        # Use a spy tool to inspect the args type
        class SpyTool(CompareOnsetOrderTool):
            name = "compare_onset_order"

            def execute(self, *, args, store):
                # Capture the type
                self._args_type = type(args)
                return super().execute(args=args, store=store)

        spy = SpyTool()
        registry = ToolRegistry()
        registry.register(spy)
        executor = _make_executor(registry=registry, store=_make_store_with_onset())
        action = _make_action()
        hypotheses = _make_hypotheses()

        executor.execute(action=action, hypotheses=hypotheses)

        assert spy._args_type is dict

    def test_modifying_dispatch_args_does_not_affect_action_args(self):
        # Use a tool that mutates args but still passes validation
        class MutatingTool(CompareOnsetOrderTool):
            name = "compare_onset_order"

            def execute(self, *, args, store):
                # Mutate dispatch args to valid values
                args["component_scope"] = ["hacked-a", "hacked-b"]
                args["new_key"] = "injected"
                if isinstance(args.get("signal_scope"), list):
                    args["signal_scope"].append("extra_signal")
                return super().execute(args=args, store=store)

        registry = ToolRegistry()
        registry.register(MutatingTool())
        executor = _make_executor(registry=registry, store=_make_store_with_onset())
        action = _make_action()
        hypotheses = _make_hypotheses()

        executor.execute(action=action, hypotheses=hypotheses)

        # Action args should be unchanged
        assert action.args["component_scope"] != ("hacked",)
        assert "new_key" not in action.args


# ===========================================================================
# 10-11. Error recovery
# ===========================================================================


class TestExecutorErrorRecovery:
    """Tool failure releases reserve; retry is allowed."""

    def test_tool_error_releases_reserve(self):
        class FailingTool(CompareOnsetOrderTool):
            name = "compare_onset_order"

            def execute(self, *, args, store):
                raise RuntimeError("simulated tool failure")

        registry = ToolRegistry()
        registry.register(FailingTool())
        gate = ActionGate()
        executor = _make_executor(gate=gate, registry=registry, store=_make_store_with_onset())
        action = _make_action()
        hypotheses = _make_hypotheses()

        with pytest.raises(RuntimeError, match="simulated"):
            executor.execute(action=action, hypotheses=hypotheses)

        # Signature should NOT be in executed or reserved
        sig = action.query_signature()
        assert sig not in gate._executed_signatures
        assert sig not in gate._reserved_signatures

    def test_retry_after_tool_error_succeeds(self):
        call_count = [0]

        class OnceFailingTool(CompareOnsetOrderTool):
            name = "compare_onset_order"

            def execute(self, *, args, store):
                call_count[0] += 1
                if call_count[0] == 1:
                    raise RuntimeError("first call fails")
                return super().execute(args=args, store=store)

        registry = ToolRegistry()
        registry.register(OnceFailingTool())
        gate = ActionGate()
        executor = _make_executor(gate=gate, registry=registry, store=_make_store_with_onset())
        action = _make_action()
        hypotheses = _make_hypotheses()

        with pytest.raises(RuntimeError):
            executor.execute(action=action, hypotheses=hypotheses)

        # Retry should succeed
        atom = executor.execute(action=action, hypotheses=hypotheses)
        assert isinstance(atom, EvidenceAtom)
        assert call_count[0] == 2


# ===========================================================================
# 12-13. Unregistered tool handling
# ===========================================================================


class TestExecutorUnregisteredTool:
    """Unregistered tool is rejected before reserve."""

    def test_unregistered_tool_rejected(self):
        executor = _make_executor(store=_make_store_with_onset())
        action = _make_action(tool_name="check_propagation_consistency")
        hypotheses = _make_hypotheses()

        with pytest.raises(ValueError, match="not registered"):
            executor.execute(action=action, hypotheses=hypotheses)

    def test_unregistered_tool_does_not_pollute_state(self):
        gate = ActionGate()
        executor = _make_executor(gate=gate, store=_make_store_with_onset())
        action = _make_action(tool_name="check_propagation_consistency")
        hypotheses = _make_hypotheses()

        with pytest.raises(ValueError, match="not registered"):
            executor.execute(action=action, hypotheses=hypotheses)

        sig = action.query_signature()
        assert sig not in gate._reserved_signatures
        assert sig not in gate._executed_signatures
        assert executor._graph.evidence_count() == 0


# ===========================================================================
# 14. Graph error recovery
# ===========================================================================


class TestExecutorGraphErrorRecovery:
    """graph.add_evidence failure releases reserve."""

    def test_graph_error_releases_reserve(self):
        gate = ActionGate()
        graph = EvidenceGraph()
        action = _make_action()
        sig = action.query_signature()
        eid = "evidence:" + sig

        # Pre-populate graph with a conflicting atom (same evidence_id,
        # different payload) so add_evidence fails
        pre_atom = EvidenceAtom(
            evidence_id=eid,
            query_signature="different-sig-xxxxxxxx",
            modality="metric",
            component_scope=("X",),
            time_window=(0.0, 1.0),
            observation={"bad": 1},
            provenance={"src": "manual"},
        )
        graph.add_evidence(pre_atom)

        executor = _make_executor(
            gate=gate, graph=graph, store=_make_store_with_onset()
        )
        hypotheses = _make_hypotheses()

        # Executor will try add_evidence with same eid but different
        # query_signature, triggering conflict
        with pytest.raises(ValueError, match="Conflicting"):
            executor.execute(action=action, hypotheses=hypotheses)

        # Reserve should be released
        assert sig not in gate._reserved_signatures
        assert sig not in gate._executed_signatures


# ===========================================================================
# 15-16. Gate state transitions
# ===========================================================================


class TestGateStateTransitions:
    """commit / release correctly manage gate sets."""

    def test_commit_moves_signature_to_executed(self):
        gate = ActionGate()
        action = _make_action()
        hypotheses = _make_hypotheses()

        gate.reserve(action, hypotheses)
        sig = action.query_signature()
        assert sig in gate._reserved_signatures
        assert sig not in gate._executed_signatures

        gate.commit(sig)
        assert sig not in gate._reserved_signatures
        assert sig in gate._executed_signatures

    def test_release_removes_from_reserved(self):
        gate = ActionGate()
        action = _make_action()
        hypotheses = _make_hypotheses()

        gate.reserve(action, hypotheses)
        sig = action.query_signature()
        assert sig in gate._reserved_signatures

        gate.release(sig)
        assert sig not in gate._reserved_signatures
        assert sig not in gate._executed_signatures

    def test_commit_without_reserve_raises(self):
        gate = ActionGate()
        with pytest.raises(ValueError, match="was not reserved"):
            gate.commit("nonexistent")

    def test_release_without_reserve_raises(self):
        gate = ActionGate()
        with pytest.raises(ValueError, match="was not reserved"):
            gate.release("nonexistent")

    def test_release_allows_re_reserve(self):
        gate = ActionGate()
        action = _make_action()
        hypotheses = _make_hypotheses()
        sig = action.query_signature()

        gate.reserve(action, hypotheses)
        gate.release(sig)

        # Should be able to reserve again
        gate.reserve(action, hypotheses)
        assert sig in gate._reserved_signatures

    def test_double_reserve_rejected(self):
        gate = ActionGate()
        action = _make_action()
        hypotheses = _make_hypotheses()
        sig = action.query_signature()

        d1 = gate.reserve(action, hypotheses)
        assert d1.accepted

        d2 = gate.reserve(action, hypotheses)
        assert not d2.accepted
        assert any("reserved" in r.lower() for r in d2.reasons)

    def test_committed_cannot_be_reserved(self):
        gate = ActionGate()
        action = _make_action()
        hypotheses = _make_hypotheses()
        sig = action.query_signature()

        gate.reserve(action, hypotheses)
        gate.commit(sig)

        d = gate.reserve(action, hypotheses)
        assert not d.accepted
        assert any("executed" in r.lower() for r in d.reasons)


# ===========================================================================
# 17. Admit backwards compatibility
# ===========================================================================


class TestAdmitCompat:
    """Old admit() behavior still works correctly."""

    def test_admit_accepts_valid_action(self):
        gate = ActionGate()
        action = _make_action()
        hypotheses = _make_hypotheses()

        d = gate.admit(action, hypotheses)
        assert d.accepted is True

    def test_admit_rejects_duplicate(self):
        gate = ActionGate()
        action = _make_action()
        hypotheses = _make_hypotheses()

        d1 = gate.admit(action, hypotheses)
        assert d1.accepted

        d2 = gate.admit(action, hypotheses)
        assert not d2.accepted
        assert any("executed" in r.lower() for r in d2.reasons)

    def test_admit_leaves_signature_in_executed_only(self):
        gate = ActionGate()
        action = _make_action()
        hypotheses = _make_hypotheses()
        sig = action.query_signature()

        gate.admit(action, hypotheses)
        assert sig in gate._executed_signatures
        assert sig not in gate._reserved_signatures

    def test_admit_still_checks_draft(self):
        gate = ActionGate()
        action = _make_action()
        h1_draft = _make_hypothesis("H1", activate=False)
        h2 = _make_hypothesis("H2")
        hm = {"H1": h1_draft, "H2": h2}

        d = gate.admit(action, hm)
        assert d.accepted is False
        assert any("draft" in r.lower() for r in d.reasons)


# ===========================================================================
# Additional integration tests
# ===========================================================================


class TestExecutorIntegration:
    """Integration scenarios."""

    def test_action_rejected_error_contains_reasons(self):
        executor = _make_executor(store=_make_store_with_onset())
        action = _make_action()
        # Use a DRAFT hypothesis to trigger rejection at gate level
        h1_draft = _make_hypothesis("H1", activate=False)
        h2 = _make_hypothesis("H2")
        hypotheses = {"H1": h1_draft, "H2": h2}

        with pytest.raises(ActionRejectedError) as exc:
            executor.execute(action=action, hypotheses=hypotheses)

        assert "rejected" in str(exc.value)
        assert "draft" in str(exc.value).lower()

    def test_multiple_different_actions_execute_independently(self):
        store = _make_store_with_onset()
        executor = _make_executor(store=store)
        hypotheses = _make_hypotheses()

        a1 = _make_action(action_id="A1")
        a2 = _make_action(
            action_id="A2",
            tool_name="inspect_trace_path",
            args={
                "source_component": "payment-svc",
                "target_component": "order-svc",
                "time_window": [1000.0, 2000.0],
            },
            expected_outcomes={
                "H1": "payment-svc calls order-svc",
                "H2": "order-svc calls payment-svc",
            },
        )
        registry = _make_registry()
        registry.register(InspectTracePathTool())
        executor = _make_executor(registry=registry, store=store)

        a1_atom = executor.execute(action=a1, hypotheses=hypotheses)
        a2_atom = executor.execute(action=a2, hypotheses=hypotheses)

        assert a1_atom.evidence_id != a2_atom.evidence_id
        assert executor._graph.evidence_count() == 2
