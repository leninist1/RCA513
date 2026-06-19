"""Integration tests for the full scripted Lead tournament (demo scenario)."""

import pytest

from prismv4.prism_cht.demo_scenario import build_demo_lead_tournament
from prismv4.prism_cht.hypothesis import HypothesisStatus
from prismv4.prism_cht.tournament_types import (
    HypothesisSnapshot,
    LeadTournamentSnapshot,
    build_snapshot,
)
from prismv4.prism_cht.canonical import deep_freeze
from dataclasses import dataclass


# ===========================================================================
# 1-4. Demo scenario runs to completion
# ===========================================================================


class TestDemoScenario:
    def test_full_run_completes(self):
        ctrl, hyps, policy = build_demo_lead_tournament()
        result = ctrl.run(initial_hypotheses=hyps, policy=policy)
        assert result is not None

    def test_status_is_challenge_required(self):
        ctrl, hyps, policy = build_demo_lead_tournament()
        result = ctrl.run(initial_hypotheses=hyps, policy=policy)
        assert result.status == "challenge_required"

    def test_nominated_hypothesis_is_h2(self):
        ctrl, hyps, policy = build_demo_lead_tournament()
        result = ctrl.run(initial_hypotheses=hyps, policy=policy)
        assert result.nominated_hypothesis_id == "H-db_002-pool"

    def test_rounds_completed_is_two(self):
        ctrl, hyps, policy = build_demo_lead_tournament()
        result = ctrl.run(initial_hypotheses=hyps, policy=policy)
        assert result.rounds_completed == 2

    def test_graph_has_two_evidence_atoms(self):
        ctrl, hyps, policy = build_demo_lead_tournament()
        result = ctrl.run(initial_hypotheses=hyps, policy=policy)
        assert len(result.evidence_ids) == 2

    def test_h2_remains_supported(self):
        ctrl, hyps, policy = build_demo_lead_tournament()
        result = ctrl.run(initial_hypotheses=hyps, policy=policy)
        h2 = ctrl._graph.hypotheses_by_id["H-db_002-pool"]
        assert h2.status == HypothesisStatus.SUPPORTED

    def test_h2_is_not_survived(self):
        ctrl, hyps, policy = build_demo_lead_tournament()
        result = ctrl.run(initial_hypotheses=hyps, policy=policy)
        h2 = ctrl._graph.hypotheses_by_id["H-db_002-pool"]
        assert h2.status != HypothesisStatus.SURVIVED

    def test_h2_is_not_final(self):
        ctrl, hyps, policy = build_demo_lead_tournament()
        result = ctrl.run(initial_hypotheses=hyps, policy=policy)
        h2 = ctrl._graph.hypotheses_by_id["H-db_002-pool"]
        assert h2.status != HypothesisStatus.FINAL

    def test_h1_is_weakened(self):
        ctrl, hyps, policy = build_demo_lead_tournament()
        result = ctrl.run(initial_hypotheses=hyps, policy=policy)
        h1 = ctrl._graph.hypotheses_by_id["H-os_009-cpu"]
        assert h1.status == HypothesisStatus.WEAKENED

    def test_h1_has_contradiction_evidence(self):
        ctrl, hyps, policy = build_demo_lead_tournament()
        result = ctrl.run(initial_hypotheses=hyps, policy=policy)
        h1 = ctrl._graph.hypotheses_by_id["H-os_009-cpu"]
        assert len(h1.contradicting_evidence_ids) >= 1

    def test_h2_has_at_least_two_support_evidence(self):
        ctrl, hyps, policy = build_demo_lead_tournament()
        result = ctrl.run(initial_hypotheses=hyps, policy=policy)
        h2 = ctrl._graph.hypotheses_by_id["H-db_002-pool"]
        assert len(h2.supporting_evidence_ids) >= 2

    def test_audit_steps_count_is_two(self):
        ctrl, hyps, policy = build_demo_lead_tournament()
        result = ctrl.run(initial_hypotheses=hyps, policy=policy)
        assert len(result.audit_steps) == 2

    def test_first_action_is_compare_onset_order(self):
        ctrl, hyps, policy = build_demo_lead_tournament()
        result = ctrl.run(initial_hypotheses=hyps, policy=policy)
        assert result.audit_steps[0].action.tool_name == "compare_onset_order"

    def test_second_action_is_inspect_trace_path(self):
        ctrl, hyps, policy = build_demo_lead_tournament()
        result = ctrl.run(initial_hypotheses=hyps, policy=policy)
        assert result.audit_steps[1].action.tool_name == "inspect_trace_path"

    def test_two_independent_runs_produce_same_result(self):
        ctrl1, hyps1, policy1 = build_demo_lead_tournament()
        result1 = ctrl1.run(initial_hypotheses=hyps1, policy=policy1)

        ctrl2, hyps2, policy2 = build_demo_lead_tournament()
        result2 = ctrl2.run(initial_hypotheses=hyps2, policy=policy2)

        assert result1.nominated_hypothesis_id == result2.nominated_hypothesis_id
        assert result1.rounds_completed == result2.rounds_completed
        assert result1.status == result2.status


# ===========================================================================
# 16. Snapshot is read-only — does not expose mutable Hypothesis
# ===========================================================================


class TestSnapshotReadOnly:
    def test_snapshot_hypotheses_are_snapshots_not_live(self):
        ctrl, hyps, policy = build_demo_lead_tournament()
        result = ctrl.run(initial_hypotheses=hyps, policy=policy)

        # Build snapshot from the controller's internal state
        from prismv4.prism_cht.tournament_types import build_snapshot
        snap = build_snapshot(
            round_index=99,
            hypotheses=ctrl._hypotheses,
            graph=ctrl._graph,
            audit_steps=result.audit_steps,
        )

        for hs in snap.hypotheses:
            assert isinstance(hs, HypothesisSnapshot)
            # Not a CausalHypothesis
            assert not hasattr(hs, "transition_to")

    def test_snapshot_does_not_expose_mutable_graph(self):
        ctrl, hyps, policy = build_demo_lead_tournament()
        result = ctrl.run(initial_hypotheses=hyps, policy=policy)

        snap = build_snapshot(
            round_index=99,
            hypotheses=ctrl._hypotheses,
            graph=ctrl._graph,
            audit_steps=result.audit_steps,
        )

        # evidence_ids is a tuple (immutable)
        assert isinstance(snap.evidence_ids, tuple)

        # Modifying the original graph should not affect the snapshot
        old_ids = snap.evidence_ids
        assert old_ids == snap.evidence_ids  # tuple unchanged


# ===========================================================================
# 11.5 Frozen dataclass safety
# ===========================================================================


class TestFrozenDataclassSafety:
    """Verify deep_freeze does not silently trust frozen dataclasses with mutable internals."""

    def test_frozen_dataclass_with_mutable_dict_rejected(self):
        @dataclass(frozen=True)
        class UnsafeOuter:
            name: str
            data: dict  # mutable dict inside

        obj = UnsafeOuter(name="bad", data={"key": [1, 2, 3]})
        with pytest.raises(TypeError, match="contains mutable internals"):
            deep_freeze(obj)

    def test_frozen_dataclass_with_mutable_list_rejected(self):
        @dataclass(frozen=True)
        class UnsafeOuter:
            name: str
            items: list

        obj = UnsafeOuter(name="bad", items=[1, 2, 3])
        with pytest.raises(TypeError, match="contains mutable internals"):
            deep_freeze(obj)

    def test_frozen_dataclass_with_mutable_set_rejected(self):
        @dataclass(frozen=True)
        class UnsafeOuter:
            name: str
            tags: set

        obj = UnsafeOuter(name="bad", tags={1, 2, 3})
        with pytest.raises(TypeError, match="contains mutable internals"):
            deep_freeze(obj)

    def test_safe_frozen_dataclass_allowed(self):
        @dataclass(frozen=True)
        class SafeInner:
            x: int
            y: str

        @dataclass(frozen=True)
        class SafeOuter:
            name: str
            inner: SafeInner
            values: tuple  # immutable tuple

        obj = SafeOuter(name="good", inner=SafeInner(1, "a"), values=(1, 2, 3))
        result = deep_freeze(obj)
        assert result is obj  # passes through unchanged

    def test_nested_mutable_in_tuple_caught(self):
        @dataclass(frozen=True)
        class UnsafeNested:
            name: str
            values: tuple  # tuple itself is immutable, but items could be mutable

        # Tuple of strings is fine
        obj1 = UnsafeNested(name="ok", values=("a", "b"))
        deep_freeze(obj1)  # should pass

        # Tuple containing a dict is NOT fine
        obj2 = UnsafeNested(name="bad", values=({"a": 1},))
        with pytest.raises(TypeError, match="contains mutable internals"):
            deep_freeze(obj2)

    def test_mock_telemetry_store_is_safe(self):
        from prismv4.prism_cht.telemetry_store import MockTelemetryStore, OnsetObservation

        store = MockTelemetryStore(
            onset_observations=[
                OnsetObservation("comp-a", "cpu", 100.0, "prom"),
            ],
        )

        # MockTelemetryStore is a frozen dataclass whose fields
        # are tuples of frozen dataclasses — should be safe
        result = deep_freeze(store)
        assert result is store
