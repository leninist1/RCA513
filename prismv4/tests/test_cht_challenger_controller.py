"""Tests for ChallengerController: full demo scenarios, reuse, shared infrastructure."""

import pytest

from prismv4.prism_cht.action_schema import DiscriminativeAction
from prismv4.prism_cht.challenge_demo import (
    build_demo_inconclusive_challenge,
    build_demo_refutation_challenge,
    build_demo_survival_challenge,
)
from prismv4.prism_cht.challenge_types import ChallengeVerdict
from prismv4.prism_cht.challenger_controller import ChallengeControllerReuseError
from prismv4.prism_cht.hypothesis import HypothesisStatus


# ===========================================================================
# 1-5. Survival demo
# ===========================================================================


class TestSurvivalDemo:
    def test_survival_demo_runs(self):
        lead_ctrl, hyps, lead_policy, chall_ctrl, chall_policy = (
            build_demo_survival_challenge()
        )
        lead_result = lead_ctrl.run(initial_hypotheses=list(hyps), policy=lead_policy)
        challenge_result = chall_ctrl.run(
            lead_result=lead_result,
            hypotheses=lead_ctrl._graph.hypotheses_by_id,
            policy=chall_policy,
        )
        assert challenge_result is not None

    def test_survival_status_is_survived_challenge(self):
        lead_ctrl, hyps, lead_policy, chall_ctrl, chall_policy = (
            build_demo_survival_challenge()
        )
        lead_result = lead_ctrl.run(initial_hypotheses=list(hyps), policy=lead_policy)
        challenge_result = chall_ctrl.run(
            lead_result=lead_result,
            hypotheses=lead_ctrl._graph.hypotheses_by_id,
            policy=chall_policy,
        )
        assert challenge_result.status == "survived_challenge"

    def test_survival_h2_is_survived(self):
        lead_ctrl, hyps, lead_policy, chall_ctrl, chall_policy = (
            build_demo_survival_challenge()
        )
        lead_result = lead_ctrl.run(initial_hypotheses=list(hyps), policy=lead_policy)
        challenge_result = chall_ctrl.run(
            lead_result=lead_result,
            hypotheses=lead_ctrl._graph.hypotheses_by_id,
            policy=chall_policy,
        )
        h2 = lead_ctrl._graph.hypotheses_by_id["H-db_002-pool"]
        assert h2.status == HypothesisStatus.SURVIVED

    def test_survival_h2_not_final(self):
        lead_ctrl, hyps, lead_policy, chall_ctrl, chall_policy = (
            build_demo_survival_challenge()
        )
        lead_result = lead_ctrl.run(initial_hypotheses=list(hyps), policy=lead_policy)
        challenge_result = chall_ctrl.run(
            lead_result=lead_result,
            hypotheses=lead_ctrl._graph.hypotheses_by_id,
            policy=chall_policy,
        )
        h2 = lead_ctrl._graph.hypotheses_by_id["H-db_002-pool"]
        assert h2.status != HypothesisStatus.FINAL


# ===========================================================================
# 6-8. Refutation demo
# ===========================================================================


class TestRefutationDemo:
    def test_refutation_demo_runs(self):
        lead_ctrl, hyps, lead_policy, chall_ctrl, chall_policy = (
            build_demo_refutation_challenge()
        )
        lead_result = lead_ctrl.run(initial_hypotheses=list(hyps), policy=lead_policy)
        challenge_result = chall_ctrl.run(
            lead_result=lead_result,
            hypotheses=lead_ctrl._graph.hypotheses_by_id,
            policy=chall_policy,
        )
        assert challenge_result is not None

    def test_refutation_status_is_returned_to_lead(self):
        lead_ctrl, hyps, lead_policy, chall_ctrl, chall_policy = (
            build_demo_refutation_challenge()
        )
        lead_result = lead_ctrl.run(initial_hypotheses=list(hyps), policy=lead_policy)
        challenge_result = chall_ctrl.run(
            lead_result=lead_result,
            hypotheses=lead_ctrl._graph.hypotheses_by_id,
            policy=chall_policy,
        )
        assert challenge_result.status == "returned_to_lead"

    def test_refutation_h2_is_weakened(self):
        lead_ctrl, hyps, lead_policy, chall_ctrl, chall_policy = (
            build_demo_refutation_challenge()
        )
        lead_result = lead_ctrl.run(initial_hypotheses=list(hyps), policy=lead_policy)
        challenge_result = chall_ctrl.run(
            lead_result=lead_result,
            hypotheses=lead_ctrl._graph.hypotheses_by_id,
            policy=chall_policy,
        )
        h2 = lead_ctrl._graph.hypotheses_by_id["H-db_002-pool"]
        assert h2.status == HypothesisStatus.WEAKENED

    def test_refutation_no_auto_rerun_lead(self):
        lead_ctrl, hyps, lead_policy, chall_ctrl, chall_policy = (
            build_demo_refutation_challenge()
        )
        lead_result = lead_ctrl.run(initial_hypotheses=list(hyps), policy=lead_policy)
        challenge_result = chall_ctrl.run(
            lead_result=lead_result,
            hypotheses=lead_ctrl._graph.hypotheses_by_id,
            policy=chall_policy,
        )
        # Verify no second nomination was produced automatically
        assert challenge_result.status == "returned_to_lead"
        assert challenge_result.verdict == ChallengeVerdict.NOMINATION_REFUTED


# ===========================================================================
# 9-11. Inconclusive demo
# ===========================================================================


class TestInconclusiveDemo:
    def test_inconclusive_demo_runs(self):
        lead_ctrl, hyps, lead_policy, chall_ctrl, chall_policy = (
            build_demo_inconclusive_challenge()
        )
        lead_result = lead_ctrl.run(initial_hypotheses=list(hyps), policy=lead_policy)
        challenge_result = chall_ctrl.run(
            lead_result=lead_result,
            hypotheses=lead_ctrl._graph.hypotheses_by_id,
            policy=chall_policy,
        )
        assert challenge_result is not None

    def test_inconclusive_status(self):
        lead_ctrl, hyps, lead_policy, chall_ctrl, chall_policy = (
            build_demo_inconclusive_challenge()
        )
        lead_result = lead_ctrl.run(initial_hypotheses=list(hyps), policy=lead_policy)
        challenge_result = chall_ctrl.run(
            lead_result=lead_result,
            hypotheses=lead_ctrl._graph.hypotheses_by_id,
            policy=chall_policy,
        )
        assert challenge_result.status == "challenge_inconclusive"

    def test_inconclusive_h2_still_supported(self):
        lead_ctrl, hyps, lead_policy, chall_ctrl, chall_policy = (
            build_demo_inconclusive_challenge()
        )
        lead_result = lead_ctrl.run(initial_hypotheses=list(hyps), policy=lead_policy)
        challenge_result = chall_ctrl.run(
            lead_result=lead_result,
            hypotheses=lead_ctrl._graph.hypotheses_by_id,
            policy=chall_policy,
        )
        h2 = lead_ctrl._graph.hypotheses_by_id["H-db_002-pool"]
        assert h2.status == HypothesisStatus.SUPPORTED


# ===========================================================================
# 12-20. Controller invariants
# ===========================================================================


class TestControllerInvariants:
    def test_cannot_run_twice(self):
        lead_ctrl, hyps, lead_policy, chall_ctrl, chall_policy = (
            build_demo_survival_challenge()
        )
        lead_result = lead_ctrl.run(initial_hypotheses=list(hyps), policy=lead_policy)
        chall_ctrl.run(
            lead_result=lead_result,
            hypotheses=lead_ctrl._graph.hypotheses_by_id,
            policy=chall_policy,
        )
        with pytest.raises(ChallengeControllerReuseError, match="only be called once"):
            chall_ctrl.run(
                lead_result=lead_result,
                hypotheses=lead_ctrl._graph.hypotheses_by_id,
                policy=chall_policy,
            )

    def test_only_one_investigation(self):
        lead_ctrl, hyps, lead_policy, chall_ctrl, chall_policy = (
            build_demo_survival_challenge()
        )
        lead_result = lead_ctrl.run(initial_hypotheses=list(hyps), policy=lead_policy)
        evidence_before = lead_ctrl._graph.evidence_count()
        challenge_result = chall_ctrl.run(
            lead_result=lead_result,
            hypotheses=lead_ctrl._graph.hypotheses_by_id,
            policy=chall_policy,
        )
        evidence_after = lead_ctrl._graph.evidence_count()
        # Exactly one new evidence added
        assert evidence_after == evidence_before + 1

    def test_shares_evidence_graph(self):
        lead_ctrl, hyps, lead_policy, chall_ctrl, chall_policy = (
            build_demo_survival_challenge()
        )
        lead_result = lead_ctrl.run(initial_hypotheses=list(hyps), policy=lead_policy)
        challenge_result = chall_ctrl.run(
            lead_result=lead_result,
            hypotheses=lead_ctrl._graph.hypotheses_by_id,
            policy=chall_policy,
        )
        # Both controllers reference the same graph
        assert chall_ctrl._graph is lead_ctrl._graph

    def test_shares_action_gate(self):
        lead_ctrl, hyps, lead_policy, chall_ctrl, chall_policy = (
            build_demo_survival_challenge()
        )
        lead_result = lead_ctrl.run(initial_hypotheses=list(hyps), policy=lead_policy)
        challenge_result = chall_ctrl.run(
            lead_result=lead_result,
            hypotheses=lead_ctrl._graph.hypotheses_by_id,
            policy=chall_policy,
        )
        assert chall_ctrl._executor._gate is lead_ctrl._executor._gate

    def test_shares_telemetry_store(self):
        lead_ctrl, hyps, lead_policy, chall_ctrl, chall_policy = (
            build_demo_survival_challenge()
        )
        lead_result = lead_ctrl.run(initial_hypotheses=list(hyps), policy=lead_policy)
        challenge_result = chall_ctrl.run(
            lead_result=lead_result,
            hypotheses=lead_ctrl._graph.hypotheses_by_id,
            policy=chall_policy,
        )
        assert chall_ctrl._executor._store is lead_ctrl._executor._store

    def test_new_evidence_in_same_graph(self):
        lead_ctrl, hyps, lead_policy, chall_ctrl, chall_policy = (
            build_demo_survival_challenge()
        )
        lead_result = lead_ctrl.run(initial_hypotheses=list(hyps), policy=lead_policy)
        challenge_result = chall_ctrl.run(
            lead_result=lead_result,
            hypotheses=lead_ctrl._graph.hypotheses_by_id,
            policy=chall_policy,
        )
        # Challenger's new evidence is in the shared graph
        assert challenge_result.evidence_id in lead_ctrl._graph.evidence_by_id

    def test_returned_to_lead_no_new_nomination(self):
        lead_ctrl, hyps, lead_policy, chall_ctrl, chall_policy = (
            build_demo_refutation_challenge()
        )
        lead_result = lead_ctrl.run(initial_hypotheses=list(hyps), policy=lead_policy)
        challenge_result = chall_ctrl.run(
            lead_result=lead_result,
            hypotheses=lead_ctrl._graph.hypotheses_by_id,
            policy=chall_policy,
        )
        assert challenge_result.status == "returned_to_lead"
        # Lead controller has already run; no auto re-nomination
        assert challenge_result.verdict == ChallengeVerdict.NOMINATION_REFUTED

    def test_survived_challenge_no_final(self):
        lead_ctrl, hyps, lead_policy, chall_ctrl, chall_policy = (
            build_demo_survival_challenge()
        )
        lead_result = lead_ctrl.run(initial_hypotheses=list(hyps), policy=lead_policy)
        challenge_result = chall_ctrl.run(
            lead_result=lead_result,
            hypotheses=lead_ctrl._graph.hypotheses_by_id,
            policy=chall_policy,
        )
        assert challenge_result.status == "survived_challenge"
        for h in lead_ctrl._graph.hypotheses_by_id.values():
            assert h.status != HypothesisStatus.FINAL

    def test_challenge_inconclusive_no_final(self):
        lead_ctrl, hyps, lead_policy, chall_ctrl, chall_policy = (
            build_demo_inconclusive_challenge()
        )
        lead_result = lead_ctrl.run(initial_hypotheses=list(hyps), policy=lead_policy)
        challenge_result = chall_ctrl.run(
            lead_result=lead_result,
            hypotheses=lead_ctrl._graph.hypotheses_by_id,
            policy=chall_policy,
        )
        assert challenge_result.status == "challenge_inconclusive"
        for h in lead_ctrl._graph.hypotheses_by_id.values():
            assert h.status != HypothesisStatus.FINAL
