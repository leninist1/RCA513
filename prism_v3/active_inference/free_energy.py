"""Expected Free Energy computation for action selection.

EFE(d) = Epistemic Value(d) + Pragmatic Value(d)

Epistemic Value: Expected reduction in uncertainty about root cause
  ≈ E_q(o_d|d) [ H(q(r)) - H(q(r|o_d)) ]
  = Expected Information Gain

Pragmatic Value: Expected divergence from preferred outcome
  ≈ E_q(o_d|d) [ -log P(o_d | desired) ]
  Rewards convergence / penalizes risk

This replaces the separate EIG + Φ(e,d) blending in prism.py _score_action().
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import math
import numpy as np

from .variational import MeanFieldState, VariationalInference

EPS = 1e-9


@dataclass
class EFEResult:
    action_name: str
    epistemic_value: float
    pragmatic_value: float
    efe_total: float
    detail: Dict = None

    def __post_init__(self):
        if self.detail is None:
            self.detail = {}


class ExpectedFreeEnergyComputer:
    """Computes EFE for candidate actions under the variational state."""

    def __init__(
        self,
        vi: VariationalInference,
        risk_aversion: float = 0.3,
        n_outcome_bins: int = 5,
    ):
        self.vi = vi
        self.risk_aversion = risk_aversion
        self.n_outcome_bins = n_outcome_bins

    def _discretize_delta(self, n_bins: int = None) -> List[float]:
        n = n_bins or self.n_outcome_bins
        return [(i + 0.5) / n for i in range(n)]

    def epistemic_value_metric(
        self,
        state: MeanFieldState,
        entity: str,
        a_obs: np.ndarray,
    ) -> float:
        idx = state.entities.index(entity) if entity in state.entities else -1
        if idx < 0:
            return 0.0
        p_r = state.q_r[idx]
        entropy_current = -float(
            np.sum(state.q_r * np.log(np.clip(state.q_r, EPS, None)))
        )

        delta_values = self._discretize_delta()
        expected_entropy_after = 0.0
        prob_sum = 0.0

        for delta in delta_values:
            signal = float(a_obs[idx]) * (0.5 + 0.5 * delta)
            prob_outcome = p_r * signal + (1.0 - p_r) * (1.0 - signal)
            prob_outcome = max(prob_outcome, EPS)

            q_r_after = state.q_r.copy()
            q_r_after[idx] *= 1.0 + 0.8 * signal
            q_r_after = np.clip(q_r_after, EPS, None)
            q_r_after /= q_r_after.sum()
            entropy_after = -float(
                np.sum(q_r_after * np.log(np.clip(q_r_after, EPS, None)))
            )
            expected_entropy_after += prob_outcome * entropy_after
            prob_sum += prob_outcome

        if prob_sum < EPS:
            return 0.0
        expected_entropy_after /= prob_sum
        return entropy_current - expected_entropy_after

    def epistemic_value_counterfactual(
        self,
        state: MeanFieldState,
        entity: str,
    ) -> float:
        idx = state.entities.index(entity) if entity in state.entities else -1
        if idx < 0:
            return 0.0
        p_r = state.q_r[idx]
        entropy_current = -float(
            np.sum(state.q_r * np.log(np.clip(state.q_r, EPS, None)))
        )

        outcomes = [
            ("root_recovery", 0.8, p_r * 0.7 + (1.0 - p_r) * 0.1),
            ("moderate_recovery", 0.5, p_r * 0.2 + (1.0 - p_r) * 0.3),
            ("no_recovery", 0.2, p_r * 0.1 + (1.0 - p_r) * 0.6),
        ]

        expected_entropy_after = 0.0
        prob_sum = 0.0
        for _, target_delta, prob_outcome in outcomes:
            q_r_after = state.q_r.copy()
            if target_delta > 0.5:
                q_r_after[idx] *= 3.0
            elif target_delta > 0.2:
                q_r_after[idx] *= 1.5
            else:
                q_r_after[idx] *= 0.5
            q_r_after = np.clip(q_r_after, EPS, None)
            q_r_after /= q_r_after.sum()
            entropy_after = -float(
                np.sum(q_r_after * np.log(np.clip(q_r_after, EPS, None)))
            )
            expected_entropy_after += prob_outcome * entropy_after
            prob_sum += prob_outcome

        if prob_sum < EPS:
            return 0.0
        expected_entropy_after /= prob_sum
        return max(0.0, entropy_current - expected_entropy_after)

    def epistemic_value_trace(
        self,
        state: MeanFieldState,
        edge: Tuple[str, str],
    ) -> float:
        u, v = edge
        if u not in state.entities or v not in state.entities:
            return 0.0
        i, j = state.entities.index(u), state.entities.index(v)
        weight = state.q_W_mean[i, j]
        entropy_current = -float(
            np.sum(state.q_r * np.log(np.clip(state.q_r, EPS, None)))
        )

        prob_present = max(weight, EPS)
        prob_absent = max(1.0 - weight, EPS)
        prob_sum = prob_present + prob_absent

        expected_entropy_after = 0.0
        for prob, w_val in [
            (prob_present, min(1.0, weight * 1.2)),
            (prob_absent, max(0.0, weight * 0.3)),
        ]:
            q_r_after = state.q_r.copy()
            if w_val > 0.5:
                q_r_after[j] *= 1.3
                q_r_after[i] *= 1.1
            else:
                q_r_after[j] *= 0.7
            q_r_after = np.clip(q_r_after, EPS, None)
            q_r_after /= q_r_after.sum()
            entropy_after = -float(
                np.sum(q_r_after * np.log(np.clip(q_r_after, EPS, None)))
            )
            expected_entropy_after += prob * entropy_after

        expected_entropy_after /= prob_sum
        return max(0.0, entropy_current - expected_entropy_after)

    def pragmatic_value(
        self,
        state: MeanFieldState,
        cost: float,
    ) -> float:
        return -self.risk_aversion * cost

    def efe_metric(
        self,
        state: MeanFieldState,
        entity: str,
        a_obs: np.ndarray,
        cost: float,
    ) -> EFEResult:
        ev = self.epistemic_value_metric(state, entity, a_obs)
        pv = self.pragmatic_value(state, cost)
        return EFEResult(
            action_name="METRIC_DEEP",
            epistemic_value=ev,
            pragmatic_value=pv,
            efe_total=ev + pv,
            detail={"entity": entity, "cost": cost},
        )

    def efe_counterfactual(
        self,
        state: MeanFieldState,
        entity: str,
    ) -> EFEResult:
        ev = self.epistemic_value_counterfactual(state, entity)
        pv = self.pragmatic_value(state, 60.0)
        return EFEResult(
            action_name="COUNTERFACTUAL",
            epistemic_value=ev,
            pragmatic_value=pv,
            efe_total=ev + pv,
            detail={"entity": entity, "cost": 60.0},
        )

    def efe_log(
        self,
        state: MeanFieldState,
        entity: str,
        log_signal: np.ndarray,
    ) -> EFEResult:
        idx = state.entities.index(entity) if entity in state.entities else -1
        if idx < 0:
            return EFEResult("LOG_QUERY", 0.0, 0.0, 0.0, {"entity": entity})
        p_r = state.q_r[idx]
        log_local = float(log_signal[idx]) if idx < len(log_signal) else 0.0
        ev = max(0.0, p_r * (1.0 - p_r) * (0.5 + log_local))
        pv = self.pragmatic_value(state, 0.5)
        return EFEResult(
            action_name="LOG_QUERY",
            epistemic_value=ev,
            pragmatic_value=pv,
            efe_total=ev + pv,
            detail={"entity": entity, "cost": 0.5},
        )

    def efe_trace(
        self,
        state: MeanFieldState,
        edge: Tuple[str, str],
    ) -> EFEResult:
        ev = self.epistemic_value_trace(state, edge)
        pv = self.pragmatic_value(state, 2.0)
        return EFEResult(
            action_name="TRACE_VERIFY",
            epistemic_value=ev,
            pragmatic_value=pv,
            efe_total=ev + pv,
            detail={"edge": edge, "cost": 2.0},
        )

    def rank_actions(
        self,
        state: MeanFieldState,
        efe_results: List[EFEResult],
    ) -> List[EFEResult]:
        scored = []
        for efe in efe_results:
            scored.append((efe, efe.efe_total))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [item[0] for item in scored]
