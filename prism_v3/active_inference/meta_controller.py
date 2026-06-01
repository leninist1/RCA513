"""EFE-based meta-controller: action selection and stopping.

Replaces the _enumerate_actions() + _score_action() + _converged() logic
with EFE-grounded decision making.

Action selection: d* = argmin_d EFE(d)
Stopping: stop if max EFE gain < threshold and belief concentrated
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math
import numpy as np

from .variational import MeanFieldState
from .free_energy import ExpectedFreeEnergyComputer, EFEResult

EPS = 1e-9


@dataclass
class ControllerConfig:
    efe_stop_threshold: float = 0.015
    belief_concentration_threshold: float = 0.80
    max_steps: int = 15
    risk_aversion: float = 0.3
    diversity_bonus: float = 0.15


class ActiveInferenceController:
    """Action selector based on Expected Free Energy minimization."""

    def __init__(
        self,
        efe_computer: ExpectedFreeEnergyComputer,
        config: Optional[ControllerConfig] = None,
    ):
        self.efe = efe_computer
        self.config = config or ControllerConfig()

    def select_action(
        self,
        state: MeanFieldState,
        action_candidates: List[EFEResult],
        visit_counts: Optional[Dict[str, int]] = None,
    ) -> EFEResult:
        scored = []
        for efe in action_candidates:
            utility = efe.efe_total
            if visit_counts:
                entity_key = efe.detail.get("entity", "")
                visits = visit_counts.get(entity_key, 0)
                utility *= math.exp(-self.config.diversity_bonus * visits)
            scored.append((efe, utility))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[0][0] if scored else EFEResult("STOP", 0.0, 0.0, 0.0)

    def should_stop(
        self,
        state: MeanFieldState,
        top_efe: float,
        step: int,
        top1_history: Optional[List[str]] = None,
        stable_rounds: int = 3,
    ) -> Tuple[bool, str]:
        if step >= self.config.max_steps:
            return True, "max_steps"
        if (
            np.max(state.q_r) > self.config.belief_concentration_threshold
            and top_efe < self.config.efe_stop_threshold
        ):
            if top1_history and len(top1_history) >= stable_rounds:
                tail = top1_history[-stable_rounds:]
                if len(set(tail)) == 1:
                    return True, "converged"
        if top_efe < self.config.efe_stop_threshold * 0.5:
            return True, "low_efe"
        return False, "continue"

    def compute_all_efe(
        self,
        state: MeanFieldState,
        a_obs: np.ndarray,
        log_signal: np.ndarray,
        top_k: int = 6,
    ) -> List[EFEResult]:
        results = []
        top_idx = np.argsort(state.q_r)[::-1][: min(top_k, len(state.entities))]

        for idx in top_idx:
            entity = state.entities[int(idx)]

            res = self.efe.efe_metric(state, entity, a_obs, cost=0.1)
            results.append(res)

            if idx < len(log_signal) and log_signal is not None:
                res = self.efe.efe_log(state, entity, log_signal)
                results.append(res)

            if state.q_r[idx] > 0.05:
                res = self.efe.efe_counterfactual(state, entity)
                results.append(res)

        n = len(state.entities)
        uncertain_edges = []
        for i in range(n):
            for j in range(n):
                w = state.q_W_mean[i, j]
                if 0.1 < w < 0.9:
                    uncertain_edges.append((state.entities[i], state.entities[j], w))
        uncertain_edges.sort(key=lambda item: abs(item[2] - 0.5))
        for u, v, _ in uncertain_edges[:10]:
            res = self.efe.efe_trace(state, (u, v))
            results.append(res)

        return results
