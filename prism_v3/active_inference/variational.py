"""Variational inference with mean-field approximation.

E-step: q(r) ∝ exp( E_q(W)[log p(obs | r, W)] ) · p(r)
M-step: q(W[i,j]) ≈ Bernoulli(σ(log_prior_odds + E_q(r)[grad_log_obs]))

This replaces the decoupled _emotion_vector(), _likelihood(), _em_refine()
with a single unified variational loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math
import numpy as np

from .generative_model import GenerativeModel

EPS = 1e-9


@dataclass
class MeanFieldState:
    q_r: np.ndarray
    q_W_mean: np.ndarray
    entities: List[str]
    entity_types: Dict[str, str]
    elbo_history: List[float] = field(default_factory=list)
    step: int = 0


class VariationalInference:
    """Mean-field variational inference for the generative model."""

    def __init__(
        self,
        gen_model: GenerativeModel,
        max_em_iters: int = 5,
        lr_W: float = 0.02,
    ):
        self.gen_model = gen_model
        self.max_em_iters = max_em_iters
        self.lr_W = lr_W

    def initialize(
        self,
        a_obs: np.ndarray,
        log_signal: np.ndarray,
        W0: np.ndarray,
        W_frozen: np.ndarray,
        entity_types: Dict[str, str],
        entities: List[str],
        root_prior: np.ndarray,
    ) -> MeanFieldState:
        n = len(entities)
        q_r = (
            root_prior.copy() if root_prior is not None else np.ones(n, dtype=float) / n
        )
        q_r = np.clip(q_r, EPS, None)
        q_r /= q_r.sum()

        return MeanFieldState(
            q_r=q_r,
            q_W_mean=W0.copy(),
            entities=list(entities),
            entity_types=dict(entity_types),
        )

    def e_step(
        self,
        state: MeanFieldState,
        a_obs: np.ndarray,
        log_signal: np.ndarray,
        trace_edges: Dict[Tuple[str, str], float],
        W_frozen: np.ndarray,
        root_prior: np.ndarray,
        fault_category: str = "generic",
    ) -> np.ndarray:
        lp = self.gen_model.joint_log_prob_all(
            a_obs=a_obs,
            log_signal=log_signal,
            W=state.q_W_mean,
            W_frozen=W_frozen,
            entity_types=state.entity_types,
            entities=state.entities,
            trace_edges=trace_edges,
            root_prior=root_prior,
            fault_category=fault_category,
        )
        lp = lp - np.max(lp)
        q_r_new = np.exp(lp)
        q_r_new = np.clip(q_r_new, EPS, None)
        q_r_new /= q_r_new.sum()
        return q_r_new

    def m_step(
        self,
        state: MeanFieldState,
        a_obs: np.ndarray,
        W_frozen: np.ndarray,
        fault_category: str = "generic",
    ) -> np.ndarray:
        n = len(state.entities)
        W = state.q_W_mean.copy()
        propagation = self.gen_model.propagation
        mu_matrix = propagation.propagate_all(
            a_obs, W, state.entity_types, state.entities, fault_category
        )
        expected_mu = np.sum(state.q_r[:, None] * mu_matrix, axis=0)
        residual = a_obs - expected_mu

        for j in range(n):
            if abs(float(residual[j])) < 0.05:
                continue
            for i in range(n):
                if i == j:
                    continue
                if W_frozen[i, j] >= 1.0:
                    continue
                grad = residual[j] * expected_mu[i]
                grad -= self.gen_model.params.graph_sparsity * np.sign(W[i, j])
                W[i, j] = float(np.clip(W[i, j] + self.lr_W * grad, 0.0, 1.0))
                if W[i, j] < 0.05 and W_frozen[i, j] < 1.0:
                    W[i, j] = 0.0
        return W

    def compute_elbo(
        self,
        state: MeanFieldState,
        a_obs: np.ndarray,
        log_signal: np.ndarray,
        trace_edges: Dict[Tuple[str, str], float],
        W_frozen: np.ndarray,
        root_prior: np.ndarray,
        fault_category: str = "generic",
    ) -> float:
        n = len(state.entities)
        log_joint = self.gen_model.joint_log_prob_all(
            a_obs,
            log_signal,
            state.q_W_mean,
            W_frozen,
            state.entity_types,
            state.entities,
            trace_edges,
            root_prior,
            fault_category,
        )
        expected_log_joint = float(np.sum(state.q_r * log_joint))
        entropy = -float(np.sum(state.q_r * np.log(np.clip(state.q_r, EPS, None))))
        elbo = expected_log_joint + entropy
        return elbo

    def iterate(
        self,
        state: MeanFieldState,
        a_obs: np.ndarray,
        log_signal: np.ndarray,
        trace_edges: Dict[Tuple[str, str], float],
        W_frozen: np.ndarray,
        root_prior: np.ndarray,
        fault_category: str = "generic",
        n_iters: Optional[int] = None,
    ) -> MeanFieldState:
        n_iters = n_iters or self.max_em_iters
        for _ in range(n_iters):
            state.q_r = self.e_step(
                state,
                a_obs,
                log_signal,
                trace_edges,
                W_frozen,
                root_prior,
                fault_category,
            )
            state.q_W_mean = self.m_step(
                state,
                a_obs,
                W_frozen,
                fault_category,
            )
            elbo = self.compute_elbo(
                state,
                a_obs,
                log_signal,
                trace_edges,
                W_frozen,
                root_prior,
                fault_category,
            )
            state.elbo_history.append(elbo)
            state.step += 1
        return state

    def evidence_lower_bound(self, state: MeanFieldState) -> float:
        if not state.elbo_history:
            return float("-inf")
        return state.elbo_history[-1]
