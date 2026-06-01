"""Generative model: joint distribution over observations, root cause, and graph.

P(obs, r, W) = P(a_obs | r, W) · P(logs | r) · P(traces | W) · P(W | prior) · P(r)

The observation model uses damped weighted propagation μ_r(W) from the
current PRISM framework, augmented with learned likelihood parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math
import numpy as np

from ..priors.graph_prior import LayerTransitionsConditional

EPS = 1e-9


@dataclass
class GenerativeModelParams:
    sigma2: float = 0.1
    alpha_base: float = 0.8
    log_signal_weight: float = 0.3
    trace_edge_weight: float = 0.5
    graph_sparsity: float = 0.05
    graph_smoothness: float = 0.10


class PropagationModel:
    """Damped weighted propagation: μ_r = how anomaly from root r spreads."""

    def __init__(
        self,
        alpha_base: float = 0.8,
        max_iter: int = 20,
        convergence_eps: float = 1e-3,
    ):
        self.alpha_base = alpha_base
        self.max_iter = max_iter
        self.convergence_eps = convergence_eps
        self._layer_transitions = LayerTransitionsConditional()

    def propagate(
        self,
        root_idx: int,
        a_obs: np.ndarray,
        W: np.ndarray,
        entity_types: Dict[str, str],
        entities: List[str],
        fault_category: str = "generic",
    ) -> np.ndarray:
        n = len(entities)
        mu = np.zeros(n, dtype=float)
        mu[root_idx] = max(0.5, float(a_obs[root_idx]))
        alpha_matrix = self._layer_transitions.build_alpha_matrix(
            entities=entities,
            entity_types=entity_types,
            fault_category=fault_category,
            alpha_base=self.alpha_base,
        )
        mu_prev = mu.copy()
        for _ in range(self.max_iter):
            for j in range(n):
                incoming = sum(
                    alpha_matrix[i, j] * W[i, j] * mu[i]
                    for i in range(n)
                    if W[i, j] > 0
                )
                mu[j] = max(float(a_obs[j]), incoming)
            delta = float(np.max(np.abs(mu - mu_prev)))
            if delta < self.convergence_eps:
                break
            mu_prev = mu.copy()
        return mu

    def propagate_all(
        self,
        a_obs: np.ndarray,
        W: np.ndarray,
        entity_types: Dict[str, str],
        entities: List[str],
        fault_category: str = "generic",
    ) -> np.ndarray:
        n = len(entities)
        mu_matrix = np.zeros((n, n), dtype=float)
        for r in range(n):
            mu_matrix[r] = self.propagate(
                r, a_obs, W, entity_types, entities, fault_category
            )
        return mu_matrix


class GenerativeModel:
    """Joint probabilistic model P(a_obs, logs, traces, r, W)."""

    def __init__(
        self,
        params: Optional[GenerativeModelParams] = None,
        propagation: Optional[PropagationModel] = None,
    ):
        self.params = params or GenerativeModelParams()
        self.propagation = propagation or PropagationModel(
            alpha_base=self.params.alpha_base
        )

    def log_lik_obs(
        self,
        a_obs: np.ndarray,
        mu_r: np.ndarray,
    ) -> float:
        diff = a_obs - mu_r
        n = len(diff)
        var = max(self.params.sigma2, EPS)
        return (
            -0.5 * n * math.log(2.0 * math.pi * var) - 0.5 * np.sum(diff * diff) / var
        )

    def log_lik_logs(
        self,
        log_signal: np.ndarray,
        root_idx: int,
    ) -> float:
        n = len(log_signal)
        ll = 0.0
        for v in range(n):
            p = log_signal[v] if v == root_idx else 0.1 * log_signal[v]
            p_clip = min(max(p, EPS), 1.0 - EPS)
            ll += p_clip * math.log(p_clip)
        return ll

    def log_lik_traces(
        self,
        W: np.ndarray,
        trace_edges: Dict[Tuple[str, str], float],
        entities: List[str],
    ) -> float:
        ll = 0.0
        for i, u in enumerate(entities):
            for j, v in enumerate(entities):
                key = (u, v)
                trace_w = trace_edges.get(key, 0.0)
                if trace_w > 0:
                    p = W[i, j]
                    p_clip = min(max(p, EPS), 1.0 - EPS)
                    ll += trace_w * math.log(p_clip) + (1.0 - trace_w) * math.log(
                        1.0 - p_clip
                    )
        return ll

    def log_prior_W(
        self,
        W: np.ndarray,
        W_frozen: np.ndarray,
    ) -> float:
        n = W.shape[0]
        lp = 0.0
        for i in range(n):
            for j in range(n):
                if W_frozen[i, j] >= 1.0:
                    continue
                w = W[i, j]
                if w > 0:
                    lp -= self.params.graph_sparsity * w
                lp -= 0.5 * self.params.graph_smoothness * (W[i, j] * W[i, j])
        return lp

    def log_prior_r(self, root_probs: np.ndarray) -> float:
        return float(np.sum(np.log(np.clip(root_probs, EPS, None))))

    def joint_log_prob(
        self,
        root_idx: int,
        a_obs: np.ndarray,
        log_signal: np.ndarray,
        W: np.ndarray,
        W_frozen: np.ndarray,
        entity_types: Dict[str, str],
        entities: List[str],
        trace_edges: Dict[Tuple[str, str], float],
        root_prior: np.ndarray,
        fault_category: str = "generic",
    ) -> float:
        mu_r = self.propagation.propagate(
            root_idx, a_obs, W, entity_types, entities, fault_category
        )
        lp = 0.0
        lp += self.log_lik_obs(a_obs, mu_r)
        lp += self.log_lik_logs(log_signal, root_idx)
        lp += self.log_lik_traces(W, trace_edges, entities)
        lp += self.log_prior_W(W, W_frozen)
        lp += math.log(max(root_prior[root_idx], EPS))
        return float(lp)

    def joint_log_prob_all(
        self,
        a_obs: np.ndarray,
        log_signal: np.ndarray,
        W: np.ndarray,
        W_frozen: np.ndarray,
        entity_types: Dict[str, str],
        entities: List[str],
        trace_edges: Dict[Tuple[str, str], float],
        root_prior: np.ndarray,
        fault_category: str = "generic",
    ) -> np.ndarray:
        n = len(entities)
        lp = np.zeros(n, dtype=float)
        for r in range(n):
            lp[r] = self.joint_log_prob(
                r,
                a_obs,
                log_signal,
                W,
                W_frozen,
                entity_types,
                entities,
                trace_edges,
                root_prior,
                fault_category,
            )
        return lp
