"""Learned observation likelihood network.

Replaces the hardcoded Gaussians at prism.py:3463:
  p(delta_v | v=r) = N(0.80, 0.10²)
  p(delta_v | v≠r) = N(0.20, 0.20²)

with a network that predicts per-entity, per-system, per-fault-type
μ_root, σ_root, μ_nonroot, σ_nonroot based on:
  - Entity embedding
  - Graph position (in-degree, out-degree, page_rank)
  - System one-hot
  - Fault type one-hot
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import math
import numpy as np

EPS = 1e-9


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-min(max(x, -20.0), 20.0)))


def _softplus(x: float) -> float:
    return math.log(1.0 + math.exp(min(max(x, 0.0), 20.0)))


class LikelihoodNetwork:
    """Predicts counterfactual recovery distribution parameters.

    Supports two pretrained formats:
      - MLP weights (legacy): {"params": {"W1":..., "b1":..., ...}}
      - System-table (new):   {"systems": {"Bank": {"cpu": {...}, ...}, ...},
                                "default": {...}, "system_index": {...},
                                "subcat_index": {...}}

    Falls back to heuristic defaults when no training data loaded.
    """

    def __init__(
        self,
        input_dim: int = 80,
        hidden_dim: int = 64,
        load_pretrained: Optional[str] = None,
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self._is_trained = False
        self._params: Optional[Dict[str, np.ndarray]] = None
        self._system_params: Optional[Dict[str, dict]] = None
        self._system_index: Dict[str, int] = {}
        self._subcat_index: Dict[str, int] = {}
        self._default_params: Dict[str, float] = {
            "mu_root": 0.80,
            "sigma_root": 0.10,
            "mu_nonroot": 0.20,
            "sigma_nonroot": 0.20,
        }
        if load_pretrained:
            self._load(load_pretrained)

    def _encode_features(
        self,
        entity_embedding: np.ndarray,
        graph_features: np.ndarray,
        system_int: int = 0,
        fault_type_int: int = 0,
    ) -> np.ndarray:
        if len(entity_embedding) > 48:
            entity_embedding = entity_embedding[:48]
        elif len(entity_embedding) < 48:
            padded = np.zeros(48, dtype=float)
            padded[: len(entity_embedding)] = entity_embedding
            entity_embedding = padded
        if len(graph_features) > 16:
            graph_features = graph_features[:16]
        elif len(graph_features) < 16:
            padded = np.zeros(16, dtype=float)
            padded[: len(graph_features)] = graph_features
            graph_features = padded
        system_vec = np.zeros(3, dtype=float)
        if 0 <= system_int < 3:
            system_vec[system_int] = 1.0
        fault_vec = np.zeros(9, dtype=float)
        if 0 <= fault_type_int < 9:
            fault_vec[fault_type_int] = 1.0
        features = np.concatenate(
            [entity_embedding, graph_features, system_vec, fault_vec]
        )
        if len(features) > self.input_dim:
            features = features[: self.input_dim]
        elif len(features) < self.input_dim:
            padded = np.zeros(self.input_dim, dtype=float)
            padded[: len(features)] = features
            features = padded
        return features

    def _heuristic_params(
        self, features: np.ndarray
    ) -> Tuple[float, float, float, float]:
        return (0.80, 0.10, 0.20, 0.20)

    def predict(
        self,
        entity_embedding: np.ndarray,
        graph_features: np.ndarray,
        system_int: int = 0,
        fault_type_int: int = 0,
    ) -> Tuple[float, float, float, float]:
        features = self._encode_features(
            entity_embedding, graph_features, system_int, fault_type_int
        )
        if self._is_trained and self._system_params is not None:
            return self._system_lookup(system_int, fault_type_int)
        if self._is_trained and self._params is not None:
            return self._trained_forward(features)
        return self._heuristic_params(features)

    def _system_lookup(
        self, system_int: int, fault_type_int: int
    ) -> Tuple[float, float, float, float]:
        system_names = ["Bank", "Telecom", "Market"]
        system_name = (
            system_names[system_int] if 0 <= system_int < len(system_names) else "Bank"
        )
        subcat_names = [
            "cpu",
            "memory",
            "disk_io",
            "disk_space",
            "network_latency",
            "network_packet_loss",
            "network_fault",
            "jvm_oom",
            "process_kill",
        ]
        subcat_name = (
            subcat_names[fault_type_int]
            if 0 <= fault_type_int < len(subcat_names)
            else "cpu"
        )

        system_params = self._system_params.get(system_name, {})
        params = system_params.get(subcat_name, self._default_params)
        return (
            params.get("mu_root", 0.80),
            params.get("sigma_root", 0.10),
            params.get("mu_nonroot", 0.20),
            params.get("sigma_nonroot", 0.20),
        )

    def _trained_forward(
        self, features: np.ndarray
    ) -> Tuple[float, float, float, float]:
        if self._params is None:
            return self._heuristic_params(features)
        W1, b1, W2, b2 = (
            self._params["W1"],
            self._params["b1"],
            self._params["W2"],
            self._params["b2"],
        )
        h = np.tanh(W1 @ features + b1)
        out = W2 @ h + b2
        mu_root = _sigmoid(float(out[0]))
        sigma_root = _softplus(float(out[1])) + 0.02
        mu_nonroot = _sigmoid(float(out[2])) * 0.4
        sigma_nonroot = _softplus(float(out[3])) + 0.05
        return (mu_root, sigma_root, mu_nonroot, sigma_nonroot)

    def log_likelihood(
        self,
        delta_v: float,
        is_root: bool,
        entity_embedding: np.ndarray,
        graph_features: np.ndarray,
        system_int: int = 0,
        fault_type_int: int = 0,
    ) -> float:
        mu_root, sigma_root, mu_nonroot, sigma_nonroot = self.predict(
            entity_embedding, graph_features, system_int, fault_type_int
        )
        mu = mu_root if is_root else mu_nonroot
        sigma = sigma_root if is_root else sigma_nonroot
        var = sigma * sigma
        log_p = -0.5 * math.log(2.0 * math.pi * max(var, EPS)) - 0.5 * (
            (delta_v - mu) ** 2
        ) / max(var, EPS)
        return float(log_p)

    def _load(self, path: str):
        import pickle as _pickle

        try:
            with open(path, "rb") as f:
                data = _pickle.load(f)
            if isinstance(data, dict) and "systems" in data:
                self._system_params = data["systems"]
                self._system_index = data.get("system_index", {})
                self._subcat_index = data.get("subcat_index", {})
                self._default_params = data.get("default", self._default_params)
                self._is_trained = bool(self._system_params)
            else:
                self._params = (
                    data.get("params", None) if isinstance(data, dict) else None
                )
                self._is_trained = self._params is not None
        except (FileNotFoundError, Exception):
            self._is_trained = False

    def save(self, path: str):
        import pickle as _pickle

        if self._system_params is not None:
            data = {
                "systems": self._system_params,
                "default": self._default_params,
                "system_index": self._system_index,
                "subcat_index": self._subcat_index,
            }
        else:
            data = {"params": self._params}
        with open(path, "wb") as f:
            _pickle.dump(data, f)
