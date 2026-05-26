from dataclasses import dataclass
from typing import Dict

PARAM_BOUNDS = {
    "anomaly_z_threshold": (2.0, 5.0), "top_k": (5, 15), "propagation_weight": (0.05, 0.60),
    "recovery_gamma": (0.1, 0.9), "soft_cf_beta": (0.3, 0.95), "causal_window_seconds": (5, 30),
    "istio_error_weight": (1.0, 5.0), "memory_failures_weight": (1.0, 6.0),
    "cpu_usage_weight": (0.5, 3.0), "latency_p99_weight": (0.5, 3.0),
}

@dataclass
class RCAConfig:
    anomaly_z_threshold: float = 3.0; top_k: int = 7; propagation_weight: float = 0.25
    recovery_gamma: float = 0.5; soft_cf_beta: float = 0.7; causal_window_seconds: int = 15
    istio_error_weight: float = 2.5; memory_failures_weight: float = 3.0
    cpu_usage_weight: float = 1.5; latency_p99_weight: float = 1.5
    # 策略选择
    propagation_strategy: str = "downstream_1_hop"
    time_penalty_strategy: str = "linear_decay"
    fusion_strategy: str = "multiplicative"

    def update(self, mutations: Dict, inertia: float = 1.0) -> 'RCAConfig':
        new_config = RCAConfig()
        for fn in self.__dataclass_fields__:
            setattr(new_config, fn, getattr(self, fn))
        for param, target_val in mutations.items():
            if not hasattr(new_config, param): continue
            old_val = getattr(self, param)
            if isinstance(old_val, str):
                setattr(new_config, param, target_val)
            elif isinstance(old_val, bool):
                setattr(new_config, param, bool(target_val))
            else:
                blended = (1 - inertia) * old_val + inertia * target_val
                if param in PARAM_BOUNDS:
                    lo, hi = PARAM_BOUNDS[param]; blended = max(lo, min(hi, blended))
                if isinstance(old_val, int): blended = int(round(blended))
                setattr(new_config, param, blended)
        return new_config

    def to_dict(self) -> Dict:
        return {f: getattr(self, f) for f in self.__dataclass_fields__}

    def diff(self, base: 'RCAConfig') -> Dict:
        diffs = {}
        for f in self.__dataclass_fields__:
            o, n = getattr(base, f), getattr(self, f)
            if o != n: diffs[f] = {"from": o, "to": n}
        return diffs
