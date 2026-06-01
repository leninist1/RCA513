from __future__ import annotations

from prism_v3.noise_native.negative_controls import apply_negative_controls


def test_negative_control_prefers_specific_collapse_over_broad_explainer() -> None:
    profiles = {
        "root": {"collapse_gain": 0.8, "recovery_score": 0.4, "broad_explainer": 0.1},
        "hub": {"collapse_gain": 0.9, "recovery_score": 0.2, "broad_explainer": 0.9},
    }
    out = apply_negative_controls(profiles, ["root", "hub"])
    assert out["root"]["passes_negative_control"] is True
    assert out["hub"]["passes_negative_control"] is False
