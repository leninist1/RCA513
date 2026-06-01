"""Counterfactual negative-control helpers."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence


def negative_control_advantage(
    profiles: Mapping[str, Mapping[str, Any]],
    candidate: str,
    alternatives: Sequence[str],
) -> Dict[str, Any]:
    """Compare one candidate's CF profile against alternative explainers."""
    own = profiles.get(candidate, {})
    own_gain = _score_profile(own)
    alt_scores = {alt: _score_profile(profiles.get(alt, {})) for alt in alternatives if alt != candidate}
    best_alt = max(alt_scores.values()) if alt_scores else 0.0
    advantage = float(own_gain - best_alt)
    return {
        "candidate": candidate,
        "candidate_score": own_gain,
        "best_alternative_score": best_alt,
        "negative_control_advantage": advantage,
        "passes_negative_control": advantage >= 0.0,
        "alternative_scores": alt_scores,
    }


def apply_negative_controls(
    profiles: Mapping[str, Mapping[str, Any]],
    candidates: Sequence[str],
) -> Dict[str, Dict[str, Any]]:
    return {
        candidate: negative_control_advantage(profiles, candidate, candidates)
        for candidate in candidates
    }


def _score_profile(profile: Mapping[str, Any]) -> float:
    collapse = float(profile.get("collapse_gain", profile.get("residual_collapse", 0.0)) or 0.0)
    recovery = float(profile.get("recovery_score", profile.get("downstream_recovery", 0.0)) or 0.0)
    broad = float(profile.get("broad_explainer", 0.0) or 0.0)
    hotspot = float(profile.get("hotspot_self_collapse", 0.0) or 0.0)
    return collapse + 0.5 * recovery - 0.7 * broad - 0.4 * hotspot
