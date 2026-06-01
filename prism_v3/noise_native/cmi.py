"""Online causal mechanism intervention features for noise-native PRISM."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple
import math

import numpy as np
import pandas as pd


EPS = 1e-9


def build_cmi_profiles(
    entities: Sequence[str],
    baseline_df: pd.DataFrame,
    fault_df: pd.DataFrame,
    graph: np.ndarray,
    candidate_entities: Sequence[str],
    max_conditioners: int = 6,
    max_effect_scope: int = 10,
) -> Dict[str, Dict[str, Any]]:
    """Compute no-GT CMI profiles for the current event candidate pool."""
    entity_list = [str(entity) for entity in entities]
    entity_index = {entity: idx for idx, entity in enumerate(entity_list)}
    candidates = [entity for entity in candidate_entities if entity in entity_index]
    if not candidates:
        return {}
    signal, baseline_mask, fault_mask = _build_signal(
        baseline_df=baseline_df,
        fault_df=fault_df,
        entities=entity_list,
    )
    if signal.empty or not np.any(baseline_mask) or not np.any(fault_mask):
        return {}
    graph_arr = np.asarray(graph, dtype=float)
    if graph_arr.shape != (len(entity_list), len(entity_list)):
        graph_arr = np.zeros((len(entity_list), len(entity_list)), dtype=float)

    pair_cache: Dict[Tuple[str, str], Dict[str, float]] = {}
    raw: Dict[str, Dict[str, Any]] = {}
    for candidate in candidates:
        conditioners = _conditioners(candidate, entity_list, entity_index, graph_arr, max_conditioners)
        cond = _conditional_residual(signal, candidate, conditioners, baseline_mask, fault_mask)
        y = _series(signal, candidate)
        marginal_delta_z = _marginal_delta_z(y, baseline_mask, fault_mask)
        marginal_fault_energy = _mean_abs(y[fault_mask])
        parent_explainability = _clip(
            (marginal_delta_z - cond["conditional_residual_z"]) / max(marginal_delta_z, 1e-6),
            0.0,
            1.0,
        )
        effect_scope = _effect_scope(candidate, entity_list, entity_index, graph_arr, max_effect_scope)
        effect = _intervention_effect_features(
            signal=signal,
            source=candidate,
            effect_scope=effect_scope,
            candidate_entities=candidates,
            baseline_mask=baseline_mask,
            fault_mask=fault_mask,
            pair_cache=pair_cache,
        )
        raw[candidate] = {
            "candidate": candidate,
            "conditioners": conditioners,
            "effect_scope": [name for name, _weight in effect_scope],
            "marginal_delta_z": float(marginal_delta_z),
            "marginal_fault_energy": float(marginal_fault_energy),
            "conditional_residual_z": float(cond["conditional_residual_z"]),
            "conditional_fault_residual_energy": float(cond["fault_residual_energy"]),
            "conditional_base_residual_std": float(cond["base_residual_std"]),
            "conditional_model_r2": float(cond["baseline_r2"]),
            "parent_explainability": float(parent_explainability),
            **effect,
        }

    _normalize_and_score(raw)
    return raw


def aggregate_anchor_marginalized_cmi(
    profiles_by_anchor: Sequence[Tuple[float, Dict[str, Dict[str, Any]]]],
    *,
    stability_by_entity: Optional[Dict[str, float]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Aggregate CMI profiles across safe anchor hypotheses.

    This implements the roadmap's robust score:
    0.60 * expected_CMI + 0.25 * worst_case_CMI + 0.15 * anchor_stability.
    """
    stability_by_entity = stability_by_entity or {}
    totals: Dict[str, List[Tuple[float, Dict[str, Any]]]] = {}
    weight_total = sum(max(0.0, float(weight)) for weight, _profiles in profiles_by_anchor)
    if weight_total <= 0:
        weight_total = 1.0
    for weight, profiles in profiles_by_anchor:
        norm_weight = max(0.0, float(weight)) / weight_total
        for entity, profile in profiles.items():
            totals.setdefault(str(entity), []).append((norm_weight, dict(profile)))
    out: Dict[str, Dict[str, Any]] = {}
    for entity, weighted_profiles in totals.items():
        scores = [float(profile.get("cmi_score", profile.get("root_admissibility", 0.0)) or 0.0) for _w, profile in weighted_profiles]
        expected = sum(weight * score for (weight, _profile), score in zip(weighted_profiles, scores))
        worst = min(scores) if scores else 0.0
        stability = float(stability_by_entity.get(entity, 0.0))
        robust = 0.60 * expected + 0.25 * worst + 0.15 * stability
        best_profile = max((profile for _weight, profile in weighted_profiles), key=lambda item: float(item.get("cmi_score", 0.0)))
        out[entity] = {
            **best_profile,
            "robust_cmi_score": float(robust),
            "cmi_by_anchor": scores,
            "cmi_variance": float(np.var(scores)) if scores else 0.0,
            "cmi_stability": stability,
        }
    return out


def _build_signal(
    baseline_df: pd.DataFrame,
    fault_df: pd.DataFrame,
    entities: Sequence[str],
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    required = {"timestamp", "entity", "metric_name", "value"}
    if baseline_df is None or fault_df is None:
        return pd.DataFrame(), np.array([], dtype=bool), np.array([], dtype=bool)
    if baseline_df.empty or fault_df.empty:
        return pd.DataFrame(), np.array([], dtype=bool), np.array([], dtype=bool)
    if not required.issubset(baseline_df.columns) or not required.issubset(fault_df.columns):
        return pd.DataFrame(), np.array([], dtype=bool), np.array([], dtype=bool)

    base = baseline_df.loc[:, ["timestamp", "entity", "metric_name", "value"]].copy()
    fault = fault_df.loc[:, ["timestamp", "entity", "metric_name", "value"]].copy()
    base["phase"] = "baseline"
    fault["phase"] = "fault"
    stats = (
        base.groupby(["entity", "metric_name"])["value"]
        .agg(["mean", "std"])
        .reset_index()
    )
    frame = pd.concat([base, fault], ignore_index=True)
    frame["entity"] = frame["entity"].astype(str)
    frame = frame[frame["entity"].isin(set(entities))]
    if frame.empty:
        return pd.DataFrame(), np.array([], dtype=bool), np.array([], dtype=bool)
    frame = frame.merge(stats, on=["entity", "metric_name"], how="left").dropna(subset=["mean"])
    if frame.empty:
        return pd.DataFrame(), np.array([], dtype=bool), np.array([], dtype=bool)
    values = pd.to_numeric(frame["value"], errors="coerce").to_numpy(dtype=float)
    means = pd.to_numeric(frame["mean"], errors="coerce").to_numpy(dtype=float)
    stds = pd.to_numeric(frame["std"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    denom = np.maximum(stds, np.maximum(np.abs(means) * 0.05, 1e-6))
    frame["z"] = np.clip((values - means) / denom, -20.0, 20.0)
    frame["abs_z"] = np.abs(frame["z"])
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=["z", "abs_z"])
    if frame.empty:
        return pd.DataFrame(), np.array([], dtype=bool), np.array([], dtype=bool)
    idx = frame.groupby(["phase", "timestamp", "entity"])["abs_z"].idxmax()
    dominant = frame.loc[idx, ["phase", "timestamp", "entity", "z"]].copy()
    dominant["row_time"] = dominant["phase"].map({"baseline": 0, "fault": 1}).astype(int) * 10**16 + dominant["timestamp"].astype(float)
    signal = dominant.pivot_table(index=["row_time", "phase", "timestamp"], columns="entity", values="z", aggfunc="mean")
    signal = signal.sort_index().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    phases = np.array([idx_tuple[1] for idx_tuple in signal.index], dtype=object)
    baseline_mask = phases == "baseline"
    fault_mask = phases == "fault"
    signal.index = range(len(signal))
    for entity in entities:
        if entity not in signal.columns:
            signal[entity] = 0.0
    return signal.loc[:, list(entities)], baseline_mask, fault_mask


def _conditioners(
    candidate: str,
    entities: Sequence[str],
    index: Dict[str, int],
    graph: np.ndarray,
    max_conditioners: int,
) -> List[str]:
    idx = index[candidate]
    weights: Dict[str, float] = {}
    for child_idx in np.where(graph[idx, :] > 0)[0]:
        child = entities[int(child_idx)]
        if child != candidate:
            weights[child] = max(weights.get(child, 0.0), 1.0 + min(1.0, float(graph[idx, child_idx])))
    for parent_idx in np.where(graph[:, idx] > 0)[0]:
        parent = entities[int(parent_idx)]
        if parent != candidate:
            weights[parent] = max(weights.get(parent, 0.0), 0.45 + min(0.5, float(graph[parent_idx, idx])))
    ranked = sorted(weights, key=lambda item: weights[item], reverse=True)
    return ranked[:max_conditioners]


def _effect_scope(
    candidate: str,
    entities: Sequence[str],
    index: Dict[str, int],
    graph: np.ndarray,
    max_scope: int,
) -> List[Tuple[str, float]]:
    idx = index[candidate]
    weights: Dict[str, float] = {}
    for parent_idx in np.where(graph[:, idx] > 0)[0]:
        parent = entities[int(parent_idx)]
        if parent != candidate:
            weights[parent] = max(weights.get(parent, 0.0), 1.0 + min(1.0, float(graph[parent_idx, idx])))
            for grand_idx in np.where(graph[:, parent_idx] > 0)[0]:
                grand = entities[int(grand_idx)]
                if grand != candidate:
                    weights[grand] = max(weights.get(grand, 0.0), 0.45 + 0.25 * min(1.0, float(graph[grand_idx, parent_idx])))
    for child_idx in np.where(graph[idx, :] > 0)[0]:
        child = entities[int(child_idx)]
        if child != candidate:
            weights[child] = max(weights.get(child, 0.0), 0.55 + 0.35 * min(1.0, float(graph[idx, child_idx])))
    ranked = sorted(weights.items(), key=lambda item: item[1], reverse=True)
    return ranked[:max_scope]


def _conditional_residual(
    signal: pd.DataFrame,
    target: str,
    conditioners: Sequence[str],
    baseline_mask: np.ndarray,
    fault_mask: np.ndarray,
) -> Dict[str, float]:
    y = _series(signal, target)
    cols = [name for name in conditioners if name in signal.columns and name != target]
    X = np.column_stack([_series(signal, name) for name in cols]) if cols else np.zeros((len(y), 0), dtype=float)
    lag_y = np.roll(y, 1)
    lag_y[0] = 0.0
    X = np.column_stack([X, lag_y])
    beta = _fit_ridge(X[baseline_mask], y[baseline_mask], alpha=1.0)
    if beta is None:
        residual = y - _safe_mean(y[baseline_mask])
        base_resid = residual[baseline_mask]
        fault_resid = residual[fault_mask]
        return {
            "conditional_residual_z": _marginal_delta_z(residual, baseline_mask, fault_mask),
            "fault_residual_energy": _mean_abs(fault_resid),
            "base_residual_std": float(np.std(base_resid)) if base_resid.size else 0.0,
            "baseline_r2": 0.0,
        }
    pred = _predict_ridge(beta, X)
    residual = y - pred
    base_resid = residual[baseline_mask]
    fault_resid = residual[fault_mask]
    denom = max(float(np.std(base_resid)) if base_resid.size else 0.0, 1.0)
    cond_z = abs(_safe_mean(fault_resid) - _safe_mean(base_resid)) / denom
    y_base = y[baseline_mask]
    pred_base = pred[baseline_mask]
    sse = float(np.sum((y_base - pred_base) ** 2))
    sst = float(np.sum((y_base - _safe_mean(y_base)) ** 2))
    r2 = 1.0 - sse / max(sst, 1e-9)
    return {
        "conditional_residual_z": float(max(0.0, cond_z)),
        "fault_residual_energy": _mean_abs(fault_resid),
        "base_residual_std": float(np.std(base_resid)) if base_resid.size else 0.0,
        "baseline_r2": _clip(r2, -1.0, 1.0),
    }


def _intervention_effect_features(
    signal: pd.DataFrame,
    source: str,
    effect_scope: Sequence[Tuple[str, float]],
    candidate_entities: Sequence[str],
    baseline_mask: np.ndarray,
    fault_mask: np.ndarray,
    pair_cache: Dict[Tuple[str, str], Dict[str, float]],
) -> Dict[str, Any]:
    total_energy = 0.0
    own_explained = 0.0
    alt_explained = 0.0
    best_observer = ""
    best_amount = -1.0
    for target, weight in effect_scope:
        if target not in signal.columns:
            continue
        target_energy = _mean_abs(_series(signal, target)[fault_mask])
        if target_energy <= 1e-9:
            continue
        own = _pair_effect(signal, source, target, baseline_mask, fault_mask, pair_cache)
        alt_amount = 0.0
        for alt in candidate_entities:
            if alt == source or alt == target or alt not in signal.columns:
                continue
            alt_payload = _pair_effect(signal, alt, target, baseline_mask, fault_mask, pair_cache)
            alt_amount = max(alt_amount, alt_payload["explained_amount"])
        weighted_energy = float(weight) * target_energy
        own_amount = float(weight) * own["explained_amount"]
        total_energy += weighted_energy
        own_explained += own_amount
        alt_explained += float(weight) * alt_amount
        if own_amount > best_amount:
            best_amount = own_amount
            best_observer = target
    if total_energy <= 1e-9:
        return {
            "effect_observer_energy": 0.0,
            "counterfactual_effect_coverage": 0.0,
            "alternative_explainability": 0.0,
            "repair_uniqueness": 0.0,
            "best_effect_observer": "",
        }
    coverage = _clip(own_explained / total_energy, 0.0, 1.0)
    alternative = _clip(alt_explained / total_energy, 0.0, 1.0)
    relative_advantage = own_explained / max(own_explained + alt_explained, 1e-9)
    return {
        "effect_observer_energy": float(total_energy),
        "counterfactual_effect_coverage": coverage,
        "alternative_explainability": alternative,
        "repair_uniqueness": _clip(coverage * relative_advantage, 0.0, 1.0),
        "best_effect_observer": best_observer,
    }


def _pair_effect(
    signal: pd.DataFrame,
    source: str,
    target: str,
    baseline_mask: np.ndarray,
    fault_mask: np.ndarray,
    cache: Dict[Tuple[str, str], Dict[str, float]],
) -> Dict[str, float]:
    key = (source, target)
    if key in cache:
        return cache[key]
    if source not in signal.columns or target not in signal.columns or source == target:
        cache[key] = {"coverage": 0.0, "explained_amount": 0.0}
        return cache[key]
    x = _series(signal, source)
    y = _series(signal, target)
    lag_y = np.roll(y, 1)
    lag_y[0] = 0.0
    X = np.column_stack([x, lag_y])
    beta = _fit_ridge(X[baseline_mask], y[baseline_mask], alpha=1.0)
    if beta is None:
        cache[key] = {"coverage": 0.0, "explained_amount": 0.0}
        return cache[key]
    source_beta = float(beta[1])
    dx = _safe_mean(x[fault_mask]) - _safe_mean(x[baseline_mask])
    dy = _safe_mean(y[fault_mask]) - _safe_mean(y[baseline_mask])
    explained_delta = source_beta * dx
    if abs(dy) <= 1e-9 or explained_delta * dy <= 0:
        coverage = 0.0
    else:
        coverage = min(abs(explained_delta), abs(dy)) / max(abs(dy), 1.0)
    marginal = _marginal_delta_z(y, baseline_mask, fault_mask)
    cond = _conditional_residual(signal, target, [source], baseline_mask, fault_mask)
    conditioned_collapse = _clip(
        (marginal - cond["conditional_residual_z"]) / max(marginal, 1e-6),
        0.0,
        1.0,
    )
    coverage = max(coverage, conditioned_collapse)
    cache[key] = {
        "coverage": _clip(coverage, 0.0, 1.0),
        "explained_amount": float(max(0.0, coverage * _mean_abs(y[fault_mask]))),
    }
    return cache[key]


def _normalize_and_score(profiles: Dict[str, Dict[str, Any]]) -> None:
    if not profiles:
        return
    items = list(profiles.values())
    norm_cols = [
        "marginal_delta_z",
        "marginal_fault_energy",
        "conditional_residual_z",
        "conditional_fault_residual_energy",
        "counterfactual_effect_coverage",
        "repair_uniqueness",
    ]
    for col in norm_cols:
        values = np.array([float(item.get(col, 0.0) or 0.0) for item in items], dtype=float)
        norm = _minmax(values)
        for item, value in zip(items, norm):
            item[f"{col}_norm"] = float(value)
    cond_values = np.array([item["conditional_residual_z_norm"] for item in items], dtype=float)
    repair_values = np.array([item["repair_uniqueness_norm"] for item in items], dtype=float)
    cond_bar = max(float(np.median(cond_values)), 0.50)
    repair_bar = max(float(np.median(repair_values)), 0.30)
    for item in items:
        parent = _clip(float(item.get("parent_explainability", 0.0)), 0.0, 1.0)
        alt = _clip(float(item.get("alternative_explainability", 0.0)), 0.0, 1.0)
        cond = float(item.get("conditional_residual_z_norm", 0.0))
        repair = float(item.get("repair_uniqueness_norm", 0.0))
        effect = float(item.get("counterfactual_effect_coverage_norm", 0.0))
        r2 = _clip(float(item.get("conditional_model_r2", 0.0)), 0.0, 1.0)
        admissible = bool(cond >= cond_bar and repair >= repair_bar and parent <= 0.65 and alt <= 0.78)
        cmi_score = (
            0.44 * cond
            + 0.25 * repair
            + 0.16 * effect
            + 0.08 * r2
            - 0.24 * parent
            - 0.18 * alt
        )
        item["cmi_root_admissible"] = admissible
        item["cmi_score"] = float(cmi_score)
        item["mechanism_break_likelihood"] = _clip(
            1.35 * cond + 0.45 * float(admissible) - 0.45 * parent - 0.28 * alt,
            -1.0,
            1.6,
        )
        item["mechanism_parent_refutation"] = _clip(
            1.00 - 1.45 * parent - 0.45 * alt,
            -1.0,
            1.0,
        )
        item["intervention_uniqueness"] = _clip(
            1.10 * repair + 0.35 * effect - 0.35 * alt,
            -0.7,
            1.3,
        )


def _series(signal: pd.DataFrame, column: str) -> np.ndarray:
    if column not in signal.columns:
        return np.zeros(len(signal), dtype=float)
    return (
        pd.to_numeric(signal[column], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .to_numpy(dtype=float)
    )


def _fit_ridge(X: np.ndarray, y: np.ndarray, alpha: float) -> Optional[np.ndarray]:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    mask = np.isfinite(y)
    if X.size:
        mask &= np.all(np.isfinite(X), axis=1)
    X = X[mask]
    y = y[mask]
    if len(y) < max(4, X.shape[1] + 2):
        return None
    design = np.column_stack([np.ones(len(y)), X])
    penalty = alpha * np.eye(design.shape[1])
    penalty[0, 0] = 0.0
    try:
        return np.linalg.solve(design.T @ design + penalty, design.T @ y)
    except np.linalg.LinAlgError:
        try:
            return np.linalg.lstsq(design.T @ design + penalty, design.T @ y, rcond=None)[0]
        except np.linalg.LinAlgError:
            return None


def _predict_ridge(beta: np.ndarray, X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    return np.column_stack([np.ones(len(X)), X]) @ beta


def _marginal_delta_z(y: np.ndarray, baseline_mask: np.ndarray, fault_mask: np.ndarray) -> float:
    base = y[baseline_mask]
    fault = y[fault_mask]
    return abs(_safe_mean(fault) - _safe_mean(base)) / max(float(np.std(base)) if base.size else 0.0, 1.0)


def _safe_mean(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if values.size else 0.0


def _mean_abs(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.mean(np.abs(values))) if values.size else 0.0


def _minmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = np.where(np.isfinite(values), values, 0.0)
    if values.size == 0:
        return values
    lo = float(np.min(values))
    hi = float(np.max(values))
    if hi <= lo + 1e-12:
        return np.zeros_like(values, dtype=float)
    return (values - lo) / (hi - lo)


def _clip(value: float, low: float, high: float) -> float:
    return float(max(low, min(high, value)))
