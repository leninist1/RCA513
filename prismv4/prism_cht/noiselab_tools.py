"""NoiseLab fact-only tools for NoiseNative LLM agents.

These tools expose NoiseLab as a feature/evidence extraction layer.
They deliberately do not produce root-cause decisions, rankings,
posteriors, or final component scores.  Their output is suitable for
LLM context and downstream deterministic gating.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Mapping, Sequence

from .canonical import canonicalize_json_value
from .telemetry_store import TelemetryStore
from .tool_types import FactToolResult


_RESERVED_FACT_KEYS = {
    "against",
    "against_score",
    "component_score",
    "confidence",
    "confidence_score",
    "likelihood",
    "net_score",
    "posterior",
    "probability",
    "rank",
    "ranking",
    "root_cause",
    "root_score",
    "score",
    "support",
    "support_score",
    "verdict",
    "winner",
}


def _component_scope(args: Mapping[str, Any]) -> tuple[str, ...]:
    raw = args.get("component_scope")
    if not isinstance(raw, (list, tuple)) or not raw:
        raise ValueError("component_scope must be a non-empty array")
    values = tuple(str(item).strip() for item in raw if str(item).strip())
    if not values:
        raise ValueError("component_scope must contain at least one component")
    return values


def _single_component(args: Mapping[str, Any], *, field: str) -> str:
    value = str(args.get(field, "")).strip()
    if not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _time_window(args: Mapping[str, Any]) -> tuple[float, float]:
    raw = args.get("time_window")
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        raise ValueError("time_window must be [start, end]")
    start = float(raw[0])
    end = float(raw[1])
    if start > end:
        raise ValueError("time_window start must be <= end")
    return (start, end)


def _string_tuple(args: Mapping[str, Any], *, field: str) -> tuple[str, ...]:
    raw = args.get(field, ())
    if raw is None:
        return ()
    if not isinstance(raw, (list, tuple)):
        raise ValueError(f"{field} must be an array when provided")
    return tuple(str(item).strip() for item in raw if str(item).strip())


def _json_fact_value(value: Any, *, renamed: set[str]) -> Any:
    if dataclasses.is_dataclass(value):
        value = dataclasses.asdict(value)

    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            key_str = str(key)
            norm = key_str.strip().lower()
            safe_key = key_str
            if norm in _RESERVED_FACT_KEYS:
                safe_key = f"feature_{norm}"
                renamed.add(f"{key_str}->{safe_key}")
            result[safe_key] = _json_fact_value(item, renamed=renamed)
        return result

    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_fact_value(item, renamed=renamed) for item in value]

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    return repr(value)


def _canonical_fact_payload(value: Any) -> tuple[Any, tuple[str, ...]]:
    renamed: set[str] = set()
    safe = _json_fact_value(value, renamed=renamed)
    return canonicalize_json_value(safe), tuple(sorted(renamed))


def _call_store_method(
    store: TelemetryStore,
    method_name: str,
    **kwargs: Any,
) -> tuple[Any | None, bool]:
    method = getattr(store, method_name, None)
    if not callable(method):
        return None, False
    return method(**kwargs), True


def _fallback_noise_records(
    *,
    store: TelemetryStore,
    component_scope: Sequence[str],
    time_window: tuple[float, float],
    limit: int = 128,
) -> tuple[Mapping[str, Any], ...]:
    retrieve = getattr(store, "retrieve_records", None)
    if not callable(retrieve):
        return ()
    return tuple(
        retrieve(
            modality="noise_lab_features",
            component_scope=component_scope,
            time_window=time_window,
            limit=limit,
        )
    )


class InspectNoiseFeaturesTool:
    """Return NoiseLab feature rows for a component/time scope."""

    name = "inspect_noise_features"

    def execute(
        self,
        *,
        args: Mapping[str, Any],
        store: TelemetryStore,
    ) -> FactToolResult:
        scope = _component_scope(args)
        window = _time_window(args)
        feature_groups = _string_tuple(args, field="feature_groups")

        payload, delegated = _call_store_method(
            store,
            "inspect_noise_features",
            component_scope=scope,
            time_window=window,
            feature_groups=feature_groups,
        )
        if not delegated:
            payload = {
                "feature_rows": list(
                    _fallback_noise_records(
                        store=store,
                        component_scope=scope,
                        time_window=window,
                    )
                )
            }

        observation, renamed = _canonical_fact_payload(payload)
        missing = () if observation else ("noise_lab_features",)
        return FactToolResult(
            modality="noise_features",
            component_scope=scope,
            time_window=window,
            observation={
                "feature_groups": list(feature_groups),
                "noise_lab_observation": observation,
            },
            provenance={
                "tool": self.name,
                "store_method": "inspect_noise_features" if delegated else "retrieve_records",
                "renamed_reserved_keys": list(renamed),
            },
            missing_fields=missing,
            reliability_note=(
                "Delegated to NoiseLab store method."
                if delegated
                else "Fallback used raw noise_lab_features records only."
            ),
        )


class CompareSourceSymptomTool:
    """Compare candidate source-vs-symptom feature patterns."""

    name = "compare_source_symptom"

    def execute(
        self,
        *,
        args: Mapping[str, Any],
        store: TelemetryStore,
    ) -> FactToolResult:
        scope = _component_scope(args)
        if len(set(scope)) < 2:
            raise ValueError("component_scope must contain at least two components")
        window = _time_window(args)

        payload, delegated = _call_store_method(
            store,
            "compare_source_symptom",
            component_scope=scope,
            time_window=window,
        )
        if not delegated:
            payload = {
                "compared_components": list(scope),
                "feature_rows": list(
                    _fallback_noise_records(
                        store=store,
                        component_scope=scope,
                        time_window=window,
                    )
                ),
            }

        observation, renamed = _canonical_fact_payload(payload)
        return FactToolResult(
            modality="source_symptom_comparison",
            component_scope=scope,
            time_window=window,
            observation={"comparison": observation},
            provenance={
                "tool": self.name,
                "store_method": "compare_source_symptom" if delegated else "retrieve_records",
                "renamed_reserved_keys": list(renamed),
            },
            reliability_note=(
                "Delegated to NoiseLab source/symptom comparator."
                if delegated
                else "Fallback exposes comparable feature rows, not a role judgement."
            ),
        )


class CounterfactualRemoveTool:
    """Expose NoiseLab counterfactual facts for removing one component."""

    name = "counterfactual_remove"

    def execute(
        self,
        *,
        args: Mapping[str, Any],
        store: TelemetryStore,
    ) -> FactToolResult:
        component = _single_component(args, field="component")
        symptoms = _string_tuple(args, field="symptom_components")
        window = _time_window(args)
        scope = tuple(sorted(set((component, *symptoms))))

        payload, delegated = _call_store_method(
            store,
            "counterfactual_remove",
            component=component,
            symptom_components=symptoms,
            time_window=window,
        )
        if not delegated:
            payload = {
                "removed_component": component,
                "symptom_components": list(symptoms),
                "available_feature_rows": list(
                    _fallback_noise_records(
                        store=store,
                        component_scope=scope,
                        time_window=window,
                    )
                ),
            }

        observation, renamed = _canonical_fact_payload(payload)
        return FactToolResult(
            modality="counterfactual_removal",
            component_scope=scope,
            time_window=window,
            observation={"counterfactual": observation},
            provenance={
                "tool": self.name,
                "store_method": "counterfactual_remove" if delegated else "retrieve_records",
                "renamed_reserved_keys": list(renamed),
            },
            reliability_note=(
                "Delegated to NoiseLab counterfactual method."
                if delegated
                else "Fallback has feature rows only; no counterfactual simulation was available."
            ),
        )


class TestDownstreamExplanationTool:
    """Check which downstream symptoms are explained by a candidate."""

    name = "test_downstream_explanation"

    def execute(
        self,
        *,
        args: Mapping[str, Any],
        store: TelemetryStore,
    ) -> FactToolResult:
        candidate = _single_component(args, field="candidate_component")
        symptoms = _string_tuple(args, field="symptom_components")
        if not symptoms:
            raise ValueError("symptom_components must contain at least one component")
        window = _time_window(args)
        scope = tuple(sorted(set((candidate, *symptoms))))

        payload, delegated = _call_store_method(
            store,
            "test_downstream_explanation",
            candidate_component=candidate,
            symptom_components=symptoms,
            time_window=window,
        )
        if not delegated:
            payload = {
                "candidate_component": candidate,
                "symptom_components": list(symptoms),
                "available_feature_rows": list(
                    _fallback_noise_records(
                        store=store,
                        component_scope=scope,
                        time_window=window,
                    )
                ),
            }

        observation, renamed = _canonical_fact_payload(payload)
        return FactToolResult(
            modality="downstream_explanation",
            component_scope=scope,
            time_window=window,
            observation={"downstream_explanation": observation},
            provenance={
                "tool": self.name,
                "store_method": "test_downstream_explanation" if delegated else "retrieve_records",
                "renamed_reserved_keys": list(renamed),
            },
            reliability_note=(
                "Delegated to NoiseLab downstream explanation method."
                if delegated
                else "Fallback exposes feature rows only; no explanation test was available."
            ),
        )
