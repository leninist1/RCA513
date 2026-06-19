"""Inspect local telemetry signatures for proposed reason families."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from prismv4.prism_cht.telemetry_store import TelemetryStore
from prismv4.prism_cht.tool_types import FactToolResult


class InspectReasonSignatureTool:
    """Retrieve per-component reason-family observations."""

    name = "inspect_reason_signature"

    def execute(
        self,
        *,
        args: Mapping[str, Any],
        store: TelemetryStore,
    ) -> FactToolResult:
        component_scope = args.get("component_scope")
        reason_by_component = args.get("reason_by_component", {})
        time_window = args.get("time_window")

        if not isinstance(component_scope, Sequence) or isinstance(component_scope, str):
            raise ValueError("component_scope must be a non-string sequence")
        if len(component_scope) < 2:
            raise ValueError("component_scope must contain at least two components")
        if not isinstance(reason_by_component, Mapping):
            raise ValueError("reason_by_component must be a mapping")
        if not isinstance(time_window, (list, tuple)) or len(time_window) != 2:
            raise ValueError("time_window must be a sequence of [start, end]")

        t0, t1 = float(time_window[0]), float(time_window[1])
        if t0 > t1:
            raise ValueError(f"time_window start ({t0}) must be <= end ({t1})")

        if not hasattr(store, "inspect_reason_signatures"):
            signatures = [
                {
                    "component": str(component),
                    "reason_family": str(reason_by_component.get(component, "unknown")),
                    "magnitude": 0.0,
                    "signals": [],
                    "note": "store does not expose reason signatures",
                }
                for component in component_scope
            ]
        else:
            signatures = list(
                store.inspect_reason_signatures(
                    component_scope=list(component_scope),
                    reason_by_component=dict(reason_by_component),
                    time_window=(t0, t1),
                )
            )

        return FactToolResult(
            modality="reason_signature",
            component_scope=tuple(sorted(set(component_scope))),
            time_window=(t0, t1),
            observation={"reason_signatures": signatures},
            provenance={
                "tool": "inspect_reason_signature",
                "store_type": type(store).__name__,
            },
        )
