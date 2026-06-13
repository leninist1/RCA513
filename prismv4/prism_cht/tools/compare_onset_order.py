"""Compare onset order across components via a TelemetryStore.

Returns only observed onset timestamps sorted by time — no ranking,
no winner, no root-cause judgement.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence, Tuple

from prismv4.prism_cht.tool_types import FactToolResult
from prismv4.prism_cht.telemetry_store import TelemetryStore


class CompareOnsetOrderTool:
    """Retrieve per-component, per-signal onset times."""

    name = "compare_onset_order"

    def execute(
        self,
        *,
        args: Mapping[str, Any],
        store: TelemetryStore,
    ) -> FactToolResult:
        component_scope = args.get("component_scope")
        signal_scope = args.get("signal_scope")
        time_window = args.get("time_window")

        if not isinstance(component_scope, Sequence) or isinstance(component_scope, str):
            raise ValueError(
                "component_scope must be a non-string sequence"
            )
        if len(component_scope) < 2:
            raise ValueError(
                "component_scope must contain at least two distinct components"
            )
        if len(set(component_scope)) != len(component_scope):
            raise ValueError("component_scope contains duplicate entries")

        if not isinstance(signal_scope, Sequence) or isinstance(signal_scope, str):
            raise ValueError(
                "signal_scope must be a non-string sequence"
            )
        if len(signal_scope) < 1:
            raise ValueError("signal_scope must contain at least one signal")

        if not isinstance(time_window, (list, tuple)) or len(time_window) != 2:
            raise ValueError(
                "time_window must be a sequence of [start, end]"
            )

        t0, t1 = float(time_window[0]), float(time_window[1])
        if t0 > t1:
            raise ValueError(f"time_window start ({t0}) must be <= end ({t1})")

        observations = store.get_onset_observations(
            component_scope=list(component_scope),
            signal_scope=list(signal_scope),
            time_window=(t0, t1),
        )

        observed_components = {o.component for o in observations}
        missing = [
            c for c in sorted(set(component_scope))
            if c not in observed_components
        ]

        observed_onsets = sorted(
            [
                {
                    "component": o.component,
                    "signal": o.signal,
                    "onset_time": o.onset_time,
                    "source": o.source,
                }
                for o in observations
            ],
            key=lambda x: (x["onset_time"], x["component"], x["signal"]),
        )

        return FactToolResult(
            modality="onset",
            component_scope=tuple(sorted(set(component_scope))),
            time_window=(t0, t1),
            observation={
                "observed_onsets": observed_onsets,
                "missing_components": missing,
            },
            provenance={
                "tool": "compare_onset_order",
                "store_type": type(store).__name__,
            },
        )
