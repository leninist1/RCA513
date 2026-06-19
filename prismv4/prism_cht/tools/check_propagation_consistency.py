"""Inspect observed propagation consistency without causal intervention."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from prismv4.prism_cht.telemetry_store import TelemetryStore
from prismv4.prism_cht.tool_types import FactToolResult


class CheckPropagationConsistencyTool:
    """Return observed path/onset consistency facts for candidate pairs."""

    name = "check_propagation_consistency"

    def execute(
        self,
        *,
        args: Mapping[str, Any],
        store: TelemetryStore,
    ) -> FactToolResult:
        source_component = args.get("source_component")
        symptom_components = args.get("symptom_components")
        time_window = args.get("time_window")

        if not source_component or not str(source_component).strip():
            raise ValueError("source_component must be non-empty")
        if not isinstance(symptom_components, Sequence) or isinstance(symptom_components, str):
            raise ValueError("symptom_components must be a non-string sequence")
        if len(symptom_components) < 1:
            raise ValueError("symptom_components must contain at least one component")
        if not isinstance(time_window, (list, tuple)) or len(time_window) != 2:
            raise ValueError("time_window must be a sequence of [start, end]")

        t0, t1 = float(time_window[0]), float(time_window[1])
        if t0 > t1:
            raise ValueError(f"time_window start ({t0}) must be <= end ({t1})")

        facts = []
        for symptom in symptom_components:
            paths = store.find_trace_paths(
                source_component=str(source_component),
                target_component=str(symptom),
                time_window=(t0, t1),
                max_hops=6,
                max_paths=5,
            )
            facts.append(
                {
                    "source_component": str(source_component),
                    "symptom_component": str(symptom),
                    "path_count": len(paths),
                    "has_observed_path": bool(paths),
                }
            )

        scope = tuple(sorted({str(source_component), *[str(s) for s in symptom_components]}))
        return FactToolResult(
            modality="propagation",
            component_scope=scope,
            time_window=(t0, t1),
            observation={"propagation_facts": facts},
            provenance={
                "tool": "check_propagation_consistency",
                "store_type": type(store).__name__,
            },
        )
