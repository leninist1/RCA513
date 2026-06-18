"""Retrieve observed symptoms not covered by a proposed component scope."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from prismv4.prism_cht.telemetry_store import TelemetryStore
from prismv4.prism_cht.tool_types import FactToolResult


class FindUnexplainedSymptomsTool:
    """Return symptom observations outside the claimed component scope."""

    name = "find_unexplained_symptoms"

    def execute(
        self,
        *,
        args: Mapping[str, Any],
        store: TelemetryStore,
    ) -> FactToolResult:
        explained_components = args.get("explained_components")
        time_window = args.get("time_window")
        limit = args.get("limit", 20)

        if not isinstance(explained_components, Sequence) or isinstance(explained_components, str):
            raise ValueError("explained_components must be a non-string sequence")
        if not isinstance(limit, int) or limit <= 0:
            raise ValueError("limit must be a positive int")
        if not isinstance(time_window, (list, tuple)) or len(time_window) != 2:
            raise ValueError("time_window must be a sequence of [start, end]")

        t0, t1 = float(time_window[0]), float(time_window[1])
        if t0 > t1:
            raise ValueError(f"time_window start ({t0}) must be <= end ({t1})")

        if hasattr(store, "find_symptom_records"):
            symptoms = list(
                store.find_symptom_records(
                    explained_components=list(explained_components),
                    time_window=(t0, t1),
                    limit=limit,
                )
            )
        else:
            symptoms = []

        return FactToolResult(
            modality="symptom",
            component_scope=tuple(sorted(set(explained_components))),
            time_window=(t0, t1),
            observation={"unexplained_symptoms": symptoms},
            provenance={
                "tool": "find_unexplained_symptoms",
                "store_type": type(store).__name__,
            },
        )
