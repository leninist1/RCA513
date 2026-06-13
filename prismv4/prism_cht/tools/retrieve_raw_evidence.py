"""Retrieve raw telemetry records by modality, components, and time window.

Returns only raw records — no summarisation, no ranking, no scores.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from prismv4.prism_cht.tool_types import FactToolResult
from prismv4.prism_cht.telemetry_store import TelemetryStore


class RetrieveRawEvidenceTool:
    """Fetch raw telemetry records matching modality and component scope."""

    name = "retrieve_raw_evidence"

    def execute(
        self,
        *,
        args: Mapping[str, Any],
        store: TelemetryStore,
    ) -> FactToolResult:
        modality = args.get("modality")
        component_scope = args.get("component_scope")
        time_window = args.get("time_window")
        limit = args.get("limit", 20)

        if not modality or not str(modality).strip():
            raise ValueError("modality must be non-empty")
        modality_str = str(modality)

        if not isinstance(component_scope, Sequence) or isinstance(component_scope, str):
            raise ValueError(
                "component_scope must be a non-string sequence"
            )
        if len(component_scope) < 1:
            raise ValueError("component_scope must contain at least one component")

        if not isinstance(limit, int) or limit <= 0:
            raise ValueError(f"limit must be a positive int, got {limit}")
        if limit > 50:
            raise ValueError(
                f"limit must be at most 50, got {limit}"
            )

        if not isinstance(time_window, (list, tuple)) or len(time_window) != 2:
            raise ValueError(
                "time_window must be a sequence of [start, end]"
            )

        t0, t1 = float(time_window[0]), float(time_window[1])
        if t0 > t1:
            raise ValueError(f"time_window start ({t0}) must be <= end ({t1})")

        records = store.retrieve_records(
            modality=modality_str,
            component_scope=list(component_scope),
            time_window=(t0, t1),
            limit=limit,
        )

        # Convert sealed (MappingProxyType) records to plain dicts for
        # the observation payload
        records_data = [dict(r) for r in records]

        return FactToolResult(
            modality=modality_str,
            component_scope=tuple(sorted(set(component_scope))),
            time_window=(t0, t1),
            observation={"records": records_data},
            provenance={
                "tool": "retrieve_raw_evidence",
                "store_type": type(store).__name__,
            },
        )
