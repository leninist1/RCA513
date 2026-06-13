"""Inspect trace paths between two components.

Returns only explicitly stored paths — no hidden-edge inference,
no propagation scoring, no root-cause judgement.
"""

from __future__ import annotations

from typing import Any, Mapping

from prismv4.prism_cht.tool_types import FactToolResult
from prismv4.prism_cht.telemetry_store import TelemetryStore


class InspectTracePathTool:
    """Retrieve trace paths between a source and a target component."""

    name = "inspect_trace_path"

    def execute(
        self,
        *,
        args: Mapping[str, Any],
        store: TelemetryStore,
    ) -> FactToolResult:
        source_component = args.get("source_component")
        target_component = args.get("target_component")
        time_window = args.get("time_window")
        max_hops = args.get("max_hops", 4)
        max_paths = args.get("max_paths", 10)

        if not source_component or not str(source_component).strip():
            raise ValueError("source_component must be non-empty")
        if not target_component or not str(target_component).strip():
            raise ValueError("target_component must be non-empty")

        source = str(source_component)
        target = str(target_component)

        if not isinstance(max_hops, int) or max_hops <= 0:
            raise ValueError(f"max_hops must be a positive int, got {max_hops}")
        if not isinstance(max_paths, int) or max_paths <= 0:
            raise ValueError(f"max_paths must be a positive int, got {max_paths}")

        if not isinstance(time_window, (list, tuple)) or len(time_window) != 2:
            raise ValueError(
                "time_window must be a sequence of [start, end]"
            )

        t0, t1 = float(time_window[0]), float(time_window[1])
        if t0 > t1:
            raise ValueError(f"time_window start ({t0}) must be <= end ({t1})")

        paths = store.find_trace_paths(
            source_component=source,
            target_component=target,
            time_window=(t0, t1),
            max_hops=max_hops,
            max_paths=max_paths,
        )

        paths_data = []
        for path in paths:
            hops = [
                {
                    "source_component": h.source_component,
                    "target_component": h.target_component,
                    "timestamp": h.timestamp,
                    "latency_ms": h.latency_ms,
                    "status": h.status,
                }
                for h in path.hops
            ]
            paths_data.append({"hops": hops})

        return FactToolResult(
            modality="trace",
            component_scope=(source, target),
            time_window=(t0, t1),
            observation={"paths": paths_data},
            provenance={
                "tool": "inspect_trace_path",
                "store_type": type(store).__name__,
            },
        )
