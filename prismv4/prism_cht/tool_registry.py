"""Tool registry for PRISM-CHT.

Holds registered fact-only tools and dispatches execution.  The
registry does not gate action admission, manage evidence graphs, or
compute scores.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping

from .tool_types import FactTool, FactToolResult, validate_fact_only_payload
from .telemetry_store import TelemetryStore
from .tools.compare_onset_order import CompareOnsetOrderTool
from .tools.inspect_trace_path import InspectTracePathTool
from .tools.retrieve_raw_evidence import RetrieveRawEvidenceTool


class ToolRegistry:
    """Named registry of ``FactTool`` instances."""

    def __init__(self) -> None:
        self._tools: Dict[str, FactTool] = {}

    def register(self, tool: FactTool) -> None:
        """Register *tool* under its ``name`` attribute.

        Raises ValueError if a tool with the same name is already
        registered.
        """
        name = tool.name
        if name in self._tools:
            raise ValueError(
                f"Tool '{name}' is already registered"
            )
        self._tools[name] = tool

    def has(self, tool_name: str) -> bool:
        """Return True if *tool_name* is registered."""
        return tool_name in self._tools

    def get(self, tool_name: str) -> FactTool:
        """Return the registered ``FactTool`` for *tool_name*.

        Raises ValueError if the tool is not registered.
        """
        if tool_name not in self._tools:
            raise ValueError(
                f"Tool '{tool_name}' is not registered"
            )
        return self._tools[tool_name]

    def execute(
        self,
        *,
        tool_name: str,
        args: Mapping[str, Any],
        store: TelemetryStore,
    ) -> FactToolResult:
        """Execute the named tool and re-validate the result.

        ``validate_fact_only_payload`` is called on the observation
        and provenance of the returned ``FactToolResult`` as a
        secondary guard.
        """
        tool = self.get(tool_name)
        result = tool.execute(args=args, store=store)

        # Secondary validation guard
        validate_fact_only_payload(result.observation)
        validate_fact_only_payload(result.provenance)

        return result


def build_default_tool_registry() -> ToolRegistry:
    """Return a ``ToolRegistry`` pre-loaded with the three default tools.

    * ``compare_onset_order``
    * ``inspect_trace_path``
    * ``retrieve_raw_evidence``
    """
    registry = ToolRegistry()
    registry.register(CompareOnsetOrderTool())
    registry.register(InspectTracePathTool())
    registry.register(RetrieveRawEvidenceTool())
    return registry
