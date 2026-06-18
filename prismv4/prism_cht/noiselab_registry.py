"""Tool registry builders for NoiseLab-backed NoiseNative agents."""

from __future__ import annotations

from .noiselab_tools import (
    CompareSourceSymptomTool,
    CounterfactualRemoveTool,
    InspectNoiseFeaturesTool,
    TestDownstreamExplanationTool,
)
from .tool_registry import ToolRegistry, build_default_tool_registry


NOISELAB_TOOL_NAMES = (
    "inspect_noise_features",
    "compare_source_symptom",
    "counterfactual_remove",
    "test_downstream_explanation",
)


def build_noiselab_tool_registry(
    *,
    include_default_tools: bool = True,
) -> ToolRegistry:
    """Return a registry loaded with NoiseLab fact tools.

    ``include_default_tools=True`` keeps the useful prism-cht trace/onset
    tools available while adding the NoiseLab-specific feature tools.
    The NoiseNative controller can still restrict the LLM allowlist to
    NoiseLab-only tools by passing ``allowed_tool_names``.
    """
    registry = build_default_tool_registry() if include_default_tools else ToolRegistry()
    registry.register(InspectNoiseFeaturesTool())
    registry.register(CompareSourceSymptomTool())
    registry.register(CounterfactualRemoveTool())
    registry.register(TestDownstreamExplanationTool())
    return registry
