"""PRISM-CHT: Causal Hypothesis Tournament skeleton.

Phase CHT-0 provides the minimal protocol for falsifiable causal
hypotheses, deduplicated evidence atoms, discriminative actions,
and a gate that rejects invalid or duplicate agent requests.

Phase CHT-1 adds the fact-only tool boundary and single-round
investigation executor.
"""

from .hypothesis import CausalHypothesis, HypothesisStatus
from .evidence_graph import EvidenceAtom, EvidenceGraph, build_query_signature
from .action_schema import DiscriminativeAction
from .action_gate import ActionGate, GateDecision
from .canonical import build_tool_call_signature, canonicalize_json_value, deep_freeze, to_dispatch_args
from .tool_types import FactToolResult, FactTool, validate_fact_only_payload
from .telemetry_store import TelemetryStore, MockTelemetryStore, OnsetObservation, TraceHop, TracePath
from .tool_registry import ToolRegistry, build_default_tool_registry
from .executor import ActionRejectedError, InvestigationExecutor

__all__ = [
    "CausalHypothesis",
    "HypothesisStatus",
    "EvidenceAtom",
    "EvidenceGraph",
    "build_query_signature",
    "DiscriminativeAction",
    "GateDecision",
    "ActionGate",
    "deep_freeze",
    "canonicalize_json_value",
    "build_tool_call_signature",
    "to_dispatch_args",
    "FactToolResult",
    "FactTool",
    "validate_fact_only_payload",
    "TelemetryStore",
    "MockTelemetryStore",
    "OnsetObservation",
    "TraceHop",
    "TracePath",
    "ToolRegistry",
    "build_default_tool_registry",
    "ActionRejectedError",
    "InvestigationExecutor",
]
