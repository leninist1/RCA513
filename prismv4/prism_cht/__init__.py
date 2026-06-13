"""PRISM-CHT: Causal Hypothesis Tournament.

Phase CHT-0 provides the minimal protocol for falsifiable causal
hypotheses, deduplicated evidence atoms, discriminative actions,
and a gate that rejects invalid or duplicate agent requests.

Phase CHT-1 adds the fact-only tool boundary and single-round
investigation executor.

Phase CHT-2 adds the scripted Lead tournament with grounded
evidence assessment and nomination guardrails.
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

# Phase CHT-2: tournament types
from .tournament_types import (
    EvidenceRelation,
    AssessmentOutcome,
    EvidenceLinkProposal,
    HypothesisStatusUpdate,
    EvidenceAssessment,
    LeadNomination,
    InvestigationAuditStep,
    HypothesisSnapshot,
    LeadTournamentSnapshot,
    LeadTournamentResult,
    build_snapshot,
)

# Phase CHT-2: assessment gate
from .assessment_gate import AssessmentRejectedError, EvidenceAssessmentGate

# Phase CHT-2: lead policy
from .lead_policy import (
    LeadPolicy,
    PolicyDecision,
    ScriptedInvestigationTurn,
    ScriptedLeadPolicy,
)

# Phase CHT-2: lead controller
from .lead_controller import (
    TournamentBudgetExhaustedError,
    NominationRejectedError,
    LeadTournamentController,
)

# Phase CHT-2: demo scenario
from .demo_scenario import build_demo_lead_tournament


__all__ = [
    # CHT-0 / CHT-1
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
    # CHT-2: tournament types
    "EvidenceRelation",
    "AssessmentOutcome",
    "EvidenceLinkProposal",
    "HypothesisStatusUpdate",
    "EvidenceAssessment",
    "LeadNomination",
    "InvestigationAuditStep",
    "HypothesisSnapshot",
    "LeadTournamentSnapshot",
    "LeadTournamentResult",
    "build_snapshot",
    # CHT-2: assessment gate
    "AssessmentRejectedError",
    "EvidenceAssessmentGate",
    # CHT-2: lead policy
    "LeadPolicy",
    "PolicyDecision",
    "ScriptedInvestigationTurn",
    "ScriptedLeadPolicy",
    # CHT-2: lead controller
    "TournamentBudgetExhaustedError",
    "NominationRejectedError",
    "LeadTournamentController",
    # CHT-2: demo
    "build_demo_lead_tournament",
]
