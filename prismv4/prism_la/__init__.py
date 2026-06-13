"""PRISM-LA: LLM-as-Investigator Root Cause Agent.

LLM does not directly guess answers. It builds hypotheses, selects tools,
compares evidence, and resolves conflicts. All evidence comes from the
current case's metrics/logs/traces via structured tool calls.
"""

from .config import PRISMLAConfig, DEFAULT_CONFIG
from .state import CaseState, EventHypothesis, EvidenceLedger, HypothesisStatus, InvestigationPlan, PRISMLAState
from .controller import PRISMLAController
from .sandbox import EvidenceSandbox
from .tools import PRISMLAToolbox
from .synthesis import synthesize_answer

__all__ = [
    "PRISMLAConfig",
    "DEFAULT_CONFIG",
    "CaseState",
    "EventHypothesis",
    "EvidenceLedger",
    "HypothesisStatus",
    "InvestigationPlan",
    "PRISMLAState",
    "PRISMLAController",
    "EvidenceSandbox",
    "PRISMLAToolbox",
    "synthesize_answer",
]
