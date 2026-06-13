"""PRISM-CHT: Causal Hypothesis Tournament skeleton.

Phase CHT-0 provides the minimal protocol for falsifiable causal
hypotheses, deduplicated evidence atoms, discriminative actions,
and a gate that rejects invalid or duplicate agent requests.
"""

from .hypothesis import CausalHypothesis, HypothesisStatus
from .evidence_graph import EvidenceAtom, EvidenceGraph, build_query_signature
from .action_schema import DiscriminativeAction
from .action_gate import ActionGate, GateDecision

__all__ = [
    "CausalHypothesis",
    "HypothesisStatus",
    "EvidenceAtom",
    "EvidenceGraph",
    "build_query_signature",
    "DiscriminativeAction",
    "GateDecision",
    "ActionGate",
]
