"""Evidence atoms and a deduplication graph for PRISM-CHT.

EvidenceAtom represents raw factual observations — never scores.
EvidenceGraph stores evidence indexed by canonical query signature
and maintains hypothesis-evidence linkage edges.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

from .hypothesis import CausalHypothesis


def build_query_signature(
    *,
    tool_name: str,
    component_scope: Sequence[str],
    signal_scope: Sequence[str],
    time_window: Tuple[float, float],
    parameters: Mapping[str, Any],
) -> str:
    """Build a canonical SHA-256 query signature.

    Signatures are deterministic, independent of input ordering,
    and free of natural-language questions, timestamps, or random data.
    """
    canonical: Dict[str, Any] = {
        "tool_name": tool_name,
        "component_scope": sorted(component_scope),
        "signal_scope": sorted(signal_scope),
        "time_window": list(time_window),
        "parameters": dict(sorted(parameters.items())),
    }
    # canonical JSON: sorted keys, no spaces after separators, ASCII-safe
    canonical_json = json.dumps(
        canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class EvidenceAtom:
    """Immutable observable fact returned by a tool.

    Must NOT contain root score, posterior, support score, or against score.
    """

    evidence_id: str
    query_signature: str
    modality: str
    component_scope: Tuple[str, ...]
    time_window: Tuple[float, float]
    observation: Mapping[str, Any]
    provenance: Mapping[str, Any]
    missing_fields: Tuple[str, ...] = ()
    reliability_note: str = ""

    def __post_init__(self):
        if not self.evidence_id or not self.evidence_id.strip():
            raise ValueError("evidence_id must be non-empty")
        if not self.query_signature or not self.query_signature.strip():
            raise ValueError("query_signature must be non-empty")
        if not self.modality or not self.modality.strip():
            raise ValueError("modality must be non-empty")

        if len(self.time_window) != 2:
            raise ValueError(
                f"time_window must have exactly 2 elements, got {len(self.time_window)}"
            )
        start, end = self.time_window
        if start > end:
            raise ValueError(f"time_window start ({start}) must be <= end ({end})")

        if not self.component_scope:
            raise ValueError("component_scope must not be empty")

        if self.observation is None:
            raise ValueError("observation must not be None")
        if self.provenance is None:
            raise ValueError("provenance must not be None")


class EvidenceGraph:
    """Fact graph: stores evidence atoms and hypothesis-evidence links.

    This is a fact store, NOT a scorer.  It does not compute weighted
    support/against scores.
    """

    def __init__(self) -> None:
        self.evidence_by_id: Dict[str, EvidenceAtom] = {}
        self.evidence_id_by_signature: Dict[str, str] = {}
        self.hypotheses_by_id: Dict[str, CausalHypothesis] = {}
        self.support_edges: Dict[str, set] = {}
        self.contradiction_edges: Dict[str, set] = {}

    # -- hypothesis registration -------------------------------------------

    def register_hypothesis(self, hypothesis: CausalHypothesis) -> None:
        hid = hypothesis.hypothesis_id
        if hid in self.hypotheses_by_id:
            raise ValueError(f"Hypothesis '{hid}' is already registered")
        self.hypotheses_by_id[hid] = hypothesis
        self.support_edges.setdefault(hid, set())
        self.contradiction_edges.setdefault(hid, set())

    # -- evidence ----------------------------------------------------------

    def add_evidence(self, atom: EvidenceAtom) -> EvidenceAtom:
        sig = atom.query_signature

        if sig in self.evidence_id_by_signature:
            existing_id = self.evidence_id_by_signature[sig]
            existing = self.evidence_by_id[existing_id]
            if atom.evidence_id != existing_id:
                raise ValueError(
                    f"Evidence signature collision: '{atom.evidence_id}' "
                    f"has same query_signature as existing '{existing_id}' "
                    f"but different evidence_id"
                )
            return existing

        if atom.evidence_id in self.evidence_by_id:
            if self.evidence_by_id[atom.evidence_id] != atom:
                raise ValueError(
                    f"evidence_id '{atom.evidence_id}' already exists "
                    f"with different payload"
                )
            return self.evidence_by_id[atom.evidence_id]

        self.evidence_by_id[atom.evidence_id] = atom
        self.evidence_id_by_signature[sig] = atom.evidence_id
        return atom

    def get_evidence(self, evidence_id: str) -> EvidenceAtom:
        if evidence_id not in self.evidence_by_id:
            raise KeyError(f"Evidence '{evidence_id}' not found")
        return self.evidence_by_id[evidence_id]

    def evidence_count(self) -> int:
        return len(self.evidence_by_id)

    # -- linkage -----------------------------------------------------------

    def link_support(self, hypothesis_id: str, evidence_id: str) -> None:
        if hypothesis_id not in self.hypotheses_by_id:
            raise ValueError(f"Hypothesis '{hypothesis_id}' not registered")
        if evidence_id not in self.evidence_by_id:
            raise ValueError(f"Evidence '{evidence_id}' not found")
        self.support_edges.setdefault(hypothesis_id, set()).add(evidence_id)

    def link_contradiction(self, hypothesis_id: str, evidence_id: str) -> None:
        if hypothesis_id not in self.hypotheses_by_id:
            raise ValueError(f"Hypothesis '{hypothesis_id}' not registered")
        if evidence_id not in self.evidence_by_id:
            raise ValueError(f"Evidence '{evidence_id}' not found")
        self.contradiction_edges.setdefault(hypothesis_id, set()).add(evidence_id)
