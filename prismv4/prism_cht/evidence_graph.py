"""Evidence atoms and a deduplication graph for PRISM-CHT.

EvidenceAtom represents raw factual observations — never scores.
EvidenceGraph stores evidence indexed by canonical query signature
and maintains hypothesis-evidence linkage edges.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Set, Tuple

from .canonical import build_tool_call_signature, deep_freeze
from .hypothesis import CausalHypothesis


def build_query_signature(
    *,
    tool_name: str,
    component_scope: Sequence[str],
    signal_scope: Sequence[str],
    time_window: Tuple[float, float],
    parameters: Mapping[str, Any],
) -> str:
    """Build a canonical SHA-256 query signature (compatibility wrapper).

    Delegates to ``build_tool_call_signature`` with the arguments
    packed into a canonical ``args`` dict.
    """
    return build_tool_call_signature(
        tool_name=tool_name,
        args={
            "component_scope": sorted(set(component_scope)),
            "signal_scope": sorted(set(signal_scope)),
            "time_window": time_window,
            "parameters": parameters,
        },
    )


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

        # Normalize component_scope: dedup, sort, store as tuple
        scope = tuple(sorted(set(self.component_scope)))
        if not scope:
            raise ValueError("component_scope must not be empty")
        object.__setattr__(self, "component_scope", scope)

        # Normalize missing_fields: dedup, sort, store as tuple
        object.__setattr__(
            self,
            "missing_fields",
            tuple(sorted(set(self.missing_fields))),
        )

        # observation and provenance must be Mapping
        if self.observation is None:
            raise ValueError("observation must not be None")
        if not isinstance(self.observation, Mapping):
            raise ValueError(
                f"observation must be a Mapping, got {type(self.observation).__name__}"
            )
        if self.provenance is None:
            raise ValueError("provenance must not be None")
        if not isinstance(self.provenance, Mapping):
            raise ValueError(
                f"provenance must be a Mapping, got {type(self.provenance).__name__}"
            )

        # Deep-freeze nested containers
        object.__setattr__(self, "observation", deep_freeze(self.observation))
        object.__setattr__(self, "provenance", deep_freeze(self.provenance))


class EvidenceGraph:
    """Fact graph: stores evidence atoms and hypothesis-evidence links.

    This is a fact store, NOT a scorer.  It does not compute weighted
    support/against scores.
    """

    def __init__(self) -> None:
        self.evidence_by_id: Dict[str, EvidenceAtom] = {}
        self.evidence_id_by_signature: Dict[str, str] = {}
        self.hypotheses_by_id: Dict[str, CausalHypothesis] = {}
        self.support_edges: Dict[str, Set[str]] = {}
        self.contradiction_edges: Dict[str, Set[str]] = {}

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
        eid = atom.evidence_id

        # Check if this exact evidence_id already exists
        if eid in self.evidence_by_id:
            existing = self.evidence_by_id[eid]
            if existing != atom:
                raise ValueError(
                    f"Conflicting evidence payload for evidence_id '{eid}': "
                    f"existing atom differs from new atom"
                )
            return existing

        # Check if this query_signature already maps to existing evidence
        if sig in self.evidence_id_by_signature:
            existing_id = self.evidence_id_by_signature[sig]
            existing = self.evidence_by_id[existing_id]
            if existing != atom:
                raise ValueError(
                    f"Conflicting evidence payload for query_signature "
                    f"'{sig[:16]}...': evidence_id '{eid}' differs from "
                    f"existing '{existing_id}'"
                )
            return existing

        self.evidence_by_id[eid] = atom
        self.evidence_id_by_signature[sig] = eid
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

        h = self.hypotheses_by_id[hypothesis_id]
        self.support_edges.setdefault(hypothesis_id, set()).add(evidence_id)

        # Synchronize into the hypothesis object (sole write entry)
        if evidence_id not in h.supporting_evidence_ids:
            h.attach_support(evidence_id)

    def link_contradiction(self, hypothesis_id: str, evidence_id: str) -> None:
        if hypothesis_id not in self.hypotheses_by_id:
            raise ValueError(f"Hypothesis '{hypothesis_id}' not registered")
        if evidence_id not in self.evidence_by_id:
            raise ValueError(f"Evidence '{evidence_id}' not found")

        h = self.hypotheses_by_id[hypothesis_id]
        self.contradiction_edges.setdefault(hypothesis_id, set()).add(evidence_id)

        # Synchronize into the hypothesis object (sole write entry)
        if evidence_id not in h.contradicting_evidence_ids:
            h.attach_contradiction(evidence_id)

    # -- consistency -------------------------------------------------------

    def validate_consistency(self) -> None:
        """Verify bidirectional consistency between graph edges and hypotheses.

        Raises ValueError if any inconsistency is found:
        - graph edge missing from hypothesis evidence_ids
        - hypothesis evidence_id not present in graph
        - hypothesis edge type mismatch with graph edges
        """
        for hid, h in self.hypotheses_by_id.items():
            sup_edges = self.support_edges.get(hid, set())
            con_edges = self.contradiction_edges.get(hid, set())

            # Every graph support edge must appear in hypothesis
            for eid in sup_edges:
                if eid not in h.supporting_evidence_ids:
                    raise ValueError(
                        f"Inconsistency: graph has support edge "
                        f"{hid} -> {eid} but hypothesis '{hid}' does not "
                        f"list it in supporting_evidence_ids"
                    )

            # Every graph contradiction edge must appear in hypothesis
            for eid in con_edges:
                if eid not in h.contradicting_evidence_ids:
                    raise ValueError(
                        f"Inconsistency: graph has contradiction edge "
                        f"{hid} -> {eid} but hypothesis '{hid}' does not "
                        f"list it in contradicting_evidence_ids"
                    )

            # Every hypothesis support id must be in the graph
            for eid in h.supporting_evidence_ids:
                if eid not in self.evidence_by_id:
                    raise ValueError(
                        f"Inconsistency: hypothesis '{hid}' references "
                        f"supporting evidence '{eid}' not in graph"
                    )
                if eid not in sup_edges:
                    raise ValueError(
                        f"Inconsistency: hypothesis '{hid}' lists "
                        f"'{eid}' as supporting but graph has no "
                        f"support edge {hid} -> {eid}"
                    )

            # Every hypothesis contradiction id must be in the graph
            for eid in h.contradicting_evidence_ids:
                if eid not in self.evidence_by_id:
                    raise ValueError(
                        f"Inconsistency: hypothesis '{hid}' references "
                        f"contradicting evidence '{eid}' not in graph"
                    )
                if eid not in con_edges:
                    raise ValueError(
                        f"Inconsistency: hypothesis '{hid}' lists "
                        f"'{eid}' as contradicting but graph has no "
                        f"contradiction edge {hid} -> {eid}"
                    )
