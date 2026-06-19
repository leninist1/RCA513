"""Single-round investigation executor for PRISM-CHT.

Translates a ``DiscriminativeAction`` through the action gate, tool
registry, and fact-only tool into a single ``EvidenceAtom`` stored
in the ``EvidenceGraph``.  The executor never computes scores, never
links evidence to hypotheses, and never calls an LLM.
"""

from __future__ import annotations

from typing import Any, Mapping

from .action_gate import ActionGate
from .action_schema import DiscriminativeAction
from .canonical import to_dispatch_args
from .evidence_graph import EvidenceAtom, EvidenceGraph
from .hypothesis import CausalHypothesis
from .telemetry_store import TelemetryStore
from .tool_registry import ToolRegistry


class ActionRejectedError(ValueError):
    """Raised when the action gate rejects an action."""


class InvestigationExecutor:
    """Single-round executor of discriminative investigation actions.

    The executor implements a strict transaction protocol:

    1. ``gate.evaluate()`` — check validity without side effects.
    2. ``gate.reserve()`` — tentatively claim the query signature.
    3. ``registry.execute()`` — run the fact-only tool.
    4. ``graph.add_evidence()`` — persist the resulting atom.
    5. ``gate.commit()`` — finalise the reservation.

    If any step between reserve and commit fails, the executor calls
    ``gate.release()`` and re-raises the original exception.
    """

    def __init__(
        self,
        *,
        gate: ActionGate,
        registry: ToolRegistry,
        graph: EvidenceGraph,
        store: TelemetryStore,
    ) -> None:
        self._gate = gate
        self._registry = registry
        self._graph = graph
        self._store = store

    def execute(
        self,
        *,
        action: DiscriminativeAction,
        hypotheses: Mapping[str, CausalHypothesis],
    ) -> EvidenceAtom:
        # 1. Evaluate (no side effects)
        decision = self._gate.evaluate(action, hypotheses)
        if not decision.accepted:
            raise ActionRejectedError(
                f"Action '{action.action_id}' rejected: "
                + "; ".join(decision.reasons)
            )

        # 2. Check tool registration before reserving
        if not self._registry.has(action.tool_name):
            raise ValueError(
                f"Tool '{action.tool_name}' is not registered"
            )

        # 3. Reserve the query signature
        reserve_decision = self._gate.reserve(action, hypotheses)
        if not reserve_decision.accepted:
            # Should not happen if evaluate() passed, but guard anyway
            raise ActionRejectedError(
                f"Action '{action.action_id}' reserve rejected: "
                + "; ".join(reserve_decision.reasons)
            )

        sig = action.query_signature()

        try:
            # 4. Convert args to plain mutable dict
            dispatch_args = to_dispatch_args(action.args)

            # 5. Execute the tool
            result = self._registry.execute(
                tool_name=action.tool_name,
                args=dispatch_args,
                store=self._store,
            )

            # 6. Construct EvidenceAtom
            evidence_id = "evidence:" + sig
            atom = EvidenceAtom(
                evidence_id=evidence_id,
                query_signature=sig,
                modality=result.modality,
                component_scope=result.component_scope,
                time_window=result.time_window,
                observation=result.observation,
                provenance=result.provenance,
                missing_fields=result.missing_fields,
                reliability_note=result.reliability_note,
            )

            # 7. Store in graph
            stored = self._graph.add_evidence(atom)

            # 8. Commit
            self._gate.commit(sig)

            return stored

        except Exception:
            # Release the reservation on any failure
            self._gate.release(sig)
            raise
