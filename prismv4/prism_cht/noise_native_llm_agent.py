"""LLM-based NoiseNative agent built on PRISM-CHT primitives.

The controller runs a fixed-budget active investigation loop.  The LLM
selects the next fact-only action and interprets returned evidence, while
the executor, gates, registry, and evidence graph enforce deterministic
boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .action_gate import ActionGate
from .action_schema import DiscriminativeAction
from .assessment_gate import EvidenceAssessmentGate
from .canonical import canonicalize_json_value
from .evidence_graph import EvidenceAtom, EvidenceGraph
from .executor import InvestigationExecutor
from .hypothesis import CausalHypothesis, HypothesisStatus
from .llm_audit import AuditedModelClient, LLMIORecorder
from .llm_json import (
    _parse_action_payload,
    parse_evidence_assessment,
    parse_json_object,
    require_exact_keys,
    require_mapping,
    require_sequence,
    require_string,
    require_string_tuple,
)
from .llm_policy import StructuredModelRequester
from .llm_prompts import (
    build_evidence_catalog,
    build_policy_request,
    serialize_evidence_atom,
)
from .llm_types import ModelClient, ModelRequest, StructuredOutputError
from .noiselab_registry import NOISELAB_TOOL_NAMES, build_noiselab_tool_registry
from .telemetry_store import TelemetryStore
from .tool_registry import ToolRegistry
from .tournament_types import (
    EvidenceAssessment,
    HypothesisSnapshot,
    InvestigationAuditStep,
    LeadTournamentSnapshot,
    build_snapshot,
)


@dataclass(frozen=True)
class NoiseNativeFinalDecision:
    """Final structured RCA decision produced after the fixed step budget."""

    root_component: str
    reason_family: str
    onset_interval: tuple[float, float]
    supporting_evidence_ids: tuple[str, ...]
    refuted_alternatives: tuple[str, ...]
    remaining_uncertainties: tuple[str, ...]
    rationale: str

    def __post_init__(self) -> None:
        if not self.root_component.strip():
            raise ValueError("root_component must be non-empty")
        if not self.reason_family.strip():
            raise ValueError("reason_family must be non-empty")
        if len(self.onset_interval) != 2:
            raise ValueError("onset_interval must have exactly two values")
        if self.onset_interval[0] > self.onset_interval[1]:
            raise ValueError("onset_interval start must be <= end")
        if not self.supporting_evidence_ids:
            raise ValueError("supporting_evidence_ids must not be empty")
        if len(set(self.supporting_evidence_ids)) != len(self.supporting_evidence_ids):
            raise ValueError("supporting_evidence_ids contains duplicate entries")
        if len(set(self.refuted_alternatives)) != len(self.refuted_alternatives):
            raise ValueError("refuted_alternatives contains duplicate entries")
        if not self.rationale.strip():
            raise ValueError("rationale must be non-empty")


@dataclass(frozen=True)
class NoiseNativeRunResult:
    status: str
    steps_completed: int
    evidence_ids: tuple[str, ...]
    audit_steps: tuple[InvestigationAuditStep, ...]
    final_decision: NoiseNativeFinalDecision


def _serialize_hypothesis(snapshot: HypothesisSnapshot) -> Mapping[str, Any]:
    return canonicalize_json_value({
        "hypothesis_id": snapshot.hypothesis_id,
        "root_component": snapshot.root_component,
        "reason_family": snapshot.reason_family,
        "onset_interval": list(snapshot.onset_interval),
        "local_trigger": snapshot.local_trigger,
        "propagation_path": list(snapshot.propagation_path),
        "explained_symptoms": list(snapshot.explained_symptoms),
        "predicted_observations": list(snapshot.predicted_observations),
        "falsifiers": list(snapshot.falsifiers),
        "supporting_evidence_ids": list(snapshot.supporting_evidence_ids),
        "contradicting_evidence_ids": list(snapshot.contradicting_evidence_ids),
        "unresolved_questions": list(snapshot.unresolved_questions),
        "status": snapshot.status.value,
    })


def _serialize_action(action: DiscriminativeAction) -> Mapping[str, Any]:
    return canonicalize_json_value({
        "action_id": action.action_id,
        "action_type": action.action_type,
        "target_hypothesis_ids": list(action.target_hypothesis_ids),
        "question": action.question,
        "tool_name": action.tool_name,
        "args": dict(action.args),
        "expected_outcomes": dict(action.expected_outcomes),
        "why_discriminative": action.why_discriminative,
    })


def _serialize_assessment(assessment: EvidenceAssessment) -> Mapping[str, Any]:
    return canonicalize_json_value({
        "action_id": assessment.action_id,
        "evidence_id": assessment.evidence_id,
        "outcome": assessment.outcome.value,
        "links": [
            {
                "hypothesis_id": link.hypothesis_id,
                "evidence_id": link.evidence_id,
                "relation": link.relation.value,
                "rationale": link.rationale,
            }
            for link in assessment.links
        ],
        "status_updates": [
            {
                "hypothesis_id": update.hypothesis_id,
                "new_status": update.new_status.value,
                "rationale": update.rationale,
            }
            for update in assessment.status_updates
        ],
        "rationale": assessment.rationale,
    })


def _serialize_snapshot(
    *,
    snapshot: LeadTournamentSnapshot,
    graph: EvidenceGraph,
    max_steps: int,
    allowed_tool_names: Sequence[str],
) -> Mapping[str, Any]:
    return canonicalize_json_value({
        "agent": "noise_native_llm",
        "objective": (
            "Choose fact-only NoiseLab actions that maximize source-vs-symptom "
            "information gain over a fixed investigation budget."
        ),
        "step_index": snapshot.round_index,
        "max_steps": max_steps,
        "remaining_action_steps": max(0, max_steps - snapshot.round_index),
        "allowed_tool_names": list(allowed_tool_names),
        "hypotheses": [_serialize_hypothesis(h) for h in snapshot.hypotheses],
        "evidence_catalog": list(
            build_evidence_catalog(graph=graph, evidence_ids=snapshot.evidence_ids)
        ),
        "action_history": [
            {
                "step_index": step.round_index,
                "action": _serialize_action(step.action),
                "evidence_id": step.evidence_id,
                "assessment": _serialize_assessment(step.assessment),
            }
            for step in snapshot.audit_steps
        ],
        "policy_constraints": [
            "Use actions, not direct guesses, until the fixed action budget is spent.",
            "Prefer tests that distinguish source from propagated symptom.",
            "Do not emit scores, probabilities, confidence values, or ranks.",
            "Do not invent hypothesis IDs, evidence IDs, or tools.",
        ],
    })


_ACTION_RESPONSE_SCHEMA = {
    "kind": "action",
    "payload": {
        "action_id": "string",
        "action_type": "run_discriminative_test",
        "target_hypothesis_ids": ["hypothesis_id_a", "hypothesis_id_b"],
        "question": "string",
        "tool_name": "one allowed tool name",
        "args": {},
        "expected_outcomes": {"hypothesis_id_a": "string", "hypothesis_id_b": "string"},
        "why_discriminative": "string",
    },
}


_ASSESSMENT_RESPONSE_SCHEMA = {
    "action_id": "string",
    "evidence_id": "string",
    "outcome": "informative | inconclusive",
    "links": [
        {
            "hypothesis_id": "string",
            "evidence_id": "string",
            "relation": "supports | contradicts",
            "rationale": "string",
        }
    ],
    "status_updates": [
        {
            "hypothesis_id": "string",
            "new_status": "supported | weakened | refuted | survived",
            "rationale": "string",
        }
    ],
    "rationale": "string",
}


_FINAL_RESPONSE_SCHEMA = {
    "root_component": "string",
    "reason_family": "string",
    "onset_interval": [0.0, 0.0],
    "supporting_evidence_ids": ["evidence:id"],
    "refuted_alternatives": ["component_or_hypothesis"],
    "remaining_uncertainties": ["string"],
    "rationale": "string",
}


def _build_action_request(
    *,
    snapshot: LeadTournamentSnapshot,
    graph: EvidenceGraph,
    allowed_tool_names: Sequence[str],
    max_steps: int,
    attempt_index: int,
    repair_error: str | None = None,
) -> ModelRequest:
    return build_policy_request(
        purpose="noise_native_choose_next_action",
        context=_serialize_snapshot(
            snapshot=snapshot,
            graph=graph,
            max_steps=max_steps,
            allowed_tool_names=allowed_tool_names,
        ),
        response_schema=_ACTION_RESPONSE_SCHEMA,
        attempt_index=attempt_index,
        repair_error=repair_error,
    )


def _build_assessment_request(
    *,
    snapshot: LeadTournamentSnapshot,
    action: DiscriminativeAction,
    evidence: EvidenceAtom,
    attempt_index: int,
    repair_error: str | None = None,
) -> ModelRequest:
    return build_policy_request(
        purpose="noise_native_assess_evidence",
        context={
            "agent": "noise_native_llm",
            "step_index": snapshot.round_index,
            "hypotheses": [_serialize_hypothesis(h) for h in snapshot.hypotheses],
            "action": _serialize_action(action),
            "evidence": serialize_evidence_atom(evidence),
            "assessment_constraints": [
                "Only link the current evidence_id to action target hypotheses.",
                "Use inconclusive with empty links/status_updates when evidence does not distinguish targets.",
                "Do not emit scores, probabilities, confidence values, or ranks.",
            ],
        },
        response_schema=_ASSESSMENT_RESPONSE_SCHEMA,
        attempt_index=attempt_index,
        repair_error=repair_error,
    )


def _build_final_request(
    *,
    snapshot: LeadTournamentSnapshot,
    graph: EvidenceGraph,
    allowed_tool_names: Sequence[str],
    max_steps: int,
    attempt_index: int,
    repair_error: str | None = None,
) -> ModelRequest:
    context = dict(
        _serialize_snapshot(
            snapshot=snapshot,
            graph=graph,
            max_steps=max_steps,
            allowed_tool_names=allowed_tool_names,
        )
    )
    context["finalization_constraints"] = [
        "Pick the best grounded root component after exactly the completed evidence budget.",
        "supporting_evidence_ids must reference existing evidence IDs.",
        "Represent unresolved ambiguity in remaining_uncertainties instead of inventing certainty.",
        "Do not emit scores, probabilities, confidence values, or ranks.",
    ]
    return build_policy_request(
        purpose="noise_native_final_decision",
        context=context,
        response_schema=_FINAL_RESPONSE_SCHEMA,
        attempt_index=attempt_index,
        repair_error=repair_error,
    )


def parse_noise_native_action(text: str) -> DiscriminativeAction:
    context = "parse_noise_native_action"
    obj = parse_json_object(text)
    require_exact_keys(obj, required={"kind", "payload"}, context=context)
    kind = require_string(obj["kind"], context=f"{context}.kind")
    if kind != "action":
        raise StructuredOutputError(f"{context}.kind must be 'action', got '{kind}'")
    payload = require_mapping(obj["payload"], context=f"{context}.payload")
    return _parse_action_payload(payload, context=context)


def parse_noise_native_final(text: str) -> NoiseNativeFinalDecision:
    context = "parse_noise_native_final"
    obj = parse_json_object(text)
    require_exact_keys(
        obj,
        required={
            "root_component",
            "reason_family",
            "onset_interval",
            "supporting_evidence_ids",
            "refuted_alternatives",
            "remaining_uncertainties",
            "rationale",
        },
        context=context,
    )
    onset_raw = require_sequence(obj["onset_interval"], context=f"{context}.onset_interval")
    if len(onset_raw) != 2:
        raise StructuredOutputError(f"{context}.onset_interval must have two values")
    try:
        onset = (float(onset_raw[0]), float(onset_raw[1]))
    except (TypeError, ValueError) as exc:
        raise StructuredOutputError(
            f"{context}.onset_interval values must be numeric"
        ) from exc

    return NoiseNativeFinalDecision(
        root_component=require_string(
            obj["root_component"], context=f"{context}.root_component"
        ),
        reason_family=require_string(
            obj["reason_family"], context=f"{context}.reason_family"
        ),
        onset_interval=onset,
        supporting_evidence_ids=require_string_tuple(
            obj["supporting_evidence_ids"],
            context=f"{context}.supporting_evidence_ids",
        ),
        refuted_alternatives=require_string_tuple(
            obj["refuted_alternatives"],
            context=f"{context}.refuted_alternatives",
        ),
        remaining_uncertainties=require_string_tuple(
            obj["remaining_uncertainties"],
            context=f"{context}.remaining_uncertainties",
        ),
        rationale=require_string(obj["rationale"], context=f"{context}.rationale"),
    )


class NoiseNativeLLMPolicy:
    """Structured LLM policy with mandatory full I/O recording."""

    def __init__(
        self,
        *,
        client: ModelClient,
        graph: EvidenceGraph,
        allowed_tool_names: Sequence[str] = NOISELAB_TOOL_NAMES,
        max_steps: int = 12,
        max_attempts: int = 2,
        llm_io_jsonl_path: str | Path | None = None,
        llm_io_recorder: LLMIORecorder | None = None,
    ) -> None:
        self.io_recorder = llm_io_recorder or LLMIORecorder(
            jsonl_path=Path(llm_io_jsonl_path) if llm_io_jsonl_path else None
        )
        audited_client = AuditedModelClient(
            inner=client,
            recorder=self.io_recorder,
        )
        self._requester = StructuredModelRequester(
            client=audited_client,
            max_attempts=max_attempts,
        )
        self._graph = graph
        self._allowed_tool_names = tuple(allowed_tool_names)
        self._max_steps = max_steps

    def decide_next(self, *, snapshot: LeadTournamentSnapshot) -> DiscriminativeAction:
        action = self._requester.request_parsed(
            request_builder=_build_action_request,
            parser=parse_noise_native_action,
            builder_kwargs={
                "snapshot": snapshot,
                "graph": self._graph,
                "allowed_tool_names": self._allowed_tool_names,
                "max_steps": self._max_steps,
            },
        )
        assert isinstance(action, DiscriminativeAction)
        return action

    def assess_evidence(
        self,
        *,
        snapshot: LeadTournamentSnapshot,
        action: DiscriminativeAction,
        evidence: EvidenceAtom,
    ) -> EvidenceAssessment:
        assessment = self._requester.request_parsed(
            request_builder=_build_assessment_request,
            parser=parse_evidence_assessment,
            builder_kwargs={
                "snapshot": snapshot,
                "action": action,
                "evidence": evidence,
            },
        )
        assert isinstance(assessment, EvidenceAssessment)
        return assessment

    def finalize(self, *, snapshot: LeadTournamentSnapshot) -> NoiseNativeFinalDecision:
        decision = self._requester.request_parsed(
            request_builder=_build_final_request,
            parser=parse_noise_native_final,
            builder_kwargs={
                "snapshot": snapshot,
                "graph": self._graph,
                "allowed_tool_names": self._allowed_tool_names,
                "max_steps": self._max_steps,
            },
        )
        assert isinstance(decision, NoiseNativeFinalDecision)
        return decision


class NoiseNativeLLMController:
    """Fixed-step NoiseNative investigation controller."""

    def __init__(
        self,
        *,
        policy: NoiseNativeLLMPolicy,
        store: TelemetryStore,
        graph: EvidenceGraph,
        registry: ToolRegistry | None = None,
        allowed_tool_names: Sequence[str] = NOISELAB_TOOL_NAMES,
        max_steps: int = 12,
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        self._policy = policy
        self._store = store
        self._graph = graph
        self._registry = registry or build_noiselab_tool_registry(
            include_default_tools=False
        )
        self._allowed_tool_names = tuple(allowed_tool_names)
        self._max_steps = max_steps
        self._executor = InvestigationExecutor(
            gate=ActionGate(allowed_tools=set(self._allowed_tool_names)),
            registry=self._registry,
            graph=self._graph,
            store=self._store,
        )
        self._assessment_gate = EvidenceAssessmentGate()

    def run(
        self,
        *,
        initial_hypotheses: Sequence[CausalHypothesis],
    ) -> NoiseNativeRunResult:
        hypotheses = self._register_and_activate(initial_hypotheses)
        audit_steps: list[InvestigationAuditStep] = []

        for step_index in range(self._max_steps):
            snapshot = build_snapshot(
                round_index=step_index,
                hypotheses=hypotheses,
                graph=self._graph,
                audit_steps=audit_steps,
                max_rounds=self._max_steps,
            )
            action = self._policy.decide_next(snapshot=snapshot)
            evidence = self._executor.execute(
                action=action,
                hypotheses=hypotheses,
            )
            assessment = self._policy.assess_evidence(
                snapshot=snapshot,
                action=action,
                evidence=evidence,
            )
            self._assessment_gate.apply(
                assessment=assessment,
                action=action,
                evidence=evidence,
                hypotheses=hypotheses,
                graph=self._graph,
            )
            audit_steps.append(
                InvestigationAuditStep(
                    round_index=step_index,
                    action=action,
                    evidence_id=evidence.evidence_id,
                    assessment=assessment,
                )
            )

        final_snapshot = build_snapshot(
            round_index=self._max_steps,
            hypotheses=hypotheses,
            graph=self._graph,
            audit_steps=audit_steps,
            max_rounds=self._max_steps,
        )
        final_decision = self._policy.finalize(snapshot=final_snapshot)
        self._validate_final_decision(final_decision)

        return NoiseNativeRunResult(
            status="completed",
            steps_completed=self._max_steps,
            evidence_ids=tuple(sorted(self._graph.evidence_by_id.keys())),
            audit_steps=tuple(audit_steps),
            final_decision=final_decision,
        )

    def _register_and_activate(
        self,
        initial_hypotheses: Sequence[CausalHypothesis],
    ) -> Mapping[str, CausalHypothesis]:
        hypotheses: dict[str, CausalHypothesis] = {}
        for hypothesis in initial_hypotheses:
            if hypothesis.hypothesis_id in hypotheses:
                raise ValueError(
                    f"Duplicate hypothesis_id '{hypothesis.hypothesis_id}'"
                )
            if hypothesis.hypothesis_id not in self._graph.hypotheses_by_id:
                self._graph.register_hypothesis(hypothesis)
            if hypothesis.status == HypothesisStatus.DRAFT:
                hypothesis.activate()
            hypotheses[hypothesis.hypothesis_id] = hypothesis
        if len(hypotheses) < 2:
            raise ValueError("NoiseNativeLLMController requires at least two hypotheses")
        return hypotheses

    def _validate_final_decision(self, decision: NoiseNativeFinalDecision) -> None:
        missing = [
            evidence_id
            for evidence_id in decision.supporting_evidence_ids
            if evidence_id not in self._graph.evidence_by_id
        ]
        if missing:
            raise ValueError(
                "final_decision.supporting_evidence_ids reference missing evidence: "
                + ", ".join(missing)
            )
