"""Deterministic prompt builders for PRISM-CHT structured LLM policy.

Every builder produces a ``ModelRequest`` with a deterministic system
message and a canonical JSON user message.  No mutable objects, memory
addresses, or Graph references leak into prompts.  Prompts exceeding
the character budget raise ``PromptBudgetExceededError``.
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from .canonical import canonicalize_json_value
from .evidence_graph import EvidenceAtom, EvidenceGraph
from .llm_types import (
    ModelMessage,
    ModelRequest,
    PromptBudgetExceededError,
)
from .tournament_types import (
    HypothesisSnapshot,
    LeadTournamentSnapshot,
)
from .challenge_types import ChallengeSnapshot, ChallengeProposal
from .action_schema import DiscriminativeAction


DEFAULT_POLICY_TOOL_NAMES = (
    "compare_onset_order",
    "inspect_trace_path",
    "retrieve_raw_evidence",
)


# ===========================================================================
# Evidence serializers
# ===========================================================================


def serialize_evidence_atom(
    atom: EvidenceAtom,
) -> Mapping[str, Any]:
    """Serialize an EvidenceAtom to a canonical, JSON-compatible dict.

    Only exposes: evidence_id, query_signature, modality, component_scope,
    time_window, observation, provenance, missing_fields, reliability_note.
    No object repr or memory addresses.
    """
    return canonicalize_json_value({
        "evidence_id": atom.evidence_id,
        "query_signature": atom.query_signature,
        "modality": atom.modality,
        "component_scope": list(atom.component_scope),
        "time_window": list(atom.time_window),
        "observation": dict(atom.observation),
        "provenance": dict(atom.provenance),
        "missing_fields": list(atom.missing_fields),
        "reliability_note": atom.reliability_note,
    })


def build_evidence_catalog(
    *,
    graph: EvidenceGraph,
    evidence_ids: Sequence[str],
) -> tuple[Mapping[str, Any], ...]:
    """Build a read-only evidence catalog for the given *evidence_ids*.

    Evidence IDs are deduplicated but kept in first-appearance order.
    Raises ``ValueError`` if any ID does not exist in *graph*.
    """
    seen: set[str] = set()
    ordered: list[str] = []
    for eid in evidence_ids:
        if eid not in seen:
            if eid not in graph.evidence_by_id:
                raise ValueError(
                    f"build_evidence_catalog: evidence '{eid}' not found in graph"
                )
            seen.add(eid)
            ordered.append(eid)

    return tuple(
        serialize_evidence_atom(graph.evidence_by_id[eid])
        for eid in ordered
    )


# ===========================================================================
# Hypothesis snapshot serializer
# ===========================================================================


def _serialize_hypothesis_snapshot(s: HypothesisSnapshot) -> Mapping[str, Any]:
    """Serialize a HypothesisSnapshot to a canonical JSON dict."""
    return canonicalize_json_value({
        "hypothesis_id": s.hypothesis_id,
        "root_component": s.root_component,
        "reason_family": s.reason_family,
        "onset_interval": list(s.onset_interval),
        "local_trigger": s.local_trigger,
        "propagation_path": list(s.propagation_path),
        "explained_symptoms": list(s.explained_symptoms),
        "predicted_observations": list(s.predicted_observations),
        "falsifiers": list(s.falsifiers),
        "supporting_evidence_ids": list(s.supporting_evidence_ids),
        "contradicting_evidence_ids": list(s.contradicting_evidence_ids),
        "unresolved_questions": list(s.unresolved_questions),
        "status": s.status.value,
    })


# ===========================================================================
# Core policy request builder
# ===========================================================================


_SYSTEM_INSTRUCTIONS = (
    "You are a structured-output causal analysis agent. "
    "You MUST output exactly one JSON object and nothing else. "
    "Do NOT output markdown code fences (no ```). "
    "Do NOT output any text before or after the JSON object. "
    "Do NOT invent evidence IDs that do not exist. "
    "Do NOT invent hypothesis IDs that do not exist. "
    "Do NOT add fields not specified in the schema. "
    "Do NOT generate scores, probabilities, confidence values, or ranks. "
    "You may only request facts through the allowed tools; "
    "you must NOT directly modify the evidence graph. "
    "Output only the JSON object."
)


def _serialize_context_json(context: Mapping[str, Any]) -> str:
    """Serialize a context dict to a canonical JSON string."""
    return json.dumps(
        canonicalize_json_value(dict(context)),
        sort_keys=True,
        indent=2,
        ensure_ascii=True,
    )


def build_policy_request(
    *,
    purpose: str,
    context: Mapping[str, Any],
    response_schema: Mapping[str, Any],
    attempt_index: int,
    repair_error: str | None = None,
    max_prompt_chars: int = 100000,
) -> ModelRequest:
    """Build a deterministic ``ModelRequest`` for a policy decision.

    * *purpose* — short description of what the model should decide.
    * *context* — the serialized state snapshot and auxiliary data.
    * *response_schema* — the expected output JSON schema (for the prompt).
    * *attempt_index* — 0 for first attempt, incremented on retry.
    * *repair_error* — if non-None, a brief message about prior parse failure.
    * *max_prompt_chars* — total character budget for all messages.
    """
    system_msg = ModelMessage(
        role="system",
        content=(
            f"{_SYSTEM_INSTRUCTIONS}\n\n"
            f"Purpose: {purpose}\n\n"
            f"Expected response schema:\n"
            f"{json.dumps(canonicalize_json_value(dict(response_schema)), sort_keys=True, indent=2, ensure_ascii=True)}"
        ),
    )

    context_json = _serialize_context_json(context)
    user_msg = ModelMessage(role="user", content=context_json)

    messages: list[ModelMessage] = [system_msg, user_msg]

    if repair_error:
        repair_msg = ModelMessage(
            role="user",
            content=(
                "Your previous output could not be parsed as valid JSON. "
                f"Error: {repair_error}\n"
                "Please respond again with exactly one JSON object "
                "matching the required schema."
            ),
        )
        messages.append(repair_msg)

    total_chars = sum(len(m.content) for m in messages)
    if total_chars > max_prompt_chars:
        raise PromptBudgetExceededError(
            f"build_policy_request: total prompt length {total_chars} "
            f"exceeds max_prompt_chars {max_prompt_chars}"
        )

    return ModelRequest(
        purpose=purpose,
        messages=tuple(messages),
        attempt_index=attempt_index,
    )


# ===========================================================================
# Lead snapshot serializer
# ===========================================================================


def _serialize_lead_snapshot(
    snapshot: LeadTournamentSnapshot,
) -> Mapping[str, Any]:
    """Serialize a LeadTournamentSnapshot to canonical JSON."""
    return canonicalize_json_value({
        "round_index": snapshot.round_index,
        "hypotheses": [
            _serialize_hypothesis_snapshot(h) for h in snapshot.hypotheses
        ],
        "evidence_ids": list(snapshot.evidence_ids),
        "audit_step_count": len(snapshot.audit_steps),
    })


def _serialize_action(action: DiscriminativeAction) -> Mapping[str, Any]:
    """Serialize a DiscriminativeAction to canonical JSON (read-only view)."""
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


def _serialize_challenge_snapshot(
    snapshot: ChallengeSnapshot,
) -> Mapping[str, Any]:
    """Serialize a ChallengeSnapshot to canonical JSON (read-only view)."""
    return canonicalize_json_value({
        "nominated_hypothesis": _serialize_hypothesis_snapshot(
            snapshot.nominated_hypothesis
        ),
        "competitor_hypotheses": [
            _serialize_hypothesis_snapshot(h)
            for h in snapshot.competitor_hypotheses
        ],
        "evidence_ids": list(snapshot.evidence_ids),
        "lead_nomination_hypothesis_id": snapshot.lead_nomination.hypothesis_id,
        "lead_audit_step_count": len(snapshot.lead_audit_steps),
    })


def _serialize_challenge_proposal(
    proposal: ChallengeProposal,
) -> Mapping[str, Any]:
    """Serialize a ChallengeProposal to canonical JSON (read-only view)."""
    return canonicalize_json_value({
        "challenge_id": proposal.challenge_id,
        "nominated_hypothesis_id": proposal.nominated_hypothesis_id,
        "competitor_hypothesis_ids": list(proposal.competitor_hypothesis_ids),
        "challenge_claim": proposal.challenge_claim,
        "falsification_target": proposal.falsification_target,
        "action": _serialize_action(proposal.action),
        "rationale": proposal.rationale,
    })


# ===========================================================================
# Lead decision request builder
# ===========================================================================


_LEAD_DECISION_RESPONSE_SCHEMA = {
    "description": (
        "Either a discriminative action (kind='action') or a "
        "final nomination (kind='nomination')."
    ),
    "oneOf": [
        {
            "kind": "action",
            "payload": {
                "action_id": "<string>",
                "action_type": "run_discriminative_test",
                "target_hypothesis_ids": ["<string>", "<string>"],
                "question": "<string>",
                "tool_name": "<string from allowed_tool_names>",
                "args": {},
                "expected_outcomes": {"<hypothesis_id>": "<string>"},
                "why_discriminative": "<string>",
            },
        },
        {
            "kind": "nomination",
            "payload": {
                "hypothesis_id": "<string>",
                "supporting_evidence_ids": ["<string>"],
                "addressed_competitor_ids": ["<string>"],
                "triplet_grounding": {
                    "component_evidence_ids": ["<string>"],
                    "reason_evidence_ids": ["<string>"],
                    "onset_evidence_ids": ["<string>"],
                },
                "rationale": "<string>",
            },
        },
    ],
}


def build_lead_decision_request(
    *,
    snapshot: LeadTournamentSnapshot,
    graph: EvidenceGraph,
    allowed_tool_names: Sequence[str],
    attempt_index: int = 0,
    repair_error: str | None = None,
) -> ModelRequest:
    """Build a request for the Lead to decide its next action or nomination."""
    catalog = build_evidence_catalog(
        graph=graph, evidence_ids=snapshot.evidence_ids
    )

    context = {
        "instruction": (
            "You are the Lead Investigator. Review the tournament snapshot "
            "and decide your next step. You may nominate a hypothesis only "
            "if you have sufficient supporting evidence and all competitors "
            "are addressed. Otherwise, propose a discriminative action "
            "using one of the allowed tools."
        ),
        "snapshot": _serialize_lead_snapshot(snapshot),
        "evidence_catalog": list(catalog),
        "allowed_tool_names": list(allowed_tool_names),
    }

    return build_policy_request(
        purpose="Lead Investigator: decide next action or nominate",
        context=context,
        response_schema=_LEAD_DECISION_RESPONSE_SCHEMA,
        attempt_index=attempt_index,
        repair_error=repair_error,
    )


# ===========================================================================
# Lead assessment request builder
# ===========================================================================


_LEAD_ASSESSMENT_RESPONSE_SCHEMA = {
    "description": "An evidence assessment for the current investigation round.",
    "type": "object",
    "properties": {
        "action_id": {"type": "string"},
        "evidence_id": {"type": "string"},
        "outcome": {"enum": ["informative", "inconclusive"]},
        "links": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "hypothesis_id": {"type": "string"},
                    "evidence_id": {"type": "string"},
                    "relation": {"enum": ["supports", "contradicts"]},
                    "rationale": {"type": "string"},
                },
            },
        },
        "status_updates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "hypothesis_id": {"type": "string"},
                    "new_status": {
                        "enum": [
                            "active", "supported", "weakened",
                            "refuted", "survived",
                        ]
                    },
                    "rationale": {"type": "string"},
                },
            },
        },
        "rationale": {"type": "string"},
    },
}


def build_lead_assessment_request(
    *,
    snapshot: LeadTournamentSnapshot,
    action: DiscriminativeAction,
    evidence: EvidenceAtom,
    attempt_index: int = 0,
    repair_error: str | None = None,
) -> ModelRequest:
    """Build a request for the Lead to assess the latest evidence."""
    context = {
        "instruction": (
            "You are the Lead Investigator. Assess the evidence collected "
            "from the most recent investigation action. Determine whether "
            "the evidence supports or contradicts each target hypothesis, "
            "and propose status updates consistent with the state machine."
        ),
        "snapshot": _serialize_lead_snapshot(snapshot),
        "action": _serialize_action(action),
        "new_evidence": serialize_evidence_atom(evidence),
    }

    return build_policy_request(
        purpose="Lead Investigator: assess evidence",
        context=context,
        response_schema=_LEAD_ASSESSMENT_RESPONSE_SCHEMA,
        attempt_index=attempt_index,
        repair_error=repair_error,
    )


# ===========================================================================
# Challenge proposal request builder
# ===========================================================================


_CHALLENGE_PROPOSAL_RESPONSE_SCHEMA = {
    "description": "A challenge proposal against the Lead nomination.",
    "type": "object",
    "properties": {
        "challenge_id": {"type": "string"},
        "nominated_hypothesis_id": {"type": "string"},
        "competitor_hypothesis_ids": {
            "type": "array", "items": {"type": "string"}
        },
        "challenge_claim": {"type": "string"},
        "falsification_target": {"type": "string"},
        "action": {
            "type": "object",
            "properties": {
                "action_id": {"type": "string"},
                "action_type": {"const": "run_discriminative_test"},
                "target_hypothesis_ids": {
                    "type": "array", "items": {"type": "string"}
                },
                "question": {"type": "string"},
                "tool_name": {"type": "string"},
                "args": {"type": "object"},
                "expected_outcomes": {
                    "type": "object",
                    "additionalProperties": {"type": "string"},
                },
                "why_discriminative": {"type": "string"},
            },
        },
        "rationale": {"type": "string"},
    },
}


def build_challenge_proposal_request(
    *,
    snapshot: ChallengeSnapshot,
    graph: EvidenceGraph,
    allowed_tool_names: Sequence[str],
    attempt_index: int = 0,
    repair_error: str | None = None,
) -> ModelRequest:
    """Build a request for the Challenger to propose a challenge."""
    all_evidence_ids = tuple(snapshot.evidence_ids)
    catalog = build_evidence_catalog(graph=graph, evidence_ids=all_evidence_ids)

    context = {
        "instruction": (
            "You are the Challenger. Review the Lead nomination and "
            "propose a challenge. Identify a weakness in the nominated "
            "hypothesis and design a discriminative action to test it."
        ),
        "snapshot": _serialize_challenge_snapshot(snapshot),
        "evidence_catalog": list(catalog),
        "allowed_tool_names": list(allowed_tool_names),
    }

    return build_policy_request(
        purpose="Challenger: propose challenge",
        context=context,
        response_schema=_CHALLENGE_PROPOSAL_RESPONSE_SCHEMA,
        attempt_index=attempt_index,
        repair_error=repair_error,
    )


# ===========================================================================
# Challenge resolution request builder
# ===========================================================================


_CHALLENGE_RESOLUTION_RESPONSE_SCHEMA = {
    "description": "A challenge resolution with verdict and evidence assessment.",
    "type": "object",
    "properties": {
        "challenge_id": {"type": "string"},
        "action_id": {"type": "string"},
        "evidence_id": {"type": "string"},
        "verdict": {
            "enum": [
                "nomination_survived",
                "nomination_refuted",
                "inconclusive",
            ]
        },
        "assessment": {
            "type": "object",
            "properties": {
                "action_id": {"type": "string"},
                "evidence_id": {"type": "string"},
                "outcome": {"enum": ["informative", "inconclusive"]},
                "links": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "hypothesis_id": {"type": "string"},
                            "evidence_id": {"type": "string"},
                            "relation": {"enum": ["supports", "contradicts"]},
                            "rationale": {"type": "string"},
                        },
                    },
                },
                "status_updates": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "hypothesis_id": {"type": "string"},
                            "new_status": {
                                "enum": [
                                    "active", "supported", "weakened",
                                    "refuted", "survived",
                                ]
                            },
                            "rationale": {"type": "string"},
                        },
                    },
                },
                "rationale": {"type": "string"},
            },
        },
        "rationale": {"type": "string"},
    },
}


def build_challenge_resolution_request(
    *,
    snapshot: ChallengeSnapshot,
    proposal: ChallengeProposal,
    evidence: EvidenceAtom,
    attempt_index: int = 0,
    repair_error: str | None = None,
) -> ModelRequest:
    """Build a request for the Challenger to assess challenge resolution."""
    context = {
        "instruction": (
            "You are the Challenger. Assess the evidence from your "
            "challenge investigation and determine the verdict. "
            "If the evidence supports the nomination, the verdict is "
            "'nomination_survived'. If it refutes the nomination, the "
            "verdict is 'nomination_refuted'. If insufficient, "
            "the verdict is 'inconclusive'."
        ),
        "snapshot": _serialize_challenge_snapshot(snapshot),
        "proposal": _serialize_challenge_proposal(proposal),
        "new_evidence": serialize_evidence_atom(evidence),
    }

    return build_policy_request(
        purpose="Challenger: assess challenge resolution",
        context=context,
        response_schema=_CHALLENGE_RESOLUTION_RESPONSE_SCHEMA,
        attempt_index=attempt_index,
        repair_error=repair_error,
    )
