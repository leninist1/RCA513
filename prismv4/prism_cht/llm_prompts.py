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
    "inspect_reason_signature",
    "check_propagation_consistency",
    "find_unexplained_symptoms",
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
    "You are a structured-output causal diagnosis agent for distributed "
    "systems. You MUST output exactly one JSON object and nothing else. "
    "Do NOT output markdown code fences. Do NOT output any text before or "
    "after the JSON object. Do NOT invent evidence IDs, hypothesis IDs, "
    "components, tools, scores, probabilities, confidence values, or ranks. "
    "Do NOT add fields not specified in the schema. You may only request "
    "facts through the allowed tools; you must NOT directly modify the "
    "evidence graph. Keep every rationale string under 160 characters. "
    "Keep arrays as short as the schema permits. Return compact JSON.\n\n"
    "Diagnostic doctrine: the root cause is the initiating component and "
    "failure mechanism that best explains the observed symptoms. It is NOT "
    "necessarily the component with the largest local anomaly, the earliest "
    "entry-service symptom, or the component with the most logs. Distinguish "
    "source from propagated symptom. Prefer hypotheses that combine: "
    "(1) plausible local mechanism, (2) temporal consistency, and "
    "(3) propagation/explanation of downstream symptoms. Treat conflicting "
    "evidence as ambiguity to resolve, not as permission to force a winner.\n\n"
    "Evidence doctrine: reason-signature magnitude is local anomaly strength, "
    "not root-cause proof; it can nominate candidates but must be corroborated "
    "by onset, propagation, trace, or raw evidence. Onset order is useful, but "
    "ties are ambiguity, not decisive support for every tied component. A "
    "single missing propagation path is weak evidence and must not by itself "
    "support the opposite root. Missing data is not proof of absence unless "
    "the tool explicitly reports exhaustive coverage. A downstream symptom can "
    "have a high local anomaly magnitude. Use only evidence returned by tools.\n\n"
    "Tool doctrine: compare_onset_order is for temporal triage and tie "
    "detection; inspect_reason_signature is for local mechanism features; "
    "check_propagation_consistency and inspect_trace_path are for source-vs-"
    "symptom direction; retrieve_raw_evidence is for validating a concrete "
    "mechanism or resolving contradictory summaries; find_unexplained_symptoms "
    "tests whether a candidate explains the remaining symptom set. Choose the "
    "highest information-gain tool for the named unresolved competition. Do "
    "not prematurely narrow from many active hypotheses to two candidates "
    "unless the context names that unresolved pair; broad triage should cover "
    "all plausible active candidates or all tied candidates. Do not repeat a "
    "used tool_name+args signature. For action.expected_outcomes, keys MUST "
    "be hypothesis IDs, not outcome labels. Every action MUST target at least "
    "two competing hypothesis IDs; never output a single-target action. For a single-pair propagation "
    "test, expected_outcomes for 'no path found' may weaken the proposed "
    "source but must not support the opposite root by itself.\n\n"
    "Decision doctrine: reason backwards from the remaining budget. If a "
    "nomination is guardrail-ready, prefer nomination over more exploration. "
    "If this is the last action opportunity, choose only an action that can "
    "resolve a listed blocker. If decision_mode is nomination_only, action is "
    "invalid and you must nominate the best grounded candidate from existing "
    "evidence. A normal nomination requires all non-nominated competitors to "
    "be addressed by contradiction evidence; addressing only one competitor is "
    "not enough when more candidates remain. Do not nominate from one evidence "
    "type alone while action budget remains; seek an orthogonal corroborating "
    "tool. For assessments, outcome=inconclusive requires empty links and "
    "empty status_updates; any support/contradiction link requires "
    "outcome=informative. For challenge resolution, assess the actual proposal "
    "and evidence; do not silently switch tests."
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
        "max_rounds": snapshot.max_rounds,
        "remaining_rounds": snapshot.remaining_rounds,
        "decision_mode": snapshot.decision_mode,
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


def _serialize_assessment_for_context(assessment) -> Mapping[str, Any]:
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


def _evidence_digest(atom: EvidenceAtom) -> Mapping[str, Any]:
    observation = dict(atom.observation)
    digest: dict[str, Any] = {
        "evidence_id": atom.evidence_id,
        "modality": atom.modality,
        "component_scope": list(atom.component_scope),
        "time_window": list(atom.time_window),
        "missing_fields": list(atom.missing_fields),
        "reliability_note": atom.reliability_note,
    }

    if atom.modality == "reason_signature":
        rows = observation.get("reason_signatures", ())
        digest["summary"] = [
            {
                "component": row.get("component"),
                "reason_family": row.get("reason_family"),
                "signals": row.get("signals", ()),
                "magnitude": row.get("magnitude"),
            }
            for row in rows
            if isinstance(row, Mapping)
        ]
    elif atom.modality == "onset":
        rows = observation.get("observed_onsets", ())
        first_by_component: dict[str, Any] = {}
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            component = str(row.get("component", ""))
            onset_time = row.get("onset_time")
            if component and (
                component not in first_by_component
                or onset_time < first_by_component[component].get("onset_time")
            ):
                first_by_component[component] = {
                    "onset_time": onset_time,
                    "signal": row.get("signal"),
                }
        digest["summary"] = {
            "first_onset_by_component": first_by_component,
            "observed_onset_count": len(rows) if isinstance(rows, (list, tuple)) else 0,
            "missing_components": observation.get("missing_components", ()),
        }
    elif atom.modality == "trace":
        paths = observation.get("paths", ())
        digest["summary"] = {
            "path_count": len(paths) if isinstance(paths, (list, tuple)) else 0,
            "paths_present": bool(paths),
        }
    elif atom.modality == "propagation":
        digest["summary"] = observation.get("propagation_facts", observation)
    else:
        digest["summary"] = {
            "observation_keys": sorted(str(k) for k in observation.keys()),
        }
    return canonicalize_json_value(digest)


def _build_round_trace(
    *,
    snapshot: LeadTournamentSnapshot,
    graph: EvidenceGraph | None,
) -> tuple[Mapping[str, Any], ...]:
    rows: list[Mapping[str, Any]] = []
    for step in snapshot.audit_steps:
        atom = graph.evidence_by_id.get(step.evidence_id) if graph is not None else None
        rows.append(canonicalize_json_value({
            "round_index": step.round_index,
            "action": _serialize_action(step.action),
            "evidence_digest": _evidence_digest(atom) if atom is not None else {
                "evidence_id": step.evidence_id,
                "evidence_digest_unavailable": True,
            },
            "applied_assessment": _serialize_assessment_for_context(step.assessment),
            "status_delta": [
                {
                    "hypothesis_id": update.hypothesis_id,
                    "new_status": update.new_status.value,
                    "rationale": update.rationale,
                }
                for update in step.assessment.status_updates
            ],
            "feedback": {
                "outcome": step.assessment.outcome.value,
                "support_links": [
                    link.hypothesis_id
                    for link in step.assessment.links
                    if link.relation.value == "supports"
                ],
                "contradiction_links": [
                    link.hypothesis_id
                    for link in step.assessment.links
                    if link.relation.value == "contradicts"
                ],
            },
        }))
    return tuple(rows)


def _build_used_queries(snapshot: LeadTournamentSnapshot) -> tuple[Mapping[str, Any], ...]:
    rows = []
    for step in snapshot.audit_steps:
        rows.append(canonicalize_json_value({
            "round_index": step.round_index,
            "tool_name": step.action.tool_name,
            "query_signature": step.action.query_signature(),
            "args": dict(step.action.args),
            "result_evidence_id": step.evidence_id,
            "assessment_outcome": step.assessment.outcome.value,
            "do_not_repeat_reason": (
                "This exact tool_name+args signature has already been executed."
            ),
        }))
    return tuple(rows)


def _nomination_readiness(snapshot: LeadTournamentSnapshot) -> Mapping[str, Any]:
    hypotheses = list(snapshot.hypotheses)
    rows = []
    for h in hypotheses:
        support_ids = tuple(h.supporting_evidence_ids)
        competitor_ids = tuple(
            c.hypothesis_id
            for c in hypotheses
            if c.hypothesis_id != h.hypothesis_id
        )
        addressable = tuple(
            c.hypothesis_id
            for c in hypotheses
            if c.hypothesis_id != h.hypothesis_id
            and c.status.value in ("weakened", "refuted")
            and c.contradicting_evidence_ids
        )
        unresolved = tuple(
            c.hypothesis_id
            for c in hypotheses
            if c.hypothesis_id != h.hypothesis_id
            and c.hypothesis_id not in addressable
        )
        blockers: list[str] = []
        if h.status.value != "supported":
            blockers.append(f"candidate status is {h.status.value}, not supported")
        if not support_ids:
            blockers.append("candidate has no linked supporting evidence")
        if unresolved:
            blockers.append(
                "unaddressed competitors remain: " + ", ".join(unresolved)
            )
        rows.append(canonicalize_json_value({
            "hypothesis_id": h.hypothesis_id,
            "root_component": h.root_component,
            "status": h.status.value,
            "all_competitor_ids": list(competitor_ids),
            "supporting_evidence_ids": list(support_ids),
            "contradicting_evidence_ids": list(h.contradicting_evidence_ids),
            "addressable_competitor_ids": list(addressable),
            "unresolved_competitor_ids": list(unresolved),
            "nomination_ready_by_guardrail": (
                h.status.value == "supported"
                and bool(support_ids)
                and set(addressable) == set(competitor_ids)
            ),
            "triplet_grounding_candidates": {
                "component_evidence_ids": list(support_ids[:1]),
                "reason_evidence_ids": list(support_ids[:1]),
                "onset_evidence_ids": list(support_ids[:1]),
            },
            "blockers": blockers,
        }))
    return {
        "candidates": rows,
        "ready_hypothesis_ids": [
            row["hypothesis_id"]
            for row in rows
            if row["nomination_ready_by_guardrail"]
        ],
    }


def _decision_state(snapshot: LeadTournamentSnapshot) -> Mapping[str, Any]:
    hypotheses = list(snapshot.hypotheses)
    supported = [h.hypothesis_id for h in hypotheses if h.status.value == "supported"]
    active = [h.hypothesis_id for h in hypotheses if h.status.value == "active"]
    weakened = [h.hypothesis_id for h in hypotheses if h.status.value == "weakened"]
    refuted = [h.hypothesis_id for h in hypotheses if h.status.value == "refuted"]
    unresolved_pairs = []
    unresolved = [
        h for h in hypotheses if h.status.value in ("active", "supported", "survived")
    ]
    for i, left in enumerate(unresolved):
        for right in unresolved[i + 1:]:
            unresolved_pairs.append([left.hypothesis_id, right.hypothesis_id])

    readiness = _nomination_readiness(snapshot)
    ready_ids = list(readiness["ready_hypothesis_ids"])
    return canonicalize_json_value({
        "budget": {
            "round_index": snapshot.round_index,
            "max_rounds": snapshot.max_rounds,
            "remaining_rounds": snapshot.remaining_rounds,
            "decision_mode": snapshot.decision_mode,
        },
        "status_groups": {
            "supported": supported,
            "active": active,
            "weakened": weakened,
            "refuted": refuted,
        },
        "unresolved_pairwise_competitions": unresolved_pairs,
        "nomination_readiness": readiness,
        "recommended_policy": _recommended_policy(snapshot, ready_ids, unresolved_pairs),
    })


def _recommended_policy(
    snapshot: LeadTournamentSnapshot,
    ready_ids: Sequence[str],
    unresolved_pairs: Sequence[Sequence[str]],
) -> str:
    if snapshot.decision_mode == "nomination_only":
        if ready_ids:
            return (
                "Action is invalid because no action budget remains. Return a "
                "nomination using one ready_hypothesis_id."
            )
        return (
            "Action is invalid because no action budget remains. Return the best "
            "grounded nomination possible from existing evidence; mention unresolved "
            "uncertainty in rationale."
        )
    if snapshot.decision_mode == "last_action_or_nomination":
        if ready_ids and not unresolved_pairs:
            return "Nominate now; do not spend the final action when guardrails are met."
        return (
            "This is the final action opportunity. If you choose action, it must "
            "directly resolve the top unresolved_pairwise_competition or a listed "
            "nomination blocker."
        )
    if ready_ids and not unresolved_pairs:
        return "Nomination is allowed; prefer nomination over further exploration."
    return (
        "Choose the highest information-gain action only if it resolves a named "
        "unresolved competition or nomination blocker."
    )


def _evidence_interpretation_guidance() -> tuple[str, ...]:
    return (
        "Reason-signature magnitudes are local anomaly strength, not root-cause proof.",
        "A reason-signature result alone is not enough for nomination; corroborate it with onset or propagation evidence.",
        "Onset ties should be treated as ambiguity, not as evidence for all tied components.",
        "Propagation/trace evidence is most useful for source-vs-symptom direction.",
        "A single missing propagation path is weak evidence; it must not by itself support the opposite root.",
        "If two candidates have tied onset and one directed path is missing, keep the pair unresolved and seek an orthogonal signal.",
        "Do not repeat a used query signature; use round_trace feedback instead.",
        "A high-magnitude downstream symptom can be contradicted by missing propagation.",
    )


def _build_evidence_catalog_digest(
    *,
    graph: EvidenceGraph,
    evidence_ids: Sequence[str],
) -> tuple[Mapping[str, Any], ...]:
    seen: set[str] = set()
    rows = []
    for evidence_id in evidence_ids:
        if evidence_id in seen:
            continue
        seen.add(evidence_id)
        atom = graph.evidence_by_id.get(evidence_id)
        if atom is not None:
            rows.append(_evidence_digest(atom))
    return tuple(rows)


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


def _challenge_decision_state(snapshot: ChallengeSnapshot) -> Mapping[str, Any]:
    nominee = snapshot.nominated_hypothesis
    competitors = list(snapshot.competitor_hypotheses)
    return canonicalize_json_value({
        "nominated_hypothesis_id": nominee.hypothesis_id,
        "nominated_component": nominee.root_component,
        "nominated_supporting_evidence_ids": list(nominee.supporting_evidence_ids),
        "lead_nomination": {
            "supporting_evidence_ids": list(snapshot.lead_nomination.supporting_evidence_ids),
            "addressed_competitor_ids": list(snapshot.lead_nomination.addressed_competitor_ids),
            "rationale": snapshot.lead_nomination.rationale,
        },
        "eligible_competitors": [
            {
                "hypothesis_id": h.hypothesis_id,
                "root_component": h.root_component,
                "status": h.status.value,
                "supporting_evidence_ids": list(h.supporting_evidence_ids),
                "contradicting_evidence_ids": list(h.contradicting_evidence_ids),
            }
            for h in competitors
        ],
        "proposal_constraints": [
            "action.target_hypothesis_ids must equal nominated_hypothesis_id plus competitor_hypothesis_ids.",
            "action.expected_outcomes keys must be hypothesis IDs, not labels like path_found/no_path_found.",
            "Challenge only the nomination with a direct counterfactual or source-vs-symptom test.",
            "Do not repeat a Lead query unless the challenge explicitly tests a different direction.",
        ],
    })


def _lead_round_trace_for_challenge(
    *,
    snapshot: ChallengeSnapshot,
    graph: EvidenceGraph | None,
) -> tuple[Mapping[str, Any], ...]:
    rows = []
    for step in snapshot.lead_audit_steps:
        atom = graph.evidence_by_id.get(step.evidence_id) if graph is not None else None
        rows.append(canonicalize_json_value({
            "round_index": step.round_index,
            "action": _serialize_action(step.action),
            "evidence_digest": _evidence_digest(atom) if atom is not None else {
                "evidence_id": step.evidence_id,
                "evidence_digest_unavailable": True,
            },
            "applied_assessment": _serialize_assessment_for_context(step.assessment),
        }))
    return tuple(rows)


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


_LEAD_NOMINATION_ONLY_RESPONSE_SCHEMA = {
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
    context = {
        "instruction": (
            "You are the Lead Investigator. Review the tournament snapshot "
            "and decide your next step. Prefer nomination when the "
            "nomination_readiness guardrail is satisfied. Propose an action "
            "only when it resolves a named unresolved competition or blocker. "
            "If decision_mode is nomination_only, action is invalid."
        ),
        "snapshot": _serialize_lead_snapshot(snapshot),
        "decision_state": _decision_state(snapshot),
        "round_trace": list(_build_round_trace(snapshot=snapshot, graph=graph)),
        "used_queries": list(_build_used_queries(snapshot)),
        "evidence_interpretation_guidance": list(_evidence_interpretation_guidance()),
        "evidence_catalog_digest": list(
            _build_evidence_catalog_digest(
                graph=graph,
                evidence_ids=snapshot.evidence_ids,
            )
        ),
        "allowed_tool_names": list(allowed_tool_names),
        "tool_argument_contracts": _tool_argument_contracts(allowed_tool_names),
    }

    return build_policy_request(
        purpose="Lead Investigator: decide next action or nominate",
        context=context,
        response_schema=(
            _LEAD_NOMINATION_ONLY_RESPONSE_SCHEMA
            if snapshot.decision_mode == "nomination_only"
            else _LEAD_DECISION_RESPONSE_SCHEMA
        ),
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
            "and propose status updates consistent with the state machine. "
            "If outcome is inconclusive, links and status_updates MUST be empty. "
            "If you include any support or contradiction link, outcome MUST be informative."
        ),
        "snapshot": _serialize_lead_snapshot(snapshot),
        "decision_state_before_assessment": _decision_state(snapshot),
        "prior_round_trace": list(_build_round_trace(snapshot=snapshot, graph=None)),
        "action": _serialize_action(action),
        "new_evidence_digest": _evidence_digest(evidence),
        "evidence_interpretation_guidance": list(_evidence_interpretation_guidance()),
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
                    "<nominated_hypothesis_id>": "what evidence would mean for nominee",
                    "<competitor_hypothesis_id>": "what evidence would mean for competitor",
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
    context = {
        "instruction": (
            "You are the Challenger. Review the Lead nomination and "
            "propose a challenge. Identify a weakness in the nominated "
            "hypothesis and design a discriminative action to test it. "
            "Your action.expected_outcomes keys MUST exactly match the "
            "hypothesis IDs in action.target_hypothesis_ids."
        ),
        "snapshot": _serialize_challenge_snapshot(snapshot),
        "challenge_decision_state": _challenge_decision_state(snapshot),
        "lead_round_trace": list(
            _lead_round_trace_for_challenge(snapshot=snapshot, graph=graph)
        ),
        "evidence_interpretation_guidance": list(_evidence_interpretation_guidance()),
        "evidence_catalog_digest": list(
            _build_evidence_catalog_digest(
                graph=graph,
                evidence_ids=all_evidence_ids,
            )
        ),
        "allowed_tool_names": list(allowed_tool_names),
        "tool_argument_contracts": _tool_argument_contracts(allowed_tool_names),
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
    graph: EvidenceGraph | None = None,
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
            "the verdict is 'inconclusive'. Judge the actual proposal action "
            "and new_evidence; do not silently switch to a different test. "
            "If verdict is inconclusive, assessment.links and "
            "assessment.status_updates MUST be empty."
        ),
        "snapshot": _serialize_challenge_snapshot(snapshot),
        "challenge_decision_state": _challenge_decision_state(snapshot),
        "lead_round_trace": list(
            _lead_round_trace_for_challenge(snapshot=snapshot, graph=graph)
        ),
        "proposal": _serialize_challenge_proposal(proposal),
        "new_evidence_digest": _evidence_digest(evidence),
        "evidence_interpretation_guidance": list(_evidence_interpretation_guidance()),
    }

    return build_policy_request(
        purpose="Challenger: assess challenge resolution",
        context=context,
        response_schema=_CHALLENGE_RESOLUTION_RESPONSE_SCHEMA,
        attempt_index=attempt_index,
        repair_error=repair_error,
    )


def _tool_argument_contracts(allowed_tool_names: Sequence[str]) -> Mapping[str, Any]:
    contracts = {
        "compare_onset_order": {
            "component_scope": ["component-a", "component-b"],
            "signal_scope": ["cpu", "latency"],
            "time_window": [0.0, 1.0],
        },
        "inspect_reason_signature": {
            "component_scope": ["component-a", "component-b"],
            "reason_by_component": {"component-a": "cpu saturation"},
            "time_window": [0.0, 1.0],
        },
        "inspect_trace_path": {
            "source_component": "component-a",
            "target_component": "component-b",
            "time_window": [0.0, 1.0],
            "max_hops": 6,
            "max_paths": 10,
        },
        "retrieve_raw_evidence": {
            "modality": "metric",
            "component_scope": ["component-a", "component-b"],
            "time_window": [0.0, 1.0],
            "limit": 20,
        },
        "check_propagation_consistency": {
            "source_component": "component-a",
            "symptom_components": ["component-b"],
            "time_window": [0.0, 1.0],
        },
        "find_unexplained_symptoms": {
            "explained_components": ["component-a"],
            "time_window": [0.0, 1.0],
            "limit": 20,
        },
    }
    return {
        name: contracts[name]
        for name in allowed_tool_names
        if name in contracts
    }
