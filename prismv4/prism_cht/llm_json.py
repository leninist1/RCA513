"""Strict JSON parsing for structured LLM output in PRISM-CHT.

Every parser in this module rejects markdown fences, extra text,
NaN/Infinity, duplicate keys, and unknown fields.  No silent fallback
or type coercion is performed.  Errors are always wrapped in
``StructuredOutputError``.
"""

from __future__ import annotations

import json
import math
from typing import Any, Mapping, Sequence, Tuple

from .action_schema import DiscriminativeAction
from .llm_types import StructuredOutputError
from .tournament_types import (
    AssessmentOutcome,
    EvidenceAssessment,
    EvidenceLinkProposal,
    EvidenceRelation,
    HypothesisStatus,
    HypothesisStatusUpdate,
    LeadNomination,
    TripletEvidenceCoverage,
)
from .challenge_types import (
    ChallengeProposal,
    ChallengeResolution,
    ChallengeVerdict,
)


# ===========================================================================
# Core JSON validation
# ===========================================================================


def _strip_and_validate_text(text: str, *, max_chars: int, context: str) -> str:
    """Strip leading/trailing whitespace and validate basic constraints."""
    if not isinstance(text, str):
        raise StructuredOutputError(
            f"{context}: text must be str, got {type(text).__name__}"
        )
    stripped = text.strip()
    if not stripped:
        raise StructuredOutputError(f"{context}: text is empty")
    if len(stripped) > max_chars:
        raise StructuredOutputError(
            f"{context}: text length {len(stripped)} exceeds "
            f"max_chars {max_chars}"
        )
    return stripped


def _reject_markdown_fence(text: str, *, context: str) -> None:
    """Reject text wrapped in markdown code fences."""
    # Check for ```json or ``` at the start after stripping
    if text.startswith("```"):
        raise StructuredOutputError(
            f"{context}: text starts with markdown code fence; "
            f"output must be a plain JSON object only"
        )
    if text.endswith("```"):
        raise StructuredOutputError(
            f"{context}: text ends with markdown code fence; "
            f"output must be a plain JSON object only"
        )


def _parse_json_with_guards(
    text: str, *, context: str
) -> Any:
    """Parse JSON string with NaN/Infinity/duplicate-key guards."""

    def _raise_on_nan_inf(constant: str) -> Any:
        raise StructuredOutputError(
            f"{context}: JSON contains {constant}; "
            f"NaN, Infinity, -Infinity are not allowed"
        )

    seen_keys: list[str] = []

    def _detect_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise StructuredOutputError(
                    f"{context}: duplicate key '{key}' in JSON object"
                )
            result[key] = value
        return result

    try:
        parsed = json.loads(
            text,
            object_pairs_hook=_detect_duplicate_keys,
            parse_constant=_raise_on_nan_inf,
        )
    except json.JSONDecodeError as e:
        raise StructuredOutputError(
            f"{context}: invalid JSON: {e}"
        ) from e
    except StructuredOutputError:
        raise
    except Exception as e:
        raise StructuredOutputError(
            f"{context}: JSON parsing failed: {e}"
        ) from e

    return parsed


# ===========================================================================
# Public: parse_json_object
# ===========================================================================


def parse_json_object(
    text: str,
    *,
    max_chars: int = 65536,
) -> Mapping[str, Any]:
    """Parse *text* as a single, complete JSON object.

    Rejects: markdown fences, extra text, NaN/Infinity, duplicate keys,
    non-object top-level, and text exceeding *max_chars*.
    """
    context = "parse_json_object"
    stripped = _strip_and_validate_text(text, max_chars=max_chars, context=context)
    _reject_markdown_fence(stripped, context=context)

    # Ensure the text starts with '{' and ends with '}' — reject
    # leading/trailing non-whitespace text.
    # But we already stripped, so check the first non-space char.
    # After stripping, first char must be '{'
    if not stripped.startswith("{"):
        raise StructuredOutputError(
            f"{context}: text does not start with '{{'; "
            f"found leading non-JSON content"
        )
    if not stripped.endswith("}"):
        raise StructuredOutputError(
            f"{context}: text does not end with '}}'; "
            f"found trailing non-JSON content"
        )

    # Sanity: try to detect extra text by checking if there's a second
    # JSON object or array.  We do this by attempting to parse once
    # and ensuring the entire string is consumed.
    parsed = _parse_json_with_guards(stripped, context=context)

    if not isinstance(parsed, dict):
        raise StructuredOutputError(
            f"{context}: top-level value must be a JSON object, "
            f"got {type(parsed).__name__}"
        )

    return parsed


# ===========================================================================
# Field validators
# ===========================================================================


def require_exact_keys(
    value: Mapping[str, Any],
    *,
    required: set[str],
    optional: set[str] = frozenset(),
    context: str,
) -> None:
    """Validate that *value* has exactly the expected keys.

    - Missing required keys → ``StructuredOutputError``.
    - Unknown keys not in required ∪ optional → ``StructuredOutputError``.
    """
    actual = set(value.keys())
    allowed = required | optional
    missing = required - actual
    if missing:
        raise StructuredOutputError(
            f"{context}: missing required fields: {sorted(missing)}"
        )
    extra = actual - allowed
    if extra:
        raise StructuredOutputError(
            f"{context}: unknown fields: {sorted(extra)}"
        )


def require_mapping(
    value: Any, *, context: str
) -> Mapping[str, Any]:
    """Require *value* to be a Mapping (dict)."""
    if not isinstance(value, Mapping):
        raise StructuredOutputError(
            f"{context}: expected a JSON object, got {type(value).__name__}"
        )
    return value


def require_string(value: Any, *, context: str, allow_empty: bool = False) -> str:
    """Require *value* to be a str."""
    if not isinstance(value, str):
        raise StructuredOutputError(
            f"{context}: expected string, got {type(value).__name__}"
        )
    if not allow_empty and not value.strip():
        raise StructuredOutputError(
            f"{context}: string must be non-empty"
        )
    return value


def require_int(value: Any, *, context: str) -> int:
    """Require *value* to be an int (not bool, not float)."""
    if isinstance(value, bool):
        raise StructuredOutputError(
            f"{context}: expected int, got bool (True/False are not integers)"
        )
    if not isinstance(value, int):
        raise StructuredOutputError(
            f"{context}: expected int, got {type(value).__name__}"
        )
    return value


def require_string_tuple(value: Any, *, context: str) -> tuple[str, ...]:
    """Require *value* to be a list/tuple of non-empty strings."""
    if not isinstance(value, (list, tuple)):
        raise StructuredOutputError(
            f"{context}: expected a JSON array, got {type(value).__name__}"
        )
    result: list[str] = []
    for i, item in enumerate(value):
        result.append(
            require_string(item, context=f"{context}[{i}]")
        )
    return tuple(result)


def require_mapping_of_strings(
    value: Any, *, context: str
) -> Mapping[str, str]:
    """Require *value* to be a Mapping of str → str."""
    m = require_mapping(value, context=context)
    result: dict[str, str] = {}
    for key, val in m.items():
        k = require_string(key, context=f"{context} key")
        v = require_string(val, context=f"{context}.{k}")
        result[k] = v
    return result


def require_sequence(value: Any, *, context: str) -> tuple[Any, ...]:
    """Require *value* to be a list or tuple."""
    if not isinstance(value, (list, tuple)):
        raise StructuredOutputError(
            f"{context}: expected a JSON array, got {type(value).__name__}"
        )
    return tuple(value)


# ===========================================================================
# Action parser helper (shared by lead decision and challenge proposal)
# ===========================================================================


def _parse_action_payload(
    value: Mapping[str, Any], *, context: str
) -> DiscriminativeAction:
    """Parse a DiscriminativeAction from a validated JSON payload dict."""
    require_exact_keys(
        value,
        required={
            "action_id", "action_type", "target_hypothesis_ids",
            "question", "tool_name", "args", "expected_outcomes",
            "why_discriminative",
        },
        context=f"{context}.action_payload",
    )

    action_id = require_string(value["action_id"], context=f"{context}.action_id")
    action_type = require_string(value["action_type"], context=f"{context}.action_type")

    target_ids = require_string_tuple(
        value["target_hypothesis_ids"],
        context=f"{context}.target_hypothesis_ids",
    )

    question = require_string(value["question"], context=f"{context}.question")
    tool_name = require_string(value["tool_name"], context=f"{context}.tool_name")

    args = require_mapping(value["args"], context=f"{context}.args")

    expected_outcomes = require_mapping_of_strings(
        value["expected_outcomes"],
        context=f"{context}.expected_outcomes",
    )

    why = require_string(
        value["why_discriminative"], context=f"{context}.why_discriminative"
    )

    return DiscriminativeAction(
        action_id=action_id,
        action_type=action_type,
        target_hypothesis_ids=target_ids,
        question=question,
        tool_name=tool_name,
        args=dict(args),
        expected_outcomes=dict(expected_outcomes),
        why_discriminative=why,
    )


# ===========================================================================
# Lead decision parsers
# ===========================================================================


def parse_lead_policy_decision(
    text: str,
) -> DiscriminativeAction | LeadNomination:
    """Parse an LLM output as either a Lead action or nomination.

    Dispatches on the top-level ``"kind"`` field:
    - ``"action"`` → ``DiscriminativeAction``
    - ``"nomination"`` → ``LeadNomination``
    """
    context = "parse_lead_policy_decision"
    obj = parse_json_object(text)

    require_exact_keys(
        obj,
        required={"kind", "payload"},
        context=context,
    )

    kind = require_string(obj["kind"], context=f"{context}.kind")
    payload = require_mapping(obj["payload"], context=f"{context}.payload")

    if kind == "action":
        return _parse_action_payload(payload, context=context)
    elif kind == "nomination":
        return _parse_nomination_payload(payload, context=context)
    else:
        raise StructuredOutputError(
            f"{context}: kind must be 'action' or 'nomination', "
            f"got '{kind}'"
        )


def _parse_nomination_payload(
    value: Mapping[str, Any], *, context: str
) -> LeadNomination:
    """Parse a LeadNomination from a validated JSON payload dict."""
    require_exact_keys(
        value,
        required={
            "hypothesis_id",
            "supporting_evidence_ids",
            "addressed_competitor_ids",
            "triplet_grounding",
            "rationale",
        },
        context=f"{context}.nomination_payload",
    )

    hid = require_string(value["hypothesis_id"], context=f"{context}.hypothesis_id")
    supporting = require_string_tuple(
        value["supporting_evidence_ids"],
        context=f"{context}.supporting_evidence_ids",
    )
    addressed = require_string_tuple(
        value["addressed_competitor_ids"],
        context=f"{context}.addressed_competitor_ids",
    )
    rationale = require_string(value["rationale"], context=f"{context}.rationale")

    grounding_raw = require_mapping(
        value["triplet_grounding"], context=f"{context}.triplet_grounding"
    )

    require_exact_keys(
        grounding_raw,
        required={
            "component_evidence_ids",
            "reason_evidence_ids",
            "onset_evidence_ids",
        },
        context=f"{context}.triplet_grounding",
    )

    component_ids = require_string_tuple(
        grounding_raw["component_evidence_ids"],
        context=f"{context}.triplet_grounding.component_evidence_ids",
    )
    reason_ids = require_string_tuple(
        grounding_raw["reason_evidence_ids"],
        context=f"{context}.triplet_grounding.reason_evidence_ids",
    )
    onset_ids = require_string_tuple(
        grounding_raw["onset_evidence_ids"],
        context=f"{context}.triplet_grounding.onset_evidence_ids",
    )

    grounding = TripletEvidenceCoverage(
        component_evidence_ids=component_ids,
        reason_evidence_ids=reason_ids,
        onset_evidence_ids=onset_ids,
    )

    return LeadNomination(
        hypothesis_id=hid,
        supporting_evidence_ids=supporting,
        addressed_competitor_ids=addressed,
        triplet_grounding=grounding,
        rationale=rationale,
    )


# ===========================================================================
# Evidence assessment parser
# ===========================================================================


def _parse_links(
    value: Any, *, context: str
) -> tuple[EvidenceLinkProposal, ...]:
    """Parse an array of EvidenceLinkProposal objects."""
    seq = require_sequence(value, context=context)
    links: list[EvidenceLinkProposal] = []
    for i, item in enumerate(seq):
        item_ctx = f"{context}[{i}]"
        m = require_mapping(item, context=item_ctx)
        require_exact_keys(
            m,
            required={"hypothesis_id", "evidence_id", "relation", "rationale"},
            context=item_ctx,
        )
        hid = require_string(m["hypothesis_id"], context=f"{item_ctx}.hypothesis_id")
        eid = require_string(m["evidence_id"], context=f"{item_ctx}.evidence_id")
        rel_str = require_string(m["relation"], context=f"{item_ctx}.relation")
        rationale = require_string(m["rationale"], context=f"{item_ctx}.rationale")

        try:
            relation = EvidenceRelation(rel_str)
        except ValueError:
            raise StructuredOutputError(
                f"{item_ctx}.relation: invalid value '{rel_str}'; "
                f"must be 'supports' or 'contradicts'"
            )

        links.append(
            EvidenceLinkProposal(
                hypothesis_id=hid,
                evidence_id=eid,
                relation=relation,
                rationale=rationale,
            )
        )
    return tuple(links)


def _parse_status_updates(
    value: Any, *, context: str
) -> tuple[HypothesisStatusUpdate, ...]:
    """Parse an array of HypothesisStatusUpdate objects."""
    seq = require_sequence(value, context=context)
    updates: list[HypothesisStatusUpdate] = []
    for i, item in enumerate(seq):
        item_ctx = f"{context}[{i}]"
        m = require_mapping(item, context=item_ctx)
        require_exact_keys(
            m,
            required={"hypothesis_id", "new_status", "rationale"},
            context=item_ctx,
        )
        hid = require_string(m["hypothesis_id"], context=f"{item_ctx}.hypothesis_id")
        status_str = require_string(m["new_status"], context=f"{item_ctx}.new_status")
        rationale = require_string(m["rationale"], context=f"{item_ctx}.rationale")

        try:
            new_status = HypothesisStatus(status_str)
        except ValueError:
            raise StructuredOutputError(
                f"{item_ctx}.new_status: invalid value '{status_str}'; "
                f"must be one of the HypothesisStatus enum values"
            )

        updates.append(
            HypothesisStatusUpdate(
                hypothesis_id=hid,
                new_status=new_status,
                rationale=rationale,
            )
        )
    return tuple(updates)


def parse_evidence_assessment(
    text: str,
) -> EvidenceAssessment:
    """Parse an LLM output as an ``EvidenceAssessment``."""
    context = "parse_evidence_assessment"
    obj = parse_json_object(text)

    require_exact_keys(
        obj,
        required={
            "action_id", "evidence_id", "outcome",
            "links", "status_updates", "rationale",
        },
        context=context,
    )

    action_id = require_string(obj["action_id"], context=f"{context}.action_id")
    evidence_id = require_string(obj["evidence_id"], context=f"{context}.evidence_id")
    outcome_str = require_string(obj["outcome"], context=f"{context}.outcome")
    rationale = require_string(obj["rationale"], context=f"{context}.rationale")

    try:
        outcome = AssessmentOutcome(outcome_str)
    except ValueError:
        raise StructuredOutputError(
            f"{context}.outcome: invalid value '{outcome_str}'; "
            f"must be 'informative' or 'inconclusive'"
        )

    links = _parse_links(obj["links"], context=f"{context}.links")
    status_updates = _parse_status_updates(
        obj["status_updates"], context=f"{context}.status_updates"
    )

    return EvidenceAssessment(
        action_id=action_id,
        evidence_id=evidence_id,
        outcome=outcome,
        links=links,
        status_updates=status_updates,
        rationale=rationale,
    )


# ===========================================================================
# Challenge proposal parser
# ===========================================================================


def parse_challenge_proposal(
    text: str,
) -> ChallengeProposal:
    """Parse an LLM output as a ``ChallengeProposal``."""
    context = "parse_challenge_proposal"
    obj = parse_json_object(text)

    require_exact_keys(
        obj,
        required={
            "challenge_id",
            "nominated_hypothesis_id",
            "competitor_hypothesis_ids",
            "challenge_claim",
            "falsification_target",
            "action",
            "rationale",
        },
        context=context,
    )

    challenge_id = require_string(
        obj["challenge_id"], context=f"{context}.challenge_id"
    )
    nominated_id = require_string(
        obj["nominated_hypothesis_id"],
        context=f"{context}.nominated_hypothesis_id",
    )
    competitor_ids = require_string_tuple(
        obj["competitor_hypothesis_ids"],
        context=f"{context}.competitor_hypothesis_ids",
    )
    challenge_claim = require_string(
        obj["challenge_claim"], context=f"{context}.challenge_claim"
    )
    falsification_target = require_string(
        obj["falsification_target"],
        context=f"{context}.falsification_target",
    )
    rationale = require_string(
        obj["rationale"], context=f"{context}.rationale"
    )

    action_payload = require_mapping(
        obj["action"], context=f"{context}.action"
    )
    action = _parse_action_payload(action_payload, context=f"{context}.action")

    return ChallengeProposal(
        challenge_id=challenge_id,
        nominated_hypothesis_id=nominated_id,
        competitor_hypothesis_ids=competitor_ids,
        challenge_claim=challenge_claim,
        falsification_target=falsification_target,
        action=action,
        rationale=rationale,
    )


# ===========================================================================
# Challenge resolution parser
# ===========================================================================


def parse_challenge_resolution(
    text: str,
) -> ChallengeResolution:
    """Parse an LLM output as a ``ChallengeResolution``."""
    context = "parse_challenge_resolution"
    obj = parse_json_object(text)

    require_exact_keys(
        obj,
        required={
            "challenge_id",
            "action_id",
            "evidence_id",
            "verdict",
            "assessment",
            "rationale",
        },
        context=context,
    )

    challenge_id = require_string(
        obj["challenge_id"], context=f"{context}.challenge_id"
    )
    action_id = require_string(
        obj["action_id"], context=f"{context}.action_id"
    )
    evidence_id = require_string(
        obj["evidence_id"], context=f"{context}.evidence_id"
    )
    verdict_str = require_string(
        obj["verdict"], context=f"{context}.verdict"
    )
    rationale = require_string(
        obj["rationale"], context=f"{context}.rationale"
    )

    try:
        verdict = ChallengeVerdict(verdict_str)
    except ValueError:
        raise StructuredOutputError(
            f"{context}.verdict: invalid value '{verdict_str}'; "
            f"must be 'nomination_survived', 'nomination_refuted', "
            f"or 'inconclusive'"
        )

    assessment_payload = require_mapping(
        obj["assessment"], context=f"{context}.assessment"
    )

    # Parse the nested EvidenceAssessment using the same logic
    require_exact_keys(
        assessment_payload,
        required={
            "action_id", "evidence_id", "outcome",
            "links", "status_updates", "rationale",
        },
        context=f"{context}.assessment",
    )

    assess_action_id = require_string(
        assessment_payload["action_id"],
        context=f"{context}.assessment.action_id",
    )
    assess_evidence_id = require_string(
        assessment_payload["evidence_id"],
        context=f"{context}.assessment.evidence_id",
    )
    assess_outcome_str = require_string(
        assessment_payload["outcome"],
        context=f"{context}.assessment.outcome",
    )
    assess_rationale = require_string(
        assessment_payload["rationale"],
        context=f"{context}.assessment.rationale",
    )

    try:
        assess_outcome = AssessmentOutcome(assess_outcome_str)
    except ValueError:
        raise StructuredOutputError(
            f"{context}.assessment.outcome: invalid value "
            f"'{assess_outcome_str}'; must be 'informative' or 'inconclusive'"
        )

    assess_links = _parse_links(
        assessment_payload["links"],
        context=f"{context}.assessment.links",
    )
    assess_status_updates = _parse_status_updates(
        assessment_payload["status_updates"],
        context=f"{context}.assessment.status_updates",
    )

    assessment = EvidenceAssessment(
        action_id=assess_action_id,
        evidence_id=assess_evidence_id,
        outcome=assess_outcome,
        links=assess_links,
        status_updates=assess_status_updates,
        rationale=assess_rationale,
    )

    return ChallengeResolution(
        challenge_id=challenge_id,
        action_id=action_id,
        evidence_id=evidence_id,
        verdict=verdict,
        assessment=assessment,
        rationale=rationale,
    )
