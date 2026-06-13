"""Tests for llm_json.py — strict JSON parsing and field validators."""

import json

import pytest

from prismv4.prism_cht.llm_json import (
    parse_json_object,
    require_exact_keys,
    require_int,
    require_mapping,
    require_mapping_of_strings,
    require_sequence,
    require_string,
    require_string_tuple,
)
from prismv4.prism_cht.llm_types import StructuredOutputError


# ===========================================================================
# parse_json_object
# ===========================================================================


class TestParseJsonObject:
    def test_valid_simple_object(self):
        result = parse_json_object('{"key": "value"}')
        assert result == {"key": "value"}

    def test_valid_nested_object(self):
        result = parse_json_object('{"outer": {"inner": [1, 2, 3]}}')
        assert result == {"outer": {"inner": [1, 2, 3]}}

    def test_whitespace_handling_deterministic(self):
        a = parse_json_object('{"key":"value"}')
        b = parse_json_object('  { "key" : "value" }  \n')
        assert a == b

    def test_rejects_malformed_json_structure(self):
        """Text with '{' but no closing '}' is rejected before full JSON parse."""
        with pytest.raises(StructuredOutputError, match="does not end with"):
            parse_json_object("{bad json")

    def test_rejects_non_object_array(self):
        with pytest.raises(StructuredOutputError, match="does not start with"):
            parse_json_object('[1, 2, 3]')

    def test_rejects_non_object_string(self):
        with pytest.raises(StructuredOutputError, match="does not start with"):
            parse_json_object('"just a string"')

    def test_rejects_non_object_number(self):
        with pytest.raises(StructuredOutputError, match="does not start with"):
            parse_json_object("42")

    def test_rejects_non_object_boolean(self):
        with pytest.raises(StructuredOutputError, match="does not start with"):
            parse_json_object("true")

    def test_rejects_non_object_null(self):
        with pytest.raises(StructuredOutputError, match="does not start with"):
            parse_json_object("null")

    def test_rejects_markdown_fence_start(self):
        with pytest.raises(StructuredOutputError, match="markdown code fence"):
            parse_json_object('```json\n{"key": "value"}')

    def test_rejects_markdown_fence_end(self):
        with pytest.raises(StructuredOutputError, match="markdown code fence"):
            parse_json_object('{"key": "value"}\n```')

    def test_rejects_nan(self):
        with pytest.raises(StructuredOutputError, match="NaN"):
            parse_json_object('{"value": NaN}')

    def test_rejects_infinity(self):
        with pytest.raises(StructuredOutputError, match="Infinity"):
            parse_json_object('{"value": Infinity}')

    def test_rejects_negative_infinity(self):
        with pytest.raises(StructuredOutputError, match="-Infinity"):
            parse_json_object('{"value": -Infinity}')

    def test_rejects_duplicate_keys(self):
        with pytest.raises(StructuredOutputError, match="duplicate key"):
            parse_json_object('{"key": 1, "key": 2}')

    def test_rejects_extra_text_after_object(self):
        with pytest.raises(StructuredOutputError, match="does not end with"):
            parse_json_object('{"key": "value"} extra text')

    def test_rejects_empty_string(self):
        with pytest.raises(StructuredOutputError, match="empty"):
            parse_json_object("")

    def test_rejects_whitespace_only(self):
        with pytest.raises(StructuredOutputError, match="empty"):
            parse_json_object("   \t\n  ")

    def test_rejects_exceeding_max_chars(self):
        obj = json.dumps({"key": "value"})
        with pytest.raises(StructuredOutputError, match="exceeds"):
            parse_json_object(obj, max_chars=len(obj) - 1)

    def test_accepts_unicode(self):
        result = parse_json_object('{"key": "\u00e9"}')
        assert result == {"key": "\u00e9"}

    def test_accepts_empty_object(self):
        result = parse_json_object("{}")
        assert result == {}

    def test_rejects_non_string_input(self):
        with pytest.raises(StructuredOutputError, match="text must be str"):
            parse_json_object(42)  # type: ignore

    def test_error_includes_context_tag(self):
        with pytest.raises(StructuredOutputError, match="parse_json_object"):
            parse_json_object("{broken")


# ===========================================================================
# require_exact_keys
# ===========================================================================


class TestRequireExactKeys:
    def test_all_required_present(self):
        require_exact_keys(
            {"a": 1, "b": 2},
            required={"a", "b"},
            context="test",
        )

    def test_missing_required_key(self):
        with pytest.raises(StructuredOutputError, match="missing"):
            require_exact_keys(
                {"a": 1},
                required={"a", "b"},
                context="test",
            )

    def test_unknown_key_rejected(self):
        with pytest.raises(StructuredOutputError, match="unknown"):
            require_exact_keys(
                {"a": 1, "b": 2, "c": 3},
                required={"a", "b"},
                context="test",
            )

    def test_optional_keys_allowed(self):
        require_exact_keys(
            {"a": 1, "b": 2, "c": 3},
            required={"a", "b"},
            optional={"c"},
            context="test",
        )

    def test_optional_keys_not_required(self):
        require_exact_keys(
            {"a": 1, "b": 2},
            required={"a", "b"},
            optional={"c"},
            context="test",
        )

    def test_error_includes_context(self):
        with pytest.raises(StructuredOutputError, match="my_context"):
            require_exact_keys(
                {"a": 1},
                required={"a", "b"},
                context="my_context",
            )


# ===========================================================================
# require_string
# ===========================================================================


class TestRequireString:
    def test_valid_string(self):
        result = require_string("hello", context="test")
        assert result == "hello"

    def test_rejects_non_string(self):
        with pytest.raises(StructuredOutputError, match="expected string"):
            require_string(42, context="test")

    def test_rejects_empty_by_default(self):
        with pytest.raises(StructuredOutputError, match="non-empty"):
            require_string("", context="test")

    def test_allows_empty_when_configured(self):
        result = require_string("", context="test", allow_empty=True)
        assert result == ""

    def test_rejects_boolean(self):
        with pytest.raises(StructuredOutputError, match="expected string"):
            require_string(True, context="test")


# ===========================================================================
# require_int
# ===========================================================================


class TestRequireInt:
    def test_valid_int(self):
        result = require_int(42, context="test")
        assert result == 42

    def test_rejects_float(self):
        with pytest.raises(StructuredOutputError, match="expected int"):
            require_int(3.14, context="test")

    def test_rejects_bool(self):
        with pytest.raises(StructuredOutputError, match="bool"):
            require_int(True, context="test")
        with pytest.raises(StructuredOutputError, match="bool"):
            require_int(False, context="test")

    def test_rejects_string(self):
        with pytest.raises(StructuredOutputError, match="expected int"):
            require_int("42", context="test")

    def test_rejects_none(self):
        with pytest.raises(StructuredOutputError, match="expected int"):
            require_int(None, context="test")


# ===========================================================================
# require_mapping
# ===========================================================================


class TestRequireMapping:
    def test_valid_dict(self):
        result = require_mapping({"a": 1}, context="test")
        assert result == {"a": 1}

    def test_rejects_list(self):
        with pytest.raises(StructuredOutputError, match="expected a JSON object"):
            require_mapping([1, 2], context="test")

    def test_rejects_string(self):
        with pytest.raises(StructuredOutputError, match="expected a JSON object"):
            require_mapping("hello", context="test")

    def test_rejects_none(self):
        with pytest.raises(StructuredOutputError, match="expected a JSON object"):
            require_mapping(None, context="test")


# ===========================================================================
# require_string_tuple
# ===========================================================================


class TestRequireStringTuple:
    def test_valid_list_of_strings(self):
        result = require_string_tuple(["a", "b", "c"], context="test")
        assert result == ("a", "b", "c")

    def test_valid_tuple_of_strings(self):
        result = require_string_tuple(("x", "y"), context="test")
        assert result == ("x", "y")

    def test_rejects_non_array(self):
        with pytest.raises(StructuredOutputError, match="expected a JSON array"):
            require_string_tuple("not a list", context="test")

    def test_rejects_item_non_string(self):
        with pytest.raises(StructuredOutputError, match="expected string"):
            require_string_tuple(["a", 42, "c"], context="test")

    def test_rejects_empty_string_item(self):
        with pytest.raises(StructuredOutputError, match="non-empty"):
            require_string_tuple(["a", "", "c"], context="test")

    def test_empty_array_returns_empty_tuple(self):
        result = require_string_tuple([], context="test")
        assert result == ()


# ===========================================================================
# require_sequence
# ===========================================================================


class TestRequireSequence:
    def test_valid_list(self):
        result = require_sequence([1, 2, 3], context="test")
        assert result == (1, 2, 3)

    def test_valid_tuple(self):
        result = require_sequence(("a", "b"), context="test")
        assert result == ("a", "b")

    def test_rejects_string(self):
        with pytest.raises(StructuredOutputError, match="expected a JSON array"):
            require_sequence("hello", context="test")

    def test_rejects_dict(self):
        with pytest.raises(StructuredOutputError, match="expected a JSON array"):
            require_sequence({"a": 1}, context="test")

    def test_empty_array(self):
        result = require_sequence([], context="test")
        assert result == ()


# ===========================================================================
# require_mapping_of_strings
# ===========================================================================


class TestRequireMappingOfStrings:
    def test_valid_string_dict(self):
        result = require_mapping_of_strings({"k1": "v1", "k2": "v2"}, context="test")
        assert result == {"k1": "v1", "k2": "v2"}

    def test_rejects_non_mapping(self):
        with pytest.raises(StructuredOutputError):
            require_mapping_of_strings([1, 2], context="test")

    def test_rejects_non_string_value(self):
        with pytest.raises(StructuredOutputError, match="expected string"):
            require_mapping_of_strings({"k": 42}, context="test")

    def test_rejects_non_string_key(self):
        with pytest.raises(StructuredOutputError, match="expected string"):
            require_mapping_of_strings({42: "value"}, context="test")


# ===========================================================================
# parse_lead_policy_decision
# ===========================================================================


class TestParseLeadPolicyDecision:
    def test_parse_action_kind(self):
        from prismv4.prism_cht.action_schema import DiscriminativeAction

        text = json.dumps({
            "kind": "action",
            "payload": {
                "action_id": "act-1",
                "action_type": "run_discriminative_test",
                "target_hypothesis_ids": ["H1", "H2"],
                "question": "Which component?",
                "tool_name": "compare_onset_order",
                "args": {"comp": "A"},
                "expected_outcomes": {"H1": "earlier", "H2": "later"},
                "why_discriminative": "H1 and H2 differ on onset",
            },
        })
        from prismv4.prism_cht.llm_json import parse_lead_policy_decision
        decision = parse_lead_policy_decision(text)
        assert isinstance(decision, DiscriminativeAction)
        assert decision.action_id == "act-1"

    def test_parse_nomination_kind(self):
        from prismv4.prism_cht.tournament_types import LeadNomination, TripletEvidenceCoverage

        text = json.dumps({
            "kind": "nomination",
            "payload": {
                "hypothesis_id": "H1",
                "supporting_evidence_ids": ["e1", "e2"],
                "addressed_competitor_ids": ["H2"],
                "triplet_grounding": {
                    "component_evidence_ids": ["e3"],
                    "reason_evidence_ids": ["e4"],
                    "onset_evidence_ids": ["e5"],
                },
                "rationale": "H1 is best supported",
            },
        })
        from prismv4.prism_cht.llm_json import parse_lead_policy_decision
        decision = parse_lead_policy_decision(text)
        assert isinstance(decision, LeadNomination)
        assert decision.hypothesis_id == "H1"
        assert isinstance(decision.triplet_grounding, TripletEvidenceCoverage)
        assert decision.triplet_grounding.component_evidence_ids == ("e3",)

    def test_rejects_missing_kind(self):
        text = json.dumps({"payload": {"hypothesis_id": "H1"}})
        from prismv4.prism_cht.llm_json import parse_lead_policy_decision
        with pytest.raises(StructuredOutputError, match="missing"):
            parse_lead_policy_decision(text)

    def test_rejects_missing_payload(self):
        text = json.dumps({"kind": "action"})
        from prismv4.prism_cht.llm_json import parse_lead_policy_decision
        with pytest.raises(StructuredOutputError, match="missing"):
            parse_lead_policy_decision(text)

    def test_rejects_unknown_kind(self):
        text = json.dumps({"kind": "unknown", "payload": {}})
        from prismv4.prism_cht.llm_json import parse_lead_policy_decision
        with pytest.raises(StructuredOutputError, match="kind must be"):
            parse_lead_policy_decision(text)

    def test_rejects_malformed_payload(self):
        text = json.dumps({"kind": "action", "payload": "not_an_object"})
        from prismv4.prism_cht.llm_json import parse_lead_policy_decision
        with pytest.raises(StructuredOutputError, match="expected a JSON object"):
            parse_lead_policy_decision(text)

    def test_rejects_action_with_missing_field(self):
        text = json.dumps({
            "kind": "action",
            "payload": {
                "action_id": "act-1",
                "action_type": "run_discriminative_test",
                "target_hypothesis_ids": ["H1", "H2"],
                "question": "What?",
                "tool_name": "tool",
                "args": {},
            },
        })
        from prismv4.prism_cht.llm_json import parse_lead_policy_decision
        with pytest.raises(StructuredOutputError, match="missing"):
            parse_lead_policy_decision(text)

    def test_action_with_extra_field_rejected(self):
        text = json.dumps({
            "kind": "action",
            "payload": {
                "action_id": "act-1",
                "action_type": "run_discriminative_test",
                "target_hypothesis_ids": ["H1", "H2"],
                "question": "Which component?",
                "tool_name": "compare_onset_order",
                "args": {"comp": "A"},
                "expected_outcomes": {"H1": "earlier", "H2": "later"},
                "why_discriminative": "H1 and H2 differ",
                "extra_field": "should be rejected",
            },
        })
        from prismv4.prism_cht.llm_json import parse_lead_policy_decision
        with pytest.raises(StructuredOutputError, match="unknown"):
            parse_lead_policy_decision(text)


# ===========================================================================
# parse_evidence_assessment
# ===========================================================================


class TestParseEvidenceAssessment:
    def test_happy_path(self):
        from prismv4.prism_cht.tournament_types import EvidenceAssessment, AssessmentOutcome

        text = json.dumps({
            "action_id": "act-1",
            "evidence_id": "e1",
            "outcome": "informative",
            "links": [
                {
                    "hypothesis_id": "H1",
                    "evidence_id": "e1",
                    "relation": "supports",
                    "rationale": "Evidence confirms prediction",
                }
            ],
            "status_updates": [
                {
                    "hypothesis_id": "H1",
                    "new_status": "supported",
                    "rationale": "Accumulated evidence",
                }
            ],
            "rationale": "Overall assessment",
        })
        from prismv4.prism_cht.llm_json import parse_evidence_assessment
        result = parse_evidence_assessment(text)
        assert isinstance(result, EvidenceAssessment)
        assert result.outcome == AssessmentOutcome.INFORMATIVE
        assert len(result.links) == 1
        assert len(result.status_updates) == 1

    def test_rejects_invalid_outcome(self):
        text = json.dumps({
            "action_id": "act-1",
            "evidence_id": "e1",
            "outcome": "invalid_outcome",
            "links": [],
            "status_updates": [],
            "rationale": "test",
        })
        from prismv4.prism_cht.llm_json import parse_evidence_assessment
        with pytest.raises(StructuredOutputError, match="outcome"):
            parse_evidence_assessment(text)

    def test_rejects_invalid_relation(self):
        text = json.dumps({
            "action_id": "act-1",
            "evidence_id": "e1",
            "outcome": "informative",
            "links": [
                {
                    "hypothesis_id": "H1",
                    "evidence_id": "e1",
                    "relation": "invalid_rel",
                    "rationale": "test",
                }
            ],
            "status_updates": [],
            "rationale": "test",
        })
        from prismv4.prism_cht.llm_json import parse_evidence_assessment
        with pytest.raises(StructuredOutputError, match="relation"):
            parse_evidence_assessment(text)

    def test_rejects_invalid_status(self):
        text = json.dumps({
            "action_id": "act-1",
            "evidence_id": "e1",
            "outcome": "informative",
            "links": [],
            "status_updates": [
                {
                    "hypothesis_id": "H1",
                    "new_status": "bogus",
                    "rationale": "test",
                }
            ],
            "rationale": "test",
        })
        from prismv4.prism_cht.llm_json import parse_evidence_assessment
        with pytest.raises(StructuredOutputError, match="new_status"):
            parse_evidence_assessment(text)

    def test_rejects_missing_required_field(self):
        text = json.dumps(
            {"action_id": "act-1", "evidence_id": "e1", "outcome": "informative"}
        )
        from prismv4.prism_cht.llm_json import parse_evidence_assessment
        with pytest.raises(StructuredOutputError, match="missing"):
            parse_evidence_assessment(text)

    def test_deterministic_for_same_input(self):
        from prismv4.prism_cht.llm_json import parse_evidence_assessment

        text = json.dumps({
            "action_id": "act-1",
            "evidence_id": "e1",
            "outcome": "inconclusive",
            "links": [],
            "status_updates": [],
            "rationale": "Insufficient evidence",
        })
        r1 = parse_evidence_assessment(text)
        r2 = parse_evidence_assessment(text)
        assert r1 == r2


# ===========================================================================
# parse_challenge_proposal
# ===========================================================================


class TestParseChallengeProposal:
    def test_happy_path(self):
        from prismv4.prism_cht.challenge_types import ChallengeProposal

        text = json.dumps({
            "challenge_id": "ch-1",
            "nominated_hypothesis_id": "H1",
            "competitor_hypothesis_ids": ["H2"],
            "challenge_claim": "H1 onset is later than claimed",
            "falsification_target": "onset_interval",
            "action": {
                "action_id": "act-c1",
                "action_type": "run_discriminative_test",
                "target_hypothesis_ids": ["H1", "H2"],
                "question": "Which is earlier?",
                "tool_name": "compare_onset_order",
                "args": {"comp": "A"},
                "expected_outcomes": {"H1": "earlier", "H2": "later"},
                "why_discriminative": "Distinguishes onset",
            },
            "rationale": "H1 claim is suspect",
        })
        from prismv4.prism_cht.llm_json import parse_challenge_proposal
        result = parse_challenge_proposal(text)
        assert isinstance(result, ChallengeProposal)
        assert result.challenge_id == "ch-1"

    def test_rejects_missing_field(self):
        text = json.dumps({"challenge_id": "ch-1"})
        from prismv4.prism_cht.llm_json import parse_challenge_proposal
        with pytest.raises(StructuredOutputError, match="missing"):
            parse_challenge_proposal(text)

    def test_rejects_non_object_top_level(self):
        from prismv4.prism_cht.llm_json import parse_challenge_proposal
        with pytest.raises(StructuredOutputError):
            parse_challenge_proposal("not json")

    def test_rejects_malformed_action_payload(self):
        text = json.dumps({
            "challenge_id": "ch-1",
            "nominated_hypothesis_id": "H1",
            "competitor_hypothesis_ids": ["H2"],
            "challenge_claim": "claim",
            "falsification_target": "target",
            "action": "not_an_object",
            "rationale": "reason",
        })
        from prismv4.prism_cht.llm_json import parse_challenge_proposal
        with pytest.raises(StructuredOutputError):
            parse_challenge_proposal(text)


# ===========================================================================
# parse_challenge_resolution
# ===========================================================================


class TestParseChallengeResolution:
    def test_happy_path(self):
        from prismv4.prism_cht.challenge_types import ChallengeResolution, ChallengeVerdict

        text = json.dumps({
            "challenge_id": "ch-1",
            "action_id": "act-r1",
            "evidence_id": "e1",
            "verdict": "nomination_survived",
            "assessment": {
                "action_id": "act-r1",
                "evidence_id": "e1",
                "outcome": "informative",
                "links": [],
                "status_updates": [],
                "rationale": "Evidence supports nomination",
            },
            "rationale": "Challenge did not refute",
        })
        from prismv4.prism_cht.llm_json import parse_challenge_resolution
        result = parse_challenge_resolution(text)
        assert isinstance(result, ChallengeResolution)
        assert result.verdict == ChallengeVerdict.NOMINATION_SURVIVED

    def test_rejects_invalid_verdict(self):
        text = json.dumps({
            "challenge_id": "ch-1",
            "action_id": "act-r1",
            "evidence_id": "e1",
            "verdict": "bogus",
            "assessment": {
                "action_id": "act-r1",
                "evidence_id": "e1",
                "outcome": "informative",
                "links": [],
                "status_updates": [],
                "rationale": "test",
            },
            "rationale": "test",
        })
        from prismv4.prism_cht.llm_json import parse_challenge_resolution
        with pytest.raises(StructuredOutputError, match="verdict"):
            parse_challenge_resolution(text)

    def test_rejects_missing_field(self):
        text = json.dumps({"challenge_id": "ch-1"})
        from prismv4.prism_cht.llm_json import parse_challenge_resolution
        with pytest.raises(StructuredOutputError, match="missing"):
            parse_challenge_resolution(text)


# ===========================================================================
# Error type consistency
# ===========================================================================


class TestErrorTypeConsistency:
    def test_all_parse_errors_are_structured_output_errors(self):
        """Verify that all parsing failures raise StructuredOutputError."""
        from prismv4.prism_cht.llm_json import (
            parse_lead_policy_decision,
            parse_evidence_assessment,
            parse_challenge_proposal,
            parse_challenge_resolution,
        )

        parsers = [
            parse_lead_policy_decision,
            parse_evidence_assessment,
            parse_challenge_proposal,
            parse_challenge_resolution,
        ]
        for parser in parsers:
            with pytest.raises(StructuredOutputError):
                parser("not json at all")

    def test_error_message_does_not_include_full_input(self):
        """Error messages should not echo the entire input string.
        Only a brief context hint should appear."""
        from prismv4.prism_cht.llm_json import parse_json_object
        long_text = '{"key": "' + ("X" * 1000) + 'xyz"}'
        with pytest.raises(StructuredOutputError) as exc:
            parse_json_object(long_text + ' trailing')
        error_str = str(exc.value)
        assert "X" * 1000 not in error_str
