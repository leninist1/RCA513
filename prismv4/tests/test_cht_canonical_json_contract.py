"""Tests for CHT-5.2B-3: canonical JSON semantic contract."""

import copy
import pytest

from prismv4.prism_cht.canonical import canonicalize_json_value


# ---------------------------------------------------------------------------
# Conspicuous sentinels
# ---------------------------------------------------------------------------

SUPER_SECRET_API_KEY_SENTINEL = "sk-super-secret-api-key-the-quick-brown-fox"
RAW_PROMPT_CONTENT_SENTINEL = (
    "You are an RCA system.  The root cause is POD_FAILURE in cluster X."
)
RAW_PROVIDER_BODY_SENTINEL = (
    '{"id":"chatcmpl-xxx","choices":[{"message":{"content":"PARSED"}}]}'
)
GROUND_TRUTH_SENTINEL = "pod-failure was caused by OOMKill at node-7"


# ===========================================================================
# A — Verbatim String Preservation
# ===========================================================================


class TestVerbatimStringPreservation:
    """Strings pass through canonicalize_json_value verbatim — no redaction."""

    def test_api_key_preserved(self):
        assert (
            canonicalize_json_value(SUPER_SECRET_API_KEY_SENTINEL)
            == SUPER_SECRET_API_KEY_SENTINEL
        )

    def test_prompt_content_preserved(self):
        assert (
            canonicalize_json_value(RAW_PROMPT_CONTENT_SENTINEL)
            == RAW_PROMPT_CONTENT_SENTINEL
        )

    def test_provider_body_preserved(self):
        assert (
            canonicalize_json_value(RAW_PROVIDER_BODY_SENTINEL)
            == RAW_PROVIDER_BODY_SENTINEL
        )

    def test_ground_truth_preserved(self):
        assert (
            canonicalize_json_value(GROUND_TRUTH_SENTINEL)
            == GROUND_TRUTH_SENTINEL
        )

    def test_string_identity_means_no_redaction(self):
        """Prove that the function is intentionally NOT a redactor."""
        assert (
            canonicalize_json_value("sk-" + "secret")
            == "sk-secret"
        )


# ===========================================================================
# B — Recursive String Preservation
# ===========================================================================


class TestRecursiveStringPreservation:
    def test_nested_structure_preserves_all_sentinels(self):
        nested = {
            "credential_like": SUPER_SECRET_API_KEY_SENTINEL,
            "nested": {
                "prompt_like": RAW_PROMPT_CONTENT_SENTINEL,
                "items": [
                    RAW_PROVIDER_BODY_SENTINEL,
                    {"label_like": GROUND_TRUTH_SENTINEL},
                ],
            },
        }
        result = canonicalize_json_value(nested)

        assert result["credential_like"] == SUPER_SECRET_API_KEY_SENTINEL
        assert result["nested"]["prompt_like"] == RAW_PROMPT_CONTENT_SENTINEL
        assert result["nested"]["items"][0] == RAW_PROVIDER_BODY_SENTINEL
        assert result["nested"]["items"][1]["label_like"] == GROUND_TRUTH_SENTINEL

    def test_sentinels_not_truncated_or_masked(self):
        nested = {"secrets": [SUPER_SECRET_API_KEY_SENTINEL, GROUND_TRUTH_SENTINEL]}
        result = canonicalize_json_value(nested)
        for i, sentinel in enumerate(
            [SUPER_SECRET_API_KEY_SENTINEL, GROUND_TRUTH_SENTINEL]
        ):
            assert result["secrets"][i] == sentinel
            assert len(result["secrets"][i]) == len(sentinel)


# ===========================================================================
# C — No In-Place Mutation
# ===========================================================================


class TestNoInPlaceMutation:
    def test_input_object_not_mutated(self):
        original = {
            "key": "alpha",
            "nested": [1, {"inner": "beta"}],
        }
        expected = copy.deepcopy(original)

        canonicalize_json_value(original)

        assert original == expected

    def test_input_list_not_mutated(self):
        original = ["a", "b", {"c": "d"}]
        expected = copy.deepcopy(original)

        canonicalize_json_value(original)

        assert original == expected

    def test_input_tuple_not_mutated(self):
        original = (1, "hello", None)
        expected = (1, "hello", None)

        canonicalize_json_value(original)

        assert original == expected


# ===========================================================================
# D — Determinism
# ===========================================================================


class TestDeterminism:
    def test_same_dict_different_order_same_output(self):
        a = canonicalize_json_value({"b": 2, "a": 1})
        b = canonicalize_json_value({"a": 1, "b": 2})
        assert a == b

    def test_repeated_calls_produce_identical_results(self):
        value = {"x": [1, None, True, "str"], "y": {"z": 3.14}}
        first = canonicalize_json_value(value)
        for _ in range(10):
            assert canonicalize_json_value(value) == first

    def test_set_canonicalization_deterministic(self):
        a = canonicalize_json_value({3, 1, 2})
        b = canonicalize_json_value({1, 2, 3})
        assert a == b


# ===========================================================================
# E — Existing Supported Scalar Behavior
# ===========================================================================


class TestSupportedScalars:
    def test_none_passthrough(self):
        assert canonicalize_json_value(None) is None

    def test_bool_passthrough(self):
        assert canonicalize_json_value(True) is True
        assert canonicalize_json_value(False) is False

    def test_int_passthrough(self):
        assert canonicalize_json_value(42) == 42
        assert canonicalize_json_value(-1) == -1
        assert canonicalize_json_value(0) == 0

    def test_float_passthrough(self):
        assert canonicalize_json_value(3.14) == 3.14
        assert canonicalize_json_value(0.0) == 0.0
        assert canonicalize_json_value(-1.5) == -1.5

    def test_str_passthrough(self):
        assert canonicalize_json_value("hello") == "hello"
        assert canonicalize_json_value("") == ""


# ===========================================================================
# F — Existing Supported Container Behavior
# ===========================================================================


class TestSupportedContainers:
    def test_mapping_sorted_keys(self):
        value = {"z": 1, "a": 2, "m": 3}
        result = canonicalize_json_value(value)
        assert list(result.keys()) == ["a", "m", "z"]

    def test_mapping_recursive(self):
        value = {"outer": {"inner": 42}}
        result = canonicalize_json_value(value)
        assert result == {"outer": {"inner": 42}}
        assert isinstance(result["outer"], dict)

    def test_list_preserves_order(self):
        value = [3, 1, 2]
        result = canonicalize_json_value(value)
        assert result == [3, 1, 2]
        assert isinstance(result, list)

    def test_tuple_becomes_list(self):
        value = (1, 2, 3)
        result = canonicalize_json_value(value)
        assert result == [1, 2, 3]
        assert isinstance(result, list)

    def test_set_becomes_sorted_list(self):
        value = {3, 1, 2}
        result = canonicalize_json_value(value)
        assert result == [1, 2, 3]


# ===========================================================================
# G — Existing Rejection Behavior
# ===========================================================================


class TestRejectionBehavior:
    def test_bytes_rejected(self):
        with pytest.raises(TypeError, match="Cannot canonicalize"):
            canonicalize_json_value(b"binary data")

    def test_custom_object_rejected(self):
        class Foo:
            pass

        with pytest.raises(TypeError, match="Cannot canonicalize"):
            canonicalize_json_value(Foo())

    def test_custom_object_in_nesting_rejected(self):
        class Bad:
            pass

        with pytest.raises(TypeError, match="Cannot canonicalize"):
            canonicalize_json_value({"key": Bad()})
