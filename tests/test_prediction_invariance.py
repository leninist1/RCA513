from __future__ import annotations

from types import SimpleNamespace

import pytest

from prism_v3.leakage_guard import UnsafeInferenceQueryError, assert_inference_query_safe


def test_random_scoring_points_do_not_enter_inference() -> None:
    query = SimpleNamespace(ground_truth=None, scoring_points=["mutated-answer"])
    with pytest.raises(UnsafeInferenceQueryError, match="scoring_points"):
        assert_inference_query_safe(query)


def test_clean_inference_query_remains_accepted() -> None:
    query = SimpleNamespace(ground_truth=None, scoring_points=[])
    assert_inference_query_safe(query)
