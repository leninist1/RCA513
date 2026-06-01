from __future__ import annotations

import inspect

from prism_v3.noise_lab import safe_runner


def test_safe_runner_has_no_ground_truth_record_access() -> None:
    """Feature generation must remain executable with record.csv hidden."""
    source = inspect.getsource(safe_runner.generate_no_gt_feature_rows)
    feature_source = inspect.getsource(safe_runner._feature_rows_for_query)
    combined = source + "\n" + feature_source
    assert "match_query_to_records" not in combined
    assert "load_records" not in combined
    assert "record.csv" not in combined


def test_safe_runner_strips_answer_fields_before_feature_generation() -> None:
    source = inspect.getsource(safe_runner._clean_query)
    assert "ground_truth = None" in source
    assert "scoring_points = []" in source
    assert "assert_inference_query_safe" in source


def test_safe_runner_manifest_marks_artifact_as_label_free() -> None:
    source = inspect.getsource(safe_runner.write_manifest)
    assert '"generated_without_gt": True' in source
    assert '"contains_labels": False' in source
    assert 'AnchorSource.PUBLIC_QUERY_WINDOW.value' in source
