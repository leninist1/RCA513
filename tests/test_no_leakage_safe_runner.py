from __future__ import annotations

import ast
import inspect

from prism_v3.noise_lab import safe_runner


def _called_names(function) -> set[str]:
    tree = ast.parse(inspect.getsource(function))
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        target = node.func
        if isinstance(target, ast.Name):
            names.add(target.id)
        elif isinstance(target, ast.Attribute):
            names.add(target.attr)
    return names


def test_safe_runner_has_no_ground_truth_record_access() -> None:
    """Feature generation must remain executable with evaluation files hidden."""
    called = _called_names(safe_runner.generate_no_gt_feature_rows)
    called |= _called_names(safe_runner._feature_rows_for_query)
    assert "match_query_to_records" not in called
    assert "load_records" not in called


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
