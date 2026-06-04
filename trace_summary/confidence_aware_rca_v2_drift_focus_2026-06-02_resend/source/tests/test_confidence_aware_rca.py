import numpy as np

from confidence_aware_rca import selective_eval


def test_numeric_columns_excludes_labels_requirements_and_hit_counts():
    rows = [
        {
            "heldout_date": "2021_03_04",
            "strict_label": "1",
            "partial_label": "1",
            "time_hit_any_label": "1",
            "component_hit_count": "2",
            "reason_required_total": "1",
            "actionable_required": "1",
            "top1_rebuttal_score": "-4.2",
            "top1_support_strength": "6.0",
        }
    ]

    cols = selective_eval.numeric_columns(rows, feature_set="all")

    assert "top1_rebuttal_score" in cols
    assert "top1_support_strength" in cols
    assert "strict_label" not in cols
    assert "time_hit_any_label" not in cols
    assert "component_hit_count" not in cols
    assert "reason_required_total" not in cols
    assert "actionable_required" not in cols


def test_core_feature_set_keeps_core_features_and_validated_prefixes_only():
    rows = [
        {
            "heldout_date": "2021_03_04",
            "strict_label": "0",
            "top1_rebuttal_score": "-3.0",
            "matched_rule_validation_confidence_max": "0.8",
            "top1_source_layer1_cluster": "1",
            "debug_only_feature": "99",
        }
    ]

    core_cols = selective_eval.numeric_columns(rows, feature_set="core")
    all_cols = selective_eval.numeric_columns(rows, feature_set="all")

    assert "top1_rebuttal_score" in core_cols
    assert "matched_rule_validation_confidence_max" in core_cols
    assert "top1_source_layer1_cluster" not in core_cols
    assert "debug_only_feature" not in core_cols
    assert "top1_source_layer1_cluster" in all_cols
    assert "debug_only_feature" in all_cols


def test_target_specific_head_uses_target_relevant_features():
    rows = [
        {
            "heldout_date": "2021_03_04",
            "strict_label": "0",
            "top1_rebuttal_score": "-3.0",
            "time_anchor_score_margin": "0.7",
            "top1_component_evidence_prior": "0.6",
            "top1_reason_prior_weight": "0.5",
            "matched_cluster_similarity_max": "0.8",
            "matched_rule_validation_confidence_max": "0.9",
        }
    ]

    time_cols = selective_eval.numeric_columns(rows, feature_set="target_specific", target="time_any")
    component_cols = selective_eval.numeric_columns(rows, feature_set="target_specific", target="component_any")
    reason_cols = selective_eval.numeric_columns(rows, feature_set="target_specific", target="reason_any")

    assert "time_anchor_score_margin" in time_cols
    assert "top1_component_evidence_prior" not in time_cols
    assert "top1_reason_prior_weight" not in time_cols
    assert "top1_component_evidence_prior" in component_cols
    assert "matched_cluster_similarity_max" in component_cols
    assert "time_anchor_score_margin" not in component_cols
    assert "top1_reason_prior_weight" in reason_cols
    assert "matched_rule_validation_confidence_max" in reason_cols
    assert "top1_component_evidence_prior" not in reason_cols


def test_target_rows_filters_by_required_field_for_field_targets():
    rows = [
        {"component_required": "1", "component_hit_any_label": "1"},
        {"component_required": "0", "component_hit_any_label": "0"},
        {"component_required": "1", "component_hit_any_label": "0"},
    ]

    filtered = selective_eval.target_rows(rows, "component_any")

    assert len(filtered) == 2
    assert selective_eval.labels(filtered, "component_any").tolist() == [1, 0]


def test_append_drift_features_uses_train_distribution():
    train_rows = [
        {"x": "0", "y": "0"},
        {"x": "2", "y": "2"},
        {"x": "4", "y": "4"},
    ]
    test_rows = [{"x": "100", "y": "100"}]

    aug_train, aug_test, columns = selective_eval.append_drift_features(train_rows, test_rows, ["x", "y"])

    assert "drift_l1_to_train_median" in columns
    assert aug_test[0]["drift_l1_to_train_median"] > max(row["drift_l1_to_train_median"] for row in aug_train)
    assert aug_test[0]["drift_max_abs_z"] > 10


def test_threshold_selection_and_selective_stats_use_train_accuracy():
    prob = np.asarray([0.95, 0.80, 0.20])
    y = np.asarray([1, 0, 1])

    threshold = selective_eval.choose_threshold(prob, y, 0.8)
    stats = selective_eval.selective_stats(prob, y, threshold)

    assert threshold == 0.95
    assert stats["accepted"] == 1
    assert stats["correct"] == 1
    assert stats["selective_accuracy"] == 1.0


def test_wilson_lower_bound_is_conservative():
    lower = selective_eval.wilson_lower_bound(successes=8, total=10, z=1.64)

    assert 0.0 < lower < 0.8
    assert selective_eval.wilson_lower_bound(successes=0, total=0, z=1.64) == 0.0


def test_auto_candidate_parser_rejects_unknown_feature_set():
    assert selective_eval.parse_feature_set_candidates("core,target_specific") == ["core", "target_specific"]

    try:
        selective_eval.parse_feature_set_candidates("core,leaky")
    except ValueError as exc:
        assert "unsupported auto feature-set candidate" in str(exc)
    else:
        raise AssertionError("expected invalid candidate to fail")


def test_threshold_transfer_selection_prefers_reliable_coverage():
    reports = [
        {
            "feature_set": "tiny",
            "aurc": 0.1,
            "coverage": 0.05,
            "selective_accuracy": 1.0,
            "wilson_lower": 0.35,
            "utility": 0.2,
        },
        {
            "feature_set": "stable",
            "aurc": 0.2,
            "coverage": 0.3,
            "selective_accuracy": 0.7,
            "wilson_lower": 0.5,
            "utility": 0.3,
        },
    ]

    selected = selective_eval.select_candidate_report(
        reports,
        selection_objective="threshold_transfer",
        min_coverage=0.2,
    )

    assert selected["feature_set"] == "stable"


def test_inner_feature_selection_returns_declared_candidate():
    rows = []
    for idx, date in enumerate(["d1", "d1", "d2", "d2", "d3", "d3"]):
        label = int(idx % 2 == 0)
        rows.append(
            {
                "heldout_date": date,
                "partial_label": str(label),
                "top1_rebuttal_score": str(-label),
                "top1_support_strength": str(label),
                "time_anchor_score_margin": str(label),
                "top1_component_evidence_prior": str(label),
                "top1_reason_prior_weight": str(label),
            }
        )

    selected, report = selective_eval.select_feature_set(
        rows,
        target="partial",
        candidates=["core", "target_specific"],
        model_kind="logistic",
        target_accuracy=0.8,
        risk_mode="empirical",
        delta=0.1,
    )

    assert selected in {"core", "target_specific"}
    assert report["selected_feature_set"] == selected
    assert {row["feature_set"] for row in report["candidates"]} == {"core", "target_specific"}
