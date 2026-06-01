from __future__ import annotations

from prism_v3.ltr.split import assert_no_group_overlap, group_k_fold


def test_group_k_fold_has_no_incident_overlap() -> None:
    groups = ["q1", "q1", "q2", "q2", "q3", "q3", "q4", "q4"]
    splits = group_k_fold(groups, n_splits=3, seed=1)
    assert splits
    for train_idx, test_idx in splits:
        assert_no_group_overlap(groups, train_idx, test_idx)
