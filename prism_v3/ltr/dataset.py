"""Dataset helpers for label-free PRISM LTR features."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd


FORBIDDEN_FEATURE_TOKENS = (
    "ground_truth",
    "gt_",
    "record_",
    "scoring_point",
    "expected_value",
    "label",
)
DEFAULT_EXCLUDE_COLUMNS = {
    "query_id",
    "query_index",
    "task_index",
    "system",
    "sub_system",
    "telemetry_date",
    "object_id",
    "entity",
    "reason",
    "feature_pipeline_version",
    "generated_without_gt",
    "graph_query",
    "training_component",
}


@dataclass
class LTRDataset:
    frame: pd.DataFrame
    feature_columns: List[str]
    label_column: str = "label"
    group_column: str = "query_id"

    @property
    def X(self) -> np.ndarray:
        return self.frame[self.feature_columns].astype(float).fillna(0.0).to_numpy(dtype=float)

    @property
    def y(self) -> np.ndarray:
        if self.label_column not in self.frame.columns:
            return np.zeros(len(self.frame), dtype=float)
        return self.frame[self.label_column].astype(float).fillna(0.0).to_numpy(dtype=float)

    @property
    def groups(self) -> np.ndarray:
        return self.frame[self.group_column].astype(str).to_numpy()


def load_ltr_dataset(
    path: str | Path,
    *,
    label_column: str = "label",
    group_column: str = "query_id",
    extra_exclude: Sequence[str] = (),
) -> LTRDataset:
    frame = pd.read_csv(path)
    assert_label_free_features(frame, label_column=label_column)
    features = numeric_feature_columns(
        frame,
        label_column=label_column,
        group_column=group_column,
        extra_exclude=extra_exclude,
    )
    return LTRDataset(frame=frame, feature_columns=features, label_column=label_column, group_column=group_column)


def assert_label_free_features(frame: pd.DataFrame, *, label_column: str = "label") -> None:
    """Reject answer-bearing columns in raw feature tables.

    The training-only label column is allowed only after label join.
    """
    forbidden = []
    for col in frame.columns:
        lower = str(col).lower()
        if col == label_column:
            continue
        if any(token in lower for token in FORBIDDEN_FEATURE_TOKENS):
            forbidden.append(str(col))
    if forbidden:
        raise ValueError("feature table contains forbidden answer-bearing columns: " + ", ".join(sorted(forbidden)))
    if "generated_without_gt" in frame.columns and not frame["generated_without_gt"].astype(bool).all():
        raise ValueError("feature table contains rows not marked generated_without_gt")


def numeric_feature_columns(
    frame: pd.DataFrame,
    *,
    label_column: str = "label",
    group_column: str = "query_id",
    extra_exclude: Iterable[str] = (),
) -> List[str]:
    exclude = set(DEFAULT_EXCLUDE_COLUMNS)
    exclude.add(label_column)
    exclude.add(group_column)
    exclude.update(str(item) for item in extra_exclude)
    cols: List[str] = []
    for col in frame.columns:
        if col in exclude:
            continue
        series = pd.to_numeric(frame[col], errors="coerce")
        if series.notna().any():
            cols.append(str(col))
            frame[col] = series.fillna(0.0)
    return cols
