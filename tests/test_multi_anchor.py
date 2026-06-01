from __future__ import annotations

import pandas as pd

from prism_v3.config import QueryCase, UnifiedTelemetry
from prism_v3.leakage_guard import AnchorSource
from prism_v3.time_anchor import anchor_stability, build_anchor_set


def test_build_anchor_set_uses_no_gt_telemetry_and_public_fallback() -> None:
    query = QueryCase(
        task_index="task_1",
        system="Bank",
        sub_system="",
        instruction="window",
        time_window=("2026-01-01 00:10:00", "2026-01-01 00:20:00"),
    )
    start = 1767226200.0
    metrics = pd.DataFrame(
        {
            "timestamp": [start - 300, start - 240, start - 180, start, start + 60, start + 120],
            "entity": ["svc"] * 6,
            "metric_name": ["cpu"] * 6,
            "value": [1, 1, 1, 20, 25, 24],
        }
    )
    telemetry = UnifiedTelemetry(metrics=metrics, logs=None, traces=None, entities=["svc"], system="Bank")
    anchors = build_anchor_set(telemetry, query, top_k=3)
    assert anchors.anchors
    assert all(anchor.source in {item.value for item in AnchorSource} for anchor in anchors.anchors)
    assert anchors.fallback_anchor is not None


def test_anchor_stability_weights_top_ranked_entities() -> None:
    stability = anchor_stability([(0.7, ["a", "b"]), (0.3, ["b", "c"])], top_k=1)
    assert stability["a"] == 0.7
    assert stability["b"] == 0.3
