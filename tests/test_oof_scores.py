from __future__ import annotations

import pandas as pd

from prism_v3.ltr.predict_oof import generate_oof_scores


def test_oof_scores_are_grouped_and_complete() -> None:
    rows = []
    for q in range(4):
        for rank in range(3):
            rows.append(
                {
                    "query_id": f"q{q}",
                    "query_index": q,
                    "task_index": "task_3",
                    "object_id": f"obj{rank}",
                    "rank": rank + 1,
                    "base_score": float(3 - rank),
                    "metric_score": float(rank == 0),
                    "generated_without_gt": True,
                    "label": int(rank == 0),
                }
            )
    out = generate_oof_scores(pd.DataFrame(rows), n_splits=2, seed=2)
    assert "ltr_score" in out.columns
    assert (out["oof_fold"] >= 0).all()
    assert len(out) == len(rows)
