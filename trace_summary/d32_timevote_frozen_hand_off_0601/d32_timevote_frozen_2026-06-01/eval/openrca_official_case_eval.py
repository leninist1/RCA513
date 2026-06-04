"""OpenRCA case-level strict/partial evaluator.

Strict: all required scoring fields match under the best one-to-one assignment.
Partial: at least one required scoring field matches under that assignment.
Time uses the official <=1 minute tolerance encoded in Bank query.csv.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from field_hit_diagnostics import FIELDS, field_match, parse_prediction, parse_truth  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred", required=True)
    parser.add_argument("--query", default="/home/yan/workspace/data/openrca/Bank/query.csv")
    parser.add_argument("--out", default="")
    return parser.parse_args()


def load_csv(path: str | Path) -> list[dict]:
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def best_hits(preds: list[dict[str, str]], truths: list[dict[str, str]]) -> tuple[int, int]:
    required_total = sum(1 for truth in truths for field in FIELDS if field in truth)
    if not truths or len(preds) != len(truths):
        return 0, required_total
    best = 0
    for perm in itertools.permutations(preds):
        hits = 0
        for truth, pred in zip(truths, perm):
            for field in FIELDS:
                if field in truth:
                    hits += int(field_match(field, truth, pred))
        best = max(best, hits)
    return best, required_total


def evaluate(pred_path: str | Path, query_path: str | Path) -> dict:
    pred_rows = load_csv(pred_path)
    if pred_rows and "row_id" in pred_rows[0]:
        pred_rows.sort(key=lambda row: int(row["row_id"]))
    query_rows = load_csv(query_path)
    if len(pred_rows) != len(query_rows):
        raise ValueError(f"prediction/query length mismatch: {len(pred_rows)} vs {len(query_rows)}")
    rows = []
    strict = 0
    partial = 0
    fractional_sum = 0.0
    for idx, (pred_row, query_row) in enumerate(zip(pred_rows, query_rows)):
        preds = parse_prediction(pred_row["prediction"])
        truths = parse_truth(query_row["scoring_points"])
        hits, total = best_hits(preds, truths)
        is_strict = bool(total and hits == total)
        is_partial = bool(hits > 0)
        strict += int(is_strict)
        partial += int(is_partial)
        fractional_sum += hits / total if total else 0.0
        rows.append({
            "row_id": idx,
            "task_index": query_row.get("task_index", ""),
            "hits": hits,
            "required_total": total,
            "strict": is_strict,
            "partial": is_partial,
            "fractional_score": hits / total if total else 0.0,
        })
    n = len(rows)
    return {
        "n": n,
        "strict": strict,
        "partial": partial,
        "strict_rate": strict / n if n else 0.0,
        "partial_rate": partial / n if n else 0.0,
        "fractional_partial_sum": fractional_sum,
        "fractional_partial_rate": fractional_sum / n if n else 0.0,
        "definition": {
            "strict": "all required fields match under best one-to-one assignment",
            "partial": "at least one required field matches under best one-to-one assignment",
            "time_tolerance": "<=1 minute",
        },
        "rows": rows,
    }


def main() -> int:
    args = parse_args()
    report = evaluate(args.pred, args.query)
    print(json.dumps({k: report[k] for k in (
        "n",
        "strict",
        "partial",
        "strict_rate",
        "partial_rate",
        "fractional_partial_rate",
        "definition",
    )}, ensure_ascii=False, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
