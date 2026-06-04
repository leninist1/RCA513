"""Field-level diagnostics for OpenRCA official query predictions.

The official evaluator reports a single partial score. This script decomposes
that score into time/component/reason hit rates and pairwise conditional hit
rates under the same one-to-one root-cause matching assumption.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import itertools
import json
from pathlib import Path
import re
from typing import Iterable


FIELDS = ("time", "component", "reason")

PREDICT_PATTERN = (
    r'{\s*'
    r'(?:"root cause occurrence datetime":\s*"(.*?)")?,?\s*'
    r'(?:"root cause component":\s*"(.*?)")?,?\s*'
    r'(?:"root cause reason":\s*"(.*?)")?\s*}'
)

TRUTH_PATTERNS = {
    "component": r"The (?:\d+-th|only) predicted root cause component is ([^\n]+)",
    "reason": r"The (?:\d+-th|only) predicted root cause reason is ([^\n]+)",
    "time": r"The (?:\d+-th|only) root cause occurrence time is within 1 minutes \(i.e., <=1min\) of ([^\n]+)",
}

PRED_KEYS = {
    "time": "root cause occurrence datetime",
    "component": "root cause component",
    "reason": "root cause reason",
}

COMPONENT_EXPANSION: dict[str, set[str]] = {}


def build_component_expansion(node_graph: dict) -> dict[str, set[str]]:
    """Build mapping from component names to expanded matching sets.

    - Node names (e.g. node-6) → {node_name} ∪ all hosted containers
    - Service names without suffix (e.g. recommendationservice) → all pods with that prefix
    - Exact container names → {container_name}
    """
    containers = node_graph.get("containers", {})
    nodes = node_graph.get("nodes", {})

    expansion: dict[str, set[str]] = {}
    for container_name in containers:
        expansion[str(container_name)] = {str(container_name)}

    for node_name, node_info in nodes.items():
        hosted = {str(c) for c in node_info.get("hosted_containers", [])}
        expansion[str(node_name)] = {str(node_name)} | hosted

    from collections import defaultdict
    prefix_map: dict[str, set[str]] = defaultdict(set)
    for container_name in containers:
        name = str(container_name)
        prefix = name.rstrip("0123456789-")
        if prefix and prefix != name and not prefix.endswith("."):
            prefix_map[prefix].add(name)

    for prefix, matched in prefix_map.items():
        if prefix in expansion:
            expansion[prefix] |= matched
        else:
            expansion[prefix] = matched

    return expansion


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred", required=True, help="Prediction CSV with row_id,prediction columns.")
    parser.add_argument("--query", default="/home/yan/workspace/data/openrca/Bank/query.csv")
    parser.add_argument("--node-graph", default="", help="Node-container graph JSON for component expansion (Market)")
    parser.add_argument("--out", default="")
    return parser.parse_args()


def parse_prediction(text: str) -> list[dict[str, str]]:
    matches = re.findall(PREDICT_PATTERN, str(text))
    return [
        {
            "time": (dt or "").strip(),
            "component": (component or "").strip(),
            "reason": (reason or "").strip(),
        }
        for dt, component, reason in matches
    ]


def parse_truth(scoring_points: str) -> list[dict[str, str]]:
    found = {field: [x.strip() for x in re.findall(pattern, str(scoring_points))] for field, pattern in TRUTH_PATTERNS.items()}
    n = max((len(values) for values in found.values()), default=0)
    truths = []
    for idx in range(n):
        row = {}
        for field, values in found.items():
            if len(values) == n:
                row[field] = values[idx]
        truths.append(row)
    return truths


def time_match(expected: str, predicted: str) -> bool:
    try:
        a = datetime.strptime(expected, "%Y-%m-%d %H:%M:%S")
        b = datetime.strptime(predicted, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return False
    return abs((a - b).total_seconds()) <= 60


def field_match(field: str, truth: dict[str, str], pred: dict[str, str]) -> bool:
    if field not in truth:
        return False
    expected = truth[field]
    got = pred.get(field, "")
    if field == "time":
        return time_match(expected, got)
    if field == "component" and COMPONENT_EXPANSION:
        expanded = COMPONENT_EXPANSION.get(expected, {expected})
        return got in expanded
    return expected == got


def permutations_or_empty(preds: list[dict[str, str]], truths: list[dict[str, str]]) -> Iterable[tuple[dict[str, str], ...]]:
    if len(preds) != len(truths):
        return []
    return itertools.permutations(preds)


def max_hits_for_fields(preds: list[dict[str, str]], truths: list[dict[str, str]], fields: tuple[str, ...]) -> dict[str, int]:
    best_score = -1
    best = {field: 0 for field in fields}
    for perm in permutations_or_empty(preds, truths):
        hits = {field: 0 for field in fields}
        for truth, pred in zip(truths, perm):
            for field in fields:
                hits[field] += int(field_match(field, truth, pred))
        score = sum(hits.values())
        if score > best_score:
            best_score = score
            best = hits
    return best


def max_single_field_hits(preds: list[dict[str, str]], truths: list[dict[str, str]], field: str) -> int:
    return max_hits_for_fields(preds, truths, (field,))[field]


def max_joint_pair_hits(preds: list[dict[str, str]], truths: list[dict[str, str]], left: str, right: str) -> int:
    best = 0
    for perm in permutations_or_empty(preds, truths):
        hits = 0
        for truth, pred in zip(truths, perm):
            hits += int(field_match(left, truth, pred) and field_match(right, truth, pred))
        best = max(best, hits)
    return best


def safe_rate(num: float, den: float) -> float:
    return num / den if den else 0.0


def load_rows(pred_path: str, query_path: str) -> list[dict]:
    with open(pred_path, newline="", encoding="utf-8") as f:
        pred_rows = list(csv.DictReader(f))
    if pred_rows and "row_id" in pred_rows[0]:
        pred_rows.sort(key=lambda row: int(row["row_id"]))
    with open(query_path, newline="", encoding="utf-8") as f:
        query_rows = list(csv.DictReader(f))
    if len(pred_rows) != len(query_rows):
        raise ValueError(f"prediction/query length mismatch: {len(pred_rows)} vs {len(query_rows)}")
    rows = []
    for idx, (pred_row, query_row) in enumerate(zip(pred_rows, query_rows)):
        rows.append({
            "row_id": idx,
            "task_index": query_row.get("task_index", ""),
            "preds": parse_prediction(pred_row["prediction"]),
            "truths": parse_truth(query_row["scoring_points"]),
        })
    return rows


def summarize(rows: list[dict]) -> dict:
    field_item = {field: {"total": 0, "hit": 0} for field in FIELDS}
    field_case = {field: {"total": 0, "all_hit": 0} for field in FIELDS}
    task_field = {}
    pair_stats = {}
    row_details = []

    for row in rows:
        preds = row["preds"]
        truths = row["truths"]
        required = {field for truth in truths for field in truth}
        full_hits = max_hits_for_fields(preds, truths, tuple(sorted(required))) if required else {}
        detail = {"row_id": row["row_id"], "task_index": row["task_index"], "required": sorted(required), "hits": full_hits}
        row_details.append(detail)
        task = task_field.setdefault(row["task_index"], {field: {"total": 0, "hit": 0} for field in FIELDS})

        for field in FIELDS:
            total = sum(1 for truth in truths if field in truth)
            if total == 0:
                continue
            hit = max_single_field_hits(preds, truths, field)
            field_item[field]["total"] += total
            field_item[field]["hit"] += hit
            field_case[field]["total"] += 1
            field_case[field]["all_hit"] += int(hit == total)
            task[field]["total"] += total
            task[field]["hit"] += hit

        for left, right in itertools.permutations(FIELDS, 2):
            pair_truths = [truth for truth in truths if left in truth and right in truth]
            if not pair_truths:
                continue
            key = f"{left}->{right}"
            stat = pair_stats.setdefault(key, {"left_hit": 0, "joint_hit": 0, "pair_items": 0})
            stat["pair_items"] += len(pair_truths)
            stat["left_hit"] += max_single_field_hits(preds, truths, left)
            stat["joint_hit"] += max_joint_pair_hits(preds, truths, left, right)

    for field, row in field_item.items():
        row["hit_rate"] = safe_rate(row["hit"], row["total"])
    for field, row in field_case.items():
        row["case_all_hit_rate"] = safe_rate(row["all_hit"], row["total"])
    for task, fields in task_field.items():
        for row in fields.values():
            row["hit_rate"] = safe_rate(row["hit"], row["total"])
    for row in pair_stats.values():
        row["conditional_rate"] = safe_rate(row["joint_hit"], row["left_hit"])
        row["joint_rate_over_pair_items"] = safe_rate(row["joint_hit"], row["pair_items"])

    return {
        "n_rows": len(rows),
        "field_item": field_item,
        "field_case": field_case,
        "pair_conditionals": dict(sorted(pair_stats.items())),
        "by_task_index": dict(sorted(task_field.items())),
        "rows": row_details,
    }


def main() -> int:
    global COMPONENT_EXPANSION
    args = parse_args()
    if args.node_graph:
        with open(args.node_graph, encoding="utf-8") as f:
            COMPONENT_EXPANSION = build_component_expansion(json.load(f))
    report = summarize(load_rows(args.pred, args.query))
    print(json.dumps({
        "n_rows": report["n_rows"],
        "field_item": report["field_item"],
        "field_case": report["field_case"],
        "pair_conditionals": report["pair_conditionals"],
    }, ensure_ascii=False, indent=2))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
