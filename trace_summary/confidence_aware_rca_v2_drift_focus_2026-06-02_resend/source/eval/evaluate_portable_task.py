#!/usr/bin/env python3
"""Dataset-specific evaluation for portable D32 outputs.

This script intentionally avoids reusing OpenRCA's case-level metric for every
dataset.  Eadro is evaluated as Root Cause Localization (ranked service HR/NDCG).
AIOps2021 is evaluated as service localization plus anomaly-type classification.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["eadro", "aiops2021"], required=True)
    parser.add_argument("--cases-csv", required=True)
    parser.add_argument("--pred", required=True, help="Portable predictions.csv")
    parser.add_argument("--debug-json", default=None, help="Portable debug.json or checkpoint JSONL")
    parser.add_argument("--out-json", "--out", dest="out_json", required=True)
    parser.add_argument("--details-csv", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cases = pd.read_csv(args.cases_csv)
    predictions = _load_predictions(args.pred)
    debug_rows = _load_debug_rows(args.debug_json) if args.debug_json else []

    details_csv = args.details_csv
    if not details_csv:
        out_path = Path(args.out_json)
        details_csv = str(out_path.with_name(out_path.stem + "_details.csv"))

    if args.dataset == "eadro":
        summary, details = evaluate_eadro(cases, predictions, debug_rows, details_csv)
    elif args.dataset == "aiops2021":
        summary, details = evaluate_aiops2021(cases, predictions, debug_rows, details_csv)
    else:
        raise ValueError(f"unsupported dataset: {args.dataset}")

    out = {
        "dataset": args.dataset,
        "metric_policy": summary.pop("metric_policy"),
        "summary": summary,
        "details_csv": details_csv,
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    Path(details_csv).parent.mkdir(parents=True, exist_ok=True)
    details.to_csv(details_csv, index=False)
    print(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def evaluate_eadro(
    cases: pd.DataFrame,
    predictions: Mapping[str, Mapping[str, str]],
    debug_rows: list[Mapping[str, Any]],
    details_csv: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    debug_by_case, debug_by_row = _index_debug_rows(debug_rows)
    detail_rows: list[dict[str, Any]] = []
    for row_idx, row in cases.reset_index(drop=True).iterrows():
        case_id = str(row.get("case_id", row_idx))
        gt_service = _first_nonempty(row, ["root_cause_component", "component", "root_cause"])
        pred = predictions.get(case_id, {})
        pred_top1 = str(pred.get("component", ""))
        debug_row = debug_by_case.get(case_id) or debug_by_row.get(int(row_idx))
        ranked, ranking_source = _ranked_services(debug_row, pred_top1)
        rank_of_gt = _rank_of(ranked, gt_service)
        detail_rows.append({
            "case_id": case_id,
            "task": "eadro_root_cause_localization",
            "gt_service": gt_service,
            "fault_type": _normalize_type(row.get("failure_type", "")),
            "pred_top1_service": ranked[0] if ranked else "",
            "ranked_services_json": json.dumps(ranked, ensure_ascii=False),
            "rank_of_gt": rank_of_gt,
            "ranking_source": ranking_source,
            "ranking_len": len(ranked),
            "HR@1": _hr_at_k(ranked, gt_service, 1),
            "HR@3": _hr_at_k(ranked, gt_service, 3),
            "HR@5": _hr_at_k(ranked, gt_service, 5),
            "NDCG@3": _ndcg_at_k(ranked, gt_service, 3),
            "NDCG@5": _ndcg_at_k(ranked, gt_service, 5),
        })
    details = pd.DataFrame(detail_rows)
    return {
        "metric_policy": {
            "primary_task": "Eadro Root Cause Localization",
            "primary_metrics": ["HR@1", "HR@3", "HR@5", "NDCG@3", "NDCG@5"],
            "note": "Eadro summary ignores reason/time correctness; failure_type is diagnostic only.",
        },
        "n_cases": int(len(details)),
        "HR@1": _mean(details, "HR@1"),
        "HR@3": _mean(details, "HR@3"),
        "HR@5": _mean(details, "HR@5"),
        "NDCG@3": _mean(details, "NDCG@3"),
        "NDCG@5": _mean(details, "NDCG@5"),
        "avg_ranking_len": _mean(details, "ranking_len"),
        "missing_gt_in_ranking": int(details["rank_of_gt"].isna().sum()),
        "ranking_source_counts": _counts(details, "ranking_source"),
        "by_failure_type_diagnostic": _aggregate(details, "fault_type", ["HR@1", "HR@3", "HR@5", "NDCG@5"]),
        "details_csv": details_csv,
    }, details


def evaluate_aiops2021(
    cases: pd.DataFrame,
    predictions: Mapping[str, Mapping[str, str]],
    debug_rows: list[Mapping[str, Any]],
    details_csv: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    debug_by_case, debug_by_row = _index_debug_rows(debug_rows)
    detail_rows: list[dict[str, Any]] = []
    for row_idx, row in cases.reset_index(drop=True).iterrows():
        case_id = str(row.get("case_id", row_idx))
        gt_service = _first_nonempty(row, ["root_cause_component", "component", "root_cause"])
        gt_type = _normalize_type(row.get("failure_type", ""))
        pred = predictions.get(case_id, {})
        pred_top1 = str(pred.get("component", ""))
        pred_type = _normalize_type(pred.get("reason", ""))
        debug_row = debug_by_case.get(case_id) or debug_by_row.get(int(row_idx))
        ranked, ranking_source = _ranked_services(debug_row, pred_top1)
        rank_of_gt = _rank_of(ranked, gt_service)
        service_top1_correct = _hr_at_k(ranked, gt_service, 1)
        type_correct = 1.0 if gt_type and pred_type == gt_type else 0.0
        detail_rows.append({
            "case_id": case_id,
            "task": "aiops2021_service_and_anomaly_type",
            "split": str(row.get("data_type", "all")),
            "gt_service": gt_service,
            "gt_type": gt_type,
            "pred_top1_service": ranked[0] if ranked else "",
            "pred_type": pred_type,
            "ranked_services_json": json.dumps(ranked, ensure_ascii=False),
            "rank_of_gt": rank_of_gt,
            "ranking_source": ranking_source,
            "ranking_len": len(ranked),
            "service_top1_accuracy": service_top1_correct,
            "service_HR@3": _hr_at_k(ranked, gt_service, 3),
            "service_HR@5": _hr_at_k(ranked, gt_service, 5),
            "service_NDCG@3": _ndcg_at_k(ranked, gt_service, 3),
            "service_NDCG@5": _ndcg_at_k(ranked, gt_service, 5),
            "anomaly_type_accuracy": type_correct,
            "service_type_tuple_accuracy": 1.0 if service_top1_correct and type_correct else 0.0,
        })
    details = pd.DataFrame(detail_rows)
    metrics = [
        "service_top1_accuracy",
        "service_HR@3",
        "service_HR@5",
        "service_NDCG@3",
        "service_NDCG@5",
        "anomaly_type_accuracy",
        "service_type_tuple_accuracy",
    ]
    summary = {
        "metric_policy": {
            "primary_task": "AIOps2021 service localization plus anomaly-type classification",
            "primary_metrics": metrics,
            "note": "AIOps2021 summary does not use OpenRCA time/partial-credit scoring.",
        },
        "n_cases": int(len(details)),
        "avg_ranking_len": _mean(details, "ranking_len"),
        "missing_gt_in_ranking": int(details["rank_of_gt"].isna().sum()),
        "ranking_source_counts": _counts(details, "ranking_source"),
        "by_split": _aggregate(details, "split", metrics),
        "by_failure_type": _aggregate(details, "gt_type", metrics),
        "details_csv": details_csv,
    }
    summary.update({metric: _mean(details, metric) for metric in metrics})
    return summary, details


def _load_predictions(path: str | Path) -> dict[str, dict[str, str]]:
    df = pd.read_csv(path)
    out: dict[str, dict[str, str]] = {}
    for idx, row in df.reset_index(drop=True).iterrows():
        case_id = str(row.get("case_id", idx))
        item = _prediction_item(row.get("prediction", {}))
        out[case_id] = {
            "component": str(item.get("root cause component", item.get("component", ""))).strip(),
            "reason": str(item.get("root cause reason", item.get("reason", ""))).strip(),
            "time": str(item.get("root cause occurrence datetime", item.get("time", ""))).strip(),
        }
    return out


def _prediction_item(prediction: Any) -> dict[str, Any]:
    if isinstance(prediction, str):
        try:
            prediction = json.loads(prediction)
        except Exception:
            component = _regex_group(prediction, r'"root cause component"\s*:\s*"([^"]+)"')
            reason = _regex_group(prediction, r'"root cause reason"\s*:\s*"([^"]+)"')
            return {"root cause component": component, "root cause reason": reason}
    if isinstance(prediction, dict):
        if "1" in prediction and isinstance(prediction["1"], dict):
            return prediction["1"]
        for value in prediction.values():
            if isinstance(value, dict):
                return value
    return {}


def _load_debug_rows(path: str | Path) -> list[Mapping[str, Any]]:
    fp = Path(path)
    text = fp.read_text(encoding="utf-8").strip()
    if not text:
        return []
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and isinstance(obj.get("debug"), list):
            rows = obj["debug"]
        elif isinstance(obj, list):
            rows = obj
        else:
            rows = [obj]
    except json.JSONDecodeError:
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    rows = [row for row in rows if isinstance(row, dict)]
    rows.sort(key=lambda row: int(row.get("row_id", 0) or 0))
    return rows


def _index_debug_rows(rows: Iterable[Mapping[str, Any]]) -> tuple[dict[str, Mapping[str, Any]], dict[int, Mapping[str, Any]]]:
    by_case: dict[str, Mapping[str, Any]] = {}
    by_row: dict[int, Mapping[str, Any]] = {}
    for idx, row in enumerate(rows):
        case_id = str(row.get("case_id", "")).strip()
        if not case_id:
            outer = row.get("debug") if isinstance(row.get("debug"), dict) else {}
            case_id = str(outer.get("case_id", "")).strip()
        if case_id:
            by_case[case_id] = row
        try:
            row_id = int(row.get("row_id", idx) or idx)
        except Exception:
            row_id = idx
        by_row[row_id] = row
    return by_case, by_row


def _ranked_services(debug_row: Mapping[str, Any] | None, pred_top1: str = "") -> tuple[list[str], str]:
    if not debug_row:
        return ([pred_top1] if pred_top1 else []), "prediction_top1_fallback"
    inner = _inner_debug(debug_row)

    decisions = inner.get("all_decisions", []) if isinstance(inner, dict) else []
    ranked = _dedup([
        _candidate_component(decision.get("candidate", {}) if isinstance(decision, dict) else {})
        for decision in decisions or []
    ])
    if len(ranked) > 1:
        return ranked, "all_decisions"
    if len(ranked) == 1 and not pred_top1:
        return ranked, "all_decisions_single"

    candidate_space = inner.get("candidate_space", []) if isinstance(inner, dict) else []
    scored: list[tuple[float, int, str]] = []
    for idx, candidate in enumerate(candidate_space or []):
        if not isinstance(candidate, dict):
            continue
        component = _candidate_component(candidate)
        if component:
            scored.append((_score_candidate(candidate), -idx, component))
    scored.sort(reverse=True)
    ranked = _dedup([component for _, _, component in scored])
    if ranked:
        return ranked, "candidate_space"
    return ([pred_top1] if pred_top1 else []), "prediction_top1_fallback"


def _inner_debug(row: Mapping[str, Any]) -> Mapping[str, Any]:
    if "all_decisions" in row or "candidate_space" in row:
        return row
    d32_result = row.get("d32_result")
    if isinstance(d32_result, dict) and isinstance(d32_result.get("debug"), dict):
        return d32_result["debug"]
    outer_debug = row.get("debug")
    if isinstance(outer_debug, dict):
        d32_result = outer_debug.get("d32_result")
        if isinstance(d32_result, dict) and isinstance(d32_result.get("debug"), dict):
            return d32_result["debug"]
        if "all_decisions" in outer_debug or "candidate_space" in outer_debug:
            return outer_debug
    return {}


def _score_candidate(candidate: Mapping[str, Any]) -> float:
    details = candidate.get("details", {}) if isinstance(candidate.get("details"), dict) else {}
    scores: list[float] = []
    for obj, key in [
        (candidate, "prior"),
        (candidate, "score"),
        (candidate, "joint_score"),
        (candidate, "rank_score"),
        (details, "joint_rerank_score"),
        (details, "joint_local_score"),
        (details, "signal_strength"),
        (details, "component_score"),
        (details, "metric_score"),
        (details, "trace_score"),
        (details, "log_score"),
    ]:
        try:
            scores.append(float(obj.get(key, 0.0) or 0.0))
        except Exception:
            pass
    return max(scores) if scores else 0.0


def _candidate_component(candidate: Any) -> str:
    if not isinstance(candidate, dict):
        return ""
    for key in ("component", "service", "cmdb_id", "node"):
        value = candidate.get(key)
        if value:
            return str(value).strip()
    nested = candidate.get("candidate")
    if isinstance(nested, dict):
        return _candidate_component(nested)
    return ""


def _normalize_type(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"\s+", " ", text)
    if not text or text == "nan":
        return ""
    if "packet" in text or "loss" in text or "retransmission" in text or "corruption" in text:
        return "network_packet_loss"
    if "latency" in text or "delay" in text:
        return "network_latency"
    if "jvm" in text or "oom" in text or "out of memory" in text:
        return "jvm_oom"
    if "filesystem" in text or "file system" in text or "space" in text:
        return "filesystem"
    if "disk" in text or "i/o" in text or " io" in text or "read" in text or "write" in text:
        return "disk_io"
    if "memory" in text or re.search(r"\bmem\b", text):
        return "memory"
    if "cpu" in text:
        return "cpu"
    return text.replace(" ", "_")


def _first_nonempty(row: Mapping[str, Any], keys: Iterable[str]) -> str:
    for key in keys:
        value = row.get(key, "")
        if pd.notna(value) and str(value).strip():
            return str(value).strip()
    return ""


def _dedup(items: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        item = str(item or "").strip()
        if item and item not in seen:
            out.append(item)
            seen.add(item)
    return out


def _rank_of(ranked: list[str], gt: str) -> int | None:
    gt = str(gt or "").strip()
    if not gt:
        return None
    try:
        return ranked.index(gt) + 1
    except ValueError:
        return None


def _hr_at_k(ranked: list[str], gt: str, k: int) -> float:
    return 1.0 if str(gt or "").strip() in ranked[:k] else 0.0


def _ndcg_at_k(ranked: list[str], gt: str, k: int) -> float:
    gt = str(gt or "").strip()
    for idx, service in enumerate(ranked[:k], start=1):
        if service == gt:
            return 1.0 / math.log2(idx + 1)
    return 0.0


def _mean(df: pd.DataFrame, column: str) -> float:
    if df.empty or column not in df:
        return 0.0
    return float(df[column].mean())


def _counts(df: pd.DataFrame, column: str) -> dict[str, int]:
    if df.empty or column not in df:
        return {}
    return {str(k): int(v) for k, v in df[column].value_counts().to_dict().items()}


def _aggregate(df: pd.DataFrame, group_col: str, metrics: list[str]) -> dict[str, dict[str, Any]]:
    if df.empty or group_col not in df:
        return {}
    out: dict[str, dict[str, Any]] = {}
    for key, group in df.groupby(group_col, dropna=False):
        item: dict[str, Any] = {"n_cases": int(len(group))}
        for metric in metrics:
            item[metric] = _mean(group, metric)
        out[str(key)] = item
    return out


def _regex_group(text: str, pattern: str) -> str:
    match = re.search(pattern, text)
    return match.group(1).strip() if match else ""


if __name__ == "__main__":
    raise SystemExit(main())
