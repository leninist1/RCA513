"""Extract self-knowledge features from frozen d32 RCA debug artifacts."""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from pathlib import Path
import sys
from statistics import mean
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = PROJECT_ROOT / "eval"
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

from field_hit_diagnostics import FIELDS, field_match, max_joint_pair_hits, max_single_field_hits, parse_truth  # noqa: E402


CONFIDENCE_LEVEL = {"LOW": 0.0, "MEDIUM": 1.0, "HIGH": 2.0}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--debug-json", default="logs/d32_lodo_timevote_trace_debug.json")
    parser.add_argument("--official-case-eval", default="logs/d32_lodo_timevote_trace_official_case_eval.json")
    parser.add_argument("--field-diag", default="logs/d32_lodo_timevote_trace_field_diag.json")
    parser.add_argument("--query", default="/home/yan/workspace/data/openrca/Bank/query.csv")
    parser.add_argument("--out", default="confidence_aware_rca/experiments/d32_timevote_confidence_features.csv")
    return parser.parse_args()


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def load_csv(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def prediction_list(prediction: Mapping[str, Mapping[str, Any]]) -> list[dict[str, str]]:
    rows = []
    for key in sorted(prediction, key=lambda item: int(item) if str(item).isdigit() else str(item)):
        pred = prediction[key]
        rows.append({
            "time": str(pred.get("root cause occurrence datetime", "")).strip(),
            "component": str(pred.get("root cause component", "")).strip(),
            "reason": str(pred.get("root cause reason", "")).strip(),
        })
    return rows


def field_case_labels(prediction: Mapping[str, Mapping[str, Any]], scoring_points: str) -> dict[str, int]:
    preds = prediction_list(prediction)
    truths = parse_truth(scoring_points)
    out: dict[str, int] = {}
    for field in FIELDS:
        total = sum(1 for truth in truths if field in truth)
        hit = max_single_field_hits(preds, truths, field) if total else 0
        out[f"{field}_required"] = int(total > 0)
        out[f"{field}_required_total"] = int(total)
        out[f"{field}_hit_any_label"] = int(hit > 0)
        out[f"{field}_hit_label"] = int(total > 0 and hit == total)
        out[f"{field}_hit_count"] = int(hit)
    pair_total = sum(1 for truth in truths if "component" in truth and "reason" in truth)
    pair_hit = max_joint_pair_hits(preds, truths, "component", "reason") if pair_total else 0
    out["component_reason_pair_required"] = int(pair_total > 0)
    out["component_reason_pair_required_total"] = pair_total
    out["component_reason_pair_hit_any_label"] = int(pair_hit > 0)
    out["component_reason_pair_hit_label"] = int(pair_total > 0 and pair_hit == pair_total)
    out["component_reason_pair_hit_count"] = pair_hit

    actionable_hit, actionable_total = max_actionable_hits(preds, truths)
    out["actionable_required"] = int(actionable_total > 0)
    out["actionable_required_total"] = actionable_total
    out["actionable_hit_any_label"] = int(actionable_hit > 0)
    out["actionable_hit_label"] = int(actionable_total > 0 and actionable_hit == actionable_total)
    out["actionable_hit_count"] = actionable_hit
    return out


def max_actionable_hits(preds: list[dict[str, str]], truths: list[dict[str, str]]) -> tuple[int, int]:
    total = sum(1 for truth in truths if "component" in truth)
    if not total or len(preds) != len(truths):
        return 0, total
    best = 0
    for perm in itertools.permutations(preds):
        hits = 0
        for truth, pred in zip(truths, perm):
            if "component" not in truth or not field_match("component", truth, pred):
                continue
            side_fields = [field for field in ("reason", "time") if field in truth]
            if not side_fields or any(field_match(field, truth, pred) for field in side_fields):
                hits += 1
        best = max(best, hits)
    return best, total


def scalar_stats(values: list[float], prefix: str) -> dict[str, float]:
    if not values:
        return {
            f"{prefix}_count": 0.0,
            f"{prefix}_max": 0.0,
            f"{prefix}_mean": 0.0,
            f"{prefix}_min": 0.0,
        }
    return {
        f"{prefix}_count": float(len(values)),
        f"{prefix}_max": max(values),
        f"{prefix}_mean": mean(values),
        f"{prefix}_min": min(values),
    }


def entropy(values: list[str], weights: list[float] | None = None) -> float:
    if not values:
        return 0.0
    weights = weights or [1.0] * len(values)
    totals: dict[str, float] = {}
    for value, weight in zip(values, weights):
        totals[str(value)] = totals.get(str(value), 0.0) + max(0.0, float(weight))
    denom = sum(totals.values())
    if denom <= 0:
        return 0.0
    probs = [value / denom for value in totals.values() if value > 0]
    return float(-sum(p * math.log(p) for p in probs))


def source_diversity(items: list[str]) -> float:
    return float(len({str(item) for item in items if item}))


def card_counts(decision: Mapping[str, Any]) -> dict[str, float]:
    cards = list(decision.get("cards", []))
    out = {
        "card_count": float(len(cards)),
        "support_card_count": 0.0,
        "refute_card_count": 0.0,
        "blind_card_count": 0.0,
        "support_card_strength": 0.0,
        "refute_card_strength": 0.0,
    }
    for card in cards:
        polarity = str(card.get("polarity", ""))
        strength = safe_float(card.get("strength"))
        if polarity == "support":
            out["support_card_count"] += 1.0
            out["support_card_strength"] += strength
        elif polarity == "refute":
            out["refute_card_count"] += 1.0
            out["refute_card_strength"] += strength
        elif polarity == "blind":
            out["blind_card_count"] += 1.0
    return out


def signature_features(signature: Mapping[str, Any]) -> dict[str, float]:
    services = list(signature.get("services", []))
    out = {
        "signature_service_count": float(len(services)),
        "signature_dominant_count": float(len(signature.get("dominant_evidence_types", []))),
        "signature_blind_spot_count": float(len(signature.get("blind_spots", []))),
        "metric_service_count": 0.0,
        "log_service_count": 0.0,
        "trace_service_count": 0.0,
        "topology_service_count": 0.0,
        "metric_strength_max": 0.0,
        "log_strength_max": 0.0,
        "trace_strength_max": 0.0,
        "topology_strength_max": 0.0,
    }
    for service in services:
        for modality in ("metric", "log", "trace", "topology"):
            items = list(service.get(modality, {}).values())
            if items:
                out[f"{modality}_service_count"] += 1.0
            for item in items:
                out[f"{modality}_strength_max"] = max(
                    out[f"{modality}_strength_max"],
                    abs(safe_float(item.get("strength"))),
                )
    return out


def top_component_evidence_features(signature: Mapping[str, Any], component: str, bucket: str) -> dict[str, float]:
    service = None
    for row in signature.get("services", []):
        if str(row.get("service", "")) == str(component):
            service = row
            break
    out = {
        "top1_component_metric_bucket_support": 0.0,
        "top1_component_log_bucket_support": 0.0,
        "top1_component_trace_bucket_support": 0.0,
        "top1_component_topology_bucket_support": 0.0,
        "top1_component_modality_agreement": 0.0,
        "top1_component_bucket_strength_sum": 0.0,
        "top1_component_bucket_strength_max": 0.0,
    }
    if not service:
        return out
    supported = []
    for modality in ("metric", "log", "trace", "topology"):
        item = service.get(modality, {}).get(bucket)
        if not item:
            continue
        if str(item.get("state", "")) != "support":
            continue
        strength = abs(safe_float(item.get("strength")))
        out[f"top1_component_{modality}_bucket_support"] = 1.0
        out["top1_component_bucket_strength_sum"] += strength
        out["top1_component_bucket_strength_max"] = max(out["top1_component_bucket_strength_max"], strength)
        supported.append(modality)
    out["top1_component_modality_agreement"] = float(len(supported))
    return out


def decision_pool_features(decisions: list[Mapping[str, Any]]) -> dict[str, float]:
    top = decisions[:10]
    components = [str(row.get("candidate", {}).get("component", "")) for row in top]
    reasons = [str(row.get("candidate", {}).get("reason", "")) for row in top]
    buckets = [str(row.get("candidate", {}).get("reason_bucket", "")) for row in top]
    sources = [str(row.get("candidate", {}).get("source", "")) for row in top]
    rebuttals = [safe_float(row.get("rebuttal_score"), 999.0) for row in top]
    if rebuttals:
        min_rebuttal = min(rebuttals)
        weights = [math.exp(-min(50.0, max(-50.0, value - min_rebuttal))) for value in rebuttals]
    else:
        weights = []
    top1 = top[0] if top else {}
    top2 = top[1] if len(top) > 1 else {}
    top1_cand = dict(top1.get("candidate", {}))
    top2_cand = dict(top2.get("candidate", {}))
    support_values = [safe_float(row.get("support_strength")) for row in top]
    return {
        "topk_component_unique": source_diversity(components),
        "topk_reason_unique": source_diversity(reasons),
        "topk_reason_bucket_unique": source_diversity(buckets),
        "topk_source_unique": source_diversity(sources),
        "topk_component_entropy": entropy(components, weights),
        "topk_reason_entropy": entropy(reasons, weights),
        "topk_reason_bucket_entropy": entropy(buckets, weights),
        "top1_top2_same_component": float(top1_cand.get("component") == top2_cand.get("component") and bool(top1_cand)),
        "top1_top2_same_reason": float(top1_cand.get("reason") == top2_cand.get("reason") and bool(top1_cand)),
        "top1_top2_same_reason_bucket": float(top1_cand.get("reason_bucket") == top2_cand.get("reason_bucket") and bool(top1_cand)),
        "topk_rebuttal_range": (max(rebuttals) - min(rebuttals)) if rebuttals else 0.0,
        "topk_support_range": (max(support_values) - min(support_values)) if support_values else 0.0,
    }


def time_anchor_features(anchor_row: Mapping[str, Any]) -> dict[str, float]:
    anchor = dict(anchor_row.get("time_anchor", {}))
    votes = list(anchor.get("votes", []))
    rejected = list(anchor.get("rejected_votes", []))
    rejected_scores = [safe_float(row.get("score")) for row in rejected]
    selected_score = safe_float(anchor.get("score"))
    by_source = {"metric": 0.0, "log": 0.0, "trace": 0.0, "fallback": 0.0}
    vote_sources = []
    vote_kinds = []
    kind_flags: dict[str, float] = {}
    for vote in votes:
        source = str(vote.get("source", ""))
        by_source[source] = by_source.get(source, 0.0) + 1.0
        vote_sources.append(source)
        vote_kinds.append(str(vote.get("kind", "")))
        kind_flags[f"time_vote_kind_{vote.get('kind', 'unknown')}"] = 1.0
    policy = str(anchor.get("policy", ""))
    out = {
        "time_anchor_score": selected_score,
        "time_anchor_vote_count": float(len(votes)),
        "time_anchor_rejected_count": float(len(rejected)),
        "time_anchor_rejected_best_score": max(rejected_scores) if rejected_scores else 0.0,
        "time_anchor_score_margin": selected_score - (max(rejected_scores) if rejected_scores else 0.0),
        "time_anchor_metric_votes": by_source.get("metric", 0.0),
        "time_anchor_log_votes": by_source.get("log", 0.0),
        "time_anchor_trace_votes": by_source.get("trace", 0.0),
        "time_anchor_fallback_votes": by_source.get("fallback", 0.0),
        "time_anchor_vote_source_diversity": source_diversity(vote_sources),
        "time_anchor_vote_kind_diversity": source_diversity(vote_kinds),
        "time_policy_metric_sustained_cross_vote": float(policy == "metric_sustained_cross_vote"),
        "time_policy_network_trace_first_seen": float(policy == "network_trace_first_seen_with_metric_vote"),
        "time_policy_jvm_log_then_heap_onset": float(policy == "jvm_log_then_heap_onset"),
        "time_policy_fallback_window_start": float(policy == "fallback_window_start"),
    }
    out.update(kind_flags)
    return out


def row_features(
    debug_row: Mapping[str, Any],
    official_row: Mapping[str, Any],
    field_row: Mapping[str, Any],
    query_row: Mapping[str, Any],
) -> dict[str, Any]:
    result = debug_row["d32_result"]
    inner = result["debug"]
    decisions = list(inner.get("all_decisions", []))
    top1 = decisions[0] if decisions else {}
    top2 = decisions[1] if len(decisions) > 1 else {}
    candidate = dict(top1.get("candidate", {}))
    details = dict(candidate.get("details", {}))

    support = safe_float(top1.get("support_strength"))
    refute = safe_float(top1.get("refute_strength"))
    blind = safe_float(top1.get("blind_count"))
    top1_rebuttal = safe_float(top1.get("rebuttal_score"), 999.0)
    top2_rebuttal = safe_float(top2.get("rebuttal_score"), 999.0)
    top1_prior = safe_float(candidate.get("prior"))
    top2_prior = safe_float(dict(top2.get("candidate", {})).get("prior"))
    top1_support = support
    top2_support = safe_float(top2.get("support_strength"))

    clusters = list(inner.get("matched_clusters", []))
    rules = list(inner.get("mined_rules_matched", []))
    cluster_sims = [safe_float(row.get("similarity")) for row in clusters]
    rule_conf = [safe_float(row.get("confidence")) for row in rules]
    rule_val_conf = [safe_float(row.get("validation_confidence")) for row in rules]
    rule_val_dates = [safe_float(row.get("validation_dates")) for row in rules]
    rule_val_support = [safe_float(row.get("validation_support")) for row in rules]

    field_labels = field_case_labels(result["prediction"], str(query_row.get("scoring_points", "")))
    modal_status = dict(debug_row.get("modal_status", {}))
    time_anchors = list(inner.get("selected_time_anchors", []))
    anchor = time_anchors[0] if time_anchors else {}

    row: dict[str, Any] = {
        "row_id": safe_int(debug_row.get("row_id")),
        "heldout_date": str(debug_row.get("heldout_date", "")),
        "task_index": str(official_row.get("task_index", "")),
        "strict_label": int(bool(official_row.get("strict"))),
        "partial_label": int(bool(official_row.get("partial"))),
        "fractional_score": safe_float(official_row.get("fractional_score")),
        "required_total": safe_float(official_row.get("required_total")),
        "hit_total": safe_float(official_row.get("hits")),
        "top1_rebuttal_score": top1_rebuttal,
        "top1_support_strength": support,
        "top1_refute_strength": refute,
        "top1_blind_count": blind,
        "top1_prior": top1_prior,
        "top1_confidence_level": CONFIDENCE_LEVEL.get(str(top1.get("confidence", "LOW")), 0.0),
        "top1_source_layer1_cluster": float(candidate.get("source") == "layer1_fault_cluster"),
        "top1_source_layer1_rule": float(candidate.get("source") == "layer1_mined_rule"),
        "top1_source_event_fallback": float(candidate.get("source") == "event_fallback"),
        "top1_component_evidence_prior": safe_float(details.get("component_evidence_prior")),
        "top1_cluster_similarity": safe_float(details.get("similarity")),
        "top1_reason_prior_weight": safe_float(details.get("weight")),
        "top1_reason_prior_count": safe_float(details.get("count")),
        "top2_rebuttal_score": top2_rebuttal,
        "rebuttal_margin_top2_minus_top1": top2_rebuttal - top1_rebuttal,
        "support_margin_top1_minus_top2": top1_support - top2_support,
        "prior_margin_top1_minus_top2": top1_prior - top2_prior,
        "decision_count": float(len(decisions)),
        "high_suspicion_count": float(len(result.get("high_suspicion", []))),
        "low_suspicion_count": float(len(result.get("low_suspicion", []))),
        "candidate_space_count": float(len(inner.get("candidate_space", []))),
        "data_blind_spot_count": float(len(result.get("data_blind_spots", []))),
        "support_refute_ratio": support / (refute + 1.0),
        "support_per_blind": support / (blind + 1.0),
        "metric_present": float(modal_status.get("metric") == "present"),
        "log_present": float(modal_status.get("log") == "present"),
        "trace_present": float(modal_status.get("trace") == "present"),
    }
    row.update(field_labels)
    row.update(card_counts(top1))
    row.update(signature_features(inner.get("signature", {})))
    row.update(top_component_evidence_features(inner.get("signature", {}), str(candidate.get("component", "")), str(candidate.get("reason_bucket", ""))))
    row.update(decision_pool_features(decisions))
    row.update(time_anchor_features(anchor))
    row.update(scalar_stats(cluster_sims, "matched_cluster_similarity"))
    row.update(scalar_stats(rule_conf, "matched_rule_confidence"))
    row.update(scalar_stats(rule_val_conf, "matched_rule_validation_confidence"))
    row.update(scalar_stats(rule_val_dates, "matched_rule_validation_dates"))
    row.update(scalar_stats(rule_val_support, "matched_rule_validation_support"))
    return row


def write_csv(rows: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    debug = load_json(args.debug_json)
    official = load_json(args.official_case_eval)
    field_diag = load_json(args.field_diag)
    query_rows = load_csv(args.query)

    debug_rows = sorted(debug["debug"], key=lambda row: int(row["row_id"]))
    official_rows = sorted(official["rows"], key=lambda row: int(row["row_id"]))
    field_rows = sorted(field_diag["rows"], key=lambda row: int(row["row_id"]))
    if not (len(debug_rows) == len(official_rows) == len(field_rows) == len(query_rows)):
        raise ValueError(f"row mismatch: debug={len(debug_rows)} official={len(official_rows)} field={len(field_rows)} query={len(query_rows)}")

    rows = [row_features(drow, orow, frow, qrow) for drow, orow, frow, qrow in zip(debug_rows, official_rows, field_rows, query_rows)]
    write_csv(rows, args.out)
    positives = {
        "strict": sum(row["strict_label"] for row in rows),
        "partial": sum(row["partial_label"] for row in rows),
        "time_hit": sum(row["time_hit_label"] for row in rows if row["time_required"]),
        "component_hit": sum(row["component_hit_label"] for row in rows if row["component_required"]),
        "reason_hit": sum(row["reason_hit_label"] for row in rows if row["reason_required"]),
        "component_reason_pair": sum(row["component_reason_pair_hit_label"] for row in rows if row["component_reason_pair_required"]),
        "actionable": sum(row["actionable_hit_label"] for row in rows if row["actionable_required"]),
    }
    print(json.dumps({"out": args.out, "n": len(rows), "positives": positives}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
