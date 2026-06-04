"""Layer 1: offline d32 knowledge construction."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import json
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from refute_b_v2.query_windows import parse_query_window
from refute_b_v2_d32.signature import build_case_signature, cosine_sparse, flatten_signature


@dataclass(frozen=True)
class KnowledgeBuildConfig:
    cluster_threshold: float = 0.48
    min_rule_support: int = 3
    min_rule_confidence: float = 0.62
    max_rule_len: int = 2
    validate_rules_lodo: bool = True
    min_rule_validation_support: int = 1
    min_rule_validation_confidence: float = 0.50
    min_rule_validation_dates: int = 2


def load_trace_summary(base_dir: str | Path, row_id: int) -> dict | None:
    if not base_dir:
        return None
    path = Path(base_dir) / f"query_{int(row_id):03d}.json"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def gather_window(cache, window, modalities: set[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = []
    logs = []
    for date_key in window.date_keys:
        if "metric" in modalities:
            metrics.append(_filter_between(cache.metric_day(date_key), window.start_ts, window.end_ts))
        if "log" in modalities:
            logs.append(_filter_between(cache.log_day(date_key), window.start_ts, window.end_ts))
    metric_df = pd.concat(metrics, ignore_index=True) if metrics else pd.DataFrame(columns=["timestamp", "cmdb_id", "kpi_name", "value"])
    log_df = pd.concat(logs, ignore_index=True) if logs else pd.DataFrame(columns=["timestamp", "cmdb_id", "value"])
    return metric_df, log_df


def _filter_between(df: pd.DataFrame, start_ts: int, end_ts: int) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    return df[(df["timestamp"] >= start_ts) & (df["timestamp"] < end_ts)].copy()


def case_rows_from_openrca(query_csv: str | Path, record_csv: str | Path) -> list[dict[str, Any]]:
    query_df = pd.read_csv(query_csv)
    record_df = pd.read_csv(record_csv)
    rows = []
    for idx, qrow in query_df.iterrows():
        rec = record_df.iloc[int(idx)]
        rows.append({
            "row_id": int(idx),
            "task_index": str(qrow.get("task_index", "")),
            "instruction": str(qrow["instruction"]),
            "component": str(rec["component"]).strip(),
            "reason": str(rec["reason"]).strip(),
            "datetime": str(rec.get("datetime", "")),
            "date_key": pd.to_datetime(rec.get("datetime", "")).strftime("%Y_%m_%d"),
        })
    return rows


def build_knowledge(
    *,
    case_rows: Iterable[Mapping[str, Any]],
    cache,
    baseline,
    trace_summary_dir: str | Path,
    modalities: set[str],
    config: KnowledgeBuildConfig | None = None,
) -> dict[str, Any]:
    config = config or KnowledgeBuildConfig()
    cases = []
    for row in case_rows:
        row_id = int(row["row_id"])
        window = parse_query_window(str(row["instruction"]))
        metric_df, log_df = gather_window(cache, window, modalities)
        trace_summary = load_trace_summary(trace_summary_dir, row_id) if "trace" in modalities else None
        modal_status = {
            "metric": "present" if "metric" in modalities and not metric_df.empty else ("empty_window" if "metric" in modalities else "disabled"),
            "log": "present" if "log" in modalities and not log_df.empty else ("empty_window" if "log" in modalities else "disabled"),
            "trace": (trace_summary or {}).get("trace_status", "unloaded") if "trace" in modalities else "disabled",
        }
        signature = build_case_signature(f"query_{row_id:03d}", metric_df, log_df, trace_summary, baseline, modal_status)
        cases.append({
            "case_id": f"query_{row_id:03d}",
            "row_id": row_id,
            "component": str(row["component"]),
            "reason": str(row["reason"]),
            "task_index": str(row.get("task_index", "")),
            "date_key": str(row.get("date_key", "")),
            "signature": signature,
            "features": flatten_signature(signature),
        })
    clusters = _cluster_cases(cases, config.cluster_threshold)
    mined_rules = _mine_rules(cases, config)
    return {
        "version": 1,
        "design": "d32_layer1_knowledge",
        "config": config.__dict__,
        "cases": [_case_public(row) for row in cases],
        "clusters": clusters,
        "mined_rules": mined_rules,
        "metadata": {"n_cases": len(cases), "modalities": sorted(modalities)},
    }


def _case_public(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "case_id": row["case_id"],
        "row_id": row["row_id"],
        "component": row["component"],
        "reason": row["reason"],
        "task_index": row["task_index"],
        "date_key": row.get("date_key", ""),
        "signature": row["signature"],
    }


def _cluster_cases(cases: list[Mapping[str, Any]], threshold: float) -> list[dict[str, Any]]:
    unassigned = {row["case_id"] for row in cases}
    by_id = {row["case_id"]: row for row in cases}
    clusters = []
    while unassigned:
        seed_id = sorted(unassigned)[0]
        seed = by_id[seed_id]
        members = [seed_id]
        unassigned.remove(seed_id)
        for other_id in sorted(list(unassigned)):
            if cosine_sparse(seed["features"], by_id[other_id]["features"]) >= threshold:
                members.append(other_id)
                unassigned.remove(other_id)
        member_rows = [by_id[mid] for mid in members]
        centroid = _centroid([row["features"] for row in member_rows])
        reason_counts = Counter(row["reason"] for row in member_rows)
        reasons = [
            {"reason": reason, "count": count, "weight": count / len(member_rows)}
            for reason, count in reason_counts.most_common()
        ]
        clusters.append({
            "cluster_id": f"cluster_{len(clusters):03d}",
            "size": len(member_rows),
            "members": members,
            "centroid": centroid,
            "reason_prior": reasons,
            "reason_distribution": dict(reason_counts.most_common()),
        })
    return clusters


def _centroid(features: list[Mapping[str, float]]) -> dict[str, float]:
    sums: dict[str, float] = defaultdict(float)
    for row in features:
        for key, value in row.items():
            sums[key] += float(value)
    n = max(1, len(features))
    return {key: value / n for key, value in sorted(sums.items())}


def _mine_rules(cases: list[Mapping[str, Any]], config: KnowledgeBuildConfig) -> list[dict[str, Any]]:
    raw_rules = _mine_rules_once(cases, config.min_rule_support, config.min_rule_confidence, config.max_rule_len)
    if not config.validate_rules_lodo:
        return raw_rules
    dates = sorted({str(row.get("date_key", "")) for row in cases if row.get("date_key")})
    if len(dates) < 2:
        return raw_rules
    candidates: dict[tuple[tuple[str, ...], str], dict[str, Any]] = {}
    for heldout in dates:
        train_rows = [row for row in cases if str(row.get("date_key", "")) != heldout]
        validation_rows = [row for row in cases if str(row.get("date_key", "")) == heldout]
        fold_rules = _mine_rules_once(train_rows, config.min_rule_support, config.min_rule_confidence, config.max_rule_len)
        for rule in fold_rules:
            stats = _validate_rule_on_rows(rule, validation_rows)
            if stats["validation_support"] == 0:
                continue
            key = (tuple(rule.get("antecedent", [])), str(rule.get("target_reason", "")))
            acc = candidates.setdefault(key, {
                "antecedent": list(rule.get("antecedent", [])),
                "target_reason": str(rule.get("target_reason", "")),
                "support": 0,
                "confidence": 0.0,
                "lift": 0.0,
                "validation_support": 0,
                "validation_hits": 0,
                "validation_dates_set": set(),
            })
            acc["support"] = max(int(acc["support"]), int(rule.get("support", 0)))
            acc["confidence"] = max(float(acc["confidence"]), float(rule.get("confidence", 0.0)))
            acc["lift"] = max(float(acc["lift"]), float(rule.get("lift", 0.0)))
            acc["validation_support"] += stats["validation_support"]
            acc["validation_hits"] += stats["validation_hits"]
            if stats["validation_support"]:
                acc["validation_dates_set"].add(heldout)
    validated = []
    for row in candidates.values():
        validation_support = int(row["validation_support"])
        validation_hits = int(row["validation_hits"])
        validation_dates = len(row.pop("validation_dates_set"))
        validation_confidence = validation_hits / validation_support if validation_support else 0.0
        if validation_support < config.min_rule_validation_support:
            continue
        if validation_dates < config.min_rule_validation_dates:
            continue
        if validation_confidence < config.min_rule_validation_confidence:
            continue
        row["validation_confidence"] = validation_confidence
        row["validation_dates"] = validation_dates
        validated.append(row)
    return sorted(
        validated,
        key=lambda row: (
            -float(row.get("validation_confidence", 0.0)),
            -int(row.get("validation_support", 0)),
            -float(row.get("confidence", 0.0)),
            -float(row.get("lift", 0.0)),
            row.get("antecedent", []),
            row.get("target_reason", ""),
        ),
    )[:500]


def _mine_rules_once(cases: list[Mapping[str, Any]], min_support: int, min_confidence: float, max_len: int) -> list[dict[str, Any]]:
    rows = [(tuple(sorted(row["features"])), str(row["reason"])) for row in cases]
    target_counts = Counter(target for _, target in rows)
    item_counts: Counter[tuple[str, ...]] = Counter()
    item_target_counts: Counter[tuple[tuple[str, ...], str]] = Counter()
    for features, target in rows:
        for size in range(1, max_len + 1):
            for item in combinations(features, size):
                item_counts[item] += 1
                item_target_counts[(item, target)] += 1
    out = []
    n = max(1, len(rows))
    for (item, target), support in item_target_counts.items():
        if support < min_support:
            continue
        confidence = support / item_counts[item]
        if confidence < min_confidence:
            continue
        prior = target_counts[target] / n
        lift = confidence / prior if prior else 0.0
        out.append({
            "antecedent": list(item),
            "target_reason": target,
            "support": support,
            "confidence": confidence,
            "lift": lift,
        })
    return sorted(out, key=lambda row: (-row["confidence"], -row["lift"], -row["support"], row["antecedent"], row["target_reason"]))[:500]


def _validate_rule_on_rows(rule: Mapping[str, Any], rows: list[Mapping[str, Any]]) -> dict[str, int]:
    antecedent = set(rule.get("antecedent", []))
    target = str(rule.get("target_reason", ""))
    support = 0
    matched = 0
    for row in rows:
        if antecedent and antecedent.issubset(set(row["features"])):
            support += 1
            if str(row["reason"]) == target:
                matched += 1
    return {
        "validation_support": support,
        "validation_hits": matched,
    }


class D32Knowledge:
    def __init__(self, data: Mapping[str, Any], min_cluster_similarity: float = 0.12):
        self.data = dict(data)
        self.clusters = list(data.get("clusters", []))
        self.mined_rules = list(data.get("mined_rules", []))
        self.min_cluster_similarity = min_cluster_similarity

    @classmethod
    def load_json(cls, path: str | Path, min_cluster_similarity: float = 0.12) -> "D32Knowledge":
        with Path(path).open("r", encoding="utf-8") as f:
            return cls(json.load(f), min_cluster_similarity=min_cluster_similarity)

    def match_clusters(self, signature: Mapping[str, Any], case_id: str | None = None, k: int = 3) -> list[dict[str, Any]]:
        query = flatten_signature(signature)
        matches = []
        for cluster in self.clusters:
            sim = cosine_sparse(query, cluster.get("centroid", {}))
            if sim < self.min_cluster_similarity:
                continue
            reasons = _reason_prior(cluster)
            matches.append({
                "cluster_id": cluster["cluster_id"],
                "similarity": sim,
                "size": cluster.get("size", 0),
                "reason_prior": reasons,
                "members": cluster.get("members", []),
            })
        return sorted(matches, key=lambda row: (-row["similarity"], row["cluster_id"]))[:k]

    def mined_reason_priors(self, signature: Mapping[str, Any]) -> list[dict[str, Any]]:
        features = set(flatten_signature(signature))
        out = []
        for rule in self.mined_rules:
            antecedent = set(rule.get("antecedent", []))
            if antecedent and antecedent.issubset(features):
                out.append(dict(rule))
        return sorted(out, key=lambda row: (-float(row.get("confidence", 0.0)), -float(row.get("lift", 0.0))))[:20]


def _reason_prior(cluster: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [dict(row) for row in cluster.get("reason_prior", [])]


def save_knowledge(data: Mapping[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
