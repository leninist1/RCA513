"""Offline candidate-recall evaluation (deterministic, no LLM).

Reproduces and extends the recall table from ``PRISM_CHT_HANDOFF.md``:
for each RCAEval case, find the rank of the expected root component
under several ranking strategies, and report hit@k distributions.

This is the upper-bound check for whether recall alone limits RE3-TT
accuracy, before any prompt or structural work.  It calls no LLM and
needs no provider env vars.

Run from the repo root so ``python -m prismv4...`` resolves:

    python -m prismv4.scripts.offline_recall_eval \
        --data-root /home/dell2/RCA513/ysj/dataset/RCAEval/RE3 \
        --system RE3-TT --system RE3-OB
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from prismv4.experiments.rcaeval_adapter import (
    RCAEvalTelemetryStore,
    discover_re3_cases,
    load_re3_case,
)

NEAR_ONSET_INTERNAL = {"cpu", "disk", "error", "log", "memory", "socket"}
ENTRY_HINTS = ("frontend", "gateway", "ingress")


@dataclass(frozen=True)
class ComponentScore:
    component: str
    # Raw signals reused across strategies.
    total_magnitude: float
    near_onset_magnitude: float
    earliest_time: float | None
    near_onset_internal: tuple[str, ...]
    latency_only_near_onset: bool
    log_count: int
    has_emitter_exception: bool
    mechanism_level: str
    direct_callers: tuple[str, ...]
    direct_callees: tuple[str, ...]
    is_entry_like_name: bool
    has_workload_signal: bool
    workload_magnitude: float


@dataclass
class CaseRecall:
    case_id: str
    expected: str
    n_components: int
    expected_present: bool
    ranks: dict[str, int | None] = field(default_factory=dict)
    pool_hits: dict[str, bool] = field(default_factory=dict)
    top5_magnitude: tuple[str, ...] = ()


def _score_components(
    store: RCAEvalTelemetryStore,
    *,
    case_components: Sequence[str],
    event_time: float,
) -> tuple[ComponentScore, ...]:
    """Build per-component score rows reusing the adapter's own feature row."""
    window = (event_time - 30.0, event_time + 600.0)
    trace_context = store._trace_dependency_context(
        component_scope=case_components,
        time_window=window,
    )
    rows: list[ComponentScore] = []
    for component in case_components:
        feature = store._noise_feature_row(component=component, time_window=window)
        metrics = store._component_metric_observations(component)
        workload_mag = sum(
            float(m["magnitude"]) for m in metrics if m["signal"] == "workload"
        )
        has_workload = any(m["signal"] == "workload" for m in metrics)
        tc = trace_context.get(component, {})
        callers = tuple(
            str(item.get("peer"))
            for item in tc.get("direct_callers", [])
            if isinstance(item, Mapping) and item.get("peer")
        )
        callees = tuple(
            str(item.get("peer"))
            for item in tc.get("direct_callees", [])
            if isinstance(item, Mapping) and item.get("peer")
        )
        from prismv4.experiments.rcaeval_adapter import _mechanism_strength

        mech = _mechanism_strength(feature, tc)
        log_features = feature.get("log_features", []) or []
        has_emitter_exception = any(
            (f.get("emitter_exception_observed") if isinstance(f, Mapping) else False)
            or (f.get("diagnostic_role") == "emitter_error_or_internal_exception"
                if isinstance(f, Mapping) else False)
            for f in log_features
        )
        rows.append(
            ComponentScore(
                component=component,
                total_magnitude=float(feature.get("local_anomaly_magnitude", 0.0)),
                near_onset_magnitude=float(feature.get("near_onset_anomaly_magnitude", 0.0)),
                earliest_time=feature.get("earliest_observed_time"),
                near_onset_internal=tuple(feature.get("near_onset_internal_signals", []) or ()),
                latency_only_near_onset=bool(feature.get("latency_only_near_onset")),
                log_count=int(feature.get("log_count", 0) or 0),
                has_emitter_exception=has_emitter_exception,
                mechanism_level=str(mech.get("level", "unknown")),
                direct_callers=callers,
                direct_callees=callees,
                is_entry_like_name=any(h in component.lower() for h in ENTRY_HINTS),
                has_workload_signal=has_workload,
                workload_magnitude=workload_mag,
            )
        )
    return tuple(rows)


def _rank_of(scores: Sequence[ComponentScore], expected: str, *, key) -> int | None:
    ordered = sorted(scores, key=key)
    for idx, s in enumerate(ordered, start=1):
        if s.component == expected:
            return idx
    return None


def _magnitude_key(s: ComponentScore):
    # Current adapter behaviour: (-magnitude, first_seen, component).
    return (-s.total_magnitude, s.earliest_time if s.earliest_time is not None else float("inf"), s.component)


def _magnitude_no_workload_key(s: ComponentScore):
    adjusted = s.total_magnitude - s.workload_magnitude
    return (-adjusted, s.earliest_time if s.earliest_time is not None else float("inf"), s.component)


def _near_onset_magnitude_key(s: ComponentScore):
    return (-s.near_onset_magnitude, s.earliest_time if s.earliest_time is not None else float("inf"), s.component)


def _onset_key(s: ComponentScore):
    return (s.earliest_time if s.earliest_time is not None else float("inf"), -s.near_onset_magnitude, s.component)


def _mechanism_key(s: ComponentScore):
    # Prefer strong/moderate mechanism, then near-onset internal breadth, then near-onset magnitude.
    level_rank = {"strong": 0, "moderate": 1, "weak": 2, "symptom_like": 3, "unknown": 4}.get(
        s.mechanism_level, 4
    )
    internal_count = len(set(s.near_onset_internal) - {"log"})
    return (level_rank, -internal_count, -s.near_onset_magnitude, s.earliest_time if s.earliest_time is not None else float("inf"), s.component)


def _tiered_union(scores: Sequence[ComponentScore], *, pool_size: int) -> tuple[str, ...]:
    """Union of several tiered selections, capped at pool_size.

    Tiers mirror the Next Agent Checklist #2:
      - magnitude (workload down-weighted)
      - earliest onset
      - near-onset internal mechanism
      - log/error emitters
      - topology-central / entry-like (callers with no callers, or high degree)
    """
    pool: list[str] = []
    seen: set[str] = set()

    def add(top: Sequence[str], quota: int) -> None:
        for comp in top:
            if comp in seen:
                continue
            if len(pool) >= pool_size:
                return
            pool.append(comp)
            seen.add(comp)
            if sum(1 for _ in filter(lambda c: c in seen, pool)) >= quota:
                break

    per_tier = max(2, pool_size // 5)
    mag_nowl = [s.component for s in sorted(scores, key=_magnitude_no_workload_key)]
    onset = [s.component for s in sorted(scores, key=_onset_key)]
    mechanism = [s.component for s in sorted(scores, key=_mechanism_key)]
    emitters = [
        s.component
        for s in sorted(
            scores,
            key=lambda s: (-int(s.has_emitter_exception), -s.log_count, -s.near_onset_magnitude, s.component),
        )
        if s.has_emitter_exception or s.log_count > 0
    ]
    # Topology-central: components that act as callers (have callees) and few/no callers,
    # i.e. near the trace root; break ties by total degree.
    degree = {
        s.component: (len(s.direct_callers), len(s.direct_callees))
        for s in scores
    }
    topo = sorted(
        scores,
        key=lambda s: (
            len(s.direct_callers) == 0 and len(s.direct_callees) > 0,
            -(len(s.direct_callers) + len(s.direct_callees)),
            s.component,
        ),
        reverse=False,
    )
    # Sort topo by the tuple directly (True sorts after False; we want True first).
    topo.sort(key=lambda s: (not (len(s.direct_callers) == 0 and len(s.direct_callees) > 0), -(len(s.direct_callers) + len(s.direct_callees)), s.component))
    topo_names = [s.component for s in topo if (len(s.direct_callers) + len(s.direct_callees)) > 0]

    add(mag_nowl, per_tier)
    add(onset, per_tier)
    add(mechanism, per_tier)
    add(emitters, per_tier)
    add(topo_names, per_tier)
    # Fill remaining from workload-down-weighted magnitude.
    add(mag_nowl, pool_size)
    return tuple(pool[:pool_size])


def evaluate_case(loaded, *, pool_size: int = 20) -> CaseRecall:
    case = loaded.case
    expected = loaded.expected_component
    scores = _score_components(
        loaded.store,
        case_components=case.components,
        event_time=case.event_time,
    )
    present = any(s.component == expected for s in scores)
    recall = CaseRecall(
        case_id=case.case_id,
        expected=expected,
        n_components=len(case.components),
        expected_present=present,
    )
    if not present:
        return recall

    strategies = {
        "magnitude (baseline)": _magnitude_key,
        "magnitude_no_workload": _magnitude_no_workload_key,
        "near_onset_magnitude": _near_onset_magnitude_key,
        "earliest_onset": _onset_key,
        "mechanism_priority": _mechanism_key,
    }
    for name, key in strategies.items():
        recall.ranks[name] = _rank_of(scores, expected, key=key)

    pool = _tiered_union(scores, pool_size=pool_size)
    recall.pool_hits[f"tiered_union@{pool_size}"] = expected in pool
    recall.pool_hits[f"tiered_union@{min(pool_size, 15)}"] = expected in pool[:15]
    recall.pool_hits[f"tiered_union@10"] = expected in pool[:10]
    recall.top5_magnitude = tuple(
        s.component for s in sorted(scores, key=_magnitude_key)[:5]
    )
    return recall


def _hit_bucket(rank: int | None, k: int) -> bool:
    return rank is not None and rank <= k


def summarize(cases: list[CaseRecall], label: str) -> dict[str, Any]:
    total = len(cases)
    present = sum(1 for c in cases if c.expected_present)
    out: dict[str, Any] = {
        "system": label,
        "n_cases": total,
        "expected_in_components": present,
    }
    for name in (
        "magnitude (baseline)",
        "magnitude_no_workload",
        "near_onset_magnitude",
        "earliest_onset",
        "mechanism_priority",
    ):
        ranks = [c.ranks.get(name) for c in cases if c.expected_present]
        for k in (1, 3, 5, 10, 15, 20):
            out[f"hit@{k}_{name}"] = sum(1 for r in ranks if _hit_bucket(r, k))
        present_for_strategy = len(ranks)
        out[f"recall_present_{name}"] = present_for_strategy
    for key in ("tiered_union@20", "tiered_union@15", "tiered_union@10"):
        out[key] = sum(1 for c in cases if c.pool_hits.get(key))
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--system", action="append", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--pool-size", type=int, default=20)
    parser.add_argument("--show-misses", action="store_true")
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    all_summaries: list[dict[str, Any]] = []
    all_cases: list[CaseRecall] = []
    for system in args.system:
        cases_paths = discover_re3_cases(args.data_root, system=system, limit=args.limit)
        print(f"== {system}: {len(cases_paths)} cases ==", flush=True)
        cases: list[CaseRecall] = []
        for case_dir in cases_paths:
            loaded = load_re3_case(case_dir, top_k=9999)
            recall = evaluate_case(loaded, pool_size=args.pool_size)
            cases.append(recall)
        all_cases.extend(cases)
        summ = summarize(cases, system)
        all_summaries.append(summ)
        print(json.dumps(summ, indent=2), flush=True)

        if args.show_misses:
            misses = [
                c
                for c in cases
                if c.expected_present
                and not _hit_bucket(c.ranks.get("magnitude (baseline)"), 5)
            ]
            print(f"-- {system}: {len(misses)} cases where expected not in top-5 (magnitude) --", flush=True)
            for c in misses:
                rank = c.ranks.get("magnitude (baseline)")
                nowl = c.ranks.get("magnitude_no_workload")
                mech = c.ranks.get("mechanism_priority")
                onset = c.ranks.get("earliest_onset")
                pool15 = c.pool_hits.get("tiered_union@15")
                print(
                    f"  {c.case_id}: expected={c.expected} mag_rank={rank} "
                    f"magNoWL_rank={nowl} mech_rank={mech} onset_rank={onset} "
                    f"pool15_hit={pool15} top5={list(c.top5_magnitude)}",
                    flush=True,
                )

    report = {
        "pool_size": args.pool_size,
        "summaries": all_summaries,
        "cases": [
            {
                "system": c.case_id.split("/")[0],
                "case_id": c.case_id,
                "expected": c.expected,
                "n_components": c.n_components,
                "expected_present": c.expected_present,
                "ranks": c.ranks,
                "pool_hits": c.pool_hits,
                "top5_magnitude": list(c.top5_magnitude),
            }
            for c in all_cases
        ],
    }
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
