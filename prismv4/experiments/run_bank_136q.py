#!/usr/bin/env python
"""PRISM-LA Experimental Script — Bank 136q accuracy evaluation with DeepSeek.

V3: NoiseLab signal extraction + PELT v2 time anchor detection.

Usage:
    python experiments/run_bank_136q.py [--max-steps 6] [--max-events 8] [--dry-run]
    nohup python experiments/run_bank_136q.py --max-steps 6 --max-events 8 > experiment.log 2>&1 &
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("prism_la.experiment")

ROOT = Path("/home/dell2/RCA513/yyx")
sys.path.insert(0, str(ROOT / "prismv3-no-leakage-hardening"))
sys.path.insert(0, str(ROOT / "prismv4"))

from prism_la.config import PRISMLAConfig
from prism_la.controller import PRISMLAController
from prism_la.time_anchor_v2 import build_anchor_set_v2
from prism_v3.data.loader import OpenRCALoader
from prism_v3.config import QueryCase
from prism_v3.leakage_guard import build_query_id
from prism_v3.evaluation.scorer import evaluate_prediction
from prism_v3.evaluation.aggregator import EvalAggregator
from prism_v3.noise_native.noiselab_adapter import NoiseLabEvidenceAdapter
from prism_v3.noise_native.evidence_frame import canonical_entity_name
from prism_v3.mace.graph import build_object_graph
from prism_v3.reason_taxonomy import reason_candidates_from_votes, normalize_reason

# Bank ground-truth reason vocabulary normalization
BANK_REASON_MAP = {
    "cpu fault": "high CPU usage",
    "high memory usage": "high memory usage",
    "disk i/o consumption": "high disk I/O read usage",
    "disk io consumption": "high disk I/O read usage",
    "disk space consumption": "high disk space usage",
    "network fault": "network packet loss",
    "network latency": "network latency",
    "network packet loss": "network packet loss",
    "db fault": "network latency",
    "process termination": "high memory usage",
    "jvm out of memory oom heap": "JVM Out of Memory (OOM) Heap",
    "high jvm cpu load": "high JVM CPU load",
    "high cpu usage": "high CPU usage",
    "high disk i/o read usage": "high disk I/O read usage",
    "high disk i/o usage": "high disk I/O read usage",
    "high disk io read usage": "high disk I/O read usage",
    "configuration change": "network latency",
    "unspecified anomaly": "high memory usage",
    "unspecified service error": "high memory usage",
    "high disk io usage": "high disk I/O read usage",
    "disk i/o read usage": "high disk I/O read usage",
}


def bank_normalize_reason(raw_reason: str) -> str:
    key = raw_reason.lower().strip()
    return BANK_REASON_MAP.get(key, raw_reason)

DEEPSEEK_API_KEY = "sk-e094fac2651a45f38cb93b3e15093604"
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL = "deepseek-chat"
RESULT_DIR = ROOT / "prismv4" / "results" / "prism_la"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PRISM-LA Bank 136q Experiment v3")
    p.add_argument("--max-steps", type=int, default=6)
    p.add_argument("--max-events", type=int, default=8)
    p.add_argument("--model", default=DEEPSEEK_MODEL)
    p.add_argument("--temperature", type=float, default=0.15)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--max-queries", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def process_query(
    query: Any,
    telemetry: Any,
    config: PRISMLAConfig,
) -> Dict[str, Any]:
    query_id = build_query_id(query)
    entities = list(telemetry.entities) if hasattr(telemetry, "entities") else []
    entity_types = dict(telemetry.entity_types or {}) if hasattr(telemetry, "entity_types") else {}
    if not entities:
        return {"component": [], "reason": [], "time": [], "top_score": 0.0, "fault_count": 0,
                "root_cause_events": [], "error": "no_entities"}

    inject_time = float(getattr(query, "inject_time", 0.0) or 0.0)

    # --- Object graph (needed for time + reason from NoiseLab) ---
    object_graph, _graph_debug = build_object_graph(telemetry, query, inject_time)
    entity_reason_votes: Dict[str, Dict[str, float]] = {}
    entity_earliest_ts: Dict[str, float] = {}
    entity_best_reason: Dict[str, str] = {}
    for node_id, node in getattr(object_graph, "nodes", {}).items():
        votes = getattr(node, "reason_votes", {}) or {}
        if votes:
            cn = canonical_entity_name(node_id)
            entity_reason_votes[cn] = dict(votes)
            best = max(votes, key=votes.get)
            entity_best_reason[cn] = normalize_reason(str(best), system="Bank")
        ets = getattr(node, "earliest_timestamp", None)
        if ets is not None:
            entity_earliest_ts[canonical_entity_name(node_id)] = float(ets)

    # --- Time anchors: PELT v2 ---
    anchors_obj = build_anchor_set_v2(telemetry, query, top_k=5)
    pelt_anchors = [
        {"timestamp": float(a.timestamp), "confidence": float(a.confidence), "source": str(a.source)}
        for a in anchors_obj.anchors
    ]

    # --- NoiseLab signal extraction ---
    try:
        adapter = NoiseLabEvidenceAdapter(strategy="base", temperature=1.0)
        evidence_frame = adapter.from_runtime_scorer(telemetry, query, inject_time, entities)
        candidates = evidence_frame.candidates if evidence_frame.applied else []
    except Exception as exc:
        logger.warning("NoiseLab adapter failed for %s: %s", query_id, exc)
        candidates = []

    n = len(entities)
    eidx = {canonical_entity_name(e): i for i, e in enumerate(entities)}
    metric_arr = np.zeros(n, dtype=float)
    log_arr = np.zeros(n, dtype=float)
    candidate_scores: Dict[str, Dict[str, float]] = {}
    metric_detail: Dict[str, Dict[str, float]] = {}
    log_detail: Dict[str, Dict[str, Any]] = {}
    entity_reason_hypothesis: Dict[str, str] = {}
    best_noiselab_time: Optional[float] = None

    for rank_i, c in enumerate(candidates):
        cid = c.canonical_component
        idx = eidx.get(cid)
        if idx is not None:
            e_name = str(entities[idx])
            root_score = float(c.source_likelihood or 0.0) * 0.6 + float(c.noise_score or 0.0) * 0.4
            metric_arr[idx] = max(metric_arr[idx], root_score)
            log_arr[idx] = max(log_arr[idx], float(c.log_evidence.get("log_score", 0.0) or 0.0))
            candidate_scores[e_name] = {
                "root_score": float(c.noise_score or 0.0),
                "symptom_score": float(c.symptomness or 0.0),
                "source_isolation": float(c.source_likelihood or 0.0),
                "hotspot_symptom": float(c.symptomness or 0.0),
                "downstream_recovery": 0.0,
                "broad_explainer": 0.0,
            }
            m_ev = c.metric_evidence
            if m_ev:
                metric_detail[e_name] = {
                    "anomaly_score": float(m_ev.get("anomaly_score", 0.0) or 0.0),
                    "metric_score": float(m_ev.get("metric_score", 0.0) or 0.0),
                    "change_score": float(m_ev.get("change_score", 0.0) or 0.0),
                }
            log_detail[e_name] = {
                "count": int(float(c.log_evidence.get("log_score", 0.0) or 0.0) * 10),
                "fatal_boost": 1.0 + float(c.log_evidence.get("log_score", 0.0) or 0.0),
                "text": "",
                "total": 1,
            }
            # Reason from NoiseLab reason_votes (mapped to Bank GT vocabulary)
            rv = entity_reason_votes.get(cid, {})
            if rv:
                rc = reason_candidates_from_votes(rv, limit=1)
                if rc:
                    entity_reason_hypothesis[e_name] = bank_normalize_reason(rc[0].reason)
            # Time from NoiseLab earliest_timestamp
            ets = entity_earliest_ts.get(cid)
            if ets and (best_noiselab_time is None or rank_i == 0):
                if rank_i == 0:
                    best_noiselab_time = ets

    # Fill in any entities not covered by NoiseLab
    for i, e in enumerate(entities):
        if e not in candidate_scores:
            candidate_scores[e] = {"root_score": 0.0, "symptom_score": 0.0, "source_isolation": 0.0,
                                   "hotspot_symptom": 0.0, "downstream_recovery": 0.0, "broad_explainer": 0.0}

    # --- Anchor set: use NoiseLab best time as primary anchor, PELT as backup ---
    anchor_set = list(pelt_anchors)
    if best_noiselab_time:
        # Prepend NoiseLab time as highest-confidence anchor
        anchor_set.insert(0, {
            "timestamp": float(best_noiselab_time),
            "confidence": 0.95,
            "source": "noiselab_earliest_ts",
        })

    # --- Trace graph from candidate structural features ---
    graph_mat = np.zeros((n, n), dtype=float)
    for c in candidates:
        cid = c.canonical_component
        idx = eidx.get(cid)
        if idx is None:
            continue
        features = c.structural_features
        if features:
            deg_in = float(c.trace_evidence.get("degree_in", 0.0) or 0.0)
            deg_out = float(c.trace_evidence.get("degree_out", 0.0) or 0.0)
            if deg_out > 0 or deg_in > 0:
                total = max(deg_in + deg_out, 1.0)
                graph_mat[idx, :] = deg_out / total
                graph_mat[:, idx] = deg_in / total
    if graph_mat.sum() < 1e-9:
        np.fill_diagonal(graph_mat, 0.01)

    topology_edges: List[Tuple[str, str, float]] = [
        (entities[i], entities[j], float(graph_mat[i, j]))
        for i in range(n) for j in range(n) if i != j and graph_mat[i, j] > 1e-6
    ]

    telemetry_meta = {
        "has_metrics": True, "has_logs": True, "has_traces": True,
        "metric_entity_count": int(np.count_nonzero(metric_arr)),
        "log_entity_count": int(np.count_nonzero(log_arr)),
        "trace_span_count": int(graph_mat.sum()),
        "error_keyword_count": sum(d.get("count", 0) for d in log_detail.values()),
    }
    anomaly_times = {a.get("component", ""): float(a.get("timestamp", inject_time))
                     for a in anchor_set}

    controller = PRISMLAController(config=config)
    pred = controller.run(
        query=query, telemetry=telemetry, entities=entities,
        entity_types=entity_types, topology=topology_edges,
        metric_signal=metric_arr, log_signal=log_arr, graph=graph_mat,
        metric_detail=metric_detail, log_detail=log_detail, trace_detail={},
        anomaly_times=anomaly_times, anchor_set=anchor_set,
        candidate_scores=candidate_scores, cmi_profiles={}, cf_profile_cache={},
        telemetry_meta=telemetry_meta,
    )

    # Post-hoc reason normalization to Bank GT vocabulary
    if entity_reason_hypothesis:
        events = pred.get("root_cause_events", [])
        for ev in events:
            comp = ev.get("root cause component", "")
            if comp in entity_reason_hypothesis:
                ev["root cause reason"] = entity_reason_hypothesis[comp]
        if pred.get("reason"):
            pred["reason"] = [
                entity_reason_hypothesis.get(c, r)
                for c, r in zip(pred.get("component", []), pred.get("reason", []))
            ]
    return pred


def main() -> None:
    args = parse_args()
    logger.info("PRISM-LA v3: model=%s steps=%d events=%d temp=%.2f",
                args.model, args.max_steps, args.max_events, args.temperature)

    config = PRISMLAConfig(
        max_investigation_steps=args.max_steps,
        max_event_hypotheses=args.max_events,
        evidence_mode="structured",
        llm_model=args.model,
        llm_temperature=args.temperature,
        llm_api_key=DEEPSEEK_API_KEY,
        llm_base_url=DEEPSEEK_BASE_URL,
        log_llm_messages=True,
        debug_dir=RESULT_DIR,
    )
    os.environ["OPENAI_API_KEY"] = DEEPSEEK_API_KEY
    os.environ["OPENAI_BASE_URL"] = DEEPSEEK_BASE_URL
    os.environ["PRISM_STRICT_NO_LEAKAGE"] = "1"
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    loader = OpenRCALoader("Bank")
    all_queries = loader.load_queries("")
    logger.info("Loaded %d Bank queries", len(all_queries))

    queries_with_gt: List[Tuple[Any, Any]] = []
    for query in all_queries:
        try:
            gt_records, inject_time = loader.match_query_to_records(query)
        except Exception:
            gt_records, inject_time = [], None
        if gt_records and inject_time is not None:
            query_clone = QueryCase(
                task_index=str(getattr(query, "task_index", "")),
                system="Bank", sub_system="",
                instruction=str(getattr(query, "instruction", "")),
                time_window=tuple(getattr(query, "time_window", ("", ""))),
                scoring_points=[], inject_time=float(inject_time),
                telemetry_date=getattr(query, "telemetry_date", None),
                ground_truth=None,
            )
            queries_with_gt.append((query_clone, query))
    logger.info("Queries with GT: %d", len(queries_with_gt))

    if args.dry_run:
        queries_with_gt = queries_with_gt[:3]
        logger.info("DRY RUN: %d queries", len(queries_with_gt))
    elif args.max_queries > 0:
        queries_with_gt = queries_with_gt[:args.max_queries]

    by_date: Dict[str, List[Tuple[Any, Any]]] = defaultdict(list)
    for inf_q, eval_q in queries_with_gt:
        try:
            ds = loader.resolve_telemetry_date(inf_q)
        except Exception:
            ds = "unknown"
        by_date[ds or "unknown"].append((inf_q, eval_q))
    logger.info("Date groups: %d", len(by_date))

    aggregator = EvalAggregator()
    results: List[Dict[str, Any]] = []
    total = len(queries_with_gt)
    correct = partial = wrong = 0
    t0 = time.time()

    for date_i, (date_str, dqs) in enumerate(sorted(by_date.items())):
        logger.info("Loading telemetry for %s...", date_str)
        t_load = time.time()
        try:
            telemetry = loader.load_telemetry(date_str, "")
        except Exception as exc:
            logger.error("Telemetry load failed for %s: %s", date_str, exc)
            results.extend({"error": str(exc), "query_id": build_query_id(inf_q)}
                           for inf_q, _ in dqs)
            continue
        n_entities = len(telemetry.entities) if hasattr(telemetry, "entities") else 0
        logger.info("Loaded in %.0fs: %d entities", time.time() - t_load, n_entities)

        for qi, (inf_q, eval_q) in enumerate(dqs):
            qid = build_query_id(inf_q)
            logger.info("[%d/%d][date %d/%d] %s [%s]",
                        qi + 1, total, date_i + 1, len(by_date), qid,
                        getattr(eval_q, "task_index", ""))

            t_start = time.time()
            try:
                pred = process_query(inf_q, telemetry, config)
            except Exception as exc:
                logger.error("Crash for %s: %s", qid, exc)
                pred = {"component": [], "reason": [], "time": [], "top_score": 0.0,
                        "fault_count": 0, "root_cause_events": [], "error": str(exc)}
            elapsed = time.time() - t_start

            eval_res = evaluate_prediction(pred, eval_q)
            aggregator.add(eval_res)

            if eval_res.correct:
                correct += 1; status = "CORRECT"
            elif eval_res.partial:
                partial += 1; status = "PARTIAL"
            else:
                wrong += 1; status = "WRONG"

            record = {
                "query_id": qid, "task_index": getattr(eval_q, "task_index", ""),
                "date": date_str, "correct": eval_res.correct, "partial": eval_res.partial,
                "official_score": round(float(eval_res.official_score), 4),
                "field_scores": dict(eval_res.field_scores),
                "prediction": {
                    "component": pred.get("component", []),
                    "reason": pred.get("reason", []),
                    "time": pred.get("time", []),
                    "top_score": pred.get("top_score", 0.0),
                },
                "ground_truth": {
                    "component": getattr(eval_q.ground_truth, "component", ""),
                    "reason": getattr(eval_q.ground_truth, "reason", ""),
                    "datetime": getattr(eval_q.ground_truth, "datetime_str", ""),
                },
                "elapsed_seconds": round(elapsed, 1),
            }
            results.append(record)
            logger.info("  -> %s score=%.2f time=%.0fs (c=%d p=%d w=%d/%d)",
                        status, eval_res.official_score, elapsed,
                        correct, partial, wrong, total)

            ckpt = RESULT_DIR / "checkpoint_bank_136q.json"
            ckpt.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")

    elapsed_total = time.time() - t0
    summary = aggregator.summary()
    summary["experiment"] = {
        "framework": "PRISM-LA v3", "model": args.model,
        "max_steps": args.max_steps, "max_events": args.max_events,
        "temperature": args.temperature, "total_queries": total,
        "total_seconds": round(elapsed_total, 1),
    }
    summary["results"] = results

    out = RESULT_DIR / "prism_la_bank_136q_results.json"
    out.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print()
    print("=" * 60)
    print("  PRISM-LA Bank 136q Results")
    print("=" * 60)
    print(f"  Total:     {total}")
    acc = 100 * correct / max(1, total)
    part = 100 * partial / max(1, total)
    print(f"  Correct:   {correct} ({acc:.1f}%)")
    print(f"  Partial:   {partial} ({part:.1f}%)")
    print(f"  Wrong:     {wrong}")
    print(f"  Offscore:  {summary.get('official_mean_score', 0):.4f}")
    print(f"  Time:      {elapsed_total:.0f}s ({elapsed_total/60:.1f}min)")
    if total:
        print(f"  Avg/query: {elapsed_total/total:.1f}s")
    print(f"  Saved:     {out}")
    print("=" * 60)


if __name__ == "__main__":
    main()
