"""Strict no-leakage entry point for PRISM-LA.

Usage:
    python -m prism_la.strict_main --systems Bank --prism-llm-agent \\
        --llm-agent-max-steps 6 --llm-agent-max-events 8 \\
        --llm-agent-evidence-mode structured

This entry point:
1. Validates all CLI arguments for leakage
2. Rejects convenience flags and un-certified artifacts
3. Loads telemetry via existing PRISM v3 loaders
4. Runs the PRISM-LA controller
5. Outputs OpenRCA-format predictions
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("prism_la.strict_main")

FORBIDDEN_FLAGS = {
    "--v2-all",
    "--v2-learned",
    "--prism-noise-lab-scores",
    "--v2-learned-embedding-path",
    "--v2-learned-likelihood-path",
    "--v2-learned-classifier-path",
    "--v2-entity-profile-path",
    "--entity-profile",
    "--research-mode",
    "--research-mode-artifacts",
}


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PRISM-LA: LLM-as-Investigator Root Cause Agent",
    )
    parser.add_argument(
        "--systems",
        nargs="+",
        default=["Bank"],
        choices=["Bank", "Telecom", "Market"],
        help="Target benchmark systems",
    )
    parser.add_argument(
        "--prism-llm-agent",
        action="store_true",
        default=True,
        help="Enable PRISM-LA agent mode (default for this entry point)",
    )
    parser.add_argument(
        "--llm-agent-max-steps",
        type=int,
        default=6,
        help="Maximum investigation steps per case",
    )
    parser.add_argument(
        "--llm-agent-max-events",
        type=int,
        default=8,
        help="Maximum hypothesis events per case",
    )
    parser.add_argument(
        "--llm-agent-evidence-mode",
        default="structured",
        choices=["structured"],
        help="Evidence delivery mode (only structured supported)",
    )
    parser.add_argument(
        "--llm-model",
        default=os.environ.get("PRISM_LLM_MODEL", "gpt-4o"),
        help="LLM model name",
    )
    parser.add_argument(
        "--llm-temperature",
        type=float,
        default=0.15,
        help="LLM sampling temperature",
    )
    parser.add_argument(
        "--output-dir",
        default="results/prism_la",
        help="Output directory for results",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=0,
        help="Maximum queries to process (0 = all)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose logging",
    )
    return parser


def check_forbidden_flags(argv: Sequence[str]) -> List[str]:
    found = []
    for flag in FORBIDDEN_FLAGS:
        if flag in argv:
            found.append(flag)
    return found


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    forbidden = check_forbidden_flags(sys.argv)
    if forbidden:
        logger.error(
            "PRISM-LA strict mode rejects the following flags: %s",
            ", ".join(forbidden),
        )
        sys.exit(1)

    os.environ["PRISM_STRICT_NO_LEAKAGE"] = "1"
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info("PRISM-LA starting: systems=%s, max_steps=%d, model=%s",
                args.systems, args.llm_agent_max_steps, args.llm_model)

    from prism_v3 import data
    from prism_v3 import config as v3_config
    from prism_v3.leakage_guard import assert_inference_query_safe, build_query_id
    from prism_v3.time_anchor import build_anchor_set
    from prism_v3.noise_native.noiselab_adapter import NoiseLabEvidenceAdapter
    from prism_v3.noise_native.evidence_frame import CandidateFrame, EvidenceFrame

    from prism_la.config import PRISMLAConfig
    from prism_la.controller import PRISMLAController

    la_config = PRISMLAConfig(
        max_investigation_steps=args.llm_agent_max_steps,
        max_event_hypotheses=args.llm_agent_max_events,
        evidence_mode=args.llm_agent_evidence_mode,
        llm_model=args.llm_model,
        llm_temperature=args.llm_temperature,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []
    processed = 0

    for system_name in args.systems:
        logger.info("Loading system: %s", system_name)
        system_cfg = v3_config.SYSTEM_PATHS.get(system_name)
        if not system_cfg:
            logger.error("Unknown system: %s", system_name)
            continue

        try:
            loader = data.loader.DataLoader(system_name)
            queries = loader.load_queries()
        except Exception as exc:
            logger.error("Failed to load queries for %s: %s", system_name, exc)
            continue

        for query in queries:
            if args.max_queries > 0 and processed >= args.max_queries:
                break

            query_id = build_query_id(query)
            logger.info("Processing query: %s (%d/%d)", query_id, processed + 1,
                         min(len(queries), args.max_queries) if args.max_queries else len(queries))

            try:
                assert_inference_query_safe(query)
            except Exception as exc:
                logger.error("Unsafe query %s: %s", query_id, exc)
                continue

            try:
                telemetry = loader.load_telemetry(query)
            except Exception as exc:
                logger.error("Failed to load telemetry for %s: %s", query_id, exc)
                continue

            entities = telemetry.entities or []
            entity_types = telemetry.entity_types or {}

            import numpy as np
            if hasattr(telemetry, 'metrics') and telemetry.metrics is not None:
                metric_detail = _build_metric_detail(telemetry.metrics, entities)
            else:
                metric_detail = {}
            if hasattr(telemetry, 'logs') and telemetry.logs is not None:
                log_detail = _build_log_detail(telemetry.logs, entities)
            else:
                log_detail = {}
            trace_detail: Dict[str, Dict[str, Any]] = {}

            inject_time = getattr(query, "inject_time", None) or 0.0
            if hasattr(telemetry, 'metrics') and telemetry.metrics is not None:
                anchors_obj = build_anchor_set(telemetry, query, top_k=la_config.top_anchor_count)
                anchor_set = [
                    {"timestamp": a.timestamp, "confidence": a.confidence, "source": str(a.source)}
                    for a in anchors_obj.anchors
                ]
                anomaly_times = {
                    str(a.get("component", "")): float(a.get("timestamp", inject_time))
                    for a in anchor_set
                }
            else:
                anchor_set = [{"timestamp": inject_time, "confidence": 1.0, "source": "query_window"}]
                anomaly_times = {}

            try:
                adapter = NoiseLabEvidenceAdapter(strategy="base", temperature=1.0)
                evidence_frame = adapter.from_runtime_scorer(
                    telemetry, query, inject_time, entities
                )
            except Exception as exc:
                logger.warning("NoiseLab adapter failed: %s, using empty frame", exc)
                evidence_frame = EvidenceFrame(
                    source="none",
                    applied=False,
                    candidates=[],
                    reason=str(exc),
                )

            candidate_scores = _candidates_to_scores(evidence_frame.candidates)
            cmi_profiles: Dict[str, Dict[str, Any]] = {}
            cf_profile_cache: Dict[str, Dict[str, Any]] = {}

            n = len(entities)
            metric_arr = np.zeros(n, dtype=float)
            log_arr = np.zeros(n, dtype=float)
            graph_arr = np.eye(n, dtype=float)

            for candidate in evidence_frame.candidates:
                cid = candidate.canonical_component
                for idx, entity in enumerate(entities):
                    from prism_v3.noise_native.evidence_frame import canonical_entity_name
                    if canonical_entity_name(entity) == cid:
                        metric_arr[idx] = max(metric_arr[idx],
                                              float(candidate.metric_evidence.get("score", 0.0) or 0.0))
                        log_arr[idx] = max(log_arr[idx],
                                           float(candidate.log_evidence.get("score", 0.0) or 0.0))
                        break

            if hasattr(telemetry, 'traces') and telemetry.traces is not None:
                graph_arr = _build_trace_graph(telemetry.traces, entities)
            elif evidence_frame.candidates:
                for i in range(n):
                    for j in range(n):
                        if i != j:
                            graph_arr[i, j] = max(graph_arr[i, j], 0.05)

            topology_edges: List[Tuple[str, str, float]] = []
            for i in range(n):
                for j in range(n):
                    if i != j and graph_arr[i, j] > 1e-6:
                        topology_edges.append((
                            str(entities[i]), str(entities[j]), float(graph_arr[i, j])
                        ))

            telemetry_meta = {
                "has_metrics": hasattr(telemetry, 'metrics') and telemetry.metrics is not None,
                "has_logs": hasattr(telemetry, 'logs') and telemetry.logs is not None,
                "has_traces": hasattr(telemetry, 'traces') and telemetry.traces is not None,
                "metric_entity_count": int(np.count_nonzero(metric_arr)),
                "log_entity_count": int(np.count_nonzero(log_arr)),
                "trace_span_count": int(graph_arr.sum()) if graph_arr.size else 0,
                "trace_entity_count": sum(1 for e in entities if e in trace_detail),
                "entity_list": [str(e) for e in entities][:30],
            }

            controller = PRISMLAController(config=la_config)
            try:
                prediction = controller.run(
                    query=query,
                    telemetry=telemetry,
                    entities=entities,
                    entity_types=entity_types,
                    topology=topology_edges,
                    metric_signal=metric_arr,
                    log_signal=log_arr,
                    graph=graph_arr,
                    metric_detail=metric_detail,
                    log_detail=log_detail,
                    trace_detail=trace_detail,
                    anomaly_times=anomaly_times,
                    anchor_set=anchor_set,
                    candidate_scores=candidate_scores,
                    cmi_profiles=cmi_profiles,
                    cf_profile_cache=cf_profile_cache,
                    telemetry_meta=telemetry_meta,
                )
            except Exception as exc:
                logger.error("Controller failed for %s: %s", query_id, exc)
                prediction = {
                    "component": [], "reason": [], "time": [],
                    "top_score": 0.0, "fault_count": 0, "root_cause_events": [],
                    "error": str(exc),
                }

            prediction["query_id"] = query_id
            prediction["system"] = system_name
            prediction["task_index"] = getattr(query, "task_index", "")

            results.append(prediction)
            processed += 1
            logger.info("Query %s prediction: %s", query_id,
                         json.dumps(prediction, indent=2, default=str)[:500])

    out_path = output_dir / "prism_la_predictions.json"
    out_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    logger.info("PRISM-LA finished: %d queries, results in %s", processed, out_path)
    print(f"Results written to {out_path}")


def _build_metric_detail(metrics: Any, entities: Sequence[str]) -> Dict[str, Dict[str, float]]:
    detail: Dict[str, Dict[str, float]] = {}
    if metrics is None:
        return detail
    import pandas as pd
    df = metrics if isinstance(metrics, pd.DataFrame) else pd.DataFrame()
    if df.empty:
        return detail
    entity_col = next((c for c in ("entity", "cmdb_id", "serviceName") if c in df.columns), None)
    metric_col = next((c for c in ("metric_name", "kpi_name") if c in df.columns), None)
    value_col = "value"
    if not entity_col or not metric_col:
        return detail
    entity_set = set(entities)
    for _, row in df.iterrows():
        e = str(row.get(entity_col, ""))
        if e not in entity_set:
            continue
        m = str(row.get(metric_col, ""))
        try:
            v = float(row.get(value_col, 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        inner = detail.setdefault(e, {})
        inner[m] = inner.get(m, 0.0) + v
    return detail


def _build_log_detail(logs: Any, entities: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    detail: Dict[str, Dict[str, Any]] = {}
    if logs is None:
        return detail
    import pandas as pd
    df = logs if isinstance(logs, pd.DataFrame) else pd.DataFrame()
    if df.empty:
        return detail
    entity_col = next((c for c in ("entity", "cmdb_id", "serviceName") if c in df.columns), None)
    msg_col = next((c for c in ("message", "log") if c in df.columns), None)
    if not entity_col:
        return detail
    entity_set = set(entities)
    fatal_kw = ["oom", "killed", "exception", "timeout", "refused", "error", "failed", "connection"]
    for e in df.groupby(entity_col):
        e_name = str(e[0])
        if e_name not in entity_set:
            continue
        group = e[1]
        msgs = []
        error_count = 0
        if msg_col:
            for _, row in group.iterrows():
                msg = str(row.get(msg_col, "") or "")
                msgs.append(msg[:200])
                for kw in fatal_kw:
                    if kw in msg.lower():
                        error_count += 1
                        break
        detail[e_name] = {
            "count": error_count,
            "total": len(group),
            "text": "\n".join(msgs[:5]),
            "fatal_boost": max(1.0, min(3.0, 1.0 + error_count * 0.5)),
        }
    return detail


def _build_trace_graph(traces: Any, entities: Sequence[str]) -> "np.ndarray":  # type: ignore
    import numpy as np
    n = len(entities)
    graph = np.eye(n, dtype=float)
    if traces is None:
        return graph
    import pandas as pd
    df = traces if isinstance(traces, pd.DataFrame) else pd.DataFrame()
    if df.empty:
        return graph
    entity_col = next((c for c in ("entity", "cmdb_id", "serviceName") if c in df.columns), None)
    parent_col = next((c for c in ("parent_entity", "parent_id") if c in df.columns), None)
    if not entity_col or not parent_col:
        return graph
    entity_idx = {e: i for i, e in enumerate(entities)}
    for _, row in df.iterrows():
        child = str(row.get(entity_col, ""))
        parent = str(row.get(parent_col, ""))
        if child in entity_idx and parent in entity_idx:
            i = entity_idx[parent]
            j = entity_idx[child]
            graph[i, j] += 1.0
    row_sums = graph.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        graph = graph / np.maximum(row_sums, 1e-9)
    return graph


def _candidates_to_scores(candidates: Sequence[Any]) -> Dict[str, Dict[str, float]]:
    scores: Dict[str, Dict[str, float]] = {}
    for c in candidates:
        cid = getattr(c, "canonical_component", str(c))
        if hasattr(c, "canonical_component"):
            cid = c.canonical_component
        elif hasattr(c, "component_id"):
            cid = c.component_id
        scores[cid] = {
            "root_score": float(getattr(c, "noise_score", 0.0) or 0.0),
            "downstream_recovery": 0.0,
            "symptom_score": float(getattr(c, "symptomness", 0.0) or 0.0),
            "broad_explainer": 0.0,
            "source_isolation": float(getattr(c, "source_likelihood", 0.0) or 0.0),
            "hotspot_symptom": float(getattr(c, "symptomness", 0.0) or 0.0),
        }
    return scores


if __name__ == "__main__":
    main()
