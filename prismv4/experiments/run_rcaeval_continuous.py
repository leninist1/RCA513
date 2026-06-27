#!/usr/bin/env python3
"""Run a continuous NoiseNative RCA agent on RCAEval cases.

This experiment intentionally avoids the Lead/Challenger stepwise tournament.
The LLM maintains one case-level working memory, updates belief after each
NoiseLab fact extraction, and gives a final decision when the action budget is
exhausted.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

from prismv4.experiments.rcaeval_adapter import (
    discover_aiops2021_cases,
    discover_eadro_cases,
    discover_openrca_cases,
    discover_re3_cases,
    load_aiops2021_case,
    load_eadro_case,
    load_openrca_case,
    load_re3_case,
)
from prismv4.prism_cht.canonical import build_tool_call_signature, canonicalize_json_value
from prismv4.prism_cht.diagnostic_policy import build_hypotheses_from_case
from prismv4.prism_cht.evidence_graph import EvidenceAtom, EvidenceGraph
from prismv4.prism_cht.http_transport import UrllibHttpTransport
from prismv4.prism_cht.llm_audit import AuditedModelClient, CostWindow, LLMIORecorder
from prismv4.prism_cht.llm_json import parse_json_object
from prismv4.prism_cht.llm_types import StructuredOutputError
from prismv4.prism_cht.llm_types import (
    ModelClient,
    ModelMessage,
    ModelRequest,
    ModelResponse,
)
from prismv4.prism_cht.noiselab_registry import (
    NOISELAB_TOOL_NAMES,
    build_noiselab_tool_registry,
)
from prismv4.prism_cht.openai_compatible_client import OpenAICompatibleChatModelClient
from prismv4.prism_cht.provider_config import load_openai_compatible_config_from_mapping


DEFAULT_RE3_ROOT = "/home/dell2/RCA513/ysj/dataset/RCAEval/RE3"

_INSTANCE_SUFFIX_RE = re.compile(r"-\d+$")


def _component_match(predicted: str | None, expected: str | None) -> bool:
    """Check if predicted component matches expected.

    OpenRCA ground-truth labels include instance suffixes (e.g. ``shippingservice-1``,
    ``node-4``) that do not exist as separate entities in the telemetry metrics. When
    the recall pool / IVD only sees the base name (``shippingservice``, ``node``), we
    match on the stripped base name so that ``predicted=node`` correctly hits
    ``expected=node-1``. This does NOT affect ``adservice2`` (no hyphen) or other
    service-renamed instances.
    """
    if predicted is None or expected is None:
        return False
    if predicted == expected:
        return True
    exp_base = _INSTANCE_SUFFIX_RE.sub("", expected)
    pred_base = _INSTANCE_SUFFIX_RE.sub("", predicted)
    return exp_base == pred_base and exp_base != expected


class RetryingModelClient:
    """Retry transient provider failures while preserving audited attempts."""

    def __init__(
        self,
        *,
        inner: ModelClient,
        max_attempts: int = 3,
        initial_delay_seconds: float = 2.0,
    ) -> None:
        self._inner = inner
        self._max_attempts = max(1, max_attempts)
        self._initial_delay_seconds = initial_delay_seconds

    def complete(self, *, request: ModelRequest) -> ModelResponse:
        last_exc: Exception | None = None
        for attempt in range(self._max_attempts):
            try:
                return self._inner.complete(request=request)
            except Exception as exc:
                last_exc = exc
                if attempt + 1 >= self._max_attempts:
                    break
                time.sleep(self._initial_delay_seconds * (2**attempt))
        assert last_exc is not None
        raise last_exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Continuous NoiseNative RCAEval runner")
    parser.add_argument("--data-root", default=DEFAULT_RE3_ROOT)
    parser.add_argument("--dataset", default="RE3", choices=["RE3", "Eadro", "OpenRCA", "AIOps2021"])
    parser.add_argument("--system", default="RE3-OB")
    parser.add_argument("--max-cases", type=int, default=5)
    parser.add_argument("--max-hypotheses", type=int, default=10)
    parser.add_argument(
        "--recall-pool-size",
        type=int,
        default=int(os.environ.get("PRISM_CHT_RECALL_POOL_SIZE", "15")),
        help="Width of the tiered recall pool that feeds EventCausalizer and "
        "hypothesis seeding.  Decoupled from --max-hypotheses (final reasoning "
        "width).  Offline validation showed 15 saturates recall on RE3-TT.",
    )
    parser.add_argument("--max-steps", type=int, default=4)
    parser.add_argument(
        "--mode",
        default="full",
        choices=["full", "lightweight"],
        help="'full' = original multi-step agent. 'lightweight' = signal-first: "
        "extract deterministic features + IVD, skip LLM when IVD consensus is "
        "strong, otherwise make one compact LLM call with <3k token summary.",
    )
    parser.add_argument(
        "--output",
        default="prismv4/results/prism_cht/rcaeval_continuous_results.json",
    )
    parser.add_argument(
        "--llm-io-output",
        default="",
        help="JSONL path for full-fidelity LLM request/response records.",
    )
    parser.add_argument(
        "--only-cases",
        default="",
        help="Comma-separated case name fragments to run (e.g. 'front-end_f1/2,emailservice_f1/1'). Empty = run all.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.dataset == "Eadro":
        eadro_root = str(
            Path(__file__).resolve().parent.parent.parent
            / "../syh/datasets/Eadro/.adapter_work_v22"
        )
        eadro_root = str(Path(eadro_root).resolve())
        if args.data_root == DEFAULT_RE3_ROOT:
            args.data_root = eadro_root
        case_specs = discover_eadro_cases(
            args.data_root,
            system=args.system,
            limit=args.max_cases if args.max_cases > 0 else None,
        )
        if args.only_cases:
            fragments = [f.strip() for f in args.only_cases.split(",") if f.strip()]

            def _eadro_case_key(cs):
                from pathlib import Path
                bn = Path(str(cs["case_dir"])).name
                return f"{bn}/fault{cs['fault_index']}"

            case_specs = [
                cs for cs in case_specs
                if any(frag in str(cs["case_dir"])
                       or frag in cs["fault"]["name"]
                       or frag == _eadro_case_key(cs)
                       or frag in _eadro_case_key(cs)
                       for frag in fragments)
            ]
        if not case_specs:
            raise SystemExit("no Eadro cases discovered")
    elif args.dataset == "OpenRCA":
        if args.max_cases > 0:
            case_specs = discover_openrca_cases(
                None, system=args.system, limit=args.max_cases
            )
        else:
            case_specs = discover_openrca_cases(None, system=args.system)
        if args.only_cases:
            fragments = [f.strip() for f in args.only_cases.split(",") if f.strip()]
            case_specs = [
                cs for cs in case_specs
                if any(frag in str(cs.get("case_id", "")) or frag in str(cs.get("telemetry_date", "")) for frag in fragments)
            ]
        if not case_specs:
            raise SystemExit("no OpenRCA cases discovered")
    elif args.dataset == "AIOps2021":
        # AIOps2021 only has test split (47 cases). --system is ignored; --max-cases=0 = all.
        case_specs = discover_aiops2021_cases(
            split=getattr(args, "split", "test"),
            limit=args.max_cases if args.max_cases > 0 else None,
        )
        if args.only_cases:
            fragments = [f.strip() for f in args.only_cases.split(",") if f.strip()]
            case_specs = [
                cs for cs in case_specs
                if any(frag in str(cs.get("case_id", "")) or frag in str(cs.get("groundtruth_id", "")) for frag in fragments)
            ]
        if not case_specs:
            raise SystemExit("no AIOps2021 cases discovered")
    else:
        case_dirs = discover_re3_cases(
            args.data_root,
            system=args.system,
            limit=args.max_cases if args.max_cases > 0 else None,
        )
        if args.only_cases:
            fragments = [f.strip() for f in args.only_cases.split(",") if f.strip()]
            case_dirs = [
                cd for cd in case_dirs
                if any(frag in str(cd) for frag in fragments)
            ]
        if not case_dirs:
            raise SystemExit("no RCAEval cases discovered")

    config = load_openai_compatible_config_from_mapping(dict(os.environ))
    provider_client = OpenAICompatibleChatModelClient(
        config=config,
        transport=UrllibHttpTransport(),
    )
    llm_io_output = (
        Path(args.llm_io_output)
        if args.llm_io_output
        else Path(str(args.output) + ".llm_io.jsonl")
    )
    cost_window = CostWindow()
    audited_client = AuditedModelClient(
        inner=provider_client,
        recorder=LLMIORecorder(jsonl_path=llm_io_output),
        cost_window=cost_window,
    )
    client = RetryingModelClient(
        inner=audited_client,
        max_attempts=int(os.environ.get("PRISM_CHT_MODEL_RETRIES", "3")),
    )

    results: list[dict[str, Any]] = []
    started_all = time.time()

    if args.dataset in ("Eadro", "OpenRCA", "AIOps2021"):
        case_iter = case_specs
    else:
        case_iter = case_dirs

    for case_item in case_iter:
        started = time.time()
        cost_before = cost_window.snapshot()
        if args.dataset == "Eadro":
            loaded = load_eadro_case(
                case_item, top_k=args.recall_pool_size, eadro_root=args.data_root
            )
        elif args.dataset == "OpenRCA":
            loaded = load_openrca_case(case_item, top_k=args.recall_pool_size)
        elif args.dataset == "AIOps2021":
            loaded = load_aiops2021_case(case_item, top_k=args.recall_pool_size)
        else:
            loaded = load_re3_case(case_item, top_k=args.recall_pool_size)
        try:
            if args.mode == "lightweight":
                result = run_case_lightweight(
                    loaded=loaded,
                    client=client,
                    recall_pool_size=args.recall_pool_size,
                )
            else:
                result = run_case(
                    loaded=loaded,
                    max_hypotheses=args.max_hypotheses,
                    max_steps=args.max_steps,
                    client=client,
                    recall_pool_size=args.recall_pool_size,
                )
            predicted = result.get("predicted_component")
            hit = _component_match(predicted, loaded.expected_component)
            case_cost = _case_cost_delta(cost_before, cost_window.snapshot())
            result.update(
                {
                    "case_id": loaded.case.case_id,
                    "expected_component": loaded.expected_component,
                    "hit": hit,
                    "elapsed_sec": round(time.time() - started, 3),
                    "error": None,
                    "token_usage": case_cost,
                }
            )
        except Exception as exc:
            case_cost = _case_cost_delta(cost_before, cost_window.snapshot())
            result = {
                "case_id": loaded.case.case_id,
                "expected_component": loaded.expected_component,
                "predicted_component": None,
                "hit": False,
                "elapsed_sec": round(time.time() - started, 3),
                "error": f"{type(exc).__name__}: {exc}",
                "token_usage": case_cost,
            }
        results.append(result)
        print(
            f"{loaded.case.case_id}: predicted={result.get('predicted_component')} "
            f"expected={loaded.expected_component} hit={result['hit']} "
            f"tokens={case_cost.get('total_tokens', 0)} "
            f"error={result.get('error')}",
            flush=True,
        )

    total = len(results)
    hits = sum(1 for item in results if item.get("hit"))
    cost_summary = cost_window.summary()
    avg_tokens = cost_summary["total_tokens"] // total if total else 0
    summary = {
        "dataset": args.dataset,
        "system": args.system,
        "mode": ("lightweight" if args.mode == "lightweight" else "continuous-noise-native-event-causalizer"),
        "total_cases": total,
        "top1_accuracy": hits / total if total else 0.0,
        "top1_hits": hits,
        "recall_pool_size": args.recall_pool_size,
        "max_hypotheses": args.max_hypotheses,
        "elapsed_sec": round(time.time() - started_all, 3),
        "cost": {
            "total_calls": cost_summary["call_count"],
            "total_prompt_tokens": cost_summary["prompt_tokens"],
            "total_completion_tokens": cost_summary["completion_tokens"],
            "total_tokens": cost_summary["total_tokens"],
            "avg_tokens_per_case": avg_tokens,
            "by_purpose": cost_summary["by_purpose"],
        },
        "results": results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, indent=2), flush=True)
    print(f"wrote {output}", flush=True)
    return 0


def _case_cost_delta(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> dict[str, Any]:
    """Compute per-case token usage as the delta between two snapshots."""
    before_by = before.get("by_purpose", {}) or {}
    after_by = after.get("by_purpose", {}) or {}
    delta_by_purpose: dict[str, dict[str, int]] = {}
    for purpose, stats in after_by.items():
        prev = before_by.get(purpose, {})
        delta_by_purpose[purpose] = {
            "call_count": int(stats.get("call_count", 0)) - int(prev.get("call_count", 0)),
            "prompt_tokens": int(stats.get("prompt_tokens", 0)) - int(prev.get("prompt_tokens", 0)),
            "completion_tokens": int(stats.get("completion_tokens", 0)) - int(prev.get("completion_tokens", 0)),
            "total_tokens": int(stats.get("total_tokens", 0)) - int(prev.get("total_tokens", 0)),
        }
    return {
        "call_count": int(after.get("call_count", 0)) - int(before.get("call_count", 0)),
        "prompt_tokens": int(after.get("prompt_tokens", 0)) - int(before.get("prompt_tokens", 0)),
        "completion_tokens": int(after.get("completion_tokens", 0)) - int(before.get("completion_tokens", 0)),
        "total_tokens": int(after.get("total_tokens", 0)) - int(before.get("total_tokens", 0)),
        "by_purpose": delta_by_purpose,
    }


# ---------------------------------------------------------------------------
# Lightweight signal-first agent (low token cost)
# ---------------------------------------------------------------------------

_LIGHTWEIGHT_SYSTEM_PROMPT = """You are a root cause analyst for microservice incidents.

You receive a compact JSON summary of pre-extracted deterministic signals: per-component anomaly features, IVD (Initiator-Victim Disambiguation) verdicts, and recall pool ranking. You do NOT receive raw metrics, logs, or traces.

Rules:
- If IVD provides an initiator with copeland_score > 0, select it.
- Source_candidate with emitter_exception=True outranks source_candidate without, UNLESS that candidate has near-zero anomaly magnitude (``mag`` <= 0.5) AND another candidate has mag >= 2x larger resource anomaly — in that case the emitter may be a reactive caller (``Failed to call X`` is a symptom, not the root cause).
- Storage components (mongo/redis/mysql/db suffix) with resource anomalies but no emitter exceptions are reactive dependencies, NOT the root cause.
- Entry-like components (frontend/gateway/ingress) are traffic surfaces, not root causes.
- Pick exactly one root_component from the candidates list, and also provide a top-5 ranking (best first) of all candidates.
- Output one valid JSON object: {"root_component": "...", "ranking": ["...", "...", ...], "rationale": "..."}"""


def _extract_compact_signals(
    loaded,
    *,
    recall_pool_size: int = 15,
) -> dict[str, Any]:
    """Extract deterministic signals into a compact dict (~1-2k tokens)."""
    store = loaded.store
    case = loaded.case
    recall_pool = _build_recall_pool(loaded, pool_size=recall_pool_size)
    time_window = (
        max(0.0, case.event_time - 30.0),
        case.event_time + 600.0,
    )
    facts = store.build_event_causal_observations(
        component_scope=list(recall_pool),
        time_window=time_window,
        max_events=32,
    )
    ivd_raw = facts.get("ivd")
    ivd = _compact_ivd(ivd_raw) if ivd_raw else None

    # Compact per-component features
    comp_features = []
    for row in facts.get("component_feature_rows", []):
        comp_features.append({
            "c": row["component"],
            "role": row.get("causal_role", "?"),
            "sl": round(float(row.get("source_likelihood_score", 0)), 2),
            "mag": round(float(row.get("near_onset_anomaly_magnitude", 0)), 1),
            "sig": list(row.get("near_onset_internal_signals", [])),
            "log_cnt": int(row.get("log_count", 0)),
            "emit": bool(row.get("has_emitter_exception", False)),
            "entry": bool(row.get("is_entry_like_component", False)),
            "storage": bool(row.get("is_storage_component", False)),
        })

    # Compact IVD verdicts
    ivd_compact = None
    if ivd and ivd.get("ivd_enabled"):
        ivd_compact = {
            "ivd_enabled": True,
            "ranking": ivd.get("ranking", []),
            "verdicts": {},
        }
        for comp, v in ivd.get("verdicts", {}).items():
            ivd_compact["verdicts"][comp] = {
                "role": v.get("ivd_role"),
                "cs": v.get("copeland_score"),
                "fs": v.get("fault_signature_label"),
                "cov": v.get("downstream_coverage_count"),
                "call": v.get("caller_anomalies_count"),
                "res": round(float(v.get("resource_magnitude", 0) or 0), 1),
                "wl": round(float(v.get("workload_magnitude", 0) or 0), 1),
            }

    return {
        "recall_pool": list(recall_pool),
        "ivd": ivd_compact,
        "components": comp_features,
        "observations": [
            {"c": o.component, "fs": round(o.first_seen, 1), "sig": list(o.signals)}
            for o in case.observations[:10]
        ],
        "entry_components": list(case.entry_components),
    }


def _ivd_has_strong_consensus(
    ivd: dict[str, Any] | None,
    *,
    comp_features: list[dict[str, Any]] | None = None,
) -> tuple[bool, str | None]:
    """Check if IVD has a clear winner we can use without LLM.

    An ``emitter`` component flagged by IVD as initiator but whose anomaly
    magnitude (``res``) is near-zero — typically a downstream service that
    merely emitted ``Failed to call X`` error logs — should NOT qualify for
    the shortcut. In such cases the true root-cause (which has resource/metric
    anomalies but no error-stack) is better handled by the LLM path that can
    see the full component-feature summary. This is the Eadro-TT failure
    pattern: fault service emits no error logs, downstream ``Failed to call``
    get misclassified as emitter/initiator.
    """
    if not ivd or not ivd.get("ivd_enabled"):
        return False, None
    ranking = ivd.get("ranking", [])
    if not ranking:
        return False, None
    verdicts = ivd.get("verdicts", {})
    if not verdicts:
        return False, None
    top = ranking[0]
    v = verdicts.get(top)
    if not v:
        return False, None
    # Strong consensus: initiator with copeland score >= 3 and no tie
    if (v.get("role") == "initiator" and
            (v.get("cs") or 0) > 2 and
            (len(ranking) == 1 or
             (verdicts.get(ranking[1], {}).get("cs") or 0) < (v.get("cs") or 0))):
        # NEW emitter-penalty: reject shortcut if top has emitter=True with
        # near-zero resource anomaly magnitude (reactive caller), particularly
        # when another candidate has resource anomaly but no emitter.
        if comp_features:
            top_feat = next((f for f in comp_features if f["c"] == top), None)
            res_top = float(top_feat.get("mag") or 0) if top_feat else 0.0
            emit_top = bool(top_feat.get("emit")) if top_feat else False
            # caller_anomalies signals the top is a *reactive* caller, not
            # the true root-cause. ``v.get("call")`` exposes this count.
            caller_anomaly_count = v.get("call") or 0
            has_alt_resource = any(
                f["c"] != top and float(f.get("mag") or 0) > res_top * 2.0
                for f in comp_features
            )
            if emit_top and res_top <= 0.5 and (caller_anomaly_count > 0 or has_alt_resource):
                # Likely a downstream "Failed to call X" emitter, not the
                # root cause. Force LLM disambiguation.
                return False, None
        return True, top
    return False, None


def run_case_lightweight(
    *,
    loaded,
    client: ModelClient,
    recall_pool_size: int = 15,
) -> dict[str, Any]:
    """Signal-first lightweight agent.

    1. Extract deterministic features + IVD.
    2. If IVD has strong consensus → return immediately (0 LLM calls).
    3. Otherwise → ONE compact LLM call with <3k token summary.
    """
    signals = _extract_compact_signals(
        loaded, recall_pool_size=recall_pool_size
    )
    # Attempt deterministic shortcut
    strong, winner = _ivd_has_strong_consensus(
        signals.get("ivd"), comp_features=signals.get("components")
    )
    if strong and winner:
        return {
            "status": "lightweight_ivd_shortcut",
            "hypothesis_id": None,
            "predicted_component": winner,
            "predicted_ranking": (
                signals.get("ivd", {}).get("ranking", [])[:5]
                if signals.get("ivd") else [winner]
            ),
            "reason_family": "ivd_consensus",
            "onset_interval": [loaded.case.event_time, loaded.case.event_time + 60],
            "steps_completed": 0,
            "evidence_count": 0,
            "referenced_evidence_ids": [],
            "rationale": "IVD strong consensus, no LLM needed",
            "uncertainties": [],
            "global_rescue": False,
            "outside_hypothesis_set": False,
            "recall_pool": signals["recall_pool"],
            "event_causal_profile": {},
            "event_causal_fact_count": 0,
            "final_belief_state": [],
            "transcript": [],
            "llm_calls": 0,
        }

    # Build compact LLM prompt
    # Keep system prompt fixed for cache reuse; vary only case-specific user msg
    user_content = json.dumps(signals, ensure_ascii=False, sort_keys=True)
    request = ModelRequest(
        purpose="lightweight_rca_decision",
        messages=(
            ModelMessage(role="system", content=_LIGHTWEIGHT_SYSTEM_PROMPT),
            ModelMessage(role="user", content=user_content),
        ),
        attempt_index=0,
    )
    response = client.complete(request=request)
    # Parse response
    ivd = signals.get("ivd")
    ivd_ranking = ivd.get("ranking", []) if ivd else []
    try:
        parsed = parse_json_object(response.content)
        predicted = parsed.get("root_component", "")
        rationale = parsed.get("rationale", "")
        llm_ranking = parsed.get("ranking", [])
        # Normalise to list[str]
        if not isinstance(llm_ranking, list):
            llm_ranking = []
        llm_ranking = [str(x) for x in llm_ranking][:5]
    except Exception:
        # Fallback: use IVD top-1 if available
        predicted = ivd_ranking[0] if ivd_ranking else ""
        rationale = "LLM parse failed, fallback to IVD top-1"
        llm_ranking = []

    # Compose final ranking: LLM's list first, fall back to IVD ranking
    predicted_ranking: list[str] = []
    for c in llm_ranking:
        if c not in predicted_ranking:
            predicted_ranking.append(c)
    for c in ivd_ranking:
        if c not in predicted_ranking:
            predicted_ranking.append(c)
    if predicted and predicted not in predicted_ranking:
        predicted_ranking.insert(0, predicted)
    predicted_ranking = predicted_ranking[:5]

    return {
        "status": "lightweight_llm",
        "hypothesis_id": None,
        "predicted_component": predicted,
        "predicted_ranking": predicted_ranking,
        "reason_family": "lightweight",
        "onset_interval": [loaded.case.event_time, loaded.case.event_time + 60],
        "steps_completed": 1,
        "evidence_count": 0,
        "referenced_evidence_ids": [],
        "rationale": rationale,
        "uncertainties": [],
        "global_rescue": False,
        "outside_hypothesis_set": False,
        "recall_pool": signals["recall_pool"],
        "event_causal_profile": {},
        "event_causal_fact_count": 0,
        "final_belief_state": [],
        "transcript": [{"agent_response": response.content[:2000]}],
        "llm_calls": 1,
    }


def run_case(
    *,
    loaded,
    max_hypotheses: int,
    max_steps: int,
    client: ModelClient,
    recall_pool_size: int = 15,
) -> dict[str, Any]:
    recall_pool = _build_recall_pool(loaded, pool_size=recall_pool_size)
    bundle = build_hypotheses_from_case(
        loaded.case,
        max_hypotheses=max_hypotheses,
        candidates=recall_pool,
    )
    graph = EvidenceGraph()
    for hypothesis in bundle.hypotheses:
        if hypothesis.hypothesis_id not in graph.hypotheses_by_id:
            graph.register_hypothesis(hypothesis)
        if hypothesis.status.value == "draft":
            hypothesis.activate()

    registry = build_noiselab_tool_registry(include_default_tools=False)
    event_causal_facts = _build_event_causal_facts(
        store=loaded.store,
        case=loaded.case,
        hypotheses=bundle.hypotheses,
        recall_pool=recall_pool,
    )
    event_causal_profile = _request_event_causal_profile(
        client=client,
        case=loaded.case,
        hypotheses=bundle.hypotheses,
        event_causal_facts=event_causal_facts,
    )
    context_state: dict[str, Any] = {
        "case_narrative": "",
        "belief_state": _initial_belief_state(bundle.hypotheses),
    }
    evidence_history: list[dict[str, Any]] = []
    used_tool_calls: list[dict[str, Any]] = []
    transcript: list[dict[str, Any]] = []

    for step_index in range(max_steps):
        response = _request_agent_state(
            client=client,
            case=loaded.case,
            hypotheses=bundle.hypotheses,
            context_state=context_state,
            evidence_history=evidence_history,
            used_tool_calls=used_tool_calls,
            event_causal_profile=event_causal_profile,
            step_index=step_index,
            max_steps=max_steps,
            require_final=False,
        )
        context_state = _extract_working_memory(response, fallback=context_state)
        action = _normalize_action(
            response.get("next_action"),
            case=loaded.case,
            hypotheses=bundle.hypotheses,
            context_state=context_state,
            step_index=step_index,
            used_tool_calls=used_tool_calls,
        )
        evidence = _execute_noise_action(
            registry=registry,
            graph=graph,
            store=loaded.store,
            action=action,
        )
        evidence_record = {
            "step_index": step_index,
            "action": action,
            "evidence": _serialize_evidence(evidence),
        }
        evidence_history.append(evidence_record)
        used_tool_calls.append(
            {
                "tool_name": action["tool_name"],
                "args": action["args"],
                "query_signature": evidence.query_signature,
            }
        )
        transcript.append(
            {
                "step_index": step_index,
                "agent_response": _truncate_jsonable(response, max_chars=8000),
                "executed_action": action,
                "evidence_id": evidence.evidence_id,
            }
        )

    final_response = _request_agent_state(
        client=client,
        case=loaded.case,
        hypotheses=bundle.hypotheses,
        context_state=context_state,
        evidence_history=evidence_history,
        used_tool_calls=used_tool_calls,
        event_causal_profile=event_causal_profile,
        step_index=max_steps,
        max_steps=max_steps,
        require_final=True,
    )
    final_decision = _normalize_final_decision(
        final_response.get("final_decision"),
        case=loaded.case,
        hypotheses=bundle.hypotheses,
        graph=graph,
        context_state=_extract_working_memory(final_response, fallback=context_state),
        recall_pool=recall_pool,
    )
    return {
        "status": "continuous_final",
        "hypothesis_id": final_decision["hypothesis_id"],
        "predicted_component": final_decision["root_component"],
        "reason_family": final_decision["reason_family"],
        "onset_interval": final_decision["onset_interval"],
        "steps_completed": max_steps,
        "evidence_count": len(graph.evidence_by_id),
        "referenced_evidence_ids": final_decision["evidence_ids"],
        "rationale": final_decision["rationale"],
        "uncertainties": final_decision["uncertainties"],
        "global_rescue": bool(final_decision.get("global_rescue")),
        "outside_hypothesis_set": bool(final_decision.get("outside_hypothesis_set")),
        "recall_pool": list(recall_pool),
        "event_causal_profile": _truncate_jsonable(event_causal_profile, max_chars=20000),
        "event_causal_fact_count": len(
            event_causal_facts.get("event_causal_observations", [])
            if isinstance(event_causal_facts, Mapping)
            else []
        ),
        "final_belief_state": _truncate_jsonable(
            final_response.get("belief_state", context_state.get("belief_state", [])),
            max_chars=12000,
        ),
        "transcript": transcript,
    }


_EVENT_CAUSALIZER_SYSTEM_PROMPT = """You are EventCausalizer, a sub-agent for RCA.
Output exactly one valid JSON object and no markdown.

Your job is to convert multimodal observations into event-level causalized
features for the main NoiseNative RCA agent. You are not the final RCA judge.
Do not output probabilities, rankings, or a single winner. Instead, create a
causal event timeline, identify source-like and symptom-like cues, and explain
which ambiguities the main agent should test with NoiseLab.

Important causal rules:
- A caller->callee trace path is request direction, not automatic fault
  propagation direction. A callee/dependency fault can surface in caller errors.
- Trace error status identifies a failure boundary. It does not prove the
  boundary component is the initiating root cause.
- A log emitted by component X that mentions dependency/storage failure is
  evidence about X observing or experiencing that failure; the mentioned
  dependency is not a ground-truth label.
- Separate primary near-onset events from late dominant metric spikes.
- Latency-only early events need corroboration before they can outrank a
  slightly later component with memory/socket/cpu/error/log mechanism evidence.
- Use mechanism_strength. A weak single-signal memory event in a caller with
  callees may be queueing/blocking on a slower dependency, not an independent
  source.
- A storage/dependency component (mongo, redis, db) with resource anomalies
  but no emitter exceptions is usually a reactive dependency, not the
  initiating root cause. A service logic error can cause its storage to show
  reactive resource pressure (memory, cpu, disk) through abnormal query
  patterns.
- When a service and its storage dependency both show strong anomalies,
  list the service as source-like and the storage as symptom-like or
  ambiguous. The service is more likely the root because its logic error
  explains the storage's reactive anomalies.
- Use causal_role and source_likelihood_score from the feature row:
  source_candidate > dependency_candidate > propagation_symptom for
  root-cause likelihood. Do not promote dependency_candidate to source-like
  without independent emitter exception evidence.
- Use inferred dependency edges (inferred_from_naming in trace_context) to
  determine call direction between services and their storage dependencies.
- Never use dataset path names, fault-name labels, case IDs, or metadata.

Be concise. Reuse compact event fields instead of copying raw traces or logs.
Each list should contain at most 2 short items unless the schema explicitly
requires more.
"""


def _build_recall_pool(loaded, *, pool_size: int) -> tuple[str, ...]:
    """Build the tiered recall pool, falling back to component order."""
    method = getattr(loaded.store, "build_tiered_recall_pool", None)
    if callable(method):
        return method(pool_size=max(pool_size, max(pool_size, 2)))
    return tuple(loaded.case.components[:pool_size])


def _build_event_causal_facts(
    *,
    store,
    case,
    hypotheses,
    recall_pool: Sequence[str] | None = None,
) -> Mapping[str, Any]:
    hypothesis_components = [hypothesis.root_component for hypothesis in hypotheses]
    # Feed EventCausalizer the recall pool (the union that contains the
    # answer), not only the final top-k hypotheses.  Hypothesis components are
    # kept first so the active reasoning set is always covered.
    base: list[str] = []
    for comp in hypothesis_components:
        if comp not in base:
            base.append(comp)
    if recall_pool:
        for comp in recall_pool:
            if comp not in base:
                base.append(comp)
    components = _component_list(base, fallback=case.components)
    method = getattr(store, "build_event_causal_observations", None)
    if callable(method):
        return method(
            component_scope=components,
            time_window=tuple(_window(None, case)),
            max_events=int(os.environ.get("PRISM_CHT_EVENT_CAUSALIZER_MAX_EVENTS", "32")),
        )
    return {
        "window": _window(None, case),
        "component_scope": components,
        "event_causal_observations": [],
        "component_feature_rows": [],
        "feature_semantics": [
            "EventCausalizer deterministic store method unavailable; using case observations only."
        ],
    }


def _request_event_causal_profile(
    *,
    client: ModelClient,
    case,
    hypotheses,
    event_causal_facts: Mapping[str, Any],
) -> Mapping[str, Any]:
    schema = _event_causalizer_schema()
    compact_facts = _compact_event_causal_facts(event_causal_facts)
    context = {
        "agent": "event_causalizer",
        "instruction": (
            "Organize deterministic multimodal event facts into event-level causalized "
            "features for the main RCA agent. Do not decide the final root cause."
        ),
        "response_schema": schema,
        "case_context_without_label_leakage": _case_context(case),
        "hypotheses": [_hypothesis_context(h) for h in hypotheses],
        "event_causal_facts": _truncate_jsonable(
            {k: v for k, v in compact_facts.items() if k != "ivd"},
            max_chars=42000,
        ),
        "ivd_verdicts": compact_facts.get("ivd"),
        "output_requirements": [
            "Preserve event IDs from event_causal_facts when referring to events.",
            "Cover every event_id, but keep each event entry concise.",
            "For each event, include at most 2 source-like cues, 2 symptom-like cues, and 2 ambiguity cues.",
            "For each text field, use one short sentence. Do not copy raw trace/log examples.",
            "Prefer event-level causal relationships over component-level magnitude shortcuts.",
            "Treat mechanism_strength=weak as a caution, especially for single memory signals in caller components.",
            "Treat trace_error_boundary as symptom-surface evidence unless independent source mechanism exists.",
            "is_entry_like_component marks the request surface (where traffic enters), NOT root-cause strength. An entry component with internal anomalies is one candidate among several; do not treat entry status as evidence that it is the initiating fault.",
            "When an entry-like component and a background service both show strong internal mechanism signals, list both as source-like and let the main agent compare onset timing, dependency direction, and which component's anomaly explains the other.",
            "causal_role=dependency_candidate means the component is a storage/database layer with resource anomalies but no emitter exceptions; resource anomalies in storage are often reactive to service-level faults. Do not list dependency_candidate components as source-like unless they have independent emitter exceptions.",
            "causal_role=source_candidate means the component has emitter exceptions or strong internal mechanism as a service; prioritize these over dependency_candidate components when both show anomalies.",
            "Use inferred dependency edges (inferred_from_naming in trace_context) to determine call direction: if a service calls a storage, the service is the caller and the storage is the callee/dependency.",
            "When a service and its storage dependency both show strong anomalies, list the service as source-like and the storage as symptom-like or ambiguous, because a service logic error can cause its storage to show reactive resource pressure.",
            "IVD (Initiator-Victim Disambiguation) verdicts are provided in event_causal_facts.ivd. Use them to classify candidates: ivd_role=initiator should be listed as source-like, ivd_role=victim should be listed as symptom-like. The initiator is the candidate whose anomaly is least explainable by others.",
            "Do not use raw event timeline ordering to break ties. If two candidates have near-simultaneous onsets (gap < 10s), raw timestamp order is unreliable.",
            "Do not include case IDs, dataset paths, expected labels, rankings, scores, or probabilities.",
        ],
    }
    request = ModelRequest(
        purpose="event_causalizer_state_update",
        messages=(
            ModelMessage(role="system", content=_EVENT_CAUSALIZER_SYSTEM_PROMPT),
            ModelMessage(
                role="user",
                content=json.dumps(
                    canonicalize_json_value(context),
                    indent=2,
                    sort_keys=True,
                    ensure_ascii=True,
                ),
            ),
        ),
        attempt_index=0,
    )
    response = client.complete(request=request)
    parsed = _parse_or_repair_model_json_object(
        client=client,
        initial_text=response.content,
        original_context=context,
        required_response_schema=schema,
        require_final=False,
    )
    return _normalize_event_causal_profile(parsed, fallback_facts=event_causal_facts)


def _event_causalizer_schema() -> Mapping[str, Any]:
    return {
        "event_timeline": [
            {
                "event_id": "existing event id",
                "component": "component name",
                "event_time": 0.0,
                "event_kind": "string",
                "mechanism_strength": "strong | moderate | weak | symptom_like | unknown",
                "multimodal_evidence": ["short metric/log/trace fact"],
                "source_like_cues": ["string"],
                "symptom_like_cues": ["string"],
                "ambiguity_cues": ["string"],
                "causal_links_to_check": ["string"],
                "diagnostic_implication": "string",
            }
        ],
        "component_event_features": [
            {
                "component": "component name",
                "primary_event_ids": ["event ids"],
                "near_onset_mechanism": ["string"],
                "symptom_visibility": ["string"],
                "topology_notes": ["string"],
                "timing_cautions": ["string"],
                "missing_checks": ["string"],
            }
        ],
        "causal_story_options": [
            {
                "story_id": "string",
                "initiating_event_ids": ["event ids"],
                "propagation_reading": "string",
                "what_would_support_it": ["string"],
                "what_would_weaken_it": ["string"],
            }
        ],
        "main_agent_guidance": {
            "high_value_next_actions": ["string"],
            "pitfalls_to_avoid": ["string"],
            "event_features_to_reuse_in_final_reasoning": ["string"],
        },
    }


def _normalize_event_causal_profile(
    value: Mapping[str, Any],
    *,
    fallback_facts: Mapping[str, Any],
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        value = {}
    events = value.get("event_timeline")
    if not isinstance(events, list) or not events:
        events = [
            {
                "event_id": item.get("event_id", "unknown"),
                "component": item.get("component", "unknown"),
                "event_time": item.get("event_time"),
                "event_kind": item.get("event_kind", "unknown"),
                "mechanism_strength": (
                    ((item.get("causalized_features") or {}).get("mechanism_strength") or {}).get(
                        "level", "unknown"
                    )
                ),
                "multimodal_evidence": ["fallback from deterministic EventCausalizer facts"],
                "source_like_cues": list(
                    ((item.get("causalized_features") or {}).get("event_causal_cues") or {}).get(
                        "source_like", []
                    )
                ),
                "symptom_like_cues": list(
                    ((item.get("causalized_features") or {}).get("event_causal_cues") or {}).get(
                        "symptom_like", []
                    )
                ),
                "ambiguity_cues": list(
                    ((item.get("causalized_features") or {}).get("event_causal_cues") or {}).get(
                        "ambiguity", []
                    )
                ),
                "causal_links_to_check": [],
                "diagnostic_implication": "LLM event causalizer output missing; use deterministic event cues.",
            }
            for item in fallback_facts.get("event_causal_observations", [])[:8]
            if isinstance(item, Mapping)
        ]
    component_features = value.get("component_event_features")
    if not isinstance(component_features, list):
        component_features = []
    story_options = value.get("causal_story_options")
    if not isinstance(story_options, list):
        story_options = []
    guidance = value.get("main_agent_guidance")
    if not isinstance(guidance, Mapping):
        guidance = {
            "high_value_next_actions": [],
            "pitfalls_to_avoid": [],
            "event_features_to_reuse_in_final_reasoning": [],
        }
    return {
        "event_timeline": _compact_profile_events(events),
        "component_event_features": _compact_component_event_features(component_features),
        "causal_story_options": _compact_story_options(story_options),
        "main_agent_guidance": _compact_guidance(dict(guidance)),
        "ivd": _compact_ivd(fallback_facts.get("ivd")),
        "profile_semantics": [
            "This profile is an event-level feature map generated by a sub-agent, not a final RCA decision.",
            "Use event IDs and causal cues to plan NoiseLab actions and final reasoning.",
            "IVD verdicts (if present) classify candidates as initiator or victim. Use them to guide final selection.",
        ],
    }


def _compact_event_causal_facts(facts: Mapping[str, Any]) -> Mapping[str, Any]:
    events = []
    for event in facts.get("event_causal_observations", []):
        if not isinstance(event, Mapping):
            continue
        causal = event.get("causalized_features") or {}
        bundle = event.get("multimodal_bundle") or {}
        trace_context = bundle.get("trace_context") or {}
        mechanism = causal.get("mechanism_strength") or {}
        events.append(
            {
                "event_id": event.get("event_id"),
                "component": event.get("component"),
                "event_time": event.get("event_time"),
                "event_kind": event.get("event_kind"),
                "temporal_anchor": event.get("temporal_anchor"),
                "mechanism_strength": mechanism,
                "near_onset": {
                    "signals": causal.get("near_onset_signals", []),
                    "internal_signals": causal.get("near_onset_internal_signals", []),
                    "magnitude": causal.get("near_onset_anomaly_magnitude"),
                    "latency_only": causal.get("latency_only_near_onset"),
                },
                "causal_role": causal.get("causal_role", "ambiguous"),
                "source_likelihood_score": causal.get("source_likelihood_score", 0.0),
                "is_storage_component": causal.get("is_storage_component", False),
                "timing_flags": {
                    "late_dominant_metric": causal.get("late_dominant_metric"),
                    "dominant_metric": _compact_metric(causal.get("dominant_metric")),
                },
                "metric_summary": [
                    _compact_metric(item)
                    for item in (bundle.get("metric_features") or [])[:6]
                    if isinstance(item, Mapping)
                ],
                "log_summary": [
                    _compact_log_feature(item)
                    for item in (bundle.get("log_features") or [])[:3]
                    if isinstance(item, Mapping)
                ],
                "trace_summary": _compact_trace_context(trace_context),
                "causal_cues": causal.get("event_causal_cues", {}),
            }
        )
    return {
        "window": facts.get("window"),
        "component_scope": facts.get("component_scope"),
        "event_causal_observations": events,
        "feature_semantics": facts.get("feature_semantics", []),
        "ivd": _compact_ivd(facts.get("ivd")),
        "load_control_note": (
            "This is a compact projection preserving every event_id while omitting raw traces/log bodies."
        ),
    }


def _compact_metric(value: Any) -> Mapping[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    return {
        "signal": value.get("signal"),
        "raw_metric": value.get("raw_metric"),
        "first_seen": value.get("first_seen"),
        "magnitude": value.get("magnitude"),
        "direction": value.get("direction"),
        "seconds_after_earliest": value.get("seconds_after_earliest"),
    }


def _compact_ivd(value: Any) -> Mapping[str, Any] | None:
    if not isinstance(value, Mapping) or not value.get("ivd_enabled"):
        return None
    verdicts = value.get("verdicts", {})
    compact_verdicts = {}
    for comp, v in verdicts.items():
        compact_verdicts[comp] = {
            "ivd_role": v.get("ivd_role"),
            "copeland_score": v.get("copeland_score"),
            "fault_signature_label": v.get("fault_signature_label"),
            "temporal_verdict": v.get("temporal_verdict"),
            "temporal_gap_seconds": v.get("temporal_gap_seconds"),
            "downstream_coverage_count": v.get("downstream_coverage_count"),
            "caller_anomalies_count": v.get("caller_anomalies_count"),
            "caller_anomalies": v.get("caller_anomalies"),
            "resource_magnitude": v.get("resource_magnitude"),
            "workload_magnitude": v.get("workload_magnitude"),
        }
    return {
        "ivd_enabled": True,
        "ranking": value.get("ranking", []),
        "verdicts": compact_verdicts,
        "interpretation": value.get("interpretation", ""),
    }


def _compact_log_feature(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        "emitter_component": value.get("emitter_component"),
        "timestamp": value.get("timestamp"),
        "dependency_terms": value.get("dependency_terms", [])[:5]
        if isinstance(value.get("dependency_terms"), list)
        else [],
        "emitter_exception_observed": value.get("emitter_exception_observed"),
        "diagnostic_role": value.get("diagnostic_role"),
        "message_preview": _short_string(value.get("message_preview", ""), max_chars=120),
    }


def _compact_trace_context(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    return {
        "component": value.get("component"),
        "direct_callers": [
            _compact_trace_edge(item)
            for item in (value.get("direct_callers") or [])[:5]
            if isinstance(item, Mapping)
        ],
        "direct_callees": [
            _compact_trace_edge(item)
            for item in (value.get("direct_callees") or [])[:5]
            if isinstance(item, Mapping)
        ],
        "request_direction_semantics": value.get("request_direction_semantics"),
    }


def _compact_trace_edge(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        "component": value.get("component"),
        "call_count": value.get("call_count"),
        "error_status_count": value.get("error_status_count"),
        "latency_examples": (value.get("latency_examples") or [])[:2]
        if isinstance(value.get("latency_examples"), list)
        else [],
        "status_examples": (value.get("status_examples") or [])[:2]
        if isinstance(value.get("status_examples"), list)
        else [],
    }


def _compact_profile_events(events: Sequence[Any]) -> list[Mapping[str, Any]]:
    compact = []
    for event in events:
        if not isinstance(event, Mapping):
            continue
        compact.append(
            {
                "event_id": event.get("event_id"),
                "component": event.get("component"),
                "event_time": event.get("event_time"),
                "event_kind": event.get("event_kind"),
                "mechanism_strength": _short_string(event.get("mechanism_strength", "unknown")),
                "multimodal_evidence": _short_list(event.get("multimodal_evidence"), max_items=3),
                "source_like_cues": _short_list(event.get("source_like_cues"), max_items=2),
                "symptom_like_cues": _short_list(event.get("symptom_like_cues"), max_items=2),
                "ambiguity_cues": _short_list(event.get("ambiguity_cues"), max_items=3),
                "causal_links_to_check": _short_list(event.get("causal_links_to_check"), max_items=3),
                "diagnostic_implication": _short_string(
                    event.get("diagnostic_implication", ""),
                    max_chars=220,
                ),
            }
        )
    return compact


def _compact_component_event_features(features: Sequence[Any]) -> list[Mapping[str, Any]]:
    compact = []
    for item in features:
        if not isinstance(item, Mapping):
            continue
        compact.append(
            {
                "component": item.get("component"),
                "primary_event_ids": _short_list(item.get("primary_event_ids"), max_items=4, max_chars=80),
                "near_onset_mechanism": _short_list(item.get("near_onset_mechanism"), max_items=3),
                "symptom_visibility": _short_list(item.get("symptom_visibility"), max_items=2),
                "topology_notes": _short_list(item.get("topology_notes"), max_items=2),
                "timing_cautions": _short_list(item.get("timing_cautions"), max_items=2),
                "missing_checks": _short_list(item.get("missing_checks"), max_items=3),
            }
        )
    return compact


def _compact_story_options(stories: Sequence[Any]) -> list[Mapping[str, Any]]:
    compact = []
    story_items = stories[:4] if isinstance(stories, list) else []
    for item in story_items:
        if not isinstance(item, Mapping):
            continue
        compact.append(
            {
                "story_id": item.get("story_id"),
                "initiating_event_ids": _short_list(item.get("initiating_event_ids"), max_items=4, max_chars=80),
                "propagation_reading": _short_string(item.get("propagation_reading", ""), max_chars=220),
                "what_would_support_it": _short_list(item.get("what_would_support_it"), max_items=2),
                "what_would_weaken_it": _short_list(item.get("what_would_weaken_it"), max_items=2),
            }
        )
    return compact


def _compact_guidance(guidance: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        "high_value_next_actions": _short_list(guidance.get("high_value_next_actions"), max_items=4),
        "pitfalls_to_avoid": _short_list(guidance.get("pitfalls_to_avoid"), max_items=5),
        "event_features_to_reuse_in_final_reasoning": _short_list(
            guidance.get("event_features_to_reuse_in_final_reasoning"),
            max_items=5,
        ),
    }


def _short_list(value: Any, *, max_items: int, max_chars: int = 180) -> list[str]:
    if not isinstance(value, list):
        return []
    return [
        _short_string(item, max_chars=max_chars)
        for item in value[:max_items]
        if str(item).strip()
    ]


def _short_string(value: Any, *, max_chars: int = 180) -> str:
    text = str(value).replace("\n", " ").strip()
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 3)].rstrip() + "..."


_SYSTEM_PROMPT = """You are a continuous NoiseNative RCA agent.
Output exactly one valid JSON object and no markdown. Use double-quoted JSON
keys and strings, no comments, no duplicate keys, no trailing text, and no
Python-style literals.
Keep the JSON concise: case_narrative <= 120 words, no more than 5 belief
items, no more than 3 strings per support/contradiction/missing list, and
final rationale <= 180 words. When remaining_steps == 0, final_decision is
mandatory.

You are NOT a stepwise judge. Maintain one coherent case-level causal
working memory across the whole episode. Each tool result updates the same
belief state; do not reset reasoning between turns.

NoiseLab is a feature-fusion layer. It returns facts about local anomaly
features, source-vs-symptom cues, counterfactual observables, and downstream
explanation facts. NoiseLab does not decide the root cause for you.

An EventCausalizer sub-agent has already converted multimodal observations
into event-level causalized features. Treat event_causal_profile as a compact
timeline and feature map for reasoning, not as a final answer. Cross-check it
with NoiseLab actions when uncertainty remains. If EventCausalizer marks an
event as mechanism_strength=weak, symptom_like, or caller-side memory caution,
do not promote it to root cause without independent NoiseLab corroboration.

Root cause means the initiating faulty component and mechanism. It is not
necessarily the earliest observed component, the largest local anomaly, the
entry service, or the component with the most logs. Treat onset, local
magnitude, and missing paths as partial evidence only. A downstream symptom
can be early or loud.

Trace paths are request/call direction, not automatic fault-propagation
direction. If frontend calls currencyservice, a currencyservice fault can
surface as frontend errors even though the trace edge points frontend ->
currencyservice. Likewise, a component log mentioning a dependency proves the
emitter observed a dependency/storage failure; it does not by itself prove the
mentioned dependency is the root cause. A dominant metric spike that occurs
well after onset should not outweigh near-onset mechanism evidence.

Before final_decision, apply these hard causal checks:
- Do not choose a mentioned dependency from an emitter log unless the dependency
  has independent near-onset source-mechanism evidence and timing/topology
  support. If the dependency is later, has no trace/topology support, or only a
  late dominant spike, keep the emitting component as a serious root candidate.
- Do not choose a component whose evidence is mainly latency degradation solely
  because it is earliest. Latency-only early signals are often symptoms of a
  slower dependency. Prefer a slightly later component with internal near-onset
  mechanism signals such as memory, socket, cpu, disk, error, or emitter-side
  exception evidence.
- Do not choose a request failure boundary solely because trace status errors
  point to it. A boundary component may be timing out because one of its callees
  is slow or faulty.
- Do not choose a storage/dependency component (mongo, redis, db) over a
  service with emitter exceptions solely because the storage has stronger
  resource anomalies. A service logic error can cause its storage dependency
  to show reactive resource pressure (memory, cpu, disk) through abnormal
  query patterns. Use causal_role: source_candidate > dependency_candidate
  for root-cause likelihood. Use inferred_dependency_edges to determine call
  direction: if a service calls a storage, the service is the caller and the
  storage is the callee/dependency.
- When a service and its storage dependency both show strong anomalies, prefer
  the service as the root cause. The service's emitter exceptions (application-
  level errors) explain why the storage shows resource anomalies (reactive
  load from abnormal queries). The reverse direction (storage causes service
  failure) would typically produce timeout errors in the service, not emitter
  exceptions.
- IVD (Initiator-Victim Disambiguation): when multiple services have similar
  source_likelihood scores, do NOT select the one that appears earliest in
  the event timeline. Use the IVD verdicts instead: the initiator is the
  candidate whose anomaly is least explainable by other candidates
  (exogeneity) and most explanatory for downstream anomalies (coverage).
  A candidate with ivd_role=initiator MUST be selected over any candidate
  labeled ambiguous or victim. Earliest unexplained cause beats earliest
  observed symptom.
- Temporal fragility: if two candidates have near-simultaneous onsets
  (gap < 10s), raw timestamp order is unreliable and must not be used as
  the sole tiebreaker. Use topology-constrained ordering and IVD instead.
- FaultSignature: one of six IVD channels, not a standalone override. Own-code
  stack traces suggest internal fault, but a victim can also have own-code
  exceptions triggered by the initiator's upstream fault (e.g., invalid data
  propagated from the true root). Do not use fault signature alone to override
  IVD ranking.

Never use dataset path names, fault-name labels, case IDs, or metadata to infer
the answer. They are intentionally omitted from the case context.

When remaining_steps > 0, return a next_action that maximizes information gain
for unresolved source-vs-symptom ambiguity. When remaining_steps == 0, return
final_decision and set next_action to null.
"""


def _request_agent_state(
    *,
    client: ModelClient,
    case,
    hypotheses,
    context_state: Mapping[str, Any],
    evidence_history: Sequence[Mapping[str, Any]],
    used_tool_calls: Sequence[Mapping[str, Any]],
    event_causal_profile: Mapping[str, Any],
    step_index: int,
    max_steps: int,
    require_final: bool,
) -> Mapping[str, Any]:
    context = {
        "agent": "continuous_noise_native",
        "step_index": step_index,
        "max_steps": max_steps,
        "remaining_steps": max(0, max_steps - step_index),
        "instruction": (
            "Update the continuous belief_state, preserve a coherent case_narrative, "
            "and either choose next_action or produce final_decision."
        ),
        "response_schema": _response_schema(require_final=require_final),
        "tool_contracts": _tool_contracts(),
        "allowed_tool_names": list(NOISELAB_TOOL_NAMES),
        "case_context_without_label_leakage": _case_context(case),
        "hypotheses": [_hypothesis_context(h) for h in hypotheses],
        "event_causal_profile": _truncate_jsonable(
            {k: v for k, v in event_causal_profile.items() if k != "ivd"},
            max_chars=28000,
        ),
        "ivd_verdicts": event_causal_profile.get("ivd"),
        "previous_working_memory": _truncate_jsonable(context_state, max_chars=20000),
        "evidence_history": _truncate_jsonable(evidence_history, max_chars=40000),
        "used_tool_calls": list(used_tool_calls),
        "reasoning_rules": [
            "Do not drop a true-looking hypothesis solely because onset is later.",
            "Do not pick a component solely because local_anomaly_magnitude is largest.",
            "Use near_onset_anomaly_magnitude before trusting a late dominant metric spike.",
            "Do not treat frontend, gateway, or any entry-like component as root solely because it has request errors or is the request surface. Entry status is a propagation-direction cue, not root-cause strength.",
            "When an entry-like component and a background service both show strong internal mechanism signals, compare onset timing and dependency direction: the component whose anomaly explains the other's anomaly (via caller->callee edges) is more likely the source, regardless of which is the entry.",
            "Trace candidate_to_symptom paths are caller-to-callee request paths; they are not direct proof that the caller caused the callee.",
            "If symptom_to_candidate paths exist, consider whether the candidate is a dependency whose failure surfaced upstream.",
            "A log emitted by component X that mentions dependency/storage failure is evidence about X observing the dependency; it is not sufficient alone to choose the dependency.",
            "A dependency mentioned in an emitter log needs independent near-onset mechanism evidence before it can outrank the emitter.",
            "A latency-only earliest component needs additional internal mechanism evidence before it can outrank a slightly later memory/socket/cpu/error component.",
            "A weak EventCausalizer mechanism_strength or caller_side_memory_caution requires NoiseLab corroboration before final selection.",
            "Trace status errors identify the failure boundary; test whether a callee dependency explains the boundary component before selecting it.",
            "A storage/dependency component (mongo, redis, db) with resource anomalies but no emitter exceptions is usually a reactive dependency, not the initiating root. Use causal_role: source_candidate > dependency_candidate.",
            "When a service and its storage dependency both show strong anomalies, prefer the service: a service logic error causes abnormal DB queries that make the storage show reactive resource pressure. The reverse (storage causes service failure) would produce timeout errors, not emitter exceptions.",
            "Use source_likelihood_score for ranking: higher score = more likely initiating root. Compare top_source_candidates_by_likelihood from compare_source_symptom output.",
            "Use inferred_dependency_edges to determine call direction: if service A calls storage B, A is the caller and B is the callee/dependency; B's resource anomalies may be caused by A's fault.",
            "IVD (Initiator-Victim Disambiguation): the ivd_verdicts field provides a pairwise tournament ranking (Copeland score) among source candidates. The initiator (ivd_role=initiator) is the candidate whose anomaly is least explainable by others and most explanatory for others. When IVD labels one candidate as initiator, you MUST select it over any candidate labeled ambiguous or victim. Do NOT override IVD ranking based on fault signature, caller count, or onset timing alone — the IVD tournament already weighs all six evidence channels.",
            "Do not use raw event timeline ordering to break ties between candidates. If two candidates have near-simultaneous onsets (gap < 10s), raw timestamp order is unreliable (TemporalFragility). Use IVD ranking instead.",
            "FaultSignature is one of six IVD channels, not a standalone override. A candidate with own-code exceptions but low IVD copeland score is likely a victim whose exception was triggered by the initiator's upstream fault (e.g., invalid data propagated from the true root). Do not use fault signature alone to override IVD ranking.",
            "Exogeneity and coverage are IVD channels, not standalone rules. The IVD verdicts already incorporate caller_anomalies_count and downstream_coverage_count with resource-magnitude gating. Do not re-apply these heuristics independently to override IVD ranking.",
            "After each evidence item, explicitly say whether it suggests source, symptom, or ambiguity.",
            "Every leading hypothesis must have a why_not_others comparison.",
            "Use missing_information to drive the next action.",
        ],
        "output_limits": {
            "max_belief_items": min(5, len(hypotheses)),
            "max_factors_per_list": 3,
            "max_case_narrative_words": 120,
            "max_final_rationale_words": 180,
            "require_final_decision_when_remaining_steps_is_zero": require_final,
        },
    }
    request = ModelRequest(
        purpose="continuous_noise_native_state_update",
        messages=(
            ModelMessage(role="system", content=_SYSTEM_PROMPT),
            ModelMessage(
                role="user",
                content=json.dumps(
                    canonicalize_json_value(context),
                    indent=2,
                    sort_keys=True,
                    ensure_ascii=True,
                ),
            ),
        ),
        attempt_index=0,
    )
    response = client.complete(request=request)
    return _parse_or_repair_model_json_object(
        client=client,
        initial_text=response.content,
        original_context=context,
        required_response_schema=_response_schema(require_final=require_final),
        require_final=require_final,
    )


def _parse_or_repair_model_json_object(
    *,
    client: ModelClient,
    initial_text: str,
    original_context: Mapping[str, Any],
    required_response_schema: Mapping[str, Any],
    require_final: bool,
) -> Mapping[str, Any]:
    repair_attempt = 0
    max_repairs = max(1, int(os.environ.get("PRISM_CHT_JSON_REPAIR_ATTEMPTS", "6")))
    text = initial_text
    last_error = ""
    while True:
        try:
            return _parse_model_json_object(text)
        except StructuredOutputError as exc:
            last_error = str(exc)
            if repair_attempt >= max_repairs:
                raise
            repair_attempt += 1
            text = _request_json_repair(
                client=client,
                bad_text=text,
                parser_error=last_error,
                original_context=original_context,
                required_response_schema=required_response_schema,
                require_final=require_final,
                repair_attempt=repair_attempt,
            )


def _parse_model_json_object(text: str) -> Mapping[str, Any]:
    try:
        return parse_json_object(text, max_chars=200000)
    except StructuredOutputError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise
        return parse_json_object(text[start : end + 1], max_chars=200000)


_JSON_REPAIR_SYSTEM_PROMPT = """You repair invalid structured RCA agent output.
Return exactly one valid JSON object and nothing else. Do not add markdown,
comments, duplicate keys, trailing text, or explanations outside JSON.

Preserve the original RCA content and decisions as much as possible. Only
change formatting or structure needed to satisfy the requested JSON schema.
"""


def _request_json_repair(
    *,
    client: ModelClient,
    bad_text: str,
    parser_error: str,
    original_context: Mapping[str, Any],
    required_response_schema: Mapping[str, Any],
    require_final: bool,
    repair_attempt: int,
) -> str:
    repair_payload = {
        "task": "Repair the invalid model output into one valid JSON object.",
        "parser_error": parser_error,
        "required_response_schema": required_response_schema,
        "format_rules": [
            "Return only a JSON object that starts with { and ends with }.",
            "Use double quotes for all keys and string values.",
            "Do not use markdown fences, comments, duplicate keys, NaN, Infinity, or trailing prose.",
            "Keep the same RCA reasoning, belief_state, next_action, and final_decision content whenever recoverable.",
        ],
        "original_turn_context": {
            "step_index": original_context.get("step_index"),
            "remaining_steps": original_context.get("remaining_steps"),
            "require_final": require_final,
            "allowed_tool_names": original_context.get("allowed_tool_names"),
            "hypotheses": original_context.get("hypotheses"),
            "previous_working_memory": original_context.get("previous_working_memory"),
            "evidence_history_tail": list(original_context.get("evidence_history", []))[-2:],
            "used_tool_calls": original_context.get("used_tool_calls"),
        },
        "invalid_model_output": bad_text,
    }
    request = ModelRequest(
        purpose="continuous_noise_native_json_repair",
        messages=(
            ModelMessage(role="system", content=_JSON_REPAIR_SYSTEM_PROMPT),
            ModelMessage(
                role="user",
                content=json.dumps(
                    canonicalize_json_value(repair_payload),
                    indent=2,
                    sort_keys=True,
                    ensure_ascii=True,
                ),
            ),
        ),
        attempt_index=repair_attempt,
    )
    return client.complete(request=request).content


def _response_schema(*, require_final: bool) -> Mapping[str, Any]:
    return {
        "case_narrative": "continuous concise causal story",
        "belief_state": [
            {
                "hypothesis_id": "string",
                "root_component": "string",
                "position": "leading | plausible | weakened | unlikely",
                "source_vs_symptom_judgment": "string",
                "supporting_factors": ["string"],
                "contradicting_factors": ["string"],
                "missing_information": ["string"],
                "why_not_others": "string",
            }
        ],
        "next_action": None
        if require_final
        else {
            "action_id": "string",
            "tool_name": "one allowed tool name",
            "target_hypothesis_ids": ["hypothesis ids"],
            "question": "string",
            "args": {},
            "expected_information_gain": "string",
        },
        "final_decision": {
            "hypothesis_id": "string",
            "root_component": "string",
            "reason_family": "string",
            "onset_interval": [0.0, 0.0],
            "evidence_ids": ["existing evidence ids"],
            "rationale": "string",
            "uncertainties": ["string"],
        }
        if require_final
        else None,
    }


def _tool_contracts() -> Mapping[str, Any]:
    return {
        "inspect_noise_features": {
            "args": {
                "component_scope": ["component", "..."],
                "time_window": [0.0, 0.0],
                "feature_groups": ["onset", "local_mechanism", "logs", "propagation"],
            },
            "use_when": "Need fused local/onset/log features for one or more candidates.",
        },
        "compare_source_symptom": {
            "args": {
                "component_scope": ["component_a", "component_b", "..."],
                "time_window": [0.0, 0.0],
            },
            "use_when": "Need compare source-vs-symptom cues across candidates.",
        },
        "counterfactual_remove": {
            "args": {
                "component": "candidate component",
                "symptom_components": ["symptom", "..."],
                "time_window": [0.0, 0.0],
            },
            "use_when": "Need facts for whether removing one candidate would explain observed symptoms.",
        },
        "test_downstream_explanation": {
            "args": {
                "candidate_component": "candidate component",
                "symptom_components": ["symptom", "..."],
                "time_window": [0.0, 0.0],
            },
            "use_when": "Need whether a candidate can explain downstream affected components.",
        },
    }


def _case_context(case) -> Mapping[str, Any]:
    return {
        "event_time": case.event_time,
        "components": list(case.components),
        "entry_components": list(case.entry_components),
        "observations": [
            {
                "component": obs.component,
                "reason_family": obs.reason_family,
                "first_seen": obs.first_seen,
                "magnitude": obs.magnitude,
                "signals": list(obs.signals),
                "symptoms": list(obs.symptoms),
            }
            for obs in case.observations
        ],
    }


def _hypothesis_context(hypothesis) -> Mapping[str, Any]:
    return {
        "hypothesis_id": hypothesis.hypothesis_id,
        "root_component": hypothesis.root_component,
        "reason_family": hypothesis.reason_family,
        "onset_interval": list(hypothesis.onset_interval),
        "local_trigger": hypothesis.local_trigger,
        "propagation_path": list(hypothesis.propagation_path),
        "explained_symptoms": list(hypothesis.explained_symptoms),
        "predicted_observations": list(hypothesis.predicted_observations),
        "falsifiers": list(hypothesis.falsifiers),
    }


def _initial_belief_state(hypotheses) -> list[dict[str, Any]]:
    return [
        {
            "hypothesis_id": hypothesis.hypothesis_id,
            "root_component": hypothesis.root_component,
            "position": "plausible",
            "source_vs_symptom_judgment": "not yet evaluated",
            "supporting_factors": [],
            "contradicting_factors": [],
            "missing_information": [
                "local mechanism features",
                "source-vs-symptom comparison",
                "downstream explanation facts",
            ],
            "why_not_others": "not yet compared",
        }
        for hypothesis in hypotheses
    ]


def _extract_working_memory(
    response: Mapping[str, Any],
    *,
    fallback: Mapping[str, Any],
) -> dict[str, Any]:
    belief = response.get("belief_state")
    if not isinstance(belief, list) or not belief:
        belief = fallback.get("belief_state", [])
    narrative = response.get("case_narrative")
    if not isinstance(narrative, str) or not narrative.strip():
        narrative = str(fallback.get("case_narrative", ""))
    return {
        "case_narrative": narrative,
        "belief_state": belief,
    }


def _normalize_action(
    value: Any,
    *,
    case,
    hypotheses,
    context_state: Mapping[str, Any],
    step_index: int,
    used_tool_calls: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return _fallback_action(
            case=case,
            hypotheses=hypotheses,
            context_state=context_state,
            step_index=step_index,
            used_tool_calls=used_tool_calls,
        )
    tool_name = str(value.get("tool_name", "")).strip()
    if tool_name not in NOISELAB_TOOL_NAMES:
        return _fallback_action(
            case=case,
            hypotheses=hypotheses,
            context_state=context_state,
            step_index=step_index,
            used_tool_calls=used_tool_calls,
        )
    components = [hypothesis.root_component for hypothesis in hypotheses]
    args = _repair_tool_args(
        tool_name=tool_name,
        args=value.get("args"),
        case=case,
        components=components,
        context_state=context_state,
    )
    target_ids = [
        str(item)
        for item in value.get("target_hypothesis_ids", [])
        if str(item) in {hypothesis.hypothesis_id for hypothesis in hypotheses}
    ]
    if len(target_ids) < 2:
        target_ids = [hypothesis.hypothesis_id for hypothesis in hypotheses[: min(3, len(hypotheses))]]
    return {
        "action_id": str(value.get("action_id") or f"continuous-step-{step_index}"),
        "tool_name": tool_name,
        "target_hypothesis_ids": target_ids,
        "question": str(value.get("question") or "Extract NoiseLab facts for unresolved RCA ambiguity."),
        "args": args,
        "expected_information_gain": str(
            value.get("expected_information_gain")
            or "Clarify source-vs-symptom evidence."
        ),
    }


def _repair_tool_args(
    *,
    tool_name: str,
    args: Any,
    case,
    components: Sequence[str],
    context_state: Mapping[str, Any],
) -> dict[str, Any]:
    raw = dict(args) if isinstance(args, Mapping) else {}
    window = _window(raw.get("time_window"), case)
    leading = _leading_component(context_state, components)
    symptoms = _symptom_components(case=case, components=components, exclude=leading)

    if tool_name == "inspect_noise_features":
        scope = _component_list(raw.get("component_scope"), fallback=components)
        return {
            "component_scope": scope,
            "time_window": window,
            "feature_groups": _component_list(
                raw.get("feature_groups"),
                fallback=("onset", "local_mechanism", "logs", "propagation"),
            ),
        }
    if tool_name == "compare_source_symptom":
        return {
            "component_scope": _component_list(raw.get("component_scope"), fallback=components),
            "time_window": window,
        }
    if tool_name == "counterfactual_remove":
        return {
            "component": str(raw.get("component") or leading),
            "symptom_components": _component_list(
                raw.get("symptom_components"),
                fallback=symptoms,
            ),
            "time_window": window,
        }
    if tool_name == "test_downstream_explanation":
        return {
            "candidate_component": str(raw.get("candidate_component") or leading),
            "symptom_components": _component_list(
                raw.get("symptom_components"),
                fallback=symptoms,
            ),
            "time_window": window,
        }
    raise ValueError(f"unsupported NoiseLab tool: {tool_name}")


def _fallback_action(
    *,
    case,
    hypotheses,
    context_state: Mapping[str, Any],
    step_index: int,
    used_tool_calls: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    components = [hypothesis.root_component for hypothesis in hypotheses]
    leading = _leading_component(context_state, components)
    symptoms = _symptom_components(case=case, components=components, exclude=leading)
    candidates = [
        {
            "action_id": f"fallback-features-{step_index}",
            "tool_name": "inspect_noise_features",
            "target_hypothesis_ids": [hypothesis.hypothesis_id for hypothesis in hypotheses],
            "question": "Build fused NoiseLab feature table for all candidates.",
            "args": {
                "component_scope": components,
                "time_window": _window(None, case),
                "feature_groups": ["onset", "local_mechanism", "logs", "propagation"],
            },
            "expected_information_gain": "Establish shared feature memory before narrowing candidates.",
        },
        {
            "action_id": f"fallback-source-symptom-{step_index}",
            "tool_name": "compare_source_symptom",
            "target_hypothesis_ids": [hypothesis.hypothesis_id for hypothesis in hypotheses],
            "question": "Compare source-vs-symptom cues for all candidates.",
            "args": {
                "component_scope": components,
                "time_window": _window(None, case),
            },
            "expected_information_gain": "Separate likely source candidates from loud symptoms.",
        },
        {
            "action_id": f"fallback-downstream-{step_index}",
            "tool_name": "test_downstream_explanation",
            "target_hypothesis_ids": [hypothesis.hypothesis_id for hypothesis in hypotheses],
            "question": "Test whether the leading candidate explains observed symptoms.",
            "args": {
                "candidate_component": leading,
                "symptom_components": symptoms,
                "time_window": _window(None, case),
            },
            "expected_information_gain": "Check if the leading candidate can explain downstream observations.",
        },
        {
            "action_id": f"fallback-counterfactual-{step_index}",
            "tool_name": "counterfactual_remove",
            "target_hypothesis_ids": [hypothesis.hypothesis_id for hypothesis in hypotheses],
            "question": "Expose counterfactual facts for removing the leading candidate.",
            "args": {
                "component": leading,
                "symptom_components": symptoms,
                "time_window": _window(None, case),
            },
            "expected_information_gain": "Clarify whether the leading component accounts for remaining symptoms.",
        },
    ]
    used = {
        json.dumps(
            {"tool_name": item["tool_name"], "args": item["args"]},
            sort_keys=True,
            ensure_ascii=True,
        )
        for item in used_tool_calls
    }
    for candidate in candidates:
        key = json.dumps(
            {"tool_name": candidate["tool_name"], "args": candidate["args"]},
            sort_keys=True,
            ensure_ascii=True,
        )
        if key not in used:
            return candidate
    return candidates[step_index % len(candidates)]


def _execute_noise_action(
    *,
    registry,
    graph: EvidenceGraph,
    store,
    action: Mapping[str, Any],
) -> EvidenceAtom:
    tool_name = str(action["tool_name"])
    args = canonicalize_json_value(dict(action["args"]))
    signature = build_tool_call_signature(tool_name=tool_name, args=args)
    evidence_id = "evidence:" + signature
    if evidence_id in graph.evidence_by_id:
        return graph.evidence_by_id[evidence_id]
    result = registry.execute(tool_name=tool_name, args=args, store=store)
    atom = EvidenceAtom(
        evidence_id=evidence_id,
        query_signature=signature,
        modality=result.modality,
        component_scope=result.component_scope,
        time_window=result.time_window,
        observation=result.observation,
        provenance=result.provenance,
        missing_fields=result.missing_fields,
        reliability_note=result.reliability_note,
    )
    return graph.add_evidence(atom)


def _normalize_final_decision(
    value: Any,
    *,
    case,
    hypotheses,
    graph: EvidenceGraph,
    context_state: Mapping[str, Any],
    recall_pool: Sequence[str] | None = None,
) -> dict[str, Any]:
    raw = dict(value) if isinstance(value, Mapping) else {}
    hypothesis_by_id = {hypothesis.hypothesis_id: hypothesis for hypothesis in hypotheses}
    active_components = {hypothesis.root_component for hypothesis in hypotheses}
    hid = str(raw.get("hypothesis_id") or "").strip()
    component = str(raw.get("root_component") or "").strip()
    rescue_note = ""
    if component and component not in active_components:
        # The LLM nominated a component outside the active hypothesis set.
        # Record this as an explicit, audited global_rescue rather than
        # silently accepting it, so downstream review can distinguish
        # in-hypothesis reasoning from out-of-set jumps.  Keep the
        # nomination (it can rescue a case) but flag it for audit.
        rescue_note = (
            "global_rescue: model nominated a component outside the active "
            "hypothesis set; retained for audit but flagged as out-of-set."
        )
    if component not in set(case.components):
        if hid in hypothesis_by_id:
            component = hypothesis_by_id[hid].root_component
            rescue_note = ""
        else:
            component = _leading_component(
                context_state,
                [hypothesis.root_component for hypothesis in hypotheses],
            )
            rescue_note = ""
    if not hid or hid not in hypothesis_by_id:
        for hypothesis in hypotheses:
            if hypothesis.root_component == component:
                hid = hypothesis.hypothesis_id
                break
    hypothesis = hypothesis_by_id.get(hid)
    reason = str(raw.get("reason_family") or (hypothesis.reason_family if hypothesis else "unknown"))
    onset = _onset_pair(raw.get("onset_interval"), hypothesis)
    evidence_ids = [
        str(item)
        for item in raw.get("evidence_ids", [])
        if str(item) in graph.evidence_by_id
    ]
    rationale = str(raw.get("rationale") or "").strip()
    if not rationale:
        rationale = _fallback_final_rationale(
            context_state=context_state,
            component=component,
        )
    if rescue_note and rescue_note not in rationale:
        rationale = f"{rationale} [{rescue_note}]"
    uncertainties = [
        str(item)
        for item in raw.get("uncertainties", [])
        if str(item).strip()
    ]
    if not uncertainties:
        uncertainties = _fallback_uncertainties(
            context_state=context_state,
            component=component,
        )
    result = {
        "hypothesis_id": hid or "unknown",
        "root_component": component,
        "reason_family": reason,
        "onset_interval": onset,
        "evidence_ids": evidence_ids,
        "rationale": rationale,
        "uncertainties": uncertainties,
    }
    if rescue_note:
        result["global_rescue"] = True
        result["outside_hypothesis_set"] = True
    return result


def _fallback_final_rationale(
    *,
    context_state: Mapping[str, Any],
    component: str,
) -> str:
    belief = _belief_for_component(context_state, component)
    if not belief:
        return "Fallback final decision from continuous belief state."
    parts = []
    judgment = str(belief.get("source_vs_symptom_judgment") or "").strip()
    if judgment:
        parts.append(judgment)
    support = [
        str(item)
        for item in belief.get("supporting_factors", [])
        if str(item).strip()
    ][:3]
    contra = [
        str(item)
        for item in belief.get("contradicting_factors", [])
        if str(item).strip()
    ][:2]
    if support:
        parts.append("Support: " + "; ".join(support))
    if contra:
        parts.append("Caveats: " + "; ".join(contra))
    return " ".join(parts) or "Fallback final decision from continuous belief state."


def _fallback_uncertainties(
    *,
    context_state: Mapping[str, Any],
    component: str,
) -> list[str]:
    belief = _belief_for_component(context_state, component)
    if not belief:
        return []
    missing = [
        str(item)
        for item in belief.get("missing_information", [])
        if str(item).strip()
    ][:3]
    return missing


def _belief_for_component(
    context_state: Mapping[str, Any],
    component: str,
) -> Mapping[str, Any] | None:
    belief = context_state.get("belief_state", [])
    if not isinstance(belief, list):
        return None
    for position in ("leading", "plausible", "weakened", "unlikely"):
        for item in belief:
            if (
                isinstance(item, Mapping)
                and item.get("root_component") == component
                and item.get("position") == position
            ):
                return item
    return None


def _leading_component(context_state: Mapping[str, Any], components: Sequence[str]) -> str:
    belief = context_state.get("belief_state", [])
    if isinstance(belief, list):
        for position in ("leading", "plausible", "weakened"):
            for item in belief:
                if (
                    isinstance(item, Mapping)
                    and item.get("position") == position
                    and item.get("root_component") in components
                ):
                    return str(item["root_component"])
    return str(components[0])


def _symptom_components(*, case, components: Sequence[str], exclude: str) -> list[str]:
    values: list[str] = []
    for component in list(case.entry_components) + [obs.component for obs in case.observations]:
        if component != exclude and component in components and component not in values:
            values.append(component)
    if not values:
        values = [component for component in components if component != exclude]
    return values[:4] or [component for component in components if component != exclude][:1]


def _window(value: Any, case) -> list[float]:
    if (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in value)
    ):
        start, end = float(value[0]), float(value[1])
        if start <= end:
            return [start, end]
    return [max(0.0, float(case.event_time) - 300.0), float(case.event_time) + 900.0]


def _component_list(value: Any, *, fallback: Sequence[str]) -> list[str]:
    if isinstance(value, (list, tuple)):
        items = [str(item).strip() for item in value if str(item).strip()]
    else:
        items = []
    if not items:
        items = [str(item) for item in fallback]
    deduped: list[str] = []
    for item in items:
        if item not in deduped:
            deduped.append(item)
    return deduped


def _onset_pair(value: Any, hypothesis) -> list[float] | None:
    if (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in value)
    ):
        return [float(value[0]), float(value[1])]
    if hypothesis is not None:
        return [float(hypothesis.onset_interval[0]), float(hypothesis.onset_interval[1])]
    return None


def _serialize_evidence(atom: EvidenceAtom) -> Mapping[str, Any]:
    return {
        "evidence_id": atom.evidence_id,
        "query_signature": atom.query_signature,
        "modality": atom.modality,
        "component_scope": list(atom.component_scope),
        "time_window": list(atom.time_window),
        "observation": _truncate_jsonable(atom.observation, max_chars=12000),
        "missing_fields": list(atom.missing_fields),
        "reliability_note": atom.reliability_note,
    }


def _truncate_jsonable(value: Any, *, max_chars: int) -> Any:
    jsonable = _to_jsonable(value)
    text = json.dumps(jsonable, ensure_ascii=False, sort_keys=True)
    if len(text) <= max_chars:
        return jsonable
    return {"truncated_json": text[:max_chars] + "..."}


def _to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {
            field.name: _to_jsonable(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_to_jsonable(item) for item in value]
    return repr(value)


if __name__ == "__main__":
    raise SystemExit(main())
