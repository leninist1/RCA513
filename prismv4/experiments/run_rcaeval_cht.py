#!/usr/bin/env python3
"""Run PRISM-CHT on RCAEval cases.

Default mode uses the real LLM policy/provider path.  Use
``--offline-baseline`` only for local debugging when no Provider key is
available.
"""

from __future__ import annotations

import argparse
from dataclasses import fields, is_dataclass
from enum import Enum
import json
import os
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from prismv4.experiments.cht_policy_repair import (
    RepairingChallengerPolicy,
    RepairingLeadPolicy,
)
from prismv4.experiments.rcaeval_adapter import discover_re3_cases, load_re3_case
from prismv4.prism_cht.action_gate import ActionGate
from prismv4.prism_cht.assessment_gate import EvidenceAssessmentGate
from prismv4.prism_cht.challenge_gate import ChallengeReviewGate
from prismv4.prism_cht.challenger_controller import ChallengerController
from prismv4.prism_cht.diagnostic_policy import (
    DeterministicChallengerDiagnosticPolicy,
    DeterministicLeadDiagnosticPolicy,
    build_hypotheses_from_case,
)
from prismv4.prism_cht.evidence_graph import EvidenceGraph
from prismv4.prism_cht.executor import InvestigationExecutor
from prismv4.prism_cht.final_verifier import FinalVerifier
from prismv4.prism_cht.http_transport import UrllibHttpTransport
from prismv4.prism_cht.lead_controller import (
    LeadTournamentController,
    NominationRejectedError,
    TournamentBudgetExhaustedError,
)
from prismv4.prism_cht.llm_json import parse_json_object
from prismv4.prism_cht.llm_types import ModelClient, ModelMessage, ModelRequest, ModelResponse
from prismv4.prism_cht.llm_policy import (
    StructuredLLMChallengerPolicy,
    StructuredLLMLeadPolicy,
)
from prismv4.prism_cht.llm_audit import AuditedModelClient, LLMIORecorder
from prismv4.prism_cht.openai_compatible_client import OpenAICompatibleChatModelClient
from prismv4.prism_cht.provider_config import (
    load_openai_compatible_config_from_mapping,
)
from prismv4.prism_cht.tool_registry import build_default_tool_registry


DEFAULT_RE3_ROOT = "/home/dell2/RCA513/ysj/dataset/RCAEval/RE3"


class RetryingModelClient:
    """Retry transient provider failures while preserving audited attempts."""

    def __init__(
        self,
        *,
        inner: ModelClient,
        max_attempts: int = 3,
        initial_delay_seconds: float = 2.0,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        self._inner = inner
        self._max_attempts = max_attempts
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
    parser = argparse.ArgumentParser(description="PRISM-CHT RCAEval runner")
    parser.add_argument("--data-root", default=DEFAULT_RE3_ROOT)
    parser.add_argument("--system", default="RE3-OB")
    parser.add_argument("--max-cases", type=int, default=5)
    parser.add_argument("--max-hypotheses", type=int, default=5)
    parser.add_argument("--max-rounds", type=int, default=4)
    parser.add_argument("--offline-baseline", action="store_true")
    parser.add_argument(
        "--output",
        default="prismv4/results/prism_cht/rcaeval_cht_results.json",
    )
    parser.add_argument(
        "--llm-io-output",
        default="",
        help="JSONL path for full-fidelity LLM request/response records.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    case_dirs = discover_re3_cases(
        args.data_root,
        system=args.system,
        limit=args.max_cases if args.max_cases > 0 else None,
    )
    if not case_dirs:
        raise SystemExit("no RCAEval cases discovered")

    client = None
    if not args.offline_baseline:
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
        audited_client = AuditedModelClient(
            inner=provider_client,
            recorder=LLMIORecorder(jsonl_path=llm_io_output),
        )
        client = RetryingModelClient(
            inner=audited_client,
            max_attempts=int(os.environ.get("PRISM_CHT_MODEL_RETRIES", "3")),
        )

    results = []
    t0 = time.time()
    for case_dir in case_dirs:
        started = time.time()
        loaded = load_re3_case(case_dir, top_k=args.max_hypotheses)
        try:
            result = run_case(
                loaded=loaded,
                max_hypotheses=args.max_hypotheses,
                max_rounds=args.max_rounds,
                client=client,
                offline_baseline=args.offline_baseline,
            )
            predicted = result["predicted_component"]
            hit = predicted == loaded.expected_component
            result.update(
                {
                    "case_id": loaded.case.case_id,
                    "expected_component": loaded.expected_component,
                    "hit": hit,
                    "elapsed_sec": round(time.time() - started, 3),
                    "error": None,
                }
            )
        except Exception as exc:
            result = {
                "case_id": loaded.case.case_id,
                "expected_component": loaded.expected_component,
                "predicted_component": None,
                "hit": False,
                "elapsed_sec": round(time.time() - started, 3),
                "error": f"{type(exc).__name__}: {exc}",
            }
        results.append(result)
        print(
            f"{loaded.case.case_id}: predicted={result.get('predicted_component')} "
            f"expected={loaded.expected_component} hit={result['hit']} "
            f"error={result.get('error')}",
            flush=True,
        )

    total = len(results)
    top1 = sum(1 for item in results if item.get("hit"))
    summary = {
        "dataset": "RCAEval",
        "system": args.system,
        "mode": "offline-baseline" if args.offline_baseline else "llm-provider",
        "total_cases": total,
        "top1_accuracy": top1 / total if total else 0.0,
        "top1_hits": top1,
        "elapsed_sec": round(time.time() - t0, 3),
        "results": results,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        json.dumps({k: v for k, v in summary.items() if k != "results"}, indent=2),
        flush=True,
    )
    print(f"wrote {output}", flush=True)
    return 0


def run_case(
    *,
    loaded,
    max_hypotheses: int,
    max_rounds: int,
    client,
    offline_baseline: bool,
) -> dict[str, Any]:
    bundle = build_hypotheses_from_case(
        loaded.case,
        max_hypotheses=max_hypotheses,
    )
    graph = EvidenceGraph()
    gate = ActionGate()
    registry = build_default_tool_registry()
    assessment_gate = EvidenceAssessmentGate()
    executor = InvestigationExecutor(
        gate=gate,
        registry=registry,
        graph=graph,
        store=loaded.store,
    )
    lead_controller = LeadTournamentController(
        executor=executor,
        assessment_gate=assessment_gate,
        graph=graph,
        max_rounds=max_rounds,
    )

    if offline_baseline:
        lead_policy = DeterministicLeadDiagnosticPolicy(
            case=loaded.case,
            observation_by_hypothesis_id=bundle.observation_by_hypothesis_id,
        )
        challenger_policy = DeterministicChallengerDiagnosticPolicy()
    else:
        lead_policy = RepairingLeadPolicy(
            inner=StructuredLLMLeadPolicy(client=client, graph=graph),
            case=loaded.case,
        )
        challenger_policy = RepairingChallengerPolicy(
            inner=StructuredLLMChallengerPolicy(client=client, graph=graph),
            case=loaded.case,
        )

    try:
        lead_result = lead_controller.run(
            initial_hypotheses=list(bundle.hypotheses),
            policy=lead_policy,
        )
    except (TournamentBudgetExhaustedError, NominationRejectedError) as exc:
        best_effort = finalize_best_effort_with_llm(
            client=client,
            case=loaded.case,
            graph=graph,
            lead_snapshot=lead_controller.build_current_snapshot(),
            lead_result=None,
            challenge_result=None,
            terminal_status="lead_best_effort_after_blocked_nomination",
            terminal_reason=f"{type(exc).__name__}: {exc}",
        )
        return {
            "status": "best_effort_final",
            "hypothesis_id": best_effort["hypothesis_id"],
            "predicted_component": best_effort["root_component"],
            "reason_family": best_effort["reason_family"],
            "onset_interval": best_effort["onset_interval"],
            "rounds_completed": lead_controller.build_current_snapshot().round_index,
            "challenge_status": None,
            "evidence_count": len(graph.evidence_by_id),
            "referenced_evidence_ids": best_effort["supporting_evidence_ids"],
            "best_effort_rationale": best_effort["rationale"],
            "best_effort_uncertainties": best_effort["uncertainties"],
            "best_effort_terminal_reason": best_effort["terminal_reason"],
        }

    challenge_controller = ChallengerController(
        executor=executor,
        assessment_gate=assessment_gate,
        challenge_gate=ChallengeReviewGate(),
        graph=graph,
    )
    try:
        challenge_result = challenge_controller.run(
            lead_result=lead_result,
            hypotheses=graph.hypotheses_by_id,
            policy=challenger_policy,
        )
    except Exception as exc:
        best_effort = finalize_best_effort_with_llm(
            client=client,
            case=loaded.case,
            graph=graph,
            lead_snapshot=lead_controller.build_current_snapshot(),
            lead_result=lead_result,
            challenge_result=None,
            terminal_status="challenge_best_effort_after_error",
            terminal_reason=f"{type(exc).__name__}: {exc}",
        )
        return {
            "status": "best_effort_final",
            "hypothesis_id": best_effort["hypothesis_id"],
            "predicted_component": best_effort["root_component"],
            "reason_family": best_effort["reason_family"],
            "onset_interval": best_effort["onset_interval"],
            "rounds_completed": lead_result.rounds_completed,
            "challenge_status": "challenge_error",
            "evidence_count": len(graph.evidence_by_id),
            "referenced_evidence_ids": best_effort["supporting_evidence_ids"],
            "best_effort_rationale": best_effort["rationale"],
            "best_effort_uncertainties": best_effort["uncertainties"],
            "best_effort_terminal_reason": best_effort["terminal_reason"],
        }
    if challenge_result.status != "survived_challenge":
        best_effort = finalize_best_effort_with_llm(
            client=client,
            case=loaded.case,
            graph=graph,
            lead_snapshot=lead_controller.build_current_snapshot(),
            lead_result=lead_result,
            challenge_result=challenge_result,
            terminal_status=challenge_result.status,
            terminal_reason=(
                "challenge did not produce survived_challenge; "
                "forcing the model to choose the most likely current root cause"
            ),
        )
        return {
            "status": "best_effort_final",
            "hypothesis_id": best_effort["hypothesis_id"],
            "predicted_component": best_effort["root_component"],
            "reason_family": best_effort["reason_family"],
            "onset_interval": best_effort["onset_interval"],
            "rounds_completed": lead_result.rounds_completed,
            "challenge_status": challenge_result.status,
            "evidence_count": len(graph.evidence_by_id),
            "referenced_evidence_ids": best_effort["supporting_evidence_ids"],
            "best_effort_rationale": best_effort["rationale"],
            "best_effort_uncertainties": best_effort["uncertainties"],
            "best_effort_terminal_reason": best_effort["terminal_reason"],
        }
    final = FinalVerifier().finalize(
        lead_result=lead_result,
        challenge_result=challenge_result,
        hypotheses=graph.hypotheses_by_id,
        graph=graph,
    )
    return {
        "status": final.status,
        "hypothesis_id": final.hypothesis_id,
        "predicted_component": final.root_component,
        "reason_family": final.reason_family,
        "onset_interval": list(final.onset_interval),
        "rounds_completed": lead_result.rounds_completed,
        "challenge_status": challenge_result.status,
        "evidence_count": len(graph.evidence_by_id),
        "referenced_evidence_ids": list(final.referenced_evidence_ids),
    }


def finalize_best_effort_with_llm(
    *,
    client,
    case,
    graph: EvidenceGraph,
    lead_snapshot,
    lead_result,
    challenge_result,
    terminal_status: str,
    terminal_reason: str,
) -> dict[str, Any]:
    """Force a final RCA guess from the current reasoning state.

    This is intentionally not a tournament gate.  It preserves all LLM I/O via
    the audited client and asks for the most likely answer even when the strict
    Lead/Challenger protocol could not finalize.
    """
    if client is None:
        return _fallback_best_effort_from_graph(
            graph=graph,
            terminal_status=terminal_status,
            terminal_reason=terminal_reason,
        )

    context = {
        "case": _case_digest(case),
        "terminal_status": terminal_status,
        "terminal_reason": terminal_reason,
        "lead_snapshot": _lead_snapshot_digest(lead_snapshot),
        "lead_result": _to_jsonable(lead_result) if lead_result is not None else None,
        "challenge_result": (
            _challenge_result_digest(challenge_result)
            if challenge_result is not None
            else None
        ),
        "evidence_catalog": _evidence_catalog_digest(graph),
    }
    system = (
        "You are the final best-effort RCA decision maker. The strict "
        "tournament budget or guardrail has ended, but you MUST still choose "
        "one most likely root cause from the current reasoning state.\n"
        "Rules:\n"
        "1. Output exactly one JSON object and no markdown.\n"
        "2. Choose root_component from the provided case.components exactly.\n"
        "3. Prefer the initiating faulty component and local failure mechanism, "
        "not the loudest downstream symptom.\n"
        "4. Use all lead/challenge feedback, including contradicted, weakened, "
        "inconclusive, and missing evidence.\n"
        "5. Do not request more exploration. The budget is exhausted or the "
        "flow has ended. Give the best current answer.\n"
        "6. supporting_evidence_ids may include positive, contradictory, or "
        "challenge evidence that materially shaped the decision."
    )
    user = (
        "Return JSON with exactly these fields:\n"
        "{\n"
        '  "hypothesis_id": "ID of the closest hypothesis, or unknown",\n'
        '  "root_component": "component name from case.components",\n'
        '  "reason_family": "short mechanism family",\n'
        '  "onset_interval": [start_number, end_number],\n'
        '  "supporting_evidence_ids": ["evidence ids used"],\n'
        '  "rationale": "concise explanation of why this is most likely",\n'
        '  "uncertainties": ["remaining ambiguity or weak points"]\n'
        "}\n\n"
        "Current reasoning state:\n"
        f"{json.dumps(context, ensure_ascii=False, indent=2)}"
    )
    response = client.complete(
        request=ModelRequest(
            purpose="best_effort_final_rca_decision",
            messages=(
                ModelMessage(role="system", content=system),
                ModelMessage(role="user", content=user),
            ),
            attempt_index=0,
        )
    )
    parsed = parse_json_object(response.content)
    return _normalize_best_effort_response(
        parsed,
        graph=graph,
        case_components=case.components,
        terminal_status=terminal_status,
        terminal_reason=terminal_reason,
    )


def _normalize_best_effort_response(
    value: Mapping[str, Any],
    *,
    graph: EvidenceGraph,
    case_components: Sequence[str],
    terminal_status: str,
    terminal_reason: str,
) -> dict[str, Any]:
    hypothesis_id = _string_or_unknown(value.get("hypothesis_id"))
    root_component = _string_or_unknown(value.get("root_component"))
    if root_component not in set(case_components):
        fallback = graph.hypotheses_by_id.get(hypothesis_id)
        if fallback is not None:
            root_component = fallback.root_component
    reason_family = _string_or_unknown(value.get("reason_family"))
    onset_interval = _number_pair_or_default(value.get("onset_interval"), graph, hypothesis_id)
    evidence_ids = tuple(
        eid
        for eid in _string_list(value.get("supporting_evidence_ids"))
        if eid in graph.evidence_by_id
    )
    rationale = _string_or_unknown(value.get("rationale"))
    uncertainties = tuple(_string_list(value.get("uncertainties")))
    return {
        "hypothesis_id": hypothesis_id,
        "root_component": root_component,
        "reason_family": reason_family,
        "onset_interval": onset_interval,
        "supporting_evidence_ids": list(evidence_ids),
        "rationale": rationale,
        "uncertainties": list(uncertainties),
        "terminal_status": terminal_status,
        "terminal_reason": terminal_reason,
    }


def _fallback_best_effort_from_graph(
    *,
    graph: EvidenceGraph,
    terminal_status: str,
    terminal_reason: str,
) -> dict[str, Any]:
    ranked = sorted(
        graph.hypotheses_by_id.values(),
        key=lambda h: (
            -len(h.supporting_evidence_ids),
            len(h.contradicting_evidence_ids),
            h.hypothesis_id,
        ),
    )
    if not ranked:
        return {
            "hypothesis_id": "unknown",
            "root_component": "unknown",
            "reason_family": "unknown",
            "onset_interval": None,
            "supporting_evidence_ids": [],
            "rationale": "No hypotheses were available for fallback finalization.",
            "uncertainties": [terminal_reason],
            "terminal_status": terminal_status,
            "terminal_reason": terminal_reason,
        }
    h = ranked[0]
    return {
        "hypothesis_id": h.hypothesis_id,
        "root_component": h.root_component,
        "reason_family": h.reason_family,
        "onset_interval": list(h.onset_interval),
        "supporting_evidence_ids": list(h.supporting_evidence_ids),
        "rationale": "Offline fallback selected the hypothesis with strongest graph support.",
        "uncertainties": [terminal_reason],
        "terminal_status": terminal_status,
        "terminal_reason": terminal_reason,
    }


def _case_digest(case) -> dict[str, Any]:
    return {
        "case_id": case.case_id,
        "dataset_name": case.dataset_name,
        "system_name": case.system_name,
        "event_time": case.event_time,
        "components": list(case.components),
        "entry_components": list(case.entry_components),
        "observations": [_to_jsonable(obs) for obs in case.observations],
        "metadata": _to_jsonable(case.metadata),
    }


def _lead_snapshot_digest(snapshot) -> dict[str, Any]:
    return {
        "round_index": snapshot.round_index,
        "max_rounds": snapshot.max_rounds,
        "remaining_rounds": snapshot.remaining_rounds,
        "decision_mode": snapshot.decision_mode,
        "hypotheses": [_to_jsonable(h) for h in snapshot.hypotheses],
        "evidence_ids": list(snapshot.evidence_ids),
        "round_trace": [
            {
                "round_index": step.round_index,
                "action": {
                    "action_id": step.action.action_id,
                    "tool_name": step.action.tool_name,
                    "target_hypothesis_ids": list(step.action.target_hypothesis_ids),
                    "question": step.action.question,
                    "args": _to_jsonable(step.action.args),
                    "expected_outcomes": _to_jsonable(step.action.expected_outcomes),
                    "why_discriminative": step.action.why_discriminative,
                },
                "evidence_id": step.evidence_id,
                "assessment": _to_jsonable(step.assessment),
            }
            for step in snapshot.audit_steps
        ],
    }


def _challenge_result_digest(challenge_result) -> dict[str, Any]:
    return {
        "status": challenge_result.status,
        "nominated_hypothesis_id": challenge_result.nominated_hypothesis_id,
        "verdict": _to_jsonable(challenge_result.verdict),
        "challenge_id": challenge_result.challenge_id,
        "evidence_id": challenge_result.evidence_id,
        "audit_step": _to_jsonable(challenge_result.audit_step),
    }


def _evidence_catalog_digest(graph: EvidenceGraph) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for evidence_id, evidence in sorted(graph.evidence_by_id.items()):
        items.append(
            {
                "evidence_id": evidence_id,
                "modality": evidence.modality,
                "component_scope": list(evidence.component_scope),
                "time_window": list(evidence.time_window),
                "observation": _truncate_jsonable(evidence.observation),
                "missing_fields": list(evidence.missing_fields),
                "reliability_note": evidence.reliability_note,
                "supports": sorted(
                    hid
                    for hid, ids in graph.support_edges.items()
                    if evidence_id in ids
                ),
                "contradicts": sorted(
                    hid
                    for hid, ids in graph.contradiction_edges.items()
                    if evidence_id in ids
                ),
            }
        )
    return items


def _truncate_jsonable(value: Any, *, max_chars: int = 3000) -> Any:
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
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_to_jsonable(v) for v in value]
    return repr(value)


def _string_or_unknown(value: Any) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return "unknown"


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]


def _number_pair_or_default(
    value: Any,
    graph: EvidenceGraph,
    hypothesis_id: str,
) -> list[float] | None:
    if (
        isinstance(value, list)
        and len(value) == 2
        and all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in value)
    ):
        return [float(value[0]), float(value[1])]
    h = graph.hypotheses_by_id.get(hypothesis_id)
    if h is not None:
        return [float(h.onset_interval[0]), float(h.onset_interval[1])]
    return None


if __name__ == "__main__":
    raise SystemExit(main())
