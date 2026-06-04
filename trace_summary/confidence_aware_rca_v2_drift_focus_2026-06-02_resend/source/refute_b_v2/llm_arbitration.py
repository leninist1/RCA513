"""LLM arbitration assembly for Scheme B.

The LLM is not allowed to invent telemetry. It receives the evidence matrix and
may call small read-only tools that expose candidate cards, blind spots, and
trace summaries. The deterministic prediction path remains available when the
LLM is disabled.
"""
from __future__ import annotations

from typing import Any, Sequence

from refute_b_v2.evidence_matrix import EvidenceMatrixDecision, decide_matrix
from refute_b_v2.llm_tool_loop import LLMClient, LLMToolLoop, ToolObservation
from refute_b_v2.rule_engine import CandidateRuleResult


def needs_llm_arbitration(matrix: EvidenceMatrixDecision, force: bool = False) -> bool:
    if force:
        return True
    return (
        len(matrix.high_suspicion) != 1
        or bool(matrix.ambiguous)
        or bool(matrix.data_blind_spots)
    )


def run_llm_arbitration(
    case_id: str,
    ranked: Sequence[CandidateRuleResult],
    client: LLMClient | None,
    *,
    context: dict[str, Any] | None = None,
    similar_index: Any | None = None,
    force: bool = False,
    max_calls: int = 8,
) -> dict[str, Any]:
    matrix = decide_matrix(list(ranked))
    matrix_dict = matrix.to_dict()
    context = dict(context or {})
    if client is None or not needs_llm_arbitration(matrix, force=force):
        return {
            "called": False,
            "reason": "disabled_or_not_ambiguous",
            "matrix": matrix_dict,
        }

    def query_candidate_evidence(args: dict) -> dict:
        service = str(args.get("service", ""))
        reason = str(args.get("reason", ""))
        matches = []
        for row in matrix_dict["high_suspicion"] + matrix_dict["ambiguous"] + matrix_dict["low_suspicion"]:
            candidate = row.get("candidate", {})
            if service and candidate.get("service") != service:
                continue
            if reason and candidate.get("reason") != reason:
                continue
            matches.append(row)
        return {"matches": matches[:5], "match_count": len(matches)}

    def query_blind_spots(args: dict) -> dict:
        return {
            "data_blind_spots": matrix_dict["data_blind_spots"],
            "modal_status": context.get("modal_status", {}),
        }

    def query_trace_summary(args: dict) -> dict:
        summary = context.get("trace_summary") or {}
        service = str(args.get("service", ""))
        if not service:
            return {"trace_summary": summary}
        return {
            "trace_status": summary.get("trace_status"),
            "service": service,
            "events": summary.get("events", {}),
            "edge_stats": [
                edge for edge in summary.get("edge_stats", [])
                if str(edge.get("src")) == service or str(edge.get("dst")) == service
            ][:20],
        }

    def query_similar_cases(args: dict) -> dict:
        signature = args.get("signature") or context.get("signature") or {}
        if similar_index is None:
            return {"error": "similar_index_not_available"}
        nearest = similar_index.nearest(signature, k=int(args.get("k", 5) or 5))
        return {
            "neighbors": [
                {"case_id": row.case_id, "similarity": row.similarity, "metadata": row.metadata}
                for row in nearest
            ]
        }

    loop = LLMToolLoop(
        {
            "query_candidate_evidence": query_candidate_evidence,
            "query_blind_spots": query_blind_spots,
            "query_trace_summary": query_trace_summary,
            "query_similar_cases": query_similar_cases,
        },
        client=client,
        max_calls=max_calls,
    )
    state = loop.run(case_id, {**matrix_dict, "context": context})
    return {
        "called": True,
        "status": state.status,
        "recommendation": state.recommendation,
        "observations": [_observation_to_dict(obs) for obs in state.observations],
        "matrix": matrix_dict,
    }


def _observation_to_dict(obs: ToolObservation) -> dict[str, Any]:
    return {
        "tool": obs.call.name,
        "args": dict(obs.call.args),
        "result": dict(obs.result),
    }
