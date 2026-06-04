"""LLM tool-loop controller.

The loop is deliberately tool-first: the model may request structured evidence
queries, but final answers must cite tool observations rather than inventing
facts. A deterministic fallback policy is provided for tests and offline runs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol


@dataclass(frozen=True)
class ToolCall:
    name: str
    args: dict


@dataclass(frozen=True)
class ToolObservation:
    call: ToolCall
    result: dict


@dataclass
class ToolLoopState:
    case_id: str
    evidence_matrix: dict
    available_tools: tuple[str, ...] = ()
    observations: list[ToolObservation] = field(default_factory=list)
    status: str = "running"
    recommendation: str = ""


class LLMClient(Protocol):
    def next_tool(self, state: ToolLoopState) -> ToolCall | None:
        ...

    def finalize(self, state: ToolLoopState) -> str:
        ...


ToolFn = Callable[[dict], dict]


class DeterministicLLMPolicy:
    """Offline policy that asks for missing evidence before finalizing."""

    def next_tool(self, state: ToolLoopState) -> ToolCall | None:
        matrix = state.evidence_matrix
        if matrix.get("data_blind_spots") and not _called(state, "query_blind_spots"):
            return ToolCall("query_blind_spots", {"case_id": state.case_id})
        high = matrix.get("high_suspicion", [])
        if len(high) >= 2 and not _called(state, "query_candidate_evidence"):
            cand = high[0].get("candidate", {})
            return ToolCall("query_candidate_evidence", cand)
        return None

    def finalize(self, state: ToolLoopState) -> str:
        high = state.evidence_matrix.get("high_suspicion", [])
        if not high:
            return "No high-confidence candidate; return blind/ambiguous decision with requested evidence gaps."
        cand = high[0].get("candidate", {})
        return f"Prefer {cand.get('service')}/{cand.get('reason')} based on structured evidence; see evidence cards."


class LLMToolLoop:
    def __init__(self, tools: dict[str, ToolFn], client: LLMClient | None = None, max_calls: int = 8):
        self.tools = tools
        self.client = client or DeterministicLLMPolicy()
        self.max_calls = max_calls

    def run(self, case_id: str, evidence_matrix: dict) -> ToolLoopState:
        state = ToolLoopState(
            case_id=case_id,
            evidence_matrix=evidence_matrix,
            available_tools=tuple(sorted(self.tools)),
        )
        for _ in range(self.max_calls):
            call = self.client.next_tool(state)
            if call is None:
                state.status = "final"
                state.recommendation = self.client.finalize(state)
                return state
            if call.name not in self.tools:
                state.observations.append(ToolObservation(call, {"error": "tool_not_registered"}))
                state.status = "blocked"
                state.recommendation = f"Tool not registered: {call.name}"
                return state
            result = self.tools[call.name](call.args)
            state.observations.append(ToolObservation(call, result))
            if result.get("decisive_blind_spot"):
                state.status = "blind_spot"
                state.recommendation = "Decisive blind spot; request additional telemetry before RCA decision."
                return state
        state.status = "ambiguous"
        state.recommendation = self.client.finalize(state)
        return state


def _called(state: ToolLoopState, name: str) -> bool:
    return any(obs.call.name == name for obs in state.observations)
