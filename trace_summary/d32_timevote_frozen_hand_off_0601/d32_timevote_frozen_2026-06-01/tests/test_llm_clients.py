import pytest

from refute_b_v2.llm_clients import (
    ClaudeRelayClient,
    DeepSeekClient,
    JSONToolLLMClient,
    LLMApiError,
    MissingLLMCredentials,
    _extract_json,
)
from refute_b_v2.llm_arbitration import run_llm_arbitration
from refute_b_v2.llm_tool_loop import LLMToolLoop
from refute_b_v2.rule_engine import CandidateRuleResult, RuleEvidenceCard
from refute_b_v2.rules import Candidate


class ScriptedJSONClient(JSONToolLLMClient):
    def __init__(self, replies):
        super().__init__(model="test-json-client")
        self.replies = list(replies)
        self.user_prompts = []

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        self.user_prompts.append(user_prompt)
        return self.replies.pop(0)


def test_json_llm_client_drives_tool_call_then_final():
    client = ScriptedJSONClient([
        '{"action":"tool","tool_name":"query_candidate_evidence","tool_args":{"service":"A","reason":"cpu"}}',
        '{"action":"final","final":"choose A/cpu after checking structured evidence"}',
    ])
    loop = LLMToolLoop(
        {"query_candidate_evidence": lambda args: {"ok": True, "checked": args["service"]}},
        client=client,
    )

    state = loop.run("case-json", {"high_suspicion": [{"candidate": {"service": "A", "reason": "cpu"}}]})

    assert state.status == "final"
    assert state.observations[0].call.name == "query_candidate_evidence"
    assert state.observations[0].result["checked"] == "A"
    assert state.recommendation == "choose A/cpu after checking structured evidence"
    assert "query_candidate_evidence" in client.user_prompts[0]


def test_json_llm_client_blocks_unavailable_tool_requests():
    client = ScriptedJSONClient([
        '{"action":"tool","tool_name":"not_registered","tool_args":{}}',
    ])
    loop = LLMToolLoop({"query_candidate_evidence": lambda args: {"ok": True}}, client=client)

    state = loop.run("case-json", {"high_suspicion": []})

    assert state.status == "final"
    assert state.observations == []
    assert state.recommendation == "Requested unavailable tool: not_registered"


def test_extract_json_accepts_fenced_response():
    parsed = _extract_json('```json\n{"action":"final","final":"ambiguous"}\n```')
    assert parsed["action"] == "final"
    assert parsed["final"] == "ambiguous"


def test_extract_json_accepts_first_valid_object_with_trailing_text():
    parsed = _extract_json('notes before\n{"action":"final","final":"ok"}\n{"ignored": true}')
    assert parsed["action"] == "final"
    assert parsed["final"] == "ok"


def test_extract_json_rejects_non_json_response():
    with pytest.raises(LLMApiError):
        _extract_json("I think service A is suspicious.")


def test_llm_clients_require_credentials(monkeypatch):
    for name in (
        "SHQBB_API_KEY",
        "CLAUDE_API_KEY",
        "ANTHROPIC_API_KEY",
        "REFUTE_ANTHROPIC_API_KEY",
        "DEEPSEEK_API_KEY",
        "REFUTE_DEEPSEEK_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(MissingLLMCredentials):
        ClaudeRelayClient()
    with pytest.raises(MissingLLMCredentials):
        DeepSeekClient()


def test_llm_arbitration_exposes_evidence_matrix_tools():
    result = CandidateRuleResult(
        Candidate("A", "high CPU usage"),
        [RuleEvidenceCard("metric.cpu", "metric", "cpu", "support", 3.0, "cpu high", rule_type="hard")],
    )
    client = ScriptedJSONClient([
        '{"action":"tool","tool_name":"query_candidate_evidence","tool_args":{"service":"A"}}',
        '{"action":"final","final":"A remains plausible after evidence lookup"}',
    ])

    decision = run_llm_arbitration(
        "case-llm",
        [result],
        client,
        context={"modal_status": {"metric": "present"}},
        force=True,
    )

    assert decision["called"]
    assert decision["status"] == "final"
    assert decision["observations"][0]["tool"] == "query_candidate_evidence"
    assert decision["recommendation"] == "A remains plausible after evidence lookup"
