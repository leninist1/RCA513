import json

from refute_b_v2_d32.llm_candidate_judge import (
    LLMJudgeConfig,
    build_judge_input,
    judge_candidate,
    parse_judgment,
)


def _card():
    return {
        "case_id": "query_001",
        "case_summary": {"modality_availability": {"metric": True, "log": False, "trace": True}},
        "candidate_summary": {
            "candidate_rank": 1,
            "component": "A",
            "reason": "high CPU usage",
            "metric_support_summary": [{"component": "A", "kpi_group": "cpu", "severity": "high"}],
            "log_support_summary": [],
            "trace_support_summary": [],
        },
    }


def test_parse_judgment_clamps_and_defaults_missing_fields():
    raw = json.dumps({
        "verdict": "support",
        "support_score": 2.5,
        "refute_score": -1,
        "should_promote": "false",
        "rationale": "x" * 300,
    })
    judgment, parse_ok, error = parse_judgment(raw, rationale_max_chars=20)
    assert parse_ok is True
    assert error is None
    assert judgment["verdict"] == "support"
    assert judgment["support_score"] == 1.0
    assert judgment["refute_score"] == 0.0
    assert judgment["component_consistency"] == 0.0
    assert judgment["should_promote"] is False
    assert len(judgment["rationale"]) == 20


def test_parse_judgment_records_invalid_json():
    judgment, parse_ok, error = parse_judgment("not-json")
    assert parse_ok is False
    assert "json_parse_error" in error
    assert judgment["verdict"] == "uncertain"


def test_parse_judgment_normalizes_invalid_verdict():
    judgment, parse_ok, error = parse_judgment(json.dumps({"verdict": "promote"}))
    assert parse_ok is True
    assert "invalid_verdict" in error
    assert judgment["verdict"] == "uncertain"


def test_judge_input_is_card_and_candidate_only():
    payload = build_judge_input(_card())
    assert set(payload) == {"task", "candidate", "evidence_summary_card"}
    assert payload["task"] == "judge_candidate_root_cause"
    assert payload["candidate"] == {"component": "A", "reason": "high CPU usage"}
    assert payload["evidence_summary_card"] == _card()


def test_judge_candidate_records_configuration_failure_without_raising(monkeypatch):
    monkeypatch.delenv("RCA_LLM_API_KEY", raising=False)
    result = judge_candidate(_card(), LLMJudgeConfig(model="m", base_url="http://localhost:9"))
    assert result["parse_ok"] is False
    assert "missing API key env var" in result["error"]
    assert result["llm_judgment"]["verdict"] == "uncertain"
