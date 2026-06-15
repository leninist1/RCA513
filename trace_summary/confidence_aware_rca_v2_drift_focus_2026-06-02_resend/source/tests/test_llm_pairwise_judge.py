import json

from refute_b_v2_d32.llm_pairwise_judge import (
    build_pairwise_requests,
    parse_pairwise_judgment,
    rank_blind_card,
)


def _card(rank, component, reason):
    return {
        "case_id": "query_001",
        "case_summary": {"modality_availability": {"metric": True, "log": False, "trace": True}},
        "candidate_summary": {
            "candidate_rank": rank,
            "component": component,
            "reason": reason,
            "d32_score_summary": {"support_level": "high"},
            "metric_support_summary": [{"component": component, "kpi_group": "cpu", "severity": "high"}],
            "log_support_summary": [],
            "trace_support_summary": [],
            "counter_evidence_summary": [
                {"type": "other_candidate_ranked_higher", "component": "A", "reason": "CPU fault"},
                {"type": "explicit_rule_refute_signal", "severity": "low"},
            ],
        },
    }


def test_rank_blind_card_removes_case_rank_score_and_rank_counter_evidence():
    blinded = rank_blind_card(_card(2, "B", "network delay"))
    assert "case_id" not in blinded
    candidate_summary = blinded["candidate_summary"]
    assert "candidate_rank" not in candidate_summary
    assert "d32_score_summary" not in candidate_summary
    assert candidate_summary["counter_evidence_summary"] == [{"type": "explicit_rule_refute_signal", "severity": "low"}]


def test_pairwise_requests_use_rank_blind_inputs():
    requests = build_pairwise_requests([
        _card(1, "A", "CPU fault"),
        _card(2, "B", "network delay"),
    ])
    assert len(requests) == 1
    assert requests[0]["top1_candidate"] == {"component": "A", "reason": "CPU fault"}
    assert requests[0]["alternative_candidate"] == {"component": "B", "reason": "network delay"}
    assert requests[0]["candidate_a_role"] in {"top1", "alternative"}
    assert requests[0]["candidate_b_role"] in {"top1", "alternative"}
    payload = requests[0]["pairwise_input"]
    assert payload["task"] == "compare_candidate_root_cause_evidence"
    assert set(payload["evidence_summary_cards"]) == {"candidate_a", "candidate_b"}
    assert set([requests[0]["candidate_a_role"], requests[0]["candidate_b_role"]]) == {"top1", "alternative"}
    assert "case_id" not in json.dumps(payload, ensure_ascii=False)
    assert "candidate_rank" not in json.dumps(payload, ensure_ascii=False)
    assert "d32_score_summary" not in json.dumps(payload, ensure_ascii=False)
    assert "other_candidate_ranked_higher" not in json.dumps(payload, ensure_ascii=False)
    assert "top1" not in json.dumps(payload, ensure_ascii=False)
    assert "alternative" not in json.dumps(payload, ensure_ascii=False)


def test_parse_pairwise_judgment_clamps_defaults_and_sanitizes_atoms():
    raw = json.dumps({
        "preferred_candidate": "candidate_b",
        "candidate_a_support_score": -1,
        "candidate_a_refute_score": 2,
        "candidate_b_support_score": 0.9,
        "candidate_b_refute_score": 0.1,
        "relative_margin": 3,
        "evidence_atoms": [
            {"modality": "metric", "candidate": "candidate_b", "effect": "support", "summary": "x" * 300},
            {"modality": "weird", "candidate": "new", "effect": "magic", "summary": "bad"},
        ],
        "rationale": "clear",
    })
    judgment, parse_ok, error = parse_pairwise_judgment(raw)
    assert parse_ok is True
    assert error is None
    assert judgment["preferred_candidate"] == "candidate_b"
    assert judgment["candidate_a_support_score"] == 0.0
    assert judgment["candidate_a_refute_score"] == 1.0
    assert judgment["relative_margin"] == 1.0
    assert judgment["evidence_atoms"][0]["summary"] == "x" * 160
    assert judgment["evidence_atoms"][1]["modality"] == "case"
    assert judgment["evidence_atoms"][1]["candidate"] == "both"
    assert judgment["evidence_atoms"][1]["effect"] == "neutral"


def test_parse_pairwise_judgment_records_invalid_json():
    judgment, parse_ok, error = parse_pairwise_judgment("not-json")
    assert parse_ok is False
    assert "json_parse_error" in error
    assert judgment["preferred_candidate"] == "uncertain"
