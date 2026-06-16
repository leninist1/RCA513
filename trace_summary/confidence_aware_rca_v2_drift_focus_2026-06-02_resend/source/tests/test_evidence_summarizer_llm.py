from types import SimpleNamespace

import pandas as pd

from refute_b_v2_d32.evidence_summarizer import build_summary_cards


class _Baseline:
    def is_anomalous(self, component, kpi_name, value, threshold="p99"):
        return SimpleNamespace(is_anomalous=True, deviation=12.0)


def test_summary_card_separates_missing_and_competing_evidence_for_llm():
    metric_df = pd.DataFrame([
        {"timestamp": 1000, "cmdb_id": "os_010", "kpi_name": "cpu_usage", "value": 99.0},
    ])
    log_df = pd.DataFrame(columns=["timestamp", "cmdb_id", "value"])
    cards = build_summary_cards(
        case_id="query_001",
        metric_df=metric_df,
        log_df=log_df,
        trace_summary=None,
        baseline=_Baseline(),
        modal_status={"metric": "present", "log": "disabled", "trace": "disabled"},
        d32_debug={
            "all_decisions": [
                {
                    "candidate": {"component": "db_004", "reason": "db connection limit"},
                    "support_strength": 0.0,
                    "refute_strength": 0.0,
                    "rebuttal_score": 0.0,
                }
            ],
            "signature": {},
        },
        window_start_ts=1000,
        top_k=1,
    )
    candidate_summary = cards[0]["candidate_summary"]
    counter_types = {item["type"] for item in candidate_summary["counter_evidence_summary"]}
    competing_types = {item["type"] for item in candidate_summary["competing_evidence_summary"]}

    assert "candidate_reason_missing_log_support" not in counter_types
    assert "other_component_metric_signal" not in counter_types
    assert "other_component_metric_signal" in competing_types
    assert candidate_summary["reason_bucket"] == "db_connection"
    assert candidate_summary["canonical_reason"] == "db connection limit"
    assert "db close" in candidate_summary["known_reason_aliases"]
    assert candidate_summary["missing_evidence_summary"][0]["impact"] == "not_used_as_counter_evidence"


def test_summary_card_adds_direct_atoms_and_same_reason_sibling_context():
    metric_df = pd.DataFrame([
        {"timestamp": 1000, "cmdb_id": "docker_001", "kpi_name": "cpu_usage", "value": 99.0},
    ])
    cards = build_summary_cards(
        case_id="query_002",
        metric_df=metric_df,
        log_df=pd.DataFrame(columns=["timestamp", "cmdb_id", "value"]),
        trace_summary=None,
        baseline=_Baseline(),
        modal_status={"metric": "present", "log": "disabled", "trace": "disabled"},
        d32_debug={
            "all_decisions": [
                {
                    "candidate": {"component": "docker_001", "reason": "container CPU load"},
                    "support_strength": 0.0,
                    "refute_strength": 0.0,
                    "rebuttal_score": 0.0,
                },
                {
                    "candidate": {"component": "docker_002", "reason": "CPU fault"},
                    "support_strength": 0.0,
                    "refute_strength": 0.0,
                    "rebuttal_score": 0.0,
                },
            ],
            "signature": {},
        },
        window_start_ts=1000,
        top_k=2,
    )
    first = cards[0]["candidate_summary"]
    second = cards[1]["candidate_summary"]

    assert first["canonical_reason"] == "CPU fault"
    assert first["candidate_direct_evidence_atoms"][0]["directness"] == "component_and_reason"
    assert first["candidate_direct_evidence_atoms"][0]["promotion_eligible"] is True
    assert first["same_reason_sibling_context"]["candidate_vs_best_sibling"] == "strongest"
    assert second["same_reason_sibling_context"]["has_stronger_sibling"] is True
