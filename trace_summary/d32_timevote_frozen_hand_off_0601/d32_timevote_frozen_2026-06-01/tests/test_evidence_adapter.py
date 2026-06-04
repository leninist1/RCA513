from refute_b_v2.evidence_adapter import SummaryBackedEvidence


class BaseEvidence:
    pass


def test_trace_summary_slow_edge_reports_role_and_strength():
    summary = {
        "trace_status": "present",
        "events": {
            "slow_edges": [
                {"src": "A", "dst": "B", "slow_ratio": 4.0},
                {"src": "C", "dst": "A", "slow_ratio": 2.0},
            ]
        },
    }
    evidence = SummaryBackedEvidence(BaseEvidence(), summary)
    result = evidence.has_slow_trace_edge("B")
    assert result.matched
    assert result.strength == 4.0
    assert result.details["role"] == "dst"


def test_trace_summary_missing_is_blind():
    evidence = SummaryBackedEvidence(BaseEvidence(), {})
    result = evidence.has_trace_edge_count_drop("B")
    assert not result.matched
    assert result.unavailable


def test_first_anomalous_service_matches_candidate():
    summary = {
        "trace_status": "present",
        "events": {"first_anomalous_service": "Tomcat01"},
    }
    evidence = SummaryBackedEvidence(BaseEvidence(), summary)
    result = evidence.is_trace_first_anomalous_service("Tomcat01")
    assert result.matched
    assert result.details["first_anomalous_service"] == "Tomcat01"
