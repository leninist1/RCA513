import json

import pandas as pd

from refute_b_v2_d32.evidence_summarizer import EvidenceSummaryLimits, build_summary_cards


class FakeAnomaly:
    def __init__(self, is_anomalous=True, deviation=5.0):
        self.is_anomalous = is_anomalous
        self.deviation = deviation
        self.reason = "above_threshold"


class FakeBaseline:
    def is_anomalous(self, cmdb_id, kpi_name, value, threshold="p99"):
        return FakeAnomaly(True, float(value))


def _debug():
    return {
        "signature": {
            "services": [{
                "service": "A",
                "metric": {"cpu": {"state": "support", "strength": 8.0, "intensity": 3}},
                "log": {"network_latency": {"state": "support", "strength": 5.0, "intensity": 2}},
                "trace": {},
                "topology": {},
            }],
            "dominant_evidence_types": ["metric:cpu", "log:network_latency"],
        },
        "reason_posterior": {"cpu": 2.0, "network_latency": 1.0},
        "selected_time_anchors": [{"time_anchor": {"timestamp": 1000}}],
        "all_decisions": [
            {
                "candidate": {"component": "A", "reason": "high CPU usage", "reason_bucket": "cpu"},
                "support_strength": 2.0,
                "refute_strength": 0.0,
                "confidence": "MEDIUM",
            },
            {
                "candidate": {"component": "B", "reason": "network latency", "reason_bucket": "network_latency"},
                "support_strength": 1.0,
                "refute_strength": 0.0,
                "confidence": "LOW",
            },
        ],
    }


def test_summary_cards_are_bounded_and_json_serializable():
    metric = pd.DataFrame([
        {"timestamp": 1000 + i * 60, "cmdb_id": "A", "kpi_name": f"CPUUtil{i}", "value": 10.0 + i}
        for i in range(8)
    ])
    log_text = "timeout " + ("x" * 500)
    log = pd.DataFrame([
        {"timestamp": 1000, "cmdb_id": "A", "value": log_text},
        {"timestamp": 1060, "cmdb_id": "A", "value": "connection retry failed"},
    ])
    trace = {
        "trace_status": "present",
        "events": {
            "slow_edges": [{"src": "A", "dst": f"S{i}", "slow_ratio": 10.0 + i} for i in range(8)],
            "first_anomalous_service": "A",
        },
    }
    cards = build_summary_cards(
        case_id="query_001",
        metric_df=metric,
        log_df=log,
        trace_summary=trace,
        baseline=FakeBaseline(),
        modal_status={"metric": "present", "log": "present", "trace": "present"},
        d32_debug=_debug(),
        window_start_ts=1000,
        top_k=1,
        limits=EvidenceSummaryLimits(max_metric_patterns=2, max_log_patterns=1, max_trace_edges=2, max_counter_evidence=1),
    )
    assert len(cards) == 1
    summary = cards[0]["candidate_summary"]
    assert len(summary["metric_support_summary"]) <= 2
    assert len(summary["log_support_summary"]) <= 1
    assert len(summary["trace_support_summary"]) <= 2
    assert len(summary["counter_evidence_summary"]) <= 1
    assert len(summary["log_support_summary"][0]["short_example"]) <= 200
    serialized = json.dumps(cards[0])
    assert '"_values"' not in serialized
    assert '"_timestamps"' not in serialized
    assert '"spans"' not in serialized


def test_missing_trace_is_not_automatic_counter_evidence():
    metric = pd.DataFrame([
        {"timestamp": 1000, "cmdb_id": "A", "kpi_name": "CPUUtil", "value": 10.0},
    ])
    cards = build_summary_cards(
        case_id="query_001",
        metric_df=metric,
        log_df=pd.DataFrame(columns=["timestamp", "cmdb_id", "value"]),
        trace_summary=None,
        baseline=FakeBaseline(),
        modal_status={"metric": "present", "log": "empty_window", "trace": "disabled"},
        d32_debug=_debug(),
        window_start_ts=1000,
        top_k=1,
        limits=EvidenceSummaryLimits(),
    )
    summary = cards[0]["candidate_summary"]
    assert any(item["modality"] == "trace" for item in summary["missing_evidence_summary"])
    assert not any(item["type"] == "trace_signal_not_centered_on_candidate" for item in summary["counter_evidence_summary"])
