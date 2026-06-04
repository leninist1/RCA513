import pandas as pd

from refute_b_v2.evidence_signature import case_signature_from_results
from refute_b_v2.rule_engine import CandidateRuleResult, RuleEvidenceCard
from refute_b_v2.rules import Candidate
from refute_b_v2.trace_summary import summarize_trace_window


def test_case_signature_discretizes_strength_and_tracks_blind_spots():
    result = CandidateRuleResult(
        Candidate("Tomcat01", "high CPU usage"),
        cards=[
            RuleEvidenceCard("cpu.container_cpu_required", "cpu", "container_cpu_required", "support", 12.0, "cpu high", rule_type="hard"),
            RuleEvidenceCard("trace.slow_edge", "trace", "slow_edge", "blind", 0.0, "trace missing", rule_type="soft"),
        ],
    )
    sig = case_signature_from_results("case1", {"metric": "present", "trace": "missing"}, [result])
    data = sig.to_dict()
    assert data["blind_spots"] == ["trace"]
    assert data["services"][0]["metric"]["container_cpu_required"]["intensity"] == 3
    assert data["anomaly_distribution"]["dominant_service"] == "Tomcat01"


def test_trace_summary_detects_slow_and_dropped_edges():
    baseline = pd.DataFrame([
        {"timestamp": 1, "cmdb_id": "A", "parent_id": "", "span_id": "p1", "trace_id": "t1", "duration": 10},
        {"timestamp": 2, "cmdb_id": "B", "parent_id": "p1", "span_id": "c1", "trace_id": "t1", "duration": 10},
        {"timestamp": 3, "cmdb_id": "A", "parent_id": "", "span_id": "p2", "trace_id": "t2", "duration": 10},
        {"timestamp": 4, "cmdb_id": "B", "parent_id": "p2", "span_id": "c2", "trace_id": "t2", "duration": 10},
        {"timestamp": 5, "cmdb_id": "A", "parent_id": "", "span_id": "p3", "trace_id": "t3", "duration": 10},
        {"timestamp": 6, "cmdb_id": "B", "parent_id": "p3", "span_id": "c3", "trace_id": "t3", "duration": 10},
        {"timestamp": 7, "cmdb_id": "A", "parent_id": "", "span_id": "p4", "trace_id": "t4", "duration": 10},
        {"timestamp": 8, "cmdb_id": "B", "parent_id": "p4", "span_id": "c4", "trace_id": "t4", "duration": 10},
    ])
    current = pd.DataFrame([
        {"timestamp": 10, "cmdb_id": "A", "parent_id": "", "span_id": "p5", "trace_id": "t5", "duration": 10},
        {"timestamp": 11, "cmdb_id": "B", "parent_id": "p5", "span_id": "c5", "trace_id": "t5", "duration": 100},
    ])
    summary = summarize_trace_window(current, baseline)
    edge = summary["edge_stats"][0]
    assert edge["src"] == "A"
    assert edge["dst"] == "B"
    assert edge["count_drop_ratio"] == 0.75
    assert edge["slow_ratio"] == 10.0
    assert summary["events"]["slow_edges"]
    assert summary["events"]["dropped_edges"]
