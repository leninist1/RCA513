import pandas as pd

from refute_b_v2_d32.layer1 import D32Knowledge
from refute_b_v2_d32.layer2 import D32RefutationPipeline
from refute_b_v2_d32.schema import reason_bucket
from refute_b_v2_d32.signature import build_case_signature, flatten_signature
from refute_b_v2_d32.time_anchor import CandidateTimeAnchorer
from refute_b_v2_d32.schema import RootCandidate


class FakeAnomaly:
    def __init__(self, is_anomalous=True, deviation=5.0):
        self.is_anomalous = is_anomalous
        self.deviation = deviation
        self.reason = "above_threshold"


class FakeBaseline:
    def is_anomalous(self, cmdb_id, kpi_name, value, threshold="p99"):
        return FakeAnomaly(True, float(value))


def test_reason_bucket_normalization():
    assert reason_bucket("network packet loss") == "network_packet_loss"
    assert reason_bucket("high disk I/O read usage") == "disk_io"


def test_signature_contains_metric_and_trace_features():
    metric = pd.DataFrame([
        {"timestamp": 60, "cmdb_id": "Tomcat01", "kpi_name": "CPUUtil", "value": 9.0},
    ])
    sig = build_case_signature(
        "case",
        metric,
        pd.DataFrame(columns=["timestamp", "cmdb_id", "value"]),
        {"trace_status": "present", "events": {"slow_edges": [{"src": "Tomcat01", "dst": "Mysql01", "slow_ratio": 2.0}]}},
        FakeBaseline(),
        {"metric": "present", "log": "empty_window", "trace": "present"},
    )
    assert any(svc["service"] == "Tomcat01" for svc in sig["services"])
    assert "metric:cpu" in sig["dominant_evidence_types"]
    assert "trace:network_latency" in sig["dominant_evidence_types"]
    features = flatten_signature(sig)
    assert not any(key.startswith("svc:") for key in features)
    assert any(key.startswith("type:app:") for key in features)
    assert any(key.startswith("role:application:") for key in features)


def test_d32_pipeline_uses_cluster_candidate_space():
    knowledge = D32Knowledge({
        "clusters": [{
            "cluster_id": "cluster_0",
            "size": 3,
            "members": ["train_1", "train_2"],
            "centroid": {"metric:cpu:support": 1.0, "type:unknown:metric:cpu:support": 1.0},
            "reason_prior": [{"reason": "high CPU usage", "count": 3, "weight": 1.0}],
        }],
        "mined_rules": [],
    }, min_cluster_similarity=0.0)
    rules = {
        "rules": [{
            "id": "cpu.container_required",
            "enabled": True,
            "reason_buckets": ["cpu"],
            "evidence_query": "container_kpi_anomalous",
            "query_args": {"kpi_bucket": "cpu"},
            "support_if": "matched",
            "refute_if": "not_matched",
            "confidence": 1.0,
            "missing_policy": "blind",
        }]
    }
    metric = pd.DataFrame([
        {"timestamp": 60, "cmdb_id": "A", "kpi_name": "CPUUtil", "value": 9.0},
    ])
    pipeline = D32RefutationPipeline(
        knowledge,
        rules,
        FakeBaseline(),
        {"containers": {"A": {"node_proxy": "A"}}, "nodes": {"A": {"hosted_containers": ["A"]}}},
        ["A", "B"],
    )
    result = pipeline.select(
        case_id="case",
        metric_df=metric,
        log_df=pd.DataFrame(columns=["timestamp", "cmdb_id", "value"]),
        trace_summary=None,
        modal_status={"metric": "present", "log": "empty_window", "trace": "disabled"},
        failure_count=1,
        window_start_ts=60,
    )
    assert result.prediction["1"]["root cause component"] == "A"
    assert result.debug["matched_clusters"][0]["cluster_id"] == "cluster_0"
    assert result.high_suspicion[0].candidate.source == "layer1_fault_cluster"
    assert result.high_suspicion[0].support_strength > 0
    assert result.debug["selected_time_anchors"][0]["time_anchor"]["policy"] == "metric_sustained_cross_vote"


def test_candidate_time_anchor_prefers_sustained_onset_over_late_peak():
    metric = pd.DataFrame([
        {"timestamp": 60, "cmdb_id": "A", "kpi_name": "CPUUtil", "value": 20.0},
        {"timestamp": 120, "cmdb_id": "A", "kpi_name": "CPUIdle", "value": 20.0},
        {"timestamp": 300, "cmdb_id": "A", "kpi_name": "CPUUtil", "value": 20.0},
    ])
    anchorer = CandidateTimeAnchorer(FakeBaseline(), {"containers": {}, "nodes": {}})
    anchor = anchorer.anchor(
        RootCandidate("A", "high CPU usage"),
        metric,
        pd.DataFrame(columns=["timestamp", "cmdb_id", "value"]),
        None,
        0,
    )
    assert anchor.timestamp == 60
    assert any(vote.kind == "sustained_onset" for vote in anchor.votes)


def test_candidate_time_anchor_uses_trace_first_seen_for_network():
    anchorer = CandidateTimeAnchorer(FakeBaseline(), {"containers": {}, "nodes": {}})
    anchor = anchorer.anchor(
        RootCandidate("A", "network latency"),
        pd.DataFrame(columns=["timestamp", "cmdb_id", "kpi_name", "value"]),
        pd.DataFrame(columns=["timestamp", "cmdb_id", "value"]),
        {"trace_status": "present", "events": {"slow_edges": [{"src": "A", "dst": "B", "slow_ratio": 5.0, "first_timestamp": 180}]}},
        0,
    )
    assert anchor.timestamp == 180
    assert anchor.policy == "network_trace_first_seen_with_metric_vote"
