import pandas as pd

from refute_b_v2.background_suppression import AnomalyEvent, BackgroundCalibrator, component_scores
from refute_b_v2.candidate_generation import CandidateGenerator
from refute_b_v2.confidence_semantics import explain_confidence
from refute_b_v2.cross_service_baseline import cross_service_scores
from refute_b_v2.dataset_adapter import BankAdapter, MarketAdapter, build_topology
from refute_b_v2.evidence_cluster import SignatureIndex, threshold_clusters
from refute_b_v2.evidence_matrix import decide_matrix
from refute_b_v2.iterative_refutation import run_iterative_refutation
from refute_b_v2.llm_tool_loop import LLMToolLoop
from refute_b_v2.reason_competition import ReasonAdjustmentConfig, ReasonScoreCalibrator, compete_reasons
from refute_b_v2.rule_engine import QueryResult, QueryRegistry, RuleEngine
from refute_b_v2.rule_mining import mine_rule_suggestions
from refute_b_v2.rules import Candidate, RefutationRule, RuleSet
from refute_b_v2.time_anchor import TimeAnchorPolicy, component_time_anchors
from refute_b_v2.trace_propagation import propagation_roles, trace_evidence_for_service


def test_dataset_adapter_builds_bank_and_market_topology():
    bank = build_topology([
        ("Tomcat01", "OSLinux-OSLinux_MEMORY_used"),
        ("Tomcat01", "JVM_CPULoad"),
    ], BankAdapter())
    assert bank["containers"]["Tomcat01"]["node_kpis"] == ["OSLinux-OSLinux_MEMORY_used"]
    assert bank["containers"]["Tomcat01"]["container_kpis"] == ["JVM_CPULoad"]

    market = build_topology([("node-1.service-A", "CPU")], MarketAdapter())
    assert market["containers"]["service-A"]["node_proxy"] == "node-1"
    assert market["nodes"]["node-1"]["hosted_containers"] == ["service-A"]


def test_cross_service_baseline_scores_peer_outlier():
    df = pd.DataFrame([
        {"timestamp": 1, "cmdb_id": f"S{i}", "kpi_name": "CPU", "value": 10}
        for i in range(5)
    ] + [{"timestamp": 1, "cmdb_id": "Root", "kpi_name": "CPU", "value": 50}])
    scored = cross_service_scores(df)
    root = scored[scored["cmdb_id"] == "Root"].iloc[0]
    assert root["eligible"]
    assert root["cross_z"] > 0


def test_background_suppression_clips_and_calibrates():
    calibrator = BackgroundCalibrator()
    calibrator.add("Noisy", "cpu", 1, 10.0)
    calibrator.add("Noisy", "cpu", 1, 20.0)
    rows = component_scores([
        AnomalyEvent(3600, "Noisy", "cpu", 100000, "metric"),
        AnomalyEvent(3600, "Root", "cpu", 20, "metric"),
    ], calibrator)
    assert rows[0].score < 100000
    assert rows[0].clipped_sum < rows[0].raw_max


def test_background_suppression_uses_component_specific_baseline():
    calibrator = BackgroundCalibrator()
    # Noisy component has high historical clipped score; Root has low baseline.
    calibrator.add("Noisy", "cpu", 1, 12.0)
    calibrator.add("Noisy", "cpu", 1, 14.0)
    calibrator.add("Root", "cpu", 1, 1.0)
    calibrator.add("Root", "cpu", 1, 1.2)
    rows = component_scores([
        AnomalyEvent(3600, "Noisy", "cpu", 94006, "metric"),
        AnomalyEvent(3600, "Root", "cpu", 34, "metric"),
    ], calibrator)
    assert rows[0].component == "Root"
    assert rows[0].relative_score > rows[1].relative_score


def test_candidate_generator_uses_background_scores():
    gen = CandidateGenerator(top_k=2)
    cands = gen.generate([
        AnomalyEvent(60, "A", "high CPU usage", 9, "metric"),
        AnomalyEvent(120, "B", "network latency", 5, "trace"),
    ])
    assert len(cands) == 2
    assert cands[0].candidate.service in {"A", "B"}
    assert cands[0].candidate_times


def test_signature_index_and_threshold_clusters():
    sig1 = {"case_id": "c1", "services": [{"service": "A", "metric": {"cpu": {"state": "support", "intensity": 3}}}]}
    sig2 = {"case_id": "c2", "services": [{"service": "A", "metric": {"cpu": {"state": "support", "intensity": 2}}}]}
    sig3 = {"case_id": "c3", "services": [{"service": "B", "trace": {"slow": {"state": "support", "intensity": 1}}}]}
    idx = SignatureIndex.from_signatures([sig1, sig2, sig3])
    assert idx.nearest(sig1, k=1)[0].case_id == "c1"
    clusters = threshold_clusters([sig1, sig2, sig3], threshold=0.5)
    assert any(set(group) >= {"c1", "c2"} for group in clusters)


def test_rule_mining_suggests_high_confidence_rule():
    suggestions = mine_rule_suggestions([
        (["cpu_high", "log_error"], "cpu"),
        (["cpu_high"], "cpu"),
        (["cpu_high"], "cpu"),
        (["trace_slow"], "network"),
    ], min_support=3, min_confidence=0.8)
    assert suggestions[0].antecedent == ("cpu_high",)
    assert suggestions[0].target == "cpu"


def test_evidence_matrix_groups_high_low_and_blind():
    rules = RuleSet((RefutationRule("cpu.req", ("cpu",), "q", rule_type="hard", refute_if="not_matched"),))
    reg = QueryRegistry()
    reg.register("q", lambda evidence, candidate, args: QueryResult(candidate.service == "A", "q", strength=2.0))
    engine = RuleEngine(rules, reg)
    decision = decide_matrix([
        engine.run_candidate(Candidate("A", "high CPU usage"), object()),
        engine.run_candidate(Candidate("B", "high CPU usage"), object()),
    ])
    assert decision.high_suspicion[0].result.candidate.service == "A"
    assert decision.low_suspicion[0].result.candidate.service == "B"


def test_iterative_refutation_stops_when_single_high_candidate():
    rules = RuleSet((RefutationRule("cpu.req", ("cpu",), "q", group="round_1_cheap_metric"),))
    reg = QueryRegistry()
    reg.register("q", lambda evidence, candidate, args: QueryResult(candidate.service == "A", "q", strength=2.0))
    states = run_iterative_refutation(RuleEngine(rules, reg), [
        Candidate("A", "high CPU usage"),
        Candidate("B", "high CPU usage"),
    ], object())
    assert states[-1].stop_reason in {"single_high_candidate", "support_gap"}


class FakeBaseline:
    def is_anomalous(self, cmdb_id, kpi_name, value, threshold="p99"):
        class R:
            is_anomalous = True
            deviation = value
        return R()


def test_component_time_anchor_finds_cross_metric_vote():
    df = pd.DataFrame([
        {"timestamp": 60, "cmdb_id": "A", "kpi_name": "cpu", "value": 8},
        {"timestamp": 60, "cmdb_id": "A", "kpi_name": "mem", "value": 7},
        {"timestamp": 120, "cmdb_id": "A", "kpi_name": "cpu", "value": 2},
    ])
    anchors = component_time_anchors(df, "A", FakeBaseline())
    assert anchors[0].timestamp == 60
    assert anchors[0].metric_votes >= 2


def test_time_anchor_policy_makes_selection_explicit():
    df = pd.DataFrame([
        {"timestamp": 60, "cmdb_id": "A", "kpi_name": "cpu", "value": 8},
        {"timestamp": 120, "cmdb_id": "A", "kpi_name": "mem", "value": 30},
    ])
    anchors = component_time_anchors(
        df,
        "A",
        FakeBaseline(),
        policy=TimeAnchorPolicy(strongest_spike_bonus=100.0, cross_metric_vote_weight=0.0),
    )
    assert anchors[0].kind == "strongest_spike"
    assert anchors[0].timestamp == 120


def test_reason_competition_applies_inter_reason_adjustments():
    cal = ReasonScoreCalibrator()
    cal.add("cpu", 1)
    cal.add("cpu", 3)
    rows = compete_reasons(
        {"high CPU usage": 10, "JVM Out of Memory (OOM) Heap": 6},
        {"gc_or_heap_pressure": True},
        cal,
    )
    assert rows[0].bucket in {"jvm_oom", "cpu"}
    assert any("gc" in note or "heap" in note for row in rows for note in row.adjustments)
    assert any(row.adjustment_source == "expert_prior_v0" for row in rows)


def test_reason_competition_coefficients_are_configurable():
    rows = compete_reasons(
        {"network latency": 10, "network packet loss": 9},
        {"tcp_retransmit_or_reset": True},
        adjustment_config=ReasonAdjustmentConfig(source="test_fit", packet_loss_tcp_boost=3.0),
    )
    packet = [row for row in rows if row.bucket == "network_packet_loss"][0]
    assert packet.adjustment_source == "test_fit"
    assert packet.adjusted_score == 27


def test_trace_propagation_roles_and_path():
    summary = {
        "edge_stats": [{"src": "A", "dst": "B", "slow_ratio": 3.0}, {"src": "B", "dst": "C", "count_drop_ratio": 0.6}],
        "events": {"slow_edges": [{"src": "A", "dst": "B", "first_timestamp": 10}], "dropped_edges": [{"src": "B", "dst": "C", "first_timestamp": 12}], "first_anomalous_service": "A"},
    }
    roles = propagation_roles(summary)
    assert roles["A"].role == "upstream_source"
    assert trace_evidence_for_service(summary, "A")["path"] == ["A", "B", "C"]


def test_confidence_semantics_and_llm_tool_loop():
    rules = RuleSet((RefutationRule("cpu.req", ("cpu",), "q", rule_type="hard"),))
    reg = QueryRegistry()
    reg.register("q", lambda evidence, candidate, args: QueryResult(True, "cpu", strength=5.0))
    row = decide_matrix([RuleEngine(rules, reg).run_candidate(Candidate("A", "high CPU usage"), object())]).high_suspicion[0]
    explanation = explain_confidence(row, best_similarity=0.9)
    assert explanation.label == "MEDIUM" or explanation.label == "HIGH"

    loop = LLMToolLoop({"query_candidate_evidence": lambda args: {"ok": True}})
    state = loop.run("case1", {"high_suspicion": [row.to_dict()], "data_blind_spots": []})
    assert state.status == "final"
    assert "Prefer" in state.recommendation
