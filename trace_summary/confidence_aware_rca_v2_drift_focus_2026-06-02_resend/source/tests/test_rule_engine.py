from refute_b_v2.default_rules import default_rule_set
from refute_b_v2.audit import summarize_rule_results
from refute_b_v2.rule_engine import QueryResult, QueryRegistry, RuleEngine
from refute_b_v2.rules import Candidate, RuleSet, normalize_reason_bucket


class FakeEvidence:
    pass


def test_reason_bucket_splits_network_latency_and_packet_loss():
    assert normalize_reason_bucket("network latency") == "network_latency"
    assert normalize_reason_bucket("network packet loss") == "network_packet_loss"
    assert normalize_reason_bucket("JVM Out of Memory (OOM) Heap") == "jvm_oom"


def test_rule_engine_turns_missing_into_blind_not_refute():
    rules = default_rule_set()
    registry = QueryRegistry()
    registry.register(
        "container_kpi_anomalous",
        lambda evidence, candidate, args: QueryResult(False, "metric missing", unavailable=True),
    )
    engine = RuleEngine(rules, registry)
    result = engine.run_candidate(Candidate("Tomcat01", "high CPU usage"), FakeEvidence())
    assert result.blind_count == 1
    assert result.hard_refute == 0
    assert result.cards[0].polarity == "blind"


def test_rule_engine_supports_cpu_candidate():
    rules = default_rule_set()
    registry = QueryRegistry()
    registry.register(
        "container_kpi_anomalous",
        lambda evidence, candidate, args: QueryResult(True, "cpu high", strength=12.0),
    )
    engine = RuleEngine(rules, registry)
    result = engine.run_candidate(Candidate("Tomcat01", "high CPU usage"), FakeEvidence())
    assert result.hard_support == 1
    assert result.support_strength == 12.0


def test_packet_loss_does_not_use_trace_slow_edge_rule():
    rules = default_rule_set()
    registry = QueryRegistry()
    registry.register(
        "trace_slow_edge",
        lambda evidence, candidate, args: QueryResult(True, "slow edge", strength=100.0),
    )
    registry.register(
        "container_kpi_anomalous",
        lambda evidence, candidate, args: QueryResult(False, "no network kpi", strength=0.0),
    )
    registry.register(
        "log_keyword_match",
        lambda evidence, candidate, args: QueryResult(False, "no retry log", strength=0.0),
    )
    engine = RuleEngine(rules, registry)
    result = engine.run_candidate(Candidate("Tomcat01", "network packet loss"), FakeEvidence())
    assert all(card.rule_id != "network_latency.trace_slow_edge" for card in result.cards)
    assert result.support_strength == 0.0


def test_jvm_oom_log_rule_is_hard_support():
    rules = default_rule_set()
    registry = QueryRegistry()
    registry.register(
        "log_keyword_match",
        lambda evidence, candidate, args: QueryResult(True, "Full GC matched", strength=1.0),
    )
    registry.register(
        "container_kpi_anomalous",
        lambda evidence, candidate, args: QueryResult(True, "heap high", strength=4.0),
    )
    engine = RuleEngine(rules, registry)
    result = engine.run_candidate(Candidate("Tomcat02", "JVM Out of Memory (OOM) Heap"), FakeEvidence())
    assert any(card.rule_id == "memory.jvm_oom_log_required" and card.rule_type == "hard" for card in result.cards)
    assert any(card.rule_id == "memory.jvm_heap_plus_gc_log" for card in result.cards)
    assert result.hard_support == 2


def test_network_latency_first_anomalous_rule_can_support():
    rules = default_rule_set()
    registry = QueryRegistry()
    registry.register(
        "trace_first_anomalous_service",
        lambda evidence, candidate, args: QueryResult(candidate.service == "Tomcat01", "first is Tomcat01", strength=2.0),
    )
    registry.register(
        "trace_slow_edge",
        lambda evidence, candidate, args: QueryResult(False, "no slow edge", strength=0.0),
    )
    registry.register(
        "log_keyword_match",
        lambda evidence, candidate, args: QueryResult(False, "no timeout", strength=0.0),
    )
    engine = RuleEngine(rules, registry)
    result = engine.run_candidate(Candidate("Tomcat01", "network latency"), FakeEvidence())
    assert any(card.rule_id == "network_latency.trace_first_anomalous" for card in result.cards)
    assert result.support_strength == 1.0


def test_default_rules_round_trip_json(tmp_path):
    path = tmp_path / "rules.json"
    default_rule_set().save_json(path)
    loaded = RuleSet.load_json(path)
    assert loaded.version == 2
    assert len(loaded.rules) == len(default_rule_set().rules)
    assert loaded.rules[0].id == "cpu.container_cpu_required"


def test_audit_counts_false_support_and_true_refute():
    rules = default_rule_set()
    registry = QueryRegistry()

    def cpu_query(evidence, candidate, args):
        if candidate.service == "Tomcat01":
            return QueryResult(False, "no cpu evidence", strength=0.0)
        return QueryResult(True, "cpu high", strength=3.0)

    registry.register("container_kpi_anomalous", cpu_query)
    engine = RuleEngine(rules, registry)
    results = [
        engine.run_candidate(Candidate("Tomcat01", "high CPU usage"), FakeEvidence()),
        engine.run_candidate(Candidate("Tomcat02", "high CPU usage"), FakeEvidence()),
    ]
    audit = summarize_rule_results(results, true_service="Tomcat01")
    row = audit["cpu.container_cpu_required"]
    assert row["refute_true_root_count"] == 1
    assert row["support_false_root_count"] == 1
