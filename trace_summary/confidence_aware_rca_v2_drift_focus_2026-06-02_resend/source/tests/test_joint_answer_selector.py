import pandas as pd

from refute_b_v2.default_rules import default_rule_set
from refute_b_v2.joint_answer_selector import JointAnswerSelector, JointSelectorConfig, _fmt_ts
from refute_b_v2.llm_clients import JSONToolLLMClient
from refute_b_v2.rule_engine import QueryResult, RuleEngine


class BaselineResult:
    def __init__(self, is_anomalous, deviation):
        self.is_anomalous = is_anomalous
        self.deviation = deviation


class FakeBaseline:
    def is_anomalous(self, cmdb_id, kpi_name, value, threshold="p99"):
        return BaselineResult(float(value) > 0, float(value))


class FakeEvidence:
    def __init__(self, metric_df, log_df, trace_summary, modal_status):
        self.metric_df = metric_df
        self.log_df = log_df
        self.trace_summary = trace_summary or {}
        self.modal_status = dict(modal_status)

    def is_container_kpi_anomalous(self, service, bucket):
        rows = self.metric_df[self.metric_df["cmdb_id"] == str(service)]
        if bucket == "cpu" and not rows.empty:
            return QueryResult(True, f"{service} cpu high", 5.0)
        return QueryResult(False, f"no {bucket} for {service}", 0.0)

    def is_node_kpi_anomalous(self, service, bucket):
        return QueryResult(False, f"no node {bucket}", 0.0)

    def does_match_log_keyword(self, service, keywords):
        return QueryResult(False, "no log hit", 0.0)

    def is_specialty_kpi_anomalous(self, service, bucket):
        return QueryResult(False, f"no specialty {bucket}", 0.0)

    def has_slow_trace_edge(self, service):
        return QueryResult(False, "no slow edge", 0.0, unavailable=True)

    def has_trace_edge_count_drop(self, service):
        return QueryResult(False, "no edge drop", 0.0, unavailable=True)

    def is_trace_first_anomalous_service(self, service):
        return QueryResult(False, "no first service", 0.0, unavailable=True)

    def has_trace_propagation_role(self, service, roles):
        return QueryResult(False, "no propagation role", 0.0, unavailable=True)


class ScriptedClient(JSONToolLLMClient):
    def __init__(self, replies):
        super().__init__("scripted")
        self.replies = list(replies)

    def complete(self, system_prompt, user_prompt):
        return self.replies.pop(0)


def evidence_factory(metric_df, log_df, trace_summary, modal_status):
    return FakeEvidence(metric_df, log_df, trace_summary, modal_status)


def test_joint_selector_uses_component_conditional_time_anchor():
    metric_df = pd.DataFrame([
        {"timestamp": 60, "cmdb_id": "A", "kpi_name": "JVM_CPULoad", "value": 4.0},
        {"timestamp": 120, "cmdb_id": "A", "kpi_name": "JVM_CPULoad", "value": 20.0},
        {"timestamp": 60, "cmdb_id": "B", "kpi_name": "JVM_CPULoad", "value": 1.0},
    ])
    selector = JointAnswerSelector(
        RuleEngine(default_rule_set()),
        FakeBaseline(),
        {"containers": {"A": {}, "B": {}}},
        ["A", "B"],
        evidence_factory=evidence_factory,
    )

    result = selector.select(
        case_id="case-time",
        metric_df=metric_df,
        log_df=pd.DataFrame(columns=["timestamp", "cmdb_id", "value"]),
        trace_summary=None,
        modal_status={"metric": "present", "log": "empty_window", "trace": "unloaded"},
        failure_count=1,
        window_start_ts=0,
    )

    pred = result.prediction["1"]
    assert pred["root cause component"] == "A"
    assert pred["root cause reason"] in {"high JVM CPU load", "high CPU usage"}
    assert result.decisions[0].time_anchor.timestamp == 120


def test_joint_selector_accepts_only_structured_llm_allowed_decision():
    metric_df = pd.DataFrame([
        {"timestamp": 60, "cmdb_id": "A", "kpi_name": "JVM_CPULoad", "value": 8.0},
    ])
    client = ScriptedClient([
        '{"action":"final","final":{"root cause occurrence datetime":"'
        + _fmt_ts(60)
        + '","root cause component":"A","root cause reason":"high JVM CPU load","confidence":"HIGH","evidence_refs":["metric.cpu"]}}'
    ])
    selector = JointAnswerSelector(
        RuleEngine(default_rule_set()),
        FakeBaseline(),
        {"containers": {"A": {}, "B": {}}},
        ["A", "B"],
        llm_client=client,
        evidence_factory=evidence_factory,
        config=JointSelectorConfig(llm_enabled=True, force_llm=True),
    )

    result = selector.select(
        case_id="case-llm",
        metric_df=metric_df,
        log_df=pd.DataFrame(columns=["timestamp", "cmdb_id", "value"]),
        trace_summary=None,
        modal_status={"metric": "present", "log": "empty_window", "trace": "unloaded"},
        failure_count=1,
        window_start_ts=0,
    )

    assert result.decisions[0].structured.source == "llm_structured"
    assert result.prediction["1"]["root cause component"] == "A"
