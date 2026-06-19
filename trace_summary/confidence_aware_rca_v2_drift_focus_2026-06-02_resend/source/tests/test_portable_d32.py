from refute.src.baseline_distributions import BaselineStats, BaselineStore
from refute_b_v2_d32.adaptive_baseline import build_adaptive_baseline
from refute_b_v2_d32.entity_roles import ROLE_DATABASE, ROLE_HOST, infer_entity_role
from refute_b_v2_d32.portable_adapters import aiops2021_tabular_adapter
from refute_b_v2_d32.portable_ontology import infer_kpi_buckets
from refute_b_v2_d32.portable_schema import normalize_log_frame, normalize_metric_frame

import importlib.util
import json
import pandas as pd
from pathlib import Path
import sys


def test_portable_schema_normalizes_common_metric_and_log_aliases():
    metrics = normalize_metric_frame(pd.DataFrame([
        {"time": 1_700_000_000_000, "service": "svc-a", "metric": "cpu_usage", "val": "91.5"},
    ]))
    logs = normalize_log_frame(pd.DataFrame([
        {"ts": 1_700_000_001_000, "instance": "svc-a", "message": "timeout talking to db"},
    ]))

    assert metrics.to_dict("records") == [
        {"timestamp": 1_700_000_000, "cmdb_id": "svc-a", "kpi_name": "cpu_usage", "value": 91.5},
    ]
    assert logs.to_dict("records") == [
        {"timestamp": 1_700_000_001, "cmdb_id": "svc-a", "value": "timeout talking to db"},
    ]


def test_tabular_adapter_keeps_labels_out_of_d32_inputs():
    cases = pd.DataFrame([
        {"case_id": "c1", "start_ts": 100, "end_ts": 200, "root_cause": "svc-b"},
    ])
    metrics = pd.DataFrame([
        {"timestamp": 90, "cmdb_id": "svc-a", "kpi_name": "cpu", "value": 1.0},
        {"timestamp": 120, "cmdb_id": "svc-b", "kpi_name": "cpu", "value": 99.0},
    ])
    adapter = aiops2021_tabular_adapter(cases=cases, metrics=metrics)
    incident = next(iter(adapter.iter_incidents()))

    assert incident.labels == {"root_cause": "svc-b"}
    d32_inputs = incident.d32_inputs()
    assert "labels" not in d32_inputs
    assert d32_inputs["metric_df"]["cmdb_id"].tolist() == ["svc-b"]


def test_adaptive_baseline_preserves_eligible_historical_answers():
    historical = BaselineStore([
        BaselineStats("svc-a", "cpu", 20, 5.0, 1.0, 1.0, 2.0, 8.0, 9.0, True),
    ])
    adaptive = build_adaptive_baseline(
        pd.DataFrame(columns=["timestamp", "cmdb_id", "kpi_name", "value"]),
        historical=historical,
        window_start_ts=100,
    )

    result = adaptive.is_anomalous("svc-a", "cpu", 10.0)
    assert result.is_anomalous is True
    assert result.reason == "above_threshold"
    assert result.deviation == 5.0


def test_adaptive_baseline_uses_pre_fault_when_historical_missing():
    rows = [
        {"timestamp": ts, "cmdb_id": "svc-a", "kpi_name": "cpu", "value": float(ts)}
        for ts in range(10)
    ]
    adaptive = build_adaptive_baseline(
        pd.DataFrame(rows),
        historical=None,
        window_start_ts=100,
    )

    result = adaptive.is_anomalous("svc-a", "cpu", 50.0)
    assert result.is_anomalous is True
    assert result.reason.startswith("pre_fault:")
    assert adaptive.reliability_report(pd.DataFrame(rows))[0]["source"] == "pre_fault"


def test_entity_role_prefers_explicit_metadata_over_legacy_prefix():
    role = infer_entity_role(
        "docker_001",
        kpi_names=["mysql_session_count"],
        topology_node={"type": "database"},
    )
    host = infer_entity_role("opaque-node", kpi_names=["system.cpu.util"])

    assert role.role == ROLE_DATABASE
    assert role.confidence == 1.0
    assert host.role == ROLE_HOST


def test_entity_role_explicit_topology_is_not_overridden_by_many_kpis():
    role = infer_entity_role(
        "IG01",
        kpi_names=[
            "Mysql-MySQL_3306_Connections",
            "Tomcat-Sessions_7441--UOCP_SESSIONActiveCounter",
            "redis-Redis_6379_used_memory",
        ] * 20,
        topology_node={"role": "gateway"},
    )

    assert role.role == "gateway"
    assert role.signals == ("explicit:gateway",)


def test_portable_ontology_recognizes_common_new_dataset_kpis():
    assert infer_kpi_buckets("rx_bytes") == {"network_latency"}
    assert infer_kpi_buckets("Container_x_NetworkRxBytes") == {"network_latency"}
    assert "jvm_oom" in infer_kpi_buckets("JVM_Memory_HeapMemoryUsed")
    assert infer_kpi_buckets("disk_free_space") == {"filesystem"}


def test_portable_runner_smoke_outputs_predictions_without_label_leakage(tmp_path, monkeypatch):
    source_root = Path(__file__).resolve().parents[1]
    runner_path = source_root / "eval" / "run_portable_d32.py"
    spec = importlib.util.spec_from_file_location("run_portable_d32", runner_path)
    runner = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["run_portable_d32"] = runner
    spec.loader.exec_module(runner)

    cases_csv = tmp_path / "cases.csv"
    metrics_csv = tmp_path / "metrics.csv"
    out_csv = tmp_path / "predictions.csv"
    debug_json = tmp_path / "debug.json"
    reliability_json = tmp_path / "baseline_reliability.json"
    audit_jsonl = tmp_path / "audit.jsonl"

    pd.DataFrame([
        {"case_id": "c1", "start_ts": 100, "end_ts": 130, "root_cause": "svc-a"},
    ]).to_csv(cases_csv, index=False)
    pd.DataFrame(
        [
            *[
                {"timestamp": ts, "cmdb_id": "svc-a", "kpi_name": "cpu_usage", "value": 10.0 + (ts % 3)}
                for ts in range(90, 100)
            ],
            {"timestamp": 100, "cmdb_id": "svc-a", "kpi_name": "cpu_usage", "value": 100.0},
            {"timestamp": 110, "cmdb_id": "svc-a", "kpi_name": "cpu_usage", "value": 101.0},
        ]
    ).to_csv(metrics_csv, index=False)

    monkeypatch.setattr(sys, "argv", [
        "run_portable_d32.py",
        "--dataset", "aiops2021",
        "--cases-csv", str(cases_csv),
        "--metrics-csv", str(metrics_csv),
        "--modalities", "metric",
        "--pre-baseline-sec", "20",
        "--rules", str(source_root / "knowledge" / "refutation_rules_v2.json"),
        "--out", str(out_csv),
        "--debug-json", str(debug_json),
        "--baseline-reliability-json", str(reliability_json),
        "--input-audit-jsonl", str(audit_jsonl),
    ])

    assert runner.main() == 0
    prediction_rows = pd.read_csv(out_csv)
    debug = json.loads(debug_json.read_text(encoding="utf-8"))
    audit = json.loads(audit_jsonl.read_text(encoding="utf-8").splitlines()[0])

    assert prediction_rows["case_id"].tolist() == ["c1"]
    assert debug["n"] == 1
    assert "labels" not in json.dumps(debug["debug"][0], ensure_ascii=False)
    assert audit["labels_present"] == ["root_cause"]
    assert reliability_json.exists()


def test_train_portable_reason_classifier_writes_model_artifacts(tmp_path, monkeypatch):
    source_root = Path(__file__).resolve().parents[1]
    trainer_path = source_root / "eval" / "train_portable_reason_classifier.py"
    spec = importlib.util.spec_from_file_location("train_portable_reason_classifier", trainer_path)
    trainer = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["train_portable_reason_classifier"] = trainer
    spec.loader.exec_module(trainer)

    cases_csv = tmp_path / "cases.csv"
    metrics_csv = tmp_path / "metrics.csv"
    out_json = tmp_path / "portable_reason_classifier.json"
    summary_json = tmp_path / "summary.json"

    pd.DataFrame([
        {"case_id": "c1", "start_ts": 100, "end_ts": 130, "root_cause": "svc-a", "failure_type": "cpu", "data_type": "train"},
        {"case_id": "c2", "start_ts": 200, "end_ts": 230, "root_cause": "svc-b", "failure_type": "memory", "data_type": "train"},
        {"case_id": "c3", "start_ts": 300, "end_ts": 330, "root_cause": "svc-c", "failure_type": "cpu", "data_type": "test"},
    ]).to_csv(cases_csv, index=False)
    rows = []
    for start, svc, kpi, normal, abnormal in [
        (100, "svc-a", "cpu_usage", 10.0, 95.0),
        (200, "svc-b", "memory_usage", 12.0, 97.0),
        (300, "svc-c", "cpu_usage", 11.0, 90.0),
    ]:
        for ts in range(start - 12, start):
            rows.append({"timestamp": ts, "cmdb_id": svc, "kpi_name": kpi, "value": normal + (ts % 2)})
        rows.append({"timestamp": start, "cmdb_id": svc, "kpi_name": kpi, "value": abnormal})
        rows.append({"timestamp": start + 10, "cmdb_id": svc, "kpi_name": kpi, "value": abnormal + 1.0})
    pd.DataFrame(rows).to_csv(metrics_csv, index=False)

    monkeypatch.setattr(sys, "argv", [
        "train_portable_reason_classifier.py",
        "--dataset", "aiops2021",
        "--cases-csv", str(cases_csv),
        "--metrics-csv", str(metrics_csv),
        "--pre-baseline-sec", "20",
        "--train-split-column", "data_type",
        "--train-split-value", "train",
        "--eval-split-value", "test",
        "--out", str(out_json),
        "--summary-json", str(summary_json),
    ])

    assert trainer.main() == 0
    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    assert out_json.exists()
    assert out_json.with_suffix(".pkl").exists()
    assert summary["n_train"] == 2
    assert summary["n_eval"] == 1


def test_build_portable_d32_knowledge_writes_training_split_artifact(tmp_path, monkeypatch):
    source_root = Path(__file__).resolve().parents[1]
    builder_path = source_root / "eval" / "build_portable_d32_knowledge.py"
    spec = importlib.util.spec_from_file_location("build_portable_d32_knowledge", builder_path)
    builder = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["build_portable_d32_knowledge"] = builder
    spec.loader.exec_module(builder)

    cases_csv = tmp_path / "cases.csv"
    metrics_csv = tmp_path / "metrics.csv"
    out_json = tmp_path / "d32_knowledge_train.json"

    pd.DataFrame([
        {"case_id": "c1", "start_ts": 100, "end_ts": 130, "root_cause_component": "svc-a", "failure_type": "cpu", "data_type": "train"},
        {"case_id": "c2", "start_ts": 200, "end_ts": 230, "root_cause_component": "svc-b", "failure_type": "network latency", "data_type": "train"},
        {"case_id": "c3", "start_ts": 300, "end_ts": 330, "root_cause_component": "svc-c", "failure_type": "memory", "data_type": "test"},
    ]).to_csv(cases_csv, index=False)
    rows = []
    for start, svc, kpi in [
        (100, "svc-a", "cpu_usage"),
        (200, "svc-b", "rx_bytes"),
        (300, "svc-c", "memory_usage"),
    ]:
        for ts in range(start - 12, start):
            rows.append({"timestamp": ts, "cmdb_id": svc, "kpi_name": kpi, "value": 10.0 + (ts % 2)})
        rows.append({"timestamp": start, "cmdb_id": svc, "kpi_name": kpi, "value": 90.0})
        rows.append({"timestamp": start + 10, "cmdb_id": svc, "kpi_name": kpi, "value": 91.0})
    pd.DataFrame(rows).to_csv(metrics_csv, index=False)

    monkeypatch.setattr(sys, "argv", [
        "build_portable_d32_knowledge.py",
        "--dataset", "aiops2021",
        "--cases-csv", str(cases_csv),
        "--metrics-csv", str(metrics_csv),
        "--modalities", "metric",
        "--pre-baseline-sec", "20",
        "--train-split-column", "data_type",
        "--train-split-value", "train",
        "--disable-rule-validation",
        "--out", str(out_json),
    ])

    assert builder.main() == 0
    knowledge = json.loads(out_json.read_text(encoding="utf-8"))
    assert knowledge["metadata"]["n_cases"] == 2
    assert knowledge["metadata"]["split"] == {"column": "data_type", "value": "train", "exclude_value": None}
    assert len(knowledge["clusters"]) >= 1
    assert {row["case_id"] for row in knowledge["cases"]} == {"c1", "c2"}


def test_portable_multimodal_builder_normalizes_trace_summary_edges():
    source_root = Path(__file__).resolve().parents[1]
    builder_path = source_root / "eval" / "build_portable_multimodal_inputs.py"
    spec = importlib.util.spec_from_file_location("build_portable_multimodal_inputs", builder_path)
    builder = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["build_portable_multimodal_inputs"] = builder
    spec.loader.exec_module(builder)

    spans = pd.DataFrame([
        {"timestamp": 100, "cmdb_id": "gateway", "trace_id": "t1", "span_id": "root", "parent_id": "", "duration": 10.0},
        {"timestamp": 101, "cmdb_id": "svc-fast", "trace_id": "t1", "span_id": "s1", "parent_id": "root", "duration": 10.0},
        {"timestamp": 102, "cmdb_id": "svc-slow", "trace_id": "t1", "span_id": "s2", "parent_id": "root", "duration": 100.0},
        {"timestamp": 103, "cmdb_id": "svc-slow", "trace_id": "t2", "span_id": "s3", "parent_id": "missing", "duration": 1.0},
    ])

    summary = builder.build_trace_summary_from_spans(
        spans,
        case_id="c1",
        start_ts=100,
        end_ts=130,
        slow_ratio_threshold=1.5,
        top_k_edges=5,
        dataset="unit",
    )

    assert summary["trace_status"] == "present"
    assert summary["service_stats"]["svc-slow"]["span_count"] == 2
    assert summary["events"]["slow_edges"][0]["src"] == "gateway"
    assert summary["events"]["slow_edges"][0]["dst"] == "svc-slow"
    assert summary["events"]["first_anomalous_service"] == "gateway"
    assert all(edge["src"] != "" for edge in summary["edge_stats"])


def test_portable_multimodal_builder_maps_eadro_fault_source_to_telemetry_dir(tmp_path):
    source_root = Path(__file__).resolve().parents[1]
    builder_path = source_root / "eval" / "build_portable_multimodal_inputs.py"
    spec = importlib.util.spec_from_file_location("build_portable_multimodal_inputs", builder_path)
    builder = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["build_portable_multimodal_inputs"] = builder
    spec.loader.exec_module(builder)

    telemetry_dir = tmp_path / "extracted_sn" / "data" / "SN.2022-04-17T181245D2022-04-17T183616"
    telemetry_dir.mkdir(parents=True)

    found = builder._eadro_telemetry_dir(
        tmp_path,
        "SN.fault-2022-04-17T181245D2022-04-17T183616.json",
    )

    assert found == telemetry_dir


def test_portable_multimodal_builder_normalizes_mixed_epoch_units():
    source_root = Path(__file__).resolve().parents[1]
    builder_path = source_root / "eval" / "build_portable_multimodal_inputs.py"
    spec = importlib.util.spec_from_file_location("build_portable_multimodal_inputs", builder_path)
    builder = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["build_portable_multimodal_inputs"] = builder
    spec.loader.exec_module(builder)

    values = builder._normalize_epoch_seconds(pd.Series([1614787199628, 1616428798]))

    assert values.tolist() == [1614787200.0, 1616428798.0]
