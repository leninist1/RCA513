import importlib.util
import json
import math
from pathlib import Path
import sys

import pandas as pd


def _load_eval_module():
    source_root = Path(__file__).resolve().parents[1]
    eval_path = source_root / "eval" / "evaluate_portable_task.py"
    spec = importlib.util.spec_from_file_location("evaluate_portable_task", eval_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["evaluate_portable_task"] = module
    spec.loader.exec_module(module)
    return module


def test_eadro_uses_rcl_hr_and_ndcg_without_reason_scoring(tmp_path):
    evaluator = _load_eval_module()
    cases = pd.DataFrame([
        {"case_id": "c1", "root_cause_component": "svc-b", "failure_type": "network latency"},
        {"case_id": "c2", "root_cause_component": "svc-x", "failure_type": "cpu"},
    ])
    predictions = {
        "c1": {"component": "svc-a", "reason": "CPU fault"},
        "c2": {"component": "svc-x", "reason": "CPU fault"},
    }
    debug_rows = [
        {
            "row_id": 0,
            "case_id": "c1",
            "d32_result": {
                "debug": {
                    "all_decisions": [
                        {"candidate": {"component": "svc-a"}},
                        {"candidate": {"component": "svc-b"}},
                    ]
                }
            },
        },
        {
            "row_id": 1,
            "case_id": "c2",
            "d32_result": {
                "debug": {
                    "all_decisions": [
                        {"candidate": {"component": "svc-x"}},
                        {"candidate": {"component": "svc-y"}},
                    ]
                }
            },
        },
    ]

    summary, details = evaluator.evaluate_eadro(cases, predictions, debug_rows, str(tmp_path / "details.csv"))

    assert summary["metric_policy"]["primary_task"] == "Eadro Root Cause Localization"
    assert summary["HR@1"] == 0.5
    assert summary["HR@3"] == 1.0
    assert math.isclose(summary["NDCG@3"], (1 / math.log2(3) + 1.0) / 2)
    assert "fault_type" in details.columns


def test_aiops2021_reports_service_type_and_split_metrics(tmp_path):
    evaluator = _load_eval_module()
    cases = pd.DataFrame([
        {"case_id": "c1", "root_cause_component": "svc-b", "failure_type": "cpu", "data_type": "train"},
        {"case_id": "c2", "root_cause_component": "svc-z", "failure_type": "network latency", "data_type": "test"},
    ])
    predictions = {
        "c1": {"component": "svc-b", "reason": "CPU fault"},
        "c2": {"component": "svc-y", "reason": "CPU fault"},
    }
    debug_rows = [
        {
            "row_id": 0,
            "case_id": "c1",
            "d32_result": {"debug": {"all_decisions": [{"candidate": {"component": "svc-b"}}]}},
        },
        {
            "row_id": 1,
            "case_id": "c2",
            "d32_result": {
                "debug": {
                    "all_decisions": [
                        {"candidate": {"component": "svc-y"}},
                        {"candidate": {"component": "svc-z"}},
                    ]
                }
            },
        },
    ]

    summary, details = evaluator.evaluate_aiops2021(cases, predictions, debug_rows, str(tmp_path / "details.csv"))

    assert summary["metric_policy"]["primary_task"] == "AIOps2021 service localization plus anomaly-type classification"
    assert summary["service_top1_accuracy"] == 0.5
    assert summary["service_HR@3"] == 1.0
    assert summary["anomaly_type_accuracy"] == 0.5
    assert summary["service_type_tuple_accuracy"] == 0.5
    assert summary["by_split"]["train"]["service_type_tuple_accuracy"] == 1.0
    assert summary["by_split"]["test"]["anomaly_type_accuracy"] == 0.0
    assert details["gt_type"].tolist() == ["cpu", "network_latency"]


def test_cli_writes_json_and_details_csv(tmp_path, monkeypatch, capsys):
    evaluator = _load_eval_module()
    cases_csv = tmp_path / "cases.csv"
    pred_csv = tmp_path / "predictions.csv"
    debug_json = tmp_path / "debug.json"
    out_json = tmp_path / "eval.json"
    details_csv = tmp_path / "details.csv"

    pd.DataFrame([
        {"case_id": "c1", "root_cause_component": "svc-a", "failure_type": "cpu"},
    ]).to_csv(cases_csv, index=False)
    pd.DataFrame([
        {
            "row_id": 0,
            "case_id": "c1",
            "prediction": json.dumps({"1": {"root cause component": "svc-a", "root cause reason": "CPU fault"}}),
        }
    ]).to_csv(pred_csv, index=False)
    debug_json.write_text(json.dumps({
        "n": 1,
        "debug": [
            {
                "row_id": 0,
                "case_id": "c1",
                "d32_result": {"debug": {"all_decisions": [{"candidate": {"component": "svc-a"}}]}},
            }
        ],
    }), encoding="utf-8")

    monkeypatch.setattr(sys, "argv", [
        "evaluate_portable_task.py",
        "--dataset", "eadro",
        "--cases-csv", str(cases_csv),
        "--pred", str(pred_csv),
        "--debug-json", str(debug_json),
        "--out-json", str(out_json),
        "--details-csv", str(details_csv),
    ])

    assert evaluator.main() == 0
    captured = json.loads(capsys.readouterr().out)
    assert captured["summary"]["HR@1"] == 1.0
    assert out_json.exists()
    assert pd.read_csv(details_csv)["HR@1"].tolist() == [1.0]
