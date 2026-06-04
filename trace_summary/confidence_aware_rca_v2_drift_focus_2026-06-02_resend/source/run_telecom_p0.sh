#!/usr/bin/env bash
set -euo pipefail

BASE=/home/dell2/RCA513/yyx/trace_summary/confidence_aware_rca_v2_drift_focus_2026-06-02_resend/source
DATA_ROOT=/home/dell2/RCA513/ysj/dataset/openrca/Telecom
QUERY_CSV=$DATA_ROOT/query.csv
RECORD_CSV=$DATA_ROOT/record.csv

TRACE_DIR=trace_summaries/telecom_queries_full
BASELINE=knowledge/telecom_baseline_distributions_all_metric_dates.json
GRAPH=knowledge/telecom_node_container_graph.json

MODS=("metric" "metric,trace" "metric,log,trace")
NAMES=("metric" "metric_trace" "metric_log_trace")

cd "$BASE"
export PYTHONPATH=".:..:${PYTHONPATH:-}"
export TZ=Asia/Shanghai

mkdir -p logs knowledge trace_summaries

resume_args=()
if [[ "${RESUME_CHECKPOINT:-0}" == "1" ]]; then
  resume_args=(--resume-checkpoint)
fi

echo "=== Telecom P0 preflight ==="
test -d "$DATA_ROOT"
test -f "$QUERY_CSV"
test -f "$RECORD_CSV"
test -f "$TRACE_DIR/manifest.json"
test -f "$BASELINE"
test -f "$GRAPH"

python3 - "$DATA_ROOT" "$TRACE_DIR" "$BASELINE" "$GRAPH" <<'PY'
import json
import sys
from pathlib import Path

import pandas as pd

from refute.src.data_loader import BankDataPaths, load_log_day, load_metric_day
from refute_b_v2.query_windows import parse_query_window
from refute_b_v2.trace_propagation import trace_evidence_for_service
from refute_b_v2_d32.layer2 import D32RefutationPipeline
from refute_b_v2_d32.schema import reason_bucket

root = Path(sys.argv[1])
trace_dir = Path(sys.argv[2])
baseline_path = Path(sys.argv[3])
graph_path = Path(sys.argv[4])

window = parse_query_window(
    "During the specified time range of April 11, 2020, from 00:00 to 00:30, there was one failure reported."
)
assert window.start.year == 2020 and window.start.month == 4
assert reason_bucket("CPU fault") == "cpu"
assert reason_bucket("network delay") == "network_latency"

paths = BankDataPaths.from_root(root)
metric = load_metric_day(paths, "2020_04_11")
log = load_log_day(paths, "2020_04_11")
assert not metric.empty
assert int(metric["timestamp"].max()) < 10_000_000_000

query_n = len(pd.read_csv(root / "query.csv"))
manifest = json.loads((trace_dir / "manifest.json").read_text(encoding="utf-8"))
baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
graph = json.loads(graph_path.read_text(encoding="utf-8"))

assert manifest.get("n") == query_n
assert len(manifest.get("rows", [])) == query_n
assert len(baseline.get("stats", [])) > 0
assert len(graph.get("containers", {})) > 0

print(f"query_n={query_n}")
print(f"metric_rows_2020_04_11={len(metric)} log_rows_2020_04_11={len(log)}")
print(f"trace_manifest={trace_dir / 'manifest.json'}")
print(f"baseline_stats={len(baseline.get('stats', []))}")
print(f"graph_containers={len(graph.get('containers', {}))} nodes={len(graph.get('nodes', {}))}")
if len(log) == 0:
    print("note: Telecom has no log directory; metric,log,trace keeps table compatibility but log evidence is empty.")
print("Telecom P0 preflight ok")
PY

for i in "${!MODS[@]}"; do
  mods=${MODS[$i]}
  name=${NAMES[$i]}

  echo "=== Telecom P0 ${name}: ${mods} ==="

  python3 eval/run_openrca_d32_lodo.py \
    --data-root "$DATA_ROOT" \
    --query-csv "$QUERY_CSV" \
    --record-csv "$RECORD_CSV" \
    --baseline "$BASELINE" \
    --node-graph "$GRAPH" \
    --rules knowledge/refutation_rules_v2.json \
    --trace-summary-dir "$TRACE_DIR" \
    --modalities "$mods" \
    --knowledge-dir "knowledge/telecom_d32_lodo_p0_${name}" \
    --out "logs/telecom_d32_lodo_p0_${name}_predictions.csv" \
    --debug-json "logs/telecom_d32_lodo_p0_${name}_debug.json" \
    --checkpoint-jsonl "logs/telecom_d32_lodo_p0_${name}_checkpoint.jsonl" \
    "${resume_args[@]}" \
    2>&1 | tee "logs/telecom_d32_lodo_p0_${name}_run.log"

  python3 eval/openrca_official_case_eval.py \
    --pred "logs/telecom_d32_lodo_p0_${name}_predictions.csv" \
    --query "$QUERY_CSV" \
    --node-graph "$GRAPH" \
    --out "logs/telecom_d32_lodo_p0_${name}_official_case_eval.json"

  python3 eval/field_hit_diagnostics.py \
    --pred "logs/telecom_d32_lodo_p0_${name}_predictions.csv" \
    --query "$QUERY_CSV" \
    --node-graph "$GRAPH" \
    --out "logs/telecom_d32_lodo_p0_${name}_field_diag.json"

  python3 eval/reason_conditional_tolerance_eval.py \
    --pred "logs/telecom_d32_lodo_p0_${name}_predictions.csv" \
    --query "$QUERY_CSV" \
    --record "$RECORD_CSV" \
    --node-graph "$GRAPH" \
    --out "logs/telecom_d32_lodo_p0_${name}_reason_conditional.json"
done

echo "=== Telecom P0 summary ==="
python3 - <<'PY'
import json
from pathlib import Path

names = ["metric", "metric_trace", "metric_log_trace"]
print("experiment,n,official_strict,official_partial,time_hit,component_hit,reason_hit,reason_cond_strict,reason_cond_partial")

for name in names:
    official = json.loads(Path(f"logs/telecom_d32_lodo_p0_{name}_official_case_eval.json").read_text())
    field = json.loads(Path(f"logs/telecom_d32_lodo_p0_{name}_field_diag.json").read_text())["field_item"]
    rc = json.loads(Path(f"logs/telecom_d32_lodo_p0_{name}_reason_conditional.json").read_text())["overall"]
    print(
        f"{name},{official['n']},"
        f"{official['strict_rate']:.4f},{official['partial_rate']:.4f},"
        f"{field['time']['hit_rate']:.4f},{field['component']['hit_rate']:.4f},{field['reason']['hit_rate']:.4f},"
        f"{rc['reason_conditional_strict_rate']:.4f},{rc['reason_conditional_partial_rate']:.4f}"
    )
PY
