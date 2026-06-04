#!/usr/bin/env bash
set -euo pipefail

BASE=/home/dell2/RCA513/yyx/trace_summary/confidence_aware_rca_v2_drift_focus_2026-06-02_resend/source
MARKET_ROOT=/home/dell2/RCA513/ysj/dataset/openrca/Market
SUBSETS=("cloudbed-1" "cloudbed-2")

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

echo "=== Market P0 preflight ==="
python3 - <<'PY'
from refute.src.baseline_distributions import BaselineStore
from refute.src.data_loader import BankDataPaths, load_log_day, load_metric_day
from refute_b_v2.query_windows import parse_query_window
from refute_b_v2.trace_propagation import trace_evidence_for_service
from refute_b_v2_d32.layer2 import D32RefutationPipeline
from refute_b_v2_d32.schema import reason_bucket

w = parse_query_window("The cloud service system experienced one failure within the time range of March 20, 2022, from 09:00 to 09:30.")
assert w.start.year == 2022
assert reason_bucket("container CPU load") == "cpu"
print("imports and Market compat ok")
PY

for subset in "${SUBSETS[@]}"; do
  DATA_ROOT=$MARKET_ROOT/$subset
  QUERY_CSV=$DATA_ROOT/query.csv
  RECORD_CSV=$DATA_ROOT/record.csv
  safe=${subset//-/_}

  TRACE_DIR="trace_summaries/market_${safe}_queries_full"
  BASELINE="knowledge/market_${safe}_baseline_distributions_all_metric_dates.json"
  GRAPH="knowledge/market_${safe}_node_container_graph.json"

  test -d "$DATA_ROOT"
  test -f "$QUERY_CSV"
  test -f "$RECORD_CSV"
  test -f "$TRACE_DIR/manifest.json"
  test -f "$BASELINE"
  test -f "$GRAPH"
done

for subset in "${SUBSETS[@]}"; do
  DATA_ROOT=$MARKET_ROOT/$subset
  QUERY_CSV=$DATA_ROOT/query.csv
  RECORD_CSV=$DATA_ROOT/record.csv
  safe=${subset//-/_}

  for i in "${!MODS[@]}"; do
    mods=${MODS[$i]}
    name=${NAMES[$i]}

    echo "=== Market ${subset} P0 ${name}: ${mods} ==="

    python3 eval/run_openrca_d32_lodo.py \
      --data-root "$DATA_ROOT" \
      --query-csv "$QUERY_CSV" \
      --record-csv "$RECORD_CSV" \
      --baseline "knowledge/market_${safe}_baseline_distributions_all_metric_dates.json" \
      --node-graph "knowledge/market_${safe}_node_container_graph.json" \
      --rules knowledge/refutation_rules_v2.json \
      --trace-summary-dir "trace_summaries/market_${safe}_queries_full" \
      --modalities "$mods" \
      --knowledge-dir "knowledge/market_${safe}_d32_lodo_p0_${name}" \
      --out "logs/market_${safe}_d32_lodo_p0_${name}_predictions.csv" \
      --debug-json "logs/market_${safe}_d32_lodo_p0_${name}_debug.json" \
      --checkpoint-jsonl "logs/market_${safe}_d32_lodo_p0_${name}_checkpoint.jsonl" \
      "${resume_args[@]}" \
      2>&1 | tee "logs/market_${safe}_d32_lodo_p0_${name}_run.log"

    python3 eval/openrca_official_case_eval.py \
      --pred "logs/market_${safe}_d32_lodo_p0_${name}_predictions.csv" \
      --query "$QUERY_CSV" \
      --node-graph "$GRAPH" \
      --out "logs/market_${safe}_d32_lodo_p0_${name}_official_case_eval.json"

    python3 eval/field_hit_diagnostics.py \
      --pred "logs/market_${safe}_d32_lodo_p0_${name}_predictions.csv" \
      --query "$QUERY_CSV" \
      --node-graph "$GRAPH" \
      --out "logs/market_${safe}_d32_lodo_p0_${name}_field_diag.json"

    python3 eval/reason_conditional_tolerance_eval.py \
      --pred "logs/market_${safe}_d32_lodo_p0_${name}_predictions.csv" \
      --query "$QUERY_CSV" \
      --record "$RECORD_CSV" \
      --node-graph "$GRAPH" \
      --out "logs/market_${safe}_d32_lodo_p0_${name}_reason_conditional.json"
  done
done

echo "=== Market P0 summary ==="
python3 - "${SUBSETS[@]}" <<'PY'
import json
import sys
from pathlib import Path

names = ["metric", "metric_trace", "metric_log_trace"]
print("subset,experiment,n,official_strict,official_partial,time_hit,component_hit,reason_hit,reason_cond_strict,reason_cond_partial")

for subset in sys.argv[1:]:
    safe = subset.replace("-", "_")
    for name in names:
        official = json.loads(Path(f"logs/market_{safe}_d32_lodo_p0_{name}_official_case_eval.json").read_text())
        field = json.loads(Path(f"logs/market_{safe}_d32_lodo_p0_{name}_field_diag.json").read_text())["field_item"]
        rc = json.loads(Path(f"logs/market_{safe}_d32_lodo_p0_{name}_reason_conditional.json").read_text())["overall"]
        print(
            f"{subset},{name},{official['n']},"
            f"{official['strict_rate']:.4f},{official['partial_rate']:.4f},"
            f"{field['time']['hit_rate']:.4f},{field['component']['hit_rate']:.4f},{field['reason']['hit_rate']:.4f},"
            f"{rc['reason_conditional_strict_rate']:.4f},{rc['reason_conditional_partial_rate']:.4f}"
        )
PY
