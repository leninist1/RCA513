#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -f "$ROOT_DIR/eval/run_openrca_d32_lodo.py" ]]; then
  SOURCE_DIR="$ROOT_DIR"
elif [[ -f "$ROOT_DIR/confidence_aware_rca_v2_drift_focus_2026-06-02_resend/source/eval/run_openrca_d32_lodo.py" ]]; then
  SOURCE_DIR="$ROOT_DIR/confidence_aware_rca_v2_drift_focus_2026-06-02_resend/source"
else
  echo "Cannot find eval/run_openrca_d32_lodo.py. Put this script either in the project root or in confidence_aware_rca_v2_drift_focus_2026-06-02_resend/source." >&2
  exit 1
fi

BANK_ROOT="${BANK_ROOT:-/home/yan/workspace/data/openrca/Bank}"
BANK_QUERY="${BANK_QUERY:-$BANK_ROOT/query.csv}"
BANK_RECORD="${BANK_RECORD:-$BANK_ROOT/record.csv}"
BANK_BASELINE="${BANK_BASELINE:-refute/knowledge/baseline_distributions_all_metric_dates.json}"
BANK_NODE_GRAPH="${BANK_NODE_GRAPH:-refute/knowledge/node_container_graph_2021_03_10.json}"

cd "$SOURCE_DIR"

run_group() {
  local name="$1"
  local modalities="$2"

  echo "==> Running Bank P1 group: $name ($modalities)"
  PYTHONPATH=..:. python3 eval/run_openrca_d32_lodo.py \
    --data-root "$BANK_ROOT" \
    --query-csv "$BANK_QUERY" \
    --record-csv "$BANK_RECORD" \
    --baseline "$BANK_BASELINE" \
    --node-graph "$BANK_NODE_GRAPH" \
    --modalities "$modalities" \
    --knowledge-dir "knowledge/d32_lodo_p1_${name}" \
    --out "logs/d32_lodo_p1_${name}_predictions.csv" \
    --debug-json "logs/d32_lodo_p1_${name}_debug.json" \
    --checkpoint-jsonl "logs/d32_lodo_p1_${name}_checkpoint.jsonl" \
    2>&1 | tee "logs/d32_lodo_p1_${name}_run.log"

  PYTHONPATH=..:. python3 eval/openrca_official_case_eval.py \
    --pred "logs/d32_lodo_p1_${name}_predictions.csv" \
    --query "$BANK_QUERY" \
    --out "logs/d32_lodo_p1_${name}_official_case_eval.json"

  PYTHONPATH=..:. python3 eval/field_hit_diagnostics.py \
    --pred "logs/d32_lodo_p1_${name}_predictions.csv" \
    --query "$BANK_QUERY" \
    --out "logs/d32_lodo_p1_${name}_field_diag.json"

  PYTHONPATH=..:. python3 eval/reason_conditional_tolerance_eval.py \
    --pred "logs/d32_lodo_p1_${name}_predictions.csv" \
    --query "$BANK_QUERY" \
    --record "$BANK_RECORD" \
    --out "logs/d32_lodo_p1_${name}_reason_conditional.json"
}

run_group "log" "log"
run_group "metric_log" "metric,log"
