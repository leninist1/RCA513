#!/usr/bin/env bash
set -euo pipefail

# Runs only missing RCAEval RE1 CAPE-RCA-Full/adaptive-path experiments.
# RE1 is metrics-only. Use the paper RE1 runner: strong IVD cases finish via
# CAPE-RCA's adaptive shortcut, and unresolved cases are explicitly marked as
# timeout/budget-exhausted instead of fabricating EG-CDA output.
# Default is dry-run to prevent accidental expensive LLM calls:
#   DRY_RUN=0 PRISM_CHT_MODEL_API_KEY=... bash paper_artifacts/scripts/run_missing_experiments.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PROJECT_PARENT="$(dirname "$REPO_ROOT")"
export PYTHONPATH="$PROJECT_PARENT${PYTHONPATH:+:$PYTHONPATH}"

PYTHON_BIN="${PYTHON_BIN:-/home/dell2/RCA513/yyx/trace_summary/.venv_d32/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="${PYTHON_BIN_FALLBACK:-python3}"
fi

DATA_ROOT="${RCAEVAL_RE1_ROOT:-/home/dell2/RCA513/ysj/dataset/RCAEval/RE1}"
RAW_DIR="$REPO_ROOT/paper_artifacts/raw"
LOG_DIR="$REPO_ROOT/paper_artifacts/logs"
mkdir -p "$RAW_DIR" "$LOG_DIR"

run_one() {
  local system="$1"
  local output="$RAW_DIR/${system}_cape_rca_full.json"
  local log="$LOG_DIR/run_${system}_cape_rca_full.log"
  local cmd=(
    "$PYTHON_BIN" "$REPO_ROOT/paper_artifacts/scripts/run_re1_metrics_only_adaptive.py"
    --data-root "$DATA_ROOT"
    --system "$system"
    --output "$output"
    --recall-pool-size 15
  )

  if [[ -s "$output" ]]; then
    echo "SKIP $system: existing output $output"
    return 0
  fi

  echo "COMMAND ${cmd[*]} > $log 2>&1"
  if [[ "${DRY_RUN:-1}" == "1" ]]; then
    return 0
  fi

  (
    cd "$PROJECT_PARENT"
    "${cmd[@]}" > "$log" 2>&1
  )
}

run_one RE1-OB
run_one RE1-SS
run_one RE1-TT
