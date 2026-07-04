#!/usr/bin/env bash
set -euo pipefail

# Rerun CAPE-RCA adaptive/full path for RCAEval RE2/RE3 after optimization.
#
# Usage:
#   PRISM_CHT_MODEL_API_KEY=... bash paper_artifacts/scripts/run_re2_re3_after_optimization.sh
#   SYSTEMS="RE2-TT RE3-TT" FORCE=1 bash paper_artifacts/scripts/run_re2_re3_after_optimization.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PROJECT_PARENT="$(dirname "$REPO_ROOT")"
export PYTHONPATH="$PROJECT_PARENT${PYTHONPATH:+:$PYTHONPATH}"

PYTHON_BIN="${PYTHON_BIN:-/home/dell2/RCA513/yyx/trace_summary/.venv_d32/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="${PYTHON_BIN_FALLBACK:-python3}"
fi

RCAEVAL_ROOT="${RCAEVAL_ROOT:-/home/dell2/RCA513/ysj/dataset/RCAEval}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/results/prism_cht/v4_re2_re3_optimized}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/paper_artifacts/logs/re2_re3_after_optimization}"
mkdir -p "$OUT_DIR" "$LOG_DIR"

SYSTEMS="${SYSTEMS:-RE2-OB RE2-SS RE2-TT RE3-OB RE3-SS RE3-TT}"
MAX_CASES="${MAX_CASES:-0}"
MAX_HYPOTHESES="${MAX_HYPOTHESES:-10}"
RECALL_POOL_SIZE="${RECALL_POOL_SIZE:-15}"
MAX_STEPS="${MAX_STEPS:-0}"
MODE="${MODE:-lightweight}"
export PRISM_CHT_LOG_SCAN_LIMIT="${PRISM_CHT_LOG_SCAN_LIMIT:-20000}"
export PRISM_CHT_EVENT_CAUSALIZER_MAX_EVENTS="${PRISM_CHT_EVENT_CAUSALIZER_MAX_EVENTS:-32}"
export PRISM_CHT_OWNER_RERANK_FALLBACK="${PRISM_CHT_OWNER_RERANK_FALLBACK:-0}"

run_one() {
  local system="$1"
  local suite="${system%%-*}"
  local data_root="$RCAEVAL_ROOT/$suite"
  local output="$OUT_DIR/${system}_optimized.json"
  local llm_io="$OUT_DIR/${system}_optimized.llm_io.jsonl"
  local log="$LOG_DIR/run_${system}.log"
  local cmd=(
    "$PYTHON_BIN" -m prismv4.experiments.run_rcaeval_continuous
    --dataset RE3
    --data-root "$data_root"
    --system "$system"
    --max-cases "$MAX_CASES"
    --max-hypotheses "$MAX_HYPOTHESES"
    --recall-pool-size "$RECALL_POOL_SIZE"
    --max-steps "$MAX_STEPS"
    --mode "$MODE"
    --output "$output"
    --llm-io-output "$llm_io"
  )

  if [[ -s "$output" && "${FORCE:-0}" != "1" ]]; then
    echo "SKIP $system: existing output $output"
    return 0
  fi

  echo "RUN $system -> $output"
  (
    cd "$PROJECT_PARENT"
    "${cmd[@]}" > "$log" 2>&1
  )
}

for system in $SYSTEMS; do
  run_one "$system"
done
