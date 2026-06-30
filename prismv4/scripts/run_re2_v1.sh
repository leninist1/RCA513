#!/bin/bash
set -e
export PRISM_CHT_MODEL_API_KEY="sk-b8af2a9ac7da4c64b8484537d60b9449"
export PRISM_CHT_MODEL_BASE_URL="https://api.deepseek.com"
export PRISM_CHT_MODEL_NAME="deepseek-v4-pro"
cd /home/dell2/RCA513/yyx
PY=/home/dell2/RCA513/yyx/trace_summary/.venv_d32/bin/python
DATA_ROOT=/home/dell2/RCA513/ysj/dataset/RCAEval/RE2
OUTDIR=prismv4/results/prism_cht
mkdir -p "$OUTDIR"
COMMON_ARGS="--dataset RE3 --data-root $DATA_ROOT --max-cases 0 --max-hypotheses 10 --recall-pool-size 15 --max-steps 4"

for SYS in RE2-OB RE2-SS RE2-TT; do
  echo "=== $SYS started at $(date) ==="
  "$PY" -m prismv4.experiments.run_rcaeval_continuous \
    $COMMON_ARGS \
    --system "$SYS" \
    --output "$OUTDIR/${SYS}_v1.json" \
    --llm-io-output "$OUTDIR/${SYS}_v1.json.llm_io.jsonl" \
    2>&1 | tee "$OUTDIR/${SYS}_v1.log"
  echo "=== $SYS done at $(date) ==="
done