#!/bin/bash
set -e
export PRISM_CHT_MODEL_API_KEY="sk-b8af2a9ac7da4c64b8484537d60b9449"
export PRISM_CHT_MODEL_BASE_URL="https://api.deepseek.com"
export PRISM_CHT_MODEL_NAME="deepseek-v4-pro"
cd /home/dell2/RCA513/yyx
PY=/home/dell2/RCA513/yyx/trace_summary/.venv_d32/bin/python
OUTDIR=prismv4/results/prism_cht
mkdir -p "$OUTDIR"

COMMON_ARGS="--max-cases 0 --max-hypotheses 10 --recall-pool-size 15 --max-steps 4"

echo "=== SS retry started at $(date) ==="
nohup "$PY" -m prismv4.experiments.run_rcaeval_continuous \
  --data-root /home/dell2/RCA513/ysj/dataset/RCAEval/RE3 \
  --system RE3-SS \
  $COMMON_ARGS \
  --only-cases "front-end_f1/2" \
  --output "$OUTDIR/SS_v4_retry.json" \
  --llm-io-output "$OUTDIR/SS_v4_retry.json.llm_io.jsonl" \
  > "$OUTDIR/SS_v4_retry.log" 2>&1
echo "=== SS retry done at $(date) ==="

echo "=== OB retry started at $(date) ==="
nohup "$PY" -m prismv4.experiments.run_rcaeval_continuous \
  --data-root /home/dell2/RCA513/ysj/dataset/RCAEval/RE3 \
  --system RE3-OB \
  $COMMON_ARGS \
  --only-cases "emailservice_f1/1,emailservice_f4/3" \
  --output "$OUTDIR/OB_v4_retry.json" \
  --llm-io-output "$OUTDIR/OB_v4_retry.json.llm_io.jsonl" \
  > "$OUTDIR/OB_v4_retry.log" 2>&1
echo "=== OB retry done at $(date) ==="

echo "=== ALL RETRY DONE at $(date) ==="
