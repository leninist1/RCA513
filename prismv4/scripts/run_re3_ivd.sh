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

echo "=== RE3-TT started at $(date) ==="
nohup "$PY" -m prismv4.experiments.run_rcaeval_continuous \
  --data-root /home/dell2/RCA513/ysj/dataset/RCAEval/RE3 \
  --system RE3-TT \
  $COMMON_ARGS \
  --output "$OUTDIR/TT_v3_ivd.json" \
  > "$OUTDIR/TT_v3_ivd.log" 2>&1
echo "=== RE3-TT done at $(date) ==="

echo "=== RE3-SS started at $(date) ==="
nohup "$PY" -m prismv4.experiments.run_rcaeval_continuous \
  --data-root /home/dell2/RCA513/ysj/dataset/RCAEval/RE3 \
  --system RE3-SS \
  $COMMON_ARGS \
  --output "$OUTDIR/SS_v3_ivd.json" \
  > "$OUTDIR/SS_v3_ivd.log" 2>&1
echo "=== RE3-SS done at $(date) ==="

echo "=== RE3-OB started at $(date) ==="
nohup "$PY" -m prismv4.experiments.run_rcaeval_continuous \
  --data-root /home/dell2/RCA513/ysj/dataset/RCAEval/RE3 \
  --system RE3-OB \
  $COMMON_ARGS \
  --output "$OUTDIR/OB_v3_ivd.json" \
  > "$OUTDIR/OB_v3_ivd.log" 2>&1
echo "=== RE3-OB done at $(date) ==="

echo "=== ALL RE3 DONE at $(date) ==="
