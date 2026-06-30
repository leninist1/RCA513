#!/bin/bash
set -e
export PRISM_CHT_MODEL_API_KEY="sk-19c5f095a7a84710bbb02a2b7ea16491"
export PRISM_CHT_MODEL_BASE_URL="https://api.deepseek.com"
export PRISM_CHT_MODEL_NAME="deepseek-v4-pro"
cd /home/dell2/RCA513/yyx
PY=/home/dell2/RCA513/yyx/trace_summary/.venv_d32/bin/python
DATA_ROOT=/home/dell2/RCA513/ysj/dataset/RCAEval/RE2
OUTDIR=prismv4/results/prism_cht
mkdir -p "$OUTDIR"
COMMON_ARGS="--dataset RE3 --data-root $DATA_ROOT --max-cases 0 --max-hypotheses 10 --recall-pool-size 15 --max-steps 4"

echo "=== RE2-TT full started at $(date) ==="
"$PY" -m prismv4.experiments.run_rcaeval_continuous \
  $COMMON_ARGS \
  --system RE2-TT \
  --output "$OUTDIR/RE2-TT_v1.json" \
  --llm-io-output "$OUTDIR/RE2-TT_v1.json.llm_io.jsonl" \
  2>&1 | tee "$OUTDIR/RE2-TT_v1.log"
echo "=== RE2-TT full done at $(date) ==="