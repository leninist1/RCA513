#!/bin/bash
set -e
export PRISM_CHT_MODEL_API_KEY="sk-1da96c68534349b8b405ab9e7c90717a"
export PRISM_CHT_MODEL_BASE_URL="https://api.deepseek.com"
export PRISM_CHT_MODEL_NAME="deepseek-chat"
cd /home/dell2/RCA513/yyx
PY=/home/dell2/RCA513/yyx/trace_summary/.venv_d32/bin/python
DATA_ROOT=/home/dell2/RCA513/ysj/dataset/RCAEval/RE2
OUTDIR=prismv4/results/prism_cht
SYS=$1
COMMON="--dataset RE3 --data-root $DATA_ROOT --mode lightweight --max-cases 0 --max-hypotheses 10 --recall-pool-size 15 --max-steps 4"
echo "=== $SYS lightweight started at $(date) ==="
"$PY" -m prismv4.experiments.run_rcaeval_continuous \
  $COMMON --system "$SYS" \
  --output "$OUTDIR/${SYS}_lw_v1.json" \
  --llm-io-output "$OUTDIR/${SYS}_lw_v1.json.llm_io.jsonl" \
  2>&1 | tee "$OUTDIR/${SYS}_lw_v1.log"
echo "=== $SYS done at $(date) ==="