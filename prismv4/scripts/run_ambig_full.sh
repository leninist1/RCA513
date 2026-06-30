#!/bin/bash
set -e
export PRISM_CHT_MODEL_API_KEY="sk-1da96c68534349b8b405ab9e7c90717a"
export PRISM_CHT_MODEL_BASE_URL="https://api.deepseek.com"
export PRISM_CHT_MODEL_NAME="deepseek-chat"
cd /home/dell2/RCA513/yyx
PY=/home/dell2/RCA513/yyx/trace_summary/.venv_d32/bin/python
OUTDIR=prismv4/results/prism_cht

# Args: $1=DATASET (RE3|Eadro|OpenRCA), $2=SYSTEM, $3=DATA_ROOT (or empty), $4=FRAGMENTS
DATASET=$1
SYSTEM=$2
DATA_ROOT=$3
FRAGMENTS=$4

OUT_PREFIX="${SYSTEM}_ambig_full"
COMMON="--dataset $DATASET --mode full --max-cases 0 --max-hypotheses 10 --recall-pool-size 15 --max-steps 4 --system $SYSTEM --only-cases $FRAGMENTS"

EXTRA=""
if [ -n "$DATA_ROOT" ]; then
  EXTRA="--data-root $DATA_ROOT"
fi

echo "=== $SYSTEM ambiguous full mode started at $(date) ==="
"$PY" -m prismv4.experiments.run_rcaeval_continuous \
  $COMMON $EXTRA \
  --output "$OUTDIR/${OUT_PREFIX}.json" \
  --llm-io-output "$OUTDIR/${OUT_PREFIX}.json.llm_io.jsonl" \
  2>&1 | tee "$OUTDIR/${OUT_PREFIX}.log"
echo "=== $SYSTEM done at $(date) ==="