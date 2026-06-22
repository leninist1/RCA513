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

echo "=== Eadro-SN started at $(date) ==="
nohup "$PY" -m prismv4.experiments.run_rcaeval_continuous \
  --dataset Eadro \
  --system Eadro-SN \
  $COMMON_ARGS \
  --output "$OUTDIR/Eadro_SN_v1.json" \
  --llm-io-output "$OUTDIR/Eadro_SN_v1.json.llm_io.jsonl" \
  > "$OUTDIR/Eadro_SN_v1.log" 2>&1
echo "=== Eadro-SN done at $(date) ==="

echo "=== Eadro-TT started at $(date) ==="
nohup "$PY" -m prismv4.experiments.run_rcaeval_continuous \
  --dataset Eadro \
  --system Eadro-TT \
  $COMMON_ARGS \
  --output "$OUTDIR/Eadro_TT_v1.json" \
  --llm-io-output "$OUTDIR/Eadro_TT_v1.json.llm_io.jsonl" \
  > "$OUTDIR/Eadro_TT_v1.log" 2>&1
echo "=== Eadro-TT done at $(date) ==="

echo "=== ALL EADRO DONE at $(date) ==="
