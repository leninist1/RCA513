#!/bin/bash
set -e
export PRISM_CHT_MODEL_API_KEY="sk-1da96c68534349b8b405ab9e7c90717a"
export PRISM_CHT_MODEL_BASE_URL="https://api.deepseek.com"
export PRISM_CHT_MODEL_NAME="deepseek-chat"
cd /home/dell2/RCA513/yyx
PY=/home/dell2/RCA513/yyx/trace_summary/.venv_d32/bin/python
OUTDIR=prismv4/results/prism_cht
mkdir -p "$OUTDIR"

echo "=== RE2-OB lightweight smoke (5 cases) at $(date) ==="
"$PY" -m prismv4.experiments.run_rcaeval_continuous \
  --dataset RE3 --data-root /home/dell2/RCA513/ysj/dataset/RCAEval/RE2 \
  --system RE2-OB --mode lightweight --max-cases 5 \
  --max-hypotheses 10 --recall-pool-size 15 --max-steps 4 \
  --output "$OUTDIR/RE2-OB_lw_smoke.json" \
  --llm-io-output "$OUTDIR/RE2-OB_lw_smoke.json.llm_io.jsonl" \
  2>&1 | tee "$OUTDIR/RE2-OB_lw_smoke.log"
echo "=== smoke done at $(date) ==="