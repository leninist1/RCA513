#!/bin/bash
set -e
cd /home/dell2/RCA513/yyx
PY=/home/dell2/RCA513/yyx/trace_summary/.venv_d32/bin/python
OUTDIR=prismv4/results/prism_cht
mkdir -p "$OUTDIR"

echo "=== OB full run started at $(date) ==="
nohup "$PY" -m prismv4.experiments.run_rcaeval_continuous \
  --data-root /home/dell2/RCA513/ysj/dataset/RCAEval/RE3 \
  --system RE3-OB \
  --max-cases 0 \
  --max-hypotheses 10 \
  --recall-pool-size 15 \
  --max-steps 4 \
  --output "$OUTDIR/OB_full_recall_v1.json" \
  > "$OUTDIR/OB_full_recall_v1.log" 2>&1
echo "=== OB done at $(date) ==="

echo "=== TT full run started at $(date) ==="
nohup "$PY" -m prismv4.experiments.run_rcaeval_continuous \
  --data-root /home/dell2/RCA513/ysj/dataset/RCAEval/RE3 \
  --system RE3-TT \
  --max-cases 0 \
  --max-hypotheses 10 \
  --recall-pool-size 15 \
  --max-steps 4 \
  --output "$OUTDIR/TT_full_recall_v1.json" \
  > "$OUTDIR/TT_full_recall_v1.log" 2>&1
echo "=== TT done at $(date) ==="
