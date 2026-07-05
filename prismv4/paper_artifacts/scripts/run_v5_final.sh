#!/usr/bin/env bash
set -uo pipefail
export PYTHONPATH=/home/dell2/RCA513/yyx
PYTHON=/home/dell2/RCA513/yyx/trace_summary/.venv_d32/bin/python
OUTDIR=prismv4/results/prism_cht/v5_optimized
mkdir -p "$OUTDIR"

for SYS in RE2-OB RE2-SS RE3-OB RE3-SS RE3-TT; do
  SUITE="${SYS%%-*}"
  unset PRISM_CHT_IVD_SOURCE_LIKELIHOOD_RATIO PRISM_CHT_IVD_TOURNAMENT_CAP PRISM_CHT_IVD_RESOURCE_MAGNITUDE_MIN
  export PRISM_CHT_RECALL_POOL_SIZE=15
  echo "[$(date +%H:%M:%S)] RUN $SYS"
  $PYTHON -m prismv4.experiments.run_rcaeval_continuous \
    --dataset RE3 --data-root /home/dell2/RCA513/ysj/dataset/RCAEval/$SUITE \
    --system "$SYS" --max-cases 0 --max-hypotheses 10 \
    --recall-pool-size 15 --max-steps 0 --mode lightweight \
    --output "$OUTDIR/${SYS}_v5.json" \
    --llm-io-output /tmp/opencode/${SYS}_v5.jsonl 2>&1 | tail -1
  echo "[$(date +%H:%M:%S)] DONE $SYS"
done

export PRISM_CHT_IVD_SOURCE_LIKELIHOOD_RATIO=0.20
export PRISM_CHT_IVD_TOURNAMENT_CAP=15
export PRISM_CHT_IVD_RESOURCE_MAGNITUDE_MIN=2.0
export PRISM_CHT_RECALL_POOL_SIZE=25
echo "[$(date +%H:%M:%S)] RUN RE2-TT (wide IVD)"
$PYTHON -m prismv4.experiments.run_rcaeval_continuous \
  --dataset RE3 --data-root /home/dell2/RCA513/ysj/dataset/RCAEval/RE2 \
  --system RE2-TT --max-cases 0 --max-hypotheses 15 \
  --recall-pool-size 25 --max-steps 0 --mode lightweight \
  --output "$OUTDIR/RE2-TT_v5.json" \
  --llm-io-output /tmp/opencode/RE2-TT_v5.jsonl 2>&1 | tail -1
echo "[$(date +%H:%M:%S)] ALL DONE"