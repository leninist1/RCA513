#!/usr/bin/env bash
set -uo pipefail

export PYTHONPATH=/home/dell2/RCA513/yyx
export PRISM_CHT_MODEL_TIMEOUT_SECONDS=180
export PRISM_CHT_MODEL_RETRIES=1
export PRISM_CHT_EVENT_CAUSALIZER_MAX_EVENTS=18
export PRISM_CHT_EVENT_CAUSALIZER_MAX_CONTEXT_CHARS=24000
export PRISM_CHT_RECALL_POOL_SIZE=25
export PRISM_CHT_IVD_STRICT_GATE=0

PYTHON=/home/dell2/RCA513/yyx/trace_summary/.venv_d32/bin/python
REPO=/home/dell2/RCA513/yyx
OUTDIR=$REPO/prismv4/results/prism_cht/v5_optimized
LOGDIR=$REPO/prismv4/paper_artifacts/logs/v5_optimized
mkdir -p "$OUTDIR" "$LOGDIR"

run_one() {
  local system="$1"
  local suite="${system%%-*}"
  local data_root="/home/dell2/RCA513/ysj/dataset/RCAEval/$suite"
  local output="$OUTDIR/${system}_v5.json"
  local llm_io="$OUTDIR/${system}_v5.jsonl"
  local log="$LOGDIR/run_${system}_v5.log"

  # Large-graph systems (Train Ticket ~40 services) use wider IVD filter
  # to improve recall. Small graphs use defaults (0.35/10/5.0).
  if [[ "$system" == *"-TT" ]]; then
    export PRISM_CHT_IVD_SOURCE_LIKELIHOOD_RATIO=0.20
    export PRISM_CHT_IVD_TOURNAMENT_CAP=15
    export PRISM_CHT_IVD_RESOURCE_MAGNITUDE_MIN=2.0
  else
    unset PRISM_CHT_IVD_SOURCE_LIKELIHOOD_RATIO
    unset PRISM_CHT_IVD_TOURNAMENT_CAP
    unset PRISM_CHT_IVD_RESOURCE_MAGNITUDE_MIN
  fi

  echo "[$(date '+%H:%M:%S')] START $system (IVD_RATIO=${PRISM_CHT_IVD_SOURCE_LIKELIHOOD_RATIO:-default})" | tee -a "$LOGDIR/progress.log"
  "$PYTHON" -m prismv4.experiments.run_rcaeval_continuous \
    --dataset RE3 --data-root "$data_root" \
    --system "$system" --max-cases 0 --max-hypotheses 15 \
    --recall-pool-size 25 --max-steps 1 --mode lightweight \
    --output "$output" --llm-io-output "$llm_io" > "$log" 2>&1
  echo "[$(date '+%H:%M:%S')] DONE $system (exit=$?)" | tee -a "$LOGDIR/progress.log"
}

for system in RE2-OB RE2-SS RE2-TT RE3-OB RE3-SS RE3-TT; do
  run_one "$system"
done
echo "[$(date '+%H:%M:%S')] ALL DONE" | tee -a "$LOGDIR/progress.log"
