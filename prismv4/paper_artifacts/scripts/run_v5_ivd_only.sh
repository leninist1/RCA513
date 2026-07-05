#!/usr/bin/env bash
set -uo pipefail
export PYTHONPATH=/home/dell2/RCA513/yyx
export PRISM_CHT_IVD_STRICT_GATE=0

PYTHON=/home/dell2/RCA513/yyx/trace_summary/.venv_d32/bin/python
OUTDIR=prismv4/results/prism_cht/v5_optimized
mkdir -p "$OUTDIR"

run_one() {
  local system="$1"
  local suite="${system%%-*}"
  local data_root="/home/dell2/RCA513/ysj/dataset/RCAEval/$suite"
  # TT uses wider IVD filter; others use defaults
  if [[ "$system" == *"-TT" ]]; then
    export PRISM_CHT_IVD_SOURCE_LIKELIHOOD_RATIO=0.20
    export PRISM_CHT_IVD_TOURNAMENT_CAP=15
    export PRISM_CHT_IVD_RESOURCE_MAGNITUDE_MIN=2.0
    export PRISM_CHT_RECALL_POOL_SIZE=25
    local pool=25
  else
    unset PRISM_CHT_IVD_SOURCE_LIKELIHOOD_RATIO PRISM_CHT_IVD_TOURNAMENT_CAP PRISM_CHT_IVD_RESOURCE_MAGNITUDE_MIN
    export PRISM_CHT_RECALL_POOL_SIZE=15
    local pool=15
  fi
  echo "RUN $system (pool=$pool)"
  "$PYTHON" -m prismv4.experiments.run_rcaeval_continuous \
    --dataset RE3 --data-root "$data_root" \
    --system "$system" --max-cases 0 --max-hypotheses 10 \
    --recall-pool-size "$pool" --max-steps 0 --mode lightweight \
    --output "$OUTDIR/${system}_v5.json" \
    --llm-io-output /tmp/opencode/${system}_v5_nollm.jsonl 2>&1 | tail -1
}

for system in RE2-OB RE2-SS RE2-TT RE3-OB RE3-SS RE3-TT; do
  run_one "$system"
done
echo "ALL DONE"
