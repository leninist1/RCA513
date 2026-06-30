#!/bin/bash
set -e
TK=/home/dell2/RCA513/yyx/RCAEval-toolkit
PY312=$TK/env/bin/python
PY38=/home/yyx/.conda/envs/rcae_rcd_yyx/bin/python
OUT=/home/dell2/RCA513/yyx/prismv4/results/baseline_results

TS=$(date +%H%M%S)
LOG=/tmp/opencode/baselines_${TS}.log
echo "START $(date)" > $LOG

cd $TK

# === Metric-only baselines (py3.8 rcd env): rcd, e_diagnosis, mmrcd, ht ===
METRIC_BASICS=(rcd e_diagnosis mmrcd ht)
for m in "${METRIC_BASICS[@]}"; do
    for ds in re2-ob re2-ss re2-tt re3-ob re3-ss re3-tt; do
        echo "=== $m $ds (py38) ===" >> $LOG
        $PY38 run_baseline_custom.py --method $m --dataset $ds >> $LOG 2>&1 || echo "FAIL $m $ds" >> $LOG
    done
done

# === Py3.12 env: baro via main.py (fast and verified) ===
for ds in re2-ob re2-ss re2-tt re3-ob re3-ss re3-tt; do
    echo "=== baro $ds (py3.12 main.py) ===" >> $LOG
    rm -rf $TK/output && mkdir -p $TK/output/results
    $PY312 main.py --method baro --dataset $ds >> $LOG 2>&1 || echo "FAIL baro $ds" >> $LOG
    mkdir -p $OUT/baro_$ds/results
    cp $TK/output/results/*.json $OUT/baro_$ds/results/ 2>/dev/null || true
done

# === Py3.12 env: trace-required baselines via custom runner ===
TRACE_BASICS=(microrank tracerca pdiagnose)
for m in "${TRACE_BASICS[@]}"; do
    for ds in re2-ob re2-tt re3-ob re3-tt; do
        # trace baselines can only run on datasets with traces
        echo "=== $m $ds (py3.12 custom) ===" >> $LOG
        $PY312 run_baseline_custom.py --method $m --dataset $ds >> $LOG 2>&1 || echo "FAIL $m $ds" >> $LOG
    done
done

echo "ALL DONE $(date)" >> $LOG
echo "Final log at: $LOG"
