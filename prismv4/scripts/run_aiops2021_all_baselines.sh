#!/bin/bash
set -e
PY312=/home/dell2/RCA513/yyx/RCAEval-toolkit/env/bin/python
PY38=/home/yyx/.conda/envs/rcae_rcd_yyx/bin/python
RUNNER=/home/dell2/RCA513/yyx/prismv4/scripts/run_aiops2021_baselines.py

TS=$(date +%H%M%S)
LOG=/tmp/opencode/aiops2021_baselines_${TS}.log
echo "START $(date)" > $LOG

# BARO (py3.12 -- fast)
echo "=== baro (py3.12) ===" >> $LOG
$PY312 $RUNNER --method baro >> $LOG 2>&1 || echo "FAIL baro" >> $LOG

# E-Diagnosis (py3.8 rcd env)
echo "=== e_diagnosis (py3.8) ===" >> $LOG
$PY38 $RUNNER --method e_diagnosis >> $LOG 2>&1 || echo "FAIL e_diagnosis" >> $LOG

# RCD (py3.8 rcd env, needs link.sh done)
echo "=== rcd (py3.8) ===" >> $LOG
$PY38 $RUNNER --method rcd >> $LOG 2>&1 || echo "FAIL rcd" >> $LOG

echo "ALL DONE $(date)" >> $LOG
echo "aiops2021 baselines log: $LOG"
