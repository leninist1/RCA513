#!/bin/bash
# Run all RCAEval baselines on all datasets and save results to baseline_results/
# Usage: ./run_baselines.sh

RCATOOLKIT=/home/dell2/RCA513/yyx/RCAEval-toolkit
OUT_ROOT=/home/dell2/RCA513/yyx/prismv4/results/baseline_results
PY312=$RCATOOLKIT/env/bin/python
PY38=/home/yyx/.conda/envs/rcae_rcd_yyx/bin/python

mkdir -p $OUT_ROOT

# ====== Py3.12 default env: baro | pdiagnose | microrank | tracerca ======
PY312_METHODS=(baro pdiagnose microrank tracerca)
PY312_DATASETS=(re2-ob re2-ss re2-tt re3-ob re3-ss re3-tt)

for method in "${PY312_METHODS[@]}"; do
    for ds in "${PY312_DATASETS[@]}"; do
        tag="${method}_${ds}"
        echo "=== Running $tag (py3.12) ==="
        cd $RCATOOLKIT
        rm -rf output
        mkdir -p output/results
        $PY312 main.py --method $method --dataset $ds > $OUT_ROOT/${tag}.log 2>&1
        # move result JSONs and eval summary
        if ls output/results/*.json 2>/dev/null | head -1 > /dev/null; then
            mkdir -p $OUT_ROOT/${tag}/results
            cp output/results/*.json $OUT_ROOT/${tag}/results/ 2>/dev/null
        fi
    done
done

# ====== Py3.8 rcd env: rcd | e_diagnosis | mmrcd | ht ======
# ht, mmrcd are PyRCA methods, can run on Py3.8 with PyRCA installed
# Note: re2-ss and re3-ss without traces - rcd / e_diagnosis can still run (metric-based)
PY38_METHODS=(rcd e_diagnosis mmrcd)
PY38_DATASETS=(re2-ob re2-ss re2-tt re3-ob re3-ss re3-tt)

for method in "${PY38_METHODS[@]}"; do
    for ds in "${PY38_DATASETS[@]}"; do
        tag="${method}_${ds}"
        echo "=== Running $tag (py3.8) ==="
        cd $RCATOOLKIT
        rm -rf output
        mkdir -p output/results
        $PY38 main.py --method $method --dataset $ds > $OUT_ROOT/${tag}.log 2>&1
        if ls output/results/*.json 2>/dev/null | head -1 > /dev/null; then
            mkdir -p $OUT_ROOT/${tag}/results
            cp output/results/*.json $OUT_ROOT/${tag}/results/ 2>/dev/null
        fi
    done
done

echo "=== ALL DONE ==="
ls -d $OUT_ROOT/*/ 2>/dev/null