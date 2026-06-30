#!/bin/bash
cd /home/dell2/RCA513/yyx
PRISM_CHT_MODEL_API_KEY=sk-1da96c68534349b8b405ab9e7c90717a \
PRISM_CHT_MODEL_BASE_URL=https://api.deepseek.com \
PRISM_CHT_MODEL_NAME=deepseek-chat \
trace_summary/.venv_d32/bin/python -m prismv4.experiments.run_rcaeval_continuous \
  --dataset AIOps2021 --max-cases 0 --mode lightweight --recall-pool-size 15 \
  --output /home/dell2/RCA513/yyx/prismv4/results/prism_cht/AIOps2021-test_lw_v1.json
