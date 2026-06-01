# Noise Lab

This directory is intentionally isolated from the main `MACE` and `PRISM`
pipelines. It is used to study two questions only:

1. Can a noise-field view identify the root source under heavy hotspot noise?
2. Can a structural object representation outperform symptom-heavy ranking?

## Modules

- `noise_field.py`
  - Implements `NoiseFieldScorer`
  - Models local noise, resonance mass, temporal source score, intervention
    collapse gain, and hotspot bias
- `structural_encoder.py`
  - Implements `StructuralObjectEncoder`
  - Encodes upstream/downstream context, temporal lead, mechanism focus,
    propagation signature, source likelihood, and symptom likelihood
- `runner.py`
  - Standalone experiment runner
  - Loads OpenRCA data, builds object graphs, scores objects, ranks candidates,
    and writes JSON results into `rca513/results`

## Run

```bash
PYTHONPATH=/home/dell2/RCA513/yyx/rca513 \
python -m openrca_meta_controller.noise_lab.runner --system Bank
```

For a smaller smoke test:

```bash
PYTHONPATH=/home/dell2/RCA513/yyx/rca513 \
python -m openrca_meta_controller.noise_lab.runner --system Bank --max-queries 10
```

## Output

The runner writes files like:

```text
rca513/results/noise_lab_Bank_YYYYMMDD_HHMMSS.json
```

Each result contains:

- `summary`: top-1/top-3/top-5 hit rates and average ground-truth rank
- `per_query`: top candidates, ground-truth rank, and a compact noise heatmap

## Current Scope

This lab does **not** use:

- LLM agents
- controller debate
- MACE shared memory
- PRISM final scoring

It only reuses the existing telemetry loader and object graph constructor.
