# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

This is a research codebase for **microservice Root Cause Analysis (RCA)** using counterfactual reasoning, emotion vectors, dynamic chain-of-thought, and heterogeneous agent evolution. The work targets the RCAEval benchmark datasets (RE1-TT, RE3-OB) and is expanding toward OpenRCA (ICLR 2025).

## Research phases (directory organization)

Each top-level directory corresponds to a distinct experiment/phase. The naming convention encodes the approach and target dataset:

| Directory | Approach | Dataset | Best result |
|-----------|----------|---------|-------------|
| `反事实+情绪向量+动态思维链分析-RE3OB/` | Emotion-driven RCA with counterfactuals | RE3-OB | 0.967 (v1.18) |
| `反事实+情绪向量+动态思维链分析RE1-TT/` | Same approach, single-script form | RE1-TT | 0.448 (v1.7) |
| `反事实+情绪向量+全模态+分类分治-re3ob/` | Full-modal + fault-type classification + emotion vectors | RE3-OB | — |
| `反事实验证最终版（5模态）-RE3OB/` | Counterfactual verification with 5 modalities | RE3-OB | — |
| `反事实+情绪向量+动态思维链分析+异质agent验证跑RE1-TT/` | Heterogeneous agent evolution (Phase 4) | RE1-TT | 0.448 (no improvement over baseline) |
| `openRCA数据集评估/` | Design doc for next-phase OpenRCA work | — | — |

## The only structured project: phase4-evolution

`反事实+情绪向量+动态思维链分析+异质agent验证跑RE1-TT/phase4-evolution-final-代码/phase4-evolution/` is the only codebase with a proper `src/` layout. All other directories contain standalone single-file scripts.

### Architecture

```
run_evolution.py          # Entry point. Pass --mock to test with fake LLM responses.
configs/
  agents.yaml             # LLM agent definitions (Claude + DeepSeek), each with attribution philosophy and internal state
  operators.yaml          # RCA operator registry (scoring, counterfactual, fusion, temporal, gating)
  evaluation.yaml         # Evaluation weights, data paths, generation count
src/
  rca_system.py           # Core RCA pipeline: anomaly detection → propagation → classification → counterfactual → fusion → ranking
  rca_config.py           # Hyperparameter dataclass with bounds. update() blends mutations via inertia.
  operator_library.py     # Pluggable strategy functions (propagation tracking, time penalty, score fusion)
  operators.py            # OperatorConfig container + YAML loader
  evolution_manager.py    # Main loop: generates variants via LLM agents, evaluates with multiprocessing, selects champion
  evaluator.py            # Stratified train/holdout split, parallel case evaluation (multiprocessing Pool)
  variant.py              # Variant dataclass representing one agent's proposed modification
  selector.py             # Champion selection with diversity preservation
  prompt_builder.py       # Constructs agent prompts with PARAM_DOC + STRATEGY_DOC + internal state
  llm_clients.py          # Anthropic Claude (native SDK) and DeepSeek (OpenAI-compatible REST) clients
  internal_state_tracker.py  # Tracks agent uncertainty, fatigue, exploration impulse
  failure_memory.py       # FIFO memory of failed attempts, filtered by attribution philosophy
  mutation_logger.py      # Records parameter diffs per generation to mutations.json
```

### How the evolution loop works

1. `EvolutionManager` loads agent configs, operator registry, and evaluation config from YAML.
2. Each generation, LLM agents (Claude/DeepSeek) receive a prompt containing the champion's current metrics, their internal state (uncertainty/fatigue/exploration), failure memory, and the allowed parameter/strategy space.
3. Agents propose JSON-formatted variants with parameter mutations and/or strategy changes.
4. Each variant is evaluated by `Evaluator`:
   - Builds a new `RCASystem` with mutated config.
   - Runs `rca_system.predict()` on each case in parallel (multiprocessing Pool, 4 workers).
   - Computes Top-1 accuracy, Top-3, MRR on both train and holdout sets.
   - Derives composite score `R_total` from accuracy, stability, generalization, complexity penalty, and ablation survival.
5. `Selector` picks the top variants (by R_total) with a diversity bonus.
6. The best variant becomes the new champion; internal states and failure memory are updated.
7. Repeats for `generations` iterations (default 5).

### How the RCA pipeline works (rca_system.py)

The pipeline is a fixed sequence of steps, with strategies selected by config:

1. **Data loading** — CSV time-series from RE1-TT cases (baseline + fault windows).
2. **Anomaly detection** — Z-score thresholding across metric configs, persistence filter.
3. **Propagation tracking** — Strategy selected by `propagation_strategy`: none / downstream_1_hop / bidirectional.
4. **Fault classification** — NaiveBayesFaultClassifier predicts CPU/MEM/DELAY/LOSS.
5. **Counterfactual recovery** — For each candidate service, compute soft counterfactual (blend fault with baseline) and recovery score.
6. **Time penalty** — Strategy selected by `time_penalty_strategy`: linear / exponential / step.
7. **Score fusion** — Strategy selected by `fusion_strategy`: multiplicative / additive / max_pooling.
8. **LOSS-specific scoring** — If fault type is LOSS, add network-specific loss scores.
9. **Ranking** — Services sorted by combined score; top-1 is the predicted root cause.

### Key dependencies

- `pyyaml`, `anthropic`, `google-generativeai` (per requirements.txt)
- `pandas`, `numpy`, `requests` (used throughout)
- The standalone scripts also depend on `run_re1tt` — a module imported from `/home/admin/rca-workspace/` that provides the RE1-TT data loading and baseline RCA functions.

## Standalone scripts (all other directories)

These are single-file Python experiments, typically 400–700 lines, structured as:
1. Configuration block (API keys, metric configs, thresholds)
2. Core RCA functions (anomaly detection, counterfactual, classification)
3. Emotion vector definitions and modulation logic
4. LLM integration (DeepSeek API via `requests`)
5. Main evaluation loop iterating over case files

The scripts share the same algorithmic core (Z-score anomaly detection, Naive Bayes classification, soft counterfactual, weighted fusion) but differ in:
- Number of metrics (5, 19, or 29)
- Whether emotion vectors modulate search behavior
- Whether log-based recall is enabled
- Whether all modalities are connected

## Important notes

- **Hardcoded paths**: Many scripts reference `/home/admin/rca-workspace/` and `/home/admin/RCAEval/data/`. These need to exist or be changed to run the code.
- **Embedded API keys**: Some standalone scripts contain hardcoded DeepSeek API keys. Do not commit these.
- **No test suite**: There are no automated tests. Evaluation is done by running the scripts against RCAEval datasets and inspecting accuracy output.
- **No package structure**: Only the phase4-evolution subdirectory has a `src/` layout. The other scripts are designed to be run directly with `python script_name.py`.
- **RE1-TT ceiling**: The Phase 4 evolution work concluded that V8.6 (0.448 accuracy) is near-optimal for pure parameter tuning on RE1-TT — the parameter space has no room for heterogeneous agent diversity to manifest as performance gain.
- **Emotion vectors ineffective**: Three scenarios × two datasets verified that emotion vector modulation produces no accuracy difference when the underlying counterfactual engine already adapts to different case types automatically.
- **Next direction**: The `openRCA数据集评估/` design doc outlines a shift to OpenRCA (ICLR 2025, 335 real fault cases, SOTA only 11.34%) with a meta-controller + strategy pool architecture for resource-constrained counterfactual search scheduling.
