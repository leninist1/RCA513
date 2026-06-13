# D32 V2 — Refutation-based Root Cause Analysis Pipeline

**Final frozen version, June 2026. NO LLM dependency.**

End-to-end pipeline for OpenRCA benchmark root cause analysis. Pure rules + XGBoost/Logistic Regression + onset-first time voting. No GPT, Claude, or DeepSeek calls.

## Algorithm Overview

```
raw metric/log/trace → Case Signature → Knowledge Matching (cosine similarity + mined rules)
    → Joint Candidate Generation → S1 Reason Classifier (XGBoost/LogReg, 9-class)
    → Two-Stage Scoring (family filter + component earliness/strength)
    → rc-sort (reason confidence primary sort)
    → Onset-First Time Anchor Voting
    → Reason Name Mapping → Final Prediction
```

- **S1 Classifier**: per-dataset per-fold XGBoost (≥60 samples) or Logistic Regression, 50-dim features → 9 reason probability classes
- **rc-sort**: sorts candidates by reason classifier confidence first, then rebuttal score, then support strength
- **Onset-First**: each KPI votes only its first anomalous timestamp, component onset gets bonus, exponential decay penalizes late votes
- **Family Filter**: cpu→docker, network→os, db_connection→db (bank: no filter)

## Prerequisites

### Python Dependencies

```bash
pip install pandas numpy scikit-learn xgboost
```

### OpenRCA Dataset

Download from the OpenRCA benchmark. Structure:

```
openrca/
├── Bank/
│   ├── query.csv
│   ├── record.csv
│   └── metric/, log/, ...  (raw telemetry)
├── Market/
│   ├── cloudbed-1/
│   │   ├── query.csv
│   │   ├── record.csv
│   │   └── ...
│   └── cloudbed-2/
│       ├── query.csv
│       ├── record.csv
│       └── ...
└── Telecom/
    ├── query.csv
    ├── record.csv
    └── ...
```

## Quick Start

```bash
git clone --branch trace_summary_d32v2 git@github.com:leninist1/RCA513.git
cd RCA513/trace_summary/confidence_aware_rca_v2_drift_focus_2026-06-02_resend/source
```

All commands run from the `source/` directory with `PYTHONPATH=.:..`.

## Run: Bank (136 cases)

```bash
PYTHONPATH=.:.. python3 eval/run_openrca_d32_lodo.py \
  --data-root /path/to/openrca/Bank \
  --query-csv /path/to/openrca/Bank/query.csv \
  --record-csv /path/to/openrca/Bank/record.csv \
  --baseline refute/knowledge/baseline_distributions_all_metric_dates.json \
  --node-graph refute/knowledge/node_container_graph_2021_03_10.json \
  --rules knowledge/refutation_rules_v2.json \
  --trace-summary-dir trace_summaries/bank_queries_full \
  --modalities metric,log,trace \
  --dataset-name bank \
  --casefold-classifier-dir knowledge/casefold_classifiers \
  --knowledge-dir knowledge/d32_casefold_bank \
  --onset-first-weight 4.0 \
  --onset-decay-seconds 300 \
  --component-onset-bonus 5.0 \
  --out logs/bank_predictions.csv \
  --debug-json logs/bank_debug.json \
  --checkpoint-jsonl logs/bank_checkpoint.jsonl
```

## Run: Market Cloudbed-1 (70 cases)

```bash
PYTHONPATH=.:.. python3 eval/run_openrca_d32_lodo.py \
  --data-root /path/to/openrca/Market/cloudbed-1 \
  --query-csv /path/to/openrca/Market/cloudbed-1/query.csv \
  --record-csv /path/to/openrca/Market/cloudbed-1/record.csv \
  --baseline knowledge/market_cloudbed_1_baseline_distributions_all_metric_dates.json \
  --node-graph knowledge/market_cloudbed_1_node_container_graph.json \
  --rules knowledge/refutation_rules_v2.json \
  --trace-summary-dir trace_summaries/market_cloudbed_1_queries_full \
  --modalities metric,log,trace \
  --dataset-name market_cb1 \
  --casefold-classifier-dir knowledge/casefold_classifiers \
  --knowledge-dir knowledge/d32_casefold_market_cb1 \
  --out logs/market_cb1_predictions.csv \
  --debug-json logs/market_cb1_debug.json \
  --checkpoint-jsonl logs/market_cb1_checkpoint.jsonl
```

## Run: Market Cloudbed-2 (78 cases)

```bash
PYTHONPATH=.:.. python3 eval/run_openrca_d32_lodo.py \
  --data-root /path/to/openrca/Market/cloudbed-2 \
  --query-csv /path/to/openrca/Market/cloudbed-2/query.csv \
  --record-csv /path/to/openrca/Market/cloudbed-2/record.csv \
  --baseline knowledge/market_cloudbed_2_baseline_distributions_all_metric_dates.json \
  --node-graph knowledge/market_cloudbed_2_node_container_graph.json \
  --rules knowledge/refutation_rules_v2.json \
  --trace-summary-dir trace_summaries/market_cloudbed_2_queries_full \
  --modalities metric,log,trace \
  --dataset-name market_cb2 \
  --casefold-classifier-dir knowledge/casefold_classifiers \
  --knowledge-dir knowledge/d32_casefold_market_cb2 \
  --out logs/market_cb2_predictions.csv \
  --debug-json logs/market_cb2_debug.json \
  --checkpoint-jsonl logs/market_cb2_checkpoint.jsonl
```

## Run: Telecom (51 cases)

```bash
PYTHONPATH=.:.. python3 eval/run_openrca_d32_lodo.py \
  --data-root /path/to/openrca/Telecom \
  --query-csv /path/to/openrca/Telecom/query.csv \
  --record-csv /path/to/openrca/Telecom/record.csv \
  --baseline knowledge/telecom_baseline_distributions_all_metric_dates.json \
  --node-graph knowledge/telecom_node_container_graph.json \
  --rules knowledge/refutation_rules_v2.json \
  --trace-summary-dir trace_summaries/telecom_queries_full \
  --modalities metric,log,trace \
  --dataset-name telecom \
  --casefold-classifier-dir knowledge/casefold_classifiers \
  --knowledge-dir knowledge/d32_casefold_telecom \
  --out logs/telecom_predictions.csv \
  --debug-json logs/telecom_debug.json \
  --checkpoint-jsonl logs/telecom_checkpoint.jsonl
```

## Resume from Checkpoint

If a run is interrupted, add `--resume-checkpoint` to continue from where it left off:

```bash
PYTHONPATH=.:.. python3 eval/run_openrca_d32_lodo.py ... --resume-checkpoint
```

## Evaluation

After generating predictions, evaluate with the official OpenRCA metrics:

```bash
PYTHONPATH=.:.. python3 eval/openrca_official_case_eval.py \
  --pred logs/bank_predictions.csv \
  --query /path/to/openrca/Bank/query.csv \
  --node-graph refute/knowledge/node_container_graph_2021_03_10.json \
  --out logs/bank_eval.json
```

Field-level diagnostics (time/component/reason hit rates, conditional rates):

```bash
PYTHONPATH=.:.. python3 eval/field_hit_diagnostics.py \
  --pred logs/bank_predictions.csv \
  --query /path/to/openrca/Bank/query.csv \
  --node-graph refute/knowledge/node_container_graph_2021_03_10.json \
  --out logs/bank_field_diag.json
```

## Expected Results (LODO, onset-first)

| Dataset | n | Strict | Partial (fractional) | Time | Component | Reason |
|--------|---:|:------:|:--------------------:|:----:|:---------:|:------:|
| Bank | 136 | 30.9% | 42.8% | 31.2% | 42.2% | 59.3% |
| Market CB1 | 70 | 10.0% | 27.0% | 30.9% | 46.3% | 20.0% |
| Market CB2 | 78 | 16.7% | 27.6% | 28.3% | 38.3% | 24.2% |
| Telecom | 51 | 13.7% | 21.6% | 6.5% | 21.7% | 30.3% |
| **Combined** | **335** | **20.6%** | **32.7%** | **27.2%** | **35.8%** | **37.3%** |

**Strict**: all required fields (time, component, reason) match.  
**Partial (fractional)**: average per-case field hit rate = Σ(hits/total) / n.

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--onset-first` | True | First anomaly per KPI voting |
| `--onset-first-weight` | 4.0 | Weight for onset votes |
| `--onset-decay-seconds` | 300 | Exponential decay window |
| `--component-onset-bonus` | 5.0 | Bonus for first anomaly on blamed component |
| `--dataset-name` | bank | bank / market_cb1 / market_cb2 / telecom |
| `--modalities` | metric,log,trace | Which telemetry modalities to use |
| `--no-lodo` | off | Disable LODO: in-sample evaluation (for baseline comparison) |
| `--no-onset-first` | off | Disable onset-first: use legacy all-anomaly voting |
| `--disable-family-filter` | off | Disable reason→component family filter |
| `--resume-checkpoint` | off | Resume from checkpoint JSONL |

## File Structure

```
source/
├── refute_b_v2_d32/        # Core pipeline (10 .py files)
│   ├── layer1.py           # Knowledge building (clusters, mined rules)
│   ├── layer2.py           # Main pipeline (candidate generation, scoring, prediction)
│   ├── time_anchor.py      # Onset-first time anchor voting
│   ├── reason_classifier.py # S1 XGBoost/LogReg classifier (train + inference)
│   ├── joint_candidates.py  # Joint (component, reason) candidate generator
│   ├── signature.py         # Case signature builder
│   ├── schema.py            # Reason/family buckets, KPI patterns
│   ├── evidence.py          # Evidence query + baseline anomaly detection
│   └── layer3.py            # Reserved (unused)
├── eval/                   # Entry + evaluation scripts
│   ├── run_openrca_d32_lodo.py   # Main runner (LODO)
│   ├── openrca_official_case_eval.py  # Official strict/partial evaluator
│   └── field_hit_diagnostics.py      # Field-level hit diagnostics
├── refute_b_v2/            # Supporting modules
│   ├── query_windows.py    # Query window parser
│   ├── rules.py            # Refutation rules dataclasses
│   ├── rule_engine.py      # Rule engine
│   └── trace_propagation.py # Trace propagation logic
├── refute/src/             # Data loading + baseline
│   ├── baseline_distributions.py   # P99 baseline anomaly detection
│   ├── data_loader.py      # OpenRCA data loader
│   └── node_container_split.py     # Node/container KPI classification
├── knowledge/              # Model weights + data files
│   ├── casefold_classifiers/       # Symlinks → casefold_classifiers_xd/
│   └── casefold_classifiers_xd/    # 8 JSON + 8 PKL (4 datasets × 2 folds)
├── trace_summaries/        # Pre-built trace summaries
└── refute/knowledge/       # Baseline distributions + node graphs
```

## Non-LODO Mode (In-Sample, for Baseline Comparison)

Add `--no-lodo` for in-sample evaluation (no date held out):

```bash
PYTHONPATH=.:.. python3 eval/run_openrca_d32_lodo.py ... --no-lodo
```

This builds knowledge from all cases and predicts on all cases. Do NOT use this for reporting main results — it introduces answer leakage through the knowledge base.

## Notes

- Bank trace summaries are not included; Bank runs metric+log only (trace shows as "unloaded"). This is the path that achieved 30.9% strict.
- The casefold classifiers were trained via leave-one-dataset-out (LODO): each dataset's classifier was trained on the other three.
- Reason name mapping automatically converts pipeline-internal names to scoring-points canonical names per dataset (e.g., "CPU fault" → "high CPU usage" for Bank, "container CPU load" for Market).
- Telecom has inherently sparse metric signals; time anchor and component scoring perform poorly regardless of algorithm.
