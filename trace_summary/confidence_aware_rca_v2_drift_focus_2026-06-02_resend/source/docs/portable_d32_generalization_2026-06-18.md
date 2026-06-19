# Portable D32 Generalization Progress

Date: 2026-06-18

Goal: improve D32 so future Eadro and AIOps2021 experiments can use the same
methodology, while OpenRCA default outputs remain unchanged.

## Invariant

The existing OpenRCA entrypoint stays on the old path:

- `eval/run_openrca_d32_lodo.py` still loads `BaselineStore` directly.
- Existing candidate generation, reason classifier, time anchor, LLM shadow mode,
  and gated rerank behavior are not changed by this step.
- New modules are opt-in and not imported by the OpenRCA runner.

## Added Portable Boundary

`refute_b_v2_d32/portable_schema.py`

- Normalizes metric/log tables into D32's canonical columns:
  - metric: `timestamp, cmdb_id, kpi_name, value`
  - log: `timestamp, cmdb_id, value`
- Handles common aliases such as `service`, `instance`, `metric`, `name`,
  `message`, and millisecond timestamps.
- Defines `NormalizedIncident`, which keeps labels separate from `d32_inputs()`
  so GT cannot leak into candidate generation or scoring.

`refute_b_v2_d32/portable_adapters.py`

- Adds a schema-tolerant `TabularIncidentAdapter`.
- Adds starter constructors for Eadro-like and AIOps2021-like tabular exports.
- These adapters slice telemetry by case window and pass only telemetry/topology
  into D32 inputs.

## Added Baseline Strategy

`refute_b_v2_d32/adaptive_baseline.py`

- Provides an `AdaptiveBaselineStore` with the same `is_anomalous()` API as the
  current `BaselineStore`.
- Historical eligible baselines are preserved exactly.
- If historical data is missing or unusable, future datasets may opt into:
  - per-case pre-fault baseline,
  - cross-sectional same-window baseline,
  - within-window baseline.
- This directly addresses datasets that do not provide clean normal days.

## Added Role Inference

`refute_b_v2_d32/entity_roles.py`

- Adds portable role inference from explicit topology metadata, KPI namespaces,
  trace role hints, and only weakly from component names.
- This is the replacement path for OpenRCA-specific assumptions such as
  `docker_`, `os_`, and `db_` prefixes.

## Verification

Command:

```bash
PYTHONPATH=confidence_aware_rca_v2_drift_focus_2026-06-02_resend/source ./.venv_d32/bin/python -m pytest \
  confidence_aware_rca_v2_drift_focus_2026-06-02_resend/source/tests/test_portable_task_eval.py \
  confidence_aware_rca_v2_drift_focus_2026-06-02_resend/source/tests/test_portable_d32.py \
  confidence_aware_rca_v2_drift_focus_2026-06-02_resend/source/tests/test_evidence_summarizer_llm.py \
  confidence_aware_rca_v2_drift_focus_2026-06-02_resend/source/tests/test_llm_candidate_judge.py \
  confidence_aware_rca_v2_drift_focus_2026-06-02_resend/source/tests/test_llm_pairwise_judge.py \
  confidence_aware_rca_v2_drift_focus_2026-06-02_resend/source/tests/test_llm_gated_rerank.py
```

Result after adding the portable runner and dataset-specific evaluator tests:
`39 passed`.

## Added Portable Runner

`eval/run_portable_d32.py`

- New opt-in entrypoint for Eadro/AIOps2021-style tabular datasets.
- Defaults to empty no-GT D32 knowledge, so same-dataset labels are not used for
  prediction.
- Supports optional external `--knowledge-json` for future transfer experiments.
- Supports `--baseline-mode historical|adaptive`.
- Writes:
  - `predictions.csv`
  - `debug.json`
  - `baseline_reliability.json`
  - `input_audit.jsonl`
- Uses an adapter-local `_PortableD32Pipeline` that normalizes fallback reason
  posterior keys to reason buckets.  This avoids touching the existing OpenRCA
  pipeline implementation.

Example:

```bash
PYTHONPATH=confidence_aware_rca_v2_drift_focus_2026-06-02_resend/source \
./.venv_d32/bin/python confidence_aware_rca_v2_drift_focus_2026-06-02_resend/source/eval/run_portable_d32.py \
  --dataset aiops2021 \
  --cases-csv /path/to/cases.csv \
  --metrics-csv /path/to/metrics.csv \
  --logs-csv /path/to/logs.csv \
  --trace-summary-dir /path/to/trace_summaries \
  --baseline-mode adaptive \
  --pre-baseline-sec 3600 \
  --out logs/aiops2021_portable_predictions.csv \
  --debug-json logs/aiops2021_portable_debug.json \
  --baseline-reliability-json logs/aiops2021_baseline_reliability.json \
  --input-audit-jsonl logs/aiops2021_input_audit.jsonl
```

## Added Dataset-Specific Evaluation

`eval/evaluate_portable_task.py`

- Eadro is evaluated as Root Cause Localization:
  - `HR@1`, `HR@3`, `HR@5`
  - `NDCG@3`, `NDCG@5`
- AIOps2021 is evaluated as service localization plus anomaly-type
  classification:
  - `service_top1_accuracy`, `service_HR@3`, `service_HR@5`
  - `service_NDCG@3`, `service_NDCG@5`
  - `anomaly_type_accuracy`, `service_type_tuple_accuracy`
- OpenRCA case-level time/reason partial-credit metrics are not used for these
  two datasets.

Example:

```bash
PYTHONPATH=confidence_aware_rca_v2_drift_focus_2026-06-02_resend/source \
./.venv_d32/bin/python confidence_aware_rca_v2_drift_focus_2026-06-02_resend/source/eval/evaluate_portable_task.py \
  --dataset aiops2021 \
  --cases-csv logs/portable_exp/aiops2021_metric_adaptive_inputs_tzfix/cases.csv \
  --pred logs/portable_exp/aiops2021_metric_adaptive_run_tzfix/predictions.csv \
  --debug-json logs/portable_exp/aiops2021_metric_adaptive_run_tzfix/debug.json \
  --out-json logs/portable_exp/aiops2021_metric_adaptive_run_tzfix/task_eval.json
```

Current metric-only diagnostic results:

| Run | Task metric | Result |
| --- | --- | --- |
| Eadro-SN adaptive | HR@1 / HR@3 / HR@5 | 0.361 / 0.500 / 0.528 |
| Eadro-SN adaptive | NDCG@3 / NDCG@5 | 0.445 / 0.457 |
| Eadro-TT adaptive | HR@1 / HR@3 / HR@5 | 0.136 / 0.173 / 0.222 |
| Eadro-TT adaptive | NDCG@3 / NDCG@5 | 0.156 / 0.177 |
| AIOps2021 adaptive | service top1 / HR@3 / HR@5 | 0.491 / 0.730 / 0.824 |
| AIOps2021 adaptive | anomaly type / service+type | 0.252 / 0.239 |
| AIOps2021 hist+adaptive | service top1 / HR@3 / HR@5 | 0.371 / 0.497 / 0.654 |
| AIOps2021 hist+adaptive | anomaly type / service+type | 0.252 / 0.189 |

The strongest current failure mode is not the service ranking on AIOps2021;
it is anomaly-type collapse toward CPU.  Eadro still needs a better RCL signal
for network latency and packet-loss cases.

## Portable Reason Classifier

`eval/train_portable_reason_classifier.py`

- Trains an offline portable reason/type classifier from an explicit training
  split, e.g. AIOps2021 `data_type=train`.
- Features are telemetry-derived only:
  - portable KPI ontology bucket densities/strengths,
  - trace summary indicators when available,
  - log keyword indicators when available.
- Labels are used only as offline training targets. `run_portable_d32.py` still
  receives no labels during prediction.

`eval/run_portable_d32.py`

- Adds explicit reason-classifier loading:
  - `--reason-classifier-path`
  - `--reason-classifier-usage selector|label-only`
- Adds `--portable-profile metric-only|full-d32`.
  - `full-d32` requires a split/fold-scoped `--knowledge-json`.
- `selector` lets the classifier influence D32 candidate ranking.
- `label-only` keeps D32 service ranking unchanged and replaces only the output
  reason/type. This is a better match for AIOps2021's service+type task.
- Adds `--disable-portable-ontology` for ablations that should preserve the
  original portable service ranking.
- Adds case filtering flags for leakage-safe fold runs:
  - `--case-filter-column`
  - `--case-filter-value`
  - `--case-exclude-value`

`eval/build_portable_d32_knowledge.py`

- Builds D32 Layer-1 knowledge for portable datasets:
  - cases with signatures,
  - symptom clusters,
  - mined rules,
  - joint prior inputs.
- Supports explicit split/fold scoping:
  - `--train-split-column data_type --train-split-value train`
  - `--train-split-column source_file --exclude-split-value <heldout_file>`
- Uses labels only in the offline artifact build. Online prediction still reads
  only telemetry plus saved artifacts.

AIOps2021 command:

```bash
PYTHONPATH=confidence_aware_rca_v2_drift_focus_2026-06-02_resend/source \
./.venv_d32/bin/python confidence_aware_rca_v2_drift_focus_2026-06-02_resend/source/eval/train_portable_reason_classifier.py \
  --dataset aiops2021 \
  --cases-csv logs/portable_exp/aiops2021_metric_adaptive_inputs_tzfix/cases.csv \
  --metrics-csv logs/portable_exp/aiops2021_metric_adaptive_inputs_tzfix/metrics.csv \
  --baseline-mode adaptive \
  --pre-baseline-sec 3600 \
  --train-split-column data_type \
  --train-split-value train \
  --eval-split-value test \
  --out logs/portable_exp/aiops2021_reason_classifier_train/aiops2021_train_reason_classifier.json \
  --summary-json logs/portable_exp/aiops2021_reason_classifier_train/train_summary.json
```

AIOps2021 train-split classifier:

- train cases: 112
- train type accuracy: 0.839
- held-out test type accuracy as classifier alone: 0.447

Full-D32 AIOps2021 train-split knowledge:

```bash
PYTHONPATH=confidence_aware_rca_v2_drift_focus_2026-06-02_resend/source \
./.venv_d32/bin/python confidence_aware_rca_v2_drift_focus_2026-06-02_resend/source/eval/build_portable_d32_knowledge.py \
  --dataset aiops2021 \
  --cases-csv logs/portable_exp/aiops2021_metric_adaptive_inputs_tzfix/cases.csv \
  --metrics-csv logs/portable_exp/aiops2021_metric_adaptive_inputs_tzfix/metrics.csv \
  --modalities metric \
  --baseline-mode adaptive \
  --pre-baseline-sec 3600 \
  --train-split-column data_type \
  --train-split-value train \
  --disable-rule-validation \
  --out logs/portable_exp/aiops2021_full_d32_artifacts/d32_knowledge_train.json
```

- training cases: 112
- clusters at threshold 0.42: 1
- mined rules: 8
- cluster-threshold 0.90 variant: 4 clusters, still dominated by one 85-case cluster

Portable D32 AIOps2021 comparison:

| Run | Split | service top1 | type acc | service+type | HR@3 / HR@5 |
| --- | --- | ---: | ---: | ---: | ---: |
| adaptive old | test | 0.532 | 0.277 | 0.255 | 0.745 / 0.830 |
| selector reason-clf | test | 0.404 | 0.319 | 0.277 | 0.638 / 0.809 |
| label-only reason-clf | test | 0.532 | 0.447 | 0.277 | 0.745 / 0.830 |
| full-d32 knowledge 0.42 + label-only | test | 0.468 | 0.447 | 0.255 | 0.702 / 0.809 |
| full-d32 knowledge 0.90 + label-only | test | 0.468 | 0.447 | 0.255 | 0.723 / 0.809 |

Interpretation:

- Retraining the reason classifier does reduce AIOps2021 type collapse.
- Feeding that classifier into candidate ranking hurts service localization.
- The more dataset-appropriate design for AIOps2021 is a two-head output:
  service localization from D32's component ranking, anomaly type from the
  portable reason classifier.
- Adding train-split D32 knowledge does not improve AIOps2021 service ranking in
  metric-only mode. The training knowledge clusters are too coarse and the
  train/test component distribution shifts, so cluster/joint priors can hurt
  service localization.

Eadro leakage-safe fold setup:

- SN has 4 `source_file` folds, 9 cases each.
- TT has 9 `source_file` folds, 9 cases each.
- Use leave-one-source-file-out knowledge:

```bash
PYTHONPATH=confidence_aware_rca_v2_drift_focus_2026-06-02_resend/source \
./.venv_d32/bin/python confidence_aware_rca_v2_drift_focus_2026-06-02_resend/source/eval/build_portable_d32_knowledge.py \
  --dataset eadro \
  --cases-csv logs/portable_exp/eadro_sn_metric_adaptive_inputs/cases.csv \
  --metrics-csv logs/portable_exp/eadro_sn_metric_adaptive_inputs/metrics.csv \
  --modalities metric \
  --baseline-mode adaptive \
  --pre-baseline-sec 3600 \
  --train-split-column source_file \
  --exclude-split-value '<heldout_source_file>' \
  --disable-rule-validation \
  --out logs/portable_exp/eadro_sn_full_d32_folds/knowledge_without_<fold>.json
```

Then run the held-out fold with:

```bash
PYTHONPATH=confidence_aware_rca_v2_drift_focus_2026-06-02_resend/source \
./.venv_d32/bin/python confidence_aware_rca_v2_drift_focus_2026-06-02_resend/source/eval/run_portable_d32.py \
  --dataset eadro \
  --portable-profile full-d32 \
  --cases-csv logs/portable_exp/eadro_sn_metric_adaptive_inputs/cases.csv \
  --metrics-csv logs/portable_exp/eadro_sn_metric_adaptive_inputs/metrics.csv \
  --modalities metric \
  --baseline-mode adaptive \
  --pre-baseline-sec 3600 \
  --knowledge-json logs/portable_exp/eadro_sn_full_d32_folds/knowledge_without_<fold>.json \
  --case-filter-column source_file \
  --case-filter-value '<heldout_source_file>' \
  --out logs/portable_exp/eadro_sn_full_d32_folds/predictions_<fold>.csv \
  --debug-json logs/portable_exp/eadro_sn_full_d32_folds/debug_<fold>.json
```

## Next Step

The next experiments should:

- improve AIOps2021 network latency vs packet-loss discrimination,
- add a leakage-safe Eadro fold setup before training any Eadro classifier,
- build an Eadro RCL-specific service ranker using portable ontology and trace
  summary signals without scoring reason/time as OpenRCA fields.

## Multimodal Alignment Update

Added `eval/build_portable_multimodal_inputs.py` to recover the original D32
multimodal path on portable datasets.  The script converts raw dataset assets
into the existing portable runner contract:

- `logs.csv`: normalized `timestamp,cmdb_id,value`
- `trace_summaries/<case_id>.json`: D32-compatible `service_stats`,
  `edge_stats`, `events.slow_edges`, and `events.first_anomalous_service`
- `topology.json`: component inventory with explicit portable roles
- `summary.json`: source and modality audit

Important alignment details:

- AIOps2021 trace timestamps are mixed seconds/milliseconds; the builder
  normalizes them per row.
- AIOps2021 case-day routing uses Asia/Shanghai local day to match raw folders.
- Eadro `source_file` values map to `.adapter_work_v22/extracted_{sn,tt}/data`
  telemetry folders.
- Eadro spans are 8 hours ahead of the current portable case/metric axis; the
  builder defaults `--eadro-trace-time-shift-sec -28800`.
- Topology explicit roles now dominate KPI-name heuristics in
  `refute_b_v2_d32/entity_roles.py`; this avoids classifying gateway/app
  services as databases merely because they expose session or memory KPIs.

AIOps2021 single-case smoke:

```bash
./.venv_d32/bin/python confidence_aware_rca_v2_drift_focus_2026-06-02_resend/source/eval/build_portable_multimodal_inputs.py \
  --dataset aiops2021 \
  --cases-csv confidence_aware_rca_v2_drift_focus_2026-06-02_resend/source/logs/portable_exp/aiops2021_metric_adaptive_inputs_tzfix/cases.csv \
  --metrics-csv confidence_aware_rca_v2_drift_focus_2026-06-02_resend/source/logs/portable_exp/aiops2021_metric_adaptive_inputs_tzfix/metrics.csv \
  --raw-root /home/dell2/RCA-dataset-ysj/AIOps2021/aiops2021-2 \
  --out-dir confidence_aware_rca_v2_drift_focus_2026-06-02_resend/source/logs/portable_exp/aiops2021_multimodal_inputs_smoke \
  --max-cases 1 \
  --chunksize 300000
```

Observed output:

- log rows: 7608
- trace summaries: 1 present, 0 empty
- trace span rows for `aiops21_000`: 74351
- services in trace summary: 12
- edge stats: 16

Eadro SN single-case smoke:

```bash
./.venv_d32/bin/python confidence_aware_rca_v2_drift_focus_2026-06-02_resend/source/eval/build_portable_multimodal_inputs.py \
  --dataset eadro \
  --cases-csv confidence_aware_rca_v2_drift_focus_2026-06-02_resend/source/logs/portable_exp/eadro_sn_metric_adaptive_inputs/cases.csv \
  --metrics-csv confidence_aware_rca_v2_drift_focus_2026-06-02_resend/source/logs/portable_exp/eadro_sn_metric_adaptive_inputs/metrics.csv \
  --raw-root /home/dell2/RCA513/syh/datasets/Eadro/.adapter_work_v22 \
  --out-dir confidence_aware_rca_v2_drift_focus_2026-06-02_resend/source/logs/portable_exp/eadro_sn_multimodal_inputs_smoke \
  --max-cases 1
```

Observed output:

- log rows: 1456
- trace summaries: 1 present, 0 empty
- trace span rows for `sn_000`: 22542
- services in trace summary: 12
- edge stats: 16
- top slow edge: `nginx-web-server -> compose-post-service`
- runner smoke predicted `text-service / CPU fault` for `sn_000`, matching the
  single-case GT service/reason.

Eadro full multimodal no-GT-knowledge runs:

| Run | HR@1 | HR@3 | HR@5 | NDCG@3 | NDCG@5 | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| SN metric-only adaptive | 0.361 | 0.500 | 0.528 | 0.445 | 0.457 | earlier baseline |
| SN metric+log+trace adaptive | 0.361 | 0.611 | 0.694 | 0.508 | 0.543 | 36/36 trace present, 26950 log rows |
| TT metric-only adaptive | 0.136 | 0.173 | 0.222 | 0.156 | 0.177 | earlier baseline |
| TT metric+log+trace adaptive | 0.346 | 0.420 | 0.469 | 0.386 | 0.407 | 78/81 trace present, logs empty |

Interpretation:

- Multimodal alignment recovers a real part of D32's original strength on Eadro.
- SN top-1 did not improve, but GT coverage in the top ranks improved sharply;
  this is a good target for a trace-aware reranker.
- TT improved at top-1 and top-k even without logs, so trace summary alone is
  already useful for TrainTicket-style RCL.
- Network latency/loss cases remain much weaker than CPU cases.  For TT, CPU
  reaches HR@1 = 1.0, while network latency remains near zero; the next model
  work should focus on network-specific trace edge calibration rather than more
  CPU heuristics.

Verification:

```bash
PYTHONPATH=confidence_aware_rca_v2_drift_focus_2026-06-02_resend/source \
./.venv_d32/bin/python -m pytest confidence_aware_rca_v2_drift_focus_2026-06-02_resend/source/tests -q
```

Result: `48 passed`.

Next full experiments:

- Generate full AIOps2021 multimodal inputs, then rerun metric+log+trace with
  label-only reason classifier on the test split.
- Generate Eadro SN/TT full multimodal inputs and rerun leakage-safe
  leave-one-source-file-out folds with `--modalities metric,log,trace`.
- Diagnose AIOps2021 trace `slow_edges`: the first smoke case had no slow edge
  above threshold but did have a cross-sectional first anomalous service, so the
  threshold or edge baseline may need dataset-specific calibration.
