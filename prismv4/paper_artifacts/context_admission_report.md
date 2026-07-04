# Context Admission Report

Generated at: 2026-07-04

## Scanned Directories

- `results/prism_cht/v3_full/`: indexed CAPE-RCA result JSON filenames, sizes, top-level schema, case count, status/cost/ranking field availability.
- `results/prism_cht/`: indexed RCAEval and Eadro result JSON filenames, sizes, and candidate provenance; ignored unrelated OpenRCA/AIOps/debug/smoke outputs for the main paper tables unless used only as compatibility notes.
- `experiments/`: searched runner/adapter support for RCAEval and Eadro without loading raw telemetry.
- `scripts/`: searched existing run scripts for reproducibility commands.
- `/home/dell2/RCA513/ysj/dataset/RCAEval/`: listed dataset/subset directory names only.
- `/home/dell2/RCA513/syh/datasets/Eadro/.adapter_work_v22/`: listed Eadro extracted case directory names only.

## Files Admitted Into Context

Admitted content is limited to compact metadata, schema summaries, selected code references, and generated aggregate tables.

- `experiments/run_rcaeval_continuous.py`: runner CLI and result schema references.
- `experiments/rcaeval_adapter.py`: RCAEval/Eadro adapter entry points and dataset discovery references.
- `prism_cht/noise_native_llm_agent.py`: CAPE-RCA/NoiseNative execution output fields if needed.
- `prism_cht/action_schema.py`: diagnostic action labels if needed.
- `prism_cht/noiselab_registry.py`: tool registry references if needed.
- `prism_cht/tournament_types.py`: result/status type references if needed.
- `results/prism_cht/v3_full/*.json`: parsed only through scripts for compact summaries and case-level fields.
- `results/prism_cht/Eadro-TT_ambig_full.json` and `results/prism_cht/Eadro-SN_ambig_full.json`: parsed only through scripts for compact summaries and case-level fields.

## Files Excluded From Context

- Full raw telemetry under RCAEval and Eadro dataset directories: excluded because it may contain large metrics/logs/traces and is not needed for aggregate tables.
- Full `*.json.llm_io.jsonl` files: excluded because they contain raw model I/O traces; may be used only via targeted script extraction for one case study if needed.
- Full log files under `results/prism_cht/*.log`: excluded except for compact command/provenance references.
- `results/baseline_results/`: excluded from main published baseline tables because the task requires published/official numbers, not locally reproduced baselines.
- OpenRCA, AIOps2021, debug, smoke, and old retry outputs: excluded from main Chapter 3 results unless explicitly marked as non-main compatibility notes.
- Python caches, notebook checkpoints, temporary logs, and unrelated historical experiment directories: excluded.

## Existing Result Coverage

Existing CAPE-RCA-Full/adaptive-path candidate results:

- RCAEval RE2: `RE2-OB`, `RE2-SS`, `RE2-TT` covered in `results/prism_cht/v3_full/`.
- RCAEval RE3: `RE3-OB`, `RE3-SS`, `RE3-TT` covered in `results/prism_cht/v3_full/`.
- Eadro: `Eadro-TT` and `Eadro-SN` covered by `results/prism_cht/Eadro-TT_ambig_full.json` and `results/prism_cht/Eadro-SN_ambig_full.json`.

Current missing main-paper result coverage at admission time:

- RCAEval RE1: `RE1-OB`, `RE1-SS`, `RE1-TT` had dataset directories but no existing result JSON was found under `results/`.

## Existing Result Schema Observations

- Top-level fields commonly include `dataset`, `system`, `mode`, `total_cases`, `top1_accuracy`, `top1_hits`, `elapsed_sec`, `cost`, and `results`.
- Case-level fields commonly include `case_id`, `expected_component`, `predicted_component`, `predicted_ranking`, `hit`, `status`, `elapsed_sec`, `error`, and `token_usage`.
- Some continuous/EG-CDA cases may lack `predicted_ranking`; scripts must handle missing rankings gracefully and record notes.
- Status values observed include `lightweight_ivd_shortcut` and `continuous_final`; both are treated as CAPE-RCA-Full adaptive-path outcomes for this paper.

## Metrics Directly Computable From Existing Results

- RCAEval RE2/RE3: cases, hits@1, AC@1, hits@3/AC@3 when `predicted_ranking` is available, Avg@5 using the documented script formula, average rank when available, total/average tokens, total/average calls, elapsed time, path distribution, failed/error/timeout counts, and case-level predictions.
- Eadro TT/SN: cases, HR@1, HR@3, HR@5, NDCG@3, NDCG@5 when `predicted_ranking` or top-1 prediction is available, cost, elapsed time, path distribution, failed/error/timeout counts, and case-level predictions.
- Error analysis: computable from case-level wrong/missing/timeout cases, with conservative error-type assignment from available status, ranking, and notes fields.

## Metrics Requiring New Runs Or Format Caveats

- RCAEval RE1 requires new CAPE-RCA runs unless a result file outside the scanned locations is later identified.
- For any case without `predicted_ranking`, Top-k and rank-based metrics cannot be fully computed; the scripts will mark these cases and calculate metrics over available ranking data only where appropriate.
- Published baseline tables require external/public-source verification. If reliable published values cannot be obtained, the baseline row will be marked `TODO` rather than fabricated.
- Eadro comparison must be documented as a known-fault localization setting, not a full end-to-end anomaly-detection-plus-localization setting.

## Minimum Necessary Experiment Candidates

Only these missing runs should be considered after table parsing:

- `RE1-OB` using `/home/dell2/RCA513/ysj/dataset/RCAEval/RE1`
- `RE1-SS` using `/home/dell2/RCA513/ysj/dataset/RCAEval/RE1`
- `RE1-TT` using `/home/dell2/RCA513/ysj/dataset/RCAEval/RE1`

Before running them, generated scripts should first parse existing results and confirm no usable RE1 result is present.

## Post-Admission Execution Addendum

- RE1 adapter compatibility was added for metrics-only `data.csv` files.
- RE1 results were generated under `paper_artifacts/raw/RE1-*_cape_rca_full.json`.
- RE1 is reported as metrics-only compatibility evidence. Strong IVD consensus cases use the adaptive shortcut path. Non-strong RE1 metrics-only cases are explicitly marked `timeout_or_budget_exhausted`; EG-CDA fallback was not fabricated after an interrupted full-fallback attempt showed minute-scale, high-token behavior.
- Interrupted/smoke LLM I/O artifacts were excluded from the commit set; they are not parsed as main results.
