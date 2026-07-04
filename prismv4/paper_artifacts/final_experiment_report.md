# CAPE-RCA Final Experiment Report

## Scope

This revision admits RCAEval RE2/RE3 as the main multi-source experiment set. RE1 is excluded from the main experiments because it is metrics-only. Eadro is archived as exploratory because its published protocol is not directly comparable to CAPE-RCA's known-fault service-localization setting.

Context and validation reports:

- `paper_artifacts/reports/context_admission_report.md`
- `paper_artifacts/reports/re2_failure_mode_report.md`
- `paper_artifacts/reports/re2_optimization_report.md`
- `paper_artifacts/reports/second_dataset_selection_report.md`
- `paper_artifacts/reports/artifact_validation_report.md`

## Main RCAEval Results

Main table: `paper_artifacts/tables/rcaeval_re2_re3_summary.csv`

| dataset_id | cases | AC@1 | AC@3 | Avg@5 | shortcut_cases | egcda_cases |
|---|---:|---:|---:|---:|---:|---:|
| RE2-OB | 91 | 0.890110 | 0.956044 | 0.938462 | 89 | 2 |
| RE2-SS | 90 | 0.777778 | 0.822222 | 0.824444 | 85 | 4 |
| RE2-TT | 90 | 0.588889 | 0.711111 | 0.691111 | 78 | 12 |
| RE3-OB | 30 | 0.966667 | 1.000000 | 0.993333 | 28 | 2 |
| RE3-SS | 30 | 0.866667 | 0.933333 | 0.926667 | 24 | 6 |
| RE3-TT | 30 | 0.933333 | 1.000000 | 0.980000 | 29 | 1 |

## RE2 Optimization Status

RE2 failure analysis identified 67 top-1 misses or failed cases before optimization, including 37 on RE2-TT. The main failure modes are shortcut/fallback over-selection and candidate/ranking miss.

Implemented code changes include vectorized trace-context construction, stronger source-likelihood/resource-owner scoring, wider IVD candidate filtering, stricter shortcut acceptance, and full-path ranking emission.

No after-optimization result is admitted. Full RE2-TT rerun stalled on provider response, and the optional deterministic owner-rerank sanity sample regressed from 8/10 to 6/10 on the first RE2-TT prefix. See `paper_artifacts/tables/rcaeval_re2_re3_before_after.csv`.

## Second Dataset Decision

No second dataset is admitted to the main comparison. Candidate audits are in `paper_artifacts/tables/second_dataset_admission.csv`.

Because no candidate passes protocol and published-baseline comparability checks, no `fig6` comparison is generated. See `paper_artifacts/notes/second_dataset_figure_not_generated.md`.

## Figures

- `paper_artifacts/figures/fig4_rcaeval_re2_re3_performance.{png,pdf}`
- `paper_artifacts/figures/fig5_rcaeval_published_baseline_comparison.{png,pdf}`
- `paper_artifacts/figures/fig7_cost_and_path_distribution.{png,pdf}`

## Obsolete Artifacts

- RE1 metrics-only raw/log/table files: `paper_artifacts/obsolete/re1_metrics_only/`
- Eadro exploratory tables/figures: `paper_artifacts/obsolete/eadro_exploratory/`
- Old all-subsets figure: `paper_artifacts/obsolete/old_figures/`

## Reproduction

```bash
PYTHONPATH=/home/dell2/RCA513/yyx python3 paper_artifacts/scripts/collect_existing_results.py
PYTHONPATH=/home/dell2/RCA513/yyx python3 paper_artifacts/scripts/compute_metrics.py
PYTHONPATH=/home/dell2/RCA513/yyx python3 paper_artifacts/scripts/select_second_dataset.py
PYTHONPATH=/home/dell2/RCA513/yyx python3 paper_artifacts/scripts/make_figures.py --skip-compute
PYTHONPATH=/home/dell2/RCA513/yyx python3 paper_artifacts/scripts/validate_artifacts.py
```
