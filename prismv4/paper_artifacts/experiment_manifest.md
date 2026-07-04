# CAPE-RCA Experiment Manifest

## Repository Inputs

- Main data root: `/home/dell2/RCA513/ysj/dataset/RCAEval/{RE2,RE3}`
- Main result source: `results/prism_cht/v3_full/{RE2,RE3}-*_v3.json`
- Runner: `experiments/run_rcaeval_continuous.py`
- Adapter: `experiments/rcaeval_adapter.py`

## Main Result Provenance

| dataset | cases | AC@1 | tokens | calls |
|---|---:|---:|---:|---:|
| RE2-OB | 91 | 0.890110 | 303643 | 12 |
| RE2-SS | 90 | 0.777778 | 742128 | 29 |
| RE2-TT | 90 | 0.588889 | 0 | 0 |
| RE3-OB | 30 | 0.966667 | 301132 | 13 |
| RE3-SS | 30 | 0.866667 | 1083432 | 36 |
| RE3-TT | 30 | 0.933333 | 203294 | 6 |

## Generated Main Artifacts

- Tables under `paper_artifacts/tables/`.
- Figures under `paper_artifacts/figures/`.
- Reports under `paper_artifacts/reports/`.

## Exclusions

- RE1 is metrics-only and excluded from main multi-source experiments.
- Eadro is excluded from main comparison due protocol mismatch.
- No second dataset is admitted; `fig6` is intentionally not generated.

## Validation

`paper_artifacts/scripts/validate_artifacts.py` passed for this artifact set.
