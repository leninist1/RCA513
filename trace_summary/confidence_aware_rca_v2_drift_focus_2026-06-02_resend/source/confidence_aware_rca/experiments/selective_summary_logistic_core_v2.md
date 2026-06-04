# Confidence-Aware RCA Summary

LODO model: `logistic`
LODO risk mode: `empirical`

Cell format for coverage columns: `coverage / selective accuracy`.

## LODO Selective Evaluation

| target | feature set | base rate | coverage@80 / acc | coverage@90 / acc | AURC | ECE |
|---|---|---:|---:|---:|---:|---:|
| strict | core | 11.76% | 36.76% / 14.00% | 19.85% / 14.81% | 0.8777 | 0.3332 |
| partial | core | 32.35% | 30.88% / 57.14% | 30.15% / 58.54% | 0.5702 | 0.1892 |
| actionable_any | core | 18.31% | 18.31% / 53.85% | 15.49% / 54.55% | 0.6571 | 0.1577 |
| component_reason_pair_any | core | 18.42% | 36.84% / 50.00% | 13.16% / 60.00% | 0.6123 | 0.1808 |
| time_any | core | 11.54% | 24.36% / 26.32% | 16.67% / 30.77% | 0.8032 | 0.2014 |
| component_any | core | 32.39% | 46.48% / 51.52% | 39.44% / 53.57% | 0.4779 | 0.2043 |
| reason_any | core | 29.49% | 30.77% / 50.00% | 30.77% / 50.00% | 0.5514 | 0.1925 |
| component_reason_pair | core | 7.89% | 5.26% / 0.00% | 5.26% / 0.00% | 0.8953 | 0.2508 |
| actionable | core | 11.27% | 16.90% / 8.33% | 16.90% / 8.33% | 0.9181 | 0.3084 |
| time | core | 11.54% | 24.36% / 26.32% | 16.67% / 30.77% | 0.8032 | 0.2014 |
| component | core | 22.54% | 28.17% / 10.00% | 28.17% / 10.00% | 0.8284 | 0.3451 |
| reason | core | 21.79% | 32.05% / 28.00% | 32.05% / 28.00% | 0.7307 | 0.2622 |

## Forward Holdout

Heldout dates: `2021_03_23, 2021_03_24, 2021_03_25`

| target | feature set | test base rate | coverage@80 / acc | coverage@90 / acc | AURC |
|---|---|---:|---:|---:|---:|
| strict | core | 13.95% | 37.21% / 18.75% | 34.88% / 20.00% | 0.8338 |
| partial | core | 34.88% | 32.56% / 57.14% | 32.56% / 57.14% | 0.5033 |
| actionable_any | core | 28.57% | 28.57% / 50.00% | 23.81% / 60.00% | 0.6372 |
| component_reason_pair_any | core | 28.57% | 14.29% / 50.00% | 14.29% / 50.00% | 0.5293 |
| time_any | core | 22.22% | 18.52% / 60.00% | 18.52% / 60.00% | 0.6885 |
| component_any | core | 38.10% | 90.48% / 42.11% | 52.38% / 54.55% | 0.4818 |
| reason_any | core | 28.57% | 14.29% / 50.00% | 14.29% / 50.00% | 0.5784 |
| component_reason_pair | core | 14.29% | 0.00% / - | 0.00% / - | 0.8757 |
| actionable | core | 14.29% | 47.62% / 10.00% | 28.57% / 16.67% | 0.8591 |
| time | core | 22.22% | 18.52% / 60.00% | 18.52% / 60.00% | 0.6885 |
| component | core | 23.81% | 42.86% / 22.22% | 42.86% / 22.22% | 0.8103 |
| reason | core | 17.86% | 7.14% / 50.00% | 7.14% / 50.00% | 0.7619 |
