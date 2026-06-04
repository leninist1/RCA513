# Confidence-Aware RCA Summary

LODO model: `logistic`
LODO risk mode: `empirical`

Cell format for coverage columns: `coverage / selective accuracy`.

## LODO Selective Evaluation

| target | feature set | base rate | coverage@80 / acc | coverage@90 / acc | AURC | ECE |
|---|---|---:|---:|---:|---:|---:|
| strict | core:9 | 11.76% | 36.76% / 14.00% | 19.85% / 14.81% | 0.8777 | 0.3332 |
| partial | core:9 | 32.35% | 30.88% / 57.14% | 30.15% / 58.54% | 0.5702 | 0.1892 |
| actionable_any | core:9 | 18.31% | 18.31% / 53.85% | 15.49% / 54.55% | 0.6571 | 0.1577 |
| component_reason_pair_any | core:7, target_specific:2 | 18.42% | 34.21% / 53.85% | 13.16% / 60.00% | 0.6179 | 0.1918 |
| time_any | core:5, target_specific:4 | 11.54% | 21.79% / 23.53% | 14.10% / 36.36% | 0.8178 | 0.2331 |
| component_any | core:9 | 32.39% | 46.48% / 51.52% | 39.44% / 53.57% | 0.4779 | 0.2043 |
| reason_any | core:9 | 29.49% | 30.77% / 50.00% | 30.77% / 50.00% | 0.5514 | 0.1925 |
| component_reason_pair | core:2, target_specific:7 | 7.89% | 7.89% / 0.00% | 7.89% / 0.00% | 0.8957 | 0.2443 |
| actionable | core:9 | 11.27% | 16.90% / 8.33% | 16.90% / 8.33% | 0.9181 | 0.3084 |
| time | core:5, target_specific:4 | 11.54% | 21.79% / 23.53% | 14.10% / 36.36% | 0.8178 | 0.2331 |
| component | core:1, target_specific:8 | 22.54% | 33.80% / 25.00% | 26.76% / 21.05% | 0.8022 | 0.3801 |
| reason | core:5, target_specific:4 | 21.79% | 33.33% / 30.77% | 33.33% / 30.77% | 0.7288 | 0.2374 |

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
| component | target_specific | 23.81% | 33.33% / 14.29% | 33.33% / 14.29% | 0.8178 |
| reason | core | 17.86% | 7.14% / 50.00% | 7.14% / 50.00% | 0.7619 |
