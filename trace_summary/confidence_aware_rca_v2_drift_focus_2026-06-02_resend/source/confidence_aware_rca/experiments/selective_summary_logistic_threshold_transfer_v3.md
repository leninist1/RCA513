# Confidence-Aware RCA Summary

LODO model: `logistic`
LODO risk mode: `empirical`

Cell format for coverage columns: `coverage / selective accuracy`.

## LODO Selective Evaluation

| target | feature set | base rate | coverage@80 / acc | coverage@90 / acc | AURC | ECE |
|---|---|---:|---:|---:|---:|---:|
| strict | target_specific:9 | 11.76% | 36.76% / 14.00% | 19.85% / 14.81% | 0.8777 | 0.3332 |
| partial | target_specific:9 | 32.35% | 30.88% / 57.14% | 30.15% / 58.54% | 0.5702 | 0.1892 |
| actionable_any | target_specific:9 | 18.31% | 18.31% / 53.85% | 15.49% / 54.55% | 0.6571 | 0.1577 |
| component_reason_pair_any | core:5, target_specific:4 | 18.42% | 34.21% / 53.85% | 13.16% / 60.00% | 0.6179 | 0.1949 |
| time_any | core:3, target_specific:6 | 11.54% | 21.79% / 23.53% | 14.10% / 36.36% | 0.8180 | 0.2104 |
| component_any | core:5, target_specific:4 | 32.39% | 42.25% / 53.33% | 36.62% / 53.85% | 0.5118 | 0.2426 |
| reason_any | core:6, target_specific:3 | 29.49% | 32.05% / 44.00% | 32.05% / 44.00% | 0.5797 | 0.2037 |
| component_reason_pair | core:2, target_specific:7 | 7.89% | 5.26% / 0.00% | 5.26% / 0.00% | 0.8895 | 0.2522 |
| actionable | target_specific:9 | 11.27% | 16.90% / 8.33% | 16.90% / 8.33% | 0.9181 | 0.3084 |
| time | core:3, target_specific:6 | 11.54% | 21.79% / 23.53% | 14.10% / 36.36% | 0.8180 | 0.2104 |
| component | core:5, target_specific:4 | 22.54% | 26.76% / 10.53% | 26.76% / 10.53% | 0.8336 | 0.3224 |
| reason | core:5, target_specific:4 | 21.79% | 32.05% / 28.00% | 29.49% / 30.43% | 0.7416 | 0.2305 |

## Forward Holdout

Heldout dates: `2021_03_23, 2021_03_24, 2021_03_25`

| target | feature set | test base rate | coverage@80 / acc | coverage@90 / acc | AURC |
|---|---|---:|---:|---:|---:|
| strict | target_specific | 13.95% | 37.21% / 18.75% | 34.88% / 20.00% | 0.8338 |
| partial | target_specific | 34.88% | 32.56% / 57.14% | 32.56% / 57.14% | 0.5033 |
| actionable_any | target_specific | 28.57% | 28.57% / 50.00% | 23.81% / 60.00% | 0.6372 |
| component_reason_pair_any | target_specific | 28.57% | 14.29% / 50.00% | 14.29% / 50.00% | 0.5555 |
| time_any | target_specific | 22.22% | 18.52% / 60.00% | 18.52% / 60.00% | 0.6961 |
| component_any | target_specific | 38.10% | 61.90% / 38.46% | 42.86% / 44.44% | 0.5373 |
| reason_any | core | 28.57% | 14.29% / 50.00% | 14.29% / 50.00% | 0.5784 |
| component_reason_pair | target_specific | 14.29% | 0.00% / - | 0.00% / - | 0.7921 |
| actionable | target_specific | 14.29% | 47.62% / 10.00% | 28.57% / 16.67% | 0.8591 |
| time | target_specific | 22.22% | 18.52% / 60.00% | 18.52% / 60.00% | 0.6961 |
| component | target_specific | 23.81% | 33.33% / 14.29% | 33.33% / 14.29% | 0.8178 |
| reason | core | 17.86% | 7.14% / 50.00% | 7.14% / 50.00% | 0.7619 |
