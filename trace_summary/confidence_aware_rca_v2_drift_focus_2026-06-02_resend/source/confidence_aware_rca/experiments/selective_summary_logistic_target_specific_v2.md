# Confidence-Aware RCA Summary

LODO model: `logistic`
LODO risk mode: `empirical`

Cell format for coverage columns: `coverage / selective accuracy`.

## LODO Selective Evaluation

| target | feature set | base rate | coverage@80 / acc | coverage@90 / acc | AURC | ECE |
|---|---|---:|---:|---:|---:|---:|
| strict | target_specific | 11.76% | 36.76% / 14.00% | 19.85% / 14.81% | 0.8777 | 0.3332 |
| partial | target_specific | 32.35% | 30.88% / 57.14% | 30.15% / 58.54% | 0.5702 | 0.1892 |
| actionable_any | target_specific | 18.31% | 18.31% / 53.85% | 15.49% / 54.55% | 0.6571 | 0.1577 |
| component_reason_pair_any | target_specific | 18.42% | 34.21% / 53.85% | 13.16% / 60.00% | 0.6163 | 0.1757 |
| time_any | target_specific | 11.54% | 16.67% / 30.77% | 7.69% / 50.00% | 0.8009 | 0.1745 |
| component_any | target_specific | 32.39% | 40.85% / 55.17% | 22.54% / 68.75% | 0.5191 | 0.2302 |
| reason_any | target_specific | 29.49% | 32.05% / 44.00% | 32.05% / 44.00% | 0.5965 | 0.2052 |
| component_reason_pair | target_specific | 7.89% | 7.89% / 0.00% | 7.89% / 0.00% | 0.8817 | 0.2461 |
| actionable | target_specific | 11.27% | 16.90% / 8.33% | 16.90% / 8.33% | 0.9181 | 0.3084 |
| time | target_specific | 11.54% | 16.67% / 30.77% | 7.69% / 50.00% | 0.8009 | 0.1745 |
| component | target_specific | 22.54% | 33.80% / 25.00% | 26.76% / 21.05% | 0.7972 | 0.3631 |
| reason | target_specific | 21.79% | 26.92% / 28.57% | 24.36% / 31.58% | 0.7292 | 0.2419 |

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
| reason_any | target_specific | 28.57% | 14.29% / 50.00% | 14.29% / 50.00% | 0.5914 |
| component_reason_pair | target_specific | 14.29% | 0.00% / - | 0.00% / - | 0.7921 |
| actionable | target_specific | 14.29% | 47.62% / 10.00% | 28.57% / 16.67% | 0.8591 |
| time | target_specific | 22.22% | 18.52% / 60.00% | 18.52% / 60.00% | 0.6961 |
| component | target_specific | 23.81% | 33.33% / 14.29% | 33.33% / 14.29% | 0.8178 |
| reason | target_specific | 17.86% | 17.86% / 20.00% | 17.86% / 20.00% | 0.8082 |
