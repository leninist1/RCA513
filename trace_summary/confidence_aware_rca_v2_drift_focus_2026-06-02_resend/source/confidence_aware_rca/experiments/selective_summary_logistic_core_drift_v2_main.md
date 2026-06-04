# Confidence-Aware RCA Summary

LODO model: `logistic`
LODO risk mode: `empirical`

Cell format for coverage columns: `coverage / selective accuracy`.

## LODO Selective Evaluation

| target | feature set | base rate | coverage@80 / acc | coverage@90 / acc | AURC | ECE |
|---|---|---:|---:|---:|---:|---:|
| partial | core:9 | 32.35% | 28.68% / 58.97% | 27.21% / 62.16% | 0.5568 | 0.1807 |
| actionable_any | core:9 | 18.31% | 18.31% / 53.85% | 15.49% / 54.55% | 0.6571 | 0.1574 |

## Forward Holdout

Heldout dates: `2021_03_23, 2021_03_24, 2021_03_25`

| target | feature set | test base rate | coverage@80 / acc | coverage@90 / acc | AURC |
|---|---|---:|---:|---:|---:|
| partial | core | 34.88% | 32.56% / 57.14% | 32.56% / 57.14% | 0.5025 |
| actionable_any | core | 28.57% | 28.57% / 50.00% | 23.81% / 60.00% | 0.6449 |
