# Confidence-Aware RCA Results

Base predictor: frozen d32 time-vote LODO.

Dataset: OpenRCA Bank, 136 cases.

## Status

The v1 selective results are deprecated. They accidentally allowed non-feature
diagnostic columns such as `*_hit_count` and `*_hit_any_label` into the feature
matrix, which inflated the apparent self-knowledge quality.

The current report uses the v2 non-leaky feature table:

- `confidence_aware_rca/experiments/d32_timevote_confidence_features_v2.csv`

The calibrator excludes:

- all `*_label` columns
- all `*_required` / `*_required_total` columns
- all `*_hit_count` columns

## Feature Table

Input artifacts:

- `logs/d32_lodo_timevote_trace_debug.json`
- `logs/d32_lodo_timevote_trace_official_case_eval.json`
- `logs/d32_lodo_timevote_trace_field_diag.json`

Case-level positives in v2:

| target | positives |
|---|---:|
| strict | 16 / 136 |
| partial | 44 / 136 |
| time case-all | 9 / required cases |
| component case-all | 16 / required cases |
| reason case-all | 17 / required cases |
| component+reason case-all | 3 / required cases |
| actionable case-all | 8 / required cases |

## Main LODO Result

Primary file:

- `confidence_aware_rca/experiments/d32_timevote_selective_lodo_logistic_core_v2.json`
- summary: `confidence_aware_rca/experiments/selective_summary_logistic_core_v2.md`

Core logistic, empirical threshold transfer:

| target | base rate | coverage@90 | selective accuracy@90 |
|---|---:|---:|---:|
| strict | 11.76% | 19.85% | 14.81% |
| partial | 32.35% | 30.15% | 58.54% |
| time_any | 11.54% | 16.67% | 30.77% |
| component_any | 32.39% | 39.44% | 53.57% |
| reason_any | 29.49% | 30.77% | 50.00% |
| component_reason_pair_any | 18.42% | 13.16% | 60.00% |
| actionable_any | 18.31% | 15.49% | 54.55% |

## Forward Last3 Holdout

Heldout dates:

- `2021_03_23`
- `2021_03_24`
- `2021_03_25`

Primary file:

- `confidence_aware_rca/experiments/d32_timevote_selective_forward_last3_core_v2.json`

Core logistic, empirical threshold transfer:

| target | test base rate | coverage@90 | selective accuracy@90 |
|---|---:|---:|---:|
| strict | 13.95% | 34.88% | 20.00% |
| partial | 34.88% | 32.56% | 57.14% |
| time_any | 22.22% | 18.52% | 60.00% |
| component_any | 38.10% | 52.38% | 54.55% |
| reason_any | 28.57% | 14.29% | 50.00% |
| actionable_any | 28.57% | 23.81% | 60.00% |

## Target-Specific Head Test

New files:

- `confidence_aware_rca/experiments/d32_timevote_selective_lodo_logistic_target_specific_v2.json`
- `confidence_aware_rca/experiments/d32_timevote_selective_forward_last3_target_specific_v2.json`
- `confidence_aware_rca/experiments/selective_summary_logistic_target_specific_v2.md`

Design:

- `time*` heads use time-anchor stability and modality vote features.
- `component*` heads use component evidence prior, cluster similarity, component
  entropy, and component modality agreement.
- `reason*` heads use reason prior, reason/rule validation, and reason-bucket
  entropy.
- `partial` / `actionable*` keep the broader mixed feature view.

Result:

| target | core LODO acc@90 | target-specific LODO acc@90 | core forward acc@90 | target-specific forward acc@90 |
|---|---:|---:|---:|---:|
| partial | 58.54% | 58.54% | 57.14% | 57.14% |
| time_any | 30.77% | 50.00% | 60.00% | 60.00% |
| component_any | 53.57% | 68.75% | 54.55% | 44.44% |
| reason_any | 50.00% | 44.00% | 50.00% | 50.00% |
| actionable_any | 54.55% | 54.55% | 60.00% | 60.00% |

Interpretation:

- Target-specific heads help in LODO for `time_any` and `component_any`.
- The improvement does not transfer cleanly to forward last3, especially for
  `component_any`.
- Therefore the next reliable direction is not to simply replace core with
  target-specific. We need inner-validation model selection or grouped
  threshold transfer before claiming this as a deployment-grade improvement.

## Inner-Validation Auto Selector

New files:

- `confidence_aware_rca/experiments/d32_timevote_selective_lodo_logistic_auto_v2.json`
- `confidence_aware_rca/experiments/d32_timevote_selective_forward_last3_auto_v2.json`
- `confidence_aware_rca/experiments/selective_summary_logistic_auto_v2.md`

Protocol:

- Outer evaluation remains LODO or forward last3.
- For every outer training split, the selector runs inner LODO over training
  dates only.
- It compares `core` and `target_specific`.
- It chooses the feature set with lower inner LODO AURC; ties are broken by
  selective utility at the target reliability threshold.

Selected feature sets in LODO:

| target | selected feature sets across 9 dates |
|---|---|
| strict | core:9 |
| partial | core:9 |
| time_any | core:5, target_specific:4 |
| component_any | core:9 |
| reason_any | core:9 |
| component_reason_pair_any | core:7, target_specific:2 |
| actionable_any | core:9 |

Main auto result:

| target | auto LODO acc@90 | auto forward acc@90 |
|---|---:|---:|
| partial | 58.54% | 57.14% |
| time_any | 36.36% | 60.00% |
| component_any | 53.57% | 54.55% |
| reason_any | 50.00% | 50.00% |
| actionable_any | 54.55% | 60.00% |

Interpretation:

- Auto selection mostly chooses `core`, which is a useful negative finding:
  target-specific heads are not yet consistently better under time-shift.
- The best current main result is still the `core` / `auto` partial confidence
  result, not full strict RCA.
- The next improvement should focus on the selector objective itself, especially
  threshold-transfer reliability rather than only ranking quality.

## Threshold-Transfer Selector v3

New files:

- `confidence_aware_rca/experiments/d32_timevote_selective_lodo_logistic_threshold_transfer_v3.json`
- `confidence_aware_rca/experiments/d32_timevote_selective_forward_last3_threshold_transfer_v3.json`
- `confidence_aware_rca/experiments/selective_summary_logistic_threshold_transfer_v3.md`

Protocol:

- `--feature-set auto`
- candidates: `core,target_specific`
- selection objective: `threshold_transfer`
- inner target accuracy: `0.90`
- minimum inner coverage: `0.20`
- reliability score: Wilson lower bound with `z = 1.64`

Main v3 result:

| target | v3 LODO acc@90 | v3 forward acc@90 |
|---|---:|---:|
| partial | 58.54% | 57.14% |
| time_any | 36.36% | 60.00% |
| component_any | 53.85% | 44.44% |
| reason_any | 44.00% | 50.00% |
| actionable_any | 54.55% | 60.00% |

Interpretation:

- v3 preserves the main `partial` and `actionable_any` result.
- v3 does not improve the main claim over the simpler `core` / AURC-auto
  protocol.
- The selector becomes more willing to choose `target_specific`, but this does
  not transfer reliably to forward last3; `component_any` drops from 54.55% to
  44.44% in forward.
- This is a useful negative result: lower-bound threshold selection alone is
  not enough. The next selector needs to model date drift or use a validation
  objective that directly penalizes unstable future-date performance.

## Interpretation

The current d32-based self-knowledge route is real, but weaker than the leaked
v1 result. It should not be presented as high-confidence full RCA yet.

The useful targets are:

- official partial: strong lift from about 32-35% base rate to about 57-59%
  selective accuracy at about 30-33% coverage.
- `component_any` and `actionable_any`: useful industrial triage targets, but
  threshold transfer is still unstable.
- `time_any`: forward holdout looks good at small coverage, but the sample is
  small.

The weak targets are:

- strict full triple: no meaningful high-confidence subset yet.
- case-all component/reason/actionable: too strict for the current predictor and
  should be treated as diagnostics, not the first paper claim.

## Next Engineering Target

The next module to improve is not the d32 RCA predictor itself. The immediate
target is a target-specific confidence head:

- separate heads for `partial`, `component_any`, `reason_any`, and
  `actionable_any`
- date-stable threshold transfer
- group-aware calibration by reason bucket / modality pattern
- explicit feature ablation for top-k disagreement, time anchor stability,
  modality agreement, and mined-rule validation features

Conformal risk control was also tested, but with this sample size it is too
conservative and should be kept as a later robustness experiment rather than
the main result.
