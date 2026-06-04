# Confidence-Aware RCA

This folder is a separate research route from the frozen `refute_b_v2_d32`
predictor. It does not change the RCA predictor. It learns when the existing
prediction is reliable and when the system should abstain / hand off to a human.

Current input artifacts:

- `logs/d32_lodo_timevote_trace_debug.json`
- `logs/d32_lodo_timevote_trace_official_case_eval.json`
- `logs/d32_lodo_timevote_trace_field_diag.json`

Main scripts:

- `feature_extractor.py`
  - Extracts self-knowledge features from frozen d32 debug output.
  - Produces one row per case with labels and confidence signals.

- `selective_eval.py`
  - Runs leave-one-date-out calibration/evaluation.
  - Reports coverage, selective accuracy, risk-coverage, ECE, and Brier score.
  - Defaults to the non-leaky `core` feature set.

- `forward_holdout_eval.py`
  - Trains on earlier dates and evaluates on the last N dates.
  - This is the harsher threshold-transfer test.

- `analyze_selective_results.py`
  - Converts JSON reports into compact Markdown tables.

The key research question is:

> Can an RCA system know when it is likely to be correct?

The important outputs are not only global strict/partial, but coverage at a
target reliability level, e.g. `coverage@80%` and `coverage@90%`.

Current v2 reports:

- `experiments/d32_timevote_confidence_features_v2.csv`
- `experiments/d32_timevote_selective_lodo_logistic_core_v2.json`
- `experiments/d32_timevote_selective_forward_last3_core_v2.json`
- `experiments/selective_summary_logistic_core_v2.md`
- `experiments/d32_timevote_selective_lodo_logistic_target_specific_v2.json`
- `experiments/d32_timevote_selective_forward_last3_target_specific_v2.json`
- `experiments/selective_summary_logistic_target_specific_v2.md`
- `experiments/d32_timevote_selective_lodo_logistic_auto_v2.json`
- `experiments/d32_timevote_selective_forward_last3_auto_v2.json`
- `experiments/selective_summary_logistic_auto_v2.md`
- `experiments/d32_timevote_selective_lodo_logistic_threshold_transfer_v3.json`
- `experiments/d32_timevote_selective_forward_last3_threshold_transfer_v3.json`
- `experiments/selective_summary_logistic_threshold_transfer_v3.md`

Do not cite the earlier v1 selective results as main evidence. They were useful
for debugging the route, but their feature matrix allowed diagnostic label-like
columns such as `*_hit_count` to enter the calibrator.
