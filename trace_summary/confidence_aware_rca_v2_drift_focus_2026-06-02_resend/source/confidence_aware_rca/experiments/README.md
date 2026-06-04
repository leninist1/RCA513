# Experiments

Generated artifacts for the confidence-aware RCA simulations.

Primary v2 files:

- `d32_timevote_confidence_features_v2.csv`
- `d32_timevote_selective_lodo_logistic_core_v2.json`
- `d32_timevote_selective_lodo_isotonic_core_v2.json`
- `d32_timevote_selective_forward_last3_core_v2.json`
- `d32_timevote_selective_forward_last3_isotonic_core_v2.json`
- `selective_summary_logistic_core_v2.md`
- `selective_summary_isotonic_core_v2.md`
- `d32_timevote_selective_lodo_logistic_target_specific_v2.json`
- `d32_timevote_selective_forward_last3_target_specific_v2.json`
- `selective_summary_logistic_target_specific_v2.md`
- `d32_timevote_selective_lodo_logistic_auto_v2.json`
- `d32_timevote_selective_forward_last3_auto_v2.json`
- `selective_summary_logistic_auto_v2.md`
- `d32_timevote_selective_lodo_logistic_threshold_transfer_v3.json`
- `d32_timevote_selective_forward_last3_threshold_transfer_v3.json`
- `selective_summary_logistic_threshold_transfer_v3.md`

The main result should be read from `selective_summary_logistic_core_v2.md`.
The old non-v2 files are kept only as debugging history; do not use them as
main evidence because the first pass had feature leakage from diagnostic
label-like columns.
