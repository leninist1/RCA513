# PRISM v3 Merge Notes

PRISM v3 is an isolated package-local merge of:

- `yyx/prism_v2`: PRISM v2 feature switches (`--v2-all`, learned modules, hierarchical priors, active inference, MCTS, active perception).
- `ysj/noise_reason_taxonomy_handoff_20260526`: Bank reason canonicalization and diagnostics.
- `syh/prism_runtime_handoff_20260526`: runtime diagnostics and top-k counterfactual controls.

## Smoke Command Used

```bash
cd /home/dell2/RCA513/yyx
python -u -m prism_v3.main \
  --option PRISM --systems Bank --workers 1 --max-queries 1 \
  --output prism_v3/results \
  --prism-noise-lab \
  --prism-noise-lab-scores /home/dell2/RCA513/yyx/rca513/results/noise_lab_ltr_scores_bankfull_ndcg_noprefix_nobase_Bank_20260525_212615.csv \
  --prism-noise-lab-strategy ltr_full \
  --v2-all --v2-mcts-sims 5 \
  --prism-cf-profile-top-k 2 \
  --prism-cf-profiles-max-calls 1 \
  --prism-cf-degradation-entity-top-k 6 \
  --prism-final-cf-top-k 2 \
  --prism-final-cf-only-on-conflict
```

Result:

- Output: `prism_v3/results/option_PRISM_20260528_160939.json`
- Smoke status: ran through PRISM v2-all with counterfactual engine enabled.
- Single query runtime: about 90s after telemetry load, with `final_cf_sec` about 53s.

## Suggested Full-Run Starting Point

Use a small date or query subset first. The final counterfactual discriminator is still the dominant cost.

```bash
python -u -m prism_v3.main \
  --option PRISM --systems Bank --workers 1 --max-queries 10 \
  --output prism_v3/results \
  --prism-noise-lab \
  --prism-noise-lab-scores /home/dell2/RCA513/yyx/rca513/results/noise_lab_ltr_scores_bankfull_ndcg_noprefix_nobase_Bank_20260525_212615.csv \
  --prism-noise-lab-strategy ltr_full \
  --v2-all --v2-mcts-sims 5 \
  --prism-cf-profile-top-k 2 \
  --prism-cf-profiles-max-calls 1 \
  --prism-cf-degradation-entity-top-k 6 \
  --prism-final-cf-top-k 2 \
  --prism-final-cf-only-on-conflict
```

For a faster but still counterfactual-profile-enabled diagnostic, use `--prism-mid`; it keeps CF profiles enabled but disables final CF.
