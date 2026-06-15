# LLM pairwise gated-rerank progress - 2026-06-15

## Objective

Improve the LLM rerank judge after the first Telecom shadow/gated run showed that
candidate-level LLM judgments almost never supported GT-like beneficial
alternatives.  The current experiment focuses on a rank-blind pairwise judge fed
only by Evidence Summary Cards.

## Implemented changes

- Added rank-blind pairwise judging path:
  - `refute_b_v2_d32/llm_pairwise_judge.py`
  - CLI: `--llm-rerank-judge pairwise`
  - Pairwise input uses `candidate_a` / `candidate_b` labels and strips
    `case_id`, D32 rank, D32 score summaries, and `other_candidate_ranked_higher`.
- Extended gated rerank:
  - `apply_pairwise_gated_rerank(...)`
  - Default conservative pairwise gate promotes only when the alternative is
    strongly supported and top1 is strongly refuted.
  - The exploratory low-top1-support promotion branch is now opt-in via
    `--llm-allow-pairwise-low-top1-support`.
- Updated Evidence Summary Card for LLM use:
  - Added `reason_bucket`.
  - Added `component_metric_signal_summary`.
  - Added `candidate_positive_evidence_summary`.
  - Moved other-component signals from `counter_evidence_summary` to
    `competing_evidence_summary`.
  - Disabled/unavailable modalities no longer produce missing-support counter
    evidence; they remain in `missing_evidence_summary`.

## Telecom 10-case probe setup

Selected case ids:

```text
2, 7, 11, 22, 32, 34, 42, 43, 47, 50
```

These were chosen because D32 top-5 contained at least one counterfactual
beneficial alternative when time was kept fixed.  This is a diagnostic set, not
a random benchmark.

Model/provider:

```text
deepseek-v4-pro via OpenAI-compatible API
```

The API key was supplied through the `RCA_LLM_API_KEY` environment variable and
was not written to logs or prompts.

## Probe results

### Before card fix

Output directory:

```text
logs/telecom_llm_exp/pairwise_probe_10
```

Summary:

```json
{
  "pairwise_requests": 40,
  "parse_failures": 0,
  "preference_counts": {
    "alternative": 2,
    "top1": 4,
    "tie": 32,
    "uncertain": 2
  },
  "changed_cases": 0,
  "baseline_partial": 1,
  "pairwise_partial": 1,
  "baseline_fractional_partial_rate": 0.05,
  "pairwise_fractional_partial_rate": 0.05
}
```

Beneficial candidates by pairwise preference:

```json
{
  "tie": 12,
  "uncertain": 1,
  "top1": 4,
  "alternative": 0
}
```

### After card fix

Output directory:

```text
logs/telecom_llm_exp/pairwise_probe_10_cardfix
```

Summary with exploratory low-top1-support branch enabled during the original
probe script:

```json
{
  "pairwise_requests": 40,
  "parse_failures": 0,
  "preference_counts": {
    "alternative": 11,
    "top1": 10,
    "tie": 15,
    "uncertain": 4
  },
  "changed_cases": 1,
  "baseline_partial": 1,
  "pairwise_partial": 1,
  "baseline_fractional_partial_rate": 0.05,
  "pairwise_fractional_partial_rate": 0.05
}
```

Beneficial candidates by pairwise preference:

```json
{
  "tie": 9,
  "top1": 6,
  "alternative": 2
}
```

The one changed case was `query_011`, promoted from `db_003/db close` to
`db_007/db connection limit`.  It did not improve official case-level score.
This showed that the low-top1-support branch can create false positives, so the
committed default now disables that branch.

### After card fix with committed default conservative gate

Output directory:

```text
logs/telecom_llm_exp/pairwise_probe_10_cardfix_default_gate
```

Summary:

```json
{
  "changed_cases": 0,
  "allow_pairwise_low_top1_support": false,
  "baseline_partial": 1,
  "pairwise_partial": 1,
  "baseline_fractional_partial_rate": 0.05,
  "pairwise_fractional_partial_rate": 0.05
}
```

## Diagnosis

- The card fix improved LLM willingness to consider alternatives:
  `alternative` preferences increased from 2/40 to 11/40.
- It also improved beneficial-alternative recognition from 0/17 to 2/17.
- However, high-confidence support is still not aligned enough with official
  GT-like benefit:
  - Most beneficial alternatives remain `tie` or `top1`.
  - Some non-beneficial alternatives receive strong support.
  - Official score did not improve on this diagnostic batch.
- The main remaining issue is candidate evidence quality:
  - Many beneficial candidates only improve a partial scoring field and do not
    have distinct direct evidence.
  - Reason aliases such as `db close` vs `db connection limit` and
    `container CPU load` vs `CPU fault` still confuse evidence comparison.
  - Component-level metric signals can make the model prefer the wrong sibling
    component when reason evidence is weak.

## Next steps

1. Add reason-alias normalization into the card, not only final predictions.
2. Add a candidate-family consistency feature so sibling components with the
   same reason are compared more carefully.
3. Add a stricter pairwise audit field requiring the model to identify which
   exact candidate-level atom justifies promotion.
4. Consider a learned/calibrated post-processor over LLM pairwise scores before
   enabling any non-refute promotion path.

## Verification

Focused tests:

```bash
PYTHONPATH=confidence_aware_rca_v2_drift_focus_2026-06-02_resend/source ./.venv_d32/bin/python -m pytest \
  confidence_aware_rca_v2_drift_focus_2026-06-02_resend/source/tests/test_evidence_summarizer_llm.py \
  confidence_aware_rca_v2_drift_focus_2026-06-02_resend/source/tests/test_llm_candidate_judge.py \
  confidence_aware_rca_v2_drift_focus_2026-06-02_resend/source/tests/test_llm_pairwise_judge.py \
  confidence_aware_rca_v2_drift_focus_2026-06-02_resend/source/tests/test_llm_gated_rerank.py
```

Result:

```text
18 passed
```
