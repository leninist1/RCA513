# RE2 Optimization Report

## Summary

The code now includes generic RE2-oriented improvements, but no after-optimization result is admitted to the main paper tables.

Accepted paper metrics therefore remain the verified v3 RE2/RE3 results in `paper_artifacts/tables/rcaeval_re2_re3_summary.csv`.

## Implemented Algorithm Changes

- Vectorized trace dependency context construction in `experiments/rcaeval_adapter.py`, replacing per-trace `iterrows()` parent lookup with a span-parent join.
- Added robust span/parent id normalization for trace CSV type inference quirks.
- Strengthened source-likelihood scoring with near-onset resource-owner signals.
- Relaxed IVD candidate filtering from 50% to 35% of max source likelihood and widened the tournament cap from 8 to 10 candidates.
- Tightened the IVD shortcut gate for noisy large-service graphs: ambiguous resource-only winners now fall back instead of returning a 0-token top-1.
- Added full-path `predicted_ranking` emission for `continuous_final` outputs.
- Added an optional owner-aware deterministic rerank fallback, but it is disabled by default because the sanity sample regressed.

## Rerun Attempts

- Full RE2-TT rerun with `max_steps=4` was started and interrupted after the first case failed to finish in a usable time window.
- A single-case `max_steps=0` full-final sanity run also stalled on the provider response to EventCausalizer.
- Optional deterministic owner-rerank was tested on the first 10 RE2-TT cases: it scored 6/10, while the admitted v3 results score 8/10 on the same prefix. This candidate is rejected.

## Decision

RE2 performance is not claimed as improved in the admitted artifact tables. The code changes are retained because they fix identifiable shortcut and preprocessing failure modes, but the after-optimization experimental result is marked `not_accepted` until a complete rerun finishes.

See `paper_artifacts/tables/rcaeval_re2_re3_before_after.csv`.
