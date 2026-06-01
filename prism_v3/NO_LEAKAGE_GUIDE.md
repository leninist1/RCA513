# PRISM v3 strict no-leakage workflow

This branch introduces a hardened inference path that rejects unverifiable
artifacts and separates label-free feature generation from evaluation-only
label loading.

## Trusted inference entrypoint

Use:

```bash
python -m prism_v3.strict_main --systems Bank --prism-noise-lab
```

Do **not** use `python -m prism_v3.main` for benchmark reporting unless you are
running a historical comparison. `strict_main` validates every externally
supplied score/model/profile artifact before delegating to the legacy runner.

## Runtime-only NoiseLab mode

The safest baseline is runtime-only NoiseLab scoring:

```bash
python -m prism_v3.strict_main \
  --systems Bank \
  --prism-noise-lab
```

Do not pass legacy CSV priors, learned models, or entity profiles until they
have been certified.

## Label-free NoiseLab feature export

Generate candidate features without reading `record.csv` or loading GT labels:

```bash
python -m prism_v3.noise_lab.safe_runner \
  --system Bank \
  --output artifacts/bank_no_gt_features.csv
```

The runner writes:

```text
artifacts/bank_no_gt_features.csv
artifacts/bank_no_gt_features.csv.manifest.json
```

The generated CSV contains telemetry-derived candidate features only. Labels
must be joined later by a separate training-only process.

## Certifying an OOF LTR score artifact

After training an LTR model with incident-grouped splits and producing truly
out-of-fold scores, certify the resulting score file:

```bash
python -m prism_v3.certify_artifact artifacts/bank_oof_ltr_scores.csv \
  --artifact-type no_leak_ltr_scores \
  --feature-pipeline-version noise_lab_no_gt_v1 \
  --anchor-sources public_query_window \
  --fit-query-ids-file artifacts/train_query_ids.txt \
  --fit-systems Bank \
  --git-commit "$(git rev-parse HEAD)"
```

Note: `certify_artifact.py` currently accepts comma-separated `--fit-query-ids`.
For long lists, generate the manifest programmatically or pass a comma-separated
string.

Then use the artifact in strict inference:

```bash
python -m prism_v3.strict_main \
  --systems Bank \
  --prism-noise-lab \
  --prism-noise-lab-scores artifacts/bank_oof_ltr_scores.csv \
  --prism-noise-lab-strategy ltr_full
```

Strict mode rejects:

- CSV artifacts without a sidecar manifest;
- artifacts whose feature generation used GT;
- artifacts with forbidden anchor sources such as `record.csv` GT time;
- artifacts whose SHA-256 does not match the manifest;
- score artifacts where the current query appears in `fit_query_ids`;
- learned convenience flags such as `--v2-all` without explicit certified
  artifact paths.

## Canary tests

Run:

```bash
pytest -q tests/test_no_leakage_guard.py \
          tests/test_no_leakage_safe_runner.py \
          tests/test_strict_main.py
```

The tests verify that:

- inference queries carrying `ground_truth` or `scoring_points` are rejected;
- legacy score CSV files without manifests are rejected;
- current-query in-fold artifacts are rejected;
- checksum tampering is detected;
- the label-free feature runner does not call `match_query_to_records`,
  `load_records`, or `record.csv`.

## Legacy NoiseLab runner

`prism_v3.noise_lab.runner` remains in the repository only for historical
comparison and evaluation diagnostics. It reads `record.csv` GT timestamps
before feature extraction and therefore **must not** be used to generate
training features or benchmark priors.
