# PRISM v3 Cache Branch

This is an isolated implementation directory. It does not modify `prism_v3/`.

## Online Runner

```bash
python -m prism_v3_cache.main \
  --option PRISM \
  --systems Bank \
  --workers 1 \
  --max-queries 1 \
  --prism-fast \
  --feature-cache-dir artifacts/cache
```

Strict validation is enabled by default. For development-only recompute on invalid
artifacts:

```bash
--feature-cache-non-strict
```

## Offline Builder

```bash
python -m prism_v3_cache.cache.build_features \
  --cache-dir artifacts/cache \
  --systems Bank \
  --max-queries 10 \
  --build-cmi
```

The builder strips `ground_truth` and `scoring_points`, uses only public query
windows and telemetry-derived anchors, and writes manifests for every cache layer.

## Implemented Layers

- L0 telemetry: normalized metrics/logs/traces, entities, entity types, raw file hash.
- L1 window/signal: baseline/fault metrics, optional log/trace windows, signal matrix and masks.
- L2 graph/features: ObjectGraph parquet, graph debug, NoiseLab candidate feature parquet.
- L3 causal: CMI profile cache and per-candidate CF profile cache.

Every layer writes `manifest.json` with no-leakage flags, allowed anchor source,
pipeline version, config hash, telemetry hash, and payload checksum.
