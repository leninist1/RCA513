# Prism-CHT Handoff

This import brings the Prism-CHT RCA agent from `docs/cht53c-manual-live-smoke-plan` into the current `trace_summary_d32v2` branch. Trace summary and Prism are separate algorithm lines; keep Prism work isolated under `prismv4/` unless the project owner explicitly asks to couple them.

## Provenance

- Target branch: `trace_summary_d32v2`
- Source branch: `docs/cht53c-manual-live-smoke-plan`
- Source commit at import time: `da0a2a275a8107cab7babbca6de6c928f512a748`
- Target HEAD before import: `1e7813f89f6f1bac7b129b3dcf529e345955b042`
- Imported surface: `prismv4/prism_cht/`, CHT/RCAEval experiment runners, CHT tests, CHT docs, `cht_provider_smoke.py`, and this handoff document.
- Deliberately not imported: `prismv4/results/`, `prismv4/artifacts/`, caches, and `prismv4/prism_la/`.

## Mental Model

Prism-CHT is an LLM-based RCA agent for RCAEval. The current design direction is:

1. NoiseLab is the feature fusion and retrieval layer. It exposes normalized observations from traces, metrics, logs, topology, and derived facts.
2. EventCausalizer organizes multimodal observations into event-level causalized features so the final diagnostic prompt sees causal candidates rather than raw fragments only.
3. NoiseNative/continuous mode performs a coherent diagnostic pass over the case, preserving intermediate evidence in audit logs, then emits a strict JSON answer.

The recent RCAEval debugging found that explicit step-by-step exploration can make the LLM over-explore and postpone diagnosis. Continuous reasoning is preferred: give the model enough causalized feedback, then force a best root-cause answer when the budget is exhausted.

## Important Files

- `prismv4/experiments/run_rcaeval_continuous.py`: main continuous RCAEval runner.
- `prismv4/experiments/rcaeval_adapter.py`: RCAEval loading, case packaging, evaluation helpers, and leakage-sensitive fields.
- `prismv4/experiments/run_rcaeval_cht.py`: older Prism-CHT runner retained for comparison.
- `prismv4/experiments/cht_policy_repair.py`: structured-output repair helpers.
- `prismv4/prism_cht/noiselab_tools.py`: NoiseLab tools and derived observation extraction.
- `prismv4/prism_cht/noiselab_registry.py`: tool registry.
- `prismv4/prism_cht/diagnostic_policy.py`: diagnostic policy and evidence packaging.
- `prismv4/prism_cht/evidence_graph.py`: evidence graph data structures.
- `prismv4/prism_cht/openai_compatible_client.py`: OpenAI-compatible LLM client.
- `prismv4/prism_cht/provider_config.py`: provider/model environment configuration.
- `prismv4/prism_cht/llm_audit.py`: full LLM input/output audit logging.

## Environment

Do not commit provider keys. The runner reads these variables:

```bash
export PRISM_CHT_MODEL_BASE_URL=...
export PRISM_CHT_MODEL_NAME=...
export PRISM_CHT_MODEL_API_KEY=...
```

Useful optional knobs:

```bash
export PRISM_CHT_MODEL_TIMEOUT_SECONDS=120
export PRISM_CHT_MODEL_MAX_TOKENS=4096
export PRISM_CHT_MODEL_RETRIES=2
export PRISM_CHT_JSON_REPAIR_ATTEMPTS=3
export PRISM_CHT_EVENT_CAUSALIZER_MAX_EVENTS=64
```

## Smoke Command

Run from the repo root, not from `trace_summary/`, so `python -m prismv4...` resolves correctly:

```bash
cd /home/dell2/RCA513/yyx
/home/dell2/RCA513/yyx/trace_summary/.venv_d32/bin/python -m prismv4.experiments.run_rcaeval_continuous \
  --data-root /home/dell2/RCA513/ysj/dataset/RCAEval/RE3 \
  --system RE3-OB \
  --max-cases 15 \
  --max-hypotheses 5 \
  --max-steps 4 \
  --output prismv4/results/prism_cht/deepseek_v4_pro_continuous_event_causalizer_15.json
```

For a larger non-RE3-OB subset, use another `--system` such as `RE3-TT` or `RE3-SS` and run without the 15-case cap if the dataset/runtime budget allows it.

The runner writes an adjacent LLM audit log, usually `*.llm_io.jsonl`. Preserve this behavior. The project owner needs all LLM inputs, outputs, repair attempts, EventCausalizer material, and final reasoning to remain inspectable.

## Recent Experimental Notes

- Earlier continuous 15-case RE3-OB run: 10/15 top-1.
- EventCausalizer improved the 15-case sample, but exposed instability around downstream symptom over-weighting and occasional structured-output failures.
- Prompt/audit checks on sampled runs did not show obvious label leakage: exact case keys and direct expected-label fields were not present in the LLM prompts. Re-check this whenever adapter fields change.
- The source branch may not include every later local experiment unless those changes were committed there. The next agent should verify behavior with fresh 15-case and full-subset runs after this import.

## Known Failure Modes To Watch

- The model can over-weight downstream symptoms such as `frontend` or infrastructure helpers such as `redis` when the causalized event list does not clearly mark service-local evidence and propagation direction.
- EventCausalizer can overload the main prompt if it emits too many verbose low-value events. Prefer ranking, compression, and evidence-density control over simply lowering the max event count.
- Structured output sometimes fails. Keep the repair loop: send the malformed non-JSON fragment plus parser error back to the LLM and retry until the configured repair budget is exhausted.
- Directionality is subtle. A downstream request symptom is not automatically the root cause. The features and prompt should distinguish local anomaly, propagated symptom, and dependency-induced delay.

## Next Agent Checklist

1. Re-run the 15-case RE3-OB smoke after this import and compare against prior audit logs.
2. Run a full larger non-RE3-OB subset, preferably `RE3-TT` or `RE3-SS`, and inspect both result JSON and `*.llm_io.jsonl`.
3. For every wrong case, inspect the EventCausalizer output first, then the final LLM prompt and answer.
4. Add regression tests for JSON repair, leakage scanning, event ranking/compression, and directionality cues.
5. Keep all Prism-CHT changes inside `prismv4/`; do not couple it to the trace summary algorithm unless explicitly requested.
