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
  --system RE3-TT \
  --max-cases 0 \
  --max-hypotheses 10 \
  --recall-pool-size 15 \
  --max-steps 4 \
  --output prismv4/results/prism_cht/deepseek_v4_pro_TT_recall_v1_full.json
```

`--max-cases 0` runs the full system subset. `--max-hypotheses` (default 10) is the final reasoning width; `--recall-pool-size` (default 15) is the tiered recall pool width that feeds EventCausalizer and hypothesis seeding. Keep `recall-pool-size >= max-hypotheses`. The legacy RE3-OB command used `--max-hypotheses 5` without `--recall-pool-size`; that still works but the new defaults are recommended for TT-like systems.

For a quick deterministic recall check without any LLM/provider cost:

```bash
/home/dell2/RCA513/yyx/trace_summary/.venv_d32/bin/python -m prismv4.scripts.offline_recall_eval \
  --data-root /home/dell2/RCA513/ysj/dataset/RCAEval/RE3 \
  --system RE3-TT --system RE3-OB --pool-size 20 --show-misses
```

For a larger non-RE3-OB subset, use another `--system` such as `RE3-TT` or `RE3-SS` and run without the 15-case cap if the dataset/runtime budget allows it.

The runner writes an adjacent LLM audit log, usually `*.llm_io.jsonl`. Preserve this behavior. The project owner needs all LLM inputs, outputs, repair attempts, EventCausalizer material, and final reasoning to remain inspectable.

## Recent Experimental Notes

- Earlier continuous 15-case RE3-OB run: 10/15 top-1.
- RE3-OB full 30-case run using the EventCausalizer command with `--max-cases 0`: 25/30 top-1, 83.3%. Result file: `prismv4/results/prism_cht/deepseek_v4_pro_continuous_event_causalizer_15.json`. This run predates the later EventCausalizer mechanism-strength tuning discussed during debugging.
- RE3-TT full 30-case run using the newer command: 3/30 top-1, 10.0%. Result file: `prismv4/results/prism_cht/deepseek_v4_pro_event_causalizer_RE3_TT_full.json`.
- Prompt/audit checks on sampled runs did not show obvious label leakage. The audit logs did not contain `case_id`, `fault_name`, `expected_component`, `RE3-TT/...`, or fault-directory names such as `ts-auth-service_f...`. Re-check this whenever adapter fields change.
- All LLM input/output was available in the adjacent `*.llm_io.jsonl` audit files. Preserve this behavior; it was essential for diagnosing the failure.

## Implemented Fixes (this session)

The Next Agent Checklist items 1-6 and 8 have been implemented. See the git diff for full detail; the key changes are:

1. **Recall pool separated from reasoning width** (`rcaeval_adapter.py`, `run_rcaeval_continuous.py`). New `RCAEvalTelemetryStore.build_tiered_recall_pool(pool_size=15)` unions candidates from five tiers: magnitude-with-workload-down-weighted, earliest onset, near-onset internal-mechanism priority, log/error emitters, and topology-central services. The runner feeds this pool to `build_hypotheses_from_case(candidates=...)` and to `_build_event_causal_facts`, so EventCausalizer now sees the full recall set, not just the final top-k. New CLI knob `--recall-pool-size` (default 15), decoupled from `--max-hypotheses` (default now 10).

2. **Workload/request-total down-weighted** in candidate scoring (tier 1 subtracts workload magnitude before ranking).

3. **TT entry-component inference fixed** (`_infer_entry_components`). Uses trace-topology roots (components with callees but no callers) instead of only `frontend`/`gateway` string matching. TT entries now resolve to real trace roots like `ts-preserve-service` instead of the alphabetical-first `ts-admin-basic-info-service`.

4. **Final-decision consistency / audited global_rescue** (`_normalize_final_decision`). Nominations outside the active hypothesis set are retained but flagged with `global_rescue: true` and an audit note in the rationale, instead of being silently accepted.

5. **Entry-bias prompt hardening**. EventCausalizer output requirements and the main agent reasoning rules now explicitly state that `is_entry_like_component` marks the request surface, NOT root-cause strength, and add a comparison rule for entry vs background service when both show strong mechanism signals.

6. **Regression tests** (`prismv4/tests/test_cht_recall_and_entry.py`, 16 tests): tiered recall includes low-magnitude sources, workload down-weighting, pool dedup/size cap, observation caching, explicit-candidate hypothesis ordering, placeholder for candidates without observations, backward compatibility, entry inference (named/topology/fallback), global_rescue flagging, in-set vs out-of-set, invalid-component fallback, and leakage guards.

7. **Performance**: memoized `_component_metric_observations`, `_component_log_observations`, `_all_log_observations`, and `_trace_dependency_context` in `RCAEvalTelemetryStore`. Single-case load dropped from ~7s to ~0.9s; repeated trace-context calls are now cached.

## Validation Results

Offline recall (deterministic, no LLM, `prismv4/scripts/offline_recall_eval.py`):

| System | magnitude top-5 | magnitude_no_workload top-5 | tiered_union@10 | tiered_union@15 |
| --- | ---: | ---: | ---: | ---: |
| RE3-TT | 1/30 | 23/30 | 27/30 | 30/30 |
| RE3-OB | 30/30 | 30/30 | 30/30 | 30/30 |

Root-in-hypotheses with `recall_pool=15` (deterministic):

| System | width=5 | width=10 | width=15 |
| --- | ---: | ---: | ---: |
| RE3-TT | 11/30 | 28/30 | 30/30 |
| RE3-OB | 29/30 | 30/30 | 30/30 |

Live LLM smoke (3 TT cases before entry-bias fix): 0/3 top-1 — answer was in the pool for all 3, but LLM picked the entry component `ts-preserve-service` over the true root `ts-auth-service`.

Live LLM smoke (1 TT case after entry-bias prompt fix): **1/1 top-1** — `ts-auth-service_f1/1` correctly predicted. Result: `prismv4/results/prism_cht/smoke_TT_entry_bias_fix_1case.json`.

The remaining frontier is **reasoning disambiguation**, not recall. A full 30-case TT run with the new defaults (`--max-hypotheses 10 --recall-pool-size 15`) plus the entry-bias prompt fix is the recommended next validation. Expect each case to take ~10-15min under the current provider, so budget ~6-8h for a full TT run.

## Latest Failure Diagnosis

The RE3-TT failure is primarily a candidate-recall failure, not a JSON-format failure and not simply an LLM reasoning failure.

Current flow:

1. `run_rcaeval_continuous.py` calls `load_re3_case(..., top_k=args.max_hypotheses)`.
2. `RCAEvalTelemetryStore.build_observations()` ranks observations by total local magnitude: `(-magnitude, first_seen, component)`.
3. `build_hypotheses_from_case()` again selects only the top `max_hypotheses` observations.
4. `_build_event_causal_facts()` sends only those hypothesis components to EventCausalizer.

With `--max-hypotheses 5`, this works on RE3-OB but fails badly on RE3-TT:

| System | Accuracy | True root in top-5 observations | True root in top-10 observations |
| --- | ---: | ---: | ---: |
| RE3-OB | 25/30 | 30/30 | 30/30 |
| RE3-TT | 3/30 | 1/30 | 10/30 |

For RE3-OB, the expected component was present in `event_causal_profile`, final belief, and transcript for every case. For RE3-TT, the expected component appeared in only 8/30 event profiles and only 4/30 final belief states. Most TT cases therefore asked the LLM to choose from a candidate set that did not contain the answer.

Examples from the offline candidate-rank check:

- `RE3-TT/ts-auth-service_f1/1`: `ts-auth-service` rank 7; top-5 were contacts, consign-price, consign, security, user.
- `RE3-TT/ts-auth-service_f3/3`: `ts-auth-service` rank 15; top-5 were admin-basic-info, travel2, admin-travel, assurance, travel.
- `RE3-TT/ts-route-service_f1/1`: `ts-route-service` rank 15; top-5 were travel, admin-travel, contacts, travel2, assurance.
- `RE3-TT/ts-route-service_f4/3`: `ts-route-service` rank 15; top-5 were travel, admin-travel, travel2, assurance, contacts.

The ranking is dominated by large downstream visibility signals, especially workload, latency, and propagated error/log bursts. In TT, real roots such as `ts-auth-service` and `ts-route-service` often have smaller local magnitude than noisy downstream services, so total-magnitude top-k drops them before the LLM and EventCausalizer can reason about them.

Secondary issues found in the same analysis:

- TT entry-component detection is weak. The adapter only recognizes names containing `frontend` or `gateway`; TT falls back to the first sorted component, often `ts-admin-basic-info-service`, which can distort propagation reasoning.
- `final_decision` currently accepts any component in `case.components`, not only active hypotheses. Two of the TT hits came from the model jumping outside the hypothesis set. That can occasionally rescue a case, but it also creates unstable and sometimes inconsistent outputs.
- EventCausalizer can mark many downstream TT symptoms as `mechanism_strength=strong` because they have simultaneous CPU/memory/log changes. This amplifies the top-k recall problem: once the true root is missing, EventCausalizer builds a plausible but wrong causal story around symptoms.
- Provider instability is still present under long prompts. RE3-OB audit had many transient provider errors; RE3-TT had fewer but still saw timeouts/incomplete reads. The final result files had no case-level parser errors, but long prompts and verbose EventCausalizer responses increase runtime and drift risk.

## Known Failure Modes To Watch

- The model can over-weight downstream symptoms such as `frontend` or infrastructure helpers such as `redis` when the causalized event list does not clearly mark service-local evidence and propagation direction.
- EventCausalizer can overload the main prompt if it emits too many verbose low-value events. Prefer ranking, compression, and evidence-density control over simply lowering the max event count.
- Structured output sometimes fails. Keep the repair loop: send the malformed non-JSON fragment plus parser error back to the LLM and retry until the configured repair budget is exhausted.
- Directionality is subtle. A downstream request symptom is not automatically the root cause. The features and prompt should distinguish local anomaly, propagated symptom, and dependency-induced delay.
- TT exposes a severe candidate-recall cliff. Do not rely on `--max-hypotheses 5` plus total magnitude ranking for TT-like systems.
- Workload/request-total can dwarf other signals and should usually be treated as symptom visibility or load propagation, not direct root-cause strength.

## Next Agent Checklist

Status legend: [DONE] implemented this session, [REMAINING] open work.

1. [DONE] Separate candidate recall from final reasoning width. Load a larger recall pool, for example 15-20 components, then let NoiseLab/EventCausalizer compress it into a smaller final reasoning set. Do not truncate the case observations to `max_hypotheses` before recall analysis.
2. [DONE] Replace total-magnitude top-k with tiered recall. Include candidates from magnitude, earliest onset, near-onset internal mechanism, log/error emitters, topology-central services, dependency/callee candidates, and low-magnitude but high-causal-centrality services.
3. [DONE] Down-weight workload/request-total in candidate scoring. Keep it as visibility/load evidence, not as the dominant source score.
4. [DONE] Feed EventCausalizer the recall pool or a deterministic compact projection of it, not only the final top-5 hypotheses. Its job should be feature organization and compression, not inheriting a broken top-k selection.
5. [DONE] Fix TT entry-component inference. Use trace topology and service roles instead of only `frontend`/`gateway` string matching.
6. [DONE] Add final-decision consistency checks. Either force final answers to come from active candidates, or implement an explicit audited `global_rescue` path that can nominate a component outside the hypothesis set and records why.
7. [REMAINING] Run quick ablations before more prompt work: TT with `--max-hypotheses 15`, TT with widened recall plus compact final candidates, and TT with workload down-weighted. These should show whether recall alone lifts the upper bound. The offline recall script (`prismv4/scripts/offline_recall_eval.py`) answers the deterministic upper bound; a full live TT run with the new defaults is the remaining validation (budget ~6-8h at current provider latency).
8. [DONE] Add regression tests for candidate recall, TT entry inference, JSON repair, leakage scanning, event ranking/compression, and directionality cues. Note: JSON-repair tests already existed (`test_cht_llm_json.py`); new tests cover recall, entry, rescue, leakage (`test_cht_recall_and_entry.py`).
9. [DONE] Keep all Prism-CHT changes inside `prismv4/`; do not couple it to the trace summary algorithm unless explicitly requested.
10. [REMAINING] Reasoning disambiguation is the new frontier. The entry-bias prompt fix turned 0/1 -> 1/1 on the sampled TT case, but a full TT run is needed to measure the real lift. If TT accuracy stays low despite the answer being in the pool, the next lever is stronger directionality cues in the EventCausalizer deterministic features (e.g. explicitly rank source-like vs symptom-like per event) and/or a final-verification pass that forces the LLM to compare the top-2 candidates' onset + dependency direction before committing.
