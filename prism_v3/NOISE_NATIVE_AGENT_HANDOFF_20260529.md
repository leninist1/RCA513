# Noise-native PRISM agent redesign handoff

Date: 2026-05-29
Workspace: `/home/dell2/RCA513/yyx`

## Why this handoff exists

The current session context has been compressed many times. This note records the design direction that should guide the next session.

The current `prism_v3` can run, supports multi-fault output, records official OpenRCA-compatible scores, and has a patched final arbitration stage. However, the deeper problem is architectural: PRISM is still too much like a scoring pipeline with an agent loop attached. It is not yet a naturally active `observe -> act -> reason -> observe` RCA agent.

## Current diagnosis

In Bank 30q, NoiseLab / noise-field-scorer often keeps useful root-cause candidates in the list, but PRISM can pull the answer away from them.

Observed symptoms:

- Correct components were often present in PRISM candidate lists, but final selection was weak.
- The old final counterfactual discriminator favored broad downstream explainers, especially Tomcat-like nodes.
- NoiseLab was treated mostly as an external prior or score vector, not as PRISM's native perception layer.
- The PRISM loop selected actions, but many actions only reweighted precomputed evidence rather than producing new observations.
- Final arbitration behaved like a late-stage reranker, which breaks the feeling of an intelligent investigation loop.

Conclusion: PRISM is currently `agent-inspired`, not `agent-native`.

## Design principle

PRISM and NoiseLab should not be two systems fighting each other. NoiseLab should be PRISM's perception layer.

The target design is:

```text
NoiseLab Perceiver
  -> CandidateFrame / EvidenceFrame
  -> PRISM Agent State over FaultEvents
  -> Policy chooses active observations
  -> Tools return new evidence
  -> Reasoner updates event posterior
  -> Stop or continue
  -> Official-format answer synthesis
```

Short version:

NoiseLab is responsible for seeing candidates and evidence. PRISM is responsible for validating, organizing, splitting/merging, and explaining those candidates through an active investigation loop.

## What the new framework should be

The next major design should be `Noise-native PRISM Agent`.

It should explicitly implement the agent cycle:

```text
Observe:
  What did NoiseLab / telemetry / logs / traces reveal?

Hypothesize:
  What root-event hypotheses exist?
  Which are root candidates, symptoms, broad explainers, or duplicates?

Act:
  What is the next best observation to reduce uncertainty?
  Examples:
    inspect_metric(component, time_window)
    inspect_log(component, reason_family)
    verify_trace(src, dst)
    run_counterfactual(candidate)
    compare_pair(candidate_a, candidate_b)
    split_event(event)
    merge_events(event_a, event_b)

Observe:
  The action returns new structured evidence.

Reason:
  Update event posterior and evidence ledger.
  Decide whether candidates are root, symptom, broad explainer, or unresolved.

Stop:
  Emit official OpenRCA prediction objects.
```

## Event-level state

Do not treat `component`, `reason`, and `time` as independent fields assembled only at the end.

The agent state should be over fault events:

```text
FaultEvent {
  event_id
  component
  reason
  time
  posterior
  status: active | reserve | rejected | merged | split
  evidence_ledger
  conflict_notes
}
```

This matters because OpenRCA official evaluation scores multi-fault answers as paired prediction objects. A component set and reason set can both be right while the event pairings are wrong.

## EvidenceFrame / CandidateFrame

NoiseLab should output a structured frame, not just `score.csv -> prior`.

Suggested fields:

```text
candidate_id
component_id
object_id
noise_score
calibrated_logit
metric_evidence
log_evidence
trace_evidence
time_candidates
reason_candidates
structural_features
symptomness
source_likelihood
```

PRISM should consume this as its initial observation.

## Posterior factorization

Final scoring should become a factorized posterior, not a hard late-stage rerank:

```text
log P(root_event | evidence)
  = w_noise * NoiseLab_logit
  + w_metric * metric_likelihood
  + w_log * log_likelihood
  + w_trace * trace_direction_likelihood
  + w_cf * counterfactual_likelihood
  - w_symptom * symptomness
  - w_broad * broad_explainer
```

Important constraint:

Counterfactual evidence is only a factor. It must not be a final judge that overrides the whole candidate list.

## Counterfactual role

The counterfactual engine should become a verification tool. It should answer:

- Does this candidate explain its local evidence?
- Does it reduce the target event's anomaly?
- Is the improvement self/local, or just broad downstream cleanup?
- In a pairwise dispute, which candidate has stronger root evidence?

It should return structured evidence:

```text
cf_likelihood
local_recovery
downstream_recovery
broad_explainer_penalty
pairwise_verdict
runtime_cost
```

It should not directly replace the posterior with a reranked distribution.

## Why the current v3 patch is not enough

The current v3 final arbitration patch improved symptoms:

- final scope now includes PRISM topK, NoiseLab topK, active/proposed candidates
- CF debug now records `root_evidence` and `broad_explainer`
- conflict cases include pairwise contender debug
- official OpenRCA scores are now recorded
- multi-fault output is supported

But this remains a patch on a pipeline.

The next design should remove the conceptual split between NoiseLab and PRISM. NoiseLab should be the perception subsystem of PRISM.

## Recommended implementation roadmap

### v3.2: NoiseLab evidence adapter

Create a native adapter that converts NoiseLab/runtime-scorer output into `CandidateFrame` / `EvidenceFrame`.

Deliverables:

- `noise_native/evidence_frame.py`
- `noise_native/noiselab_adapter.py`
- store per-candidate metric/log/trace/time/reason evidence
- do not only store scalar score vectors

### v3.3: Agent state over FaultEvents

Introduce event-level state:

- active event hypotheses
- reserve hypotheses
- rejected / symptom / broad-explainer status
- evidence ledger
- conflict ledger

The agent should reason over events, not bare components.

### v3.4: Tool-based active observation

Convert current actions into real observation tools:

- `inspect_metric`
- `inspect_log`
- `verify_trace`
- `run_counterfactual`
- `compare_pair`
- `split_event`
- `merge_events`

Each action must return a structured observation and append it to the evidence ledger.

### v3.5: Factorized posterior

Replace final hard rerank with posterior-factor updates.

Every update should produce an attribution record:

```text
candidate/event
old_posterior
new_posterior
factor_name
factor_delta
evidence_ids
```

### v3.6: Official-format event answer synthesis

Generate answers as a list of root-cause event objects:

```json
{
  "root cause occurrence datetime": "...",
  "root cause component": "...",
  "root cause reason": "..."
}
```

Then map that list to the current internal `prediction` dict only for compatibility.

## Files changed recently

Useful files in current `prism_v3`:

- `prism_v3/prism.py`
  - multi-fault output
  - final scope expansion
  - root_evidence / broad_explainer debug
  - conflict pairwise arbitration debug

- `prism_v3/evaluation/scorer.py`
  - now includes official OpenRCA-compatible scoring

- `prism_v3/evaluation/aggregator.py`
  - now reports `official_score_pct`

- `prism_v3/main.py`
  - prints `Strict Correct` and `Official Score`
  - adds final scope tuning args

## Suggested first prompt for next session

Use this prompt:

```text
阅读 /home/dell2/RCA513/yyx/prism_v3/NOISE_NATIVE_AGENT_HANDOFF_20260529.md。
基于这个方向，不要继续给 final arbitration 打补丁。
请先设计并实现 v3.2 的 NoiseLab EvidenceFrame adapter，让 NoiseLab 成为 PRISM 的原生感知层。
要求保留当前实验 runner 可跑，先做小步重构和冒烟测试。
```

## Current Bank 30q command after latest patch

If needed, rerun the current patched version with:

```bash
cd /home/dell2/RCA513/yyx
mkdir -p prism_v3/results

python3 -u -m prism_v3.main \
  --option PRISM \
  --systems Bank \
  --workers 1 \
  --max-queries 30 \
  --output prism_v3/results \
  --prism-noise-lab \
  --prism-noise-lab-scores /home/dell2/RCA513/yyx/rca513/results/noise_lab_ltr_scores_bankfull_ndcg_noprefix_nobase_Bank_20260525_212615.csv \
  --prism-noise-lab-strategy ltr_full \
  --v2-all \
  --v2-mcts-sims 5 \
  --prism-cf-profile-top-k 2 \
  --prism-cf-profiles-max-calls 1 \
  --prism-cf-degradation-entity-top-k 6 \
  --prism-final-cf-top-k 2 \
  --prism-final-scope-prism-top-k 5 \
  --prism-final-scope-noise-top-k 5 \
  --prism-final-scope-min-candidates 6 \
  --prism-final-scope-max-candidates 10 \
  --prism-final-cf-only-on-conflict \
  2>&1 | tee prism_v3/results/run_bank30_v3_finalarb_$(date +%Y%m%d_%H%M%S).log
```

But this command is for diagnosing the patched pipeline, not the desired final architecture.

