# CHT-5.1 Provider Integration — Recovery Note

**Branch:** `recovery/cht51-provider`
**Recovery Date:** 2026-06-13
**Status:** Frozen — do not push

---

## Reference Commits

| Role | Commit | Description |
|------|--------|-------------|
| Trusted baseline | `b5cfc34` | `fix(prism-cht): require explicit triplet evidence grounding` |
| Broken commit | `69925c6` | `feat(prism-cht): add openai-compatible provider adapter` |
| Clean dependency source | `cf81f5d` | `feat(prism-cht): add structured llm policy boundary` |
| Recovery head | `076226a` | `feat(prism-cht): add provider adapter on trusted core` |

The broken commit `69925c6` is retained as `backup/cht51-broken-69925c6` for archival reference.

---

## File Classification

### (a) Frozen Core Files — Byte-Identical to `b5cfc34`

These core runtime files were restored from the trusted baseline and verified byte-identical with `git diff`:

| File | Status |
|------|--------|
| `prismv4/prism_cht/action_gate.py` | Identical |
| `prismv4/prism_cht/action_schema.py` | Identical |
| `prismv4/prism_cht/challenge_demo.py` | Identical |
| `prismv4/prism_cht/demo_scenario.py` | Identical |
| `prismv4/prism_cht/evidence_graph.py` | Identical |
| `prismv4/prism_cht/final_types.py` | Identical |
| `prismv4/prism_cht/final_verifier.py` | Identical |
| `prismv4/prism_cht/hypothesis.py` | Identical |
| `prismv4/prism_cht/lead_controller.py` | Identical |
| `prismv4/prism_cht/tournament_types.py` | Identical |

The `__init__.py` was **reconstructed** (214 lines) — it combines the original baseline imports with additional imports for provider and LLM policy modules. It is not byte-identical to `b5cfc34` (179 lines).

### (b) Clean Dependency Files — Byte-Identical to `cf81f5d`

These structured LLM policy boundary source files were imported from `cf81f5d` as clean dependencies. No runtime modifications were made.

| File | Status |
|------|--------|
| `prismv4/prism_cht/fake_model_client.py` | Identical |
| `prismv4/prism_cht/llm_json.py` | Identical |
| `prismv4/prism_cht/llm_policy.py` | Identical |
| `prismv4/prism_cht/llm_prompts.py` | Identical |
| `prismv4/prism_cht/llm_types.py` | Identical |

### (c) Recovered Provider Files

These files implement the OpenAI-compatible provider adapter. They were migrated from the broken commit `69925c6` and re-applied on top of the trusted core.

| File | Description |
|------|-------------|
| `prismv4/prism_cht/http_transport.py` | HTTP transport layer |
| `prismv4/prism_cht/openai_compatible_client.py` | OpenAI-compatible client adapter |
| `prismv4/prism_cht/provider_config.py` | Provider configuration |
| `prismv4/prism_cht/provider_types.py` | Provider type definitions |

### (d) Provider Test Files

| File | Description |
|------|-------------|
| `prismv4/tests/test_cht_http_transport.py` | HTTP transport tests |
| `prismv4/tests/test_cht_openai_compatible_client.py` | Client adapter tests |
| `prismv4/tests/test_cht_provider_audit.py` | Provider audit tests |
| `prismv4/tests/test_cht_provider_config.py` | Provider config tests |
| `prismv4/tests/test_cht_provider_integration.py` | Provider integration tests |
| `prismv4/tests/test_cht_provider_no_leakage.py` | No-leakage guard tests |
| `prismv4/tests/test_cht_provider_no_live_network.py` | No-live-network guard tests |

---

## Verification

- `git diff --check` — clean (no whitespace issues)
- `python3 -m pytest prismv4/tests -q` — **690 passed**, 0 failed (1.67s)
- Runtime diff from baseline `b5cfc34` contains only the additions listed above

---

## Known Deferred Issues

1. **LLM Boundary test files from `cf81f5d` not imported.** The following test files from the structured LLM policy boundary commit were excluded from this recovery and should be imported or rewritten before merging:
   - `test_cht_llm_boundary_no_leakage.py`
   - `test_cht_llm_boundary_no_network.py`
   - `test_cht_llm_json.py`
   - `test_cht_llm_policy.py`
   - `test_cht_llm_policy_integration.py`
   - `test_cht_llm_prompts.py`

2. **`__init__.py` is a merge reconstruction**, not a clean rebase. It was manually rebuilt to import both legacy core modules and the new provider/LLM modules. A future audit should verify no missing or orphaned exports.

3. **No live-network integration test** has been performed. These tests use mock/fake transports only.

4. **Branch has not been merged** into `prismv4` or `master`. Merge strategy should be a fast-forward or squash-merge of the recovery branch only.
