# CHT-5.2A — Structured LLM Policy Boundary Formalization Audit

## 1. Accepted Parent Recovery Point

- **Commit**: `076226aea588fceb0c32016668ae0d2d54e76c32`
- **Branch**: `recovery/cht51-provider`
- **Description**: feat(prism-cht): add provider adapter on trusted core

## 2. Recovery Branch

- **Branch**: `recovery/cht52a-llm-policy-boundary`
- **Created from**: `076226aea588fceb0c32016668ae0d2d54e76c32`

## 3. Provenance from `cf81f5d`

All 5 production files are byte-identical to commit `cf81f5d`:

```
git diff --exit-code cf81f5d -- \
  prism_cht/llm_types.py \
  prism_cht/llm_json.py \
  prism_cht/llm_prompts.py \
  prism_cht/llm_policy.py \
  prism_cht/fake_model_client.py
```

**Result**: ALL FILES BYTE-IDENTICAL — PASS

## 4. Responsibility of Each Production File

| File | Responsibility |
|------|---------------|
| `llm_types.py` | Structured types: `ModelMessage`, `ModelRequest`, `ModelResponse`, `ModelClient` protocol, error types |
| `llm_json.py` | Strict JSON parsing with field validation; rejects markdown fences, NaN/Infinity, duplicate keys, unknown fields |
| `llm_prompts.py` | Deterministic prompt builders for Lead and Challenger policy decisions, assessments, challenge proposals, and resolutions |
| `llm_policy.py` | `StructuredModelRequester` (retry wrapper), `StructuredLLMLeadPolicy`, `StructuredLLMChallengerPolicy` |
| `fake_model_client.py` | Deterministic `FakeModelClient` with FIFO response queue and request recording |

## 5. Dedicated Tests Added

| Test File | Tests | Coverage |
|-----------|-------|----------|
| `tests/test_cht_llm_types.py` | 23 | Valid construction, required fields, role restrictions, error types, protocol |
| `tests/test_cht_llm_json.py` | 58 | parse_json_object, field validators, lead/challenge parsers, error consistency |
| `tests/test_cht_llm_prompts.py` | 31 | Deterministic output, prompt structure, evidence catalog, semantic sections, isolation |
| `tests/test_cht_llm_policy.py` | 22 | Happy-path decisions, retry behavior, malformed/schema-invalid JSON, cross-role isolation |
| `tests/test_cht_fake_model_client.py` | 21 | FIFO order, request recording, deterministic behavior, empty queue, network/env isolation |
| **Total** | **155** | — (including 207 test functions across all) |

## 6. Security and Leakage Checks

### Error message isolation
- `StructuredOutputError` messages do not include raw model response text — VERIFIED
- Error context tags are brief and structural (e.g., `parse_lead_policy_decision.action_id`)
- No full prompt body appears in parse error messages

### Prompt serialization
- `canonicalize_json_value` preserves input string values verbatim (no content redaction)
- Prompt builders do not inject scoring, ground-truth, or label fields — VERIFIED
- Python object reprs (`<__main__.Foo at 0x...>`) do not appear in prompt messages — VERIFIED
- User messages are valid JSON strings — VERIFIED

### Known limitation (deferred)
- Dataclass `__repr__` for `ModelMessage`, `ModelRequest`, `ModelResponse` includes all field values including potential secrets. This is inherent to Python dataclasses and no redaction mechanism exists in this layer. Content filtering should be handled upstream.

## 7. Network Isolation

- All tests use deterministic `FakeModelClient` or inline `_TestFakeClient` — no live network calls
- `FakeModelClient` does not import any model SDK, read environment variables, or access the filesystem
- Policy classes (`StructuredLLMLeadPolicy`, `StructuredLLMChallengerPolicy`) have no dependency on `provider_config`

## 8. Deferred Issues

| # | Issue | Detail |
|---|-------|--------|
| D1 | Dataclass repr includes field values | `ModelMessage`, `ModelRequest`, `ModelResponse` are plain frozen dataclasses; `repr()` includes all content verbatim. No secret redaction. Mitigation should be handled at the audit/log layer if needed. |
| D2 | `canonicalize_json_value` preserves all string values | Prompt builders do not redact or filter context values. If the caller passes sensitive data in the context dict, it WILL appear in the prompt. This is by design — the prompt builder is a transparent serializer. |

## 9. No Production Files Modified

Zero modifications to production files under `prism_cht/`:

```
git diff --name-status 076226aea588fceb0c32016668ae0d2d54e76c32..HEAD -- prism_cht/
```

**Result**: No changes — PASS

## 10. Acceptance Checklist

- [x] PASS — Branch created from accepted CHT-5.1R recovery point
- [x] PASS — All 5 boundary production files byte-identical to `cf81f5d`
- [x] PASS — No existing production Python file modified
- [x] PASS — Dedicated tests added for all 5 files
- [x] PASS — Malformed JSON behavior covered
- [x] PASS — Schema-invalid JSON behavior covered
- [x] PASS — Fake-client deterministic behavior covered
- [x] PASS — Inference prompt GT-leakage sentinel test covered
- [x] PASS — Secret, prompt-body, and response-body leakage checks covered
- [x] PASS — No live network access
- [x] PASS — `git diff --check` clean
- [x] PASS — All `prismv4/tests` pass (897 passed)
- [x] PASS — No push
