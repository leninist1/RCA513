# `canonicalize_json_value` — Semantic Contract

## Scope

This contract applies to:

```python
canonicalize_json_value(value: Any) -> Any
```

defined in:

```text
prismv4/prism_cht/canonical.py
```

## Responsibility

`canonicalize_json_value` is a **general semantic canonicalizer**.  It
is not an LLM-only helper.  Its uses include:

* deterministic canonical representation of JSON-compatible values;
* prompt construction (via `llm_prompts.py`);
* dispatch argument normalization (via `to_dispatch_args`);
* tool-call signature hashing (via `build_tool_call_signature`).

Supported string values are intentionally preserved **verbatim**.  No
redaction, masking, truncation, or sanitization is performed.

## Explicit Non-Responsibilities

This function is **not**:

* a redaction helper;
* a log-safe serializer;
* an audit-safe serializer;
* an exception-safe formatter.

Callers must **not** assume that its output is safe for logs,
exception messages, audit persistence, or any outward-facing debug
surface.

## Redaction Boundary

Sensitive-content suppression belongs at outward-facing boundaries:

| Boundary | Mechanism |
|---|---|
| repr / debug surfaces | `field(repr=False)` in DTO dataclasses |
| logging | log-level filtering, structured sanitizers |
| exception text | exception-safe formatters |
| audit persistence | audit record schemas with redaction hooks |

## Frozen-Core Decision

`prismv4/prism_cht/canonical.py` is part of the frozen core restored
from trusted baseline `b5cfc34`.  No runtime source modification has
been accepted.  The semantic contract is documented externally and
verified behaviorally through the accompanying test file:

```text
prismv4/tests/test_cht_canonical_json_contract.py
```

Any future executable change to `canonicalize_json_value` requires an
explicit thaw phase with audit trail.

## Caller Inventory

Currently identified callers:

| File | Role |
|---|---|
| `prismv4/prism_cht/llm_prompts.py` | Prompt construction |
| `prismv4/prism_cht/canonical.py` | Self (`to_dispatch_args`, `build_tool_call_signature`) |
| `prismv4/prism_cht/__init__.py` | Public re-export |

Test files also exercise the function directly (see
`test_cht_invariant_hardening.py` and
`test_cht_canonical_json_contract.py`).

## Verification

Behavioral contract coverage is provided by:

```bash
python3 -m pytest prismv4/tests/test_cht_canonical_json_contract.py -v
```

The canonical source file is verified byte-identical to trusted
baseline with:

```bash
git diff --exit-code b5cfc34 -- prismv4/prism_cht/canonical.py
```
