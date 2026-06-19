# CHT-5.2 — LLM Provider Safety Baseline

## Accepted Baseline

```text
CHT-5.2 accepted baseline: 40cf7d8
Trusted frozen-core baseline: b5cfc34
```

## Accepted Commit Chain

```text
076226a  Provider integration recovered on trusted core
a02d86b  Structured LLM policy boundary formally verified
b86e6a2  LLM → Provider redaction boundary audited
0742881  Sensitive DTO repr surfaces hardened
0f5e29a  Historical CHT-5.1R recovery note backfilled
4d8d1cd  Canonical JSON semantic contract formalized
40cf7d8  Policy → Provider safety boundary verified end to end
```

## Runtime Boundary

```text
Structured LLM policy
  → StructuredModelRequester
  → ModelClient protocol
  → OpenAICompatibleChatModelClient
  → HttpTransport
  → HttpResponse
  → ModelResponse
  → ProviderCallAudit
  → structured parser
```

## Accepted Security Properties

* credentials may exist only where required for outbound HTTP transport;
* raw Prompt content may flow through semantic in-memory objects and
  outbound request bodies;
* raw Provider response bodies may flow through HTTP parsing;
* parsed model content may flow into the structured parser;
* `ProviderCallAudit` excludes credentials, prompts, raw response
  bodies, and parsed model content;
* exception text does not include those sensitive values;
* sensitive DTO fields are excluded from default `repr()` via
  `field(repr=False)`;
* inference-time Prompt construction does not inject GT-only values;
* Provider tests and integrated boundary tests do not access the live
  network;
* `canonicalize_json_value()` is a semantic canonicalizer, not a
  redaction boundary.

## Frozen-Core Status

All frozen core files remain byte-identical to `b5cfc34`:

```text
prismv4/prism_cht/action_gate.py
prismv4/prism_cht/action_schema.py
prismv4/prism_cht/evidence_graph.py
prismv4/prism_cht/hypothesis.py
prismv4/prism_cht/canonical.py
prismv4/prism_cht/executor.py
prismv4/prism_cht/assessment_gate.py
prismv4/prism_cht/lead_controller.py
prismv4/prism_cht/challenge_gate.py
prismv4/prism_cht/challenger_controller.py
prismv4/prism_cht/final_verifier.py
```

Verified with:

```bash
git diff --exit-code b5cfc34 -- <files>
```

## Test Evidence

Full regression suite at acceptance commit `40cf7d8`:

```text
986 passed
```

Integrated boundary test:

```text
13 passed — test_cht_policy_provider_boundary_e2e.py
```

## Deferred Work

The next phase may introduce an explicitly configured Provider smoke
path, but must not yet integrate real Provider calls into the full RCA
execution chain.

Any live Provider test must:

* be opt-in;
* be skipped by default;
* never run in ordinary regression tests;
* never persist secrets;
* never print Prompt or response bodies;
* use a minimal request;
* produce only log-safe metadata.
