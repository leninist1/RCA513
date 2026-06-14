# CHT-5.3B Smoke Harness CLI and Output Safety Audit

## 1. Scope and Parent Commit

- **Parent commit:** `0e6807f`
- **Parent branch:** `recovery/cht53a-explicit-smoke-harness`
- **Audit branch:** `audit/cht53b-smoke-cli-output`

This audit inspects the Provider smoke harness introduced in CHT-5.3A:

```
scripts/cht_provider_smoke.py
tests/test_cht_provider_smoke_harness.py
```

and the Provider configuration loader:

```
prism_cht/provider_config.py
tests/test_cht_provider_config.py
```

No runtime or test file was modified during this audit.

---

## 2. Harness Entrypoint and CLI Flags

**File:** `scripts/cht_provider_smoke.py:382`

### CLI Entrypoint

From the git repository toplevel:

```text
python3 prismv4/scripts/cht_provider_smoke.py --allow-live-network
```

Or via module invocation:

```text
python3 -m prismv4.scripts.cht_provider_smoke --allow-live-network
```

### Public CLI Flags

| Flag                 | Type         | Default | Description                                                    |
| -------------------- | ------------ | ------- | -------------------------------------------------------------- |
| `--allow-live-network` | `store_true` | `False` | Authorize live-network access; also requires `PRISM_CHT_ENABLE_LIVE_SMOKE=1` |

No other CLI arguments exist. The argparse `description` is:

> "CHT Provider smoke harness — minimal, fail-closed."

### Default Behavior

Without `--allow-live-network`, the harness exits immediately with a refusal message. It does not load configuration, construct a transport, or read secrets.

### Double Opt-In Gate Implementation

The gate (`scripts/cht_provider_smoke.py:166`) requires **both**:

1. `--allow-live-network` CLI flag set to `True` (explicit boolean)
2. Environment variable `PRISM_CHT_ENABLE_LIVE_SMOKE` set exactly to `"1"` (single-character string comparison via `==`)

The helper `_live_smoke_enabled()` (`scripts/cht_provider_smoke.py:49-51`) returns `True` only when `environ.get("PRISM_CHT_ENABLE_LIVE_SMOKE") == "1"`. Any other value (`"0"`, `"true"`, `""`, absent) is treated as "not enabled".

When an injected `transport` argument is supplied to `run_smoke()`, the gate is bypassed entirely — `gate_passed = True` unconditionally (`scripts/cht_provider_smoke.py:165-168`). This is the test path.

### Config-Loading Point

Config is loaded immediately after the gate check (`scripts/cht_provider_smoke.py:174`):

```python
config = load_openai_compatible_config_from_mapping(environ)
```

The loader reads only from the explicit `environ: Mapping[str, str]` dict. It never reads `os.environ` or any file.

### Transport-Construction Point

The real `UrllibHttpTransport()` is constructed only when both gates pass AND no injected transport was provided (`scripts/cht_provider_smoke.py:177-179`). It is wrapped in a `try/except` that catches any `Exception` and returns a `SmokeResult(status="transport_error", ...)` without leaking secrets.

### Output Function / Print Path

The `main()` function (`scripts/cht_provider_smoke.py:335-379`) calls `print()` directly for each line. No logging framework is used. Output categories:

- **Refused:** `"SMOKE REFUSED: live-network execution is disabled by default."` + guidance
- **Success:** `"SMOKE SUCCESS"` + safe metadata fields
- **Error:** `"SMOKE {ERROR_KIND}"` + safe metadata fields + error class and message

### Returned Result Fields

`SmokeResult` is a frozen dataclass (`scripts/cht_provider_smoke.py:59-84`) with fields:

| Field                    | Type              | Remarks                                      |
| ------------------------ | ----------------- | -------------------------------------------- |
| `status`                 | `str`             | one of six controlled values                 |
| `configured_model_name`  | `str \| None`     | `None` when gate refused                     |
| `request_count`          | `int`             | `0` when gate refused                        |
| `audit_record_count`     | `int`             | `0` when gate refused                        |
| `provider_outcome`       | `str \| None`     | from `ProviderCallAudit.outcome`             |
| `http_status_code`       | `int \| None`     | safe metadata                                |
| `elapsed_ms`             | `float \| None`   | safe metadata                                |
| `request_bytes`          | `int \| None`     | safe metadata                                |
| `response_bytes`         | `int \| None`     | safe metadata                                |
| `finish_reason`          | `str \| None`     | model-given, typically "stop"                |
| `prompt_tokens`          | `int \| None`     | safe metadata                                |
| `completion_tokens`      | `int \| None`     | safe metadata                                |
| `total_tokens`           | `int \| None`     | safe metadata                                |
| `error_class`            | `str \| None`     | `type(exc).__name__` only                    |
| `error_message`          | `str \| None`     | `str(exc)` — **see output safety notes**     |

### Exception Handling Path

Four specific exception blocks in `run_smoke()`:

1. `ProviderTransportError` → `status="transport_error"` (line 210)
2. `ProviderHTTPError` → `status="http_error"` (line 230)
3. `ProviderResponseError` → `status="response_error"` (line 250)
4. Generic `Exception` → `status="response_error"` (line 270)

The `UrllibHttpTransport()` constructor failure is also caught generically at line 181, returning `status="transport_error"`.

### `os.environ` Access

- **`run_smoke()`:** Never reads `os.environ`. Accepts `environ: Mapping[str, str]`.
- **`main()`:** Reads `os.environ` once at line 342 via `dict(os.environ)`, **after** argument parsing, and passes the snapshot to `run_smoke()`.

### Module Import Side Effects

The harness file has **no** import-time side effects. At module level it only:
- Defines imports, constants, dataclasses, and functions
- Does NOT read `os.environ`
- Does NOT construct transport
- Does NOT instantiate a client
- Does NOT print output
- Does NOT access the network

The `if __name__ == "__main__"` guard on line 381 prevents `main()` from running on import.

---

## 3. Provider Config Keys and Classification

**File:** `prism_cht/provider_config.py:23-32`

### Required Environment Variables

| Export Name                     | Classification            | Required | Default | Notes                                                              |
| ------------------------------- | ------------------------- | -------- | ------- | ------------------------------------------------------------------ |
| `PRISM_CHT_MODEL_BASE_URL`      | endpoint/config           | Yes      | None    | Must start with `https://`; no user/pass, query params, or fragment |
| `PRISM_CHT_MODEL_NAME`          | model/config              | Yes      | None    | Must be non-empty                                                  |
| `PRISM_CHT_MODEL_API_KEY`       | **secret**                | Yes      | None    | Must be non-empty; excluded from `repr()`                          |

### Optional Environment Variables

| Export Name                         | Classification              | Required | Default    | Notes                          |
| ----------------------------------- | --------------------------- | -------- | ---------- | ------------------------------ |
| `PRISM_CHT_MODEL_TIMEOUT_SECONDS`   | optional generation param   | No       | `60.0`     | Must be > 0, parseable as float |
| `PRISM_CHT_MODEL_MAX_TOKENS`        | optional generation param   | No       | `4096`     | Must be > 0, parseable as int   |
| `PRISM_CHT_MODEL_MAX_RESPONSE_BYTES`| optional generation param   | No       | `2000000`  | Must be > 0, parseable as int   |

### Config Loader Behavior

`load_openai_compatible_config_from_mapping()` (`provider_config.py:201-262`):

- Never reads `os.environ`
- Never reads files
- Never imports `dotenv`
- Error messages for missing/invalid config never include the API key value
- The `OpenAICompatibleChatConfig` class has `api_key: str = field(repr=False)` — API key is excluded from `repr()`

---

## 4. Gate Behavior Matrix

| Scenario                                        | Status   | Config Loaded? | Secret Access? | Real Transport? | Request Sent? | Output Category       | Leakage Risk |
| ----------------------------------------------- | -------- | -------------- | -------------- | --------------- | ------------- | --------------------- | ------------ |
| No CLI flag, no env gate                        | refused  | No             | No             | No              | No            | Safe refusal message  | None         |
| CLI flag only (`--allow-live-network`)          | refused  | No             | No             | No              | No            | Safe refusal message  | None         |
| Env gate only (`PRISM_CHT_ENABLE_LIVE_SMOKE=1`) | refused  | No             | No             | No              | No            | Safe refusal message  | None         |
| CLI flag + env gate = `0`                       | refused  | No             | No             | No              | No            | Safe refusal message  | None         |
| CLI flag + env gate = `true`                    | refused  | No             | No             | No              | No            | Safe refusal message  | None         |
| CLI flag + env gate = `"1"`, no env gate hex check* | refused  | No             | No             | No              | No            | Safe refusal message  | None         |
| `--help`                                        | N/A      | N/A            | N/A            | N/A             | N/A           | Usage text only       | None         |
| Both gates + missing `PRISM_CHT_MODEL_API_KEY`  | error    | Attempted      | No             | No              | No            | Config error message  | None         |
| Both gates + missing `PRISM_CHT_MODEL_BASE_URL` | error    | Attempted      | No             | No              | No            | Config error message  | None         |
| Both gates + missing `PRISM_CHT_MODEL_NAME`     | error    | Attempted      | No             | No              | No            | Config error message  | None         |
| Both gates + injected fake transport (tests)    | success  | Yes (fake env) | Fake sentinel  | No              | Yes (fake)    | Safe metadata only    | None         |
| Both gates + invalid base URL (not https)       | error    | Yes            | No             | No              | No            | Config validation err | None         |

\* The env gate uses exact string comparison `== "1"`. It does NOT parse `PRISM_CHT_ENABLE_LIVE_SMOKE` as a hex, int, bool, or any other type.

### Gate Timing (Critical Safety Guarantee)

The gate evaluation happens **before** any of:
1. Config loading (`load_openai_compatible_config_from_mapping`)
2. Transport construction (`UrllibHttpTransport()`)
3. Client instantiation (`OpenAICompatibleChatModelClient`)
4. Request building (`_build_smoke_request()`)
5. HTTP request dispatch (`client.complete()`)

This means that in the refused state, no secrets are accessed and no network activity can occur, regardless of environment variable state.

Test evidence: `TestRealTransportConstructionGate` (tests `test_cht_provider_smoke_harness.py:476-567`) uses `monkeypatch` to verify that `UrllibHttpTransport.__init__` is called exactly 0 times in refused scenarios and exactly 1 time when both gates pass.

---

## 5. Output Safety Audit

### Allowed Outputs (Present)

| Output                        | Refusal | Success | Error | Safe? |
| ----------------------------- | ------- | ------- | ----- | ----- |
| Status string                 | Yes     | Yes     | Yes   | Yes   |
| Safe refusal reason            | Yes     | —       | —     | Yes   |
| Safe exception class name      | —       | —       | Yes   | Yes   |
| Configured model name          | —       | Yes     | Yes   | Yes   |
| Request count                  | —       | Yes     | Yes   | Yes   |
| Audit record count             | —       | Yes     | Yes   | Yes   |
| Provider outcome               | —       | Yes     | Yes   | Yes   |
| HTTP status code               | —       | Yes     | Yes   | Yes   |
| Elapsed ms                     | —       | Yes     | Yes   | Yes   |
| Request bytes                  | —       | Yes     | Yes   | Yes   |
| Response bytes                 | —       | Yes     | Yes   | Yes   |
| Finish reason                  | —       | Yes     | Yes   | Yes   |
| Token usage (prompt/compl/tot) | —       | Yes     | —     | Yes   |
| `--help` usage text            | —       | —       | —     | Yes   |

### Forbidden Outputs (Absent)

| Output                       | Refusal | Success | Error | Present? |
| ---------------------------- | ------- | ------- | ----- | -------- |
| API key                      | No      | No      | No    | Absent   |
| Authorization header          | No      | No      | No    | Absent   |
| Prompt body                   | No      | No      | No    | Absent   |
| Raw HTTP response body        | No      | No      | No    | Absent   |
| Assistant content             | No      | No      | No    | Absent   |
| Parsed assistant payload      | No      | No      | No    | Absent   |
| Full DTO repr with content    | No      | No      | No    | Absent   |
| Traceback by default          | No      | No      | No    | Absent   |

Test evidence:

- `TestCredentialBoundary.test_api_key_not_in_smoke_result` — sentinel API key absent from `SmokeResult` repr
- `TestCredentialBoundary.test_api_key_not_in_cli_output` — sentinel API key absent from CLI stdout
- `TestResponseBoundary.test_assistant_content_not_in_smoke_result` — assistant sentinel absent from `SmokeResult` repr
- `TestResponseBoundary.test_assistant_content_not_in_cli_output` — assistant sentinel absent from CLI stdout
- `TestResponseBoundary.test_raw_response_body_not_in_smoke_result` — raw body sentinel absent
- `TestFailureBoundary.test_http_error_no_credential_leak_in_result` — no sentinel leak in HTTP error path
- `TestFailureBoundary.test_transport_exception_no_credential_leak` — no sentinel leak in transport error path
- `OpenAICompatibleChatConfig.repr` excludes `api_key` (`test_cht_provider_config.py:112-119`)
- Config loader error messages never include API key (`test_cht_provider_config.py:271-280`)

### Minor Concern: `str(exc)` in Error Paths

The failure paths store `error_message=str(exc)` and print it to stdout. While the harness itself only raises controlled exceptions (`ProviderTransportError`, `ProviderHTTPError`, `ProviderResponseError`), the underlying HTTP library (`urllib`) can raise exceptions whose `str()` representation might include URLs or other metadata. This is a **low-risk observation**, not a blocker, because:

- The `SmokeResult` structure never includes API keys or full bodies.
- Provider client exceptions from `OpenAICompatibleChatModelClient` are defined in `provider_types.py` and are minimal.
- The only transitive risk is from `urllib.error.URLError` or similar which could include the endpoint URL in the message. Since the URL is already a configured non-secret value, this is not a leakage risk.

---

## 6. Failure-Mode Audit

### 6.1 Gate Refusal

| Property              | Value                                  |
| --------------------- | -------------------------------------- |
| Trigger               | Missing CLI flag or env gate           |
| Config loaded?        | No                                     |
| Transport constructed? | No                                    |
| Output                 | `SMOKE REFUSED: live-network execution is disabled by default.` + guidance |
| Verification           | `TestGateMatrix` (5 test methods)      |
| Safety verdict         | **Safe** — no secret access, no network |

### 6.2 Missing Required Config

| Property              | Value                                  |
| --------------------- | -------------------------------------- |
| Trigger               | Gate passes but env missing `PRISM_CHT_MODEL_API_KEY`, `PRISM_CHT_MODEL_BASE_URL`, or `PRISM_CHT_MODEL_NAME` |
| Config loaded?        | Attempted; `ProviderConfigurationError` raised |
| Transport constructed? | No                                    |
| Output                 | `SMOKE CONFIGURATION ERROR` (or similar; error derives from exception handler) with error class and message |
| API key in error?     | Never — config loader excludes API key from error messages |
| Verification           | `test_cht_provider_config.py` tests for missing keys |
| Safety verdict         | **Safe** — fails before transport, no key exposure |

### 6.3 Provider HTTP Error (e.g., 502, 429, 401)

| Property              | Value                                  |
| --------------------- | -------------------------------------- |
| Trigger               | Provider returns error HTTP status     |
| Config loaded?        | Yes                                    |
| Transport constructed? | Yes                                   |
| Request sent?         | Yes (error from response)              |
| Output                 | `SMOKE HTTP ERROR` + safe metadata fields + error class/message |
| Sensitive in result?  | No — verified by `test_http_error_no_credential_leak_in_result` |
| Safety verdict         | **Safe** — sentinel-free result, no response body in output |

### 6.4 Transport Error

| Property              | Value                                  |
| --------------------- | -------------------------------------- |
| Trigger               | DNS failure, connection refused, timeout |
| Config loaded?        | Yes                                    |
| Transport constructed? | Yes                                   |
| Request sent?         | Possibly (error may come from send)    |
| Output                 | `SMOKE TRANSPORT ERROR` + safe metadata + error class/message |
| Sensitive in result?  | No — verified by `test_transport_exception_no_credential_leak` |
| Safety verdict         | **Safe**                                 |

### 6.5 Malformed Provider Envelope

| Property              | Value                                  |
| --------------------- | -------------------------------------- |
| Trigger               | `ProviderResponseError` from client (invalid JSON, missing choices, malformed usage) |
| Config loaded?        | Yes                                    |
| Transport constructed? | Yes                                   |
| Request sent?         | Yes                                    |
| Output                 | `SMOKE RESPONSE ERROR` + safe metadata + error class/message |
| Sensitive in result?  | No — verified by response boundary tests |
| Safety verdict         | **Safe**                                 |

### 6.6 UrllibHttpTransport Construction Failure

| Property              | Value                                  |
| --------------------- | -------------------------------------- |
| Trigger               | `Exception` during `UrllibHttpTransport()` constructor |
| Config loaded?        | Yes (config loads before transport)    |
| Transport constructed? | Failed                                 |
| Request sent?         | No                                     |
| Output                 | `SMOKE TRANSPORT ERROR` with `error_class` and `error_message` |
| Config model leaked?  | `configured_model_name=None` in this path (line 183) — model name NOT exposed |
| Safety verdict         | **Safe**                                 |

### Failure Mode Verdict Summary

| Failure Mode             | Safe? | Needs Improvement? | Blocker Before Live Smoke? |
| ------------------------ | ----- | ------------------ | -------------------------- |
| Gate refusal             | Safe  | No                 | No                         |
| Missing config            | Safe  | No                 | No                         |
| Provider HTTP error       | Safe  | No                 | No                         |
| Transport error           | Safe  | No                 | No                         |
| Malformed envelope        | Safe  | No                 | No                         |
| Transport constructor err | Safe  | No                 | No                         |

---

## 7. No-RCA-Wiring Confirmation

The smoke harness has **zero** coupling to the RCA controller chain:

- **No `lead_controller` import or reference** — verified by `TestNoRCAWiring.test_no_lead_controller_in_module_namespace`
- **No `challenger_controller` import or reference** — verified by `TestNoRCAWiring.test_no_challenger_controller_in_module_namespace`
- **No `executor` import or reference** — verified by `TestNoRCAWiring.test_no_executor_in_module_namespace`
- **No `final_verifier` import or reference** — verified by `TestNoRCAWiring.test_no_final_verifier_in_module_namespace`
- **No RCA controller in source** — verified by source inspection via `inspect.getsource`
- **No RCA telemetry, hypothesis, or evidence in smoke request** — the smoke message is `"Reply with exactly: OK"`, verified by `test_recorded_body_does_not_contain_rca_evidence`

Imports in the harness are limited to:
- `http_transport` (transport abstractions)
- `llm_types` (message/request DTOs)
- `openai_compatible_client` (Provider client)
- `provider_config` (configuration)
- `provider_types` (audit and error types)

None of these import or reference the RCA layer.

---

## 8. No-Live-Network Confirmation

No live network request was performed during this audit. The following measures confirm this:

- No real API key was set; only fake sentinel values were used
- `UrllibHttpTransport` would not be constructed without both gates passing (verified by test monkeypatching)
- The `.env` file or other credential sources were never read, printed, or inspected for content
- All test evidence comes from offline test fixtures using `RecordingFakeTransport`
- The gate behavior matrix is fully covered by tests that never touch a real network

### Usage Note — Invocation Directory

The `prismv4` Python package uses namespace-package resolution. All imports of the form `from prismv4.prism_cht.*` and `from prismv4.scripts.*` resolve correctly when Python's current working directory is the git repository root:

```bash
# Correct — from the git toplevel
cd $(git rev-parse --show-toplevel)   # typically .../yyx
python3 -m pytest prismv4/tests -q
python3 prismv4/scripts/cht_provider_smoke.py --help
```

Running from inside the `prismv4/` subdirectory (e.g., `cd prismv4`) causes Python to resolve `prismv4` to the nested `prismv4/prismv4/` directory, which is an empty namespace-package skeleton. This results in `ModuleNotFoundError` at import time. This is an invocation-path caveat, not a code defect.

**Verified:** Import tests from the git toplevel resolve all five critical modules (`prismv4.prism_cht`, `.provider_config`, `.http_transport`, `.openai_compatible_client`, `prismv4.scripts.cht_provider_smoke`) successfully.

---

## 9. Readiness Verdict

### Overall Assessment: **READY for manual live smoke** — no blockers

The harness is safe, correctly gated, properly isolated from RCA controllers, and its output contract prevents secret or semantic-content leakage. The full regression suite passes (1024 passed) from the git toplevel.

### Non-Blocking Follow-Up

#### B2: `configured_model_name` Reported as `None` for Transport Constructor Failure

At `scripts/cht_provider_smoke.py:183`, when `UrllibHttpTransport()` constructor fails, `configured_model_name` is hardcoded to `None`. Config has already been loaded at line 174, so the model name is available but not propagated to the error result. This is a minor inconsistency — not a safety issue, but makes the error output less informative.

### Non-Blockers (Acceptable for Now)

1. **`str(exc)` in error messages** — Low risk; exceptions come from controlled types. The harness does not echo raw HTTP response bodies or API keys in error messages.
2. **No JSON/structured output format** — Current output is human-readable `print()` lines. Acceptable for manual smoke; structured output (JSON) could be added in a future phase.
3. **No exit code differentiation** — The harness does not call `sys.exit()` with different codes for refusal vs. error vs. success. Acceptable for manual use; can be added later.
4. **Invocation-directory sensitivity** — The harness and tests must be run from the git toplevel, not from inside the `prismv4/` subdirectory. Documented in Section 8.

---

## 10. Required Fixes Before Live Smoke

None. The harness is ready for manual live smoke. The sole follow-up item (B2 — model name in transport-constructor error path) is cosmetic, not a safety or correctness issue.

---

## 11. Explicit Statement: No Runtime Behavior Changed

This audit added exactly one documentation file. No Python source file, test file, configuration file, or any other runtime artifact was modified. The working tree diff contains only the new audit document.

---

## 12. Audit Summary

| Item                             | Status                        |
| -------------------------------- | ----------------------------- |
| Double opt-in gate design        | Correct and safe              |
| Gate timing (before config)      | Correct and safe              |
| Config key inventory              | Complete; 3 required, 3 opt  |
| API key in `repr()`              | Excluded (`repr=False`)       |
| Config error messages             | API-key-free                  |
| Smoke message content             | Minimal and non-sensitive     |
| Output in refusal path            | Safe                          |
| Output in success path            | Safe (no content leakage)     |
| Output in error path              | Safe (sentinel-free)          |
| `--help` text                     | Safe; no examples, no secrets |
| `os.environ` read timing          | Only in `main()`, not `run_smoke()` |
| Import side effects               | None                          |
| RCA controller isolation          | Complete and verified         |
| Live network performed            | None (zero)                   |
| Module import path                | Resolves from git toplevel; invocation caveat documented |
