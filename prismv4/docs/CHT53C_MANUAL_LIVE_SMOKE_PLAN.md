# CHT-5.3C Manual Live Smoke Execution Plan

## 1. Scope and Parent Commit

- **Parent commit:** `4417324`
- **Parent branch:** `audit/cht53b-smoke-cli-output`
- **Plan branch:** `docs/cht53c-manual-live-smoke-plan`

This document defines the exact procedure a human operator must follow to execute **one** minimal
Provider live smoke via the CHT provider smoke harness (`prismv4/scripts/cht_provider_smoke.py`).
It does **not** execute a live request itself.

The harness was audited in CHT-5.3B (`docs/CHT53B_SMOKE_HARNESS_CLI_AUDIT.md`) and found
ready for manual live smoke with no safety blockers.

---

## 2. Preconditions

Before executing a live smoke:

1. **Repository root:** All commands must run from the git repository toplevel
   (`git rev-parse --show-toplevel`), not from inside the `prismv4/` subdirectory.

2. **Full regression must pass:**
   ```bash
   python3 -m pytest prismv4/tests -q
   ```
   Do not proceed if any test fails.

3. **Manual and opt-in only:** This smoke must never run in CI or automation.
   It requires a conscious human decision and explicit double-opt-in.

4. **No real secrets in logs or terminals:** Do not paste, commit, screenshot,
   or share real API keys. Do not redirect live-smoke output to a shared log file
   without sanitization.

5. **Audit review:** The operator should have read
   `docs/CHT53B_SMOKE_HARNESS_CLI_AUDIT.md` before proceeding.

---

## 3. Required Environment Variables

All environment variables must be set inline in the invocation command. Do **not**
export them or write them to `.env`, `.bashrc`, or any persistent file.

### Required (all three must be non-empty)

| Variable                   | Classification | Purpose                              |
| -------------------------- | -------------- | ------------------------------------ |
| `PRISM_CHT_MODEL_BASE_URL` | endpoint       | HTTPS base URL of the Provider       |
| `PRISM_CHT_MODEL_NAME`     | model id       | Model identifier (e.g., `gpt-4o`)    |
| `PRISM_CHT_MODEL_API_KEY`  | **secret**     | Provider API key                     |

### Required Gate Variable

| Variable                       | Value | Purpose                             |
| ------------------------------ | ----- | ----------------------------------- |
| `PRISM_CHT_ENABLE_LIVE_SMOKE`  | `1`   | Double-opt-in environment gate      |

The gate uses an exact character comparison `== "1"`. Any other value
(`"0"`, `"true"`, absent) refuses the request.

### Optional (defaults are acceptable)

| Variable                           | Default    | Purpose                        |
| ---------------------------------- | ---------- | ------------------------------ |
| `PRISM_CHT_MODEL_TIMEOUT_SECONDS`  | `60.0`     | Request timeout in seconds     |
| `PRISM_CHT_MODEL_MAX_TOKENS`       | `4096`     | Maximum completion tokens      |
| `PRISM_CHT_MODEL_MAX_RESPONSE_BYTES` | `2000000`  | Maximum response body bytes    |

---

## 4. Explicit Double Opt-In Command

From the git repository toplevel, run:

```bash
PRISM_CHT_ENABLE_LIVE_SMOKE=1 \
PRISM_CHT_MODEL_BASE_URL="<YOUR_HTTPS_ENDPOINT>" \
PRISM_CHT_MODEL_NAME="<YOUR_MODEL_NAME>" \
PRISM_CHT_MODEL_API_KEY="<YOUR_API_KEY>" \
python3 prismv4/scripts/cht_provider_smoke.py --allow-live-network
```

Placeholders `<...>` must be replaced with real values **only at invocation time**.
Never write real values into any file in the repository.

The command constructs the transport and client **only after** both gates pass:

1. `--allow-live-network` CLI flag = `True`
2. `PRISM_CHT_ENABLE_LIVE_SMOKE` = `1`

If either gate is missing, the harness prints a refusal message and exits
**without** loading configuration, constructing transport, or accessing secrets.

---

## 5. Safe Invocation Directory

```bash
cd $(git rev-parse --show-toplevel)
```

The harness and tests use `prismv4.prism_cht.*` imports that depend on namespace-package
resolution from the repository toplevel. Running from inside the `prismv4/` subdirectory
causes `ModuleNotFoundError` because Python resolves `prismv4` to the nested skeleton
directory.

---

## 6. Allowed Output Fields

The smoke harness prints only the following to stdout:

| Field                  | Example                     | Notes                                   |
| ---------------------- | --------------------------- | --------------------------------------- |
| Status banner          | `SMOKE SUCCESS`             | Or `SMOKE REFUSED`, `SMOKE HTTP ERROR`, etc. |
| Model name             | `model: gpt-4o`             | The configured model identifier         |
| Request count          | `request_count: 1`          | Always `1` on success                   |
| Audit record count     | `audit_records: 1`          | Must match request count                |
| Provider outcome       | `provider_outcome: success` | From `ProviderCallAudit.outcome`         |
| HTTP status code       | `http_status_code: 200`     | Safe metadata                           |
| Elapsed milliseconds   | `elapsed_ms: 1234.5`        | Safe metadata                           |
| Request bytes          | `request_bytes: 250`        | Safe metadata                           |
| Response bytes         | `response_bytes: 150`       | Safe metadata                           |
| Finish reason          | `finish_reason: stop`       | Model-reported finish reason            |
| Token usage            | `prompt_tokens: 10`         | If available from the provider          |
| Error class            | `error_class: ProviderHTTPError` | Safe exception class name only     |
| Error message          | `error_message: ...`        | Safe error text from exception          |
| Refusal guidance       | `live-network execution is disabled` | Gate refusal message            |

Acceptable variations:
- `elapsed_ms: N/A` when not applicable
- `model: None` when gate refused

---

## 7. Forbidden Output Fields

None of the following may appear in stdout, stderr, or any captured output:

| Forbidden Content             | Reason                                    |
| ----------------------------- | ----------------------------------------- |
| API key (any form)            | Secret — must never be printed            |
| `Authorization` header        | Contains bearer token                     |
| Prompt body                   | May contain provider-identifying data     |
| Raw HTTP response body        | May contain sensitive payload             |
| Assistant content             | Provider-generated text                   |
| Parsed assistant payload      | Derived from content                      |
| Full DTO `repr()`             | May embed semantic content                |
| Python traceback (by default) | Only exception class name + message       |
| RCA controller references     | Not part of the smoke harness             |

Verification: the CHT-5.3B audit confirmed that all output paths (refusal, success,
HTTP error, transport error, response error) are free of these forbidden fields.

---

## 8. Pass/Fail Criteria

### The live smoke **passes** only if ALL of:

1. Exactly one request is attempted (`request_count: 1`).
2. The command exits with exit code `0`.
3. Output contains only allowed fields (Section 6).
4. No forbidden field appears in output (Section 7).
5. `audit_record_count` matches `request_count` (both `1`).
6. `provider_outcome` is `success`.
7. `http_status_code` is in the expected success range (typically `200`).
8. No RCA controller name appears in the module namespace or output.

### The live smoke **fails** if ANY of:

1. Any secret (API key, auth header) appears in output.
2. Prompt body, response body, or assistant content appears in output.
3. More than one request is attempted.
4. A traceback is printed by default (exception class + message is ok).
5. Config is loaded or transport is constructed in a refused scenario.
6. The output includes any RCA controller reference.
7. The request count is `0` (indicating gate refusal when live was intended).
8. An error status (`http_error`, `transport_error`, `response_error`) is reported
   with a persistent cause (not a transient network issue).

---

## 9. Failure Handling

| Failure Type              | Action                                                           |
| ------------------------- | ---------------------------------------------------------------- |
| Gate refusal (misconfiguration) | Check both `--allow-live-network` and `PRISM_CHT_ENABLE_LIVE_SMOKE=1` are set |
| Config error (missing env)      | Verify all three required env vars are set and non-empty        |
| Provider HTTP error (4xx/5xx)   | Check endpoint URL, model name, API key validity, and quota    |
| Transport error (DNS/connection)| Verify network access to the endpoint; check firewall rules    |
| Response error (malformed)      | Verify Provider compatibility; check API version               |

Do not retry with modified API keys in the same terminal session without sanitizing
the terminal scrollback and shell history.

---

## 10. Secret-Handling Rules

1. **Never** write a real API key into any file tracked by version control.
2. **Never** export `PRISM_CHT_MODEL_API_KEY` in a shell profile (`.bashrc`, `.zshrc`, etc.).
3. **Always** pass the key inline as a single-command environment variable:
   ```
   PRISM_CHT_MODEL_API_KEY="sk-..." python3 prismv4/scripts/cht_provider_smoke.py --allow-live-network
   ```
4. **Never** redirect live-smoke stdout to a file that may be committed or shared.
5. **After** the smoke, inspect your shell history for any accidentally pasted keys
   and remove them (`history -d <line>`, or clear history for the session).
6. Rotate the API key immediately if it was ever committed, pasted into chat,
   or captured in a screenshot.

---

## 11. Rollback / Cleanup

After executing the live smoke:

1. Clear the shell environment variables if exported (open a new shell session
   if needed).
2. Verify no `.env` or config file was accidentally created during the session.
3. Run a quick refused-smoke to confirm the harness defaults back to deny:
   ```bash
   python3 prismv4/scripts/cht_provider_smoke.py
   ```
   Expected output:
   ```
   SMOKE REFUSED: live-network execution is disabled by default.
   ```
4. Remove the API key from shell history if it was typed directly.

---

## 12. Statement: This Phase Does Not Execute Live Smoke

This document is a **plan only**. No live Provider request was made during its creation.
No real API key was used, stored, or transmitted. The content of this plan is derived
entirely from static analysis of the harness source, test code, and audit documentation.
