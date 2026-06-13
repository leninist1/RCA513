"""CHT Provider smoke harness — minimal, isolated, fail-closed.

This script refuses live-network execution by default.  A real network
transport is constructed only when TWO independent opt-in signals are
present:

    1. ``--allow-live-network`` CLI flag
    2. ``PRISM_CHT_ENABLE_LIVE_SMOKE=1`` environment variable

The harness is independently testable with an injected fake transport
via the ``run_smoke()`` function.  It is fully disconnected from the
RCA Controller chain (Lead/Challenger/Executor/FinalVerifier).

Output is limited to safe operational metadata — no API key, Prompt
body, raw HTTP response body, or parsed assistant content is ever
printed.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Mapping

from prismv4.prism_cht.http_transport import HttpTransport, UrllibHttpTransport
from prismv4.prism_cht.llm_types import ModelMessage, ModelRequest
from prismv4.prism_cht.openai_compatible_client import (
    OpenAICompatibleChatModelClient,
)
from prismv4.prism_cht.provider_config import (
    load_openai_compatible_config_from_mapping,
)
from prismv4.prism_cht.provider_types import (
    ProviderCallAudit,
    ProviderHTTPError,
    ProviderResponseError,
    ProviderTransportError,
)


# ===========================================================================
# Smoke environment gate
# ===========================================================================

_ENV_LIVE_SMOKE = "PRISM_CHT_ENABLE_LIVE_SMOKE"


def _live_smoke_enabled(environ: Mapping[str, str]) -> bool:
    """Return True only when PRISM_CHT_ENABLE_LIVE_SMOKE is exactly '1'."""
    return environ.get(_ENV_LIVE_SMOKE) == "1"


# ===========================================================================
# SmokeResult
# ===========================================================================


@dataclass(frozen=True)
class SmokeResult:
    """Safe metadata returned by a single provider smoke call.

    Never includes API keys, Prompt bodies, raw response bodies,
    authorization headers, or parsed assistant content.
    """

    status: str  # "permitted", "refused", "transport_error", "http_error", "response_error", "success"
    configured_model_name: str | None  # None when gate refused
    request_count: int  # 0 when gate refused
    audit_record_count: int  # 0 when gate refused
    provider_outcome: str | None  # None when gate refused or no call made

    http_status_code: int | None
    elapsed_ms: float | None
    request_bytes: int | None
    response_bytes: int | None
    finish_reason: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None

    error_class: str | None
    error_message: str | None


_SMOKE_PURPOSE = "cht_provider_smoke_harness"
_SMOKE_MESSAGE_CONTENT = "Reply with exactly: OK"


def _build_smoke_request() -> ModelRequest:
    """Build exactly one minimal, non-sensitive model request."""
    message = ModelMessage(role="user", content=_SMOKE_MESSAGE_CONTENT)
    return ModelRequest(
        purpose=_SMOKE_PURPOSE,
        messages=(message,),
        attempt_index=0,
    )


def _safe_audit_metadata(
    audit: ProviderCallAudit,
) -> dict:
    """Extract safe metadata from a ProviderCallAudit record."""
    return {
        "outcome": audit.outcome,
        "http_status_code": audit.status_code,
        "elapsed_ms": audit.elapsed_ms,
        "request_bytes": audit.request_bytes,
        "response_bytes": audit.response_bytes,
        "finish_reason": audit.finish_reason,
        "prompt_tokens": audit.usage.prompt_tokens,
        "completion_tokens": audit.usage.completion_tokens,
        "total_tokens": audit.usage.total_tokens,
    }


def _build_refused_result() -> SmokeResult:
    """Return a refusal result with all metadata set to None/0."""
    return SmokeResult(
        status="refused",
        configured_model_name=None,
        request_count=0,
        audit_record_count=0,
        provider_outcome=None,
        http_status_code=None,
        elapsed_ms=None,
        request_bytes=None,
        response_bytes=None,
        finish_reason=None,
        prompt_tokens=None,
        completion_tokens=None,
        total_tokens=None,
        error_class=None,
        error_message=None,
    )


# ===========================================================================
# Core smoke function (testable, no os.environ)
# ===========================================================================


def run_smoke(
    *,
    allow_live_network: bool,
    environ: Mapping[str, str],
    transport: HttpTransport | None = None,
) -> SmokeResult:
    """Run exactly one minimal provider smoke request.

    Args:
        allow_live_network: Must be True, together with environment gate,
            before a real transport is constructed.
        environ: An explicit Mapping providing configuration and gate
            environment variables.  The function never reads ``os.environ``.
        transport: An optional test-local ``HttpTransport``.  When
            supplied, it is used directly and the double opt-in gate is
            bypassed for transport construction (but configuration is
            still loaded from *environ*).

    Returns:
        ``SmokeResult`` containing only safe operational metadata.
    """
    # -- Gate evaluation (before any config / transport / key access) --------
    if transport is None:
        gate_passed = allow_live_network and _live_smoke_enabled(environ)
    else:
        gate_passed = True

    if not gate_passed:
        return _build_refused_result()

    # -- Load configuration from the explicit environ mapping ----------------
    config = load_openai_compatible_config_from_mapping(environ)

    # -- Construct transport (only if no injected transport) -----------------
    if transport is None:
        try:
            transport = UrllibHttpTransport()
        except Exception as exc:
            return SmokeResult(
                status="transport_error",
                configured_model_name=None,
                request_count=0,
                audit_record_count=0,
                provider_outcome=None,
                http_status_code=None,
                elapsed_ms=None,
                request_bytes=None,
                response_bytes=None,
                finish_reason=None,
                prompt_tokens=None,
                completion_tokens=None,
                total_tokens=None,
                error_class=type(exc).__name__,
                error_message=str(exc),
            )

    # -- Build the Provider client -------------------------------------------
    client = OpenAICompatibleChatModelClient(
        config=config,
        transport=transport,
    )

    # -- Send exactly one minimal request ------------------------------------
    smoke_request = _build_smoke_request()

    try:
        client.complete(request=smoke_request)
    except ProviderTransportError as exc:
        audit = client.audit_records[0] if client.audit_records else None
        meta = (_safe_audit_metadata(audit) if audit else {})
        return SmokeResult(
            status="transport_error",
            configured_model_name=config.model,
            request_count=1,
            audit_record_count=len(client.audit_records),
            provider_outcome=meta.get("outcome"),
            http_status_code=meta.get("http_status_code"),
            elapsed_ms=meta.get("elapsed_ms"),
            request_bytes=meta.get("request_bytes"),
            response_bytes=meta.get("response_bytes"),
            finish_reason=meta.get("finish_reason"),
            prompt_tokens=meta.get("prompt_tokens"),
            completion_tokens=meta.get("completion_tokens"),
            total_tokens=meta.get("total_tokens"),
            error_class=type(exc).__name__,
            error_message=str(exc),
        )
    except ProviderHTTPError as exc:
        audit = client.audit_records[0] if client.audit_records else None
        meta = (_safe_audit_metadata(audit) if audit else {})
        return SmokeResult(
            status="http_error",
            configured_model_name=config.model,
            request_count=1,
            audit_record_count=len(client.audit_records),
            provider_outcome=meta.get("outcome"),
            http_status_code=meta.get("http_status_code"),
            elapsed_ms=meta.get("elapsed_ms"),
            request_bytes=meta.get("request_bytes"),
            response_bytes=meta.get("response_bytes"),
            finish_reason=meta.get("finish_reason"),
            prompt_tokens=meta.get("prompt_tokens"),
            completion_tokens=meta.get("completion_tokens"),
            total_tokens=meta.get("total_tokens"),
            error_class=type(exc).__name__,
            error_message=str(exc),
        )
    except ProviderResponseError as exc:
        audit = client.audit_records[0] if client.audit_records else None
        meta = (_safe_audit_metadata(audit) if audit else {})
        return SmokeResult(
            status="response_error",
            configured_model_name=config.model,
            request_count=1,
            audit_record_count=len(client.audit_records),
            provider_outcome=meta.get("outcome"),
            http_status_code=meta.get("http_status_code"),
            elapsed_ms=meta.get("elapsed_ms"),
            request_bytes=meta.get("request_bytes"),
            response_bytes=meta.get("response_bytes"),
            finish_reason=meta.get("finish_reason"),
            prompt_tokens=meta.get("prompt_tokens"),
            completion_tokens=meta.get("completion_tokens"),
            total_tokens=meta.get("total_tokens"),
            error_class=type(exc).__name__,
            error_message=str(exc),
        )
    except Exception as exc:
        audit = client.audit_records[0] if client.audit_records else None
        meta = (_safe_audit_metadata(audit) if audit else {})
        return SmokeResult(
            status="response_error",
            configured_model_name=config.model,
            request_count=1,
            audit_record_count=len(client.audit_records),
            provider_outcome=meta.get("outcome"),
            http_status_code=meta.get("http_status_code"),
            elapsed_ms=meta.get("elapsed_ms"),
            request_bytes=meta.get("request_bytes"),
            response_bytes=meta.get("response_bytes"),
            finish_reason=meta.get("finish_reason"),
            prompt_tokens=meta.get("prompt_tokens"),
            completion_tokens=meta.get("completion_tokens"),
            total_tokens=meta.get("total_tokens"),
            error_class=type(exc).__name__,
            error_message=str(exc),
        )

    # -- Success — extract safe metadata from the audit record ---------------
    audit = client.audit_records[0]
    meta = _safe_audit_metadata(audit)

    return SmokeResult(
        status="success",
        configured_model_name=config.model,
        request_count=1,
        audit_record_count=len(client.audit_records),
        provider_outcome=meta.get("outcome"),
        http_status_code=meta.get("http_status_code"),
        elapsed_ms=meta.get("elapsed_ms"),
        request_bytes=meta.get("request_bytes"),
        response_bytes=meta.get("response_bytes"),
        finish_reason=meta.get("finish_reason"),
        prompt_tokens=meta.get("prompt_tokens"),
        completion_tokens=meta.get("completion_tokens"),
        total_tokens=meta.get("total_tokens"),
        error_class=None,
        error_message=None,
    )


# ===========================================================================
# CLI wrapper (reads os.environ only in main after arg parsing)
# ===========================================================================


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CHT Provider smoke harness — minimal, fail-closed.",
    )
    parser.add_argument(
        "--allow-live-network",
        action="store_true",
        default=False,
        help=(
            "Explicitly authorize live-network provider access.  "
            "Also requires PRISM_CHT_ENABLE_LIVE_SMOKE=1."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """CLI entry point — reads os.environ only after argument parsing."""
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    result = run_smoke(
        allow_live_network=args.allow_live_network,
        environ=dict(os.environ),
    )

    if result.status == "refused":
        print("SMOKE REFUSED: live-network execution is disabled by default.")
        print(
            "  Use --allow-live-network AND set PRISM_CHT_ENABLE_LIVE_SMOKE=1 "
            "to enable."
        )
    elif result.status == "success":
        print("SMOKE SUCCESS")
        print(f"  model: {result.configured_model_name}")
        print(f"  request_count: {result.request_count}")
        print(f"  audit_records: {result.audit_record_count}")
        print(f"  provider_outcome: {result.provider_outcome}")
        print(f"  http_status_code: {result.http_status_code}")
        print(f"  elapsed_ms: {result.elapsed_ms:.1f}" if result.elapsed_ms is not None else "  elapsed_ms: N/A")
        print(f"  request_bytes: {result.request_bytes}")
        print(f"  response_bytes: {result.response_bytes}")
        print(f"  finish_reason: {result.finish_reason}")
        if result.prompt_tokens is not None:
            print(f"  prompt_tokens: {result.prompt_tokens}")
        if result.completion_tokens is not None:
            print(f"  completion_tokens: {result.completion_tokens}")
        if result.total_tokens is not None:
            print(f"  total_tokens: {result.total_tokens}")
    else:
        kind = result.status.replace("_", " ").upper()
        print(f"SMOKE {kind}")
        print(f"  model: {result.configured_model_name}")
        print(f"  request_count: {result.request_count}")
        print(f"  audit_records: {result.audit_record_count}")
        print(f"  provider_outcome: {result.provider_outcome}")
        print(f"  http_status_code: {result.http_status_code}")
        print(f"  elapsed_ms: {result.elapsed_ms:.1f}" if result.elapsed_ms is not None else "  elapsed_ms: N/A")
        print(f"  error_class: {result.error_class}")
        print(f"  error_message: {result.error_message}")


if __name__ == "__main__":
    main()
