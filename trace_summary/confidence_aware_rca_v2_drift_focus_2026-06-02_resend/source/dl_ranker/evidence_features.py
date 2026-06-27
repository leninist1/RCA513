"""Evidence card features computed per-candidate without the full rule engine.

These approximate the support/refute/blind signals that the heuristic's
rule-evaluation loop produces, using fast evidence queries.
"""
from __future__ import annotations

from typing import Any


_OOM_KEYWORDS = ("oom", "outofmemory", "out of memory", "heap space", "gc overhead")
_NET_KEYWORDS = ("timeout", "connection", "refused", "delay", "slow")


def compute_evidence_features(
    comp: str,
    bucket: str,
    trace_summary: dict[str, Any] | None,
    log_text: str,
    trace_avail: bool,
    log_avail: bool,
    metric_avail: bool,
    first_anomalous_svc: str,
    roles: dict,
    service_stats: dict,
) -> list[float]:
    """Compute 6 evidence-card-like features for a candidate.

    Returns: [trace_support, slowedge_support, prop_support, log_support,
              blind_trace, blind_log]
    """
    trace_support = 0.0
    slowedge_support = 0.0
    prop_support = 0.0
    log_support = 0.0
    blind_trace = 0.0
    blind_log = 0.0

    if trace_avail and comp in service_stats:
        # first_anomalous_service match → strong trace support
        if comp == first_anomalous_svc:
            trace_support = 2.0
        # slow edges support
        slow_edges = trace_summary.get("events", {}).get("slow_edges", []) if trace_summary else []
        for e in slow_edges:
            if str(e.get("src")) == comp or str(e.get("dst")) == comp:
                slowedge_support = float(e.get("slow_ratio", 1.0) or 1.0) - 1.0
                break
        # propagation role support: upstream_source is strongest
        role = roles.get(comp)
        if role and role.role == "upstream_source":
            prop_support = 1.5
        elif role and role.role == "middle_propagator":
            prop_support = 0.5
    else:
        blind_trace = 1.0

    if log_avail and log_text:
        if bucket in ("jvm_oom", "memory"):
            log_support = 2.0 if any(kw in log_text for kw in _OOM_KEYWORDS) else 0.0
        elif bucket in ("network_latency", "network_packet_loss"):
            log_support = 1.0 if any(kw in log_text for kw in _NET_KEYWORDS) else 0.0
    else:
        blind_log = 0.3  # soft blind

    return [
        trace_support / 2.0,
        slowedge_support / 3.0,
        prop_support / 1.5,
        log_support / 2.0,
        blind_trace,
        blind_log,
    ]
