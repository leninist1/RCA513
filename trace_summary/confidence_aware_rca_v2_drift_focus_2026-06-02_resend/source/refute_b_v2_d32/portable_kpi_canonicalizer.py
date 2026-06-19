"""Dataset-local KPI vocabulary canonicalization for portable D32 runs.

The OpenRCA D32 pipeline relies on KPI-name tokens for reason buckets.  Public
portable datasets often use different metric names, so this module rewrites only
the KPI *names* into OpenRCA-compatible tokens while keeping one stable name per
original KPI via a short hash suffix.  Values, timestamps, and components are
unchanged.
"""
from __future__ import annotations

import hashlib
import re
from typing import Any

import pandas as pd


def canonicalize_metric_kpis(
    frame: pd.DataFrame,
    *,
    dataset: str,
    mode: str = "none",
) -> pd.DataFrame:
    """Return a copy with KPI names canonicalized for portable OpenRCA flow.

    ``mode="none"`` is a no-op.  ``mode="openrca"`` maps AIOps/Eadro metric
    names to tokens consumed by the existing OpenRCA D32 reason and evidence
    bucket functions.
    """

    if frame is None or frame.empty or mode in {"", "none", None}:
        return frame.copy() if frame is not None else pd.DataFrame()
    if mode != "openrca":
        raise ValueError(f"unsupported KPI canonicalization mode: {mode}")
    if "kpi_name" not in frame.columns:
        return frame.copy()
    out = frame.copy()
    out["kpi_name"] = out["kpi_name"].map(lambda value: canonicalize_kpi_name(value, dataset=dataset))
    return out


def canonicalize_kpi_name(value: Any, *, dataset: str = "") -> str:
    original = str(value)
    low = original.lower()
    collapsed = re.sub(r"[^a-z0-9]+", "", low)
    suffix = _stable_suffix(original)

    if _is_cpu(low, collapsed):
        return f"OpenRCA_CPU_CPULoad_{suffix}"
    if _is_jvm_memory(low, collapsed):
        return f"OpenRCA_JVM_HeapMemory_{suffix}"
    if _is_memory(low, collapsed):
        return f"OpenRCA_MEMORY_MemUsage_{suffix}"
    if _is_filesystem(low, collapsed):
        return f"OpenRCA_FILESYSTEM_FSCapacity_{suffix}"
    if _is_disk_io(low, collapsed):
        return f"OpenRCA_DSK_ReadWrite_{suffix}"
    if _is_network_loss(low, collapsed):
        return f"OpenRCA_NetworkPacketErrLoss_{suffix}"
    if _is_network_latency(low, collapsed):
        return f"OpenRCA_Network_latency_{suffix}"
    if _is_process_termination(low, collapsed):
        return f"OpenRCA_process_termination_{suffix}"
    return original


def _stable_suffix(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:10]


def _is_cpu(low: str, collapsed: str) -> bool:
    return any(token in collapsed for token in (
        "cpu",
        "cpuload",
        "cpuidle",
        "singlecpu",
        "processorload",
        "procppcpuperc",
        "usedcpusys",
        "usedcpuuser",
    ))


def _is_jvm_memory(low: str, collapsed: str) -> bool:
    return ("jvm" in collapsed or "heap" in collapsed) and any(token in collapsed for token in (
        "memory",
        "mem",
        "heap",
    ))


def _is_memory(low: str, collapsed: str) -> bool:
    return any(token in collapsed for token in (
        "memory",
        "mem",
        "swap",
        "cachemem",
        "nocachemem",
        "usermem",
        "procppmem",
        "usedmemory",
        "usedmemoryrss",
    ))


def _is_filesystem(low: str, collapsed: str) -> bool:
    return any(token in collapsed for token in (
        "filesystem",
        "fscapacity",
        "fsavailable",
        "fsused",
        "fsinoden",
        "fsinode",
        "filesizemb",
    ))


def _is_disk_io(low: str, collapsed: str) -> bool:
    return any(token in collapsed for token in (
        "localdisk",
        "diskio",
        "dsk",
        "iowait",
        "await",
        "svctm",
        "readwrite",
        "readbytes",
        "writebytes",
        "disktps",
    ))


def _is_network_loss(low: str, collapsed: str) -> bool:
    explicit_loss = any(token in low for token in (
        " error",
        "_error",
        "-error",
        "err",
        "drop",
        "loss",
        "retrans",
        "reject",
        "rejected",
        "reset",
        "abort",
        "aborted",
        "failed",
        "failure",
    ))
    tomcat_loss = any(token in collapsed for token in (
        "errorcountrequestinfo",
        "sessionrejectedsessions",
        "rejectedrequests",
    ))
    return explicit_loss or tomcat_loss


def _is_network_latency(low: str, collapsed: str) -> bool:
    request_latency = any(token in collapsed for token in (
        "maxtimerequestinfo",
        "processingtimerequestinfo",
        "responsetime",
        "latency",
        "duration",
    ))
    generic_network = any(token in collapsed for token in (
        "network",
        "net",
        "tcp",
        "icmp",
        "ping",
        "packetsin",
        "packetsout",
        "packet",
        "traffic",
        "recv",
        "send",
        "sent",
        "bandwidth",
        "totalconn",
        "closewait",
        "finwait",
        "queue",
        "tnsping",
    ))
    return request_latency or generic_network


def _is_process_termination(low: str, collapsed: str) -> bool:
    return any(token in collapsed for token in (
        "zombie",
        "processrestart",
        "processkill",
        "procnoprocess",
        "procnumb",
    ))
