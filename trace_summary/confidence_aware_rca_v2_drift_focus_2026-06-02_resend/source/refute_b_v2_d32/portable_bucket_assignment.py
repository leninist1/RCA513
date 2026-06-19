"""Dataset-native KPI -> reason-bucket assignment for portable D32 runs.

The OpenRCA D32 pipeline historically decides which "reason bucket" a KPI
belongs to via hardcoded token matching (see ``evidence.kpi_in_bucket``,
``signature.reason_for_kpi``, ``joint_candidates._kpi_buckets``).  Those token
rules assume OpenRCA's own KPI vocabulary and do not generalise to public
portable datasets such as AIOps2021, where application-layer symptoms
(Tomcat error/rejected-session counters, Redis rejected connections) are
raised under *every* fault type and therefore pollute the packet-loss bucket.

This module supplies a dataset-native assignment instead: it maps each AIOps
KPI name to the reason bucket(s) it genuinely evidences, and deliberately
returns an *empty* set for cross-fault propagation symptoms that have no
discriminative value.  The result is attached to ``metric_df`` as a precomputed
``buckets`` column by the adapter layer (see ``portable_adapters``); the
algorithm-side bucketing functions then read that column via
``bucket_resolver`` and fall back to the original OpenRCA token logic when the
column is absent (so the OpenRCA dataset path stays byte-identical).

The assignment is purely lexical on the raw AIOps KPI name and uses no labels,
no OpenRCA vocabulary, and no token-matching against OpenRCA shapes.
"""
from __future__ import annotations

import re
from typing import Any

# Reason buckets recognised by the D32 pipeline (subset of schema.REASON_BUCKETS
# values).  App-symptom / load-counter KPIs intentionally map to the empty set.
_BUCKETS = (
    "cpu",
    "memory",
    "jvm_oom",
    "disk_io",
    "filesystem",
    "network_latency",
    "network_packet_loss",
)


def assign_kpi_buckets(kpi_name: Any, component: Any = "") -> set[str]:
    """Return the set of reason buckets a KPI evidences for AIOps2021.

    Returns an empty set for cross-fault propagation symptoms (Tomcat
    error/rejected-session/request counters, Redis rejected connections, Tomcat
    thread/session load counters) and for KPIs that carry no fault evidence
    (host uptime, NTP offset, generic internal stats).  Such KPIs must NOT
    enter any reason bucket, otherwise they inflate the wrong bucket's density
    and bias the reason posterior.
    """
    original = str(kpi_name)
    low = original.lower()
    collapsed = re.sub(r"[^a-z0-9]+", "", low)

    # --- Explicit exclusions: propagation symptoms / load counters / non-fault ---
    # These are raised under multiple fault types (cpu, network latency, packet
    # loss, jvm oom) and carry no discriminative reason signal.
    if _is_excluded_app_symptom(low, collapsed):
        return set()

    # --- Hardware / OS buckets (checked before generic network) ---
    if _is_cpu(low, collapsed):
        return {"cpu"}
    if _is_jvm_oom(low, collapsed):
        return {"jvm_oom"}
    if _is_memory(low, collapsed):
        return {"memory"}
    if _is_filesystem(low, collapsed):
        return {"filesystem"}
    if _is_disk_io(low, collapsed):
        return {"disk_io"}

    # --- Network: distinguish NIC error counters (loss) from throughput/latency ---
    if _is_network_packet_loss(low, collapsed):
        return {"network_packet_loss"}
    if _is_network_latency(low, collapsed):
        return {"network_latency"}

    # --- DB response time -> latency evidence ---
    if _is_db_latency(low, collapsed):
        return {"network_latency"}

    # Everything else (host uptime, NTP offset, Mysql/Redis internal stats,
    # Tomcat request/session/thread load counters not caught above) carries no
    # fault-evidence signal and is excluded.
    return set()


def excluded_buckets(kpi_name: Any, component: Any = "") -> set[str]:
    """Return the (empty) bucket set for KPIs that should enter no bucket.

    Exposed as a named helper so callers/tests can express intent.  Equivalent
    to ``assign_kpi_buckets`` returning an empty set.
    """
    return set()


# ---------------------------------------------------------------------------
# Detectors -- AIOps2021-native, no OpenRCA token shapes.
# ---------------------------------------------------------------------------


def _is_excluded_app_symptom(low: str, collapsed: str) -> bool:
    # Tomcat request error / rejected session counters: cross-fault propagation.
    if any(t in collapsed for t in (
        "errorcountrequestinfo",
        "sessionrejectedsessions",
        "rejectedrequests",
    )):
        return True
    # Tomcat session load counters (active/keepalive) and thread load counters:
    # load/availability signals, not fault evidence.
    if any(t in collapsed for t in (
        "sessionactivecounter",
        "sessionkeepalivecounter",
        "currentthreadcountthreadinfo",
        "currentthreadsbusythreadinfo",
        "maxthreadsthreadinfo",
        "requestcountrequestinfo",
    )):
        return True
    # Redis rejected connections: raised under both loss and latency faults.
    if "rejected_connections" in low:
        return True
    return False


def _is_cpu(low: str, collapsed: str) -> bool:
    if "cpuload" in collapsed or "cpuidle" in collapsed or "cpuutil" in collapsed:
        return True
    if "singlecpuutil" in collapsed or "singlecpuidle" in collapsed:
        return True
    if "cpusystime" in collapsed or "cpuusertime" in collapsed:
        return True
    if "cpuwio" in collapsed or "cpuidleutil" in collapsed:
        return True
    # Container / JVM / Redis process CPU.
    if "cpupercent" in collapsed:
        return True
    if "jvmcpuload" in collapsed or "usedcpusys" in collapsed or "usedcpuuser" in collapsed:
        return True
    if "procppcpuperc" in collapsed:
        return True
    return False


def _is_jvm_oom(low: str, collapsed: str) -> bool:
    # JVM heap memory / Tomcat JVM memory -> OOM evidence (more specific than
    # generic memory, checked before _is_memory).
    if "heapmemory" in collapsed:
        return True
    if ("jvmfreememory" in collapsed or "jvmmaxmemory" in collapsed
            or "jvmmemoryusedpercent" in collapsed or "jvmtotalmemory" in collapsed
            or "jvmusedmemory" in collapsed):
        return True
    if "jvmmemory" in collapsed and ("heap" in collapsed or "nomheap" in collapsed):
        return True
    return False


def _is_memory(low: str, collapsed: str) -> bool:
    if any(t in collapsed for t in (
        "memfreemem", "memusedmemperc", "nocachememperc", "usermem",
        "cachemem", "procppmem", "procppmemperc", "memtotalmem",
    )):
        return True
    if any(t in collapsed for t in (
        "usedmemory", "usedmemoryrss", "usedmemory",
        "usedmemorypeak", "usedmemoryrss", "maxmemory",
        "memfragmentationratio",
    )):
        return True
    # Container memory percent / usage / limit.
    if any(t in collapsed for t in ("mempercent", "memusage", "memlimit")):
        return True
    if "swaptotswapusedpercent" in collapsed or "swapsi" in collapsed or "swapso" in collapsed:
        return True
    # Mysql query-cache memory / row-lock memory bytes -> memory pressure.
    if "qcachefreememory" in collapsed or "qcachelowmemprunes" in collapsed:
        return True
    if "maxtrxlockmemorybytes" in collapsed:
        return True
    if "innodbrowlocktimemax" in collapsed:
        return True
    return False


def _is_filesystem(low: str, collapsed: str) -> bool:
    if "filesystem" in collapsed:
        return True
    if any(t in collapsed for t in (
        "fscapacity", "fsavailablespace", "fsinodesusedpercent", "fsusedspace",
    )):
        return True
    if "filesizemb" in collapsed:
        return True
    return False


def _is_disk_io(low: str, collapsed: str) -> bool:
    if "localdisk" in collapsed:
        return True
    if any(t in collapsed for t in (
        "dskavgserv", "dskbps", "dskpercentbusy", "dskrtps", "dskread",
        "dskreadwrite", "dsktps", "dskwtps", "dskwrite",
    )):
        return True
    # Mysql InnoDB data read/write rates are I/O evidence.
    if "innodbdataread" in collapsed or "innodbdatawritten" in collapsed:
        return True
    if "innodbpagesread" in collapsed or "innodbpageswritten" in collapsed:
        return True
    return False


def _is_network_packet_loss(low: str, collapsed: str) -> bool:
    # NIC-level error counters are the only genuine packet-loss indicator in
    # AIOps2021.  Note these are also raised during network-latency faults
    # (loss and latency are not fully separable at the metric layer), but they
    # are the closest metric-level loss signal available.
    if any(t in collapsed for t in (
        "netinerr", "netouterr", "neterrprc", "netinerrprc", "netouterrprcc",
    )):
        return True
    return False


def _is_network_latency(low: str, collapsed: str) -> bool:
    # NIC throughput / TCP connection-state -> latency evidence.
    if any(t in collapsed for t in (
        "netpacketsin", "netpacketsout", "netkbtotalpersec", "netbandwidthutil",
    )):
        return True
    if any(t in collapsed for t in (
        "tcpclosewait", "tcpfinwait", "totaltcpconnnum",
    )):
        return True
    # Container network throughput.
    if any(t in collapsed for t in ("networkrxbytes", "networktxbytes")):
        return True
    # Tomcat request latency.
    if "maxtimerequestinfo" in collapsed or "processingtimerequestinfo" in collapsed:
        return True
    return False


def _is_db_latency(low: str, collapsed: str) -> bool:
    # Mysql response time -> latency evidence (not a DB-connection fault).
    if "getresponsetimeofmysqld" in collapsed:
        return True
    return False
