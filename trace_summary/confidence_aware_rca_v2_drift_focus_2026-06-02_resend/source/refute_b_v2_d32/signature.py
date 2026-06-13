"""Case evidence signatures for d32 Layer 1 and Layer 2."""
from __future__ import annotations

from typing import Any, Mapping

import pandas as pd


CPU_TOKENS = ("CPU", "Cpu", "cpu", "CPULoad", "SingleCpu")
MEM_TOKENS = ("MEMORY", "Memory", "Mem", "Heap", "used_memory", "Qcache", "PROCPPMem")
DISK_TOKENS = ("DSK", "Disk", "Read", "Write", "blkio")
FS_TOKENS = ("FILESYSTEM", "FSAvailable", "FSCapacity", "FSInode")
NET_TOKENS = ("Network", "network", "traffic", "TCP", "Packet", "packet", "rejected", "Aborted", "NET")


def service_type(service: str) -> str:
    text = str(service).lower()
    if "tomcat" in text:
        return "app"
    if "mysql" in text:
        return "database"
    if "redis" in text:
        return "cache"
    if "apache" in text:
        return "gateway"
    if text.startswith("ig"):
        return "ingress"
    if text.startswith("mg"):
        return "middleware"
    return "unknown"


def service_role(service: str) -> str:
    text = str(service).lower()
    if "mysql" in text or "redis" in text:
        return "stateful_backend"
    if "tomcat" in text:
        return "application"
    if "apache" in text or text.startswith("ig"):
        return "frontdoor"
    if text.startswith("mg"):
        return "message_orchestration"
    return "service"


def reason_for_kpi(kpi_name: str) -> str | None:
    text = str(kpi_name)
    low = text.lower()
    # Market KPI detection: lowercase dot-notation (system.* or container_*)
    if text.startswith("system.") or text.startswith("container_"):
        return _reason_for_kpi_market(text, low)
    if any(token in text for token in CPU_TOKENS):
        return "high JVM CPU load" if "JVM" in text or "Tomcat" in text else "high CPU usage"
    if any(token in text for token in MEM_TOKENS):
        if any(token in text for token in ("JVM", "Heap", "Tomcat-MEMORY")):
            return "JVM Out of Memory (OOM) Heap"
        return "high memory usage"
    if any(token in text for token in FS_TOKENS):
        return "high disk space usage"
    if any(token in text for token in DISK_TOKENS):
        return "high disk I/O read usage"
    if any(token in text for token in NET_TOKENS):
        packet_tokens = ("Packet", "Err", "rejected", "Aborted", "TCP")
        return "network packet loss" if any(token in text for token in packet_tokens) else "network latency"
    return None


def _reason_for_kpi_market(text: str, low: str) -> str | None:
    """Market-dataset KPI → reason mapping using Market vocabulary."""
    is_node = text.startswith("system.")
    prefix = "node" if is_node else "container"
    if is_node:
        if "cpu" in low:
            return "node CPU load"
        if "mem" in low:
            return "node memory consumption"
        if "disk" in low:
            if "read" in low:
                return "node disk read I/O consumption"
            if "write" in low:
                return "node disk write I/O consumption"
            return "node disk space consumption"
        if any(t in low for t in ("net", "tcp", "udp")):
            if any(t in low for t in ("drop", "loss", "error", "retrans")):
                return "container network packet loss"
            return "container network latency"
    else:
        if "cpu" in low:
            return "container CPU load"
        if any(t in low for t in ("mem", "memory")):
            return "container memory load"
        if any(t in low for t in ("fs", "filesystem")):
            return "node disk space consumption"
        if any(t in low for t in ("disk_", "disk.")):
            if "read" in low:
                return "container read I/O load"
            if "write" in low:
                return "container write I/O load"
            return "container disk I/O load"
        if any(t in low for t in ("net", "network")):
            if any(t in low for t in ("drop", "loss", "error", "corrupt", "retrans")):
                return "container network packet corruption"
            return "container network latency"
        if "process" in low and any(t in low for t in ("term", "restart", "kill")):
            return "container process termination"
    return None


def reason_for_log(value: str) -> str | None:
    text = str(value)
    low = text.lower()
    if any(token in text for token in ("OutOfMemoryError", "Full GC", "SIGKILL")) or "oom" in low:
        return "JVM Out of Memory (OOM) Heap"
    if any(token in low for token in ("reset by peer", "broken pipe", "connection reset", "retry")):
        return "network packet loss"
    if any(token in low for token in ("timeout", "timed out", "connection refused")):
        return "network latency"
    if "no space" in low or "disk" in low:
        return "high disk space usage"
    return None


def _bucket(reason: str) -> str:
    from refute_b_v2_d32.schema import reason_bucket

    return reason_bucket(reason)


def build_case_signature(
    case_id: str,
    metric_df: pd.DataFrame,
    log_df: pd.DataFrame,
    trace_summary: Mapping[str, Any] | None,
    baseline,
    modal_status: Mapping[str, str],
    max_metric_events: int = 300,
) -> dict[str, Any]:
    services: dict[str, dict[str, Any]] = {}
    dominant = set()

    def service_row(service: str) -> dict[str, Any]:
        return services.setdefault(str(service), {"service": str(service), "metric": {}, "log": {}, "trace": {}, "topology": {}})

    metric_events = 0
    if metric_df is not None and not metric_df.empty:
        for row in metric_df.itertuples(index=False):
            reason = reason_for_kpi(getattr(row, "kpi_name", ""))
            if not reason:
                continue
            result = baseline.is_anomalous(str(row.cmdb_id), str(row.kpi_name), row.value, threshold="p99")
            if not result.is_anomalous:
                continue
            metric_events += 1
            bucket = _bucket(reason)
            dominant.add(f"metric:{bucket}")
            svc = service_row(str(row.cmdb_id))
            item = svc["metric"].setdefault(bucket, {"state": "support", "intensity": 0, "strength": 0.0, "examples": []})
            strength = abs(float(result.deviation or 0.0))
            item["strength"] += strength
            item["intensity"] = min(3, int(item["strength"] // 3) + 1)
            if len(item["examples"]) < 3:
                item["examples"].append({"kpi": str(row.kpi_name), "value": float(row.value), "deviation": float(result.deviation)})
            if metric_events >= max_metric_events:
                break

    if log_df is not None and not log_df.empty and "value" in log_df.columns:
        for row in log_df.itertuples(index=False):
            reason = reason_for_log(getattr(row, "value", ""))
            if not reason:
                continue
            bucket = _bucket(reason)
            dominant.add(f"log:{bucket}")
            svc = service_row(str(getattr(row, "cmdb_id")))
            item = svc["log"].setdefault(bucket, {"state": "support", "intensity": 2, "strength": 5.0, "examples": []})
            item["strength"] += 5.0
            item["intensity"] = min(3, item["intensity"] + 1)
            if len(item["examples"]) < 3:
                item["examples"].append(str(getattr(row, "value", ""))[:160])

    summary = trace_summary or {}
    if summary.get("trace_status") == "present":
        events = summary.get("events", {})
        for edge in events.get("slow_edges", []):
            dominant.add("trace:network_latency")
            for service in {str(edge.get("src")), str(edge.get("dst"))}:
                if service and service != "None":
                    svc = service_row(service)
                    item = svc["trace"].setdefault("network_latency", {"state": "support", "intensity": 0, "strength": 0.0, "examples": []})
                    strength = float(edge.get("slow_ratio", 0.0) or 0.0)
                    item["strength"] += strength
                    item["intensity"] = min(3, int(item["strength"]) + 1)
                    if len(item["examples"]) < 3:
                        item["examples"].append(dict(edge))
        for edge in events.get("dropped_edges", []):
            dominant.add("trace:network_packet_loss")
            for service in {str(edge.get("src")), str(edge.get("dst"))}:
                if service and service != "None":
                    svc = service_row(service)
                    item = svc["trace"].setdefault("network_packet_loss", {"state": "support", "intensity": 0, "strength": 0.0, "examples": []})
                    strength = float(edge.get("count_drop_ratio", 0.0) or 0.0) * 10.0
                    item["strength"] += strength
                    item["intensity"] = min(3, int(item["strength"]) + 1)
                    if len(item["examples"]) < 3:
                        item["examples"].append(dict(edge))
        first = events.get("first_anomalous_service")
        if first:
            service_row(str(first))["topology"]["first_anomalous_service"] = {"state": "support", "intensity": 2, "strength": 2.0}
            dominant.add("topology:first_anomalous_service")

    blind_spots = [f"{name}:{status}" for name, status in sorted(dict(modal_status).items()) if status not in {"present", "disabled"}]
    return {
        "case_id": str(case_id),
        "services": list(sorted(services.values(), key=lambda item: item["service"])),
        "dominant_evidence_types": sorted(dominant),
        "blind_spots": blind_spots,
    }


def flatten_signature(signature: Mapping[str, Any]) -> dict[str, float]:
    features: dict[str, float] = {}
    for service in signature.get("services", []):
        svc = str(service.get("service", ""))
        stype = service_type(svc)
        role = service_role(svc)
        service_modalities = 0
        for modality in ("metric", "log", "trace", "topology"):
            for name, item in service.get(modality, {}).items():
                state = str(item.get("state", "none"))
                intensity = float(item.get("intensity", 0.0) or 0.0)
                strength = float(item.get("strength", 0.0) or 0.0)
                weight = max(1.0, intensity, min(abs(strength), 20.0) / 5.0)
                shape = _shape_bucket(strength, intensity)
                features[f"{modality}:{name}:{state}"] = features.get(f"{modality}:{name}:{state}", 0.0) + weight
                features[f"type:{stype}:{modality}:{name}:{state}"] = features.get(f"type:{stype}:{modality}:{name}:{state}", 0.0) + weight
                features[f"role:{role}:{modality}:{name}:{state}"] = features.get(f"role:{role}:{modality}:{name}:{state}", 0.0) + weight
                features[f"shape:{modality}:{name}:{shape}"] = features.get(f"shape:{modality}:{name}:{shape}", 0.0) + 1.0
                service_modalities += 1
        if service_modalities:
            features[f"type:{stype}:active"] = features.get(f"type:{stype}:active", 0.0) + 1.0
            features[f"role:{role}:active"] = features.get(f"role:{role}:active", 0.0) + 1.0
    for item in signature.get("dominant_evidence_types", []):
        features[f"dominant:{item}"] = 1.0
    for blind in signature.get("blind_spots", []):
        features[f"blind:{blind}"] = 1.0
    return features


def _shape_bucket(strength: float, intensity: float) -> str:
    if intensity >= 3 or abs(strength) >= 12:
        return "strong"
    if intensity >= 2 or abs(strength) >= 5:
        return "medium"
    return "weak"


def cosine_sparse(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    if not left or not right:
        return 0.0
    common = set(left) & set(right)
    dot = sum(float(left[k]) * float(right[k]) for k in common)
    nl = sum(float(v) * float(v) for v in left.values()) ** 0.5
    nr = sum(float(v) * float(v) for v in right.values()) ** 0.5
    return dot / (nl * nr) if nl and nr else 0.0
