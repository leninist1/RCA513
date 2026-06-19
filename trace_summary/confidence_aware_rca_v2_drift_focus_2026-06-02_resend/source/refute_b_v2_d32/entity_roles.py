"""Dataset-portable entity role inference.

The current OpenRCA path still uses its established component-family helpers.
This module gives Eadro/AIOps2021 adapters a less brittle way to infer service,
host, database, cache, and gateway roles from topology and telemetry metadata
instead of component-name prefixes alone.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping


ROLE_HOST = "host"
ROLE_CONTAINER = "container"
ROLE_DATABASE = "database"
ROLE_CACHE = "cache"
ROLE_GATEWAY = "gateway"
ROLE_QUEUE = "queue"
ROLE_SERVICE = "service"
ROLE_UNKNOWN = "unknown"


@dataclass(frozen=True)
class EntityRole:
    entity_id: str
    role: str
    confidence: float
    signals: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "role": self.role,
            "confidence": self.confidence,
            "signals": list(self.signals),
        }


def infer_entity_role(
    entity_id: str,
    *,
    kpi_names: Iterable[str] = (),
    topology_node: Mapping[str, Any] | None = None,
    trace_role: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> EntityRole:
    """Infer a portable role from explicit metadata first, names last."""

    scores: dict[str, float] = {}
    signals: list[str] = []

    def add(role: str, score: float, signal: str) -> None:
        scores[role] = scores.get(role, 0.0) + float(score)
        signals.append(signal)

    explicit = _first_nonempty(
        (topology_node or {}).get("role"),
        (topology_node or {}).get("type"),
        (topology_node or {}).get("kind"),
        (metadata or {}).get("role"),
        (metadata or {}).get("type"),
        (metadata or {}).get("kind"),
    )
    if explicit:
        role = _normalize_role(str(explicit))
        if role != ROLE_UNKNOWN:
            return EntityRole(str(entity_id), role, 1.0, (f"explicit:{explicit}",))

    for kpi in kpi_names:
        low = str(kpi).lower()
        if any(token in low for token in ("system.", "node_", "host_", "machine", "load1", "iostat")):
            add(ROLE_HOST, 1.5, f"kpi:{kpi}")
        if any(token in low for token in ("container_", "pod_", "jvm", "heap", "gc", "cgroup")):
            add(ROLE_CONTAINER, 1.3, f"kpi:{kpi}")
        if any(token in low for token in ("mysql", "postgres", "oracle", "db_", "session", "sql", "tnsping")):
            add(ROLE_DATABASE, 1.4, f"kpi:{kpi}")
        if any(token in low for token in ("redis", "cache", "memcached")):
            add(ROLE_CACHE, 1.4, f"kpi:{kpi}")
        if any(token in low for token in ("nginx", "apache", "ingress", "gateway", "proxy")):
            add(ROLE_GATEWAY, 1.2, f"kpi:{kpi}")
        if any(token in low for token in ("kafka", "rabbitmq", "queue", "mq_")):
            add(ROLE_QUEUE, 1.2, f"kpi:{kpi}")

    if trace_role:
        low = str(trace_role).lower()
        if any(token in low for token in ("entry", "gateway", "frontdoor", "ingress")):
            add(ROLE_GATEWAY, 1.0, f"trace_role:{trace_role}")
        if any(token in low for token in ("database", "db", "storage")):
            add(ROLE_DATABASE, 1.0, f"trace_role:{trace_role}")

    # Name is intentionally the weakest signal.  It keeps legacy datasets usable
    # but prevents prefix quirks from dominating richer Eadro/AIOps metadata.
    low_id = str(entity_id).lower()
    if low_id.startswith(("os_", "node_", "host_")):
        add(ROLE_HOST, 0.6, "name_prefix")
    if low_id.startswith(("docker_", "pod_", "container_")):
        add(ROLE_CONTAINER, 0.6, "name_prefix")
    if low_id.startswith(("db_", "mysql", "postgres", "oracle")) or "database" in low_id:
        add(ROLE_DATABASE, 0.7, "name_prefix")
    if low_id.startswith(("redis", "cache")):
        add(ROLE_CACHE, 0.7, "name_prefix")
    if low_id.startswith(("ig", "apache", "nginx", "gateway", "proxy")):
        add(ROLE_GATEWAY, 0.5, "name_prefix")

    if not scores:
        return EntityRole(str(entity_id), ROLE_UNKNOWN, 0.0, tuple())
    role, score = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[0]
    confidence = min(1.0, score / 4.0)
    return EntityRole(str(entity_id), role, float(confidence), tuple(dict.fromkeys(signals)))


def infer_roles_for_metric_frame(metric_df, topology: Mapping[str, Any] | None = None) -> dict[str, EntityRole]:
    """Infer roles for all components in a normalized metric frame."""

    if metric_df is None or metric_df.empty or "cmdb_id" not in metric_df.columns:
        return {}
    topology = dict(topology or {})
    containers = dict(topology.get("containers", {}) or {})
    nodes = dict(topology.get("nodes", {}) or {})
    out: dict[str, EntityRole] = {}
    for component, group in metric_df.groupby("cmdb_id", sort=True):
        component_id = str(component)
        kpis = group["kpi_name"].dropna().astype(str).unique().tolist() if "kpi_name" in group else []
        topo = containers.get(component_id) or nodes.get(component_id) or {}
        out[component_id] = infer_entity_role(component_id, kpi_names=kpis, topology_node=topo)
    return out


def _normalize_role(value: str) -> str:
    low = value.lower().strip()
    if low in {"host", "node", "machine", "vm", "physical_node"}:
        return ROLE_HOST
    if low in {"container", "pod", "workload"}:
        return ROLE_CONTAINER
    if low in {"db", "database", "mysql", "postgres", "oracle"}:
        return ROLE_DATABASE
    if low in {"redis", "cache", "memcached"}:
        return ROLE_CACHE
    if low in {"gateway", "ingress", "proxy", "frontdoor"}:
        return ROLE_GATEWAY
    if low in {"queue", "mq", "kafka", "rabbitmq"}:
        return ROLE_QUEUE
    if low in {"service", "app", "application", "microservice"}:
        return ROLE_SERVICE
    return ROLE_UNKNOWN


def _first_nonempty(*values: Any) -> Any:
    for value in values:
        if value is not None and str(value).strip():
            return value
    return None
