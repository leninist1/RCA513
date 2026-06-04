"""Dataset adapters and topology builders for Scheme B v2.

The adapter layer is the boundary between dataset-specific schemas and the
abstract evidence layer. Bank, Market, and later Telecom should differ here,
not inside the rule engine.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Iterable, Mapping


NODE_KPI_PREFIXES = ("OSLinux-OSLinux_",)

KPI_BUCKET_PATTERNS = {
    "cpu": ("CPU", "Cpu", "CPULoad", "CpuUtil"),
    "memory": ("MEMORY", "Memory", "Mem", "Heap", "used_memory", "Qcache"),
    "disk_io": ("DSK", "Disk", "blkio", "Read", "Write"),
    "filesystem": ("FILESYSTEM", "FSAvailable", "FSCapacity", "FSInode"),
    "network": ("Network", "NET", "TCP", "Packet", "rejected", "Aborted"),
}


@dataclass(frozen=True)
class MetricIdentity:
    cmdb_id: str
    kpi_name: str
    level: str
    node_id: str
    service: str
    abstract_bucket: str

    def to_dict(self) -> dict:
        return {
            "cmdb_id": self.cmdb_id,
            "kpi_name": self.kpi_name,
            "level": self.level,
            "node_id": self.node_id,
            "service": self.service,
            "abstract_bucket": self.abstract_bucket,
        }


@dataclass
class DatasetAdapter:
    dataset: str = "generic"

    def classify_metric(self, cmdb_id: str, kpi_name: str) -> MetricIdentity:
        service = self.normalize_service(cmdb_id)
        level = self.metric_level(kpi_name)
        return MetricIdentity(
            cmdb_id=str(cmdb_id),
            kpi_name=str(kpi_name),
            level=level,
            node_id=self.node_id_for(cmdb_id, kpi_name, level),
            service=service,
            abstract_bucket=self.abstract_bucket(kpi_name),
        )

    def normalize_service(self, cmdb_id: str) -> str:
        return str(cmdb_id)

    def metric_level(self, kpi_name: str) -> str:
        return "node" if str(kpi_name).startswith(NODE_KPI_PREFIXES) else "container"

    def node_id_for(self, cmdb_id: str, kpi_name: str, level: str) -> str:
        return f"node::{cmdb_id}"

    def abstract_bucket(self, kpi_name: str) -> str:
        text = str(kpi_name)
        for bucket, patterns in KPI_BUCKET_PATTERNS.items():
            if any(pattern in text for pattern in patterns):
                return bucket
        return "other"


@dataclass
class BankAdapter(DatasetAdapter):
    dataset: str = "Bank"


@dataclass
class MarketAdapter(DatasetAdapter):
    dataset: str = "Market"

    def normalize_service(self, cmdb_id: str) -> str:
        text = str(cmdb_id)
        match = re.match(r"(?P<node>node-[^.]+)[.](?P<service>.+)", text)
        return match.group("service") if match else text

    def node_id_for(self, cmdb_id: str, kpi_name: str, level: str) -> str:
        text = str(cmdb_id)
        match = re.match(r"(?P<node>node-[^.]+)[.](?P<service>.+)", text)
        return match.group("node") if match else f"node::{cmdb_id}"


def adapter_for(dataset: str) -> DatasetAdapter:
    key = str(dataset).lower()
    if key == "bank":
        return BankAdapter()
    if key == "market":
        return MarketAdapter()
    return DatasetAdapter(dataset=str(dataset))


def build_topology(metric_pairs: Iterable[tuple[str, str]], adapter: DatasetAdapter | None = None) -> dict:
    adapter = adapter or BankAdapter()
    containers: dict[str, dict] = {}
    nodes: dict[str, dict] = {}
    for cmdb_id, kpi_name in metric_pairs:
        ident = adapter.classify_metric(cmdb_id, kpi_name)
        container = containers.setdefault(ident.service, {
            "node_proxy": ident.node_id,
            "container_kpis": [],
            "node_kpis": [],
            "abstract_buckets": {},
        })
        container["node_proxy"] = ident.node_id
        target = "node_kpis" if ident.level == "node" else "container_kpis"
        container[target].append(ident.kpi_name)
        container["abstract_buckets"].setdefault(ident.abstract_bucket, 0)
        container["abstract_buckets"][ident.abstract_bucket] += 1
        node = nodes.setdefault(ident.node_id, {"hosted_containers": set(), "node_kpis": []})
        node["hosted_containers"].add(ident.service)
        if ident.level == "node":
            node["node_kpis"].append(ident.kpi_name)
    return {
        "dataset": adapter.dataset,
        "containers": {
            svc: {
                **row,
                "container_kpis": sorted(set(row["container_kpis"])),
                "node_kpis": sorted(set(row["node_kpis"])),
            }
            for svc, row in sorted(containers.items())
        },
        "nodes": {
            node: {
                "hosted_containers": sorted(row["hosted_containers"]),
                "node_kpis": sorted(set(row["node_kpis"])),
            }
            for node, row in sorted(nodes.items())
        },
    }


def metric_pairs_from_frame(metric_df) -> list[tuple[str, str]]:
    if metric_df is None or metric_df.empty:
        return []
    return [
        (str(row.cmdb_id), str(row.kpi_name))
        for row in metric_df[["cmdb_id", "kpi_name"]].drop_duplicates().itertuples(index=False)
    ]
