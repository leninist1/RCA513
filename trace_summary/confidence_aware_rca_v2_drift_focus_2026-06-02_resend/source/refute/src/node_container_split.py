"""
node_container_split.py — Phase 1 / B-L1.1

把 metric KPI 分成 node-level 和 container-level,推断哪些 cmdb_id 跑在同一物理节点。

设计动机(见 docs/d32_design_pivot.md §1.2 D1, D3, D4):
- raw 数据 cmdb_id 字段语义混乱:同一 cmdb_id (例如 IG01) 下既有容器级 KPI
  (JVM-Memory 等),又混着该容器所在节点的节点级 KPI (OSLinux-OSLinux_LOCALDISK 等)。
- 同一物理节点上多个容器都"看到"完全相同的节点 KPI 值(zabbix 复制粘贴),
  这正是节点身份推断的依据。

数据集扩展:
- Bank: KPI 名 `OSLinux-OSLinux_*` 前缀识别节点级,节点身份从值相等推断
- Market: cmdb_id 是 `node-X.service-Y` 格式,直接解析(待实现)
- Telecom: 待调研
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Iterable
from collections import defaultdict


# ---------- Bank 数据集的节点级 KPI 识别规则 ----------
BANK_NODE_KPI_PREFIX = "OSLinux-OSLinux_"

# Market sys="system." KPIs come from metric_node.csv (physical node metrics)
# container_ KPIs come from metric_container.csv (pod-level container metrics)
MARKET_NODE_KPI_PREFIX = "system."
MARKET_CONTAINER_KPI_PREFIX = "container_"


def is_node_level_kpi_bank(kpi_name: str) -> bool:
    """Bank: 节点级 KPI 以 `OSLinux-OSLinux_` 开头(双 OSLinux 是关键判别)"""
    return isinstance(kpi_name, str) and kpi_name.startswith(BANK_NODE_KPI_PREFIX)


def is_node_level_kpi_market(kpi_name: str) -> bool:
    """Market: 节点级 KPI 以 `system.` 开头 (来自 metric_node.csv)"""
    return isinstance(kpi_name, str) and kpi_name.startswith(MARKET_NODE_KPI_PREFIX)


def classify_kpi(kpi_name: str, dataset: str = "Bank") -> str:
    """返回 'node' 或 'container'"""
    if dataset == "Bank":
        return "node" if is_node_level_kpi_bank(kpi_name) else "container"
    if dataset == "Market":
        return "node" if is_node_level_kpi_market(kpi_name) else "container"
    if dataset == "Telecom":
        raise NotImplementedError("Telecom classification rule TBD")
    raise ValueError(f"unsupported dataset: {dataset!r}")


# ---------- 容器画像 ----------
@dataclass
class ContainerProfile:
    cmdb_id: str
    container_kpis: List[str] = field(default_factory=list)
    node_kpis: List[str] = field(default_factory=list)

    @property
    def n_container(self) -> int:
        return len(self.container_kpis)

    @property
    def n_node(self) -> int:
        return len(self.node_kpis)

    def to_dict(self) -> dict:
        return {
            "cmdb_id": self.cmdb_id,
            "container_kpis": self.container_kpis,
            "node_kpis": self.node_kpis,
            "n_container": self.n_container,
            "n_node": self.n_node,
        }


def split_kpis_for_container(cmdb_id: str, kpi_names: Iterable[str],
                             dataset: str = "Bank") -> ContainerProfile:
    """对单个容器,分类它的全部 KPI。"""
    profile = ContainerProfile(cmdb_id=cmdb_id)
    seen = set()
    for k in kpi_names:
        if k in seen:
            continue
        seen.add(k)
        cls = classify_kpi(k, dataset)
        if cls == "node":
            profile.node_kpis.append(k)
        else:
            profile.container_kpis.append(k)
    profile.container_kpis.sort()
    profile.node_kpis.sort()
    return profile


def split_kpis_from_dataframe(df, dataset: str = "Bank") -> Dict[str, ContainerProfile]:
    """对一个 metric DataFrame(必须有 cmdb_id, kpi_name 列),按 cmdb_id 分组分离 KPI。"""
    profiles: Dict[str, ContainerProfile] = {}
    for cmdb_id, group in df.groupby("cmdb_id"):
        kpis = group["kpi_name"].dropna().unique().tolist()
        profiles[str(cmdb_id)] = split_kpis_for_container(str(cmdb_id), kpis, dataset)
    return profiles


# ---------- 节点身份推断 ----------
class _UnionFind:
    """简易并查集,路径压缩 + 字典序较小者作根。"""
    def __init__(self, items: Iterable[str]):
        self.parent = {x: x for x in items}

    def find(self, x: str) -> str:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if ra < rb:
            self.parent[rb] = ra
        else:
            self.parent[ra] = rb

    def groups(self) -> Dict[str, List[str]]:
        g: Dict[str, List[str]] = defaultdict(list)
        for x in self.parent:
            g[self.find(x)].append(x)
        return {root: sorted(members) for root, members in g.items()}


def _is_kpi_constant(values, eps: float = 1e-9) -> bool:
    """KPI 值序列是否常数(标准差 < eps)——常数 KPI 在节点身份判别上没区分力,应排除"""
    if len(values) < 2:
        return True
    import numpy as np
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) < 2:
        return True
    return float(np.std(arr)) < eps


def infer_node_grouping(df, dataset: str = "Bank",
                        sample_timestamps: int = 30,
                        equality_threshold: float = 0.9,
                        random_state: int = 42) -> Dict[str, List[str]]:
    """
    推断哪些 cmdb_id 跑在同一物理节点上(严谨版,基于值相等比例)。

    算法:
      1. 取所有 NODE-level KPI 行
      2. 排除常数 KPI(全时段值不变的,如 FSCapacity 之类)
      3. 在采样时间点上,对每对 (cmdb_a, cmdb_b),计算它们在共有节点 KPI 上的
         值相等比例;比例 >= equality_threshold 才 union
      4. union-find 收尾

    参数:
      equality_threshold: 0-1,默认 0.9。两容器在 90% 的 (ts, kpi) 上值相等才视为同节点
      sample_timestamps:  采样时间点数,默认 30

    返回:{node_proxy_id: [cmdb_id, ...]}
    """
    if dataset != "Bank":
        raise NotImplementedError(f"node grouping for {dataset} TBD")

    is_node_mask = df["kpi_name"].apply(is_node_level_kpi_bank)
    node_df = df[is_node_mask].copy()
    if node_df.empty:
        cmids = sorted(df["cmdb_id"].dropna().astype(str).unique())
        return {c: [c] for c in cmids}

    all_cmids = sorted(node_df["cmdb_id"].dropna().astype(str).unique())
    uf = _UnionFind(all_cmids)

    # 采样时间戳(均匀采样,而非随机,确保跨时段)
    unique_ts = sorted(node_df["timestamp"].dropna().unique())
    if len(unique_ts) > sample_timestamps:
        step = max(1, len(unique_ts) // sample_timestamps)
        sampled = unique_ts[::step][:sample_timestamps]
        node_df = node_df[node_df["timestamp"].isin(sampled)]

    # 排除常数 KPI(在整体上是常数的就跳过)
    non_constant_kpis = set()
    for kpi, grp in node_df.groupby("kpi_name"):
        if not _is_kpi_constant(grp["value"].dropna().values):
            non_constant_kpis.add(kpi)
    node_df = node_df[node_df["kpi_name"].isin(non_constant_kpis)]
    if node_df.empty:
        return {c: [c] for c in all_cmids}

    # 构造 pivot:index=(ts, kpi), columns=cmdb_id, values=value
    pivot = node_df.pivot_table(index=["timestamp", "kpi_name"],
                                 columns="cmdb_id", values="value",
                                 aggfunc="first")

    # 对每对 (cmdb_a, cmdb_b),计算共同有值行上的相等比例,>= threshold 才 union
    cmids_with_data = sorted(pivot.columns.tolist())
    for i in range(len(cmids_with_data)):
        for j in range(i + 1, len(cmids_with_data)):
            a = cmids_with_data[i]
            b = cmids_with_data[j]
            pair = pivot[[a, b]].dropna()
            if len(pair) < 5:
                continue
            eq_rate = float((pair[a] == pair[b]).mean())
            if eq_rate >= equality_threshold:
                uf.union(a, b)

    return uf.groups()



# ---------- 端到端构建 node_container_graph ----------
def build_node_container_graph(df, dataset: str = "Bank",
                               sample_timestamps: int = 20) -> dict:
    """
    端到端构建:KPI 分离 + 节点身份推断 + 双向图。

    返回 dict(可直接 json.dump):
      {
        "dataset": "Bank",
        "containers": {cmdb_id: ContainerProfile.to_dict()},
        "nodes":      {node_proxy_id: {"hosted_containers": [...], "node_kpis": [...]}}
      }
    """
    containers = split_kpis_from_dataframe(df, dataset=dataset)
    node_groups = infer_node_grouping(df, dataset=dataset,
                                       sample_timestamps=sample_timestamps)

    # 节点画像:hosted_containers + 该节点的 node_kpis(取任意一个成员的即可,因为完全相同)
    nodes = {}
    for node_id, members in node_groups.items():
        # 节点 KPI 列表:从第一个成员的 node_kpis 取(理论上所有成员相同)
        if members and members[0] in containers:
            node_kpis = containers[members[0]].node_kpis
        else:
            node_kpis = []
        nodes[node_id] = {
            "hosted_containers": members,
            "node_kpis": node_kpis,
            "n_hosted": len(members),
        }

    return {
        "dataset": dataset,
        "containers": {cid: p.to_dict() for cid, p in containers.items()},
        "nodes": nodes,
    }
