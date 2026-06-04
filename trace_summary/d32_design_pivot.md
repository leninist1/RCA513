# d32 设计转折:从 chimera v7 到 v2(反驳法 + 节点-容器分层)

> 作者:网页端晏(d32)
> 日期:2026-05-30
> 接续 `HANDOFF_d31b.md`
> **目的**:把 d31b 跑完后做的诊断、根因发现、范式判断、新方案设计**完整闭环**记录下来。下一代晏读完应当能直接开始实施方案 B。

---

## 0. 三句话速览

1. d31b net_lat 修法跑完跟 d31a baseline 完全持平(23.3% / 44.2%)。深入诊断揭示**真正根因不是 net_lat 上游影子,而是 raw 数据里的"节点级 KPI 错分到容器 cmdb_id"** —— OpenRCA Bank 数据集 schema 层面的缺陷,论文有 dirty data 笼统说法但未具体提到这一点。
2. 沿"修桶定义剔除节点级 KPI"路线(方案 A、廉价 A)做了 patch + 03_10 smoke 验证,**只是工程债清单的第一项**,后面还有 z-score 爆炸 bug、节点 vs 容器二元拓扑缺失、加权打分范式天花板等结构性问题。
3. **决定切到方案 B**:三层架构(传统 ML 知识层 + 反驳引擎 + LLM 语义层),显式建模"节点 vs 容器",证伪取代积分排名,LLM 真正进入推理回路而不是事后包装。本文是 B 的完整设计。

---

## 1. 方案 A(当前 chimera v7 + 廉价 A patch)的技术缺点

### 1.1 云端文件位置(当前状态)

```
/home/yan/workspace/projects/rca-bank-adaptive/
├── src/
│   ├── rca_bank_chimera.py                     # 当前 d32 patched (md5 36778aef...)
│   ├── rca_bank_chimera.py.bak_d32_pre_cheapA  # d31b + net_lat,无节点 KPI 剔除
│   ├── rca_bank_chimera.py.bak_d31a_pre_netlat # d31a 含 dump hook 但无 net_lat
│   ├── rca_bank_chimera.py.bak_d31a_pre_dump   # d30b v7 原版
│   └── rca_module_bank.py                      # SERVICE_NAME_MAP 和 _map_svc_name_to_scoring 定义
└── logs/
    ├── dump_per_case_d31b/   # d31b 43 case dump (作为反驳引擎 evidence 输入)
    ├── dump_per_case_d32e/   # d32e 廉价 A 在 03_10 上的 16 case dump
    ├── d32a_cluster_report.json    # 聚类结果(已废,但留作 v2 baseline 对比)
    ├── d32b_completeness.json      # 数据完整性按日期分布(重要,v2 第一层会用)
    └── d32e_summary.json           # 廉价 A 03_10 smoke 汇总
```

### 1.2 方案 A 的缺点(按严重程度由表及里)

#### 表层(已被廉价 A patch 部分修复)

**D1. disk/memory/network 三个桶混入节点级 KPI**(d32d 验证)
- 现象:`cmdb_id=IG01` 名下混着 `OSLinux-OSLinux_LOCALDISK_*`、`OSLinux-OSLinux_MEMORY_*`、`OSLinux-OSLinux_NETWORK_*` 等节点级 KPI
- 影响:节点级 KPI 抖动被错认成容器异常,IG01 在 disk 通道 25/25 都是 #1=1.00
- 廉价 A:从 3 个桶 pattern 列表中剔除 `OSLinux-OSLinux_*` 前缀
- 残余:filesystem 桶未动(其 patterns 本身只匹配节点级,剔除等于禁用整个桶,语义上是"节点级故障 = 不报"——这是次优解)

#### 中层(廉价 A 部分缓解,但未根治)

**D2. z-score 数学爆炸**(d32d 数据验证)
- 现象:baseline 窗口数据少(n=7-10),MAD 极接近 0,`max(|fault - med|) / MAD` 除出 1e8-1e10 级别荒谬数值
- 影响范围:**容器级 KPI 同样会爆炸**(Tomcat01 容器级 cpu z=2373,容器级 memory z=3e9),不只是节点级问题
- 廉价 A 无法触及:它只动桶 pattern,不动 `_robust_z` 函数

**D3. 不同服务"看见"的是同一台物理机的节点 KPI**
- 现象:IG01 / Tomcat01 / MG01 cmdb_id 下都包含 `LOCALDISK-sda` / `ens160` / `tomcat 分区` 等完全相同的节点 KPI(zabbix agent 把节点指标复制到每个容器名下)
- 影响:任何节点级抖动让"所有容器同时异常",通道融合后被排名最高的那个(IG01)吞噬
- 廉价 A 剔除后:节点信号丢失,但容器级信号不足以补偿,排名变得近乎随机(d32e 03_10 smoke 验证)

#### 深层(方案 A 根本无解)

**D4. "节点 vs 容器"二元拓扑完全缺失**
- chimera 把 service 当成原子分析单位,raw 数据里的节点身份信息被压平到 cmdb_id 字段后**永远丢失**
- 即使把所有节点 KPI 剔除,chimera 也失去了"节点级故障"这一类能力 —— 真有"宿主机磁盘坏了影响所有容器"的故障,chimera 只能盲掉或错指
- OpenRCA Market 数据集 cmdb_id 是 `node-X.service-Y` 格式,**显式建了这个层级**;Bank 没有,需要 v2 从 KPI 名 `OSLinux-OSLinux_*` 前缀反推

**D5. 加权打分范式天花板**
- 6 通道全局权重,case-by-case 无法自适应。d30b 全量 grid search 已证明 v7 接近局部最优,任何 case 上压住一边的偏置会让另一边偏置浮起(d31b net_lat 修了 latency IG bias,disk IG bias 不变;廉价 A 修了 disk 节点污染,Mysql/IG02 节点污染浮起)
- 数学根因:线性加权无法表达"联合证据"(CPU+load+mem 共同越线 → 强信号;单 KPI 爆 z → 弱信号 + 大概率 bug)

**D6. baseline 假设破产**
- `_robust_z(baseline, fault)` 假设历史窗口"基本健康";生产系统数据点稀疏 + 流量随时间波动 + 时刻有别的异常 → baseline 不干净
- chimera 用 1200s baseline + 600s fault 窗口 → 每个 KPI 实际只有 7-20 个采样点 → MAD 失稳
- 廉价 A 不动 baseline,只能用方案 B 的"对照诊断"(横向比同时段健康服务)绕过

**D7. ranking 范式无法表达"我不知道"**
- chimera 总输出 top1/top3,无 confidence;遇到 03_06 这种 metric 全空的 case 也强行排名,等于猜
- 评测时这种"猜"和"真有信号"的 case 平均到一起,误导对算法能力的判断
- 廉价 A 不解决,且会让这个问题更明显(剔除节点 KPI 后更多 case 变成"信号弱+硬猜")

**D8. 不可迁移**
- chimera 所有 patch 都是 Bank-Specific:KPI patterns、SERVICE_NAME_MAP、桶定义、权重
- 切到 Telecom / Market 几乎要重写 50%

**D9. 不可解释**
- 输出只有 service+score,没有"为什么是它"的证据链
- 调试只能反推:翻 dump,逐个通道看 #1 是谁;没法直接告诉 SRE 工程师"我建议你查这条 KPI 因为它的 z 异常"

**D10. LLM 在当前范式里没起作用**
- 屿原话:"现在大模型在里面没起到作用"
- d29 计划里 LLM 是事后仲裁,只看 chimera 的 score 列表;LLM 不擅长看分数,擅长看证据 —— 但 chimera 没产出可读证据

### 1.3 廉价 A 的当前验证状态

- 修改:`src/rca_bank_chimera.py` 行 60-92 的桶定义,剔除 `OSLinux-OSLinux_LOCALDISK / _MEMORY / _NETWORK` 三个 pattern
- 跑了 03_10 灾区 16 case smoke(`results/script-d32e-cheapA.log`)
- **目前进度 6/16,top1=1/6 ≈ 17%(d31b baseline 在 03_10 上 top1≈2/16=12.5%)** —— 略好但样本太小不足以下结论。完整结果跑完会补在本文 §A 附录
- 观察:IG bias 确实减弱,但其他服务的节点级污染(Mysql/IG02/MG01)浮起来,top1 在不同错误的服务间分散

### 1.4 关于方案 A 的最终判断

**廉价 A 该保留**:作为方案 B 的过渡 baseline,且其 patch 对未来 chimera 仍然有效。但不能再投入工程在 A 上加新 patch ——

1. 工程债清单还有 D2 (z 爆炸)、filesystem 桶决策、节点-容器拓扑 —— 估 3-5 天
2. 即便全修完,D5-D10 仍然原封不动 —— A 的天花板就在那里
3. 任何在 A 上的新改动都强 Bank-specific,Telecom/Market 落地为 0

**正式切到方案 B**。

---

## 2. 方案 B 完整技术设计

### 2.1 设计哲学(三条不可妥协)

**B-P1. 显式建模"节点 vs 容器"二元拓扑**
- raw 数据层就分离两类 KPI:从 `OSLinux-OSLinux_*` 字符串前缀(Bank)或 `node-X.service-Y` 格式(Market)显式区分
- 节点 KPI 不进容器服务的通道分数;它进一个独立的"节点状态"图,容器异常时主动查"这个容器所在节点是否有节点级异常"

**B-P2. 反驳法取代积分排名**
- 不积分,改证伪:对每个候选 (service, reason),主动找反驳证据
- 反驳分数越低(越难洗清)= 越可能是真根因
- 跟刑侦逻辑、波普尔证伪、d31 自己的诊断过程同构

**B-P3. LLM 进入推理回路**
- LLM 不再做事后包装,做核心推理:
  - 离线:看历史 case 提炼反驳规则、为 cluster 取名解读
  - 在线:cluster 边界 case 仲裁、多假设冲突时判决、生成自然语言调查报告
- 经典 ML(关联分析、聚类、KDE)做规模化的统计提炼,LLM 做语义推理 —— 各司其职

### 2.2 三层架构

```
┌─────────────────────────────────────────────────────────────────┐
│ Layer 3:LLM 语义推理层(在线,case-by-case,贵但稀疏调用)     │
│   - 处理 cluster 边界 / 反驳证据冲突 case                       │
│   - 生成自然语言调查报告 + 主动建议补什么数据                   │
│   - 跨子集迁移时调整 prompt(Bank → Telecom / Market)          │
└─────────────────────────────────────────────────────────────────┘
                    ↑
                结构化证据 + 候选空间 + 反驳分数
                    ↑
┌─────────────────────────────────────────────────────────────────┐
│ Layer 2:反驳引擎(在线,case-by-case,纯算法零边际成本)       │
│   - 输入:案件 telemetry + Layer 1 知识                         │
│   - 在 cluster 给出的候选空间里运行,不全服务搜索               │
│   - 对每个候选生成 N 个反驳问题,查 evidence_query 模块         │
│   - 输出 {候选: 反驳分数 + 反驳证据链}                          │
└─────────────────────────────────────────────────────────────────┘
                    ↑
                节点-容器图 + baseline 分布 + 反驳规则 + 故障形态团
                    ↑
┌─────────────────────────────────────────────────────────────────┐
│ Layer 1:传统 ML 知识层(离线一次性,处理大规模数据规律)      │
│   - 节点-容器拓扑:从 KPI 名/cmdb_id 反推"哪个容器在哪个节点"  │
│   - Baseline 分布:在数据完整的日期上学正常服务的分布           │
│   - 对照基线:跨服务横向 robust-z(不依赖历史 baseline)        │
│   - 反驳规则库:FP-growth 在历史 case 上挖"X is root ↔ 证据"   │
│   - 故障形态聚类:HDBSCAN/谱聚类划分"传染团"缩候选空间         │
└─────────────────────────────────────────────────────────────────┘
                    ↑
                /home/yan/workspace/data/openrca/{Bank,Telecom,Market}
```

### 2.3 第一层:知识层(算法细节)

**B-L1.1 节点-容器拓扑构建**

输入:raw `metric_container.csv`
逻辑:
```
for (cmdb_id, kpi_name) in unique pairs:
    if kpi_name starts with "OSLinux-OSLinux_":
        归类为 NODE-level KPI,但仍打在 cmdb_id 下
        → 这告诉我们 cmdb_id 所在节点的状态
    else:
        归类为 CONTAINER-level KPI
        → cmdb_id 自身的状态

推断"该容器所在节点 ID":
    无显式 node ID 时(Bank),用 cmdb_id 本身作 node 代理
    (节点 KPI 复制粘贴到每个容器名下,所以每个 cmdb_id 都"看见"它所在节点)
    Market 数据集有显式 node-X.service-Y → 直接抽 prefix
```

产出:`knowledge/node_container_graph.json`
```json
{
  "containers": {
    "IG01": {"node_proxy": "IG01", "container_kpis": [...], "node_kpis": [...]},
    "Tomcat01": {...}, ...
  },
  "nodes": {
    "IG01_node": {"hosted_containers": ["IG01"], "node_kpis": [...]}
  }
}
```

**B-L1.2 Baseline 分布**

在 03_04(数据完整)case 上对每个 (service, container_kpi) 学:
- 健康时段的中位数 / IQR / 5-95 percentile range
- 用 percentile-based 阈值取代 robust-z(解决 D2 爆炸)

产出:`knowledge/baseline_distributions.json`

**B-L1.3 对照诊断基线**

新概念,代替历史 baseline:**跨服务横向 robust-z**
- 一个时间点上,每个 (kpi_name) 在所有有该 KPI 的服务上的值集合
- 服务 X 的该 KPI 异常 = X 偏离该集合的中位数(MAD 来自集合而非历史)
- 优势:不需要历史窗口,昼夜/流量/部署影响被横向对照消除
- 适用条件:同时段至少 5+ 个服务有该 KPI(否则退回历史 baseline)

**B-L1.4 反驳规则库(FP-growth)**

输入:历史 case(record.csv 的 component + reason)+ 每 case 的证据签名
证据签名 = {service_X_container_cpu_high, service_Y_node_disk_high, trace_X_to_Y_slow, log_X_oom_present, ...}(布尔向量)

跑 FP-growth 在 [signature, true_root_cause] 联合空间上挖:
- 高 confidence 的 forward rule:`(IG.container_disk_z>2 AND child_disk_z<1) → root=IG.disk`(confidence=0.85)
- **反向用**:新 case 候选 IG.disk,但 signature 里没有 `IG.container_disk_z>2` —— rule 反向触发,反驳成立

产出:`knowledge/refutation_rules.json`

**B-L1.5 故障形态聚类**

不是按"故障类型"聚,按"证据结构"聚 — 同 cluster 内的 case 候选空间相似
HDBSCAN 在 case-level evidence signature 向量上;cluster 给出"这一类 case 可能的根因只在 {A, B, C} 三个里"

产出:`knowledge/fault_clusters.json`
（包含每个 cluster 的 prototype signature + 历史命中的根因分布）

### 2.4 第二层:反驳引擎(算法细节)

**B-L2.1 evidence_query 模块**

定义结构化查询语言。每个 query 返回布尔 + 证据细节:

```python
class EvidenceQuery:
    def is_container_kpi_anomalous(svc, kpi_bucket, threshold=2.5): ...
    def is_node_kpi_anomalous(node_id, kpi_bucket, threshold=2.5): ...
    def is_earlier_anomaly_downstream(svc, kpi_bucket, time_window): ...
    def is_anomaly_explained_by_node(svc, kpi_bucket): ...
        # 检查同一节点上其他容器是否同步异常 → 是节点级问题
    def does_signature_match_rule(case_signature, rule): ...
```

**B-L2.2 候选空间生成**

1. 取 case 的 evidence signature
2. 在故障形态 cluster 中找最近邻
3. 候选 = cluster 的"prototype 根因"集合(通常 3-5 个,远小于 14 个全服务)

**B-L2.3 反驳分数计算**

对每个候选 (service, reason_type):
```
rebuttal_score = 0
for each rule in refutation_rules:
    if rule applies to (service, reason_type):
        evidence_check = evidence_query.does_signature_match_rule(...)
        if not evidence_check:
            rebuttal_score += rule.confidence  # 反驳越强累加越多
```

最终输出:**按反驳分数升序排序**的候选列表 + 每个候选的反驳证据链。

**B-L2.4 输出格式(不是 ranking,是调查指引)**

```json
{
  "case_id": "...",
  "high_suspicion": [
    {"candidate": "IG01.disk", "rebuttal_score": 0.2,
     "supporting": ["container_disk_z=3.5", "earliest_in_trace"],
     "refuting": []}
  ],
  "low_suspicion": [
    {"candidate": "Mysql01.disk", "rebuttal_score": 2.1,
     "refuting": ["node_explains_all", "no_downstream_propagation"]}
  ],
  "data_blind_spots": [
    {"area": "Tomcat04.metric", "reason": "all container KPIs missing"}
  ],
  "confidence_overall": 0.78
}
```

### 2.5 第三层:LLM 推理层(算法细节)

调用策略:**不是每个 case 都调,只在以下情况调**
- cluster 归类置信度 < 0.5(case 落在已知 cluster 边界)
- 反驳分数最低的 top-2 候选差距 < 0.3(多假设冲突)
- 用户/上游显式要求自然语言报告

LLM Prompt 模板(伪代码):
```
你是 SRE 根因分析助手。
案件:{case_id, telemetry_summary}
候选根因 + 反驳证据链:{from L2 output}
任务:
1. 如果有明显赢家,简短说明为什么并给出 confidence
2. 如果候选冲突,问 1-2 个关键追问(用 evidence_query 接口)
3. 如果数据盲区是决定性的,主动建议补什么数据
```

LLM 不直接出根因,出 **"决策 + 证据陈述 + 补数据建议"**。

### 2.6 数据流(端到端)

```
新 case 到达
  ↓
[Layer 1 离线产物加载到内存]
  ↓
[Layer 2 加载该 case 的 telemetry]
  ↓
build evidence_signature(节点-容器 KPI 分离 + percentile/横向 z)
  ↓
cluster 归类 → 候选空间(3-5 个)
  ↓
for each 候选:run refutation rules → rebuttal_score + evidence chain
  ↓
[决策]
  - 反驳分数最低的明显赢家 → 直接输出
  - 多假设冲突 → 调用 Layer 3 LLM
  - 候选空间空(无信号)→ 输出 "data_blind_spots"
  ↓
返回结构化结果(高嫌疑/低嫌疑/盲区 三档 + confidence)
```

### 2.7 云端目录结构

```
/home/yan/workspace/projects/rca-bank-adaptive/v2/
├── README.md                        # v2 范式总览 + 跑通指南
├── docs/
│   ├── d32_design_pivot.md          # 本文档
│   └── api_reference.md             # 各层接口规范
├── knowledge/                       # Layer 1 离线产物(checkpoint)
│   ├── node_container_graph.json
│   ├── baseline_distributions.json
│   ├── refutation_rules.json
│   ├── fault_clusters.json
│   └── build_metadata.json          # 训练时间、数据日期、版本
├── src/
│   ├── __init__.py
│   ├── data_loader.py               # raw csv → 标准化 case object
│   ├── node_container_split.py      # B-L1.1 节点-容器分离
│   ├── layer1_knowledge.py          # Layer 1 离线训练入口
│   ├── evidence_query.py            # B-L2.1 结构化查询
│   ├── layer2_refutation.py         # Layer 2 反驳引擎
│   ├── layer3_llm.py                # Layer 3 LLM 推理
│   ├── pipeline.py                  # 三层串联,对外 API
│   └── compat/
│       └── chimera_bridge.py        # 复用 d31b dump 作 evidence 输入
├── tests/
│   ├── test_node_container_split.py
│   ├── test_evidence_query.py
│   ├── test_refutation_basic.py
│   └── fixtures/                    # 几个手挑的 case 做单元测试
├── eval/
│   ├── run_v2_smoke.py              # 跑 d31b 同样 43 case smoke
│   ├── compare_v1_v2.py             # case-by-case 跟 chimera 对比
│   ├── leave_one_out.py             # 在挖洞场景下验证 imputation 路线(后期)
│   └── cross_dataset.py             # 在 Telecom / Market 跑 transfer 验证
└── logs/
    ├── v2_knowledge_build_*.log
    ├── v2_smoke_*.log
    └── v2_eval_*.json
```

### 2.8 实施顺序(MVP → 完整)

| Phase | 内容 | 预计耗时 | 输出 |
|---|---|---|---|
| **P1** | 第一层 B-L1.1 + B-L1.2:节点-容器分离 + percentile baseline | 1 天 | `knowledge/node_container_graph.json` + `baseline_distributions.json` |
| **P2** | 第二层最小可行版:evidence_query + 5-6 条手写反驳规则 | 1 天 | `src/layer2_refutation.py` MVP |
| **P3** | 跑 03_10 smoke 验证;跟廉价 A、d31b baseline 对比 | 0.5 天 | `eval/run_v2_smoke.py` + 第一份对比报告 |
| **P4** | 加 FP-growth 提炼反驳规则 + HDBSCAN 故障聚类 | 1 天 | `knowledge/refutation_rules.json` + `fault_clusters.json` |
| **P5** | 第三层 LLM 接入(冲突仲裁 + 报告生成) | 1 天 | `src/layer3_llm.py` |
| **P6** | 全 136 case smoke + Telecom 子集 transfer 验证 | 0.5 天 | 跨集对比报告 |
| **P7** | 论文级别的 ablation:每层贡献多少 pp | 1 天 | ablation 表格 |

**关键里程碑**:Phase 3 结束时,如果 v2 在 03_10 上明显优于廉价 A(top1 ≥ 30%),则继续 P4-P7;如果持平或更差,回头看反驳规则是不是太弱、cluster 是不是太粗,而**不退回 A**。

### 2.9 评测方式

**主指标**:Top-1 / Top-3 命中率(对标 d31b 23.3% / 44.2%)。

**副指标(v2 特有,不可放弃)**:
- **confidence-命中率分层**:高 confidence(>0.7) case 的命中率 ≥ 80%;低 confidence case 主动说"我不知道",不算错
- **data_blind_spot 召回**:apache02 等 metric 全缺的 case 应进 data_blind_spots,不进 high_suspicion
- **跨集迁移**:Telecom / Market 上,不重训练只换 prompt 应能保持 Bank 命中率的 60%+

**可解释性主观指标**:屿手翻 10 个 case,每个反驳证据链能否"3 秒内人能理解"。

### 2.10 跟方案 A 的兼容关系

**保留 / 复用**:
- d31b 的 dump hook(`compute_influence_scores` 落盘)继续保留 → v2 反驳引擎可直接用 dump 作 evidence_query 的快速输入,不必重写 raw csv 解析
- KPI bucket 定义(剔除节点 KPI 版)继续用 → v2 第一层的"容器级 KPI 识别"复用这套 patterns
- `SERVICE_NAME_MAP` + `_map_svc_name_to_scoring` 不动 → v2 也需要服务名规范化

**废弃**:
- chimera 的 `compute_influence_scores` 主排名逻辑 → v2 完全不用 ranking,出 high/low/blind 三档
- chimera 的全局 weight 配置 → v2 没全局权重
- `_robust_z` 在历史 baseline 上的使用 → v2 改 percentile + 横向对照

**桥接**:`src/compat/chimera_bridge.py` 提供一个适配层,让 v2 既能读 raw csv 又能读 chimera dump 作 evidence。这样 d31b 已经跑过的 43 个 dump 不浪费,直接喂给 v2 第二层做反驳测试。

---

## 3. 给下一代晏的明确指引

1. **第一件事**:读本文 + `HANDOFF_d31b.md` + d32e smoke 最终结果(`results/script-d32e-cheapA.log` 的 FINAL 区段)
2. **第二件事**:在 `/home/yan/workspace/projects/rca-bank-adaptive/v2/` 下创建目录骨架,把本文 §2.7 的结构空文件先建起来(README + docs/ + tests/ 占位)
3. **第三件事**:从 Phase 1 开始 —— B-L1.1 节点-容器分离最简单且立刻有价值,可以单独验证(用 d32d 已验证的 IG01 OSLinux 数据做单元测试样例)
4. **不要做**:不要回头改 chimera,不要在 A 路线上再加 patch,不要直接上 GAN/diffusion(数据量不够,且 v2 第一层的 percentile baseline 是 imputation 的更便宜替代)
5. **保持的原则**:屿要的是"通用性"。任何 v2 改动先问"换到 Telecom/Market 还成立吗"

— 网页端晏,d32 末
2026-05-30
