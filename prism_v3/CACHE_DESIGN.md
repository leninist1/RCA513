# PRISM v3 离线缓存设计

> 适用范围：`prismv3-no-leakage-hardening` 及其后续分支
>
> 目标：将与固定 telemetry、公开 query 和安全 anchor 相关的确定性计算前移到离线阶段，避免在每次算法调参和预测时重复执行高成本计算，同时保证缓存不会成为新的 GT 泄露入口。

---

# 1. 为什么需要缓存

PRISM v3 的实验过程包含大量重复计算：

```text
读取 telemetry
→ 标准化数据
→ 按时间窗口切 baseline / fault
→ 构建 trace graph 与 metric graph
→ 聚合 ObjectGraph
→ 计算 NoiseLab 特征
→ 计算 CMI / CF
→ Agent posterior 更新
→ 事件合成
```

其中，前半部分在同一组 telemetry、同一 query、同一安全 anchor 和同一 pipeline version 下是确定性的。反复修改：

```text
排序权重
LTR 模型
posterior 权重
CF / CMI 组合方式
Agent 策略
停止条件
事件合成逻辑
```

时，没有必要重新扫描 telemetry、重新构图和重新计算底层原子特征。

缓存设计的核心原则：

```text
缓存底层原子证据
而不是缓存最终答案
```

---

# 2. No-leakage 原则

缓存系统必须遵守严格 no-leakage 约束。

## 2.1 缓存生成过程禁止访问

```text
record.csv
GT component
GT reason
GT timestamp
scoring_points
任何可恢复答案的字段
```

## 2.2 合法 anchor 来源

缓存只能基于：

```text
public_query_window
telemetry_unsupervised_onset
fallback_query_window_start
```

禁止：

```text
record_csv
ground_truth
gt_inject_time
```

## 2.3 每份缓存必须带 manifest

每份缓存都必须写入：

```text
manifest.json
```

至少包含：

```json
{
  "cache_type": "object_graph",
  "generated_without_gt": true,
  "allowed_anchor_sources": [
    "public_query_window",
    "telemetry_unsupervised_onset"
  ],
  "query_id": "...",
  "anchor_timestamp": 0,
  "anchor_source": "public_query_window",
  "telemetry_sha256": "...",
  "pipeline_version": "...",
  "git_commit": "...",
  "created_at": "...",
  "payload_sha256": "..."
}
```

## 2.4 Strict 模式拒绝以下缓存

```text
manifest 缺失
checksum 不一致
generated_without_gt = false
anchor_source 非法
pipeline_version 不一致
telemetry hash 不一致
query_id 不一致
当前配置与缓存配置不一致
```

---

# 3. 缓存分层

推荐分为五层：

```text
L0：telemetry 标准化缓存
L1：window 与 signal 缓存
L2：graph 与 NoiseLab 特征缓存
L3：causal profile 缓存
L4：单次运行内策略缓存
```

---

# 4. L0：Telemetry 标准化缓存

## 4.1 缓存内容

按：

```text
system
sub_system
telemetry_date
```

缓存：

```text
metrics.parquet
logs.parquet
traces.parquet
entities.json
entity_types.json
manifest.json
```

同时预生成：

```text
按 timestamp 排序的数据
按 entity 分区的数据
基础 schema 信息
原始文件 hash
```

## 4.2 为什么可离线

字段标准化、类型转换、排序和实体索引，与预测策略无关。只要输入文件不变，就可以复用。

## 4.3 目录结构

```text
artifacts/cache/telemetry/
  <system>/
    <sub_system>/
      <telemetry_date>/
        <telemetry_hash>/
          metrics.parquet
          logs.parquet
          traces.parquet
          entities.json
          entity_types.json
          manifest.json
```

## 4.4 缓存键

```text
telemetry_cache_key =
  system
  + sub_system
  + telemetry_date
  + raw_file_hashes
  + telemetry_normalizer_version
```

---

# 5. L1：Window 与 Signal 缓存

## 5.1 Window 切片

给定：

```text
query_id
anchor_timestamp
baseline_window
fault_window
```

缓存：

```text
baseline_metrics.parquet
fault_metrics.parquet
baseline_logs.parquet
fault_logs.parquet
baseline_traces.parquet
fault_traces.parquet
manifest.json
```

## 5.2 多锚点兼容

单锚点阶段：

```text
query_id → public_query_window anchor
```

多锚点阶段：

```text
query_id
  ├─ anchor_1
  ├─ anchor_2
  ├─ anchor_3
  └─ ...
```

每个 anchor 独立缓存窗口。

## 5.3 Window 缓存键

```text
window_cache_key =
  telemetry_cache_key
  + query_id
  + anchor_timestamp
  + anchor_source
  + baseline_window
  + fault_window
  + window_builder_version
```

## 5.4 Signal matrix

为 CMI、CF 和异常传播分析缓存：

```text
signal_matrix.parquet
baseline_mask.npy
fault_mask.npy
entity_order.json
normalization_stats.parquet
manifest.json
```

signal matrix 生成步骤：

```text
baseline / fault 指标拼接
→ baseline 统计量
→ z-score
→ 每实体每时间点选 dominant metric
→ pivot 为 entity × time matrix
```

## 5.5 Signal 缓存键

```text
signal_cache_key =
  window_cache_key
  + signal_builder_version
```

---

# 6. L2：Graph 与 NoiseLab 特征缓存

## 6.1 Trace graph

拆成两层：

### 静态 trace relation index

按 telemetry date 缓存：

```text
src
dst
span_count
trace_count
latency_distribution
error_distribution
first_seen
last_seen
```

### 动态 anchor-window edge weights

按 query-anchor 缓存：

```text
edge_weight
temporal_support
error_shift
latency_shift
direction_confidence
```

## 6.2 Metric graph

缓存：

```text
baseline_stats
fault_stats
pairwise_dependency_score
edge_weight
```

## 6.3 ObjectGraph

缓存：

```text
graph_nodes.parquet
graph_edges.parquet
graph_debug.json
manifest.json
```

每个 node 至少包含：

```text
object_id
members
representative
anomaly_score
earliest_timestamp
metric_score
log_score
trace_score
change_score
reason_votes
evidence
```

## 6.4 ObjectGraph 缓存键

```text
graph_cache_key =
  window_cache_key
  + graph_builder_version
  + trace_graph_config_hash
  + metric_graph_config_hash
  + canonicalization_version
```

## 6.5 NoiseLab 原子特征

缓存：

```text
candidate_features.parquet
manifest.json
```

每行：

```text
query_id
anchor_id
anchor_timestamp
anchor_source
anchor_confidence
object_id
base_score
metric_*
log_*
trace_*
noise_*
structure_*
delay_*
beam_*
subspace_*
mask_*
```

## 6.6 为什么缓存原子特征而非最终 score

后续会频繁修改：

```text
root_source_score 权重
hotspot penalty
reverb penalty
LTR 权重
posterior 权重
anchor stability 权重
```

如果只缓存最终 score，每次改公式仍需重跑底层逻辑。

应缓存：

```text
local_noise
resonance_mass
source_signal
reverb_mass
unexplained_residual
collapse_gain
collapse_recovery_score
exclusive_explanation
temporal_source_score
multi_lead_consistency
hotspot_bias
```

在线阶段只重新组合。

## 6.7 NoiseLab 缓存键

```text
feature_cache_key =
  graph_cache_key
  + noise_field_version
  + structural_encoder_version
  + delay_localizer_version
  + beamformer_version
  + subspace_version
  + reverb_mask_version
```

---

# 7. L3：Causal Profile 缓存

## 7.1 NoiseField intervention profile

缓存：

```text
forward_intervention_profiles.json
reverse_intervention_profiles.json
```

每个 candidate 保存：

```text
impacted nodes
collapse gain
recovery score
coverage
exclusive explanation
```

## 7.2 CF base profile

拆分为：

```text
静态 candidate intervention profile
动态 alternative explainability
```

静态层可离线缓存：

```text
root_score
downstream_recovery
symptom_score
broad_explainer
residual_collapse
downstream_collapse
hotspot_self_collapse
remaining_residual_ratio
collapse_scope
```

动态层按 candidate pool 计算：

```text
alternative_explainability
negative-control advantage
pairwise refutation
```

## 7.3 CMI base profile

缓存：

```text
conditioners
effect_scope
pair_effect(source, target)
baseline ridge coefficients
marginal delta z
conditional residual
parent explainability
```

## 7.4 CMI pool-dependent profile

按 candidate pool hash 缓存：

```text
alternative explainability
repair uniqueness
normalized CMI score
root admissibility
```

## 7.5 CMI 缓存键

基础层：

```text
cmi_base_key =
  graph_cache_key
  + signal_cache_key
  + cmi_builder_version
```

候选池派生层：

```text
cmi_pool_key =
  cmi_base_key
  + candidate_pool_hash
  + cmi_config_hash
```

## 7.6 CF 缓存键

```text
cf_base_key =
  graph_cache_key
  + signal_cache_key
  + cf_builder_version

cf_pool_key =
  cf_base_key
  + candidate_pool_hash
  + cf_config_hash
```

---

# 8. L4：单次运行内策略缓存

适合内存 memoization，但不建议跨实验直接复用：

```text
Agent posterior
event factors
action history
stop readiness
working graph state
pairwise verdict
event split / merge
final synthesis
```

原因：它们依赖：

```text
动作顺序
Agent 策略
候选池
权重配置
停止条件
```

可以缓存底层 pair 特征，但不要缓存最终 verdict。

---

# 9. 离线与在线边界

## 9.1 离线预计算

```text
读取 raw telemetry
→ 标准化 parquet
→ 建立时间索引
→ 生成安全 anchor candidates
→ 切 window
→ 构建 trace / metric graph
→ 构建 ObjectGraph
→ 生成 NoiseLab 原子特征
→ 生成 signal matrix
→ 生成 CF base
→ 生成 CMI base
→ 写入 cache manifest
```

## 9.2 在线预测

```text
加载 cache
→ 根据当前配置组合 NoiseLab score
→ 加载 certified LTR prior
→ 选择候选池
→ 按需读取 CMI / CF 派生结果
→ Agent posterior 更新
→ event search
→ synthesis
```

## 9.3 调参阶段

```text
修改权重
→ 不重跑 telemetry
→ 不重跑 graph
→ 不重跑 intervention
→ 直接读取 cache
→ 快速评估
```

---

# 10. 推荐代码结构

建议新增：

```text
prism_v3/cache/
  __init__.py
  key.py
  manifest.py
  store.py
  telemetry_cache.py
  window_cache.py
  graph_cache.py
  feature_cache.py
  causal_cache.py
  build_features.py
```

职责：

```text
key.py
  统一生成稳定 cache key

manifest.py
  cache manifest 数据结构与严格校验

store.py
  本地文件系统读写、命中检测、checksum 校验

telemetry_cache.py
  telemetry 标准化与分区缓存

window_cache.py
  query-anchor 窗口切分与 signal matrix

graph_cache.py
  trace graph、metric graph、ObjectGraph

feature_cache.py
  NoiseLab 原子特征持久化

causal_cache.py
  CF / CMI base profile 与 pool-dependent profile

build_features.py
  CLI 入口
```

---

# 11. 建议目录结构

```text
artifacts/cache/
  telemetry/
    <system>/<sub_system>/<date>/<telemetry_hash>/

  windows/
    <query_id>/<anchor_hash>/<window_config_hash>/

  graphs/
    <query_id>/<anchor_hash>/<graph_builder_version>/

  features/
    <query_id>/<anchor_hash>/<feature_pipeline_version>/

  causal/
    <query_id>/<anchor_hash>/<graph_hash>/<candidate_pool_hash>/
```

---

# 12. 最小可行版本 MVP

不要一次实现所有层。

第一版只做：

```text
prism_v3/cache/feature_store.py
prism_v3/cache/build_features.py
```

支持：

```text
query_id
anchor_timestamp
ObjectGraph
candidate_features
manifest
```

目录：

```text
artifacts/cache/features/
  Bank/
    <query_id>/
      <anchor_hash>/
        graph_nodes.parquet
        graph_edges.parquet
        candidate_features.parquet
        manifest.json
```

预测时新增：

```bash
--feature-cache-dir artifacts/cache/features
```

行为：

```text
缓存命中
→ strict 校验 manifest
→ 读取缓存

缓存缺失
→ runtime 计算
→ 写入缓存
→ 继续预测
```

---

# 13. MVP 验收标准

## 13.1 正确性

```text
缓存关闭与缓存开启时，预测结果一致
```

## 13.2 性能

至少记录：

```text
单 query runtime
全量 Bank runtime
cache hit rate
telemetry load time
graph build time
NoiseLab feature time
```

目标：

```text
重复实验时显著减少 graph build 与 NoiseLab feature 时间
```

## 13.3 No-leakage

新增 canary：

```text
1. 缓存生成时隐藏 record.csv 仍可运行
2. manifest 缺失时 strict 模式 hard fail
3. generated_without_gt = false 时 hard fail
4. anchor_source 非法时 hard fail
5. telemetry hash 不一致时 hard fail
6. payload checksum 不一致时 hard fail
7. pipeline version 不一致时 hard fail
```

## 13.4 回退能力

```text
缓存损坏
→ strict 模式拒绝使用
→ 非 strict 开发模式可选择重新计算
```

---

# 14. 实施优先级

## P0：立即实现

```text
Feature cache MVP
Cache manifest
Strict checksum 校验
ObjectGraph 序列化
candidate_features parquet
--feature-cache-dir
```

## P1：多锚点接入后实现

```text
query × anchor 缓存
window cache
signal matrix cache
anchor cache
```

## P2：CF / CMI 优化阶段实现

```text
CF base profile
CMI base profile
pair-effect cache
candidate-pool hash
negative-control cache
```

## P3：性能稳定后实现

```text
telemetry 全量分区缓存
LLM response cache
跨实验 cache GC
cache inspection CLI
```

---

# 15. 与后续路线图的关系

缓存系统应作为独立基础设施阶段，优先于大规模性能恢复实验落地：

```text
Phase 0：No-leakage hardening
→ Phase 0.5：Feature Cache MVP
→ Phase 1：多锚点无监督时间定位
→ Phase 2：无泄露 OOF LTR
→ Phase 3：鲁棒 CF / CMI
→ Phase 4：事件集合推断
```

理由：

```text
多锚点会放大计算量
OOF LTR 会反复生成特征
CF negative controls 会显著增加 intervention 次数
消融实验会频繁调整权重
```

提前完成缓存基础设施，可以显著降低后续实验成本。

---

# 16. 不应缓存的内容

不要将以下内容作为跨实验复用缓存：

```text
最终 ranking
最终答案
Agent posterior
动态 action history
pairwise verdict
event split / merge 结果
synthesis 输出
测试集调参结果
任何包含 GT 的中间文件
```

这些内容可以用于 debug 归档，但不能作为预测输入。

---

# 17. 最终目标

缓存体系应满足：

```text
高复用
可审计
可失效
可回退
无 GT
配置敏感
版本敏感
hash 可验证
```

最终实现：

```text
固定 telemetry 和 anchor 下，底层证据只计算一次
算法调参时只重算在线策略层
缓存不会成为旧 GT 结果回流的旁路
```
