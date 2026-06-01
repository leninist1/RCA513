# PRISM v3 后续工作路线图

> 适用分支：`prismv3-no-leakage-hardening`
>
> 当前目标：在完全切断 GT 泄露路径的前提下，逐步恢复并提升 RCA 准确率，同时让每个阶段都可审计、可复现、可回滚。

---

## 0. 背景与当前状态

PRISM 早期版本存在多类数据泄露风险，最典型的是：

1. LTR 训练和外部 score CSV 可能使用了真实标签；
2. 旧版 `noise_lab.runner` 在生成特征前读取 `record.csv` 中的真实注入时间；
3. 许多图构建、窗口切分、候选排序和反事实机制围绕 GT 时间进行了调优；
4. learned model、entity profile 和历史 CSV 缺少可验证的 lineage；
5. 训练、推理和评估边界不够严格，开发阶段容易把评估字段重新带入推理。

第一阶段硬化已经完成，当前分支已经具备：

- `prism_v3/leakage_guard.py`：统一 no-leakage 契约；
- `prism_v3/strict_main.py`：严格推理入口；
- `prism_v3/noise_lab/safe_runner.py`：无 GT 特征导出器；
- `prism_v3/certify_artifact.py`：artifact manifest 认证工具；
- `prism_v3/noise_native/noiselab_adapter.py`：严格外部 score CSV 校验；
- `tests/test_no_leakage_*.py`：基础 canary 测试；
- `prism_v3/NO_LEAKAGE_GUIDE.md`：可信运行说明。

当前最安全的 baseline 是：

```bash
python -m prism_v3.strict_main \
  --systems Bank \
  --prism-noise-lab
```

在完成后续认证工作前，不应启用旧版 LTR CSV、旧 learned artifact 或旧 entity profile。

---

# 1. 总体原则

后续开发必须遵守以下五条原则。

## 1.1 GT 只能进入训练标签拼接与最终评估

GT 包括：

- root cause component；
- root cause reason；
- root cause timestamp；
- `record.csv`；
- scoring points；
- 任何可还原答案的字段。

GT 不得进入：

- telemetry 特征生成；
- 时间锚点生成；
- 图构建；
- NoiseLab runtime scorer；
- CF / CMI；
- LTR 输入特征；
- 线上推理；
- 测试集统计 prior。

## 1.2 所有外部 artifact 必须带 lineage

任何 CSV、模型、embedding、profile、prior、缓存结果都必须带 sidecar manifest：

```text
<artifact>.manifest.json
```

manifest 至少包含：

```json
{
  "artifact_type": "...",
  "feature_pipeline_version": "...",
  "generated_without_gt": true,
  "allowed_anchor_sources": ["public_query_window"],
  "fit_query_ids": [],
  "fit_dates": [],
  "fit_systems": [],
  "git_commit": "...",
  "artifact_sha256": "..."
}
```

## 1.3 任何性能提升都必须可拆解

每次实验至少报告：

- candidate Recall@5；
- candidate Recall@10；
- Top-1 component accuracy；
- reason accuracy；
- time-window accuracy；
- final official score；
- broad explainer error rate；
- 平均推理耗时；
- artifact certification status。

## 1.4 默认使用 inductive evaluation

默认评估模式：

```text
当前 query 只能访问：
- 当前 query 的 telemetry
- 当前 query 的公开自然语言窗口
- 训练集 / 验证集统计
- 已认证 artifact
```

默认禁止：

```text
- 当前测试 query 的 GT
- 测试集中其他 query 的统计
- 在测试集上调权重
- 用全量模型回打训练集后作为 OOF score
```

## 1.5 先恢复可信度，再恢复分数

后续顺序必须是：

```text
泄露封堵
→ 无监督时间锚点
→ 无泄露特征
→ OOF LTR
→ 鲁棒 CF / CMI
→ 多事件推断
→ 学习型策略
```

不要跳过中间阶段直接重新堆叠 heuristic 权重。

---

# 2. Phase 1：多锚点无监督时间定位

## 2.1 问题定义

旧版系统使用 GT 注入时间切分：

```text
baseline = [T - 300, T)
fault    = [T, T + 300]
```

移除 GT 后，许多机制失去最关键的时间对齐信号。当前仅使用公开 query window 起点作为 anchor，虽然安全，但会引入显著噪声。

本阶段目标：

```text
不依赖 GT 时间
→ 生成 Top-K 候选 anchor
→ 对 anchor 分配置信度
→ 让所有后续机制对时间不确定性鲁棒
```

## 2.2 新增数据结构

建议新增：

```text
prism_v3/time_anchor.py
```

核心结构：

```python
@dataclass(frozen=True)
class AnchorHypothesis:
    timestamp: float
    confidence: float
    source_scores: Dict[str, float]
    evidence_ids: Tuple[str, ...]
    source: str

@dataclass
class AnchorSet:
    anchors: List[AnchorHypothesis]
    fallback_anchor: Optional[AnchorHypothesis]
    debug: Dict[str, Any]
```

## 2.3 Anchor 候选来源

### A. Metric 变化点

至少实现：

- rolling MAD；
- z-score first persistent anomaly；
- CUSUM；
- PELT 或其他离线变化点；
- 峰值前沿检测；
- 多指标投票。

建议输出：

```text
metric_change_score
metric_persistence
metric_cross_entity_support
metric_earliness
```

### B. Log burst

至少实现：

- `ERROR` / `Exception` / `Timeout` 突发；
- OOM / killed / refused 等高置信日志；
- exception family 首次出现；
- error density 变化；
- 多实体日志传播顺序。

建议输出：

```text
log_burst_score
fatal_keyword_score
log_family_novelty
log_earliness
```

### C. Trace shift

至少实现：

- latency 分布突变；
- error span 比例突变；
- 调用链断裂；
- 入口到下游传播时延变化；
- trace topology 变化。

建议输出：

```text
trace_latency_shift
trace_error_shift
trace_topology_shift
trace_earliness
```

### D. Public query window

始终保留：

```text
query.time_window.start
```

作为安全 fallback anchor。

## 2.4 Anchor 融合

推荐初始公式：

```text
anchor_score
  = 0.40 × metric_change_score
  + 0.20 × log_burst_score
  + 0.20 × trace_shift_score
  + 0.15 × cross_modal_agreement
  - 0.05 × isolated_noise_penalty
```

保留 Top-K：

```text
K = 3 ~ 5
```

并归一化：

```text
P(anchor_i | evidence)
```

## 2.5 机制改造

所有依赖时间窗口的模块统一改成：

```python
for anchor in anchor_set.anchors:
    result = compute(anchor)
    aggregate += anchor.confidence * result
```

涉及模块：

```text
prism_v3/mace/graph.py
prism_v3/noise_lab/safe_runner.py
prism_v3/noise_native/cmi.py
prism_v3/noise_native/tools.py
prism_v3/prism.py
```

## 2.6 Anchor stability

每个候选实体新增：

```text
anchor_stability
```

定义：

```text
候选在多个合理 anchor 下保持高排名的稳定程度
```

例如：

```text
anchor_stability(entity)
  = Σ P(anchor_i) × indicator(rank_i(entity) <= K)
```

将其用于：

- 候选排序；
- CF 权重门控；
- CMI 权重门控；
- final confidence；
- 停止条件。

## 2.7 验收标准

必须新增：

```text
Anchor MAE
Anchor Recall@3 within 60s
Anchor Recall@5 within 120s
Public-window fallback usage rate
Anchor confidence calibration
```

验收目标：

```text
- Top-3 anchor 在 60 秒内覆盖 GT 时间的比例明显高于单锚点 baseline
- 在 GT 时间不可见时，候选 Recall@10 有显著恢复
- anchor 稳定性低时，后处理权重自动减弱
```

---

# 3. Phase 2：重建完全无泄露的 OOF LTR

## 3.1 问题定义

LTR 可以继续保留，但必须重新训练。旧模型不可信的原因包括：

- 可能使用了 GT 时间对齐后的特征；
- 可能直接或间接包含标签；
- 可能没有 incident-level split；
- 可能使用全量模型回打训练数据；
- 可能在测试集反复调参。

## 3.2 新增训练目录

建议新增：

```text
prism_v3/ltr/
  __init__.py
  dataset.py
  split.py
  train.py
  predict_oof.py
  calibrate.py
  certify.py
  metrics.py
```

## 3.3 Feature pipeline

统一使用：

```bash
python -m prism_v3.noise_lab.safe_runner \
  --system Bank \
  --output artifacts/bank_no_gt_features.csv
```

未来接入多锚点后，特征至少包括：

```text
query_id
object_id
rank
base_score
anchor_source
anchor_confidence
anchor_stability
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

禁止包含：

```text
GT component
GT reason
GT timestamp
record.csv 字段
scoring_points
GT-derived earliest_offset
```

## 3.4 标签拼接

训练标签必须在 feature generation 完成后单独拼接：

```text
label-free features
      ↓
training-only label join
      ↓
LTR dataset
```

建议新增：

```text
prism_v3/ltr/join_labels.py
```

并明确：

```text
join_labels.py 不能被 strict inference import
```

## 3.5 数据切分

禁止随机按 candidate row 切分。

必须按 incident 分组：

```text
GroupKFold(group=query_id)
```

进一步建议：

```text
date-grouped split
leave-one-system-out split
```

实验至少包含：

```text
A. In-system grouped CV
B. Cross-date transfer
C. Cross-system transfer
```

## 3.6 OOF score 生成

正确流程：

```text
fold_1 model train folds 2..K → predict fold_1
fold_2 model train folds 1,3..K → predict fold_2
...
concat OOF predictions
```

禁止：

```text
全量训练模型 → 回打全部训练 query → 伪装成 OOF
```

## 3.7 模型建议

先使用可解释的轻量模型：

```text
- Logistic Regression
- LightGBM Ranker
- XGBoost Ranker
- LambdaMART
```

不要优先使用复杂神经网络。

建议先比较：

```text
base heuristic
base + Logistic Regression
base + LambdaMART
base + XGBoost Ranker
```

## 3.8 LTR 接入策略

LTR 只能作为弱 prior：

```text
final_logit
  = runtime_evidence_logit
  + α × certified_ltr_logit
```

推荐：

```text
α ∈ [0.20, 0.35]
```

并增加门控：

```text
α_effective
  = α
  × anchor_stability
  × modality_agreement
  × artifact_calibration_confidence
```

## 3.9 Artifact 认证

生成 OOF score 后：

```bash
python -m prism_v3.certify_artifact artifacts/bank_oof_ltr_scores.csv \
  --artifact-type no_leak_ltr_scores \
  --feature-pipeline-version noise_lab_no_gt_v2 \
  --anchor-sources public_query_window,telemetry_unsupervised_onset \
  --fit-query-ids "..." \
  --fit-systems Bank \
  --git-commit "$(git rev-parse HEAD)"
```

## 3.10 验收标准

必须报告：

```text
LTR Recall@5
LTR Recall@10
LTR NDCG@5
LTR NDCG@10
OOF Top-1 accuracy
OOF calibration error
Cross-system degradation
```

验收目标：

```text
- OOF score 明显优于 runtime-only baseline
- 测试 query 不得出现在 fit_query_ids
- 删除 record.csv 后 feature generation 仍可运行
- artifact manifest 通过 strict_main 校验
```

---

# 4. Phase 3：让 CF / CMI 适应无 GT 环境

## 4.1 问题定义

旧版 CF / CMI 在 GT 对齐窗口下效果较好，但在无监督 anchor 环境中容易出现：

- 时间窗口漂移；
- broad explainer 偏差；
- 中心节点天然占优；
- 下游症状被误判为根因；
- 单次反事实过度推翻其他证据。

## 4.2 Anchor-marginalized CMI

当前 CMI 保留，但改为：

```text
CMI(anchor_1)
CMI(anchor_2)
CMI(anchor_3)
```

聚合：

```text
robust_cmi_score
  = 0.60 × expected_CMI
  + 0.25 × worst_case_CMI
  + 0.15 × anchor_stability
```

新增 debug：

```text
cmi_by_anchor
cmi_variance
cmi_stability
cmi_effect_scope
cmi_alternative_explanation
```

## 4.3 CF negative controls

对每个候选实体 `c`，同时计算：

```text
CF(c)
CF(random_neighbor)
CF(same_degree_node)
CF(downstream_node)
CF(same_family_node)
```

定义：

```text
cf_advantage
  = CF(candidate) - median(CF(controls))
```

只有当：

```text
cf_advantage > threshold
```

时才允许 CF 提升候选后验。

## 4.4 Broad explainer penalty

继续强化：

```text
broad_explainer_penalty
  ∝ downstream_recovery - self_recovery
```

建议记录：

```text
self_recovery
downstream_recovery
upstream_recovery
neighborhood_recovery
local_specificity
propagation_consistency
broad_explainer_risk
```

## 4.5 权重门控

弱证据时不应激进 rerank：

```text
effective_cf_weight
  = base_cf_weight
  × anchor_stability
  × modality_agreement
  × intervention_reliability

effective_cmi_weight
  = base_cmi_weight
  × anchor_stability
  × cmi_consistency
```

## 4.6 回归用例

必须新增人工构造回归用例：

```text
root cause → Tomcat-like hub → many downstream anomalies
```

期望：

```text
- hub 的 downstream_recovery 高
- hub 的 self_recovery 低
- hub 的 broad_explainer_penalty 高
- 最终仍选择上游局部根因
```

## 4.7 验收标准

报告：

```text
GT rank before CF
GT rank after CF
GT rank after CMI
CF help rate
CF hurt rate
CMI help rate
CMI hurt rate
Broad explainer error rate
```

验收目标：

```text
CF hurt rate 明显下降
Broad explainer error rate 明显下降
弱 anchor 场景下不再出现大幅错误交换
```

---

# 5. Phase 4：事件集合推断与多故障支持

## 5.1 问题定义

独立预测 component、reason、time 后再组合，会产生错误配对：

```text
(A, R2, T1)
(B, R1, T2)
```

必须围绕 `FaultEvent` 做集合级推断。

## 5.2 事件集合搜索

建议新增：

```text
prism_v3/noise_native/event_search.py
```

目标：

```text
E* = argmax score({event_1, event_2, ...})
```

集合得分：

```text
event_set_score
  = Σ event_posterior
  + coverage_gain
  + modality_diversity
  - duplicate_penalty
  - propagation_overlap_penalty
  - incompatible_time_penalty
```

初始使用 Beam Search：

```text
beam_width = 8
max_faults = 3
```

动作：

```text
add
remove
merge
split
replace
```

## 5.3 Split / merge 规则

### Merge 候选

```text
同一组件
相近时间窗口
相同 reason family
高度重叠传播路径
```

### Split 候选

```text
明显分离的时间峰值
不同异常机制
不同传播路径
单一事件无法解释全部证据
```

## 5.4 验收标准

新增：

```text
fault_count_accuracy
event_exact_match
component_recall
reason_accuracy
time_window_iou
event_pairing_accuracy
```

---

# 6. Phase 5：Reason 与时间定位联合建模

## 6.1 Reason taxonomy

建议新增：

```text
prism_v3/reason_taxonomy.py
```

层次结构示例：

```text
resource
  ├─ cpu_saturation
  ├─ memory_pressure
  └─ disk_io

network
  ├─ latency
  ├─ packet_loss
  └─ connection_error

application
  ├─ exception
  ├─ thread_pool_exhaustion
  └─ dependency_failure

database
  ├─ slow_query
  ├─ connection_pool
  └─ lock_contention
```

先预测 family，再预测 detail：

```text
P(reason | event, evidence)
  = P(reason_family | evidence)
  × P(reason_detail | reason_family, evidence)
```

## 6.2 时间定位

不要直接使用最大异常峰值。

目标：

```text
root onset
  = earliest stable anomaly
  with propagation-consistent downstream delay
```

每个事件保存：

```text
time_candidates
  - onset
  - peak
  - propagation
  - fallback
```

## 6.3 验收标准

报告：

```text
reason_family_accuracy
reason_detail_accuracy
onset_mae
time_window_iou
propagation_delay_consistency
```

---

# 7. Phase 6：Agent replay、可解释性与实验基础设施

## 7.1 Agent replay

新增：

```text
prism_v3/replay.py
```

每个 case 输出：

```json
{
  "query_id": "...",
  "anchor_hypotheses": [],
  "initial_candidates": [],
  "steps": [
    {
      "action": "...",
      "target": "...",
      "observation": {},
      "posterior_before": {},
      "posterior_after": {},
      "runtime_cost": 0.0
    }
  ],
  "final_events": []
}
```

## 7.2 Posterior attribution

每个候选输出：

```text
runtime_score
ltr_prior
metric_factor
log_factor
trace_factor
cf_factor
cmi_factor
anchor_stability
symptom_penalty
broad_explainer_penalty
final_score
```

## 7.3 配置文件化

建议新增：

```text
prism_v3/configs/
  bank_runtime_only.yaml
  bank_multi_anchor.yaml
  bank_oof_ltr.yaml
  telecom_transfer.yaml
  market_holdout.yaml
  ablation_no_cmi.yaml
  ablation_no_cf.yaml
  ablation_no_ltr.yaml
```

## 7.4 实验记录

每次实验保存：

```text
config snapshot
git commit
artifact manifest
system
split
query count
metrics
runtime
random seed
```

---

# 8. Phase 7：严格评估协议

## 8.1 数据集角色

建议：

```text
Bank：主要开发集
Telecom：迁移验证集
Market：最终保留测试集
```

Market 在架构稳定前不要反复查看结果。

## 8.2 必跑消融实验

至少包含：

```text
A. runtime-only
B. runtime + multi-anchor
C. runtime + certified OOF LTR
D. runtime + multi-anchor + OOF LTR
E. D + robust CMI
F. E + CF negative controls
G. F + event-set search
```

## 8.3 每次实验输出

```text
Official Score
Component Recall@5
Component Recall@10
Top-1 component accuracy
Reason accuracy
Time-window accuracy
Anchor MAE
Anchor Recall@3 within 60s
Broad explainer error rate
CF help / hurt rate
CMI help / hurt rate
Average runtime
Artifact certification status
```

## 8.4 Canary 测试扩展

新增：

```text
1. 修改 record.csv 后 prediction 不变
2. 删除 record.csv 后 strict inference 仍可运行
3. 随机替换 scoring_points 后 prediction 不变
4. GT query 进入 inference 时 hard fail
5. 未认证 artifact hard fail
6. fit_query_ids 含当前 query 时 hard fail
7. checksum 不匹配 hard fail
8. safe_runner 删除 record.csv 后仍可导出特征
9. 测试集 batch prior 默认关闭
10. manifest lineage 可复现
```

---

# 9. 具体文件级 TODO

## 9.1 需要新增

```text
prism_v3/time_anchor.py
prism_v3/ltr/dataset.py
prism_v3/ltr/split.py
prism_v3/ltr/join_labels.py
prism_v3/ltr/train.py
prism_v3/ltr/predict_oof.py
prism_v3/ltr/calibrate.py
prism_v3/ltr/metrics.py
prism_v3/noise_native/event_search.py
prism_v3/reason_taxonomy.py
prism_v3/replay.py
prism_v3/configs/*.yaml
tests/test_multi_anchor.py
tests/test_ltr_group_split.py
tests/test_oof_scores.py
tests/test_cf_negative_controls.py
tests/test_event_search.py
tests/test_prediction_invariance.py
```

## 9.2 需要重点修改

```text
prism_v3/noise_lab/safe_runner.py
prism_v3/noise_native/noiselab_adapter.py
prism_v3/noise_native/cmi.py
prism_v3/noise_native/tools.py
prism_v3/noise_native/posterior.py
prism_v3/noise_native/synthesis.py
prism_v3/mace/graph.py
prism_v3/prism.py
prism_v3/strict_main.py
```

## 9.3 暂时保留但禁止正式使用

```text
prism_v3/noise_lab/runner.py
prism_v3/main.py
```

它们仅用于历史对照和回归分析。

---

# 10. 建议执行顺序

## Milestone 1：可信 baseline

目标：证明 strict runtime-only 可以稳定运行。

任务：

```text
- 在服务器运行 Phase 0 tests
- 跑 Bank runtime-only
- 保存候选 Recall@5 / Recall@10
- 导出 replay 基础日志
```

验收：

```text
- 所有 no-leakage tests 通过
- strict_main 可跑通 Bank
- 旧 CSV 无法进入推理
```

## Milestone 2：多锚点时间定位

目标：恢复窗口对齐能力。

任务：

```text
- 新增 time_anchor.py
- 实现 metric/log/trace anchor
- Top-K anchor 融合
- 接入 safe_runner 和 graph 构建
```

验收：

```text
- Anchor Recall@3 within 60s 显著优于单锚点
- Recall@10 明显恢复
```

## Milestone 3：无泄露 OOF LTR

目标：恢复排序能力。

任务：

```text
- no-GT 特征导出
- incident grouped split
- OOF score 生成
- artifact 认证
- strict_main 接入
```

验收：

```text
- OOF LTR 优于 runtime-only
- 当前 query 不在 fit_query_ids
- 删除 record.csv 后特征生成仍可运行
```

## Milestone 4：鲁棒 CF / CMI

目标：降低后处理误伤。

任务：

```text
- anchor-marginalized CMI
- CF negative controls
- broad explainer regression
- confidence-aware rerank
```

验收：

```text
- CF hurt rate 降低
- broad explainer error rate 降低
```

## Milestone 5：事件集合推断

目标：提升多故障场景。

任务：

```text
- event_search.py
- Beam Search
- split / merge / replace
- pairing metrics
```

验收：

```text
- fault_count_accuracy 提升
- event_pairing_accuracy 提升
```

## Milestone 6：完整实验报告

目标：形成可提交的系统性实验结果。

任务：

```text
- Bank 开发
- Telecom 迁移
- Market holdout
- 全部消融
- replay 案例分析
```

---

# 11. 不应优先做的事情

暂时不要：

```text
- 继续堆 final arbitration 权重
- 针对 Tomcat 类节点写黑名单
- 提高 Top-K 掩盖召回问题
- 直接重新启用旧 LTR CSV
- 直接启用 --v2-all
- 在 Market 上反复调参
- 直接训练 RL policy
- 继续使用旧 noise_lab.runner 生成训练特征
```

这些做法容易重新引入泄露、过拟合或不可解释的性能波动。

---

# 12. 最终目标

PRISM v3 的目标不是恢复到“依赖 GT 的高分”，而是实现：

```text
可信感知
→ 无监督多锚点时间定位
→ 无泄露 OOF LTR
→ 鲁棒 CF / CMI
→ 事件集合推断
→ 可审计 replay
→ 严格跨系统评估
```

最终系统必须同时满足：

```text
1. record.csv 对推理不可见
2. scoring_points 对推理不可见
3. 所有 artifact 可验证
4. 当前 query 不出现在训练集合
5. 测试集统计默认隔离
6. 性能提升可拆解
7. 每次误判可回放
8. 删除 GT 文件后推理仍可运行
```

---

# 13. 建议的下一步

下一次提交建议只聚焦一个主题：

```text
Phase 1：多锚点无监督时间定位
```

不要同时引入 LTR、CF 和 event search。先证明：

```text
多锚点 anchor
→ 改善窗口对齐
→ 恢复候选召回
```

再进入 OOF LTR 重建。这样最容易定位每个机制的真实贡献。
