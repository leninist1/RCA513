# OpenRCA-flow AIOps/Eadro 迁移交接文档

日期：2026-06-19（§10 更新于同日）  
分支：`trace_summary_d32v2`  
工作目录：`/home/dell2/RCA513/yyx/trace_summary`

## 0. 接手前最重要的原则

用户明确要求：

1. OpenRCA 上的算法输出必须保持不变。
2. Eadro/AIOps2021 上必须尽量复用 OpenRCA 原始算法流程。
3. 只能在数据集层面做适配，不要再用“portable runner 自己改 selector”的方式折中。
4. 不同数据集的评估任务不同，AIOps2021 不能用 OpenRCA 指标衡量。
5. 禁止把 GT 信息用于在线 rerank、candidate generation、selector、time anchor。

当前推荐路线：

只做数据层语义对齐：

- KPI 名称 canonicalization
- trace/log/topology/component role 到 OpenRCA 可消费证据的映射
- reason classifier 离线训练特征与在线 selector 特征对齐
- 不改 `D32RefutationPipeline` 的 online ranking/time anchor 逻辑

## 1. 已确认的历史问题

### 1.1 之前 AIOps runner 不等价 OpenRCA runner

旧入口：

`confidence_aware_rca_v2_drift_focus_2026-06-02_resend/source/eval/run_portable_d32.py`

里面的 `_PortableD32Pipeline(D32RefutationPipeline)` 重写了多处 OpenRCA 核心逻辑：

- `_candidate_space`
- `_compute_reason_posterior`
- `_learned_reason_posterior`
- `_component_earliness_strength`

并且有 portable ontology expansion、label-only reason 后处理等逻辑。

因此旧 AIOps 实验不能证明“OpenRCA 原算法迁移失败或成功”，只能证明 portable 变体的效果。

### 1.2 之前 reason classifier 不是 OpenRCA-flow classifier

旧训练脚本：

`eval/train_portable_reason_classifier.py`

使用的是：

`refute_b_v2_d32.portable_ontology.extract_portable_reason_features`

而不是：

`refute_b_v2_d32.reason_classifier.extract_features`

另外，之前 AIOps classifier 大多是 metric-only 或 portable-feature 版本。用户怀疑“没有在新数据集上用多模态重新训练 reason classifier”，这个怀疑基本成立。

### 1.3 AIOps 严格 OpenRCA-flow 失败的主要原因

不是简单的 candidate coverage 问题。

已诊断结果：

- Test 中 GT component 在 candidate space 覆盖接近 100%。
- GT pair 也常在 candidates/decisions 里。
- 但 GT pair rank 很靠后。
- reason 判对时，service top1 仍经常错。

根因更像是：

1. AIOps KPI/log/trace 词表和 OpenRCA 证据词表不一致。
2. reason posterior 和 component scorer 对同一类证据的理解不一致。
3. AIOps 的 trace slow edge / gateway / MG/IG 传播现象容易把 top1 推向网关或中间件。
4. Layer-1 cluster/mine-rule 在 AIOps 上退化严重。

## 2. 新增/修改文件

### 2.1 严格 OpenRCA-flow 入口

新增：

`confidence_aware_rca_v2_drift_focus_2026-06-02_resend/source/eval/run_portable_openrca_flow.py`

用途：

- 用 Eadro/AIOps2021 tabular input 构造 `NormalizedIncident`
- 在线 selector 只调用 base `D32RefutationPipeline`
- 不走 `_PortableD32Pipeline`
- 不做 portable ontology selector override
- 不做 label-only reason 后处理

关键参数：

- `--kpi-canonicalization none|openrca`
- `--reason-classifier-path`
- `--knowledge-json`
- `--pre-baseline-sec`
- `--use-openrca-default-family-map`

注意：

默认使用空 dataset family map：

`DATASET_FAMILY_MAPS={args.dataset: {}}`

这是数据 ontology 适配，不是 selector 改法。原因是 AIOps 的组件名不是 OpenRCA 的 `docker_`/`os_`/`db_`，直接使用 OpenRCA default family map 会过滤掉很多合法候选。

### 2.2 OpenRCA-feature reason classifier 训练脚本

新增/修改：

`confidence_aware_rca_v2_drift_focus_2026-06-02_resend/source/eval/train_portable_openrca_reason_classifier.py`

用途：

- 使用 portable 数据集 train split 离线训练 reason classifier
- 特征提取使用 OpenRCA D32 原始：
  `refute_b_v2_d32.reason_classifier.extract_features`
- 支持 metric/log/trace summary
- 支持 KPI canonicalization
- 支持训练特征与在线 selector 对齐：
  `--joint-feature-mode raw|candidate`

重点：

`--joint-feature-mode candidate` 是后续推荐使用的模式。  
它会在训练时生成在线 D32 selector 实际会看到的 joint candidates，然后用这些 candidates 调 `extract_features`。这能减少 raw-joint training feature 与 online candidate-joint inference feature 的分布偏移。

### 2.3 OpenRCA Layer-1 knowledge 构建脚本

新增：

`confidence_aware_rca_v2_drift_focus_2026-06-02_resend/source/eval/build_portable_openrca_knowledge.py`

用途：

- 使用 train split 构建 D32 Layer-1 knowledge
- 用 OpenRCA `build_case_signature` / `flatten_signature` / `extract_joint_reason_features`
- 不使用 portable ontology augmentation
- 输出 `D32Knowledge` 可读取的 JSON

注意：

AIOps train split 112 cases 下：

- non-canonical: 1 cluster, 0 mined rules
- KPI canonicalization: 1 cluster, 10 mined rules

cluster 仍然退化，这是后续重点。

### 2.4 KPI canonicalizer

新增：

`confidence_aware_rca_v2_drift_focus_2026-06-02_resend/source/refute_b_v2_d32/portable_kpi_canonicalizer.py`

修改：

`confidence_aware_rca_v2_drift_focus_2026-06-02_resend/source/refute_b_v2_d32/portable_adapters.py`

设计：

- 只改 `kpi_name`
- 不改 timestamp/cmdb_id/value
- 原始 KPI 通过 hash suffix 保留一 KPI 一分布，避免把不同 KPI 合并成一个 baseline key
- `mode=none` 默认 no-op
- `mode=openrca` 显式开启

例子：

- `OSLinux-CPU_CPU_CPULoad` -> `OpenRCA_CPU_CPULoad_<hash>`
- `JVM-Memory_7778_JVM_Memory_HeapMemoryUsage` -> `OpenRCA_JVM_HeapMemory_<hash>`
- `OSLinux-OSLinux_NETWORK_ens160_NETPacketsIn` -> `OpenRCA_Network_latency_<hash>`
- `Tomcat-Requests_*_ErrorCountRequestInfo` -> `OpenRCA_NetworkPacketErrLoss_<hash>`
- `Tomcat-Sessions_*_SESSIONRejectedSessions` -> `OpenRCA_NetworkPacketErrLoss_<hash>`

经验教训：

不要把所有 `NETPacketsIn/Out` 都映射成 packet loss。之前 reason posterior 会一边倒预测 `network_packet_loss`。当前 canonicalizer 把泛化 packets/connection/TCP 状态主要映射为 latency，只有 error/drop/loss/retrans/reject/reset/abort 等强 token 映射为 loss。

## 3. AIOps2021 输入/输出位置

输入：

```bash
confidence_aware_rca_v2_drift_focus_2026-06-02_resend/source/logs/portable_exp/aiops2021_metric_adaptive_inputs_tzfix/cases.csv
confidence_aware_rca_v2_drift_focus_2026-06-02_resend/source/logs/portable_exp/aiops2021_metric_adaptive_inputs_tzfix/metrics.csv
confidence_aware_rca_v2_drift_focus_2026-06-02_resend/source/logs/portable_exp/aiops2021_multimodal_inputs/logs.csv
confidence_aware_rca_v2_drift_focus_2026-06-02_resend/source/logs/portable_exp/aiops2021_multimodal_inputs/trace_summaries
confidence_aware_rca_v2_drift_focus_2026-06-02_resend/source/logs/portable_exp/aiops2021_multimodal_inputs/topology.json
```

split：

- train: 112 cases
- test: 47 cases
- split column: `data_type`

使用 python：

```bash
.venv_d32/bin/python
```

系统 `python3` 没有 pandas，不要用系统 python 跑实验。

## 4. 关键实验结果

### 4.1 严格 OpenRCA-flow 三个版本

全部使用 AIOps2021 正确任务指标：

- service localization
- anomaly type classification
- service+type tuple accuracy

| strict OpenRCA-flow variant | Test service@1 | Test HR@3 | Test HR@5 | Test type acc | Test tuple |
|---|---:|---:|---:|---:|---:|
| non-canonical | 0.1489 | 0.3191 | 0.5319 | 0.2553 | 0.0638 |
| KPI canonical + raw classifier | 0.1702 | 0.5957 | 0.7447 | 0.2340 | 0.0851 |
| KPI canonical + candidate classifier | 0.2128 | 0.4894 | 0.7660 | 0.2766 | 0.1277 |

Overall：

| strict OpenRCA-flow variant | Overall service@1 | HR@3 | HR@5 | type acc | tuple |
|---|---:|---:|---:|---:|---:|
| non-canonical | 0.2516 | 0.4403 | 0.6730 | 0.5346 | 0.1761 |
| KPI canonical + raw classifier | 0.3270 | 0.6289 | 0.7610 | 0.5409 | 0.2264 |
| KPI canonical + candidate classifier | 0.3711 | 0.5912 | 0.7736 | 0.6478 | 0.3145 |

对照旧 metric label-only portable run：

| run | Test service@1 | Test HR@3 | Test HR@5 | Test type acc | Test tuple |
|---|---:|---:|---:|---:|---:|
| old metric label-only | 0.5319 | 0.7447 | 0.8298 | 0.4468 | 0.2766 |

解释：

- canonicalization 显著提升 top-k service coverage。
- candidate-mode classifier 提升 top1/type/tuple，但 test service@1 仍低。
- 严格 OpenRCA-flow 当前还没有超过旧 metric label-only baseline。

### 4.2 Reason classifier 独立评估

| classifier | Test top1 | Test R@3 | 说明 |
|---|---:|---:|---|
| non-canonical raw-joint | 0.4894 | 0.7872 | strict OpenRCA-feature 初版 |
| KPI canonical raw-joint | 0.5532 | 0.8298 | 独立 classifier 最好 top1 |
| KPI canonical candidate-joint | 0.5319 | 0.8723 | 最好 R@3，与在线特征更一致 |

### 4.3 Online selector posterior 诊断

non-canonical strict-flow test：

- posterior top1 reason acc: 0.255
- posterior R@3: 0.766
- top1 posterior 几乎一边倒：
  `network_packet_loss`: 37/47

KPI canonical + candidate classifier test：

- posterior top1 reason acc: 0.277
- posterior R@3: 0.809
- top1 posterior 分布改善：
  - `network_packet_loss`: 23/47
  - `network_latency`: 15/47
  - 其他类型开始出现

但仍有问题：

- CPU test top1 recall 只有 2/13
- network latency/loss 仍大量互相混淆
- filesystem test 样本只有 2 个，不稳定

### 4.4 Component ranking 诊断

non-canonical strict-flow test：

- GT component rank <= 3: 14/47
- GT pair rank <= 3: 3/47

KPI canonical + raw classifier test：

- GT component rank <= 3: 27/47
- GT pair rank <= 3: 10/47

KPI canonical + candidate classifier test：

- GT component rank <= 3: 23/47
- GT pair rank <= 3: 10/47

解释：

KPI canonicalization 很有效地把 GT service 拉进 top-k，但 top1 scorer 仍不足。后续要重点做 component-local evidence 对齐，而不是再盲目调 reason classifier。

## 5. 已跑命令

### 5.1 KPI canonical + candidate classifier 训练

```bash
.venv_d32/bin/python confidence_aware_rca_v2_drift_focus_2026-06-02_resend/source/eval/train_portable_openrca_reason_classifier.py \
  --dataset aiops2021 \
  --cases-csv confidence_aware_rca_v2_drift_focus_2026-06-02_resend/source/logs/portable_exp/aiops2021_metric_adaptive_inputs_tzfix/cases.csv \
  --metrics-csv confidence_aware_rca_v2_drift_focus_2026-06-02_resend/source/logs/portable_exp/aiops2021_metric_adaptive_inputs_tzfix/metrics.csv \
  --logs-csv confidence_aware_rca_v2_drift_focus_2026-06-02_resend/source/logs/portable_exp/aiops2021_multimodal_inputs/logs.csv \
  --trace-summary-dir confidence_aware_rca_v2_drift_focus_2026-06-02_resend/source/logs/portable_exp/aiops2021_multimodal_inputs/trace_summaries \
  --topology-json confidence_aware_rca_v2_drift_focus_2026-06-02_resend/source/logs/portable_exp/aiops2021_multimodal_inputs/topology.json \
  --pre-baseline-sec 3600 \
  --kpi-canonicalization openrca \
  --joint-feature-mode candidate \
  --train-split-column data_type \
  --train-split-value train \
  --eval-split-value test \
  --out confidence_aware_rca_v2_drift_focus_2026-06-02_resend/source/logs/portable_exp/aiops2021_openrca_kpicanon_candidateclf_artifacts/reason_classifier_openrca_features.json \
  --summary-json confidence_aware_rca_v2_drift_focus_2026-06-02_resend/source/logs/portable_exp/aiops2021_openrca_kpicanon_candidateclf_artifacts/reason_classifier_openrca_features_summary.json
```

### 5.2 KPI canonical knowledge 构建

```bash
.venv_d32/bin/python confidence_aware_rca_v2_drift_focus_2026-06-02_resend/source/eval/build_portable_openrca_knowledge.py \
  --dataset aiops2021 \
  --cases-csv confidence_aware_rca_v2_drift_focus_2026-06-02_resend/source/logs/portable_exp/aiops2021_metric_adaptive_inputs_tzfix/cases.csv \
  --metrics-csv confidence_aware_rca_v2_drift_focus_2026-06-02_resend/source/logs/portable_exp/aiops2021_metric_adaptive_inputs_tzfix/metrics.csv \
  --logs-csv confidence_aware_rca_v2_drift_focus_2026-06-02_resend/source/logs/portable_exp/aiops2021_multimodal_inputs/logs.csv \
  --trace-summary-dir confidence_aware_rca_v2_drift_focus_2026-06-02_resend/source/logs/portable_exp/aiops2021_multimodal_inputs/trace_summaries \
  --topology-json confidence_aware_rca_v2_drift_focus_2026-06-02_resend/source/logs/portable_exp/aiops2021_multimodal_inputs/topology.json \
  --pre-baseline-sec 3600 \
  --kpi-canonicalization openrca \
  --modalities metric,log,trace \
  --train-split-column data_type \
  --train-split-value train \
  --disable-rule-validation \
  --out confidence_aware_rca_v2_drift_focus_2026-06-02_resend/source/logs/portable_exp/aiops2021_openrca_kpicanon_artifacts/d32_knowledge_openrca_recipe_train.json
```

### 5.3 KPI canonical + candidate classifier strict-flow run

```bash
.venv_d32/bin/python confidence_aware_rca_v2_drift_focus_2026-06-02_resend/source/eval/run_portable_openrca_flow.py \
  --dataset aiops2021 \
  --cases-csv confidence_aware_rca_v2_drift_focus_2026-06-02_resend/source/logs/portable_exp/aiops2021_metric_adaptive_inputs_tzfix/cases.csv \
  --metrics-csv confidence_aware_rca_v2_drift_focus_2026-06-02_resend/source/logs/portable_exp/aiops2021_metric_adaptive_inputs_tzfix/metrics.csv \
  --logs-csv confidence_aware_rca_v2_drift_focus_2026-06-02_resend/source/logs/portable_exp/aiops2021_multimodal_inputs/logs.csv \
  --trace-summary-dir confidence_aware_rca_v2_drift_focus_2026-06-02_resend/source/logs/portable_exp/aiops2021_multimodal_inputs/trace_summaries \
  --topology-json confidence_aware_rca_v2_drift_focus_2026-06-02_resend/source/logs/portable_exp/aiops2021_multimodal_inputs/topology.json \
  --rules confidence_aware_rca_v2_drift_focus_2026-06-02_resend/source/knowledge/refutation_rules_v2.json \
  --knowledge-json confidence_aware_rca_v2_drift_focus_2026-06-02_resend/source/logs/portable_exp/aiops2021_openrca_kpicanon_artifacts/d32_knowledge_openrca_recipe_train.json \
  --reason-classifier-path confidence_aware_rca_v2_drift_focus_2026-06-02_resend/source/logs/portable_exp/aiops2021_openrca_kpicanon_candidateclf_artifacts/reason_classifier_openrca_features.json \
  --pre-baseline-sec 3600 \
  --kpi-canonicalization openrca \
  --modalities metric,log,trace \
  --out confidence_aware_rca_v2_drift_focus_2026-06-02_resend/source/logs/portable_exp/aiops2021_openrca_kpicanon_candidateclf_run/predictions.csv \
  --debug-json confidence_aware_rca_v2_drift_focus_2026-06-02_resend/source/logs/portable_exp/aiops2021_openrca_kpicanon_candidateclf_run/debug.json \
  --baseline-reliability-json confidence_aware_rca_v2_drift_focus_2026-06-02_resend/source/logs/portable_exp/aiops2021_openrca_kpicanon_candidateclf_run/baseline_reliability.json \
  --input-audit-jsonl confidence_aware_rca_v2_drift_focus_2026-06-02_resend/source/logs/portable_exp/aiops2021_openrca_kpicanon_candidateclf_run/input_audit.jsonl
```

### 5.4 AIOps2021 evaluation

```bash
.venv_d32/bin/python confidence_aware_rca_v2_drift_focus_2026-06-02_resend/source/eval/evaluate_portable_task.py \
  --dataset aiops2021 \
  --cases-csv confidence_aware_rca_v2_drift_focus_2026-06-02_resend/source/logs/portable_exp/aiops2021_metric_adaptive_inputs_tzfix/cases.csv \
  --pred confidence_aware_rca_v2_drift_focus_2026-06-02_resend/source/logs/portable_exp/aiops2021_openrca_kpicanon_candidateclf_run/predictions.csv \
  --debug-json confidence_aware_rca_v2_drift_focus_2026-06-02_resend/source/logs/portable_exp/aiops2021_openrca_kpicanon_candidateclf_run/debug.json \
  --out-json confidence_aware_rca_v2_drift_focus_2026-06-02_resend/source/logs/portable_exp/aiops2021_openrca_kpicanon_candidateclf_run/task_eval.json \
  --details-csv confidence_aware_rca_v2_drift_focus_2026-06-02_resend/source/logs/portable_exp/aiops2021_openrca_kpicanon_candidateclf_run/task_eval_details.csv
```

## 6. 当前最值得继续的方向

### 6.1 Component-local evidence 对齐

当前 top-k 已经有明显提升，说明 GT service 进来了，但 top1 不够。

下一步重点不要只盯 reason classifier，而要让 component scorer 看到更合理的 component-local evidence。

建议方向：

1. AIOps 服务角色映射到 OpenRCA topology/trace 证据：
   - apache/IG/MG/Tomcat/Redis/Mysql 的角色要明确
   - 当前 MG/IG 容易因为 trace first/upstream 被推成 top1
   - 需要区分 propagation symptom 和 true root service evidence

2. trace summary 数据层重写：
   - 不改 `D32RefutationPipeline`
   - 可以在 adapter/trace summary preprocessor 层添加 OpenRCA 可消费字段
   - 目标是让 Tomcat/业务服务的 direct evidence 进入 component score，而不是只让网关拿到 propagation support

3. log canonicalization：
   - 目前 access logs 多，错误日志很稀疏
   - 不要把普通 access latency/access count 全部当 root-cause log evidence
   - 需要 template/keyword 过滤，只把 OOM/error/reject/timeout/reset 等强证据注入 OpenRCA log reason tokens

4. component evidence prior 数据层改进：
   - 当前 signature 已经有一些 metric prior，但 `_component_earliness_strength` 只按 reason bucket 重扫 metric rows
   - 由于 bucket/KPI 语义仍不完整，很多 GT pair 虽有 candidate 但 score 不够
   - 继续完善 KPI canonicalization，尤其 Tomcat/JVM/session/request 指标

### 6.2 Layer-1 cluster 退化

现象：

- AIOps train 112 cases 仍然只有 1 cluster。
- mined rules canonical 后有 10 条，但 cluster 仍没有区分能力。

可做：

1. 检查 `flatten_signature(signature)` 输出是否过于稀疏/过于同质。
2. 在数据层给 signature 加更稳定的 dataset-local symptom features。
3. 不要改 `_cluster_cases` 或 `D32Knowledge.match_clusters` 算法，除非用户同意改变 OpenRCA flow。
4. 可以在 knowledge build 的输入 signature 层做 canonical feature 注入，但必须无 GT、只来自窗口证据。

### 6.3 Eadro 迁移

目前主要跑的是 AIOps2021。Eadro 也要按同样原则做：

- 先 strict OpenRCA-flow baseline
- 再 KPI canonicalization
- 再 candidate-mode reason classifier
- 评估用 Eadro root cause localization 指标，不要用 AIOps/OpenRCA 指标

Eadro 输入已有：

```bash
logs/portable_exp/eadro_sn_metric_adaptive_inputs
logs/portable_exp/eadro_sn_multimodal_inputs
logs/portable_exp/eadro_tt_metric_adaptive_inputs
logs/portable_exp/eadro_tt_multimodal_inputs
```

需要注意 Eadro 的 split/场景可能与 AIOps `data_type=train/test` 不一样，先检查 cases CSV 列。

## 7. 不要做的事

1. 不要直接修改 `D32RefutationPipeline._evaluate_two_stage` 来适配 AIOps，除非用户明确放宽“OpenRCA 输出保持不变/流程一致”的要求。
2. 不要在 online runner 中使用 GT label。
3. 不要混用 OpenRCA 指标评估 AIOps/Eadro。
4. 不要把 API key 写入文档、日志或 git。
5. 不要继续扩展 `_PortableD32Pipeline` 来证明主方法，它已经被确认不是 OpenRCA-equivalent。
6. 不要只看 overall 指标。AIOps 的 train/test 差异很大，必须看 test split。

## 8. Git/工作树注意事项

当前工作树里有大量未跟踪文件和历史输出。接手 agent 不要随手清理。

重点新增文件是：

```bash
confidence_aware_rca_v2_drift_focus_2026-06-02_resend/source/refute_b_v2_d32/portable_kpi_canonicalizer.py
confidence_aware_rca_v2_drift_focus_2026-06-02_resend/source/refute_b_v2_d32/portable_adapters.py
confidence_aware_rca_v2_drift_focus_2026-06-02_resend/source/eval/train_portable_openrca_reason_classifier.py
confidence_aware_rca_v2_drift_focus_2026-06-02_resend/source/eval/build_portable_openrca_knowledge.py
confidence_aware_rca_v2_drift_focus_2026-06-02_resend/source/eval/run_portable_openrca_flow.py
```

还有历史上已有但未跟踪的 portable 文件：

```bash
eval/run_portable_d32.py
eval/train_portable_reason_classifier.py
eval/build_portable_d32_knowledge.py
eval/build_portable_multimodal_inputs.py
eval/evaluate_portable_task.py
refute_b_v2_d32/adaptive_baseline.py
refute_b_v2_d32/entity_roles.py
refute_b_v2_d32/portable_ontology.py
refute_b_v2_d32/portable_schema.py
```

不要误删它们。

`layer2.py` 当前工作树也显示有改动。接手前请先查看 diff，确认是否是已有用户/历史改动，不要回滚。

## 9. 当前结论

最有价值的经验：

1. 严格复用 OpenRCA flow 后，AIOps 初始表现很差，不是因为没有 candidates，而是 reason/component evidence 语义不对齐。
2. KPI canonicalization 是正确方向，能显著提升 service top-k。
3. 训练 reason classifier 时必须和在线 selector 的 feature construction 对齐，否则独立 recall 好看但在线 posterior 不工作。
4. ~~下一阶段的收益点在 top1 component ranking，而不是单纯提高 reason R@3。~~ **已证伪，见 §10**：top1 错的主因是训练/在线 reason feature skew，不是 component ranking。修正该 skew 后 top1 大幅提升。
5. 数据层 trace/log/topology 语义适配是下一步关键。

一句话路线：

继续保持 `D32RefutationPipeline` 不动，把 AIOps/Eadro 的 metric/log/trace/topology 输入转写成 OpenRCA D32 原流程能理解的证据语言。

## 10. 2026-06-19 train/serve feature skew 诊断与修复

### 10.1 修正前一节的判断

§9 第 4 条原先判断"收益点在 top1 component ranking 而非 reason"。在 `aiops2021_openrca_kpicanon_candidateclf_run` 的 details 上做定向诊断后，该判断被证伪。

诊断方法：取 test split "GT 在 top5 但 top1 错" 的 26 个 case，按 `anomaly_type_accuracy` 拆分。

结果：

- type WRONG（reason 判错）: 21/26
- type RIGHT（纯 service ranking）: 5/26
- 单一最大错误：gt=cpu → pred=network_packet_loss: 9 例
- network_latency ↔ packet_loss 互混: 8 例

即 80% 的 top1 错误先由 reason 判错把错误 service 推上去，不是 component ranking 问题。

### 10.2 真正根因：训练/在线 joint feature skew

进一步对比 offline classifier 与 online posterior：

| 来源 | case 130 (gt=Tomcat02/cpu) top1 |
|---|---|
| offline classifier 独立 eval top1 | 0.5319，cpu recall 9/13 |
| online reason_posterior top1 acc | 0.277，cpu→packet_loss 9 例 |

两者严重不一致。复现 case 130 的 `extract_features` + `predict_proba`：

- 用训练时的 16 个 `generate_joint_root_candidates` 输出 → classifier 给 cpu=0.615（正确）。
- 用在线 `reason_posterior` 的实际输入 → network_packet_loss=0.378（错），与 debug.json 逐位一致。

关键链条：

1. `_candidate_space`（layer2.py:205-279）用 `_put_best` 按 `(component, reason)` 去重，保留高 prior。
2. layer1_fault_cluster 候选的 prior 高于同 key 的 joint 候选，把后者挤掉。
3. `_learned_reason_posterior` 把 `_candidate_space` 的完整输出传给 `extract_features`。
4. `extract_features`→`_joint_features`（reason_classifier.py:187-211）**只统计 `source=="joint_generator"` 的候选**。
5. 训练时 `extract_features` 收到的是未经去重的 16 个 joint 候选；在线只剩被挤占后的子集。

case 130 实测：

- candidate_space 共 52 条，joint_generator 幸存 5 条（4 network_latency + 1 cpu）。
- `joint_cpu_max_strength` 从训练态 8.779 跌到在线态 2.674，cpu joint 信号几乎消失。

test 全量量化：

- joint 幸存数中位 8（训练用 16）。
- 18/34 type-wrong case 的 GT-bucket joint 信号被 `_put_best` 完全挤掉。
- 在线 type acc 0.277 远低于 offline 0.532，gap 即 skew。

### 10.3 修复（训练侧对齐，不动 pipeline）

文件：`eval/train_portable_openrca_reason_classifier.py`

新增参数：

- `--knowledge-json`：与 run 脚本同一份 D32 knowledge。
- `--candidate-space-alignment`：强制在线 `_candidate_space` 对齐。

当 `--joint-feature-mode candidate` 且提供 `--knowledge-json` 时，训练 feature 不再用裸 `generate_joint_root_candidates`，而是实例化 `D32RefutationPipeline`（`enable_two_stage_selector=False` 跳过 classifier 加载），对每个 case 跑完整 `_candidate_space`（含 layer1 cluster + mined + `_put_best` 去重），把去重后的 candidates 喂 `extract_features`。

这样训练 feature 与在线 feature 完全一致，符合 §0 原则：不改 `D32RefutationPipeline` online ranking/time anchor，不改 `reason_classifier`，只在数据/训练层对齐。

### 10.4 已跑命令

训练（加 `--knowledge-json --candidate-space-alignment`，输出到 `aiops2021_openrca_kpicanon_candidatespaceclf_artifacts`）：

```bash
.venv_d32/bin/python confidence_aware_rca_v2_drift_focus_2026-06-02_resend/source/eval/train_portable_openrca_reason_classifier.py \
  --dataset aiops2021 \
  --cases-csv confidence_aware_rca_v2_drift_focus_2026-06-02_resend/source/logs/portable_exp/aiops2021_metric_adaptive_inputs_tzfix/cases.csv \
  --metrics-csv confidence_aware_rca_v2_drift_focus_2026-06-02_resend/source/logs/portable_exp/aiops2021_metric_adaptive_inputs_tzfix/metrics.csv \
  --logs-csv confidence_aware_rca_v2_drift_focus_2026-06-02_resend/source/logs/portable_exp/aiops2021_multimodal_inputs/logs.csv \
  --trace-summary-dir confidence_aware_rca_v2_drift_focus_2026-06-02_resend/source/logs/portable_exp/aiops2021_multimodal_inputs/trace_summaries \
  --topology-json confidence_aware_rca_v2_drift_focus_2026-06-02_resend/source/logs/portable_exp/aiops2021_multimodal_inputs/topology.json \
  --pre-baseline-sec 3600 \
  --kpi-canonicalization openrca \
  --joint-feature-mode candidate \
  --knowledge-json confidence_aware_rca_v2_drift_focus_2026-06-02_resend/source/logs/portable_exp/aiops2021_openrca_kpicanon_artifacts/d32_knowledge_openrca_recipe_train.json \
  --candidate-space-alignment \
  --train-split-column data_type \
  --train-split-value train \
  --eval-split-value test \
  --out confidence_aware_rca_v2_drift_focus_2026-06-02_resend/source/logs/portable_exp/aiops2021_openrca_kpicanon_candidatespaceclf_artifacts/reason_classifier_openrca_features.json \
  --summary-json confidence_aware_rca_v2_drift_focus_2026-06-02_resend/source/logs/portable_exp/aiops2021_openrca_kpicanon_candidatespaceclf_artifacts/reason_classifier_openrca_features_summary.json
```

run + evaluate 与 §5.3/§5.4 相同，仅 `--reason-classifier-path` 和输出目录换成 `aiops2021_openrca_kpicanon_candidatespaceclf_*`。

### 10.5 结果

| 指标 (test split) | non-canonical | KPI canonical + raw clf | KPI canonical + candidate clf | **KPI canonical + candidatespace clf (新)** | 旧 metric label-only |
|---|---:|---:|---:|---:|---:|
| service@1 | 0.1489 | 0.1702 | 0.2128 | **0.3830** | 0.5319 |
| HR@3 | 0.3191 | 0.5957 | 0.4894 | **0.6383** | 0.7447 |
| HR@5 | 0.5319 | 0.7447 | 0.7660 | **0.7872** | 0.8298 |
| type acc | 0.2553 | 0.2340 | 0.2766 | **0.4681** | 0.4468 |
| tuple | 0.0638 | 0.0851 | 0.1277 | **0.3191** | 0.2766 |

关键观察：

1. test service@1 从 0.2128 → 0.3830（+80%），tuple 从 0.1277 → 0.3191（+150%）。
2. **type acc 与 tuple 已超过旧 metric label-only baseline**（0.4681 > 0.4468，0.3191 > 0.2766）。
3. train/serve skew 归零：在线 type acc 0.4681 ≈ 对齐 classifier 独立 top1 0.4681（旧版在线 0.277 vs 独立 0.532 的 gap 消除）。
4. case 130（原判错 IG02/network loss）现判对 Tomcat02/CPU fault，posterior cpu=0.433 为 top1。

### 10.6 注意事项与下一步

1. **train split type acc=1.0**：train case 自匹配 knowledge cluster 存在自匹配泄漏（test case 不在 train knowledge 中，test 结果真实）。若做 leave-one-case-out 训练可降低过拟合，可能进一步提升 test service@1（当前 0.3830 仍低于 label-only 0.5319）。
2. **不要回退 §10.3 之前的 candidate-clf artifact**：它证明了 skew 的存在，保留作为对照。
3. 下一阶段仍可推进 §6.1 的 trace/log/component role 数据层适配，但应优先做 leave-one-case-out 训练验证 test service@1 上限，再判断 component ranking 是否仍是瓶颈。
4. Eadro 迁移（§6.3）同样需要用 `--knowledge-json --candidate-space-alignment` 训练 classifier，否则会重现同样的 skew。
