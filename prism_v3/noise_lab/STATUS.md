# Noise Lab — 项目状态与下一步路线图

> 写于 2026-05-25。这份文档面向「下一个接手这个项目的我自己」。读完应当能立刻知道：当前最强 baseline 是什么，哪些路走过了，哪些路准备走，每条路的预期收益和已知陷阱。

---

## 0. 2026-05-25 接手后更新

### LLM rerank v5/v6 结论：暂不值得全量

本轮先按“立即下一步”跑了 Bank 20q LLM smoke。

| 版本 | 关键变化 | top1 | top3 | top5 | avg_rank |
|---|---|---:|---:|---:|---:|
| fullspec5l 20q baseline | 无 LLM | 5 | 8 | 12 | 7.75 |
| llm_v4_bank20 | 旧最好 LLM smoke | 2 | 10 | 11 | 8.00 |
| llm_v5_bank20 | 40% confidence gate | 0 | 8 | 11 | 8.25 |
| llm_v6_bank20 | v5 + 方向感知 chain graph_score | 2 | 8 | 10 | 8.15 |

实现改动：
- `noise_lab/runner.py` 末尾重复 `main()` 已删除，否则模块执行会跑两遍实验。
- `noise_lab/causal_reranker.py` 的 chain graph validation 改成方向感知：图是 `caller -> callee`，链是 `root -> effect`，因此 `effect -> root` 给强证据，`root -> effect` 只给弱证据。

结论：
- v5 的 40% margin gate 只 skip 了 1/20，保护不了原本 top1 正确样本；Bank 20q 里正确 rank-1 的 margin 只有 7.2%–17.5%。
- v6 能把 top1 从 v5 的 0 拉回 2，但仍不如 baseline，也不如 v4 的 top3。
- 不要直接跑 Bank 134q LLM 全量；除非先做更强的 evidence gate / prompt 上下文 / promotion-only 合并策略。

### Telecom 关键发现：trace schema 原来读错实体列

Raw Telecom `trace_span.csv` 中：
- `serviceName` 在抽样的所有日期前 100k 行都是空。
- `cmdb_id` 才是 trace 主体实体，值包含 `docker_*`，部分日期也包含 `os_021/os_022`。
- `dsName` 包含 `db_003/db_007/db_009`，但直接作为 peer trace 信号会严重误导。

已保留的有效改动：
- `data/schema.py`：Telecom `trace_span.entity_col` 从 `serviceName` 改成 `cmdb_id`。
- `data/loader.py`：trace entity 过滤空字符串 / `nan` / `none` / `null`，避免生成假对象。

已尝试但撤销的方向：
- 把 `dsName` 作为 `peer_entity` 计入 `db_*` trace profile，并加 `cmdb_id -> dsName` 边。
- 20q 结果变差到 top1=0、top3=4、top5=5、avg_rank=19.25；原因是 `db_003` 等常见依赖被过度抬高，污染 docker/os 实例排序。

### Telecom 新结果：必须区分“层级聚合口径”和“实例严格口径”

旧结果 `noise_lab_fullspec5l_gtfix_Telecom_20260525_025059.json`：

| 口径 | total | top1 | top3 | top5 | avg_rank |
|---|---:|---:|---:|---:|---:|
| 旧 schema，GT prefix/broad | 51 | 0 | 0 | 0 | 7.6471 |

修正 trace `cmdb_id` 后，如果使用旧的层级聚合 / broad GT 口径，会得到很乐观的结果：

| 文件 | total | top1 | top3 | top5 | avg_rank |
|---|---:|---:|---:|---:|---:|
| `noise_lab_fullspec5l_tracecmdb_full_Telecom_20260525_123236.json` | 51 | 20 | 35 | 47 | 2.8431 |

但当前代码已经对 Telecom 保留实例后缀（`docker_003/db_007/os_018` 不再折叠成 `docker/db/os`），严格实例口径才是可信结果：

| 文件 | total | top1 | top3 | top5 | avg_rank |
|---|---:|---:|---:|---:|---:|
| `noise_lab_fullspec5l_tracecmdb_strict_full_Telecom_20260525_132943.json` | 51 | 2 | 7 | 10 | 20.2353 |

严格实例分布：

| GT prefix | n | top1 | top3 | top5 | avg_rank |
|---|---:|---:|---:|---:|---:|
| db | 12 | 0 | 0 | 0 | 14.75 |
| docker | 19 | 0 | 1 | 4 | 16.16 |
| os | 20 | 2 | 6 | 6 | 27.40 |

当前解释：
- `cmdb_id` 修复证明 Telecom 不是完全没救，trace 里确实有 infra 实体。
- 但实例级 RCA 仍缺少 service/docker/db/os 的精确映射，`os_021/os_022` 和少数 `docker_*` 仍会霸榜。
- 论文/报告里不要引用层级聚合 92.16% top5 作为最终 Telecom 成绩；它只能说明“层级定位”变好了，不代表实例级定位成功。

### 2026-05-25 继续更新：Telecom metric 时间单位修复，但不能默认启用

本轮发现 Telecom `metric_node/service/container/middleware` 的 `timestamp` 原始单位也是毫秒，之前 schema 没有 `time_unit: "millis"`，导致 metric 全部落在故障窗口外，GT 的 `metric_score` 基本为 0。

已保留的代码改动：
- `data/schema.py`：Telecom 四类 key-value metric 补 `time_unit: "millis"`。
- `noise_lab/runner.py`：Telecom metric scoring 改成显式 opt-in；只有 variant 包含 `metricms` 才启用修复后的 metric 时间。默认非 `metricms` variant 会保留 metric 实体但屏蔽 metric 时间戳，避免污染当前 trace-only baseline。

实验结果：

| 版本 | total | top1 | top3 | top5 | avg_rank | 结论 |
|---|---:|---:|---:|---:|---:|---|
| `fullspec5l_metricms_instance_20q` | 20 | 1 | 1 | 1 | 16.35 | 直接启用 metric 更差 |
| `fullspec5l_metricms_robust_instance_20q` | 20 | 0 | 1 | 1 | 16.10 | 通用 robust scorer 仍无效，已撤回 |
| `fullspec5l_metricms_robusttime_instance_5q` | 5 | 0 | 0 | 0 | 17.60 | trace earliest 改动更差，已撤回 |
| `fullspec5l_tracecmdb_guard_5q` | 5 | 0 | 0 | 0 | 9.20 | 默认 guard 基本回到 trace-only 行为 |

诊断结论：
- 时间单位 bug 是真实的，但 generic metric scorer 对 Telecom 不可直接用；大量持久症状和下游 DB/OS 指标会被打成强异常。
- 典型例子：`docker_003` CPU fault 时，`db_007` 的 `Redo_Per_Sec/Physical_Read_Per_Sec`、`os_018/os_021/os_022` 的网络计数也很强，直接启用 metric 会把症状层排到 root 前面。
- 不要跑 `metricms` full；下一步需要 Telecom 专用 metric admission，比如按实体层级、指标族、故障后首个显著变点、历史持久异常清理来筛 metric，而不是使用通用 z-score。

### 当前下一步

1. Bank：LLM rerank 暂停全量，除非先实现 promotion-only / stricter evidence gate；非 LLM 方向优先看孤立图 fallback / Tomcat 层。
2. Market：先修严格实例口径与 node 实体 canonical，再继续做 hub 抑制；不要直接用旧 broad node 成绩。
3. Telecom：暂时搁置。后续要做严格实例级映射和专用 metric admission，而不是再调通用 scorer。

### 2026-05-25 Bank/Market 继续更新

本轮按“先放下 Telecom，集中 Bank + Market”执行。

Bank 20q：

| 版本 | total | top1 | top3 | top5 | avg_rank | 结论 |
|---|---:|---:|---:|---:|---:|---|
| `fullspec5l_bank20_current` | 20 | 5 | 8 | 12 | 7.85 | 当前非 LLM 基线，基本复现旧 20q |
| `fullspec5l_bank20_globalprior` | 20 | 5 | 10 | 10 | 8.15 | top3 增加但 top5/avg 变差，不适合作 Bank 默认 |

Bank 结论：
- 跨 query global prior 会救回少数 Tomcat/MG 样本，但也压低 IG/apache/MG 等真实重复组件。
- 不建议跑 Bank full globalprior；Bank 下一步应做孤立图 fallback 或更保守的 promotion-only LLM，而不是频率先验。

Market 20q（当前严格实例口径）：

| 版本 | total | top1 | top3 | top5 | avg_rank | 结论 |
|---|---:|---:|---:|---:|---:|---|
| `fullspec5l_market20_current` | 20 | 0 | 1 | 1 | 78.40 | 当前严格 node 匹配后，旧 20q top5=5 的乐观结果不再成立 |
| `fullspec5l_market20_globalprior` | 20 | 0 | 2 | 4 | 77.35 | hub 抑制有效，但没有解决 node 实体问题 |
| `fullspec5l_market20_nodefix_current` | 20 | 0 | 1 | 1 | 44.30 | 保留 Market `node-*` 实例后缀，node GT 可排名，avg 大幅改善 |
| `fullspec5l_market20_nodefix_globalprior` | 20 | 0 | 1 | 1 | 42.55 | 组合只小幅改善 avg，没有提升 top5 |

已保留的 Market correctness fix：
- `mace/graph.py`：`build_object_graph` 现在对 `Telecom` 和 `Market` 都保留 instance suffix。否则 raw 里的裸 `node-1/node-2/node-4` 会被 canonical 成 `node`，严格 GT 永远匹配不到。

Market 结论：
- 旧 Market 成绩里 `node-*` 宽匹配确实偏乐观；严格口径下 20q 只有 top5=1。
- nodefix 是正确性修复，avg_rank 从 78.40 降到 44.30，但 service GT 有局部回归，说明还需要区分“host/node root”和“service replica root”的聚合策略。
- globalprior 对 adservice hub 有作用：无 nodefix 时 top5 1→4；但在 nodefix 后没有继续提升 top5。暂不建议直接跑 full。

### 2026-05-25 Bank LTR / 对比学习方向更新

本轮先不做端到端 contrastive/GNN，先实现候选级 learning-to-rank，因为它能直接复用现有 NoiseFieldScorer / structure / delay / beam / subspace / mask 信号，并且可做 query-grouped CV。

新增实验模块：
- `noise_lab/learning_ranker.py`：抽取候选特征，按 `system:component:datetime` 分组做 5-fold `XGBRanker`。
- 支持 `--features-csv` 复用特征表、`blend_*` / `guarded_*` 多策略评估、feature importance、预测 CSV。
- 已加 telemetry LRU cache 和抽取进度日志。无缓存 Bank full 运行 60 分钟未落盘，已停止；cached full 只需读取 9 个日期，`cache_hits=125/cache_misses=9`。

Bank full 134q 结果：

| 版本 | total | top1 | top3 | top5 | avg_rank | 回归情况 | 结论 |
|---|---:|---:|---:|---:|---:|---|---|
| base from LTR features | 134 | 15 | 33 | 56 | 8.664 | - | 与 `fullspec5l` 接近，作为本轮对照 |
| `xgbrank_all/ltr_full` | 134 | 20 | 50 | 77 | 5.396 | improved 78 / regressed 42 | top3/top5/avg 最强，适合论文主线候选 |
| `xgbrank_all/blend_a1` | 134 | 16 | 44 | 74 | 6.119 | improved 81 / regressed 13 | 更安全，回归少很多 |
| `xgbrank_all/guarded_m0.5` | 134 | 25 | 46 | 67 | 7.291 | improved 28 / regressed 21 | top1 最高，但 top5/avg 损失明显 |
| `noprefix/ltr_full` | 134 | 19 | 52 | 75 | 5.687 | improved 76 / regressed 43 | 去掉组件名前缀仍有效 |
| `noprefix_nobase/ltr_full` | 134 | 20 | 52 | 79 | 5.560 | improved 73 / regressed 50 | 不靠组件名前缀或原始 rank/score，说明信号本身可学习 |

关键产物：
- `results/noise_lab_ltr_bankfull_xgbrank_top100_cached_Bank_20260525_193839.json`
- `results/noise_lab_ltr_features_bankfull_xgbrank_top100_cached_Bank_20260525_193839.csv`
- `results/noise_lab_ltr_scores_bankfull_xgbrank_top100_cached_Bank_20260525_193839.csv`
- `results/noise_lab_ltr_bankfull_reuse_noprefix_nobase_Bank_20260525_193943.json`

Bank LTR 信号结论：
- 最有价值的信号不只是原始 rank。去掉 `prefix_*`、`base_*`、`rank_*` 后仍有大幅提升。
- 高重要度信号包括 `metric_score`、`anomaly_score`、`noise_local_noise`、`noise_root_source_score`、`noise_multi_lead_consistency`、`delay_observer_coverage`、`degree_out/degree_total`、`structure_topological_eccentricity`、`subspace_residual_distinctiveness`、`earliest_offset`、`mask_replaceability_penalty`。
- `noise_local_noise` 单独看像饱和信号，但在 LTR 中很有价值，作用更像非线性 gate：配合 degree/hub、delay、subspace 区分真实局部源和共享症状噪声。
- `prefix_*` 有帮助但不是必要条件；如果要做更普适方法，优先沿 `noprefix_nobase` 特征集推进。

策略结论：
- 若目标是 top3/top5/avg，优先 `ltr_full`，尤其是 `noprefix_nobase/ltr_full` 或 `xgbrank_all/ltr_full`。
- 若目标是减少负迁移，优先 `blend_a1` / `blend_a1.5`，牺牲少量 top5 换显著更少回归。
- 若目标是 top1，`guarded_m0.5` 最强，但它更像“强信号 promotion-only”，不是综合最优。
- 下一步不应直接上端到端 contrastive；更现实的升级是 query-aware pairwise/listwise loss、hard-negative mining、Bank+Market 跨系统 no-prefix 训练，再考虑 graph/contrastive encoder。

本轮后续并行更新：

Learning-to-rank 优化：

| 版本 | total | top1 | top3 | top5 | avg_rank | 结论 |
|---|---:|---:|---:|---:|---:|---|
| `noprefix_nobase/rank:pairwise` | 134 | 20 | 52 | 79 | 5.560 | 原 no-prefix/no-base 强基线 |
| `noprefix_nobase/rank:ndcg` | 134 | 23 | 53 | 75 | 5.649 | top1/top3 更好，适合单答案 PRISM prior |
| `rank:ndcg + hard_negative_window=15` | 134 | 14 | 42 | 68 | 6.112 | 退化，不应采用 |
| `rank:pairwise + hard_negative_window=15` | 134 | 18 | 48 | 69 | 6.157 | 退化，不应采用 |

关键结论：hard-negative 不能简单截 `base_rank<=15`，因为 Bank 里很多 Tomcat/MG 正样本原始 rank 很深，远端候选反而是必要训练上下文。当前更好的泛化候选是 `rank:ndcg + noprefix_nobase`，但若目标是 top5，旧 pairwise 仍更强。

PRISM 接入：
- `prism.py` 增加 `noise_lab_enabled/noise_lab_scores_csv/noise_lab_strategy`，可把 Noise Lab OOF scores 作为 PRISM prior / final blend。
- `main.py` 增加 `--prism-noise-lab`、`--prism-noise-lab-scores`、`--prism-noise-lab-strategy`、`--prism-fast`。
- `--prism-fast` 关闭重型 counterfactual profiles / final discriminator，用于 Bank 全量评估；完整 PRISM loop 在 7q smoke 中超过 5 分钟仍未完成，不适合当前全量 sweep。
- `main.py` 后续会保存 `field_scores`；本次已落盘结果没有该字段，field-level 是事后从 prediction/GT 反算。

Bank 全量整体框架结果（PRISM-fast + Noise Lab OOF `rank:ndcg/noprefix_nobase/ltr_full`）：

| task | 要求 | total | Correct | Partial-only | Correct+Partial |
|---|---|---:|---:|---:|---:|
| task_1 | time | 23 | 19 | 0 | 19 |
| task_2 | reason | 22 | 0 | 0 | 0 |
| task_3 | component | 15 | 1 | 0 | 1 |
| task_4 | time+reason | 18 | 0 | 16 | 16 |
| task_5 | time+component | 18 | 1 | 12 | 13 |
| task_6 | component+reason | 21 | 0 | 3 | 3 |
| task_7 | time+component+reason | 17 | 0 | 13 | 13 |
| **TOTAL** |  | **134** | **21 (15.67%)** | **44 (32.84%)** | **65 (48.51%)** |

产物：
- `results/option_PRISM_20260525_221432.json`
- `results/noise_lab_ltr_bankfull_ndcg_noprefix_nobase_Bank_20260525_212615.json`
- `results/noise_lab_ltr_scores_bankfull_ndcg_noprefix_nobase_Bank_20260525_212615.csv`

解释：
- time 字段在当前框架里基本由 matched record / inject time 直接给出，因此 task_1 和含 time 的 Partial 偏乐观；更能反映 RCA 能力的是 component/reason。
- component 字段当前仅 6/71 正确（8.45%），reason 字段 0/78 正确；PRISM-fast 接入证明流程跑通，但还没有把 Noise Lab 的高 top-k 能力转化成最终 exact answer。
- 下一步应优先修 reason taxonomy / 输出归一化，以及让 PRISM final answer 可以直接选 Noise Lab OOF top candidate 或 listwise top candidate，而不是只把它作为弱 prior。

---

## 1. 当前最强 baseline (fullspec5l)

`fullspec5l` 是把 NoiseFieldScorer + StructuralObjectEncoder + DelayPatternLocalizer + StructuralBeamformer + SourceNoiseSubspaceDecomposer + ReverbSuppressionMask 全部打开后的全谱版本。打分权重在 `noise_lab/runner.py::_rank_objects` 里。

GT 匹配在 `runner.py::_ground_truth_rank`，已加 canonical + prefix 匹配（修了 Telecom/Market 之前 90%+ rank=None 的问题）。

### 三个数据集结果（`fullspec5l` + GT 匹配修复）

| 系统 / 变体                  | total | skip | top1%  | top3%  | top5%  | avg_rank |
|------------------------------|-------|------|--------|--------|--------|----------|
| Bank p21_baseline            | 134   | 2    | 9.70   | 22.39  | 32.84  | 13.40    |
| Bank v2_baseline             | 134   | 2    | 10.45  | 25.37  | **44.03** | 9.34     |
| **Bank fullspec5l**          | 134   | 2    | **11.19** | 25.37  | 39.55  | **8.81** |
| Telecom fullspec5l_gtfix     | 51    | 0    | 0.00   | 0.00   | 0.00   | 7.65     |
| Market fullspec5l_gtfix      | 146   | 2    | 3.42   | 22.60  | 36.30  | 14.47    |

**Bank fullspec5l vs v2**：top1 +0.74pp、top3 持平、top5 **−4.48pp**、avg_rank −0.53（更好）。

整体上 fullspec5l 比 v2 更"稳"（avg_rank 改善），但 v2 在 top5 上仍然是单项最好的 Bank baseline。

### Bank 详细分布（fullspec5l 134q）

| GT          | n  | top1 | top3 | top5 | avg_rank |
|---|---|---|---|---|---|
| MG01        | 19 | 6 | 10 | 12 | 3.95 |
| IG02        | 14 | 2 | 3  | 7  | 5.43 |
| apache02    | 8  | 1 | 3  | 4  | 5.25 |
| MG02        | 18 | 2 | 5  | 8  | 8.72 |
| Mysql02     | 4  | 1 | 1  | 2  | 8.75 |
| IG01        | 4  | 2 | 2  | 2  | 6.75 |
| Redis02     | 8  | 1 | 2  | 3  | 7.00 |
| Tomcat02    | 9  | 0 | 1  | 3  | 14.11 |
| Tomcat03    | 11 | 0 | 1  | 2  | 12.00 |
| Tomcat04    | 9  | 0 | 1  | 1  | 11.22 |
| Tomcat01    | 19 | 0 | 4  | 5  | 13.00 |
| apache01    | 7  | 0 | 0  | 3  | 11.00 |

明显模式：**叶子层（MG、IG、apache02、DB、Redis）做得不错**；**Tomcat 这层全军覆没**——19 个 Tomcat01 没有一个 top1，9 个 Tomcat02 只有 1 个进 top3。

### Market 详细分布（fullspec5l_gtfix 146q）

仅列重点：
- `node-*` 51 个：top3 15、top5 30、avg_rank 6.59 — **但 GT 匹配过宽**（`node-1` canonical=`node`，会匹配 `node-6.*`，数据偏乐观）
- `adservice` 5 个：top1 5、avg_rank 1.00 — adservice 自己就是 dominant 节点
- `emailservice` 7、`recommendationservice2` 6 — top5 全是 0
- `shippingservice` 13、`recommendationservice` 9、`checkoutservice` 9 — avg_rank 13–32

### Telecom 51 query 的"确定性失败"

经过 GT 匹配修复后，所有 query 的 rank 已知，但分布是确定性的：
- `docker_*`（19 个）总是 rank 6
- `db_*`（12 个）总是 rank 8
- `os_*`（20 个）总是 rank 9

avg_rank=7.65 的精确公式：`(19×6 + 12×8 + 20×9) / 51`。

---

## 2. 关键路径 & 已踩过的坑

### 2.1 fullspec5l 的当前打分组合

`runner.py::_rank_objects` 加权（部分摘录，全文看代码）：

| 信号                                          | 权重    |
|---|---|
| structural_score                              | +0.18   |
| subspace.local_residual_source_energy         | +0.16   |
| noise.root_source_score                       | +0.14   |
| delay.source_time_consistency                 | +0.14   |
| beam.beamformed_explanation                   | +0.12   |
| structure.source_likelihood                   | +0.10   |
| **noise.hotspot_bias**                        | **−0.25** |
| structure.symptom_likelihood                  | −0.12   |
| delay.reverse_penalty                         | −0.10   |
| structure.hard_negative_resistance            | −0.10   |
| mask.gated_reverb_penalty                     | −0.10   |

**已经做过的几次微调**：
- 把 `_context_mass(direction="down")` 从 sum 改成 average，避免"高出度自动加分"
- `hotspot_bias` 加了 `is_isolated` 项（孤立节点强惩罚）
- `_topological_eccentricity` 对孤立节点返回 0.0
- `noise_field` 加了 connectivity gate（在 `noise_field.py`）

调更多权重（hotspot −0.25 → −0.20、hard_neg −0.10 → −0.08）的实验做过 trial，**得不偿失**：救回的是个别 IG02 样本，但 Mysql02/MG01 这种叶子的大幅改善又被吃掉。Bank 上 weight tuning 已基本到顶。

### 2.2 GT 匹配修复（`runner.py::_ground_truth_rank`）

修复前：Telecom 90%+ rank=None、Market 75%+ rank=None。修复后所有 query 都能命中。规则：
1. 原始字符串完全相等
2. canonical 形式相等（去掉 `-1`/`_003` 等）
3. **前缀匹配**：候选的 canonical 以 `gt_canonical + .` 或 `gt_canonical + -` 开头

**已知问题**：第 3 条对 Market 的 `node-*` 标签过宽。`node-1` canonical=`node`，会匹配任何 `node-6.something`、`node-2.x`。这让 Market `node-*` 类 GT 的 top3/5 数据偏乐观，但还没改 — 改了之后 Market 总分会掉一点，需要再做一组对照实验看真实损失。

### 2.3 SemanticCausalChainReranker（LLM 重排，刚做完一轮迭代）

文件：`noise_lab/causal_reranker.py`。

思路：让 LLM 在 fullspec5l 给出的 top-15 候选基础上提因果链，**再用图证据反向验证链**（不让 LLM 直接出排名）。每条链按 graph_score（边邻接）+ temporal_score（earliest_ts 单调）+ mechanism_score（reason 兼容矩阵）打验证分；最后每条 elite 链的起点拿正票，下游位置拿症状惩罚。

**Bank 20q 上的迭代历史（DeepSeek-chat，每个版本完整跑一遍）**：

| 版本 | 关键变化 | top1 | top3 | top5 | avg_rank |
|---|---|---|---|---|---|
| fullspec5l (matched 20q baseline) | — | 5 | 8 | 12 | 7.75 |
| llm_v1 | 初版，所有链都投票，position-decay 0.5^pos | 5 | 7 | 7 | 8.85 |
| llm_v2 | 缺证据时 validation=0；加 evidence-coverage gate | 5 | 7 | 10 | 8.20 |
| llm_v3 | 只看 elite 链（top3 cutoff 0.85×best）；下游位置当 symptom | 3 | 9 | 11 | 7.95 |
| llm_v4 | 同时被某链当 head 的候选不再受 effect 惩罚 | 2 | 10 | 11 | 8.00 |
| llm_v5 (尚未跑) | rank-1 vs rank-2 score margin ≥ 40% 时整段 skip | — | — | — | — |

**核心结论**：top3、top5 都能涨（v3/v4 已超过 baseline），但**top1 一直在退**。LLM 很容易"晃动"原本 rank-1 的稳定样本。

**为什么调权重还不够**：
- Bank 不是全孤立图，链验证里 graph_score 几乎都给 1.0、temporal_score 也接近 1.0（因为 earliest_ts 普遍存在）
- `chain_weight=0.25` 已经偏温和，但对 base 信心强、score 差距大的 query 还是会扰动
- v5 的"高 margin 跳过"是对症的方案，**下次首先跑这个**

每次完整 20q 跑大约 9 分钟（DeepSeek API + 5 个并发候选链）。

---

## 3. 三个数据集的瓶颈、根因诊断

| 失败模式                              | 影响系统           | 根因                                                  | 是否能算法解决                          |
|---|---|---|---|
| 全孤立子图归一化噪声放大              | Bank 部分 query    | trace 缺失导致 callees=[]、ts=None；min-max 归一化把 0.001 差异放成 0/1 | 部分（已加 connectivity gate） |
| 观测层 vs GT 层语义错位               | **Telecom 全部**   | telemetry 监控 osb/csf/local.method 应用层；GT 标在 docker/db/os 基础设施层；GT 节点 anomaly=0 | **必须补 infra-level 信号** |
| 持续高 anomaly hub 节点霸榜            | Market（adservice）| adservice 在 146 个 query 几乎都 rank-1；scorer 没有跨 query 的"重复 hub"识别 | **能**（全局先验或 LLM 语义判断） |
| Hub 类 GT 被 hotspot+hard_neg 双罚    | Bank（Tomcat、IG02）| 高出度真根因被同时当噪声和"被多方异常包围"惩罚 | 难，hub 是真根因 vs 假阳的边界本来就模糊 |
| 前缀 GT 匹配对 `node-N` 类标签过宽    | Market             | canonical(`node-1`) = `node` 匹配 `node-6.*`         | **是**，区分 instance suffix vs machine suffix |

### Bank 的"全孤立图"

约 50% 的 Bank query 出现 trace 数据缺失，整张图 9–29 个节点全部 callees=[]。在这类图上：
- propagation/beam/subspace 全模块失效
- 排名几乎纯靠 anomaly 强度 + 归一化随机性
- ServiceTest1 (anom=0.12) 反而能拿到 structural_score=1.0 后被排在 GT (anom=0.91) 上面

这类图占 Bank 错误样本的相当一部分（Tomcat01 19 例中只有 5 个进 top5，多数错的就在这里）。

### Telecom 的"观测鸿沟"

```
Telecom 9-node 图（disconnected）
  应用层（anom=1.0）：osb / csf / fly.remote / local.method
  基础设施层（anom=0.0）：docker / db / os / redis
GT 全在基础设施层 ⟹ 任何基于 anomaly 的排序都把应用层放前面
```

**这不是 NoiseFieldScorer 的设计缺陷**。哪怕用反事实、用结构、用什么花式 scorer，输入信号里 GT 节点根本没有 anomaly 信号，就是排不上去。

破局必须从**数据层**开始：service→host/container 映射 + 亚阈值变点检测（CUSUM、Mann-Whitney），把"看似无异常但有持续小漂移"的 infra 节点挖出来。这是路线图里方向 ③（见下）。

### Market 的"adservice 霸榜"

161-node 连通图，`adservice-{0,1,2}.source.adservice.jaeger-collector` 在大部分 query 都拿 root_source_score=1.0、structural_score=1.0。fullspec5l 没有跨 query 的重复识别能力，所以无差别让 adservice 赢。

---

## 4. 路线图（按预期 ROI 排序）

### ① LLM 因果链重排（已动工，下次接着调）

文件：`noise_lab/causal_reranker.py`。

**还没做完**：
1. **跑 v5（confidence gate）** — 已经写完代码（margin ≥ 40% 时跳过 LLM），等下次 smoke
2. **跑全量 Bank 134q + Telecom + Market** — smoke 通过后做完整对照
3. **加 sub-graph 上下文** — 当前 prompt 只给 candidates 之间的边；可以给每个候选的 1-hop 邻居信息（哪些上游、哪些下游、对应 anomaly）让 LLM reasoning 更有据
4. **多模型对比** — DeepSeek-chat vs Claude Sonnet 4.6 / Opus 4.7（已有 Anthropic key）；看链质量是否随模型升级线性提升

**已知陷阱**：
- 链 head 投票会扰动原本 rank-1 的稳定样本 → v5 confidence gate 是对症方案
- DeepSeek `response_format={"type":"json_object"}` 时 LLMClient 直接返回 dict，不是 string
- API 调用单 query ~3 秒，全量 Bank 134q ≈ 7 分钟（可接受）

### ② 跨 query 全局先验（解决 Market hub 霸榜）

**���路**：扫一遍所有 query，统计每个 canonical object 在 top5 出现的频率 `P_top5(obj)`。推理时 `final_score *= (1 - α·P_top5)`。

**预期收益**：
- Market：`adservice` 的 P_top5 ≈ 1.0，会被压到接近 0；只有 GT 真是 adservice 时才能赢
- Bank：`ServiceTest*` 一类反复在错误样本里夺冠的伪源也会被压
- Telecom：无效（all GTs 都不出现在 top5）

**实现要点**：
- 不能用本次 query 自己的统计（leakage）；需要用 leave-one-out 或离线 baseline 统计
- α 需要调；过大会把真实重复出现的核心服务也吃掉

**参考**：方向二 LLM 增强版本 — 让 LLM 看 5 个 query 的 top1，如果完全不同 reason 但 top1 都是同一节点，直接把它打成"背景假阳"。

### ③ Telecom 基础设施归因层（最大胆，唯一能突破 0% 的路径）

**前置探查**（必须先做）：
1. Telecom telemetry CSV 里的 trace span tag 有没有 `host`、`container`、`pod` 字段？
2. log 里有没有 `host=docker_NNN` 这种字段？
3. 如果没有显式映射，能否从 metric_name 或 entity 命名规律里推断？

**如果能挖到映射**：
- 把应用层异常分（osb anom=1.0）按映射反向分发给 infra 候选
- 加一个对 infra 层 metric 的**亚阈值变点检测**（CUSUM、Mann-Whitney U）
- 应用层 anomaly 信号 + infra 层细粒度变点 = 突破 0%

**如果没有映射**：
- 让 LLM 用世界知识（"OSB 跑在 JVM 上、JVM 跑在 Docker"）做类比推理
- 这是论文价值最大的方向之一，但依赖 LLM 的领域知识泛化

### ④ 反事实扰动验证（解决 cause vs symptom）

**思路**：对 fullspec5l top-K 候选逐个做扰动（把该节点 telemetry 替换为基线），重跑 anomaly 检测，看全图异常质量恢复量 Δ。用 Δ 重排。

**vs 现有 collapse_gain**：现有的是图结构上的 propagate 近似；这里是**全局真实扰动**。

**预期收益**：
- Market 上 emailservice (anom=0.15) 类小信号 GT 突围（扰动它能让 frontend 等下游恢复）
- Bank 上能区分 Tomcat 是真因（扰动后 IG/MG 都恢复）vs 症状（扰动后下游不变）

**成本**：每个 candidate 需要重跑一次 anomaly detector → 单 query 慢 K 倍。要做近似（只对 metric mode 重算，不重算 trace/log）。

### ⑤ Window-aware first-mover detector（解决 Bank Tomcat 困境）

**思路**：故障窗切成 1-2 分钟 slice，对每个候选用相对自身基线做 CUSUM；输出 first-deviation-slice 作为新信号。

**预期收益**：Bank 中"叶子早期小幅变化 vs Tomcat 后期大幅变化"，让早期信号赢。

---

## 5. 工程坑位备忘

### 文件位置
- 算法核心：`/home/dell2/RCA513/yyx/rca513/openrca_meta_controller/noise_lab/`
- 数据加载：`/home/dell2/RCA513/yyx/rca513/openrca_meta_controller/data/loader.py`
- 图构建：`/home/dell2/RCA513/yyx/rca513/openrca_meta_controller/mace/graph.py`
- 结果输出：`/home/dell2/RCA513/yyx/rca513/results/noise_lab_{variant}_{system}_{ts}.json`
- DeepSeek key（在仓库里硬编码）：`/home/dell2/RCA513/yyx/rca513/prism/dev/test_re3ob_on_openrca.py`

### 数据集尺寸（重要！跑全量前心里有数）
- Bank: 1 sub-system × ~22 dates，**完整跑 134q ≈ 25 分钟**
- Telecom: 1 sub-system × 11 dates，**完整跑 51q ≈ 25 分钟**（CSV 加载占大部分）
- Market: 2 sub-systems × 2 dates, **24GB telemetry**，**完整跑 146q ≈ 60 分钟**

### LLM 集成
- `LLMClient` 在 `openrca_meta_controller/utils/llm_client.py`
- 当 `response_format={"type":"json_object"}` 时**返回的是 dict 不是 string**，要先 isinstance 判断
- DeepSeek API 单 call ~3s；Claude 通常更慢但更准

### 跑实验的标准命令

```bash
cd /home/dell2/RCA513/yyx/rca513
# 不带 LLM
python -m openrca_meta_controller.noise_lab.runner --system Bank --variant fullspec5l

# 带 LLM 重排
DEEPSEEK_API_KEY=sk-... python -m openrca_meta_controller.noise_lab.runner \
  --system Bank --variant fullspec5l_llm --llm-rerank --llm-provider deepseek
```

### 跑后立刻做的对照分析

```python
import json, glob
base = json.load(open(sorted(glob.glob('rca513/results/noise_lab_fullspec5l_Bank_*.json'))[-1]))
new  = json.load(open(sorted(glob.glob('rca513/results/noise_lab_<NEW>_Bank_*.json'))[-1]))
def key(q): return q.get('query_id','') + '|' + str(q.get('ground_truth',{}).get('datetime',''))
base_map = {key(q): q.get('gt_rank') for q in base['per_query'] if q.get('status')=='ok'}
new_map  = {key(q): q.get('gt_rank') for q in new['per_query']  if q.get('status')=='ok'}
common = set(base_map) & set(new_map)
imp = sum(1 for k in common if (new_map[k] or 99) < (base_map[k] or 99))
reg = sum(1 for k in common if (new_map[k] or 99) > (base_map[k] or 99))
print(f'matched={len(common)} improved={imp} regressed={reg}')
```

### 别再踩的坑
1. **每次重跑全量都要 25-60 分钟**。先在 20q smoke 上验证再跑全量。
2. **stdout 缓冲**：nohup 后台进程的 print 不会立刻 flush 到日志；用 `python3 -u` 或者直接读 result 文件来判断进度。
3. **Market 24GB telemetry** 加载需要 ~15GB RAM；不要在内存紧张时启动。
4. **`top_candidates` JSON 字段只存 5 个**；想分析 rank>5 的位置必须在 runner 里改，或者重跑保留更多。
5. **Telecom 上不要花时间调 ranking 算法**——已经多次确认是数据层问题，不是算法层。

---

## 6. 立即下一步（按顺序）

1. **跑 v5 LLM smoke（confidence gate）** — 验证能否回收 top1
2. **如果 v5 OK → 全量 Bank 134q LLM 跑** — 出第一组对照
3. **挖 Telecom telemetry tag** — 决定方向 ③ 是否可走
4. **实现方向 ② 全局先验** — 一次性预统计 + 推理时门控；最简单，最快产出对照
5. **方向 ④ 反事实扰动**：实现起来不便宜，但和方向 ① LLM 链能形成天然组合（链假设 → 扰动验证 → LLM 没有的真实效应度量），写论文时是有重量级的章节。
