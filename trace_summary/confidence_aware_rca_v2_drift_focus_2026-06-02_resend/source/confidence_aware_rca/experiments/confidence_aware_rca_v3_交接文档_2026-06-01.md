# Confidence-Aware RCA v3 交接文档

日期：2026-06-01

项目服务器：

```text
yan@8.222.254.207
/home/yan/workspace/projects/rca-bank-adaptive/refute_b_v2
```

本地镜像：

```text
C:\Users\echo_\Documents\Codex\2026-05-30\new-chat-2\work\refute_b_v2
```

## 1. 当前路线定位

当前路线不是继续追求全量 RCA strict 准确率，而是做：

```text
RCA Confidence / Self-Knowledge / Selective RCA
```

核心问题：

```text
RCA 系统能不能知道自己什么时候比较可靠？
```

因此本路线分成两层：

1. RCA 预测底座：`refute_b_v2_d32`
2. 可信度选择层：`confidence_aware_rca`

可信度层不改原始 RCA 答案，只判断哪些 case 可以自动采纳，哪些 case 应该 abstain / 转人工。

## 2. RCA 预测底座

当前底座是：

```text
Bank 136 cases
严格 LODO
trace-summary
d32 time-vote
```

核心代码：

```text
refute_b_v2_d32/
eval/run_openrca_d32_lodo.py
eval/build_d32_knowledge.py
```

核心输出：

```text
logs/d32_lodo_timevote_trace_predictions.csv
logs/d32_lodo_timevote_trace_debug.json
logs/d32_lodo_timevote_trace_official_case_eval.json
logs/d32_lodo_timevote_trace_field_diag.json
```

全量 RCA 指标：

| 指标 | 当前 Bank LODO |
|---|---:|
| official strict | 16 / 136 = 11.76% |
| official partial | 44 / 136 = 32.35% |
| time hit | 11.83% |
| component hit | 30.12% |
| reason hit | 27.91% |

注意：

- 这组结果是 Bank-only 136 cases。
- OpenRCA 官方论文结果是全量 335 cases，不能直接宣称全量超过 SOTA。
- 当前全量 strict 仍然低，不适合把论文主张写成 full automation RCA。

## 3. Confidence-Aware 模块

代码目录：

```text
confidence_aware_rca/
```

核心文件：

```text
confidence_aware_rca/feature_extractor.py
confidence_aware_rca/selective_eval.py
confidence_aware_rca/forward_holdout_eval.py
confidence_aware_rca/analyze_selective_results.py
```

测试文件：

```text
tests/test_confidence_aware_rca.py
```

当前测试结果：

```text
PYTHONPATH=..:. python3 -m pytest tests -q
51 passed in 2.32s
```

## 4. 重要修正：v1 结果废弃

最早的 confidence v1 结果看起来较好，但不能作为主证据。

原因：

```text
feature matrix 中误放入了诊断字段，例如：
*_hit_count
*_hit_any_label
*_required_total
```

这些字段与标签高度相关，属于 label-like leakage。

当前 v2/v3 已修正：

```text
selective_eval.py 会排除：
*_label
*_required
*_required_total
*_hit_count
```

交接时不要引用旧文件作为主结果：

```text
confidence_aware_rca/experiments/d32_timevote_selective_lodo_logistic.json
confidence_aware_rca/experiments/d32_timevote_selective_forward_last3.json
```

主结果应使用 `*_v2` 或 `*_v3` 文件。

## 5. v2 Core 结果

主文件：

```text
confidence_aware_rca/experiments/d32_timevote_confidence_features_v2.csv
confidence_aware_rca/experiments/d32_timevote_selective_lodo_logistic_core_v2.json
confidence_aware_rca/experiments/d32_timevote_selective_forward_last3_core_v2.json
confidence_aware_rca/experiments/selective_summary_logistic_core_v2.md
```

核心结论：

| target | LODO coverage@90 | LODO acc@90 | Forward coverage@90 | Forward acc@90 |
|---|---:|---:|---:|---:|
| strict | 19.85% | 14.81% | 34.88% | 20.00% |
| partial | 30.15% | 58.54% | 32.56% | 57.14% |
| component_any | 39.44% | 53.57% | 52.38% | 54.55% |
| reason_any | 30.77% | 50.00% | 14.29% | 50.00% |
| actionable_any | 15.49% | 54.55% | 23.81% | 60.00% |
| time_any | 16.67% | 30.77% | 18.52% | 60.00% |

当前最稳主结果：

```text
official partial selective RCA
LODO:    30.15% coverage, 58.54% selective accuracy
Forward: 32.56% coverage, 57.14% selective accuracy
```

解释：

- 全量 official partial 是 32.35%。
- confidence 模块能筛出约三成高置信 case。
- 在这些 case 上 official partial 提升到约 57%-59%。

这是目前最适合作为论文主线的结果。

## 6. 为什么 strict 不成立

全量 strict：

```text
16 / 136 = 11.76%
```

selective strict：

| 验证方式 | coverage@90 | acc@90 |
|---|---:|---:|
| LODO core | 19.85% | 14.81% |
| Forward core | 34.88% | 20.00% |

如果 strict self-knowledge 成立，应该看到高置信 strict 子集达到 50%、70% 或更高。

当前只有 14.81%-20.00%，说明系统还不能可靠判断完整三元组什么时候全对。

原因：

```text
strict = time + component + reason 同时正确
```

任何一个字段错，strict 就失败。当前 reason/time/component 的误差在同一 case 上会叠加，因此 strict 很难成立。

结论：

```text
不要把当前路线写成 high-confidence strict RCA。
应该写成 selective partial / actionable RCA。
```

## 7. Target-Specific Head 尝试

目标：

不同 target 使用不同 confidence 特征。

设计：

| head | 主要特征 |
|---|---|
| time* | time anchor score, vote count, rejected votes, modality votes |
| component* | component evidence prior, cluster similarity, component entropy |
| reason* | reason prior, mined rule validation, reason-bucket entropy |
| partial/actionable* | 混合特征 |

文件：

```text
confidence_aware_rca/experiments/d32_timevote_selective_lodo_logistic_target_specific_v2.json
confidence_aware_rca/experiments/d32_timevote_selective_forward_last3_target_specific_v2.json
confidence_aware_rca/experiments/selective_summary_logistic_target_specific_v2.md
```

结果：

| target | core LODO acc@90 | target-specific LODO acc@90 | core Forward acc@90 | target-specific Forward acc@90 |
|---|---:|---:|---:|---:|
| partial | 58.54% | 58.54% | 57.14% | 57.14% |
| time_any | 30.77% | 50.00% | 60.00% | 60.00% |
| component_any | 53.57% | 68.75% | 54.55% | 44.44% |
| reason_any | 50.00% | 44.00% | 50.00% | 50.00% |
| actionable_any | 54.55% | 54.55% | 60.00% | 60.00% |

结论：

- target-specific 在 LODO 上对 `time_any` 和 `component_any` 有提升。
- 但提升不能稳定迁移到 forward last3。
- 尤其 `component_any` forward 从 54.55% 掉到 44.44%。
- 不能直接用 target-specific 全面替代 core。

## 8. Auto Selector v2

目标：

避免手工挑 `core` 或 `target_specific`。

协议：

```text
外层：LODO 或 forward last3
内层：只在训练日期内部做 inner LODO
候选：core, target_specific
选择指标：inner LODO AURC，再用 selective utility 打破平局
```

文件：

```text
confidence_aware_rca/experiments/d32_timevote_selective_lodo_logistic_auto_v2.json
confidence_aware_rca/experiments/d32_timevote_selective_forward_last3_auto_v2.json
confidence_aware_rca/experiments/selective_summary_logistic_auto_v2.md
```

主结果：

| target | auto LODO acc@90 | auto Forward acc@90 |
|---|---:|---:|
| partial | 58.54% | 57.14% |
| time_any | 36.36% | 60.00% |
| component_any | 53.57% | 54.55% |
| reason_any | 50.00% | 50.00% |
| actionable_any | 54.55% | 60.00% |

选择分布：

| target | LODO 9 dates 内选择情况 |
|---|---|
| strict | core:9 |
| partial | core:9 |
| time_any | core:5, target_specific:4 |
| component_any | core:9 |
| reason_any | core:9 |
| component_reason_pair_any | core:7, target_specific:2 |
| actionable_any | core:9 |

结论：

- AURC-auto 大多数时候选回 core。
- 这是合理的，说明 target-specific 的 LODO 局部提升不够稳定。
- auto/core 是当前最稳主线。

## 9. Threshold-Transfer Selector v3

最新尝试。

目标：

把 selector 的目标从“排序质量”改成“阈值迁移后的高置信可靠性”。

新增逻辑：

```text
--selection-objective threshold_transfer
--selection-min-coverage 0.2
--selection-z 1.64
```

选择依据：

1. inner validation coverage 是否达到下限
2. Wilson lower bound
3. selective accuracy
4. coverage
5. AURC 作为辅助

代码位置：

```text
confidence_aware_rca/selective_eval.py
confidence_aware_rca/forward_holdout_eval.py
```

新增函数：

```text
wilson_lower_bound()
select_candidate_report()
inner_candidate_report()
select_feature_set()
```

文件：

```text
confidence_aware_rca/experiments/d32_timevote_selective_lodo_logistic_threshold_transfer_v3.json
confidence_aware_rca/experiments/d32_timevote_selective_forward_last3_threshold_transfer_v3.json
confidence_aware_rca/experiments/selective_summary_logistic_threshold_transfer_v3.md
```

主结果：

| target | v3 LODO acc@90 | v3 Forward acc@90 |
|---|---:|---:|
| partial | 58.54% | 57.14% |
| time_any | 36.36% | 60.00% |
| component_any | 53.85% | 44.44% |
| reason_any | 44.00% | 50.00% |
| actionable_any | 54.55% | 60.00% |

v3 LODO 选择分布：

| target | LODO 9 dates 内选择情况 |
|---|---|
| strict | target_specific:9 |
| partial | target_specific:9 |
| time_any | core:3, target_specific:6 |
| component_any | core:5, target_specific:4 |
| reason_any | core:6, target_specific:3 |
| component_reason_pair_any | core:5, target_specific:4 |
| actionable_any | target_specific:9 |

v3 结论：

- v3 保住了 partial 和 actionable_any。
- v3 没有超过 core/auto 主结果。
- v3 更愿意选择 target_specific，但这种选择不能稳定迁移到 future dates。
- component_any forward 从 54.55% 掉到 44.44%。

这是一条有价值的负结果：

```text
仅靠训练内部 Wilson lower-bound threshold selector 不足以解决 date drift。
下一步需要显式建模 date drift / per-date stability。
```

## 10. 推荐论文口径

不推荐：

```text
我们实现了高准确率 OpenRCA strict RCA。
```

推荐：

```text
Under official OpenRCA evaluation, the base Bank-only RCA system achieves
11.76% strict and 32.35% partial accuracy. Rather than pursuing full automation,
we study selective RCA: whether the RCA system can identify cases where its
output is likely reliable. The confidence-aware model identifies about 30%
high-confidence cases where official partial accuracy improves to 58.54% under
LODO and 57.14% under forward holdout.
```

中文口径：

```text
全量 strict 仍然较低，但系统具备一定 self-knowledge：
它可以识别约三成高置信 case，使官方 partial 从 32.35% 提升到约 57%-59%。
这比盲目追求全自动 RCA 更符合工业场景。
```

## 11. 下一步建议

优先级 P0：

```text
date-drift-aware selector
```

当前 v3 的问题是：inner validation 可靠，不代表 forward 可靠。

建议 selector 加入：

1. per-date worst-case selective accuracy
2. per-date Wilson lower bound 的最小值
3. 日期间方差惩罚
4. 如果某个 head 在多个 inner dates 上失效，直接降权
5. 不只比较平均 AURC / 平均 lower bound

候选目标函数：

```text
score =
  min_date_wilson_lower
  + 0.3 * mean_wilson_lower
  + 0.1 * coverage
  - 0.5 * date_variance
```

优先级 P1：

```text
主 target 聚焦 official partial 和 actionable_any
```

原因：

- `partial` 与官方指标直接挂钩。
- `actionable_any` 工业意义强。
- strict 目前不成立，建议作为失败边界分析。

优先级 P2：

```text
field-conditional confidence
```

不要只预测 case-level confidence，可拆成：

```text
P(time reliable)
P(component reliable)
P(reason reliable)
P(partial reliable)
P(actionable reliable)
```

然后解释为什么：

```text
partial 成立，但 strict 不成立。
```

优先级 P3：

```text
reason evidence 改造
```

reason 仍是重要瓶颈，可继续：

1. network / memory / cpu / configuration reason-bucket classifier
2. reason-bucket 反证特征
3. 跨日期稳定 mined rules
4. trace-summary 与 metric shape 的显式映射

## 12. 复现实验命令

进入项目：

```bash
cd /home/yan/workspace/projects/rca-bank-adaptive/refute_b_v2
```

测试：

```bash
PYTHONPATH=..:. python3 -m pytest tests -q
```

v2 core LODO：

```bash
PYTHONPATH=..:. python3 confidence_aware_rca/selective_eval.py \
  --features confidence_aware_rca/experiments/d32_timevote_confidence_features_v2.csv \
  --out confidence_aware_rca/experiments/d32_timevote_selective_lodo_logistic_core_v2.json \
  --model logistic \
  --risk-mode empirical \
  --feature-set core
```

v2 core forward：

```bash
PYTHONPATH=..:. python3 confidence_aware_rca/forward_holdout_eval.py \
  --features confidence_aware_rca/experiments/d32_timevote_confidence_features_v2.csv \
  --out confidence_aware_rca/experiments/d32_timevote_selective_forward_last3_core_v2.json \
  --model logistic \
  --risk-mode empirical \
  --feature-set core
```

v2 auto：

```bash
PYTHONPATH=..:. python3 confidence_aware_rca/selective_eval.py \
  --features confidence_aware_rca/experiments/d32_timevote_confidence_features_v2.csv \
  --out confidence_aware_rca/experiments/d32_timevote_selective_lodo_logistic_auto_v2.json \
  --model logistic \
  --risk-mode empirical \
  --feature-set auto \
  --auto-candidates core,target_specific \
  --selection-target-accuracy 0.9
```

v3 threshold-transfer：

```bash
PYTHONPATH=..:. python3 confidence_aware_rca/selective_eval.py \
  --features confidence_aware_rca/experiments/d32_timevote_confidence_features_v2.csv \
  --out confidence_aware_rca/experiments/d32_timevote_selective_lodo_logistic_threshold_transfer_v3.json \
  --model logistic \
  --risk-mode empirical \
  --feature-set auto \
  --auto-candidates core,target_specific \
  --selection-target-accuracy 0.9 \
  --selection-objective threshold_transfer \
  --selection-min-coverage 0.2 \
  --selection-z 1.64
```

生成摘要：

```bash
PYTHONPATH=..:. python3 confidence_aware_rca/analyze_selective_results.py \
  --lodo confidence_aware_rca/experiments/d32_timevote_selective_lodo_logistic_threshold_transfer_v3.json \
  --forward confidence_aware_rca/experiments/d32_timevote_selective_forward_last3_threshold_transfer_v3.json \
  --out confidence_aware_rca/experiments/selective_summary_logistic_threshold_transfer_v3.md
```

## 13. 当前最重要文件清单

RCA 底座：

```text
refute_b_v2_d32/
eval/run_openrca_d32_lodo.py
logs/d32_lodo_timevote_trace_predictions.csv
logs/d32_lodo_timevote_trace_debug.json
logs/d32_lodo_timevote_trace_official_case_eval.json
logs/d32_lodo_timevote_trace_field_diag.json
```

Confidence 路线：

```text
confidence_aware_rca/
confidence_aware_rca/experiments/d32_timevote_confidence_features_v2.csv
confidence_aware_rca/experiments/selective_summary_logistic_core_v2.md
confidence_aware_rca/experiments/selective_summary_logistic_auto_v2.md
confidence_aware_rca/experiments/selective_summary_logistic_threshold_transfer_v3.md
```

测试：

```text
tests/test_confidence_aware_rca.py
```

## 14. 最终交接结论

当前最稳结论：

```text
official partial selective RCA 成立。
strict selective RCA 不成立。
target-specific head 有局部价值，但不能稳定 forward。
threshold-transfer v3 是有用负结果，说明还需要 date-drift-aware selector。
```

下一位接手时建议不要再从 strict 开始硬调。

最合理的下一步：

```text
围绕 official partial / actionable_any 做 date-drift-aware selective RCA。
```
