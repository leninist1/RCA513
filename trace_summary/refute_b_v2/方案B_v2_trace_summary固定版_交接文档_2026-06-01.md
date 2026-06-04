# 方案 B v2 trace-summary 固定版交接文档

更新时间：2026-06-01

服务器目录：

```text
/home/yan/workspace/projects/rca-bank-adaptive/refute_b_v2
```

本交接包固定的技术路线是：

```text
metric/log 按 query window 读取
+ trace_span.csv 离线构建 deterministic trace summary
+ runtime 读取 trace summary JSON
+ JointAnswerSelector
+ optional DeepSeek LLM tool-loop arbitration
+ checkpoint/resume
```

注意：trace-summary 不是 LLM 总结，而是离线 Python 程序从原始 trace span 中确定性提取结构化特征。

---

## 1. 当前固定结论

当前最优版本是：

```text
DeepSeek + metric/log + trace-summary
```

不是：

```text
DeepSeek + metric/log + runtime raw trace full-window
```

原因是 raw trace full-window 已完整跑完 136 条，结果没有超过 trace-summary 版本，但运行成本明显更高。

---

## 2. 当前最优结果

最优输出文件：

```text
logs/joint_full136_deepseek_trace_predictions.csv
logs/joint_full136_deepseek_trace_debug.json
logs/joint_full136_deepseek_trace_checkpoint.jsonl
logs/joint_full136_deepseek_trace_field_diag.json
logs/joint_full136_deepseek_trace_reason_conditional.json
```

完整 136 条指标：

| 指标 | trace-summary + DeepSeek |
|---|---:|
| official strict | 10.29% |
| official partial | 20.59% |
| reason-cond strict | 16.18% |
| reason-cond partial | 27.21% |
| time hit | 15.05% |
| component hit | 36.14% |
| reason hit | 23.26% |
| reason-cond time hit | 37.63% |

字段命中明细：

```text
time:      14 / 93 = 15.05%
component: 30 / 83 = 36.14%
reason:    20 / 86 = 23.26%
```

---

## 3. 与旧 no-LLM 证据矩阵版本对比

旧版本文件：

```text
logs/openrca_query_v2_predictions.csv
logs/field_hit_diag_notrace.json
logs/calibration_report_136_notrace.json
logs/lodo_calibration_136_notrace.json
```

| 指标 | 旧 no-LLM | 当前最优 trace-summary + DeepSeek | 变化 |
|---|---:|---:|---:|
| official strict | 8.82% | 10.29% | +1.47 pp |
| official partial | 18.44% | 20.59% | +2.14 pp |
| reason-cond strict | 13.97% | 16.18% | +2.21 pp |
| reason-cond partial | 24.63% | 27.21% | +2.57 pp |
| time hit | 7.53% | 15.05% | +7.53 pp |
| component hit | 30.12% | 36.14% | +6.02 pp |
| reason hit | 27.91% | 23.26% | -4.65 pp |

结论：

```text
当前版本相对旧 no-LLM 版本有小幅整体提升，主要来自 time 和 component。
代价是 reason 判断下降，尤其 network/memory 等细粒度 reason 仍容易混。
```

旧版本真正强项仍然是 confidence 自知能力：

```text
LODO high confidence:
  n = 29
  top1 = 26
  top1_rate = 89.66%

Forward last3 holdout high confidence:
  n = 17
  top1 = 14
  top1_rate = 82.35%
  coverage = 39.53%
```

---

## 4. 与 raw trace full-window 对比

raw trace full-window 已完整跑完 136 条。

输出文件：

```text
logs/joint_full136_deepseek_rawtrace_fullwindow_predictions.csv
logs/joint_full136_deepseek_rawtrace_fullwindow_debug.json
logs/joint_full136_deepseek_rawtrace_fullwindow_checkpoint.jsonl
logs/joint_full136_deepseek_rawtrace_fullwindow_field_diag.json
logs/joint_full136_deepseek_rawtrace_fullwindow_reason_conditional.json
```

| 指标 | trace-summary | raw trace full-window |
|---|---:|---:|
| official strict | 10.29% | 8.82% |
| official partial | 20.59% | 19.85% |
| reason-cond strict | 16.18% | 14.71% |
| reason-cond partial | 27.21% | 26.47% |
| time hit | 15.05% | 13.98% |
| component hit | 36.14% | 36.14% |
| reason hit | 23.26% | 23.26% |

结论：

```text
runtime 直接读取 full-window 原始 trace 没有超过 trace-summary。
它打平 component/reason，但 time 和 strict/partial 略低，运行成本更高。
因此后续主线应保留 trace-summary，而不是把 raw trace 直接塞进 runtime。
```

---

## 5. trace-summary 是怎么来的

相关文件：

```text
eval/build_query_trace_summaries.py
eval/build_bank_trace_summaries.py
refute_b_v2/trace_summary.py
refute_b_v2/trace_propagation.py
```

当前 query trace summary 目录：

```text
trace_summaries/openrca_queries_full_batch18
```

summary 内容包括：

```text
service duration stats
slow_edges
dropped_edges
edge count drop
duration p95 slow ratio
first anomalous service
upstream_source / middle_propagator / downstream_sink
propagation path
```

这是一种鲁棒性设计：

```text
运行时不直接处理巨大的原始 trace_span.csv。
先离线把 raw span 压缩成 RCA 相关的传播结构证据。
runtime 只读小 JSON summary，成本低、噪声少、可解释。
```

---

## 6. 当前主运行命令

环境变量：

```bash
source /home/yan/.rca_scheme_b_llm.env
```

固定版运行命令：

```bash
cd /home/yan/workspace/projects/rca-bank-adaptive/refute_b_v2
PYTHONPATH=..:. python3 eval/run_openrca_query_v2.py \
  --modalities metric,log,trace \
  --query-trace-summary-dir trace_summaries/openrca_queries_full_batch18 \
  --use-llm \
  --llm-provider deepseek \
  --checkpoint-jsonl logs/joint_full136_deepseek_trace_checkpoint.jsonl \
  --resume-checkpoint \
  --llm-error-policy fallback \
  --out logs/joint_full136_deepseek_trace_predictions.csv \
  --debug-json logs/joint_full136_deepseek_trace_debug.json
```

不要无 checkpoint 跑 LLM 全量。

---

## 7. Evaluator 命令

字段级诊断：

```bash
PYTHONPATH=..:. python3 eval/field_hit_diagnostics.py \
  --pred logs/joint_full136_deepseek_trace_predictions.csv \
  --query /home/yan/workspace/data/openrca/Bank/query.csv \
  --out logs/joint_full136_deepseek_trace_field_diag.json
```

reason-conditional tolerance 诊断：

```bash
PYTHONPATH=..:. python3 eval/reason_conditional_tolerance_eval.py \
  --pred logs/joint_full136_deepseek_trace_predictions.csv \
  --query /home/yan/workspace/data/openrca/Bank/query.csv \
  --record /home/yan/workspace/data/openrca/Bank/record.csv \
  --out logs/joint_full136_deepseek_trace_reason_conditional.json
```

---

## 8. 关键代码索引

主装配：

```text
eval/run_openrca_query_v2.py
refute_b_v2/joint_answer_selector.py
```

证据与规则：

```text
refute_b_v2/rules.py
refute_b_v2/rule_engine.py
refute_b_v2/default_rules.py
refute_b_v2/evidence_adapter.py
refute_b_v2/evidence_matrix.py
refute_b_v2/iterative_refutation.py
knowledge/refutation_rules_v2.json
```

候选、时间、reason：

```text
refute_b_v2/candidate_generation.py
refute_b_v2/background_suppression.py
refute_b_v2/time_anchor.py
refute_b_v2/reason_competition.py
```

trace：

```text
refute_b_v2/trace_summary.py
refute_b_v2/trace_propagation.py
eval/build_query_trace_summaries.py
eval/build_bank_trace_summaries.py
```

LLM：

```text
refute_b_v2/llm_clients.py
refute_b_v2/llm_tool_loop.py
refute_b_v2/llm_arbitration.py
```

confidence：

```text
refute_b_v2/confidence_semantics.py
refute_b_v2/confidence_calibration.py
eval/calibration_report.py
eval/leave_one_date_out_calibration.py
eval/cross_date_holdout_calibration.py
```

---

## 9. 后续优化建议

### 9.1 第一优先级：恢复 reason hit

当前 reason hit 从旧版 27.91% 降到 23.26%。建议优先做：

```text
case-level diff:
  旧版 reason 对、当前 reason 错
  旧版 reason 错、当前 reason 对

按 reason group 汇总：
  network latency vs packet loss
  memory vs JVM OOM
  CPU vs JVM CPU
  disk I/O vs filesystem
```

推荐新增：

```text
eval/reason_regression_report.py
```

### 9.2 第二优先级：改 reason competition

当前 `reason_competition.py` 仍是 expert_prior_v0。建议：

```text
在 component 已高置信时，reason 选择更多依赖 component 内部 evidence ranking。
增加 pairwise discriminator：
  latency vs packet loss
  memory pressure vs JVM OOM
  disk I/O vs filesystem
```

### 9.3 第三优先级：改 trace summary 特征，而不是 runtime raw trace

建议增强 summary：

```text
保留 onset time，而不仅是 strongest spike
为 propagation/progressive case 生成 time interval candidate
区分 latency 与 packet loss 的 trace-derived 特征
保留 edge stability / edge disappearance 的原因归因信号
```

### 9.4 第四优先级：校准 joint_score

当前 `_joint_score` 仍是启发式。建议：

```text
用 leave-one-date-out 学习或估计权重
对 background relative_score 做 clipping / quantile calibration
固定 time/component 不退步后，再调全局 joint score
```

### 9.5 第五优先级：保留 confidence 作为研究卖点

不要只报 official strict。论文/报告建议双线评价：

```text
公共对齐线：
  official strict
  official partial

研究解释线：
  component hit
  reason hit
  time hit
  reason-conditional tolerance
  confidence high bucket Top1 / coverage
  data blind spot
  trace-summary vs raw-trace ablation
```

---

## 10. 当前可声明与不可声明

可以声明：

```text
方案 B v2 的工程骨架与主要算法模块已经完整装配。
trace-summary 是当前最优且更鲁棒的 trace 接入方式。
DeepSeek + trace-summary 相比旧 no-LLM 版本在 official 三元组上有小幅提升。
confidence 分层仍然是方案 B 当前最强研究卖点。
```

不应声明：

```text
方案 B 已全面超过 OpenRCA archive baseline。
time detection 已解决。
LLM 是主要性能来源。
runtime raw trace full-window 优于 trace-summary。
跨数据集迁移已经验证。
```

