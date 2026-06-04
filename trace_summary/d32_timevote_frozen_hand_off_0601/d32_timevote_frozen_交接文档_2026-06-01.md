# d32 time-vote 冻结版交接文档

日期：2026-06-01

服务器：`yan@8.222.254.207`

项目目录：`/home/yan/workspace/projects/rca-bank-adaptive/refute_b_v2`

本地备份包：`C:\Users\echo_\Documents\Codex\2026-06-01\new-chat\outputs\d32_timevote_frozen_2026-06-01.tar.gz`

云端备份包：`/home/yan/workspace/projects/rca-bank-adaptive/refute_b_v2/backups/d32_timevote_frozen_2026-06-01.tar.gz`

## 1. 当前状态

当前冻结版本是 d32 主干的严格 LODO 版本，不再使用 in-sample 结果作为主结果。

核心路线：

- Layer 1：cluster candidate space，但 cluster 特征已去服务名化。
- Layer 1：cluster 输出为 `reason_prior`，不再记忆 `(component, reason)` 的 `root_distribution`。
- Layer 1：mined rules 必须经过内层 LODO 验证后入库。
- Layer 2：component 交给当前 case evidence prior；reason 交给 cluster reason-prior + LODO-stable mined rules。
- Layer 2：保留 `rebuttal_score`、`high_suspicion`、`low_suspicion`、`data_blind_spots`。
- Time：新增 candidate-local time-vote anchor，不再只取最大异常点。
- Trace：使用离线 `trace-summary`，不是 raw trace 全量直接喂入。

这一版不是“工程可运行版 Scheme B v2”补丁，而是独立 `refute_b_v2_d32` 主干。

## 2. 当前代码变更

主要代码：

- `refute_b_v2_d32/signature.py`
  - 去掉 `svc:Tomcat01:*` 这类服务名记忆特征。
  - 增加 `type:*`、`role:*`、`shape:*` 特征。

- `refute_b_v2_d32/layer1.py`
  - cluster 知识改为 `reason_prior`。
  - mined rules 改为内层 LODO 验证后入库。
  - knowledge build 默认支持 LODO / split，不再鼓励 all-train 主结果。

- `refute_b_v2_d32/layer2.py`
  - candidate space 改为 reason-prior × 当前 evidence component prior。
  - 接入新的 time anchor 模块。
  - debug 中写入 `selected_time_anchors`。

- `refute_b_v2_d32/time_anchor.py`
  - 新增 candidate-local time-vote。
  - 支持 metric cross-vote、sustained onset、strongest peak、log keyword onset、trace first_seen。
  - reason-specific 权重：
    - CPU / memory / disk I/O：偏 metric onset。
    - network latency / packet loss：偏 trace first_seen。
    - JVM OOM：偏 log / heap evidence。

- `eval/openrca_official_case_eval.py`
  - 新增官方 case-level strict / partial 评估。
  - partial 口径：一个 case 中至少一个 required field 命中即 partial=1。

## 3. 测试状态

服务器测试命令：

```bash
cd /home/yan/workspace/projects/rca-bank-adaptive/refute_b_v2
PYTHONPATH=..:. python3 -m pytest tests -q
```

结果：

```text
42 passed in 0.87s
```

## 4. 当前最优结果

当前最优输出：

- `logs/d32_lodo_timevote_trace_predictions.csv`
- `logs/d32_lodo_timevote_trace_debug.json`
- `logs/d32_lodo_timevote_trace_checkpoint.jsonl`
- `logs/d32_lodo_timevote_trace_official_case_eval.json`
- `logs/d32_lodo_timevote_trace_field_diag.json`
- `logs/d32_lodo_timevote_trace_reason_conditional.json`

评估设置：

- 数据集：OpenRCA Bank 136 条。
- 训练/测试：leave-one-date-out。
- Knowledge：每个 heldout date 单独构建。
- Rules：训练折内部做 inner-LODO validation 后入库。
- 模态：metric + log + trace-summary。
- Trace：离线 summary，不是 raw trace。
- Time tolerance：official <=1 minute。

官方口径结果：

| 指标 | 当前 d32 time-vote LODO |
|---|---:|
| official strict | 11.76% |
| official partial | 32.35% |
| time hit | 11.83% |
| component hit | 30.12% |
| reason hit | 27.91% |

计数：

```text
official strict  = 16 / 136
official partial = 44 / 136
time hit         = 11 / 93
component hit    = 25 / 83
reason hit       = 24 / 86
```

诊断口径补充：

| 指标 | 当前值 |
|---|---:|
| fractional partial | 21.38% |
| reason-cond strict | 16.18% |
| reason-cond partial | 27.57% |
| reason-cond time hit | 31.18% |
| reason-cond component hit | 30.12% |
| reason-cond reason hit | 27.91% |

注意：`fractional partial` 和 `reason-cond` 是诊断指标，不是 OpenRCA 官方 partial。

## 5. 和上一版对比

| 版本 | official strict | official partial | time hit | component hit | reason hit |
|---|---:|---:|---:|---:|---:|
| old d32 LODO | 8.82% | 28.68% | 10.75% | 21.69% | 22.09% |
| shape + reason-prior + innerLODO，改 time 前 | 9.56% | 30.88% | 6.45% | 30.12% | 27.91% |
| shape + reason-prior + innerLODO + time-vote | 11.76% | 32.35% | 11.83% | 30.12% | 27.91% |

解释：

- 去服务名化 + reason-prior 后，component/reason 泛化明显恢复。
- time-vote 把 time 从 6.45% 拉到 11.83%，没有牺牲 component/reason。
- 当前最大的短板仍然是 time，尤其 network/progressive 类时间定位。

## 6. 当前算法状态判断

当前版本可以作为 d32 主干的冻结版。

已经稳定下来的判断：

- 不再汇报 in-sample 作为主结果。
- cluster 不再记忆具体服务名。
- Layer 1 只学泛化的 fault shape 和 reason prior。
- component 由当前证据决定，而不是历史 cluster 记忆决定。
- mined rules 必须经过 LODO 稳定性验证。
- time 必须是 candidate-local evidence voting。

还不能宣称的事情：

- 不能直接宣称超过 OpenRCA 全数据集 SOTA，因为当前结果是 Bank-only 136 条。
- 不能说三模态融合一定优于单模态或双模态，因为还没做模态消融。
- 不能把 reason-cond 当官方指标。

## 7. 下一步最应该补的实验：模态消融

目标：

判断三模态融合到底带来增益，还是在 reason/time 上引入干扰。

必须使用同一套设置：

- Bank 136 条。
- LODO。
- inner-LODO mined rules。
- official strict / official partial / time hit / component hit / reason hit。
- 每个实验都保留 checkpoint、debug、official case eval、field diag。

建议实验矩阵：

| 实验 | modalities | 目的 |
|---|---|---|
| metric only | `metric` | 看 metric 对 time/component 的基础贡献 |
| log only | `log` | 看 log 对 OOM/reason 的贡献 |
| trace-summary only | `trace` | 看 trace 对 network/component/time 的贡献 |
| metric + log | `metric,log` | 非 trace 路线 |
| metric + trace | `metric,trace` | 预期最可能强的组合 |
| log + trace | `log,trace` | 看文本 + 传播链是否补 reason |
| metric + log + trace | `metric,log,trace` | 当前完整路线，作为对照 |

建议输出命名：

```text
knowledge/d32_lodo_ablation_<modalities>/
logs/d32_lodo_ablation_<modalities>_predictions.csv
logs/d32_lodo_ablation_<modalities>_debug.json
logs/d32_lodo_ablation_<modalities>_checkpoint.jsonl
logs/d32_lodo_ablation_<modalities>_official_case_eval.json
logs/d32_lodo_ablation_<modalities>_field_diag.json
logs/d32_lodo_ablation_<modalities>_reason_conditional.json
```

推荐命令模板：

```bash
cd /home/yan/workspace/projects/rca-bank-adaptive/refute_b_v2

PYTHONPATH=..:. python3 eval/run_openrca_d32_lodo.py \
  --modalities metric,trace \
  --knowledge-dir knowledge/d32_lodo_ablation_metric_trace \
  --out logs/d32_lodo_ablation_metric_trace_predictions.csv \
  --debug-json logs/d32_lodo_ablation_metric_trace_debug.json \
  --checkpoint-jsonl logs/d32_lodo_ablation_metric_trace_checkpoint.jsonl

PYTHONPATH=..:. python3 eval/openrca_official_case_eval.py \
  --pred logs/d32_lodo_ablation_metric_trace_predictions.csv \
  --out logs/d32_lodo_ablation_metric_trace_official_case_eval.json

PYTHONPATH=..:. python3 eval/field_hit_diagnostics.py \
  --pred logs/d32_lodo_ablation_metric_trace_predictions.csv \
  --out logs/d32_lodo_ablation_metric_trace_field_diag.json

PYTHONPATH=..:. python3 eval/reason_conditional_tolerance_eval.py \
  --pred logs/d32_lodo_ablation_metric_trace_predictions.csv \
  --out logs/d32_lodo_ablation_metric_trace_reason_conditional.json
```

## 8. 预期分析重点

模态消融后重点看：

1. `metric only` 是否已经能支撑主要 component 和 time。
2. `trace-summary only` 是否提升 network，但拖累 CPU/memory/disk reason。
3. `metric+trace` 是否接近或超过三模态。
4. `log` 是否只在 OOM/exception 类有价值，还是引入噪声。
5. 三模态完整路线的 component/reason 提升是否抵消 time 漂移。

如果三模态不如双模态，下一步应该改为：

```text
reason-specific modality gate
```

例如：

- network 类：trace 优先，metric 网络 KPI 辅助。
- CPU / memory / disk I/O：metric 优先，trace 降权。
- JVM OOM：log + heap metric 优先。
- disk space：filesystem metric 优先，log 辅助。

## 9. 备份内容

云端备份：

```text
/home/yan/workspace/projects/rca-bank-adaptive/refute_b_v2/backups/d32_timevote_frozen_2026-06-01.tar.gz
```

本地备份：

```text
C:\Users\echo_\Documents\Codex\2026-06-01\new-chat\outputs\d32_timevote_frozen_2026-06-01.tar.gz
```

备份包包含：

- `refute_b_v2_d32/`
- `eval/`
- `tests/`
- `knowledge/d32_lodo_shape_reasonprior_innerlodo/`
- 当前最优 time-vote 预测、debug、checkpoint、评估结果
- 改 time 前的 innerLODO 对照预测、debug、checkpoint、评估结果

## 10. 接手建议

下一位接手者不要从旧 `JointAnswerSelector` 继续 patch。

建议从这里继续：

1. 先跑模态消融。
2. 根据消融结果决定是否加入 reason-specific modality gate。
3. 再针对 time 错误样本做 `selected_time_anchors` debug 对齐。
4. 最后才考虑调 rebuttal_score 参数。

当前最应该避免：

- 避免回到 in-sample 指标。
- 避免把服务名重新放进 cluster 特征。
- 避免让 Layer 1 再输出 `(component, reason)` 记忆。
- 避免把 raw 三模态直接拼进 LLM prompt。
