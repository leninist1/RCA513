# Confidence-Aware RCA V2 + OOD Drift Focus 交接文档

日期：2026-06-02

## 1. 当前决定

采纳 Claude 建议中的两点：

1. 加入 OOD-aware / drift detection confidence features。
2. 主表放弃 strict，聚焦 `official partial` 和 `actionable_any`。

不继续做 date-stratified conformal，也不把 strict selective accuracy 作为主卖点。

## 2. 代码位置

服务器：

```text
yan@8.222.254.207
/home/yan/workspace/projects/rca-bank-adaptive/refute_b_v2
```

本地镜像：

```text
C:\Users\echo_\Documents\Codex\2026-05-30\new-chat-2\work\refute_b_v2
```

核心代码：

```text
confidence_aware_rca/selective_eval.py
confidence_aware_rca/forward_holdout_eval.py
confidence_aware_rca/analyze_selective_results.py
tests/test_confidence_aware_rca.py
```

底座 RCA 代码：

```text
refute_b_v2_d32/
eval/run_openrca_d32_lodo.py
eval/build_d32_knowledge.py
```

## 3. 底座版本

当前底座仍然是：

```text
d32 time-vote
trace-summary
Bank 136 cases
strict LODO
```

底座输出：

```text
logs/d32_lodo_timevote_trace_predictions.csv
logs/d32_lodo_timevote_trace_debug.json
logs/d32_lodo_timevote_trace_official_case_eval.json
logs/d32_lodo_timevote_trace_field_diag.json
```

底座全量指标：

| 指标 | 当前 Bank LODO |
|---|---:|
| official strict | 16 / 136 = 11.76% |
| official partial | 44 / 136 = 32.35% |
| time hit | 11.83% |
| component hit | 30.12% |
| reason hit | 27.91% |

confidence-aware 不改变这些 RCA 预测，只做 selective / abstention 判断。

## 4. 与 OpenRCA 官方公开参考对比

OpenRCA ICLR 2025 公开参考是全量 335 cases，不是 Bank-only。

| 指标 | OpenRCA 官方公开参考 | 我们当前 Bank LODO | 对比 |
|---|---:|---:|---:|
| official strict | 11.34% | 11.76% | +0.42 pp |
| official partial | 24.18% | 32.35% | +8.17 pp |
| time hit | 14.23% | 11.83% | -2.40 pp |
| component hit | 18.00% | 30.12% | +12.12 pp |
| reason hit | 19.67% | 27.91% | +8.24 pp |

注意：

```text
不能直接宣称超过 OpenRCA 全量 SOTA，因为当前是 Bank-only 136 cases。
```

## 5. V1 泄漏修正

旧 v1 confidence 结果废弃。

原因：

```text
特征矩阵中误放入了 label-like diagnostic columns：
*_hit_count
*_hit_any_label
*_required_total
```

V2 已修正，`selective_eval.py` 排除：

```text
*_label
*_required
*_required_total
*_hit_count
```

## 6. V2 Core / Auto 主结果

主结果文件：

```text
confidence_aware_rca/experiments/d32_timevote_confidence_features_v2.csv
confidence_aware_rca/experiments/d32_timevote_selective_lodo_logistic_core_v2.json
confidence_aware_rca/experiments/d32_timevote_selective_forward_last3_core_v2.json
confidence_aware_rca/experiments/d32_timevote_selective_lodo_logistic_auto_v2.json
confidence_aware_rca/experiments/d32_timevote_selective_forward_last3_auto_v2.json
confidence_aware_rca/experiments/selective_summary_logistic_core_v2.md
confidence_aware_rca/experiments/selective_summary_logistic_auto_v2.md
```

核心结果：

| target | full/base rate | LODO coverage@90 | LODO acc@90 | Forward coverage@90 | Forward acc@90 |
|---|---:|---:|---:|---:|---:|
| official partial | 32.35% | 30.15% | 58.54% | 32.56% | 57.14% |
| actionable_any | 18.31% | 15.49% | 54.55% | 23.81% | 60.00% |

结论：

```text
V2 core/auto 是当前最稳主版本。
主 claim 应聚焦 official partial / actionable_any。
```

## 7. OOD / Drift Features 新增内容

新增参数：

```text
--drift-features
```

新增特征：

```text
drift_l1_to_train_median
drift_l2_to_train_median
drift_max_abs_z
drift_missing_rate
```

实现方式：

```text
每个 LODO / forward split 内，只用 train rows 计算 median/IQR，
再计算 train/test rows 到训练分布中心的距离。
```

这样避免泄漏测试日期分布。

核心代码函数：

```text
append_drift_features()
model_probabilities(..., drift_features=True)
```

测试已覆盖：

```text
test_append_drift_features_uses_train_distribution
```

## 8. Drift 实验结果

新增结果文件：

```text
confidence_aware_rca/experiments/d32_timevote_selective_lodo_logistic_core_drift_v2.json
confidence_aware_rca/experiments/d32_timevote_selective_forward_last3_core_drift_v2.json
confidence_aware_rca/experiments/d32_timevote_selective_lodo_logistic_auto_drift_v2.json
confidence_aware_rca/experiments/d32_timevote_selective_forward_last3_auto_drift_v2.json
confidence_aware_rca/experiments/selective_summary_logistic_core_drift_v2_main.md
confidence_aware_rca/experiments/selective_summary_logistic_auto_drift_v2_main.md
```

主 target 对比：

| target | 原 V2 LODO acc@90 | drift V2 LODO acc@90 | 原 V2 Forward acc@90 | drift V2 Forward acc@90 |
|---|---:|---:|---:|---:|
| official partial | 58.54% | 62.16% | 57.14% | 57.14% |
| actionable_any | 54.55% | 54.55% | 60.00% | 60.00% |

coverage 对比：

| target | 原 V2 LODO coverage@90 | drift V2 LODO coverage@90 | 原 V2 Forward coverage@90 | drift V2 Forward coverage@90 |
|---|---:|---:|---:|---:|
| official partial | 30.15% | 27.21% | 32.56% | 32.56% |
| actionable_any | 15.49% | 15.49% | 23.81% | 23.81% |

解释：

```text
drift features 对 LODO official partial 有正向作用：
58.54% -> 62.16%，但 coverage 从 30.15% 降到 27.21%。

Forward official partial 没有变化：
57.14%，coverage 32.56%。

actionable_any 基本不变。
```

所以：

```text
drift features 可以作为 OOD-aware confidence 的增强实验 / ablation。
但不能宣称它解决了 forward/date drift。
```

## 9. strict 为什么不进主表

strict 正例太少：

```text
16 / 136 = 11.76%
```

V2 core strict selective：

| split | coverage@90 | acc@90 |
|---|---:|---:|
| LODO | 19.85% | 14.81% |
| Forward | 34.88% | 20.00% |

这说明系统不能可靠识别 strict 正确子集。

推荐写法：

```text
Full-triple strict RCA remains difficult and is reported as a limitation.
The main study focuses on selective partial/actionable RCA.
```

## 10. 当前推荐主表

建议主论文表只放：

| method | target | coverage@90 LODO | selective acc@90 LODO | coverage@90 Forward | selective acc@90 Forward |
|---|---|---:|---:|---:|---:|
| V2 core/auto | official partial | 30.15% | 58.54% | 32.56% | 57.14% |
| V2 core+drift | official partial | 27.21% | 62.16% | 32.56% | 57.14% |
| V2 core/auto | actionable_any | 15.49% | 54.55% | 23.81% | 60.00% |
| V2 core+drift | actionable_any | 15.49% | 54.55% | 23.81% | 60.00% |

建议主 claim：

```text
V2 confidence-aware identifies about 30% high-confidence cases where official
partial accuracy improves from 32.35% to 58.54% under LODO and 57.14% under
forward holdout. OOD-aware drift features further improve LODO selective
partial accuracy to 62.16% at lower coverage, but do not improve forward
holdout.
```

## 11. 复现命令

进入服务器项目：

```bash
cd /home/yan/workspace/projects/rca-bank-adaptive/refute_b_v2
```

测试：

```bash
PYTHONPATH=..:. python3 -m pytest tests -q
```

当前测试结果：

```text
52 passed
```

V2 core + drift LODO：

```bash
PYTHONPATH=..:. python3 confidence_aware_rca/selective_eval.py \
  --features confidence_aware_rca/experiments/d32_timevote_confidence_features_v2.csv \
  --out confidence_aware_rca/experiments/d32_timevote_selective_lodo_logistic_core_drift_v2.json \
  --model logistic \
  --risk-mode empirical \
  --feature-set core \
  --drift-features
```

V2 core + drift forward：

```bash
PYTHONPATH=..:. python3 confidence_aware_rca/forward_holdout_eval.py \
  --features confidence_aware_rca/experiments/d32_timevote_confidence_features_v2.csv \
  --out confidence_aware_rca/experiments/d32_timevote_selective_forward_last3_core_drift_v2.json \
  --model logistic \
  --risk-mode empirical \
  --feature-set core \
  --drift-features
```

生成主 target 摘要：

```bash
PYTHONPATH=..:. python3 confidence_aware_rca/analyze_selective_results.py \
  --lodo confidence_aware_rca/experiments/d32_timevote_selective_lodo_logistic_core_drift_v2.json \
  --forward confidence_aware_rca/experiments/d32_timevote_selective_forward_last3_core_drift_v2.json \
  --out confidence_aware_rca/experiments/selective_summary_logistic_core_drift_v2_main.md \
  --display-targets partial,actionable_any
```

## 12. 最终结论

```text
1. V1 泄漏，废弃。
2. V2 core/auto 是当前最稳主版本。
3. strict 不进主表，只放 limitation。
4. 主表聚焦 official partial 和 actionable_any。
5. OOD/drift features 有 LODO partial 增益，但 forward 不变。
6. 推荐论文主线仍是 selective RCA / RCA self-knowledge。
```
