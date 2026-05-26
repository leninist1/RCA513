# SYH 任务交接：完整 PRISM Runtime / 可运行性优化

日期：2026-05-26

## 任务定位

你负责方向 3：让完整 PRISM 在 Bank 全量上可运行，并产出可对比的 Correct / Partial 结果。

当前状态：

- `noise_lab` 已有 Bank LTR OOF prior。
- PRISM 已接入 NoiseLab prior / final blend。
- Bank 全量目前使用 `--prism-fast` 才能完成。
- 完整 PRISM loop 曾在 7q smoke 中超过 5 分钟仍未完成，因此没有 Bank 134q full PRISM 结果。

你的目标不是修 reason taxonomy，也不是训练新的 NoiseLab scorer；你只负责 PRISM runtime、缓存、top-k counterfactual gate、调试输出和完整/半完整模式评估。

## 交接目录内容

本目录下应包含：

- `code/openrca_meta_controller/`：从 yyx 当前工作区复制的相关代码包。
- `reference_results/option_PRISM_20260525_221432.json`：当前 Bank PRISM-fast + NoiseLab OOF 结果。
- `reference_results/noise_lab_ltr_bankfull_ndcg_noprefix_nobase_Bank_20260525_212615.json`：当前 Bank LTR OOF 排名结果。
- `reference_results/noise_lab_ltr_scores_bankfull_ndcg_noprefix_nobase_Bank_20260525_212615.csv`：PRISM 接入使用的 NoiseLab OOF score CSV。

源工作区参考路径：

```bash
/home/dell2/RCA513/yyx/rca513/openrca_meta_controller
/home/dell2/RCA513/yyx/rca513/results
```

## 需要重点看的代码

- `code/openrca_meta_controller/main.py`
  - `--prism-fast`
  - `--prism-noise-lab`
  - `--prism-noise-lab-scores`
  - `--prism-noise-lab-strategy`
  - `field_scores` 持久化位置
- `code/openrca_meta_controller/prism.py`
  - `PRISMConfig`
  - `PRISMPipeline.run()`
  - `cf_profiles_enabled`
  - `final_counterfactual_enabled`
  - `_noise_lab_prior_from_scores(...)`
  - `_noise_lab_prior_from_runtime_scorer(...)`
  - state transition / counterfactual profile 相关逻辑
- `code/openrca_meta_controller/counterfactual/engine.py`
  - counterfactual 成本来源。
- `code/openrca_meta_controller/data/loader.py`
  - telemetry 加载可能重复 IO。
- `code/openrca_meta_controller/mace/graph.py`
  - object graph 构建成本和 trace/metric graph 推断。
- `code/openrca_meta_controller/evaluation/aggregator.py`
  - Correct / Partial / by_task / by_field 汇总。

## 当前已知 baseline

当前可完成的 Bank 全量命令：

```bash
cd /home/dell2/RCA513/yyx/rca513

python -u -m openrca_meta_controller.main \
  --option PRISM --systems Bank --workers 4 --output results \
  --prism-fast \
  --prism-noise-lab \
  --prism-noise-lab-scores results/noise_lab_ltr_scores_bankfull_ndcg_noprefix_nobase_Bank_20260525_212615.csv \
  --prism-noise-lab-strategy ltr_full
```

结果文件：

```text
results/option_PRISM_20260525_221432.json
```

结果摘要：

- total = 134
- Correct = 21 / 134 = 15.67%
- Partial-only = 44 / 134 = 32.84%
- Correct + Partial = 65 / 134 = 48.51%

by task：

- task_1 time：19/23 Correct
- task_2 reason：0/22 Correct
- task_3 component：1/15 Correct
- task_4 time+reason：0/18 Correct，16 Partial
- task_5 time+component：1/18 Correct，12 Partial
- task_6 component+reason：0/21 Correct，3 Partial
- task_7 time+component+reason：0/17 Correct，13 Partial

## 为什么完整 PRISM 慢

当前 `--prism-fast` 做了这些减法：

- `t_max=1`
- `n_cf_max=0`
- `unexplained_trigger=2.0`
- `probe_edge_budget=0`
- `cf_profiles_enabled=False`
- `final_counterfactual_enabled=False`

也就是说 full PRISM 慢的主要嫌疑是：

- counterfactual profiles 构建。
- final counterfactual discriminator。
- 每个 query 重复加载 telemetry / 重复建图。
- degradation map / metric detail / graph computations 没有跨 query 缓存。
- 对全候选运行重逻辑，而不是只对 LTR top-k / belief top-k。

## 建议实现路线

第一步：加 profiling，不要先盲目改算法。

建议在 `PRISMPipeline.run()` 内记录：

```json
"runtime_debug": {
  "load_telemetry_sec": ...,
  "build_graph_sec": ...,
  "noise_lab_prior_sec": ...,
  "state_init_sec": ...,
  "cf_profiles_sec": ...,
  "state_loop_sec": ...,
  "final_cf_sec": ...,
  "total_sec": ...
}
```

第二步：做缓存。

优先级：

- telemetry cache：key = `(system, telemetry_date, sub_system)`。
- graph cache：key = `(system, telemetry_date, sub_system, inject_time)`，或者至少缓存 telemetry 后减少 IO。
- NoiseLab score CSV cache：只读一次，不要每个 pipeline/query 重读。
- degradation detail / metric profile cache：按 query 或 date 缓存。

第三步：新增一个中间模式，不要只有 fast/full。

建议新增 `--prism-mid` 或配置等价模式：

- `t_max=1 or 2`
- `n_cf_max` 小于 full，例如 2
- `cf_profiles_enabled=True`
- `final_counterfactual_enabled=False`
- counterfactual 只对 top-k 候选运行

这样能区分“state/cf profile 是否有收益”和“final discriminator 是否太慢”。

第四步：top-k gate。

用 NoiseLab OOF prior 或当前 belief 选 top-k：

- full profile / final CF 只对 top 5 或 top 10 component 候选运行。
- 如果 NoiseLab top1/top2 gap 很大，跳过 final CF。
- 如果 PRISM belief 和 NoiseLab prior 冲突，再触发 CF。

第五步：保证结果可对比。

每次跑完都输出：

- config：fast/mid/full/top-k 参数。
- summary：Correct / Partial / Correct+Partial。
- by_task。
- by_field。
- runtime_debug 聚合：mean/p50/p95/total。

## 成功标准

最低目标：

- 7q smoke 在完整或 mid 模式下可在几分钟内完成，并有 runtime breakdown。
- Bank 134q 至少能跑完一个 `PRISM-mid + NoiseLab` 结果。
- 结果 JSON 保留 `field_scores` 和 runtime debug，方便 ysj 做 reason 方向时对照。

理想目标：

- Bank 134q full/top-k PRISM 在可接受时间内完成。
- 相比 `--prism-fast`，Correct 或 component exact 有提升，或者证明 full CF 不带来收益但成本很高。

## 分工边界

- 不要改 reason taxonomy；reason 归一化由 ysj 负责。
- 不要重新训练 LTR；NoiseLab prior 直接使用现有 OOF scores CSV。
- 可以改 PRISM 的调度、缓存、top-k gate、debug 输出。
- 如果必须改 `main.py` / `aggregator.py`，保留原有 `--prism-fast` 行为，避免破坏 yyx 当前 baseline。

