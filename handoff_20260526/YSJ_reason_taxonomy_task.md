# YSJ 任务交接：Bank Reason Taxonomy / Reason Canonicalization

日期：2026-05-26

## 任务定位

你负责方向 2：修复 `noise_lab + meta-controller + PRISM` 在 Bank 上的 reason 字段。

当前 Bank 全量 PRISM-fast + NoiseLab OOF 结果已经跑通，但 reason exact 基本是硬瓶颈：

- Bank 有效 query：134
- 整体 Correct：21/134 = 15.67%
- Partial-only：44/134 = 32.84%
- Correct + Partial：65/134 = 48.51%
- 手工复盘字段级结果：reason 约 0/78 命中
- 受影响任务：task_2, task_4, task_6, task_7

你的目标不是改 component 排名，也不是优化 PRISM runtime；只做 reason 输出归一化和可解释诊断。

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

- `code/openrca_meta_controller/evaluation/scorer.py`
  - `evaluate_prediction()`：最终字段评分入口。
  - `_match_field("reason", ...)`：当前 reason 是 lowercase exact set match。
  - `construct_answer()`：非 PRISM 路径的最终答案构造。
  - `_infer_reason()` / `_infer_reason_from_hypothesis()`：旧 meta-controller 的 reason 推断。
- `code/openrca_meta_controller/prism.py`
  - `PRISMPipeline.run()` 最终 prediction 输出。
  - `_infer_reason(...)`：PRISM 当前 reason 生成逻辑。
  - `noise_lab_prior` 只影响 component prior/final belief，不解决 reason。
- `code/openrca_meta_controller/mace/graph.py`
  - `RESOURCE_REASON_MAP`
  - `RESOURCE_KEYWORDS`
  - `ObjectNode.best_reason()`
- `code/openrca_meta_controller/data/loader.py`
  - GT record 匹配和 `query.scoring_points` 生成。
- `code/openrca_meta_controller/config.py`
  - `GroundTruth`, `QueryCase`, `ScoringPoint`, `EvalResult` 数据结构。
- `code/openrca_meta_controller/noise_lab/runner.py`
  - `top_candidates[*].reason` 可作为 reason 证据来源之一，但不要直接相信。

## 建议实现方案

第一步：统计 Bank GT canonical reasons。

从 `QueryCase.scoring_points` 或 matched `GroundTruth.reason` 中收集 Bank reason 标签，形成固定集合。当前观察到的典型标签包括：

- `high CPU usage`
- `high memory usage`
- `network latency`
- `network packet loss`
- `high disk I/O read usage`
- `high disk space usage`
- `JVM Out of Memory (OOM) Heap`
- `high JVM CPU load`

第二步：新增一个独立 normalizer，建议文件：

```text
code/openrca_meta_controller/evaluation/reason_normalizer.py
```

建议提供这些函数：

```python
def canonical_bank_reasons() -> list[str]: ...
def normalize_reason_text(text: str) -> str: ...
def infer_bank_reason(entity: str, evidence_pool: dict, query, metric_detail=None, log_detail=None) -> str: ...
def choose_reason_from_candidates(raw_reason: str, instruction: str, metric_names: list[str]) -> str: ...
```

第三步：接入两个输出点。

- 在 `evaluation/scorer.py` 的 `_infer_reason()` 和 `_infer_reason_from_hypothesis()` 中返回 canonical reason。
- 在 `prism.py` 的 `_infer_reason(...)` 中返回同一套 canonical reason。

注意：不要靠 `query.ground_truth.reason` 直接回填 prediction，这会数据泄漏。可以使用 query instruction、metric/log evidence、candidate object 的 reason votes，但不能用当前样本 GT。

第四步：评分侧可以加 debug，但不要改变正式 exact 口径。

可以在结果里额外输出：

```json
"reason_debug": {
  "raw_reason": "...",
  "canonical_reason": "...",
  "matched_rule": "...",
  "evidence": [...]
}
```

但最终 `prediction["reason"]` 必须是 OpenRCA GT 里的 canonical 字符串。

## 推荐 baseline 命令

在完整项目根目录运行：

```bash
cd /home/dell2/RCA513/yyx/rca513

python -u -m openrca_meta_controller.main \
  --option PRISM --systems Bank --workers 4 --output results \
  --prism-fast \
  --prism-noise-lab \
  --prism-noise-lab-scores results/noise_lab_ltr_scores_bankfull_ndcg_noprefix_nobase_Bank_20260525_212615.csv \
  --prism-noise-lab-strategy ltr_full
```

如果你在自己的目录单独跑，需要保证 Python 可以 import 到你的 `code/openrca_meta_controller`，并且数据路径仍能访问 `/home/dell2/RCA513/yyx/rca513` 下的 OpenRCA 数据。

## 成功标准

最低目标：

- Bank reason 字段从 0 命中提升到可观的非零命中。
- task_2/task_4/task_6/task_7 的 Correct 有提升。
- 不改变 component 排名逻辑，不让 task_3/task_5 component 指标明显退化。

建议汇报指标：

- Overall Correct / Partial / Correct+Partial。
- by_task 的 Correct / Partial。
- by_field 的 `reason/component/time`。
- reason confusion matrix：GT reason -> predicted reason。
- 每条规则命中数和错误样例。

## 已知风险

- Bank time 字段当前偏乐观，因为框架常从 matched record / inject time 推出 time。
- reason exact 是严格字符串匹配，`CPU fault` 不等于 `high CPU usage`。
- `mace/graph.py` 的 `RESOURCE_REASON_MAP` 当前更偏通用原因族，不等同 Bank GT taxonomy。
- LLM 可以用于 canonical label 选择，但不要自由生成 reason；自由生成会继续 exact mismatch。

