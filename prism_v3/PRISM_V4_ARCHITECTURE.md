# PRISMv4 架构图

> 当前仓库没有单独的 `prism_v4/` 目录；这里的 PRISMv4 指
> `prism_v3/noise_native/*` 已经实现的最新 Noise-native PRISM Agent 架构。

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│ Layer 3: 事件级答案合成 / 语义解释层                                         │
│   - 输入: FaultEvent 后验 + evidence ledger + reason/time candidates          │
│   - 生成 OpenRCA 官方格式 root_cause_events                                  │
│   - 做 reason 归一化、time 解析、多故障 component-reason-time 配对             │
│   - 输出: {component, reason, time, root_cause_events, debug}                 │
└──────────────────────────────────────────────────────────────────────────────┘
                                   ↑
                 事件后验 + root 选择分数 + 结构化证据账本
                                   ↑
┌──────────────────────────────────────────────────────────────────────────────┐
│ Layer 2: Noise-native PRISM Agent / 主动反事实推理层                         │
│   - 输入: EvidenceFrame + ToolContext                                         │
│   - 以 FaultEvent 为状态单元, 不再只维护组件级概率向量 p                      │
│   - observe -> act -> reason 循环:                                            │
│       inspect_metric / inspect_log / verify_trace                             │
│       run_counterfactual / residual_collapse / mechanism_intervention(CMI)    │
│       compare_pair / split_event / merge_events                               │
│   - 因子化后验: NoiseLab + metric + log + trace + CF + CMI - symptom penalty   │
│   - 输出: {候选事件: posterior + status_probs + evidence_ledger}              │
└──────────────────────────────────────────────────────────────────────────────┘
                                   ↑
          CandidateFrame + metric/log/trace 证据 + source/symptom 角色特征
                                   ↑
┌──────────────────────────────────────────────────────────────────────────────┐
│ Layer 1: NoiseLab 原生感知层 / 传统 PRISM 信号层                              │
│   - OpenRCA telemetry: metrics / logs / traces + query                         │
│   - Legacy PRISM: 构造 a_obs、log_signal、trace graph W、metric/log detail      │
│   - NoiseLab adapter:                                                         │
│       scores_csv: 读取 LTR OOF 候选分数                                        │
│       runtime scorer: object graph + NoiseField/structure/delay/beam/subspace  │
│   - 输出 EvidenceFrame / CandidateFrame:                                      │
│       noise_score、calibrated_logit、metric/log/trace evidence                 │
│       time_candidates、reason_candidates、structural_features                  │
│       symptomness、source_likelihood、prior_mass                               │
└──────────────────────────────────────────────────────────────────────────────┘
                                   ↑
             /home/dell2/RCA513/yyx/OpenRCA/{Bank,Telecom,Market}
```

## 每层设计思路

### Layer 1: NoiseLab 原生感知层 / 传统 PRISM 信号层

这一层解决的问题是：OpenRCA 的原始 telemetry 很杂，直接交给推理层会导致两个问题。第一，候选空间太大，Bank、Telecom、Market 里有大量服务、容器、节点、数据库对象，不能每个 case 都全量搜索。第二，metric、log、trace 的证据形态不一样，metric 是数值异常，log 是文本/错误模式，trace 是方向和传播关系，如果不先统一成结构化证据，后面只能做粗暴加权。

所以 Layer 1 的核心思想是“先看见，再推理”：它不急着给最终答案，而是把不同模态的原始观测整理成候选级证据框。

具体设计分两条线：

1. 传统 PRISM 信号线负责构造可计算世界。
   - `a_obs` 表示每个实体的 metric 异常强度。
   - `log_signal` 表示每个实体的日志异常强度。
   - `W` 表示 trace / 调用 / 传播图，用来区分上游 root 和下游 symptom。
   - `metric_detail`、`log_detail`、`anomaly_times` 保存后续解释和 reason/time 推断所需的细节。

2. NoiseLab 感知线负责压缩候选空间。
   - 如果有离线训练好的 LTR OOF score，就从 score CSV 读取候选排名。
   - 如果没有 score CSV，就在线跑 runtime scorer，从 object graph 中提取 NoiseField、structure、delay、beam、subspace、reverb 等特征。
   - 最终不是只输出一个分数，而是输出 `EvidenceFrame / CandidateFrame`。

`CandidateFrame` 是这一层最重要的接口。它把一个候选实体的证据打包起来，包括 `noise_score`、`calibrated_logit`、metric/log/trace evidence、候选时间、候选原因、结构特征、`symptomness`、`source_likelihood` 和 `prior_mass`。这样 Layer 2 拿到的不是“某个组件排第几”，而是“这个组件为什么像 root、又为什么可能只是 symptom”。

这一层的设计边界是：只做感知、候选生成和证据结构化，不做最终裁决。它可以强烈提示候选，但不能直接把 LTR 第一名当根因答案。

### Layer 2: Noise-native PRISM Agent / 主动反事实推理层

这一层是 PRISMv4 和旧 PRISMv3 patch 最大的区别。旧版本主要维护组件级概率向量 `p`，后面再用 final counterfactual discriminator 或 final blend 做末端重排。问题是这种结构容易让系统变成“打分流水线”：NoiseLab 给一个排序，PRISM 再改一遍，CF 最后再裁判一次。模块之间会互相拉扯，而不是形成一个真正的调查过程。

Layer 2 的核心思想是把 RCA 改成事件级主动调查：PRISM 不再只问“哪个组件分最高”，而是维护一组 `FaultEvent`，每个事件都有 component、reason、time、posterior、status_probs 和 evidence ledger。这样多故障场景里 component、reason、time 的配对关系不会到最后才硬拼。

每个 `FaultEvent` 会维护五类状态：

1. `posterior`：当前认为它是根因事件的概率。
2. `status_probs`：它是 root、symptom、broad explainer、duplicate、unresolved 的概率。
3. `factor_states`：每类证据因子的当前值和置信度。
4. `evidence_ledger`：所有工具返回的结构化证据。
5. `conflict_notes`：比如 source evidence 和 symptom evidence 冲突时的记录。

Agent 的循环是：

```text
Observe: 读取 EvidenceFrame, 初始化 FaultEvent
Act:     按不确定性和冲突选择下一步观测工具
Reason:  把工具返回的 EvidenceObservation 写入 ledger, 更新因子化后验
Stop:    熵、top gap、root probability、工具覆盖率满足条件后停止
```

工具层的设计重点是让 action 真正产生新证据，而不是简单重新加权：

- `inspect_metric` 检查候选实体的 metric 异常强度和 top metrics。
- `inspect_log` 检查日志信号、错误数量和文本片段。
- `verify_trace` 检查候选在图中的输入/输出方向，判断它更像上游源头还是下游受害者。
- `run_counterfactual` 把反事实从“最终裁判”降级为一个证据因子，回答这个候选能否解释局部和下游异常。
- `residual_collapse` 看移除该候选解释后，图上的剩余异常是否明显塌缩。
- `mechanism_intervention(CMI)` 检查机制破坏、父节点解释能力、修复唯一性，用来处理“看起来很异常但其实被上游解释”的候选。
- `compare_pair` 专门处理 top candidates 分数接近或角色冲突的情况。
- `split_event / merge_events` 处理多故障和重复候选。

Layer 2 的 posterior 不是一个黑盒分数，而是因子化组合：

```text
root posterior
  = NoiseLab_logit
  + metric_likelihood
  + log_likelihood
  + trace_direction_likelihood
  + counterfactual_likelihood
  + residual_collapse
  + mechanism_break_likelihood
  + intervention_uniqueness
  - symptomness
  - hotspot_symptom
  - broad_explainer
```

这里最关键的设计取舍是：CF 和 CMI 都只是证据因子，不再拥有“一票否决”或“一票定案”的地位。这样可以避免 Tomcat 这类 broad downstream explainer 因为能解释很多下游现象而压过真正 root，也可以保留 NoiseLab top-k 中有价值的候选。

Layer 2 输出给下一层的不是简单 top1，而是一组带证据账本的事件候选：每个候选都有 posterior、root/symptom 状态、反事实证据、机制证据和选择分数。Layer 3 才基于这些事件对象生成最终答案。

### Layer 3: 事件级答案合成 / 语义解释层

这一层解决的问题是：OpenRCA 的评分对象不是单纯的组件排名，而是 `{root cause occurrence datetime, root cause component, root cause reason}` 这样的事件对象。组件对了但 reason 配错，或者多故障时 component 和 reason 配对错，最终都会影响官方得分。

所以 Layer 3 的核心思想是“先选事件，再填字段”。它不再从组件列表、reason 列表、time 列表里分别取 top 值拼答案，而是从 Layer 2 的 `FaultEvent` 中选出最可信的根因事件，再对每个事件分别解析 component、reason、time。

具体做法：

1. 用 `root_selection_score` 对事件排序。
   - 排序不只看 posterior，还看 root probability、source isolation、trace/CF/CMI 支持、residual collapse、hotspot penalty 等。
   - 已被标记为 duplicate / merged 的事件不会优先输出。

2. 用 reason resolver 生成规范原因。
   - 优先利用 `reason_candidates`。
   - 再结合 metric names、log excerpt、已有 reason taxonomy 做归一化。
   - 目标是把候选原因转成 OpenRCA 可评分的 canonical reason。

3. 用 time resolver 生成故障发生时间。
   - 优先使用事件自身 time candidates。
   - 缺失时退回 query / inject time / anomaly_times 推断。

4. 输出 `root_cause_events`。
   - 这是官方格式的事件级答案。
   - 同时再转换成旧兼容字段 `component/reason/time`，保证原有评估器和结果 JSON 不被破坏。

当前代码里的 Layer 3 更偏“规则归一化 + 事件级合成”，默认运行不依赖在线 LLM。如果后续论文或系统设计要加入 LLM，它最自然的位置就在这一层：LLM 不负责全量搜索，也不直接改 posterior，而是基于 Layer 2 提供的结构化证据链，做 case-by-case 的自然语言调查报告、边界冲突解释、跨数据集 prompt 迁移和缺失数据建议。这样 LLM 只处理少量高价值 case，成本可控，也不破坏底层可复现的数值证据链。

### 三层之间的核心关系

```text
Layer 1 负责把原始 telemetry 变成候选和结构化证据:
  raw telemetry -> EvidenceFrame / CandidateFrame

Layer 2 负责把候选变成经过主动验证的事件后验:
  CandidateFrame -> FaultEvent -> factorized posterior

Layer 3 负责把事件后验变成可评分、可解释的最终答案:
  FaultEvent posterior -> root_cause_events -> component/reason/time
```

这个分层的好处是职责清楚：

- Layer 1 管“看见什么”，避免候选空间爆炸。
- Layer 2 管“怎么验证”，避免把排序分数误当因果证据。
- Layer 3 管“怎么回答”，避免多故障场景里字段错配。

它也解释了为什么 PRISMv4 要叫 Noise-native：NoiseLab 不再是 PRISM 外面的一张分数表，而是整个 agent 的第一层感知接口；后续所有推理、反事实、CMI 和答案合成都围绕 `FaultEvent` 这个事件级状态展开。

## 对应源码

```text
入口/配置:
  prism_v3/main.py
  prism_v3/prism.py

Layer 1:
  prism_v3/noise_native/evidence_frame.py
  prism_v3/noise_native/noiselab_adapter.py
  prism_v3/noise_lab/*

Layer 2:
  prism_v3/noise_native/agent.py
  prism_v3/noise_native/fault_event.py
  prism_v3/noise_native/tools.py
  prism_v3/noise_native/posterior.py
  prism_v3/noise_native/cmi.py

Layer 3:
  prism_v3/noise_native/synthesis.py
  prism_v3/evaluation/scorer.py
```

## 当前最新运行口径

```text
数据集: Bank 136 cases
结果文件:
  prism_v3/results/bank136_noise_native_cmi_guarded_20260531/
  option_PRISM_20260601_002754.json

关键开关:
  --prism-noise-lab
  --prism-noise-lab-strategy ltr_full
  noise_native_agent_enabled = True
  noise_native_cmi_enabled = True
  noise_native_max_events = 10
  noise_native_max_rounds = 2

结果摘要:
  Strict Correct = 37 / 136 = 27.21%
  Official Score = 46.63%
  Nonzero Partial = 60 / 136 = 44.12%
  Any Nonzero Score = 71.32%
```
