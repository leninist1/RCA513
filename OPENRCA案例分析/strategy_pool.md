# 基于反事实验证的 RCA 策略池

本文基于 `mydocs/反事实验证思路.md` 与 `mydocs` 下的三类案例画像（`Bank`、`Market`、`Telecom`）设计一套可被 Meta-Controller 调度的策略池。目标不是“看到异常就下结论”，而是围绕候选根因做反事实验证：如果某候选不是真正根因，那么当前跨模态观测是否仍然成立。

从案例画像看，OpenRCA 的主导难点有四类：

1. 资源类与局部负载类案例占比高。
`Bank` 以 CPU / I/O / JVM 类为主，`Telecom` 以 CPU fault 为主，`Market` 同时覆盖 container 与 node 两侧的 CPU / Memory / I/O / disk space。

2. 网络类故障表现强，但子类型不一致。
`Bank` 和 `Telecom` 以 delay / loss 为主，`Market` 还额外出现 corruption / retransmission；其中 retransmission 往往局部信号强、端到端症状弱。

3. 层级与传播结构不一致。
`Bank` 基本是 pod 层单点问题；`Market` 同时包含 node / service / pod，且有副本容错；`Telecom` 需要借助 trace/JDBC 才能分清 db/service/node 的传播关系。

4. 业务症状并不总是单调恶化。
更新后的画像显示，`Bank` 的部分 CPU / Memory / I/O / OOM 案例、`Market` 的 retransmission / process termination / 部分 node 资源类案例，业务 `mrt` / `sr` 可能近乎不变甚至反向；`Telecom` 的 db 类问题也更常体现为成功率下降，而不是时延升高。

因此策略池不按“具体故障名”硬编码，而按“验证动作模式”划分，便于在不同数据子集之间复用。

## 共通接口

### 共通输入

所有策略使用同一组输入对象：

```json
{
  "case_context": {
    "dataset": "Bank | Market | Telecom",
    "failure_time": "YYYY-MM-DD HH:MM:SS",
    "analysis_window": {
      "baseline": "[t-30m, t)",
      "incident": "[t, t+10m]"
    },
    "topology": "可为空；包含 node/service/pod 部署关系与 trace 依赖边",
    "budget": {
      "max_rounds": 1,
      "max_candidates": 5,
      "max_hops": 4
    }
  },
  "candidate_pool": [
    {
      "component": "候选组件",
      "layer": "node | service | pod",
      "reason_family": "cpu | mem | io | net | db | jvm | process | unknown",
      "seed_evidence": ["初始化证据"]
    }
  ],
  "evidence_pool": {
    "metrics": "已抽取的 KPI 异常",
    "logs": "已抽取的关键词/事件锚点，可为空",
    "traces": "已抽取的依赖边/时延/错误率，可为空",
    "business_symptoms": "mrt / sr / avg_time / succee_rate 等症状，可弱化或反向"
  }
}
```

### 共通判定标签

- `support`：当前策略支持该候选为真。
- `refute`：当前策略发现关键反证，排除或显著降权该候选。
- `inconclusive`：证据不足，需切换或加深其他策略。

### 共通判定补充

- 业务症状是重要证据，但不是所有原因族的必要条件。
- 对画像中已知的弱症状类型，允许“局部异常强 + peer 排他 + 语义或 trace/JDBC 辅证”直接支持候选。
- 反事实比较优先问：如果候选为假，是否还能解释局部排他性、层级分布和语义锚点；端到端时延是否上升是次级条件，而不是统一门槛。

### 共通输出格式

所有策略输出统一为：

```json
{
  "strategy": "策略名",
  "status": "support | refute | inconclusive",
  "ranked_candidates": [
    {
      "component": "组件名",
      "layer": "node | service | pod",
      "reason_family": "原因族",
      "score": 0.0,
      "verdict": "support | refute | inconclusive",
      "evidence": [
        "证据 1",
        "证据 2"
      ],
      "counterfactual_check": "本策略执行的关键反事实结论"
    }
  ],
  "stop_reason": "停止或切换原因",
  "next_action": "STOP | DEEPEN | SWITCH:Sx"
}
```

## 策略总览

| 编号 | 策略名 | 核心用途 | 优先适用 |
| --- | --- | --- | --- |
| S1 | 广撒网浅验证 | 在候选较多、根因层不清晰时快速筛选 | `Bank` 网络类，`Market` 初筛，`Telecom` node 网络类 |
| S2 | 单候选深挖验证 | 对资源/局部负载类候选做强反事实验证 | CPU / Memory / I/O / JVM / disk space 类 |
| S3 | 传播链回溯验证 | 从症状前沿沿依赖边向上游追根溯源 | `Market`、`Telecom`，以及有 trace 的 `Bank` |
| S4 | 副本-层级消歧验证 | 区分 node / service / pod 哪一层最能解释症状 | `Market` 多层场景，`Telecom` 混合层级 |
| S5 | 语义锚点验证 | 用日志/JDBC/状态切换确认离散事件型根因 | OOM、db close、db connection limit、process termination |

## S1 广撒网浅验证

### 定义

在候选集较大、故障类型尚不明确时，对多个候选执行低成本、多组件、浅层反事实检查。它的目标不是直接输出最终根因，而是快速回答两个问题：

1. 哪些候选明显不成立，应尽快剔除。
2. 哪一类候选最值得进入深挖或传播追踪。

该策略特别适合 `Bank` 的 packet loss / latency、`Telecom` 的 network delay / loss，以及 `Market` 初始候选很多、层级未收敛的场景。

### 输入

- `candidate_pool` 中最多前 `max_candidates` 个候选。
- 每个候选的 1 到 2 个最相关 KPI。
- 同层 peer 组件的同名 KPI。
- 基础业务症状。
- 可选的轻量日志/trace 摘要。

### 执行步骤

1. 对每个候选仅做最便宜的异常摘要。
计算 baseline 中位数、incident 中位数、变化方向、变化倍率，保留每个候选最强的 1 到 2 个异常指标。

2. 做同层排他性检查。
若候选组件的异常强度与同层 peer 相近，则该候选不能进入深挖优先队列；若明显高于 peer，则保留。

3. 做轻量症状一致性检查。
仅检查“是否存在与原因族相符的关键症状维度”，不把“时延必须上升”当统一模板。例如：
CPU / Memory / I/O / JVM 优先看局部 trace 时长、GC、throttling、peer 排他性；若业务症状弱，只记为弱证据。
network delay / loss / corruption 优先看 `mrt` / `avg_time`、`sr` / `succee_rate`、timeout/error、queue/TCP 状态异常。
network retransmission 允许业务症状近乎不变，但要求局部包量或链路指标与 peer 对照后具排他性。
db 类优先看 `success` / `succee_rate`、JDBC 成功率、连接或探活状态，而不是只看时延。

4. 做快速反事实否定。
如果某候选声称是 network delay / loss / corruption，但没有任何链路退化症状，也没有时延或成功率异常，则直接降权。
如果某候选属于画像中已知的弱症状类型，则“业务指标不恶化”只作为弱反证，不直接剔除。
如果某候选声称是资源类，但同层多个组件都出现同等资源抖动，且无局部排他或层级聚焦，则视为缺乏排他性。

5. 输出 top-k 候选，并给出下一跳策略建议。

### 停止条件

- 候选数已收敛到 1 到 2 个，进入 `S2` 或 `S3`。
- 所有候选都缺乏排他性，转入 `S4` 先做层级消歧。
- 某一候选出现明确语义硬锚点，直接转入 `S5`。
- 预算轮次耗尽，返回当前排序。

### 输出格式

```json
{
  "strategy": "S1_BROAD_SHALLOW",
  "status": "inconclusive",
  "ranked_candidates": [
    {
      "component": "Tomcat01",
      "layer": "pod",
      "reason_family": "net",
      "score": 0.71,
      "verdict": "support",
      "evidence": [
        "TCP-FIN-WAIT 在 incident 窗口显著上升",
        "metric_app.mrt 同步恶化"
      ],
      "counterfactual_check": "若 Tomcat01 不是网络故障源，则不应同时出现链路异常与业务时延恶化"
    }
  ],
  "stop_reason": "候选已收敛到 2 个",
  "next_action": "SWITCH:S3"
}
```

## S2 单候选深挖验证

### 定义

对单个高优先级候选执行强反事实验证，重点判断该候选是否同时满足“局部异常最强、时间上最早、替代解释最弱”；若存在清晰业务症状，还应能解释关键症状；若业务症状弱，则以局部排他性和跨模态辅证为主。这是资源类故障的主力策略，尤其适合：

- `Bank` 的 high CPU usage / high memory usage / high disk I/O / high disk space usage / high JVM CPU load / OOM
- `Market` 的 container CPU load / memory load / read-write I/O / node CPU / node CPU spike / node memory / node disk I/O / node disk space
- `Telecom` 的 CPU fault

### 输入

- 单个主候选及其 `reason_family`。
- 该候选完整 KPI 子集。
- 同层 peer KPI。
- 业务症状时间序列。
- 可选日志/trace 证据。

### 执行步骤

1. 计算局部异常强度。
围绕该候选提取同原因族 KPI，计算 baseline 与 incident 的偏移、极值、持续时长，形成局部异常画像。对 disk space 或个别反向指标，只要求“显著偏离基线”，不强依赖指标字面方向与故障名完全同向。

2. 验证时间先后关系。
检查候选 KPI 的首次越阈时间是否早于或不晚于最关键症状首次恶化时间。若端到端症状很弱，则改与 trace/JDBC 异常起点或 peer 分叉点比较。若候选信号明显滞后，则构成反证。

3. 验证同层排他性。
将该候选与同层 peer 做对照。如果 peer 也出现同幅度异常，则该候选更可能只是受害者或全局噪声。

4. 验证跨模态一致性。
按原因族检查是否存在匹配的伴随信号：
CPU 看 throttling / user time / JVM CPU；
Memory 看 failcnt / pgfault / free mem / GC；
I/O 看 await / queue / util / read-write；
disk space 看 `pct_usage` / `free` / `used` 等容量压力偏移；
JVM/OOM 看 `allocation failure`、`full gc`、heap 指标。

5. 验证替代解释是否更强。
若同时间窗内网络类或 DB 类语义锚点更硬，则把该候选从 `support` 降为 `inconclusive` 或 `refute`。

6. 输出该候选是否经得住反事实检验。

### 停止条件

- 候选满足“最强 + 最早 + 最可解释或最排他”，可停止并输出。
- 时间先后关系被破坏，或同层排他性失败，转入 `S3` 或 `S4`。
- 发现硬语义锚点指向其他原因族，转入 `S5`。

### 输出格式

```json
{
  "strategy": "S2_DEEP_SINGLE",
  "status": "support",
  "ranked_candidates": [
    {
      "component": "shippingservice-1",
      "layer": "pod",
      "reason_family": "mem",
      "score": 0.87,
      "verdict": "support",
      "evidence": [
        "container_memory_failcnt 在故障窗前后持续上升",
        "同服务其他副本未见同等强度异常",
        "metric_service.mrt 同窗口升高"
      ],
      "counterfactual_check": "若 shippingservice-1 不是根因，则无法解释其局部内存异常最强且早于业务症状"
    }
  ],
  "stop_reason": "候选通过强反事实验证",
  "next_action": "STOP"
}
```

## S3 传播链回溯验证

### 定义

从“受害症状最重的组件或调用边”出发，沿依赖关系向上游回溯，寻找最早出现局部异常且能解释下游广泛症状的起点。它的反事实核心是：

如果当前候选只是传播链上的受害者，那么它的上游应存在更早、更局部、更有解释力的异常源。

该策略尤其适合：

- `Market` 的服务间传播、网络退化、多副本服务链路
- `Telecom` 的 `docker -> db` JDBC 传播
- `Bank` 在 trace 可用时的横向 pod 传播

### 输入

- 当前 top 候选，或症状最重的组件集合。
- trace/span 依赖边或部署拓扑。
- 上游与下游组件的 KPI 摘要。
- 业务症状或 span 时延恶化摘要。

### 执行步骤

1. 定义症状前沿。
优先选择 incident 窗口内时延、错误率、timeout、失败率最差的组件或调用边；若业务症状较弱，则改用 trace/JDBC 最异常的依赖边、或同节点/同服务的异常簇作为回溯起点。

2. 沿依赖边逐跳回溯。
对每条上游边检查三个条件：
本地异常是否更强；
异常时间是否更早；
是否能覆盖更多下游受影响对象。

3. 标记“受害者模式”。
若某组件业务症状很重，但本地资源/语义异常很弱，而其上游边或上游组件异常更强，则该组件降为受害者。

4. 标记“源头模式”。
若某组件本地异常最早出现，且其下游多个对象同步恶化，则将其提升为根因源候选。

5. 若存在多个可能源头，交给 `S4` 做层级消歧，或交给 `S5` 查硬语义锚点。

### 停止条件

- 已找到单一上游起点，且下游覆盖充足。
- 回溯达到 `max_hops` 仍未收敛，转为 `inconclusive`。
- 拓扑不足或 trace 缺失，切换 `S2` 或 `S4`。

### 输出格式

```json
{
  "strategy": "S3_PROPAGATION_BACKTRACK",
  "status": "support",
  "ranked_candidates": [
    {
      "component": "db_003",
      "layer": "service",
      "reason_family": "db",
      "score": 0.82,
      "verdict": "support",
      "evidence": [
        "多个 docker->db_003 的 JDBC elapsedTime 同步恶化",
        "db_003 的连接相关指标先于应用侧症状异常",
        "下游多个 docker 组件均受影响"
      ],
      "counterfactual_check": "若 db_003 不是传播源，则无法解释多条下游 JDBC 边同时恶化"
    }
  ],
  "stop_reason": "传播链已收敛到单一上游源头",
  "next_action": "STOP"
}
```

## S4 副本-层级消歧验证

### 定义

用于判断根因应归到 `node`、`service` 还是 `pod`。其核心反事实是：

如果层级判断错误，那么异常在副本、节点共址关系、服务容错关系上的分布模式将不成立。

该策略是 `Market` 的关键策略，因为 `Market` 同时存在 node / service / pod 三层根因，且很多服务有多个 pod 副本；`Telecom` 也需要用它区分 node 网络类与 service/db 类问题。`Bank` 全部为 pod 层，可把此策略作为快速确认步骤。

### 输入

- 当前一个或多个候选。
- 部署关系：node -> pod -> service。
- 同服务副本、同节点共址组件的异常摘要。
- 业务症状覆盖范围。

### 执行步骤

1. 做副本对照。
若同一服务只有一个 pod 异常、其他副本正常，则优先判为 pod 级；若服务下多个副本一致恶化，则 pod 级解释减弱。

2. 做共址对照。
若同一 node 上多个不同服务/pod 同时出现相似异常，而其他 node 正常，则优先判为 node 级。

3. 做服务聚合对照。
若多个 pod 轻度异常但都属于同一 service，且症状主要体现在该 service 的调用质量，则优先判为 service 级。

4. 检查容错违背。
若单个 pod 理论上应被副本容错吸收，但业务症状依然大面积暴露，则说明真正根因可能位于共享上游 service 或 node。

5. 产出最小解释层级。
优先选择“能解释最多症状、引入最少额外假设”的层级。

### 停止条件

- 某一层级明显优于其他层级。
- 副本与共址模式相互矛盾，转入 `S3` 继续追传播源。
- 缺少部署信息，返回 `inconclusive`。

### 输出格式

```json
{
  "strategy": "S4_LAYER_DISAMBIGUATION",
  "status": "support",
  "ranked_candidates": [
    {
      "component": "node-6",
      "layer": "node",
      "reason_family": "io",
      "score": 0.79,
      "verdict": "support",
      "evidence": [
        "node-6 上多个不同服务 pod 同时出现 I/O 相关异常",
        "其他节点上的同服务副本未出现同等异常",
        "该层级能统一解释跨服务症状"
      ],
      "counterfactual_check": "若不是 node-6 层故障，则不应出现跨服务、同节点共址的同步异常"
    }
  ],
  "stop_reason": "node 层解释力显著高于 pod/service 层",
  "next_action": "STOP"
}
```

## S5 语义锚点验证

### 定义

利用日志、JDBC、状态位、离散错误事件等“硬语义证据”对候选做确认或反证。其核心反事实是：

某些根因类型如果真实存在，通常会留下难以被其他原因族替代解释的离散锚点；若锚点不存在，则相关候选应大幅降权。

它主要覆盖：

- `Bank` 的 JVM OOM / Full GC 类
- `Telecom` 的 db connection limit / db close
- `Market` 的 process termination、timeout/error/unavailable 类
- 一切资源指标不足以单独定性、但语义锚点比业务症状更稳定的场景

### 输入

- 当前 top 候选。
- incident 窗口内日志关键词、JDBC 指标、状态位、离散事件。
- 候选组件及其 peer 的语义锚点分布。
- 对应的业务症状时间窗。

### 执行步骤

1. 提取锚点事件。
例如：
`allocation failure`、`full gc`、`timeout`、`error`、`unavailable`、`tnsping_result_time`、`Login_Per_Sec`、`Session_pct`、`Proc_User_Used_Pct`、JDBC `success`、`Call_Per_Sec`、`container_threads`。

2. 做时间对齐。
要求锚点出现在 incident 窗口内，且不晚于核心业务症状太多；否则视为弱证据。

3. 做组件排他性检查。
若相同锚点在多个 peer 上普遍存在，则其区分力下降；若只集中在单一组件或单一依赖边，则候选加权。

4. 做原因族映射。
把锚点映射为更具体的原因族，例如：
`allocation failure/full gc` -> `jvm`
`timeout/unavailable` + 网络边退化 -> `net`
`JDBC success` / `succee_rate` down + `Session_pct` / `Proc_User_Used_Pct` up -> `db connection limit`
`tnsping_result_time` up + `Call_Per_Sec` down -> `db close`
`container_threads` 异常 + 副本或调用边断裂 -> `process`
单独 `exception` / `warning` -> 弱锚点，不能单独定性

5. 输出确认、反证或切换建议。

### 停止条件

- 存在单一组件、单一原因族的硬锚点，可直接停止。
- 锚点存在但不排他，转回 `S2` 或 `S3`。
- 完全无锚点，相关语义型候选降权。

### 输出格式

```json
{
  "strategy": "S5_SEMANTIC_ANCHOR",
  "status": "support",
  "ranked_candidates": [
    {
      "component": "db_007",
      "layer": "service",
      "reason_family": "db",
      "score": 0.9,
      "verdict": "support",
      "evidence": [
        "Session_pct 与 Proc_User_Used_Pct 在 incident 窗口同步上升",
        "JDBC success 下降且多个下游 docker 访问失败",
        "该模式与 db connection limit 高一致"
      ],
      "counterfactual_check": "若 db_007 不存在连接数受限，则不应出现连接指标与 JDBC 症状的同步恶化"
    }
  ],
  "stop_reason": "发现高排他性的 DB 语义锚点",
  "next_action": "STOP"
}
```

## 推荐调度顺序

### 默认顺序

1. 先用 `S1` 做候选压缩。
2. 若层级本身不清晰，优先插入 `S4`。
3. 若单个局部资源或负载候选已明显突出，进入 `S2`。
4. 若存在跨组件传播或 JDBC 扇出，进入 `S3`。
5. 若发现离散事件、GC/OOM、JDBC 状态或连接类强证据，随时抢占调用 `S5`。

### 按案例画像的快捷路由

- `Bank`
优先 `S1 -> S2`；若是 packet loss / latency，则 `S1 -> S3`；若出现 GC/OOM 词，则直接 `S5`。对 memory / disk / disk space / high JVM CPU load，不要求 app 侧 `mrt/sr` 必然恶化。

- `Market`
优先 `S1 -> S4`；若是 corruption / packet loss / network latency 且上下游边已明显恶化，再进 `S3`；若是 retransmission 或部分 node 资源类这类弱症状场景，则 `S1 -> S4 -> S2`；若日志有 timeout/error/unavailable 或 process termination 痕迹，补 `S5`。

- `Telecom`
CPU fault 走 `S2`；network delay / loss 走 `S1 -> S3`；db connection limit / db close 直接优先 `S5`，必要时再 `S3`。对 DB 类优先看 `succee_rate` 和 JDBC `success`，不强求 `avg_time` 上升。

## 设计原则总结

这套策略池刻意把“资源深挖”“传播回溯”“层级消歧”“语义确认”拆开，而不是把所有证据一次性混在一起。原因有三点：

1. OpenRCA 三个子数据集的可观测模态不一致。
`Bank` 偏 metric + log，`Telecom` 偏 metric + trace/JDBC，`Market` 则是多层部署 + 多模态混合。

2. 反事实验证本质上需要可切换、可终止、可比较。
Meta-Controller 只有在每个策略都明确输入、执行步骤、停止条件、输出格式时，才能真正学会“下一步做什么”，而不是重复做同一种检查。

3. 更新后的画像显示“业务症状弱或反向”并不罕见。
因此 Meta-Controller 不能把端到端时延升高当成统一前提，而应把它视为条件化证据：有则加权，无则回到局部排他性、传播结构和语义锚点。
