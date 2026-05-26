# PRISM: Probabilistic Relational Inference with Sentient Meta-Controller

## 面向 OpenRCA 的认知型多模态根因分析框架 v1.0

---

## 零、文档概述

### 0.1 本文档的定位

本文档是 PRISM 框架的**唯一权威设计规范**，包含：
- 数学定义（完整、可计算）
- 算法伪代码（可直接翻译为 Python）
- 组件接口规范（输入/输出类型、维度）
- 配置参数表（含默认值和调优范围）
- 实现路线图（分阶段，含验收标准）

后续所有代码实现以本文档为基准。本文档取代以下旧文档：
- `思路文档_反事实验证+动态思维链+资源调度_v0.1.md`
- `新框架数学设计_v0.1.md`

### 0.2 命名

**PRISM** = **P**robabilistic **R**elational **I**nference with **S**entient **M**eta-controller

### 0.3 核心创新（一句话）

**情绪向量作为信念几何→认知决策的可学习接口，反事实作为传播模型的 grounding 信号，多模态嫁接作为跨链证据收敛检测器，三者在同一数学框架内闭合。**

---

## 一、问题定义与数据特征

### 1.1 输入

| 符号 | 含义 | 类型 |
|------|------|------|
| $V$ | 实体集合 | set, $\|V\| \sim 20-60$ |
| $\mathcal{M}$ | metric 时序 | $\{(t, v, k, x)\}$ |
| $\mathcal{L}$ | log 数据（可选） | $\{(t, v, msg)\}$ |
| $\mathcal{T}$ | trace 数据 | $\{(t, v, tid, sid, pid, dur)\}$ |
| $t_0$ | 故障时间 | Unix timestamp |
| $\Delta$ | 基线/故障窗口 | 300s |

### 1.2 输出

- 根因组件 $r^* \in V$
- 根因原因类别
- 故障时间
- 完整推理轨迹（每步：动作、证据、信念、情绪）
- 最终因果图 $\mathbf{W}^*$

### 1.3 假设

**H1（单点故障）**：每 case 恰好一个根因 $r \in V$（已验证：Bank 136 cases, 14 unique roots）。

**H2（马尔可夫传播）**：异常沿有向边传播，到达 $v$ 的异常仅依赖于 $r$、图距离 $d(r,v)$、衰减 $\alpha$。

**H3（条件独立观测）**：给定 $r$ 和 $\mathbf{W}$，各实体观测条件独立。

### 1.4 OpenRCA 三系统数据特征

```
                  Bank              Telecom           Market
─────────────────────────────────────────────────────────────
模态              M + L + T          M + T              M + L + T
metrics 规模      1.3M              0.6M               7.7M
logs 规模         1.2M              ✗ 无               12.6M
traces 规模       12.2M             10.0M              9.1M
实体数 |V|        29                20+                60+ (双层)
实体类型          service/pod       node/container/    pod/service/node
                                    middleware
层级深度          1                  2                  3
根因组件种类      14                 15                 29
根因分布          Tomcat 密集        均匀                均匀
每 query 实体     29                 20+                cloudbed-1 + cloudbed-2
```

**关键差异维度**：
1. **模态缺失**：Telecom 无 logs → 语义信号盲区
2. **实体层级**：Market 三层跨 cloudbed → 传播路径更长
3. **数据噪声**：Market 7.7M+12.6M → 伪异常多，图推断过密
4. **根因分布**：Bank Tomcat 密集（信息论上更容易），Telecom/Market 均匀（更难）

---

## 二、核心数学框架

### 2.1 符号表

| 符号 | 含义 | 维度 |
|------|------|------|
| $\mathbf{a}^{obs}$ | 观测异常向量 | $\mathbb{R}^{\|V\|}$ |
| $\mathbf{W}$ | 边权矩阵 | $[0,1]^{\|V\| \times \|V\|}$ |
| $w_{uv}$ | 边 $u \to v$ 的置信度 | $[0,1]$ |
| $\boldsymbol{\mu}_r$ | 根因 $r$ 的传播预测 | $\mathbb{R}^{\|V\|}$ |
| $\alpha$ | 传播衰减系数 | $(0,1)$ |
| $p(r)$ | 根因信念分布 | $\Delta^{\|V\|-1}$ |
| $\mathbf{e}$ | 情绪向量 | $[0,1]^6$ |
| $\mathbf{W}_{frozen}$ | 确定性边集合 | $\{0,1\}^{\|V\| \times \|V\|}$ |
| $\sigma^2$ | 观测噪声方差 | 标量 |
| $c(d)$ | 动作 $d$ 的代价 | $\mathbb{R}^+$ |
| $EIG(d)$ | 期望信息增益 | bits |

### 2.2 系统状态定义

PRISM 在第 $t$ 步的状态是一个四元组：

$$\mathcal{S}_t = (p_t, \mathbf{W}_t, \mathbf{e}_t, \mathcal{H}_t)$$

- $p_t$：根因信念（在 $\|V\|$-单纯形上）
- $\mathbf{W}_t$：概率图边权矩阵
- $\mathbf{e}_t$：情绪向量（信念几何的压缩表示）
- $\mathcal{H}_t$：轨迹记忆库（历史 $(\mathbf{e}, d, \mathbf{e}', outcome)$ 记录）

---

## 三、Layer 1: 信号去噪

### 3.1 Metric 异常检测

**基线建模**（MAD-based Z-score）：
$$b_{vk} = \text{median}(x_{vk}^{baseline}), \quad \sigma_{vk} = \text{MAD}(x_{vk}^{baseline})$$
$$z_{vk}(t) = \frac{x_{vk}(t) - b_{vk}}{\sigma_{vk} + \epsilon}$$

**持久化过滤**（区分瞬时尖峰 vs 持续异常）：
$$\text{is\_anomalous}(v, k) = \mathbb{1}\left[P_{95}(z_{vk}^{fault}) > 3.0 \;\land\; \text{frac}(z_{vk}^{fault} > 2.0) > 0.3\right]$$

**实体异常分数**：
$$s_v^{obs} = \frac{1}{|K_v|} \sum_{k \in K_v} \mathbb{1}[\text{is\_anomalous}(v, k)] \cdot \min\left(\frac{P_{95}(z_{vk}^{fault})}{10}, 1\right)$$

### 3.2 Log 异常提取

$$l_v = \frac{\text{count\_error\_logs}(v)}{\max_{u} \text{count\_error\_logs}(u)} \cdot \text{fatal\_boost}(v)$$

$$\text{fatal\_boost}(v) = \begin{cases} 2.0 & \text{if } \exists \text{ keyword} \in \{\text{OOM Killed, SIGKILL, BindException, ...}\} \\ 1.0 & \text{otherwise} \end{cases}$$

### 3.3 统一异常信号

$$a_v^{obs} = \begin{cases}
\lambda_m \cdot s_v^{obs} + \lambda_l \cdot l_v & \text{if } \mathcal{L} \text{ available} \\
\lambda_m \cdot s_v^{obs} + \lambda_l \cdot \hat{l}_v & \text{if } \mathcal{L} \text{ missing (虚拟 log, 见 §6.2.2)}
\end{cases}$$

默认 $\lambda_m = 0.7, \lambda_l = 0.3$。

**输出**：$\mathbf{a}^{obs} \in \mathbb{R}^{|V|}$。

---

## 四、Layer 2: 概率依赖图

### 4.1 图结构

$$\mathbf{W} = [w_{uv}] \in [0,1]^{|V| \times |V|}$$

$w_{uv}$ = 异常从 $u$ 传播到 $v$ 的条件概率强度。

### 4.2 图初始化

**来源 1：Trace 确定性边**

$$w_{uv}^{trace} = \min\left(1.0, \frac{\text{call\_count}(u \to v)}{10}\right)$$
若 $w_{uv}^{trace} \geq 0.5$，边加入 $\mathbf{W}_{frozen}$（不可被后续修正消除）。

**来源 2：Metric 时序推断边**

$$w_{uv}^{metric} = \begin{cases} 0.5 & \text{if } t_v^{anomaly} - t_u^{anomaly} \in (0, 60s] \\ 0 & \text{otherwise} \end{cases}$$

**来源 3：LLM 先验建议边**

$$w_{uv}^{llm} \in [0.3, 0.6] \text{（LLM 输出，带置信度理由）}$$

LLM 边不是开放世界自由生成，而是只在候选对集合 $\mathcal{C}_{edge}$ 上打分：

$$\mathcal{C}_{edge} = \{(u,v) \mid v \in \mathcal{U}_{unexp},\; u \in \mathcal{P}(v)\}$$

其中：
- $\mathcal{U}_{unexp} = \{v \mid a_v^{obs} > \tau_a \land \text{explained\_ratio}(v) < \tau_{unexp}\}$ 是未解释异常实体
- $\mathcal{P}(v)$ 是 $v$ 的候选上游集合，仅包含 2-hop trace 邻居、同 service group、同 node/cloudbed 或 LLM 语义上强相关的组件

若边违反硬约束（自环、与 $\mathbf{W}_{frozen}$ 冲突、跨层非法跳转），则直接置 $w_{uv}^{llm}=0$。

**融合**：
$$w_{uv}^{(0)} = \begin{cases}
1.0 & \text{if } w_{uv}^{trace} \geq 0.5 \quad (\text{确定性，锁定}) \\
0.7 w_{uv}^{trace} + 0.3 w_{uv}^{metric} & \text{if } w_{uv}^{trace} > 0 \\
0.8 w_{uv}^{metric} + 0.2 w_{uv}^{llm} & \text{otherwise}
\end{cases}$$

### 4.3 自适应稀疏化

trace 数据量大 → 偶然共现多 → 需要更强稀疏化：

$$\lambda_1 = \lambda_1^{base} \cdot \left(1 + \gamma_{sparse} \cdot \log_{10}\left(\frac{N_{trace}}{|V| \cdot 10^6}\right)\right)$$

默认 $\lambda_1^{base} = 0.05, \gamma_{sparse} = 0.3$。

### 4.4 跨层传播衰减

$$\alpha_{uv} = \alpha_{base} \cdot \gamma_{layer(u) \to layer(v)}$$

层级转移矩阵：
```
              pod    service   node
pod           1.0    0.85     0.70
service       0.85   1.0      0.75
node          0.70   0.75     1.0
```

---

## 五、Layer 3: 传播模型与反事实 Grounding

### 5.1 阻尼加权传播

$$\boldsymbol{\mu}_r^{(0)} = \mathbf{e}_r \cdot \beta_r, \quad \beta_r = \max(0.5, a_r^{obs})$$

$$\boldsymbol{\mu}_r^{(t+1)} = \max\left(\mathbf{a}^{obs},\; \boldsymbol{\alpha} \odot (\mathbf{W}^T \boldsymbol{\mu}_r^{(t)})\right)$$

其中 $\boldsymbol{\alpha} = [\alpha_{uv}]$ 是层级感知衰减矩阵，$\odot$ 逐元素乘。

收敛条件：$\|\boldsymbol{\mu}_r^{(t+1)} - \boldsymbol{\mu}_r^{(t)}\|_\infty < 10^{-3}$，或 $t \geq \text{diam}(G) + 2$。

**观测似然**：
$$p(\mathbf{a}^{obs} \mid r, \mathbf{W}) = \frac{\exp(-\|\mathbf{a}^{obs} - \boldsymbol{\mu}_r\|_2^2 / 2\sigma^2)}{\sum_{u \in V} \exp(-\|\mathbf{a}^{obs} - \boldsymbol{\mu}_u\|_2^2 / 2\sigma^2)}$$

### 5.2 反事实 Grounding

**核心思想**：传播模型是"预测"，反事实是"实测"。残差驱动图修正。

**执行**：Counterfactual($v$) → 系统改善量 $\delta_v \in [0,1]$。

**残差**：
$$\Delta_v = \mathbb{E}_{r \sim p_t}[\mu_r(v)] - (1 - \delta_v)$$

三种情况：
- $\Delta_v > 0$（预测高估异常）→ 进入 $v$ 的边被高估 → M-step 降低上游 $w_{uv}$
- $\Delta_v < 0$（预测低估异常）→ 缺失传播路径 → 触发 LLM 边建议
- $\Delta_v \approx 0$ → 图在当前区域可信

**M-step 梯度**：
$$\frac{\partial \mathcal{L}}{\partial w_{uv}} \leftarrow -2\Delta_v \cdot \frac{\partial \mu_r(v)}{\partial w_{uv}}$$

### 5.3 EM 交替优化

**全局目标**：
$$\mathcal{L}(\mathbf{W}) = \mathbb{E}_{r \sim p_t}[\log p(\mathbf{a}^{obs} \mid r, \mathbf{W})] - \lambda_1 \|\mathbf{W}\|_1 - \lambda_2 \|\mathbf{W} - \mathbf{W}^{(0)}\|_F^2$$

**E-step**（固定 $\mathbf{W}$）：
$$p_t(r) \propto p(\text{evidence} \mid r, \mathbf{W}) \cdot p_{t-1}(r)$$

**M-step**（固定 $p_t$，坐标下降）：
$$w_{uv} \leftarrow w_{uv} + \eta_w \cdot \frac{\partial \mathcal{L}}{\partial w_{uv}}, \quad \eta_w = 0.02$$
$$w_{uv} \leftarrow \text{clamp}(w_{uv}, 0, 1)$$
$$\text{if } w_{uv} < 0.05 \text{ and } (u,v) \notin \mathbf{W}_{frozen}: w_{uv} \leftarrow 0$$

**触发条件**（非每步执行）：
- 累积 $K=3$ 个新证据后
- 或 $\max_r p_t(r) < 0.3$
- 或 LLM 建议了新候选边

每次触发执行 $M=5$ 次 EM 迭代。

---

## 六、Layer 4: 多模态推理链与嫁接

### 6.1 模态链激活

```
active_modalities = {m ∈ {M, L, T} | modality_available(m)}
```

每条链 $m$ 维护独立的信念分布 $p_t^m(r)$ 和推理轨迹。

**metric 链**：
$$p_t^M(r) \propto p(\mathbf{a}^{obs} \mid r, \mathbf{W}) \cdot p_0^M(r)$$

**log 链**（仅当 $\mathcal{L}$ 可用）：
$$p_t^L(r) \propto p(\text{log\_fatal}(r) \mid r) \cdot p_0^L(r)$$

其中 $p(\text{log\_fatal}(r) \mid r)$ 基于 fatal keyword 命中率建模。

**trace 链**：
$$p_t^T(r) \propto p(\text{graph\_centrality}(r) \mid r) \cdot p_0^T(r)$$

其中 $p(\text{graph\_centrality}(r) \mid r)$ 基于 $r$ 在图中的入度/出度异常建模。

### 6.2 缺模态补偿（Telecom 无 logs 用）

Telecom 没有原始 log，因此 PRISM 不把"无 log"视为"无语义"，而是显式构造一个**缺模态补偿层**：

1. 用传播反推回答"谁更像上游解释者"
2. 用虚拟 log 分数回答"谁更像会在 log 里留下离散故障锚点"
3. 将两者只作为**辅助证据**注入，不单独替代真实 log 链

#### 6.2.1 传播反推分数

当 log 链缺失时，用逆向图推断补偿：

$$s_r^{backprop} = \frac{\sum_{v \neq r} a_v^{obs} \cdot \mathbb{1}[r \text{ is ancestor of } v]}{\sum_{v \neq r} \mathbb{1}[r \text{ is ancestor of } v] + 1}$$

它刻画某个候选节点 $r$ 对下游异常的"可解释覆盖度"。若 $r$ 能覆盖更多异常节点，且这些节点异常更强，则 $s_r^{backprop}$ 更高。

#### 6.2.2 虚拟 log 分数

缺 log 时，为每个实体构造一个伪语义锚点分数 $\hat{l}_v$：

$$\hat{l}_v = \sigma\left(
\omega_1 \cdot s_v^{backprop}
+ \omega_2 \cdot s_v^{jdbc}
+ \omega_3 \cdot s_v^{exclusive}
+ \omega_4 \cdot s_v^{state}
\right)$$

其中：
- $s_v^{backprop}$：传播反推分数，刻画该点对下游异常的解释力
- $s_v^{jdbc}$：JDBC/trace 异常锚点，刻画 `elapsedTime`、`success`、调用错误率等离散退化
- $s_v^{exclusive}$：同层排他性分数，刻画"异常是否集中在该点附近而非全局噪声"
- $s_v^{state}$：状态切换代理分数，刻画 queue、session、connection、tnsping 等更接近"事件"而非连续波动的指标

默认 $(\omega_1,\omega_2,\omega_3,\omega_4) = (0.35, 0.30, 0.20, 0.15)$。

这不是生成自然语言日志，而是生成一个与真实 $l_v$ 同量纲的**伪 log 强度**，专门用于前述统一异常信号融合。

#### 6.2.3 Telecom 特化代理锚点

对 Telecom，$\hat{l}_v$ 的四项代理来源进一步特化为：

- `db connection limit` / `db close`：优先使用 `JDBC.success` 下降、`Session_pct` 上升、`Login_Per_Sec` 异常、`tnsping_result_time` 激增
- `network delay` / `network loss`：优先使用 `Received_queue`、`Sent_queue`、`ICMP_ping`、`System_wait_queue_length`
- `CPU fault`：优先使用 `container_cpu_used` 的局部排他性和其下游 JDBC `elapsedTime` 放大
- `os` 级根因：要求节点指标异常与下游多服务 JDBC 症状同时出现，避免把单个受害容器误判成上游根因

#### 6.2.4 注入规则与防御边界

虚拟 log 不是新模态链，而是辅助 metric 链：

$$a_v^{obs} \leftarrow \lambda_m \cdot s_v^{obs} + \lambda_l \cdot \hat{l}_v, \quad
\lambda_l^{virt} = \lambda_l \cdot c_v^{virt}$$

其中 $c_v^{virt}$ 是缺模态补偿置信度：

$$c_v^{virt} = \min\left(1,\; 0.5 \cdot q_v^{struct} + 0.3 \cdot q_v^{peer} + 0.2 \cdot q_v^{temporal}\right)$$

- $q_v^{struct}$：候选实体是否位于当前 top-$k$ 根因假设的传播主干上
- $q_v^{peer}$：是否具有同层排他性
- $q_v^{temporal}$：代理锚点是否早于或同步于关键症状

防御规则：
1. 若 $c_v^{virt} < 0.35$，则强制 $\hat{l}_v = 0$，避免把连续噪声误当事件锚点
2. 虚拟 log 只能增强已异常实体，不能让一个原本 $s_v^{obs} \approx 0$ 的实体凭空跃迁为高优先级候选
3. 若真实 log 存在，则始终优先使用真实 $l_v$，虚拟 log 不参与融合

### 6.3 模态置信度自校准

$$c^M = \frac{\max_r p^M(r) - 1/|V|}{1 - 1/|V|} \cdot (1 - H_{norm}(p^M))$$

$$c^L = \frac{|\{v : \text{fatal\_log}(v)\}|}{\max(1, |\{v : \text{has\_log\_error}(v)\}|)}$$

$$c^T = \frac{|\{(u,v) : w_{uv}^{trace} \geq 0.5\}|}{\max(1, |E|)}$$

$c^L = 0$ 在 Telecom（自动降权）。

### 6.4 嫁接条件与融合

**嫁接检测**：
$$\text{can\_graft}(A, B) \iff KL(p^A \parallel p^B) < \tau_{graft}$$

$\tau_{graft} = 0.5$。

**嫁接融合**（置信度加权 + 共识增益）：
$$p_{grafted}(r) \propto \frac{\sum_{m \in \text{active}} c^m \cdot p^m(r)}{\sum_{m \in \text{active}} c^m} \cdot \exp(\gamma \cdot \text{agreement}(r))$$

$$\text{agreement}(r) = \mathbb{1}[r = \arg\max_{r'} p^A(r') = \arg\max_{r'} p^B(r')]$$

$\gamma = 0.5$（共识增益系数）。

**冲突标记**：如果 $KL(p^A \parallel p^B) > 1.5$ 且 $c^A, c^B > 0.3$ → 两个模态给出矛盾结论 → **高优先级诊断信号**，触发额外 exploration。

---

## 七、Layer 5: 情绪向量

### 7.1 定义

情绪向量 $\mathbf{e} \in [0,1]^6$ 是信念分布 $p(r)$ 和图 $\mathbf{W}$ 的**可微压缩表示**。

| 维度 | 符号 | 数学定义 | 认知含义 |
|------|------|---------|---------|
| $e_1$ | conviction | $\max_r p(r)$ | 对 top-1 的确信度 |
| $e_2$ | curiosity | $H_{norm}(p) \cdot (1 - \text{coverage})$ | 未探索空间引力 |
| $e_3$ | perplexity | $\text{conflict\_ratio} \cdot KL(p \parallel p_{prev})$ | 证据矛盾 + 信念突变 |
| $e_4$ | vigilance | $\text{fragility}$（闭式近似） | 排名翻转风险 |
| $e_5$ | satiety | $\text{coverage} \cdot \sigma(iter - 3)$ | 证据充分度 |
| $e_6$ | anxiety | $(1 - \text{budget\_ratio}) \cdot (1 - e_1)$ | 资源压力 × 不确定性 |

**辅助统计量**：
$$H_{norm}(p) = -\frac{\sum_r p(r) \log p(r)}{\log |V|}, \quad \text{coverage} = \frac{|\{v: \text{has\_evidence}(v)\}|}{|V|}$$

$$\text{conflict\_ratio} = \frac{|\{v: \text{support}(v) \land \text{refute}(v)\}|}{|V|}$$

$$\text{fragility} = \max\left(0, 1 - \frac{gap}{0.3}\right), \quad gap = p(r_1) - p(r_2)$$

### 7.2 关键性质：可微性

$$\frac{\partial \mathbf{e}}{\partial p} \text{ 和 } \frac{\partial \mathbf{e}}{\partial \mathbf{W}} \text{ 存在且可计算。}$$

情绪向量可以通过梯度反向传播到图参数和动作选择——构成端到端可学习系统。

### 7.3 理想情绪参考点

$$\mathbf{e}_{ideal} = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)$$

高 conviction、高 satiety、其他为零 = 完美收敛状态。

---

## 八、Layer 6: Sentient Meta-Controller

### 8.1 决策函数

$$d^* = \arg\max_d \underbrace{(1 - \beta(\mathbf{e})) \cdot \frac{EIG(d)}{c(d)}}_{\text{理性分量: 信息论最优}} + \underbrace{\beta(\mathbf{e}) \cdot \Phi(\mathbf{e}, d)}_{\text{情绪分量: 认知直觉}}$$

### 8.2 情绪信任度 $\beta(\mathbf{e})$

$$\beta(\mathbf{e}) = \sigma(w_\beta \cdot (e_3 + e_4 + e_6) + b_\beta)$$

perplexity 高 + vigilance 高 + anxiety 高 → $\beta$ 大 → 信任情绪直觉。
conviction 高 + satiety 高 → $\beta$ 小 → 信任精确 EIG 计算。

初始：$w_\beta = 1.5, b_\beta = -2.5$（语义先验，后续在线调整）。

### 8.3 策略效果预测器 $\Phi(\mathbf{e}, d)$

$$\Phi(\mathbf{e}, d) = \underbrace{\cos\_sim(\mathbf{e}, \mathbf{e}_{ideal})}_{\text{向理想情绪靠近}} + \underbrace{\mathbb{E}_{trajectory}[\Delta \mathbf{e} \mid \mathbf{e}, d]}_{\text{历史轨迹中的预期情绪变化}}$$

历史期望从轨迹记忆库中通过余弦相似度检索：

$$\mathbb{E}[\Delta \mathbf{e} \mid \mathbf{e}, d] = \frac{\sum_{(\mathbf{e}_i, d_i, \Delta\mathbf{e}_i) \in \mathcal{H}} \cos\_sim(\mathbf{e}, \mathbf{e}_i) \cdot \Delta\mathbf{e}_i}{\sum \cos\_sim(\mathbf{e}, \mathbf{e}_i)}$$

### 8.4 期望信息增益 $EIG(d)$

$$EIG(d) = \sum_{y_k} p(y_k \mid d) \cdot [H(p_t) - H(p_{t+1} \mid y_k)]$$

$$p(y_k \mid d) = \sum_{r \in V} p(y_k \mid r, d) \cdot p_t(r)$$

$$H(p_{t+1} \mid y_k) = -\sum_r \frac{p(y_k \mid r, d) \cdot p_t(r)}{p(y_k \mid d)} \cdot \log \frac{p(y_k \mid r, d) \cdot p_t(r)}{p(y_k \mid d)}$$

**离散化**：COUNTERFACTUAL $\delta_v \in [0,1]$ → $N=10$ bins。LLM score → $N=5$ bins。

### 8.5 动作空间

| 动作 $d$ | 证据 | 代价 $c(d)$ | 似然形式 |
|-----------|------|------------|---------|
| METRIC_DEEP($v$) | $v$ 的各指标 Z-score | 0.1s | $\mathcal{N}(\cdot \mid \mu_r(v), \sigma^2)$ |
| LOG_QUERY($v$) | $v$ 的错误日志 | 0.5s | Bernoulli(· | $\mu_r(v)$) |
| TRACE_VERIFY($u$,$v$) | 边存在高置信判定 | 2s | Bernoulli(· | $w_{uv}$) |
| COUNTERFACTUAL($v$) | $\delta_v$ 系统改善量 | 60s | $\mathcal{N}(\cdot \mid v=r, 0.1^2)$ |
| LLM_SEMANTIC($v$) | $v$ 的根因语义评估 | 2s | Beta(· | $v=r$) |
| LLM_SUGGEST_EDGE | 可能缺失的边 | 2s | 先验 $w_{uv}^{(llm)}$ |

### 8.6 动作剪枝

| 动作类型 | 剪枝条件 |
|---------|---------|
| COUNTERFACTUAL | 只评 $p_t(r) > 0.05$ 的实体（2-5 个），先用闭式近似，EIG/c 超阈值才精算 |
| LLM 类 | 只在 $\max p_t < 0.3$ 或 $gap < 0.1$ 或 EM 后无改善时评估 |
| METRIC_DEEP | 全评（代价极低） |
| TRACE_VERIFY | 只评 $w_{uv} \in (0.1, 0.9)$ 的不确定边 |

### 8.7 Stop Projection（在线学习）

$$\text{readiness}(\mathbf{e}) = \sigma(\mathbf{w}^T \mathbf{e} + b)$$

初始：$\mathbf{w} = (0.30, -0.10, -0.25, -0.20, 0.20, 0.25), b = 0$（语义先验）。

Stop Projection 不直接替代收敛判定，而是作为"是否值得继续探索"的软判断器：

$$\text{STOP} \iff \text{readiness}(\mathbf{e}_t) > \tau_{stop} \land gap_t > \tau_{gap} \land \max_d EIG(d)/c(d) < \tau_{eig}^{soft}$$

其中：
- $gap_t = p_t(r_1) - p_t(r_2)$ 是 top-1/top-2 信念间隔
- $\tau_{stop}=0.80$，$\tau_{gap}=0.15$，$\tau_{eig}^{soft}=0.02$
- 若触发资源兜底（$T_{max}, N_{cf}^{max}, N_{llm}^{max}$）则允许强制停止，但该样本会记为"被资源截断"

**监督信号生成**：

每次系统决定停止后，生成标签 $target \in \{0,1\}$：

$$target = \mathbb{1}\left[\text{StableTop1AfterOneMoreStep} \land \text{NoSignificantGain}\right]$$

其中：
- `StableTop1AfterOneMoreStep`：在 shadow mode 再执行 1 个 cheapest-highest-EIG 动作后，top-1 根因不变
- `NoSignificantGain`：追加一步后的 $\max_r p(r)$ 提升 $< 0.03$ 且最终答案类型不发生变化
- 若离线评测时有真值标签，则可额外要求停止时 top-1 为真根因，此时该标签优先于 shadow label

**在线更新目标函数**：

$$\mathcal{L}_{stop} = - target \log(\text{readiness}) - (1-target)\log(1-\text{readiness}) + \lambda_{stop}\|\mathbf{w}-\mathbf{w}_0\|_2^2$$

其中 $\mathbf{w}_0$ 是初始语义先验权重，$\lambda_{stop}=0.01$ 用于防止小样本早期漂移。

SGD 更新（每次停止后）：
$$\mathbf{w} \leftarrow \mathbf{w} - lr \cdot (\text{readiness} - target) \cdot \text{readiness} \cdot (1 - \text{readiness}) \cdot \mathbf{e}$$
$$target = 1.0 \text{ if stop was correct, } 0.0 \text{ otherwise}$$

$$lr = 0.05$$

偏置项同步更新：

$$b \leftarrow b - lr \cdot (\text{readiness} - target)$$

**安全更新机制**：
1. 梯度裁剪：$\|\nabla\|_2 \leq 1.0$
2. 单次更新后若 $\|\mathbf{w}-\mathbf{w}_{prev}\|_2 > 0.2$，则投影回半径 0.2 球内
3. 按 system 分开维护 $(\mathbf{w}, b)$，避免 Bank 的停止偏好污染 Telecom/Market
4. 冷启动阶段（每个 system 样本数 $<20$）时，更新采用 replay mini-batch（最近 20 条停止样本）而不是单样本 SGD

**产出与用途**：
- 在线阶段：作为早停门控器，减少无收益的额外 counterfactual / LLM 调用
- 离线阶段：统计 stop precision / stop recall / 过早停止率，为消融实验提供独立指标

### 8.8 轨迹记忆库

存储四元组：$(\mathbf{e}_{before}, d, \mathbf{e}_{after}, outcome)$。

用途：
1. 策略效果预测 $\mathbb{E}[\Delta \mathbf{e} \mid \mathbf{e}, d]$
2. Stop projection 在线训练
3. 新 case warm-start（检索相似初始情绪状态）

容量上限：1000 条。FIFO 淘汰。按 system + reason_family 分区。

### 8.9 收敛判定（三元条件）

1. **信念集中**：$\max_r p_t(r) > \tau_p = 0.80$
2. **边际信息增益下降**：$\max_d EIG(d)/c(d) < \tau_{eig} = 0.01$
3. **图结构稳定**：$\|\mathbf{W}_t - \mathbf{W}_{t-1}\|_F < \tau_w = 0.02$

**资源兜底**：$T_{max} = 15$，$N_{cf}^{max} = 3$，$N_{llm}^{max} = 5$。

---

## 九、自适应机制汇总

| # | 机制 | 解决什么 | 关键公式 | 激活条件 |
|---|------|---------|---------|---------|
| 1 | 传播反推补偿 | Telecom 无 logs | $s_r^{backprop}$ | $\mathcal{L}$ 缺失 |
| 2 | 自适应先验强度 | 数据稀疏度不同 | $\eta(\mu)$ | 自动（按 $\mu$） |
| 3 | 图稀疏化自适应 | trace 量级不同 | $\lambda_1(N_{trace})$ | 自动（按 $N_{trace}$） |
| 4 | 跨层传播衰减 | 实体层级不同 | $\alpha_{uv} = \alpha \cdot \gamma_{layer}$ | 多层级系统 |
| 5 | 模态置信度自校准 | 各模态信噪比不同 | $c^m$ 加权融合 | 自动 |
| 6 | 虚拟模态推理 | 缺模态但需推理 | $\hat{l}_v$ | $\mathcal{L}$ 缺失 |
| 7 | LLM 补边暂挂注入 | 防止幻觉边直接污染主图 | $E_{probe} \rightarrow E_{promote}$ | 残差显著且低置信 |
| 8 | Stop Projection 在线学习 | 动态早停而非固定阈值 | $\text{readiness}(\mathbf{e})$ | 每次停止后 |

---

## 十、LLM 集成设计（精确两处）

### 10.1 位置 1：初始先验注入（初始化时）

**触发**：算法启动，图初始化后。

**输入**：结构化异常摘要 + log 摘录（如有）。

**输出**：$p_{llm}(r)$ 分布（$\|V\|$ 维概率向量）+ 推理理由。

**融合**：$p_0(r) \leftarrow (1 - \eta) \cdot p_0^{math}(r) + \eta \cdot p_{llm}(r)$。

**$\eta$ 自适应**：$\eta = 0.4 \cdot (1 - \mu)$，其中 $\mu$ 是模态覆盖率。Telecom $\eta \approx 0.22$，Bank $\eta \approx 0.12$，Market $\eta \approx 0.04$。

#### 10.1.1 输入规范

LLM 不直接看到原始全量遥测，而是只接收压缩后的 case card：

```json
{
  "task": "root_cause_prior",
  "system": "Bank|Telecom|Market",
  "top_anomalies": [
    {"entity": "Tomcat01", "score": 0.78, "top_metric": "cnt", "z_p95": 4.2, "layer": "service"},
    {"entity": "Mysql02", "score": 0.65, "top_metric": "mrt", "z_p95": 3.8, "layer": "service"}
  ],
  "fatal_logs": {
    "Tomcat01": ["connection pool exhausted", "upstream timeout"]
  },
  "trace_hints": [
    {"from": "Tomcat01", "to": "Mysql02", "strength": 0.9}
  ],
  "query": "identify the most likely faulty component"
}
```

压缩原则：
- 每个实体最多保留 3 个 metric 摘要、2 条 log 摘录、2 条 trace 提示
- 只保留 top-$k_{prior}=8$ 个异常实体，避免长上下文稀释
- 对 Telecom 缺 log 场景显式标注 `"logs_missing": true`，让 LLM 降低对语义证据的依赖

#### 10.1.2 输出规范与校准

LLM 输出必须是结构化 JSON：

```json
{
  "prior_scores": [
    {"entity": "Tomcat01", "score": 0.62, "reason": "pool exhaustion usually originates near the overloaded service"},
    {"entity": "Mysql02", "score": 0.21, "reason": "slow backend can induce upstream blocking"}
  ],
  "confidence": 0.68,
  "abstain": false
}
```

将 LLM 分数温度归一化为概率分布：

$$p_{llm}(r) = \frac{\exp(s_r / T_{llm})}{\sum_{u \in \mathcal{C}_{prior}} \exp(s_u / T_{llm})}, \quad T_{llm}=1.2$$

其中 $\mathcal{C}_{prior}$ 是候选根因集合，仅包含：
- top-$k_{prior}$ 异常实体
- fatal log 命中的实体
- trace 图中高 out-degree 且自身异常的实体

非候选实体强制置零，再统一归一化。

#### 10.1.3 自一致性与接受门槛

LLM 先验默认采样 3 次，得到 $\{p_{llm}^{(1)}, p_{llm}^{(2)}, p_{llm}^{(3)}\}$。定义：

$$q_{cons} = \frac{1}{3}\sum_{i=1}^{3}\mathbb{1}\left[\arg\max p_{llm}^{(i)} = \text{mode-top1}\right]$$

$$q_{evidence} = \frac{|\{\text{reasons 引用的实体}\} \cap \mathcal{C}_{prior}|}{\max(1, |\{\text{reasons 引用的实体}\}|)}$$

$$q_{llm} = 0.6 q_{cons} + 0.4 q_{evidence}$$

最终融合强度改为：

$$\eta_{case} = \eta_{base} \cdot (1 - \mu) \cdot q_{llm}$$

接受规则：
- 若 $q_{llm} < 0.35$ 或 `abstain=true`，则不注入 LLM 先验，退回纯数学先验
- 若 3 次采样 top-1 完全不一致，则只保留理由文本，不进入概率融合
- 若 LLM 给出的 top-1 不在 $\mathcal{C}_{prior}$，视为越权输出，整次响应丢弃

#### 10.1.4 作用边界

LLM 先验只能改变初始化信念，不能覆盖后续证据更新：

$$p_t(r) \propto p(\text{evidence}_{1:t}\mid r, \mathbf{W}) \cdot p_0(r)$$

因此即便 LLM 初始判断错误，只要后续反事实和传播残差持续反驳，该偏差会在若干步内被冲淡。

### 10.2 位置 2：图补全边建议（LLM 专用动作）

**触发**：EM 图更新后 $\max_r p_t(r) < 0.3$（传播模型无法解释观测，图可能缺边）。

**输入**：当前拓扑摘要 + 未解释的异常实体列表。

**输出**：$\{(u, v, confidence, reason)\}$ 建议边列表。

**注入**：$w_{uv}^{(llm)} = confidence$，作为新的边源参与 $\mathbf{W}$ 初始化融合。后续 EM 自适应调整。

**多次采样交验**（热乱 ensembling）：LLM 对同一输入跑 3 次，取 2+ 次出现的边作为高置信建议（$w \times 1.2$），仅 1 次的边降权（$w \times 0.6$）。

#### 10.2.1 触发的精细条件

不是所有低置信 case 都允许 LLM 补边，需同时满足：

$$\max_r p_t(r) < 0.3 \;\land\; |\mathcal{U}_{unexp}| \geq 1 \;\land\; \|\mathbf{W}_t - \mathbf{W}_{t-1}\|_F < \tau_w$$

即：
- 信念仍分散
- 至少存在 1 个显著未解释异常点
- 近期 EM 已基本收敛，说明问题不在参数微调而在结构缺失

#### 10.2.2 候选对生成而非自由生成

LLM 不负责在 $|V|^2$ 空间里自由造边，而是只对预筛后的候选对打分：

```json
{
  "task": "edge_suggestion",
  "unexplained_nodes": [
    {"entity": "MG01", "anomaly": 0.72, "explained_ratio": 0.11}
  ],
  "candidate_parents": {
    "MG01": ["Tomcat02", "Mysql02", "Redis01"]
  },
  "current_edges": [
    {"from": "Tomcat02", "to": "IG01", "confidence": 0.81}
  ],
  "forbidden_edges": [
    {"from": "MG01", "to": "MG01", "reason": "self-loop"},
    {"from": "node01", "to": "pod99", "reason": "illegal layer jump"}
  ]
}
```

候选父节点由以下规则并集生成：
1. 2-hop trace 邻居
2. 时序上领先于 $v$ 的高异常实体
3. 同 service group / 同 node / 同 cloudbed 的运维邻近实体
4. 当前 top-$3$ 根因假设所在子图中的关键节点

#### 10.2.3 接受分数与暂挂注入

对于每条候选边，定义补边接受分数：

$$q_{edge}(u,v) = c_{llm}(u,v) \cdot q_{cons}(u,v) \cdot q_{struct}(u,v) \cdot (1 + \rho \cdot \text{residual}_v)$$

其中：
- $c_{llm}(u,v)$ 是 LLM 返回置信度
- $q_{cons}(u,v) \in \{0.6, 1.0\}$ 表示 3 次采样中该边是否达到多数一致
- $q_{struct}(u,v) \in \{0,1\}$ 表示是否满足结构约束
- $\text{residual}_v = \max(0, -\Delta_v)$ 衡量该点被当前传播模型低估的程度
- $\rho = 0.5$

边权按下式裁剪：

$$w_{uv}^{llm} = \text{clip}(q_{edge}(u,v), 0.1, 0.6)$$

新边先进入暂挂集合 $E_{probe}$，而不是立即并入稳定图：
- 先执行 1 轮 EM，仅允许 $E_{probe}$ 中的边参与更新
- 若对数似然提升 $\Delta \mathcal{L} > 0.01$ 且边权未被稀疏化剪掉，则提升到 $E_{promote}$
- 否则删除该边，并在当前 case 中加入黑名单，避免反复向同一幻觉边付费

#### 10.2.4 硬约束与兜底

LLM 补边必须满足以下硬约束：
1. 不允许自环
2. 不允许覆盖或反转 $\mathbf{W}_{frozen}$ 中的确定性边
3. 不允许跨 cloudbed 非法跳边，除非 trace 已观测到弱连接
4. 单次调用最多返回 5 条边，全局累计最多提升 10 条 LLM 边

若补边后仍无提升，则系统退回"图已足够，问题在根因排序"假设，不再继续申请 LLM 边调用，避免预算耗尽。

#### 10.2.5 RE3-OB -> OpenRCA 的 LLM 补边代理验证协议

在 OpenRCA 上直接放开 LLM 补边风险较高，因此先用 RE3-OB 做**代理验证**。目标不是证明 RE3-OB 与 OpenRCA 完全同分布，而是验证补边代理是否具备三项最低能力：

1. 只在候选对空间内补边，而不是开放式造边
2. 被遮蔽后能恢复关键传播结构，而不是只给语义上合理但结构无用的边
3. 注入后能改善根因排序或传播解释度，而不是仅提高图密度

**协议步骤**：

1. **构造代理样本**
   - 从 RE3-OB 中选取具有较完整多模态和较清晰传播路径的 case
   - 每个 case 保留真值根因、真实依赖图或可观测 trace 图、异常摘要

2. **执行遮蔽**
   - 遮掉 log 与 trace 主体，只保留 metrics、基础拓扑和候选父节点集合
   - 对部分 case 进一步遮掉与真根因直接相关的 1-hop 边，模拟 OpenRCA 中"图结构缺边但仍有异常残差"的场景

3. **统一补边输入**
   - 使用与 OpenRCA 完全同构的 `edge_suggestion` case card
   - 候选父节点仍只来自 2-hop 邻居、时序领先点、运维邻近点、top-3 根因假设子图

4. **执行补边代理**
   - LLM 跑 3 次采样，按前述 $q_{cons}$ 和 $q_{edge}$ 规则聚合
   - 所有返回边先进入 $E_{probe}$，不得直接提升到稳定图

5. **离线评估**
   - 与被遮蔽前的真实边集 $E_{gold}$ 对比
   - 在注入前后分别运行传播与根因排序，比较是否带来实际收益

**评估指标**：

$$Precision@k = \frac{|E_{pred}^{@k} \cap E_{gold}|}{|E_{pred}^{@k}|}, \quad
Recall@k = \frac{|E_{pred}^{@k} \cap E_{gold}|}{|E_{gold}^{masked}|}$$

$$LegalRate = \frac{|\{e \in E_{pred}: e \text{ 满足全部结构约束}\}|}{|E_{pred}|}$$

$$Gain_{rank} = p_{after}(r^*) - p_{before}(r^*)$$

$$Gain_{fit} = \cos(\mu_{after}, a^{obs}) - \cos(\mu_{before}, a^{obs})$$

其中：
- $E_{gold}^{masked}$ 是本轮被刻意遮蔽、理论上可恢复的边集合
- $r^*$ 是真值根因
- $Gain_{rank}$ 衡量补边对真根因排序的帮助
- $Gain_{fit}$ 衡量补边后传播解释度是否提高

**通过门槛**：

1. `LegalRate = 100%`，即不能出现越层、自环、冲突边
2. `Precision@3 >= 0.50` 或相对随机/启发式候选基线提升至少 `+0.20`
3. 平均 `Gain_{fit} > 0`
4. 至少 60% 的 case 满足 `Gain_{rank} >= 0`，且不能因为补边导致 top-1 正确样本显著变差

只有通过上述门槛，LLM 补边代理才允许迁移到 OpenRCA 的在线推理流程。

#### 10.2.6 OpenRCA 迁移验收

代理验证通过后，OpenRCA 侧采用更保守的灰度验收：

1. 先在 `Telecom` 子集启用，因为它最具缺模态补偿需求
2. 仅对满足 `NeedLLMEdgeRepair()` 条件的低置信 case 开启
3. 记录 `LLM-called / probe-promoted / promoted-effective` 三层漏斗
4. 对比不开启补边时的 `Correct`、`C+P`、平均步数、平均 LLM 调用数

若 OpenRCA 灰度阶段出现以下任一情况，则回退到"只保留先验注入，不启用在线补边"：
- `promoted-effective` 低于 30%
- 平均 `Gain_{fit} <= 0`
- 非法边虽被规则拦截，但高频触发说明候选对生成或提示词设计失真

---

## 十一、完整算法

```
Algorithm: PRISM
Input: V, M, L (optional), T, t0
Output: r*, reasoning_trace

 1:  // ── 模态检测 ──
 2:  active  ← {M, L, T} ∩ available
 3:  μ       ← ModalityCoverage(active, M, L, T)
 4:  layers  ← InferLayerTypes(V, schema)
 5:  λ1      ← AdaptiveSparsity(|T|, |V|)            // §4.3
 6:
 7:  // ── Layer 1: 信号去噪 ──
 8:  s_obs   ← MetricAnomalyDetection(M, t0)          // §3.1
 9:  if L available:
10:      l   ← LogErrorAggregation(L, t0)              // §3.2
11:  else:
12:      l̂   ← VirtualLogScore()  // placeholder        // §6.2.2
13:      l   ← l̂
14:  a_obs   ← SignalFusion(s_obs, l)                  // §3.3
15:
16:  // ── Layer 2: 图初始化 ──
17:  W_trace ← TraceEdgeExtraction(T)                  // §4.2
18:  W_metric← MetricEdgeInference(a_obs)              // §4.2
19:  W_llm   ← InitEmptyLLMEdgePrior()                 // 占位；仅在后续残差显著时补边
20:  W       ← EdgeFusion(W_trace, W_metric, W_llm)    // §4.2
21:  W_frozen← {(u,v) | w_trace_uv ≥ 0.5}
22:
23:  // ── 信念初始化 ──
24:  p_0_math← PriorFromAnomaly(a_obs, W)              // out-degree惩罚
25:  p_llm, q_llm ← LLMPrior(case_card)    // if LLM   // §10.1
26:  η       ← AdaptivePriorWeight(μ, q_llm)           // §10.1
27:  p_0     ← (1-η)·p_0_math + η·p_llm
28:
29:  // ── Layer 3: 多模态推理链 ──
30:  chains  ← {}
31:  chains["M"] ← MetricChain(p_0, W, a_obs)
32:  if L available: chains["L"] ← LogChain(p_0, L, t0)
33:  else:            s_backprop ← BackpropCompensation(a_obs, W)  // §6.2
34:  chains["T"] ← TraceChain(p_0, W)
35:
36:  // ── 主循环 ──
37:  t ← 0, ev_buf ← []
38:  e_0 ← EmotionVector(p_0, W, budget=1.0, iter=0)   // §7
39:
40:  while not Converged(p_t, W_t, t):                   // §8.9
41:      // ── 各链独立推理 ──
42:      for name, chain in chains:
43:          d_m   ← chain.SelectAction(p_m, W, e_t)     // 链内 EIG
44:          y_m   ← ExecuteAction(d_m)                  // §8.5
45:          p_m   ← BayesianUpdate(p_m, y_m, W)
46:
47:      // ── 嫁接融合 ──
48:      c     ← ModalityConfidence(chains)              // §6.3
49:      p_t   ← GraftFusion(chains, c)                  // §6.4
50:      if ConflictDetected(chains):                    // KL > 1.5
51:          MarkHighPriority()  // → 增加exploration
52:
53:      // ── 反事实 grounding ──
54:      ev_buf.append(selected_d, y_d)
55:      if |ev_buf| ≥ 3 or max(p_t) < 0.3:
56:          for each counterfactual_result in ev_buf:
57:              Δ_v ← μ_pred(v) - (1 - δ_v)             // §5.2
58:              if |Δ_v| > 0.2:  // significant residual
59:                  W ← GraphUpdateMStep(W, Δ_v, λ1)    // §5.3
60:          if NeedLLMEdgeRepair(p_t, W, residuals):    // §10.2
61:              E_probe ← LLMEdgeSuggest(CandidateEdgePairs(residuals, W, T))
62:              W ← ProbeThenPromote(W, E_probe)
60:          ev_buf ← []
61:
62:      // ── 情绪 + Meta-Controller ──
63:      e_t   ← EmotionVector(p_t, W, budget, t)        // §7
64:      d_meta← ControllerSelect(e_t, p_t, W, traj_mem) // §8.1
65:      ExecuteAction(d_meta)
66:      traj_mem.add(e_{t-1}, d_meta, e_t, outcome)
67:
68:      // ── Stop check ──
69:      readiness ← σ(w^T e_t + b)
70:      if StopProjection(readiness, gap, max_eig_cost) or Converged(p_t, W, t):
71:          break
72:
73:      t ← t + 1
74:
75:  // ── 输出 ──
76:  r* ← argmax_r p_t(r)
77:  return r*, BuildTrace(p_0..p_t, W_0..W_t, chains, traj_mem)
```

---

## 十二、计算复杂度

| 组件 | 计算量 | 说明 |
|------|--------|------|
| 信号去噪 | $O(\|V\| \cdot \|K\| \cdot T_{window})$ | 一次性，秒级 |
| 图初始化 | $O(\|V\|^2)$ | 毫秒级 |
| 信念更新 | $O(\|V\|)$ | 毫秒级 |
| 传播模型 | $O(\|V\|^2 \cdot \text{diam}(G))$ | 毫秒级 |
| EIG 计算 (per action) | $O(\|V\| \cdot N_{bins})$ | 毫秒级 |
| 全候选 EIG 评估 | $O(\|V\|^2 \cdot N_{bins})$ | $30^2 \times 10 = 9000$ 次，毫秒级 |
| EM 图更新 | $O(\|E\| \cdot M)$ | $100 \times 5 = 500$ 次，毫秒级 |
| 情绪向量 | $O(\|V\|)$ | 毫秒级 |
| COUNTERFACTUAL | 引擎受限 | ~60s（唯一瓶颈） |
| LLM API | 网络 IO | ~2s |

**每 step 纯数学计算**：毫秒级。
**每 query 预估时间**：3-8 steps × (0.1s 数学 + 1-3次 COUNTERFACTUAL × 60s + 1-2次 LLM × 2s) ≈ **100-250s**。
**对比当前**：每 query 430s，且新框架有理论保证。

---

## 十三、配置参数表

| 参数 | 符号 | 默认值 | 范围 | 说明 |
|------|------|--------|------|------|
| 基线/故障窗口 | $\Delta$ | 300s | 固定 | OpenRCA 标准 |
| 异常 Z 阈值 | $z_{th}$ | 3.0 | [2.5, 4.0] | MAD-based |
| 持久化比例 | $p_{persist}$ | 0.3 | [0.2, 0.5] | 异常时间占比 |
| 传播衰减 | $\alpha_{base}$ | 0.8 | [0.6, 0.9] | 基础值 |
| metric/log 权重 | $\lambda_m, \lambda_l$ | 0.7, 0.3 | - | 信号融合 |
| 观测噪声 | $\sigma^2$ | 0.1 | [0.05, 0.2] | 高斯似然 |
| L1 稀疏化基础 | $\lambda_1^{base}$ | 0.05 | [0.01, 0.1] | 自适应调整 |
| 图偏离惩罚 | $\lambda_2$ | 0.10 | [0.05, 0.2] | Frobenius |
| EM 迭代次数 | $M$ | 5 | [3, 10] | 每触发 |
| EM 触发证据数 | $K$ | 3 | [2, 5] | 累积 |
| EM 学习率 | $\eta_w$ | 0.02 | [0.01, 0.05] | 坐标下降 |
| 嫁接 KL 阈值 | $\tau_{graft}$ | 0.5 | [0.3, 0.7] | 融合触发 |
| 共识增益系数 | $\gamma$ | 0.5 | [0.3, 0.8] | 嫁接融合 |
| 信念收敛阈值 | $\tau_p$ | 0.80 | [0.70, 0.90] | |
| EIG 收敛阈值 | $\tau_{eig}$ | 0.01 | [0.005, 0.02] | |
| Stop readiness 阈值 | $\tau_{stop}$ | 0.80 | [0.70, 0.90] | 在线早停门控 |
| top-1/top-2 间隔阈值 | $\tau_{gap}$ | 0.15 | [0.10, 0.25] | 防止过早停止 |
| 图稳定阈值 | $\tau_w$ | 0.02 | [0.01, 0.05] | Frobenius |
| 最大推理步数 | $T_{max}$ | 15 | [10, 20] | |
| 最大反事实次数 | $N_{cf}^{max}$ | 3 | [2, 5] | |
| 最大 LLM 调用 | $N_{llm}^{max}$ | 5 | [3, 8] | |
| LLM 先验基础权重 | $\eta_{base}$ | 0.4 | [0.2, 0.5] | 自适应调整 |
| LLM 先验温度 | $T_{llm}$ | 1.2 | [0.8, 2.0] | 分数转概率校准 |
| LLM 一致性门槛 | $q_{llm}^{min}$ | 0.35 | [0.25, 0.50] | 低于则拒收先验 |
| 未解释异常阈值 | $\tau_{unexp}$ | 0.30 | [0.20, 0.50] | explained ratio 上限 |
| 补边残差放大系数 | $\rho$ | 0.5 | [0.2, 1.0] | 残差驱动 LLM 边接纳 |
| 稀疏化敏感系数 | $\gamma_{sparse}$ | 0.3 | [0.1, 0.5] | |
| stop SGD 学习率 | $lr_{stop}$ | 0.05 | [0.01, 0.1] | |
| stop 正则系数 | $\lambda_{stop}$ | 0.01 | [0.001, 0.05] | 拉回语义先验 |
| 轨迹记忆容量 | | 1000 | 固定 | FIFO |

---

## 十四、实现路线图

### Phase 1：信号去噪 + 异常摘要（2-3 天）
- MetricAnomalyDetector（MAD Z-score + 持久化过滤）
- LogErrorAggregator（错误计数 + fatal boost）
- SignalFusion（统一 $\mathbf{a}^{obs}$）
- **验收**：Bank 一个 case 输出 $\mathbf{a}^{obs}$，人工检查 top-5 异常实体含真根因的比例

### Phase 2：图初始化 + 传播模型（3-4 天）
- TraceEdgeExtractor（确定性边 + $\mathbf{W}_{frozen}$）
- MetricEdgeInferrer（时序推断边）
- PropagationModel（阻尼加权传播 + 层级衰减）
- 概率图初始化融合
- **验收**：传播预测 $\boldsymbol{\mu}_r$ vs 实际 $\mathbf{a}^{obs}$ 的 cosine similarity > 0.6

### Phase 3：信念更新 + EIG + 动作选择（4-5 天）
- BayesianBeliefUpdater
- EIGComputer（含离散化和边际化）
- ActionSelector（含剪枝规则）
- 单模态推理链（无嫁接先跑通）
- **验收**：EIG 随步数单调下降，信念熵单调下降

### Phase 4：多模态链 + 嫁接（3-4 天）
- MetricChain / LogChain / TraceChain
- ModalityConfidence 计算
- GraftFusion（KL 检测 + 置信度加权 + 共识增益）
- 冲突标记
- **验收**：嫁接后 $p(r)$ 熵 $\leq$ 单链 min 熵

### Phase 5：情绪 + Meta-Controller + EM 图更新（4-5 天）
- EmotionVector（6-D + 可微化）
- SentientController（$\beta(\mathbf{e})$ 混合 + $\Phi$ 预测 + Stop Projection）
- GraphUpdate（EM M-step + 反事实残差梯度）
- TrajectoryMemory
- **验收**：Meta-Controller 选择的动作 $EIG/c$ 在跨 case 平均上优于纯 EIG 和纯情绪

### Phase 6：LLM 集成 + 自适应机制（3-4 天）
- LLMPriorInjector（先验 + 自适应 $\eta(\mu)$）
- LLMEdgeSuggester（边建议 + 3次采样交验）
- ModalityCompensation（传播反推 + 虚拟 log）
- AdaptiveSparsity（$\lambda_1(N_{trace})$）
- **验收**：按 §10.2.5 代理验证协议通过 `LegalRate`、`Precision@3`、`Gain_{fit}`、`Gain_{rank}` 四项门槛

### Phase 7：全量评测 + 消融（4-5 天）
- OpenRCA 335 queries 全量
- 消融矩阵：
  - 移除情绪分量（$\beta \equiv 0$，纯 EIG）
  - 移除嫁接（单模态最优链）
  - 移除图更新 EM（图固定）
  - 移除 LLM 先验（$\eta \equiv 0$）
  - 移除反事实 grounding（不用残差，纯传播）
- Per-task、per-system、per-modality 分析
- **验收**：Correct > 25% 或 C+P > 55%（超当前 ablation 46.7% 的显著边际）

---

## 十五、附录：关键推导

### A.1 传播模型梯度

$$\frac{\partial \mu_r^{(t+1)}(x)}{\partial w_{uv}} = \begin{cases}
\alpha_{uv} \cdot \mu_r^{(t)}(u) & \text{if } x = v \text{ and } \mu_r^{(t+1)}(v) > a_v^{obs} \\
\alpha_{xv} \cdot w_{xv} \cdot \frac{\partial \mu_r^{(t)}(x)}{\partial w_{xv}} & \text{otherwise (链式传播)}
\end{cases}$$

实际使用 autograd（PyTorch / JAX），不手写。

### A.2 反事实似然

$$p(\delta_v \mid r, \mathbf{W}) = \begin{cases}
\mathcal{N}(\delta_v; 0.80, 0.10^2) & \text{if } v = r \\
\mathcal{N}(\delta_v; 0.20, 0.20^2) & \text{if } v \neq r
\end{cases}$$
$$p(\delta_v \mid d=\text{CF}(v)) = p(\delta_v \mid v=r) \cdot p_t(v) + p(\delta_v \mid v \neq r) \cdot (1-p_t(v))$$

### A.3 EIG 对 COUNTERFACTUAL 的计算

$$EIG(CF(v)) = \sum_{k=1}^{10} p(\delta_v \in \text{bin}_k \mid CF(v)) \cdot [H(p_t) - H(p_{t+1} \mid \delta_v \in \text{bin}_k)]$$

其中 $\text{bin}_k = [\frac{k-1}{10}, \frac{k}{10})$，$p(\delta_v \in \text{bin}_k \mid CF(v))$ 由 $p_t$ 加权的高斯混合给出。

### A.4 情绪向量对信念的梯度

$$\frac{\partial e_1}{\partial p(r)} = \mathbb{1}[r = r_1], \quad r_1 = \arg\max p$$
$$\frac{\partial e_2}{\partial p(r)} = -\frac{1}{H_{max}} \cdot (1 + \log p(r)) \cdot (1 - \text{coverage})$$
$$\frac{\partial e_6}{\partial p(r)} = -(1 - \text{budget\_ratio}) \cdot \mathbb{1}[r = r_1]$$

梯度允许端到端反向传播，但实际训练中间层（meta-controller 的 $w_\beta, b_\beta$, stop projection 的 $\mathbf{w}, b$）已足够。

---

*PRISM v1.0 — 2026-05-20*
*本文档为框架唯一权威设计规范，后续所有实现以此为准。*
