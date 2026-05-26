# 20 个 OpenRCA 案例的策略匹配度初始标注

评分标准：

| 分值 | 含义 |
| --- | --- |
| 3 | 首选策略 |
| 2 | 适合作为下一步或强补充策略 |
| 1 | 可做，但边际收益较低 |
| 0 | 不适合，或容易把分析带偏 |

说明：

- 这是一版初始标注，目标是给人工复核提供第一轮参考，而不是最终金标准。
- `S1` 到 `S5` 分别对应 [strategy_pool.md](/mnt/d/Projects/OpenRCA/mydocs/strategy_pool.md:109) 中定义的五类策略。
- 目前这 20 个案例都已至少完成一轮原始 `record/query` 核验。

| Case | 数据集 | 子集 | 时间窗 | 根因层级 | 根因组件 | 根因原因 | 核验状态 | S1 | S2 | S3 | S4 | S5 | 首选 | 次选 | 初始标注备注 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Bank | - | 2021-03-06 06:00-06:30 | pod | Tomcat01 | high memory usage | 已核验（前轮） | 2 | 3 | 0 | 0 | 1 | S2 | S1 | 资源类，业务症状可能偏弱，适合单候选深挖。 |
| 2 | Bank | - | 2021-03-06 23:30-00:00 | pod | MG01 | network latency | 已核验（前轮） | 3 | 0 | 2 | 0 | 0 | S1 | S3 | 先做浅筛，再决定是否沿传播链回溯。 |
| 3 | Bank | - | 2021-03-06 18:30-19:00 | pod | apache02 | network packet loss | 已核验（前轮） | 3 | 0 | 2 | 0 | 0 | S1 | S3 | 典型网络类，S1 用于快速排他，S3 用于补传播解释。 |
| 4 | Bank | - | 2021-03-07 08:00-08:30 | pod | Tomcat03 | high CPU usage | 已核验（前轮） | 2 | 3 | 1 | 0 | 0 | S2 | S1 | 局部资源异常强，S2 最合适。 |
| 5 | Bank | - | 2021-03-23 01:00-01:30 | pod | MG01 | high disk I/O read usage | 已核验（前轮） | 2 | 3 | 0 | 0 | 0 | S2 | S1 | 典型弱症状资源类，容易被只看端到端症状的策略误伤。 |
| 6 | Bank | - | 2021-03-06 15:00-15:30 | pod | IG02 | high disk space usage | 已核验（前轮） | 2 | 3 | 0 | 0 | 0 | S2 | S1 | 稀有资源边界例，重点看局部偏离和 peer 排他。 |
| 7 | Bank | - | 2021-03-07 03:30-04:00 | pod | Tomcat02 | JVM Out of Memory (OOM) Heap | 已核验（前轮） | 1 | 2 | 0 | 0 | 3 | S5 | S2 | GC/OOM 语义锚点强，S5 应优先。 |
| 8 | Bank | - | 2021-03-04 13:30-14:00 | pod | MG02 | high JVM CPU load | 已核验（前轮） | 2 | 3 | 0 | 0 | 1 | S2 | S1 | 业务症状未必强，仍以局部 JVM/CPU 深挖为主。 |
| 9 | Market | cloudbed-1 | 2022-03-20 10:30-11:00 | node | node-1 | node memory consumption | 已核验（本轮） | 2 | 2 | 0 | 3 | 0 | S4 | S2 | 典型 node 层样本，适合验证层级消歧。 |
| 10 | Market | cloudbed-1 | 2022-03-21 13:30-14:00 | service | currencyservice | container memory load | 已核验（本轮） | 2 | 2 | 0 | 3 | 0 | S4 | S2 | 与 node/pod memory 样本构成 service 层对照。 |
| 11 | Market | cloudbed-1 | 2022-03-20 23:00-23:30 | pod | checkoutservice-2 | container memory load | 已核验（本轮） | 2 | 2 | 0 | 3 | 0 | S4 | S2 | 同属 memory 家族，但最小解释层级是 pod。 |
| 12 | Market | cloudbed-2 | 2022-03-20 09:30-10:00 | service | productcatalogservice | container network packet corruption | 已核验（本轮） | 2 | 0 | 3 | 1 | 1 | S3 | S1 | 网络传播和上下游边退化更关键，S3 应主导。 |
| 13 | Market | cloudbed-1 | 2022-03-21 00:30-01:00 | service | cartservice | container network packet retransmission | 已核验（本轮） | 3 | 2 | 1 | 2 | 0 | S1 | S4 | retransmission 是弱症状边界例，先筛后消歧更稳。 |
| 14 | Market | cloudbed-2 | 2022-03-21 03:30-04:00 | pod | recommendationservice-1 | container process termination | 已核验（本轮） | 1 | 1 | 1 | 1 | 3 | S5 | S4 | process termination 更依赖语义锚点和离散事件。 |
| 15 | Telecom | - | 2020-05-27 05:00-05:30 | pod | docker_001 | CPU fault | 已核验（本轮） | 1 | 3 | 1 | 0 | 0 | S2 | S1 | `Telecom` CPU fault 的典型资源深挖样本。 |
| 16 | Telecom | - | 2020-05-30 02:30-03:00 | node | os_009 | network delay | 已核验（本轮） | 3 | 0 | 2 | 1 | 0 | S1 | S3 | 先快速定位 node 网络候选，再看 JDBC/下游传播。 |
| 17 | Telecom | - | 2020-05-30 05:00-05:30 | node | os_018 | network loss | 已核验（本轮） | 3 | 0 | 2 | 1 | 0 | S1 | S3 | 与 delay 形成对照，同样以网络传播视角为主。 |
| 18 | Telecom | - | 2020-05-27 01:30-02:00 | service | db_003 | db connection limit | 已核验（本轮） | 1 | 0 | 2 | 1 | 3 | S5 | S3 | DB 连接型故障以连接/JDBC 锚点最有区分力。 |
| 19 | Telecom | - | 2020-05-27 02:00-02:30 | service | db_007 | db close | 已核验（本轮） | 1 | 0 | 2 | 1 | 3 | S5 | S3 | `db close` 与 `connection limit` 都走 S5，但锚点模式不同。 |
| 20 | Telecom | - | 2020-05-27 02:30-03:00 | service | db_007 | db connection limit | 已核验（本轮） | 1 | 0 | 1 | 0 | 3 | S5 | S3 | 同组件相邻窗口的 DB 子类对照样本，适合验证 S5 子类型区分力。 |

## 复核建议

| 项目 | 建议 |
| --- | --- |
| 最先复核的案例 | 13、14、18、19、20。这几类最容易暴露策略边界是否定义清楚。 |
| 最容易产生分歧的案例 | 5、6、8、13。原因是业务症状弱，但局部证据可能很强。 |
| 最适合做标注员校准的案例组 | 9、10、11。三者同属 memory 家族，但层级分别是 node/service/pod。 |
| 最适合验证 S5 子类型能力的案例组 | 18、19、20。两种 DB 原因在相近上下文中对比明显。 |
