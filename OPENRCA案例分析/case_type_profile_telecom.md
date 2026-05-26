# OpenRCA Telecom 案例画像

- 数据目录：`dataset/Telecom`
- 分析窗口：故障前 30 分钟作为基线，故障后 10 分钟作为事件窗口

## 类型分布
- 总案例数：51
- 根因原因分布：CPU fault 19 (37.3%)，network delay 13 (25.5%)，db connection limit 7 (13.7%)，network loss 7 (13.7%)，db close 5 (9.8%)
- 根因层级分布：node 20 (39.2%)，pod 19 (37.3%)，service 12 (23.5%)

## 各类典型特征
### CPU fault
- 占比：19/51 (37.3%)
- 根因层级：pod 19
- 高频根因组件：docker_001(4), docker_004(3), docker_002(3), docker_008(3), docker_006(3)
- 典型根因信号：container_cpu_used up (覆盖 19/19，均值 9.49 -> 48.32，代表性 13.31)
- 上层业务症状：`metric_app.avg_time` 中位数 0.84 -> 5.64，`succee_rate` 中位数 99.9% -> 99.9%
- Trace/JDBC 症状：JDBC `elapsedTime` 中位数 16.70 -> 187.03，`success` 中位数 100.0% -> 100.0%
- 传播相关组件：unknown(19)、db_009(12)、db_007(12)、db_003(7)
- 常见调用边：docker_001->db_009(4)、docker_001->unknown(4)、docker_001->db_007(4)、docker_004->db_009(3)、docker_004->unknown(3)

### network delay
- 占比：13/51 (25.5%)
- 根因层级：node 13
- 高频根因组件：os_021(4), os_020(3), os_018(2), os_017(2), os_001(1)
- 典型根因信号：Received_queue up (覆盖 6/13，均值 3430.47 -> 5641.00，代表性 239.34)；Sent_queue up (覆盖 13/13，均值 34.25 -> 120.81，代表性 97.31)；ICMP_ping down (覆盖 9/13，均值 1.00 -> 0.51，代表性 0.34)
- 上层业务症状：`metric_app.avg_time` 中位数 1.85 -> 8.68，`succee_rate` 中位数 100.0% -> 99.2%
- Trace/JDBC 症状：JDBC `elapsedTime` 中位数 45.26 -> 117.18，`success` 中位数 100.0% -> 100.0%
- 传播相关组件：n/a
- 常见调用边：docker_002->db_009(12)、docker_003->db_009(12)、docker_004->db_009(7)、docker_001->db_009(5)、docker_005->db_003(3)

### db connection limit
- 占比：7/51 (13.7%)
- 根因层级：service 7
- 高频根因组件：db_003(4), db_007(3)
- 典型根因信号：Proc_User_Used_Pct up (覆盖 7/7，均值 1.67 -> 494.10，代表性 173.97)；Login_Per_Sec up (覆盖 7/7，均值 1.35 -> 4.04，代表性 2.30)；Session_pct up (覆盖 7/7，均值 0.17 -> 0.78，代表性 0.61)
- 上层业务症状：`metric_app.avg_time` 中位数 0.66 -> 0.49，`succee_rate` 中位数 99.9% -> 94.2%
- Trace/JDBC 症状：JDBC `elapsedTime` 中位数 10.13 -> 6.28，`success` 中位数 100.0% -> 99.7%
- 传播相关组件：docker_006(4)、docker_003(3)、docker_001(3)、docker_008(3)、docker_007(3)
- 常见调用边：docker_006->db_003(4)、docker_003->db_007(3)、docker_001->db_007(3)、docker_008->db_003(3)、docker_007->db_003(3)

### network loss
- 占比：7/51 (13.7%)
- 根因层级：node 7
- 高频根因组件：os_018(3), os_021(2), os_009(1), os_017(1)
- 典型根因信号：Sent_queue up (覆盖 7/7，均值 2.76 -> 255.29，代表性 101.09)；Received_queue up (覆盖 6/7，均值 3.56 -> 48.33，代表性 36.97)；System_wait_queue_length up (覆盖 6/7，均值 1.06 -> 1.17，代表性 0.34)
- 上层业务症状：`metric_app.avg_time` 中位数 0.63 -> 8.36，`succee_rate` 中位数 100.0% -> 98.7%
- Trace/JDBC 症状：JDBC `elapsedTime` 中位数 40.73 -> 91.01，`success` 中位数 100.0% -> 100.0%
- 传播相关组件：n/a
- 常见调用边：docker_002->db_009(7)、docker_003->db_009(5)、docker_001->db_009(5)、docker_004->db_009(3)、docker_005->db_003(1)

### db close
- 占比：5/51 (9.8%)
- 根因层级：service 5
- 高频根因组件：db_003(3), db_007(2)
- 典型根因信号：tnsping_result_time up (覆盖 4/5，均值 3.93 -> 12501.43，代表性 4286.05)；Login_Per_Sec up (覆盖 5/5，均值 1.17 -> 2.46，代表性 1.13)；Call_Per_Sec down (覆盖 5/5，均值 139.91 -> 87.91，代表性 0.36)
- 上层业务症状：`metric_app.avg_time` 中位数 0.58 -> 0.39，`succee_rate` 中位数 100.0% -> 54.5%
- Trace/JDBC 症状：JDBC `elapsedTime` 中位数 6.92 -> 7.73，`success` 中位数 100.0% -> 92.9%
- 传播相关组件：docker_006(3)、docker_008(3)、docker_007(2)、docker_002(2)、docker_003(2)
- 常见调用边：docker_006->db_003(3)、docker_008->db_003(3)、docker_007->db_003(2)、docker_002->db_007(2)、docker_003->db_007(2)

## 限制说明
- 当前报告只分析已解压的 `Telecom` 子集；`Bank` 和 `Market` 需要先解压对应数据目录。
- `Telecom` 没有日志文件，传播分析主要依赖 `metric_app.csv` 和 `trace/trace_span.csv`。
- `Telecom` 的 trace 几乎全部是 JDBC 调用，能直接看出的传播路径主要是 `docker -> db`；对 `os` 级网络故障，更多是通过节点指标和全局 JDBC 症状做间接判断。
- 本次运行包含 trace 扫描。