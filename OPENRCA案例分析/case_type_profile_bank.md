# OpenRCA Bank 案例画像

- 数据目录：`dataset/Bank`
- 分析窗口：故障前 30 分钟作为基线，故障后 10 分钟作为事件窗口

## 类型分布
- 总案例数：136
- 根因原因分布：high CPU usage 33 (24.3%)，network packet loss 32 (23.5%)，network latency 27 (19.9%)，high disk I/O read usage 19 (14.0%)，high memory usage 10 (7.4%)，JVM Out of Memory (OOM) Heap 7 (5.1%)，high disk space usage 5 (3.7%)，high JVM CPU load 3 (2.2%)
- 高频根因组件：Tomcat01 19，MG01 17，MG02 17，Tomcat04 12，Tomcat03 12，IG02 12，apache02 10，Tomcat02 9，Redis02 8，apache01 8
- 根因层级：全部为 `pod`

## 各类典型特征
### high CPU usage
- 占比：33/136 (24.3%)
- 高频根因组件：Redis02(6), IG02(6), Tomcat03(4), IG01(3), Tomcat04(3)
- 典型根因信号：OSLinux-CPU_CPU-0_SingleCpuUtil up (覆盖 28/33，均值 10.54 -> 44.62，代表性 15.55)；OSLinux-CPU_CPU-2_SingleCpuUtil up (覆盖 27/33，均值 18.09 -> 50.20，代表性 15.50)；OSLinux-CPU_CPU_CPUUserTime up (覆盖 33/33，均值 17.53 -> 47.67，代表性 10.27)
- 上层业务症状：`metric_app.mrt` 中位数 337.27 -> 320.29，`sr` 中位数 100.0% -> 100.0%
- 日志症状：日志类型 localhost_access_log(88723)、apache_access_log(14803)、gc(158)；关键词 allocation failure(158)、full gc(158)、exception(2)；`localhost_access_log` 时延中位数 0.309 -> 0.625；`localhost_access_log` 5xx 比例中位数 0.0% -> 0.0%
- Trace 症状：根因组件 span `duration` 中位数 106.75 -> 188.13
- 传播相关组件：Tomcat04(9)、Tomcat03(9)、Tomcat02(9)、Tomcat01(8)、IG01(6)
- 常见调用边：IG02->Tomcat04(7)、IG02->Tomcat03(7)、IG02->Tomcat01(7)、IG01->Tomcat01(5)、IG01->Tomcat04(5)

### network packet loss
- 占比：32/136 (23.5%)
- 高频根因组件：Tomcat01(6), apache02(5), MG02(5), Tomcat03(5), Tomcat04(4)
- 典型根因信号：OSLinux-OSLinux_NETWORK_NETWORK_TCP-FIN-WAIT up (覆盖 31/32，均值 0.07 -> 12.36，代表性 11.77)；OSLinux-OSLinux_NETWORK_ens160_NETKBTotalPerSec down (覆盖 18/32，均值 316490.33 -> 186970.00，代表性 0.27)；OSLinux-OSLinux_NETWORK_NETWORK_TotalTcpConnNum down (覆盖 25/32，均值 1146.57 -> 731.60，代表性 0.24)
- 上层业务症状：`metric_app.mrt` 中位数 447.54 -> 883.42，`sr` 中位数 100.0% -> 99.8%
- 日志症状：日志类型 localhost_access_log(186224)、apache_access_log(120166)、gc(319)；`localhost_access_log` 时延中位数 0.403 -> 0.410；`localhost_access_log` 5xx 比例中位数 0.0% -> 0.0%
- Trace 症状：根因组件 span `duration` 中位数 175.90 -> 147.51
- 传播相关组件：IG02(7)、IG01(7)、MG01(7)、MG02(7)、dockerB1(5)
- 常见调用边：Tomcat01->MG02(7)、IG02->Tomcat01(5)、Tomcat01->MG01(4)、IG01->Tomcat01(4)、Tomcat04->MG02(3)

### network latency
- 占比：27/136 (19.9%)
- 高频根因组件：MG01(6), MG02(5), Tomcat02(4), Tomcat01(3), apache02(3)
- 典型根因信号：OSLinux-OSLinux_NETWORK_NETWORK_TCP-FIN-WAIT up (覆盖 22/27，均值 0.40 -> 3.21，代表性 2.61)；OSLinux-OSLinux_NETWORK_ens160_NETKBTotalPerSec down (覆盖 16/27，均值 338462.09 -> 199922.61，代表性 0.25)；OSLinux-OSLinux_NETWORK_NETWORK_TotalTcpConnNum down (覆盖 21/27，均值 1436.32 -> 964.32，代表性 0.25)
- 上层业务症状：`metric_app.mrt` 中位数 345.65 -> 591.07，`sr` 中位数 100.0% -> 100.0%
- 日志症状：日志类型 localhost_access_log(112565)、apache_access_log(59790)、gc(228)；`localhost_access_log` 时延中位数 0.560 -> 0.488；`localhost_access_log` 5xx 比例中位数 0.0% -> 0.0%
- Trace 症状：根因组件 span `duration` 中位数 118.33 -> 106.06
- 传播相关组件：IG01(8)、MG01(8)、MG02(8)、IG02(8)、dockerB2(4)
- 常见调用边：Tomcat02->MG01(5)、Tomcat01->MG02(5)、Tomcat01->MG01(4)、IG01->Tomcat02(3)、Tomcat02->MG02(3)

### high disk I/O read usage
- 占比：19/136 (14.0%)
- 高频根因组件：Tomcat01(5), MG01(3), MG02(3), apache01(2), Tomcat02(2)
- 典型根因信号：OSLinux-OSLinux_LOCALDISK_LOCALDISK-sdb_DSKRead up (覆盖 16/19，均值 0.00 -> 64570.70，代表性 54375.32)；OSLinux-OSLinux_LOCALDISK_LOCALDISK-sdb_DSKReadWrite up (覆盖 19/19，均值 147.71 -> 68453.26，代表性 4371.43)；OSLinux-OSLinux_LOCALDISK_LOCALDISK-sdb_DSKWrite up (覆盖 16/19，均值 93.91 -> 30319.50，代表性 1836.38)
- 上层业务症状：`metric_app.mrt` 中位数 259.71 -> 137.09，`sr` 中位数 100.0% -> 100.0%
- 日志症状：日志类型 localhost_access_log(86063)、apache_access_log(39097)、gc(170)；`localhost_access_log` 时延中位数 0.557 -> 0.494；`localhost_access_log` 5xx 比例中位数 0.0% -> 0.0%
- Trace 症状：根因组件 span `duration` 中位数 114.85 -> 98.85
- 传播相关组件：MG01(6)、IG01(6)、MG02(6)、IG02(6)、Tomcat01(4)
- 常见调用边：IG02->Tomcat01(6)、Tomcat01->MG01(5)、Tomcat01->MG02(5)、IG01->Tomcat01(4)、IG02->Tomcat03(3)

### high memory usage
- 占比：10/136 (7.4%)
- 高频根因组件：Mysql02(2), Redis02(2), Tomcat01(2), Tomcat04(2), Mysql01(1)
- 典型根因信号：OSLinux-OSLinux_MEMORY_MEMORY_MEMFreeMem up (覆盖 9/10，均值 1421.37 -> 1507.77，代表性 2.28)；OSLinux-OSLinux_MEMORY_MEMORY_NoCacheMemPerc up (覆盖 7/10，均值 35.57 -> 56.90，代表性 0.81)；OSLinux-OSLinux_MEMORY_MEMORY_CacheMem down (覆盖 8/10，均值 2745.26 -> 1416.96，代表性 0.33)
- 上层业务症状：`metric_app.mrt` 中位数 297.37 -> 165.45，`sr` 中位数 100.0% -> 100.0%
- 日志症状：日志类型 localhost_access_log(17498)、gc(34)；关键词 allocation failure(34)、full gc(34)；`localhost_access_log` 时延中位数 0.419 -> 0.189；`localhost_access_log` 5xx 比例中位数 0.0% -> 0.0%
- Trace 症状：根因组件 span `duration` 中位数 136.88 -> 67.51
- 传播相关组件：MG02(3)、IG02(3)、IG01(3)、MG01(3)、dockerB1(1)
- 常见调用边：Tomcat04->MG01(3)、IG01->Tomcat04(2)、IG02->Tomcat04(2)、Tomcat04->MG02(2)、Tomcat01->MG02(1)

### JVM Out of Memory (OOM) Heap
- 占比：7/136 (5.1%)
- 高频根因组件：Tomcat02(3), MG02(2), MG01(1), IG01(1)
- 典型根因信号：JVM-Operating System_7779_JVM_JVM_CPULoad up (覆盖 3/7，均值 0.40 -> 9.50，代表性 3.90)；JVM-Operating System_7778_JVM_JVM_CPULoad up (覆盖 1/7，均值 0.13 -> 10.93，代表性 1.54)；OSLinux-OSLinux_MEMORY_MEMORY_MEMFreeMem down (覆盖 5/7，均值 323.82 -> 258.78，代表性 0.12)
- 上层业务症状：`metric_app.mrt` 中位数 218.72 -> 109.93，`sr` 中位数 100.0% -> 100.0%
- 日志症状：日志类型 localhost_access_log(10752)、gc(947)；关键词 allocation failure(947)、full gc(947)；`localhost_access_log` 时 延中位数 1.455 -> 2.133；`localhost_access_log` 5xx 比例中位数 0.0% -> 0.0%
- Trace 症状：根因组件 span `duration` 中位数 80.56 -> 71.02
- 传播相关组件：dockerB2(3)、dockerB1(3)、Tomcat01(3)、Tomcat02(3)、Tomcat03(3)
- 常见调用边：Tomcat02->MG02(5)、IG01->Tomcat02(4)、Tomcat02->MG01(3)、IG02->Tomcat02(3)、Tomcat01->MG02(2)

### high disk space usage
- 占比：5/136 (3.7%)
- 高频根因组件：IG02(2), apache01(2), Tomcat04(1)
- 典型根因信号：OSLinux-OSLinux_LOCALDISK_LOCALDISK-sda_DSKRead up (覆盖 4/5，均值 0.00 -> 7.07，代表性 5.66)；OSLinux-OSLinux_LOCALDISK_LOCALDISK-sdb_DSKReadWrite up (覆盖 5/5，均值 45.80 -> 73.35，代表性 2.66)；OSLinux-OSLinux_LOCALDISK_LOCALDISK-sdb_DSKRead up (覆盖 3/5，均值 0.00 -> 4.28，代表性 2.57)
- 上层业务症状：`metric_app.mrt` 中位数 508.68 -> 449.88，`sr` 中位数 99.8% -> 100.0%
- 日志症状：日志类型 apache_access_log(60251)、localhost_access_log(97)、gc(2)；`localhost_access_log` 时延中位数 0.096 -> 0.039；`localhost_access_log` 5xx 比例中位数 0.0% -> 0.0%
- Trace 症状：根因组件 span `duration` 中位数 793.60 -> 611.58
- 传播相关组件：Tomcat04(2)、Tomcat03(2)、Tomcat01(2)、Tomcat02(1)、MG02(1)
- 常见调用边：IG02->Tomcat04(3)、IG02->Tomcat03(2)、IG02->Tomcat01(2)、IG02->Tomcat02(1)、Tomcat04->MG02(1)

### high JVM CPU load
- 占比：3/136 (2.2%)
- 高频根因组件：MG02(2), IG01(1)
- 典型根因信号：JVM-Operating System_7779_JVM_JVM_CPULoad up (覆盖 2/3，均值 0.49 -> 23.71，代表性 15.48)；JVM-Operating System_7778_JVM_JVM_CPULoad up (覆盖 1/3，均值 0.17 -> 25.32，代表性 8.38)；OSLinux-CPU_CPU-3_SingleCpuUtil up (覆盖 2/3，均值 15.64 -> 66.94，代表性 6.03)
- 上层业务症状：`metric_app.mrt` 中位数 190.19 -> 196.17，`sr` 中位数 100.0% -> 100.0%
- 日志症状：n/a
- Trace 症状：根因组件 span `duration` 中位数 136.41 -> 138.19
- 传播相关组件：Tomcat02(3)、dockerB2(2)、dockerB1(2)、Tomcat03(2)、dockerA1(2)
- 常见调用边：Tomcat02->MG02(2)、Tomcat03->MG02(2)、Tomcat04->MG02(2)、Tomcat01->MG02(2)、dockerB2->MG02(1)

## 限制说明
- 当前报告只分析已解压的 `Bank` 子集。
- `Bank` 的根因全部在 `pod` 层；根因 KPI 来自 `metric_container.csv`，业务症状来自 `metric_app.csv`。
- 日志只基于 `log_service.csv` 做轻量模式提取，属于经验型特征，不等于严格语义解析。
- 本次运行包含 trace 扫描，trace 结果来自 span 级时延和父子 span 组件关系。