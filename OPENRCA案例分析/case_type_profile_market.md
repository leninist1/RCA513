# OpenRCA Market 案例画像

- 数据目录：`dataset/Market`
- 分析 cloudbed：cloudbed-1, cloudbed-2
- 分析窗口：故障前 30 分钟作为基线，故障后 10 分钟作为事件窗口

## 类型分布
- 总案例数：148
- cloudbed 分布：cloudbed-1 70 (47.3%)，cloudbed-2 78 (52.7%)
- 根因层级分布：node 51 (34.5%)，service 49 (33.1%)，pod 48 (32.4%)
- 根因原因分布：container read I/O load 17 (11.5%)，container memory load 13 (8.8%)，container network packet corruption 13 (8.8%) ，container CPU load 13 (8.8%)，container network packet retransmission 13 (8.8%)，node disk write I/O consumption 10 (6.8%)，node disk space consumption 10 (6.8%)，node memory consumption 9 (6.1%)，node disk read I/O consumption 9 (6.1%)，node CPU load 9 (6.1%)，container packet loss 8 (5.4%)，container network latency 8 (5.4%)，container process termination 7 (4.7%)，container write I/O load 5 (3.4%)，node CPU spike 4 (2.7%)

## 各类典型特征
### container read I/O load
- 占比：17/148 (11.5%)
- cloudbed 分布：cloudbed-1(8), cloudbed-2(9)
- 根因层级：pod(9), service(8)
- 高频根因组件：frontend-2(3), shippingservice(2), emailservice(2), adservice(2), shippingservice-1(1)
- 典型根因信号：container_fs_reads./dev/vda up (覆盖 9/17，均值 59.80 -> 32633.37，代表性 14530.65)；container_fs_reads_MB./dev/vda up (覆盖 9/17，均值 16.88 -> 5517.78，代表性 2547.96)；container_cpu_cfs_throttled_periods up (覆盖 8/17，均值 0.25 -> 10.23，代表性 4.66)
- 上层业务症状：`metric_service.mrt` 中位数 1.05 -> 2.53，`sr` 中位数 100.0% -> 100.0%
- 服务日志症状：日志类型 log_currencyservice-service_application(87577)、log_adservice-service_application(40860)、log_frontend-service_application(19003)；关键词 exception(4086)、error(80)
- 代理日志症状：未扫描
- Trace 症状：span `duration` 中位数 152.34 -> 253.18；状态码 0(145075)、OK(17226)、2(786)、13(84)、9(62)
- 传播相关组件：frontend-1(9)、frontend-2(9)、frontend-0(9)、frontend2-0(5)、checkoutservice-1(4)
- 常见调用边：frontend-2->currencyservice-0(5)、frontend-2->currencyservice-2(4)、frontend-2->currencyservice-1(3)、frontend-2->productcatalogservice-1(3)、frontend-1->shippingservice-1(2)

### container CPU load
- 占比：13/148 (8.8%)
- cloudbed 分布：cloudbed-1(7), cloudbed-2(6)
- 根因层级：service(6), pod(7)
- 高频根因组件：productcatalogservice(2), recommendationservice(1), shippingservice(1), shippingservice2-0(1), productcatalogservice-0(1)
- 典型根因信号：container_cpu_cfs_throttled_seconds up (覆盖 7/13，均值 6.51 -> 435.97，代表性 85.65)；container_cpu_cfs_throttled_periods up (覆盖 7/13，均值 6.04 -> 312.44，代表性 65.19)；container_cpu_cfs_periods up (覆盖 7/13，均值 99.94 -> 362.14，代表性 6.04)
- 上层业务症状：`metric_service.mrt` 中位数 3.40 -> 68.53，`sr` 中位数 100.0% -> 100.0%
- 服务日志症状：日志类型 log_frontend-service_application(45827)、log_recommendationservice-service_application(8845)、log_shippingservice-service_application(7772)；关键词 warning(2828)
- 代理日志症状：未扫描
- Trace 症状：span `duration` 中位数 4125.70 -> 87687.42；状态码 0(305058)、2(2821)、4(11)
- 传播相关组件：frontend-0(6)、frontend-2(6)、frontend2-0(6)、frontend-1(4)、productcatalogservice-0(3)
- 常见调用边：frontend-0->productcatalogservice-0(3)、frontend2-0->recommendationservice2-0(2)、frontend2-0->shippingservice2-0(2) 、frontend-2->shippingservice-1(2)、checkoutservice2-0->shippingservice2-0(2)

### container memory load
- 占比：13/148 (8.8%)
- cloudbed 分布：cloudbed-1(7), cloudbed-2(6)
- 根因层级：pod(7), service(6)
- 高频根因组件：shippingservice-1(2), frontend(2), paymentservice(1), cartservice-0(1), checkoutservice-2(1)
- 典型根因信号：container_memory_failcnt up (覆盖 3/13，均值 38.02 -> 15790.91，代表性 261.83)；container_memory_failures.container.pgfault up (覆盖 7/13，均值 1989.05 -> 20295.74，代表性 236.96)；container_memory_failures.hierarchy.pgfault up (覆盖 6/13，均值 1533.43 -> 20266.55，代表性 236.71)
- 上层业务症状：`metric_service.mrt` 中位数 1.06 -> 1.55，`sr` 中位数 100.0% -> 100.0%
- 服务日志症状：日志类型 log_frontend-service_application(57691)、log_currencyservice-service_application(54152)、log_cartservice-service_application(17829)；关键词 warning(1808)、error(94)
- 代理日志症状：未扫描
- Trace 症状：span `duration` 中位数 163.62 -> 984.13；状态码 0(365527)、Ok(3752)、200(3461)、2(1806)、1(118)
- 传播相关组件：frontend-0(6)、frontend-2(5)、frontend-1(5)、checkoutservice2-0(4)、currencyservice-0(4)
- 常见调用边：frontend-2->shippingservice-1(3)、frontend-0->shippingservice-1(3)、frontend-1->shippingservice-1(3)、frontend2-0->currencyservice2-0(3)、frontend2-0->productcatalogservice2-0(3)

### container network packet corruption
- 占比：13/148 (8.8%)
- cloudbed 分布：cloudbed-1(6), cloudbed-2(7)
- 根因层级：service(7), pod(6)
- 高频根因组件：checkoutservice(2), recommendationservice(1), currencyservice-0(1), emailservice(1), shippingservice2-0(1)
- 典型根因信号：container_network_transmit_packets.eth0 down (覆盖 6/13，均值 866.68 -> 358.42，代表性 0.13)；container_network_receive_packets.eth0 down (覆盖 6/13，均值 859.02 -> 413.60，代表性 0.13)；container_network_transmit_MB.eth0 down (覆盖 6/13，均值 1.91 -> 1.37，代表性 0.07)
- 上层业务症状：`metric_service.mrt` 中位数 3.12 -> 614.18，`sr` 中位数 100.0% -> 98.8%
- 服务日志症状：日志类型 log_cartservice-service_application(30659)、log_currencyservice-service_application(18804)、log_recommendationservice-service_application(9440)；关键词 timeout(114)、error(67)、unavailable(2)
- 代理日志症状：未扫描
- Trace 症状：span `duration` 中位数 2413.72 -> 779205.62；状态码 0(158154)、Ok(4995)、200(4648)、2(609)、13(152)
- 传播相关组件：frontend-1(7)、frontend-2(7)、frontend-0(5)、frontend2-0(4)、paymentservice-1(3)
- 常见调用边：frontend2-0->productcatalogservice2-0(3)、frontend-1->recommendationservice-1(2)、frontend-1->currencyservice-0(2)、frontend2-0->shippingservice2-0(2)、frontend2-0->currencyservice2-0(2)

### container network packet retransmission
- 占比：13/148 (8.8%)
- cloudbed 分布：cloudbed-1(5), cloudbed-2(8)
- 根因层级：pod(8), service(5)
- 高频根因组件：cartservice(2), shippingservice-1(1), frontend-2(1), cartservice-0(1), adservice-1(1)
- 典型根因信号：container_network_transmit_packets.eth0 up (覆盖 8/13，均值 408.10 -> 552.79，代表性 0.22)；container_network_receive_packets.eth0 up (覆盖 8/13，均值 459.08 -> 586.51，代表性 0.14)；container_network_transmit_MB.eth0 up (覆盖 8/13，均值 0.99 -> 1.06，代表性 0.02)
- 上层业务症状：`metric_service.mrt` 中位数 1.11 -> 1.11，`sr` 中位数 100.0% -> 100.0%
- 服务日志症状：日志类型 log_cartservice-service_application(72711)、log_frontend-service_application(5632)、log_shippingservice-service_application(5386)
- 代理日志症状：未扫描
- Trace 症状：span `duration` 中位数 136.58 -> 126.28；状态码 0(109599)、Ok(11933)、200(11039)、OK(2548)
- 传播相关组件：frontend-2(8)、frontend-0(7)、frontend-1(7)、frontend2-0(4)、checkoutservice-0(3)
- 常见调用边：frontend-2->shippingservice-1(2)、frontend-2->productcatalogservice-1(2)、frontend-2->productcatalogservice-0(2)、frontend-0->cartservice-2(2)、frontend-1->cartservice-0(2)

### node disk space consumption
- 占比：10/148 (6.8%)
- cloudbed 分布：cloudbed-1(5), cloudbed-2(5)
- 根因层级：node(10)
- 高频根因组件：node-1(2), node-4(2), node-5(2), node-2(2), node-3(1)
- 典型根因信号：system.disk.used down (覆盖 3/10，均值 8550413869.78 -> 7843769985.16，代表性 0.03)；system.disk.pct_usage up (覆盖 1/10，均值 7.62 -> 8.77，代表性 0.02)；system.disk.free down (覆盖 1/10，均值 3881569883.02 -> 3684586478.93，代表性 0.01)
- 上层业务症状：`metric_service.mrt` 中位数 7.67 -> 7.91，`sr` 中位数 97.4% -> 97.4%
- 服务日志症状：日志类型 log_frontend-service_application(44544)、log_cartservice-service_application(31383)、log_currencyservice-service_application(19117)；关键词 exception(1456)
- 代理日志症状：未扫描
- Trace 症状：span `duration` 中位数 4513.71 -> 4599.21；状态码 0(322995)、Ok(13719)、200(12666)、OK(8347)、2(728)
- 传播相关组件：currencyservice-1(2)、currencyservice-2(2)、currencyservice-0(1)、productcatalogservice-0(1)、productcatalogservice-1(1)
- 常见调用边：frontend-1->currencyservice-2(2)、frontend-2->currencyservice-2(2)、frontend-1->currencyservice-0(1)、frontend-1->currencyservice-1(1)、frontend-1->adservice-2(1)

### node disk write I/O consumption
- 占比：10/148 (6.8%)
- cloudbed 分布：cloudbed-1(5), cloudbed-2(5)
- 根因层级：node(10)
- 高频根因组件：node-6(5), node-5(4), node-2(1)
- 典型根因信号：system.io.w_await up (覆盖 10/10，均值 1.00 -> 26.50，代表性 20.48)；system.io.await up (覆盖 10/10，均值 2.24 -> 19.66，代表性 12.93)；system.io.avg_q_sz up (覆盖 10/10，均值 1.66 -> 14.86，代表性 10.96)
- 上层业务症状：`metric_service.mrt` 中位数 14.85 -> 5.97，`sr` 中位数 97.3% -> 97.3%
- 服务日志症状：日志类型 log_cartservice-service_application(131166)、log_currencyservice-service_application(80262)、log_frontend-service_application(68608)；关键词 exception(4644)、error(3)
- 代理日志症状：未扫描
- Trace 症状：span `duration` 中位数 7712.67 -> 7760.69；状态码 0(526752)、Ok(37222)、200(34360)、OK(14244)、2(2350)
- 传播相关组件：frontend2-0(5)、currencyservice2-0(5)、currencyservice-0(4)、currencyservice-1(4)、productcatalogservice2-0(4)
- 常见调用边：frontend2-0->currencyservice2-0(5)、frontend2-0->productcatalogservice2-0(5)、frontend-2->currencyservice-0(4)、frontend2-0->shippingservice2-0(4)、frontend2-0->recommendationservice2-0(4)

### node CPU load
- 占比：9/148 (6.1%)
- cloudbed 分布：cloudbed-1(4), cloudbed-2(5)
- 根因层级：node(9)
- 高频根因组件：node-4(4), node-6(3), node-1(1), node-2(1)
- 典型根因信号：system.cpu.user up (覆盖 9/9，均值 4.41 -> 26.08，代表性 7.50)；system.cpu.pct_usage up (覆盖 9/9，均值 8.13 -> 29.54，代表性 3.14)；system.cpu.iowait down (覆盖 5/9，均值 2.25 -> 1.54，代表性 0.16)
- 上层业务症状：`metric_service.mrt` 中位数 2.12 -> 2.11，`sr` 中位数 95.7% -> 95.8%
- 服务日志症状：日志类型 log_cartservice-service_application(106614)、log_currencyservice-service_application(65043)、log_frontend-service_application(18857)
- 代理日志症状：未扫描
- Trace 症状：span `duration` 中位数 4772.25 -> 4536.79；状态码 0(336304)、Ok(16624)、200(15347)、OK(8962)、2(925)
- 传播相关组件：frontend2-0(3)、currencyservice2-0(3)、productcatalogservice2-0(2)、frontend-0(1)、frontend-2(1)
- 常见调用边：frontend2-0->currencyservice2-0(3)、frontend2-0->productcatalogservice2-0(3)、frontend2-0->recommendationservice2-0(2)、frontend2-0->adservice2-0(2)、frontend2-0->shippingservice2-0(2)

### node disk read I/O consumption
- 占比：9/148 (6.1%)
- cloudbed 分布：cloudbed-1(5), cloudbed-2(4)
- 根因层级：node(9)
- 高频根因组件：node-4(2), node-5(2), node-6(2), node-1(1), node-3(1)
- 典型根因信号：system.io.rkb_s up (覆盖 9/9，均值 24589.31 -> 39719.13，代表性 18551.08)；system.io.r_s down (覆盖 9/9，均值 110.58 -> 85.72，代表性 47.84)；system.io.util up (覆盖 9/9，均值 11.76 -> 16.27，代表性 10.21)
- 上层业务症状：`metric_service.mrt` 中位数 7.40 -> 7.97，`sr` 中位数 97.9% -> 98.0%
- 服务日志症状：日志类型 log_cartservice-service_application(156231)、log_currencyservice-service_application(95263)、log_frontend-service_application(54382)；关键词 exception(5168)
- 代理日志症状：未扫描
- Trace 症状：span `duration` 中位数 6018.45 -> 5432.20；状态码 0(455710)、Ok(24163)、200(22304)、OK(12714)、2(2582)
- 传播相关组件：currencyservice-0(3)、currencyservice-2(2)、currencyservice-1(2)、frontend-0(2)、frontend-1(2)
- 常见调用边：frontend-1->currencyservice-2(2)、frontend-1->currencyservice-0(2)、frontend-2->currencyservice-2(2)、frontend-2->currencyservice-0(2)、frontend-0->currencyservice-1(2)

### node memory consumption
- 占比：9/148 (6.1%)
- cloudbed 分布：cloudbed-1(5), cloudbed-2(4)
- 根因层级：node(9)
- 高频根因组件：node-1(2), node-4(2), node-6(2), node-2(2), node-5(1)
- 典型根因信号：system.mem.free down (覆盖 8/9，均值 8219.21 -> 6972.40，代表性 0.74)；system.mem.real.used up (覆盖 7/9，均值 14643.29 -> 19079.29，代表性 0.44)；system.mem.real.pct_useage up (覆盖 8/9，均值 42.24 -> 52.97，代表性 0.38)
- 上层业务症状：`metric_service.mrt` 中位数 6.10 -> 9.73，`sr` 中位数 97.6% -> 97.8%
- 服务日志症状：日志类型 log_cartservice-service_application(141480)、log_currencyservice-service_application(86461)、log_frontend-service_application(29565)；关键词 exception(2122)、warning(1061)
- 代理日志症状：未扫描
- Trace 症状：span `duration` 中位数 4226.80 -> 4483.20；状态码 0(443889)、Ok(23395)、200(21588)、OK(11861)、2(1061)
- 传播相关组件：frontend-1(2)、frontend-2(2)、frontend2-0(2)、currencyservice2-0(2)、frontend-0(1)
- 常见调用边：frontend2-0->currencyservice2-0(2)、frontend2-0->productcatalogservice2-0(2)、frontend-0->currencyservice-2(1)、frontend-0->currencyservice-1(1)、frontend-0->currencyservice-0(1)

### container network latency
- 占比：8/148 (5.4%)
- cloudbed 分布：cloudbed-1(4), cloudbed-2(4)
- 根因层级：service(5), pod(3)
- 高频根因组件：adservice(1), cartservice(1), frontend-1(1), productcatalogservice(1), adservice2-0(1)
- 典型根因信号：container_network_transmit_packets.eth0 down (覆盖 3/8，均值 829.41 -> 733.97，代表性 0.08)；container_network_receive_packets.eth0 down (覆盖 3/8，均值 795.73 -> 708.95，代表性 0.05)；container_network_transmit_MB.eth0 down (覆盖 3/8，均值 2.20 -> 2.09，代表性 0.03)
- 上层业务症状：`metric_service.mrt` 中位数 2.56 -> 116.81，`sr` 中位数 100.0% -> 100.0%
- 服务日志症状：日志类型 log_cartservice-service_application(78489)、log_frontend-service_application(11803)、log_recommendationservice-service_application(9018)
- 代理日志症状：未扫描
- Trace 症状：span `duration` 中位数 1146.98 -> 229023.85；状态码 0(167930)、Ok(9925)、200(9195)、OK(4954)、2(359)
- 传播相关组件：frontend-2(4)、frontend-0(4)、frontend2-0(4)、frontend-1(3)、currencyservice-2(2)
- 常见调用边：frontend-1->productcatalogservice-1(2)、frontend2-0->productcatalogservice2-0(2)、frontend-1->adservice-1(1)、frontend-2->adservice-1(1)、frontend-1->adservice-2(1)

### container packet loss
- 占比：8/148 (5.4%)
- cloudbed 分布：cloudbed-1(2), cloudbed-2(6)
- 根因层级：service(6), pod(2)
- 高频根因组件：cartservice(2), emailservice(1), productcatalogservice(1), checkoutservice(1), frontend-2(1)
- 典型根因信号：container_network_transmit_packets.eth0 down (覆盖 2/8，均值 2408.03 -> 802.90，代表性 0.18)；container_network_receive_packets.eth0 down (覆盖 2/8，均值 2380.38 -> 980.95，代表性 0.14)；container_network_transmit_MB.eth0 down (覆盖 2/8，均值 2.70 -> 2.06，代表性 0.06)
- 上层业务症状：`metric_service.mrt` 中位数 2.55 -> 968.16，`sr` 中位数 100.0% -> 86.7%
- 服务日志症状：日志类型 log_cartservice-service_application(151982)、log_adservice-service_application(44674)、log_frontend-service_application(2576)；关键词 timeout(390)、error(270)
- 代理日志症状：未扫描
- Trace 症状：span `duration` 中位数 1610.55 -> 242933.47；状态码 0(53211)、Ok(19432)、200(17893)、OK(9242)、2(264)
- 传播相关组件：frontend-0(5)、frontend-1(5)、frontend-2(3)、frontend2-0(3)、checkoutservice-2(2)
- 常见调用边：frontend-0->cartservice-1(2)、frontend-1->cartservice-1(2)、frontend-0->productcatalogservice-1(2)、checkoutservice-2->emailservice-2(1)、checkoutservice2-0->emailservice2-0(1)

### container process termination
- 占比：7/148 (4.7%)
- cloudbed 分布：cloudbed-1(4), cloudbed-2(3)
- 根因层级：service(4), pod(3)
- 高频根因组件：paymentservice(2), recommendationservice2-0(1), shippingservice(1), recommendationservice(1), adservice-0(1)
- 典型根因信号：container_threads up (覆盖 3/7，均值 24.69 -> 25.65，代表性 0.02)
- 上层业务症状：`metric_service.mrt` 中位数 2.49 -> 2.51，`sr` 中位数 100.0% -> 100.0%
- 服务日志症状：日志类型 log_adservice-service_application(14420)、log_shippingservice-service_application(6916)、log_recommendationservice-service_application(3455)；关键词 exception(1442)
- 代理日志症状：未扫描
- Trace 症状：span `duration` 中位数 161.02 -> 242.78；状态码 0(19187)、OK(721)
- 传播相关组件：frontend-0(4)、checkoutservice-1(3)、checkoutservice-0(3)、productcatalogservice-1(3)、frontend-2(3)
- 常见调用边：checkoutservice2-0->paymentservice2-0(2)、checkoutservice-2->paymentservice-0(2)、frontend-0->recommendationservice-1(2)、checkoutservice-1->paymentservice-0(1)、checkoutservice-2->paymentservice-2(1)

### container write I/O load
- 占比：5/148 (3.4%)
- cloudbed 分布：cloudbed-1(1), cloudbed-2(4)
- 根因层级：service(2), pod(3)
- 高频根因组件：emailservice(1), recommendationservice2-0(1), emailservice-1(1), recommendationservice(1), productcatalogservice-0(1)
- 典型根因信号：container_fs_writes./dev/vda up (覆盖 3/5，均值 49.24 -> 4494.07，代表性 1753.39)；container_fs_writes_MB./dev/vda up (覆盖 3/5，均值 23.14 -> 2142.62，代表性 839.86)；container_cpu_cfs_throttled_periods up (覆盖 3/5，均值 2.46 -> 146.80，代表性 47.45)
- 上层业务症状：`metric_service.mrt` 中位数 3.00 -> 3.88，`sr` 中位数 100.0% -> 100.0%
- 服务日志症状：日志类型 log_recommendationservice-service_application(3414)、log_emailservice-service_application(579)
- 代理日志症状：未扫描
- Trace 症状：span `duration` 中位数 2032.48 -> 3225.98；状态码 0(16580)
- 传播相关组件：checkoutservice-1(2)、checkoutservice-2(2)、frontend2-0(2)、productcatalogservice-1(2)、productcatalogservice-0(2)
- 常见调用边：checkoutservice-1->emailservice-1(2)、frontend2-0->recommendationservice2-0(2)、recommendationservice2-0->productcatalogservice-0(2)、recommendationservice-0->productcatalogservice-0(2)、checkoutservice-2->emailservice-0(1)

### node CPU spike
- 占比：4/148 (2.7%)
- cloudbed 分布：cloudbed-1(2), cloudbed-2(2)
- 根因层级：node(4)
- 高频根因组件：node-6(2), node-4(1), node-3(1)
- 典型根因信号：system.cpu.user up (覆盖 4/4，均值 3.10 -> 20.15，代表性 6.88)；system.cpu.pct_usage up (覆盖 4/4，均值 8.17 -> 25.03，代表性 3.19)；system.cpu.iowait up (覆盖 2/4，均值 1.85 -> 1.97，代表性 0.11)
- 上层业务症状：`metric_service.mrt` 中位数 4.95 -> 4.86，`sr` 中位数 96.3% -> 96.3%
- 服务日志症状：日志类型 log_cartservice-service_application(79956)、log_currencyservice-service_application(48858)、log_frontend-service_application(10704)
- 代理日志症状：未扫描
- Trace 症状：span `duration` 中位数 5850.84 -> 5783.11；状态码 0(215178)、Ok(12333)、200(11386)、OK(6057)、2(1568)
- 传播相关组件：frontend2-0(2)、currencyservice2-0(2)、frontend-0(1)、frontend-1(1)、frontend-2(1)
- 常见调用边：frontend2-0->currencyservice2-0(2)、frontend-0->currencyservice-1(1)、frontend-0->currencyservice-0(1)、frontend-0->currencyservice-2(1)、frontend-1->currencyservice-1(1)

## 限制说明
- `Market` 同时包含 `node / pod / service` 三层根因；默认只用根因层直接相关的指标文件做根因信号提取。
- 业务症状统一来自 `metric_service.csv`，节点故障会按当天部署在该节点上的 pod 推导受影响服务集合。
- 默认只扫描 `log_service.csv`；`log_proxy.csv` 和 `trace_span.csv` 体积很大，因此是可选开关。
- 本次运行未扫描 `log_proxy.csv`；如需补充网络/代理侧症状，请加 `--with-proxy-log`。
- 本次运行包含 `trace_span.csv`。