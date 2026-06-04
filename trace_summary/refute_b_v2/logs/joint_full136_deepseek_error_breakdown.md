# DeepSeek Full136 Error Breakdown

- rows: 136
- delta vs rule: {'same': 132, 'improved': 4}

## By GT Reason

| group | n | strict | partial | time | component | reason |
|---|---:|---:|---:|---:|---:|---:|
| high CPU usage | 31 | 25.81% | 31.18% | 12.50% | 50.00% | 22.22% |
| network packet loss | 29 | 3.45% | 8.05% | 23.53% | 0.00% | 0.00% |
| network latency | 24 | 0.00% | 13.19% | 0.00% | 20.00% | 33.33% |
| high disk I/O read usage | 16 | 31.25% | 50.52% | 33.33% | 69.23% | 66.67% |
| high memory usage | 9 | 0.00% | 5.56% | 0.00% | 50.00% | 0.00% |
| JVM Out of Memory (OOM) Heap | 6 | 0.00% | 0.00% | 0.00% | - | 0.00% |
| high disk space usage | 5 | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| high JVM CPU load | 3 | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| network latency+network packet loss | 3 | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| JVM Out of Memory (OOM) Heap+high disk I/O read usage | 2 | 0.00% | 62.50% | 0.00% | 100.00% | 50.00% |
| high CPU usage+high memory usage | 2 | 0.00% | 12.50% | 0.00% | 50.00% | - |
| high CPU usage+network packet loss | 2 | 0.00% | 33.33% | 0.00% | 50.00% | 50.00% |
| high disk I/O read usage+network latency | 2 | 0.00% | 75.00% | 100.00% | 50.00% | 50.00% |
| high disk I/O read usage+network packet loss | 2 | 0.00% | 29.17% | 0.00% | 50.00% | 50.00% |

## By Reason Group

| group | n | strict | partial | time | component | reason |
|---|---:|---:|---:|---:|---:|---:|
| event_like | 92 | 9.78% | 17.21% | 11.11% | 22.81% | 17.86% |
| progressive_like | 20 | 0.00% | 2.50% | 0.00% | 33.33% | 0.00% |
| io_like | 16 | 31.25% | 50.52% | 33.33% | 69.23% | 66.67% |
| mixed | 8 | 0.00% | 44.79% | 28.57% | 70.00% | 50.00% |

## By Task Index

| group | n | strict | partial | time | component | reason |
|---|---:|---:|---:|---:|---:|---:|
| task_1 | 25 | 8.00% | 8.00% | 6.90% | - | - |
| task_2 | 22 | 9.09% | 9.09% | - | - | 9.09% |
| task_6 | 21 | 14.29% | 32.14% | - | 30.43% | 34.78% |
| task_4 | 18 | 11.11% | 19.44% | 20.00% | - | 25.00% |
| task_5 | 18 | 0.00% | 29.17% | 21.74% | 43.48% | - |
| task_7 | 17 | 0.00% | 20.59% | 14.29% | 38.10% | 23.81% |
| task_3 | 15 | 33.33% | 33.33% | - | 31.25% | - |

## By Time Anchor Kind

| group | n | strict | partial | time | component | reason |
|---|---:|---:|---:|---:|---:|---:|
| candidate_seed_nearest | 71 | 16.90% | 31.34% | 21.05% | 59.09% | 32.56% |
| candidate_seed_only | 49 | 2.04% | 7.48% | 4.00% | 7.14% | 12.50% |
| cross_metric_vote | 16 | 6.25% | 13.02% | 9.09% | 18.18% | 18.18% |

## By Confidence

| group | n | strict | partial | time | component | reason |
|---|---:|---:|---:|---:|---:|---:|
| MEDIUM | 91 | 14.29% | 26.74% | 18.57% | 48.28% | 28.07% |
| LOW | 45 | 2.22% | 8.15% | 4.35% | 8.00% | 13.79% |

## Final Changed From Rule

| group | n | strict | partial | time | component | reason |
|---|---:|---:|---:|---:|---:|---:|
| same_as_rule | 132 | 10.61% | 19.13% | 11.49% | 32.47% | 21.95% |
| changed | 4 | 0.00% | 68.75% | 66.67% | 83.33% | 50.00% |

## Final Score Delta Vs Rule

| group | n | strict | partial | time | component | reason |
|---|---:|---:|---:|---:|---:|---:|
| same | 132 | 10.61% | 19.13% | 11.49% | 32.47% | 21.95% |
| improved | 4 | 0.00% | 68.75% | 66.67% | 83.33% | 50.00% |

## Focus By GT Reason

| group | n | strict | partial | time | component | reason |
|---|---:|---:|---:|---:|---:|---:|
| network packet loss | 29 | 3.45% | 8.05% | 23.53% | 0.00% | 0.00% |
| network latency | 24 | 0.00% | 13.19% | 0.00% | 20.00% | 33.33% |
| high CPU usage | 14 | 7.14% | 9.52% | 16.67% | 14.29% | 0.00% |
| high memory usage | 9 | 0.00% | 5.56% | 0.00% | 50.00% | 0.00% |
| JVM Out of Memory (OOM) Heap | 6 | 0.00% | 0.00% | 0.00% | - | 0.00% |
| high disk I/O read usage | 5 | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| high disk space usage | 3 | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| network latency+network packet loss | 3 | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| JVM Out of Memory (OOM) Heap+high disk I/O read usage | 2 | 0.00% | 62.50% | 0.00% | 100.00% | 50.00% |
| high CPU usage+high memory usage | 2 | 0.00% | 12.50% | 0.00% | 50.00% | - |
| high CPU usage+network packet loss | 2 | 0.00% | 33.33% | 0.00% | 50.00% | 50.00% |
| high disk I/O read usage+network latency | 2 | 0.00% | 75.00% | 100.00% | 50.00% | 50.00% |
| high disk I/O read usage+network packet loss | 2 | 0.00% | 29.17% | 0.00% | 50.00% | 50.00% |
| high JVM CPU load | 1 | 0.00% | 0.00% | - | - | 0.00% |

## Focus By Time Anchor Kind

| group | n | strict | partial | time | component | reason |
|---|---:|---:|---:|---:|---:|---:|
| candidate_seed_only | 49 | 2.04% | 7.48% | 4.00% | 7.14% | 12.50% |
| candidate_seed_nearest | 44 | 2.27% | 17.42% | 20.00% | 42.31% | 18.75% |
| cross_metric_vote | 11 | 0.00% | 2.27% | 0.00% | 11.11% | 0.00% |

## Focus Delta Vs Rule

| group | n | strict | partial | time | component | reason |
|---|---:|---:|---:|---:|---:|---:|
| same | 100 | 2.00% | 8.83% | 7.58% | 15.79% | 11.94% |
| improved | 4 | 0.00% | 68.75% | 66.67% | 83.33% | 50.00% |
