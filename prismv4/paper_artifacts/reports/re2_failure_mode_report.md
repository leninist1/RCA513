# RE2 Failure Mode Report (before)

## Scope

This report analyzes RE2 compact result summaries only. It does not copy raw metrics, logs, or traces into the artifact set.

## Aggregate Findings

- Total RE2 top-1 misses or failed cases: 67.
- Misses with ground truth absent from exported top candidates: 41.
- Misses on shortcut/fallback paths: 63.

## Error Type Counts

| error_type | cases |
|---|---:|
| candidate_pool_miss | 40 |
| false_cpsi_shortcut | 16 |
| missing_trace_direction | 10 |
| parser_or_adapter_issue | 1 |

## Fault Type Counts

| fault_type | cases |
|---|---:|
| delay | 18 |
| loss | 15 |
| cpu | 13 |
| socket | 12 |
| mem | 6 |
| disk | 3 |

## System Breakdown

| system | misses | dominant_error_types |
|---|---:|---|
| RE2-Online Boutique | 10 | candidate_pool_miss:4, false_cpsi_shortcut:3, missing_trace_direction:3 |
| RE2-Sock Shop | 20 | candidate_pool_miss:12, false_cpsi_shortcut:4, missing_trace_direction:3 |
| RE2-Train Ticket | 37 | candidate_pool_miss:24, false_cpsi_shortcut:9, missing_trace_direction:4 |

## Optimization Implications

- RE2-TT has many shortcut/fallback misses; the IVD shortcut should become more conservative on noisy multi-service cases.
- Ground-truth absence in top candidates points to recall/ranking width rather than final reasoning alone.
- Network and latency faults need explicit trace-direction checks before shortcut acceptance.
- Resource-fault misses should be handled by owner-sensitive resource evidence, not by case or label-specific rules.
