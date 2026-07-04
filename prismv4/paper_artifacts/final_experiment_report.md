# CAPE-RCA Final Experiment Report

## 1. Context Admission Report

See `paper_artifacts/context_admission_report.md`. The pipeline indexed relevant result and runner files, excluded full raw telemetry/model I/O/log dumps from context, and admitted only compact summary fields.

## 2. Parsed From Existing Results

- RCAEval RE2/RE3: parsed from `results/prism_cht/v3_full/*.json`.
- Eadro TT/SN: parsed from `results/prism_cht/Eadro-TT_ambig_full.json` and `results/prism_cht/Eadro-SN_ambig_full.json`.

## 3. Newly Run

RCAEval RE1-OB, RE1-SS, and RE1-TT were newly generated with `paper_artifacts/scripts/run_missing_experiments.sh`, which calls `run_re1_metrics_only_adaptive.py`. Strong IVD consensus cases use the adaptive shortcut; non-strong RE1 metrics-only cases are marked timeout/budget-exhausted rather than filled with fabricated EG-CDA outputs.

## 4. Missing Or Incompatible Experiments

- RE1-SS: timeout/budget guarded cases=5
- RE1-TT: timeout/budget guarded cases=22
- RE2-OB: rank_missing=2; top-k/Avg@5 are lower bounds where only top-1 is available; missing_cases=RE2-OB/productcatalogservice_cpu/1, RE2-OB/productcatalogservice_loss/2; top-k_missing_values_counted_as_false_lower_bound
- RE2-SS: rank_missing=5; top-k/Avg@5 are lower bounds where only top-1 is available; missing_cases=RE2-SS/carts_loss/2, RE2-SS/catalogue_socket/1, RE2-SS/orders_cpu/2, RE2-SS/orders_socket/2, RE2-SS/user_socket/2; top-k_missing_values_counted_as_false_lower_bound
- RE3-OB: rank_missing=2; top-k/Avg@5 are lower bounds where only top-1 is available; missing_cases=RE3-OB/cartservice_f1/1, RE3-OB/currencyservice_f1/3
- RE3-SS: rank_missing=6; top-k/Avg@5 are lower bounds where only top-1 is available; missing_cases=RE3-SS/carts_f1/1, RE3-SS/carts_f4/1, RE3-SS/carts_f4/2, RE3-SS/front-end_f2/1, RE3-SS/orders_f1/2, +1 more; top-k_missing_values_counted_as_false_lower_bound
- RE3-TT: rank_missing=1; top-k/Avg@5 are lower bounds where only top-1 is available; missing_cases=RE3-TT/ts-auth-service_f3/3
- Eadro-TT: CAPE-RCA evaluated in known-fault localization setting, not Eadro end-to-end anomaly detection; rank_missing=8; top-k/Avg@5 are lower bounds where only top-1 is available; missing_cases=Eadro-TT/TT.2022-04-17T212101D2022-04-17T230842/fault0, Eadro-TT/TT.2022-04-17T212101D2022-04-17T230842/fault7, Eadro-TT/TT.2022-04-17T212101D2022-04-17T230842/fault8, Eadro-TT/TT.2022-04-18T102520D2022-04-18T121301/fault8, Eadro-TT/TT.2022-04-18T121515D2022-04-18T140256/fault0, +3 more; HR@3/HR@5/NDCG are lower bounds where only top-1 is available
- Eadro-SN: CAPE-RCA evaluated in known-fault localization setting, not Eadro end-to-end anomaly detection; rank_missing=8; top-k/Avg@5 are lower bounds where only top-1 is available; missing_cases=Eadro-SN/SN.2022-04-17T181245D2022-04-17T183616/fault0, Eadro-SN/SN.2022-04-17T181245D2022-04-17T183616/fault7, Eadro-SN/SN.2022-04-17T181245D2022-04-17T183616/fault8, Eadro-SN/SN.2022-04-17T183729D2022-04-17T190100/fault0, Eadro-SN/SN.2022-04-17T183729D2022-04-17T190100/fault4, +3 more; HR@3/HR@5/NDCG are lower bounds where only top-1 is available

## 5. RCAEval All-Subset Summary

| dataset_id | cases | AC@1 | AC@3 | Avg@5 | shortcut_cases | egcda_cases | notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RE1-OB | 125 | 0.720000 | 0.936000 | 0.891200 | 125 | 0 | RE1 is metrics-only; not multi-source causal evidence |
| RE1-SS | 125 | 0.648000 | 0.904000 | 0.870400 | 120 | 0 | RE1 is metrics-only; not multi-source causal evidence; timeout_or_budget_guarded_cases=5; EG-CDA fallback not completed for those metrics-only cases |
| RE1-TT | 125 | 0.296000 | 0.496000 | 0.449600 | 103 | 0 | RE1 is metrics-only; not multi-source causal evidence; timeout_or_budget_guarded_cases=22; EG-CDA fallback not completed for those metrics-only cases |
| RE2-OB | 91 | 0.890110 | 0.956044 | 0.938462 | 89 | 2 | rank_missing=2; top-k/Avg@5 are lower bounds where only top-1 is available; missing_cases=RE2-OB/productcatalogservice_cpu/1, RE2-OB/productcatalogservice_loss/2; top-k_missing_values_counted_as_false_lower_bound |
| RE2-SS | 90 | 0.777778 | 0.822222 | 0.824444 | 85 | 4 | rank_missing=5; top-k/Avg@5 are lower bounds where only top-1 is available; missing_cases=RE2-SS/carts_loss/2, RE2-SS/catalogue_socket/1, RE2-SS/orders_cpu/2, RE2-SS/orders_socket/2, RE2-SS/user_socket/2; top-k_missing_values_counted_as_false_lower_bound |
| RE2-TT | 90 | 0.588889 | 0.711111 | 0.691111 | 78 | 12 |  |
| RE3-OB | 30 | 0.966667 | 1.000000 | 0.993333 | 28 | 2 | rank_missing=2; top-k/Avg@5 are lower bounds where only top-1 is available; missing_cases=RE3-OB/cartservice_f1/1, RE3-OB/currencyservice_f1/3 |
| RE3-SS | 30 | 0.866667 | 0.933333 | 0.926667 | 24 | 6 | rank_missing=6; top-k/Avg@5 are lower bounds where only top-1 is available; missing_cases=RE3-SS/carts_f1/1, RE3-SS/carts_f4/1, RE3-SS/carts_f4/2, RE3-SS/front-end_f2/1, RE3-SS/orders_f1/2, +1 more; top-k_missing_values_counted_as_false_lower_bound |
| RE3-TT | 30 | 0.933333 | 1.000000 | 0.980000 | 29 | 1 | rank_missing=1; top-k/Avg@5 are lower bounds where only top-1 is available; missing_cases=RE3-TT/ts-auth-service_f3/3 |

## 6. Eadro Summary

| dataset | cases | HR@1 | HR@3 | HR@5 | NDCG@5 | egcda_cases | notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Eadro-TT | 8 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 8 | CAPE-RCA evaluated in known-fault localization setting, not Eadro end-to-end anomaly detection; rank_missing=8; top-k/Avg@5 are lower bounds where only top-1 is available; missing_cases=Eadro-TT/TT.2022-04-17T212101D2022-04-17T230842/fault0, Eadro-TT/TT.2022-04-17T212101D2022-04-17T230842/fault7, Eadro-TT/TT.2022-04-17T212101D2022-04-17T230842/fault8, Eadro-TT/TT.2022-04-18T102520D2022-04-18T121301/fault8, Eadro-TT/TT.2022-04-18T121515D2022-04-18T140256/fault0, +3 more; HR@3/HR@5/NDCG are lower bounds where only top-1 is available |
| Eadro-SN | 8 | 0.125000 | 0.125000 | 0.125000 | 0.125000 | 8 | CAPE-RCA evaluated in known-fault localization setting, not Eadro end-to-end anomaly detection; rank_missing=8; top-k/Avg@5 are lower bounds where only top-1 is available; missing_cases=Eadro-SN/SN.2022-04-17T181245D2022-04-17T183616/fault0, Eadro-SN/SN.2022-04-17T181245D2022-04-17T183616/fault7, Eadro-SN/SN.2022-04-17T181245D2022-04-17T183616/fault8, Eadro-SN/SN.2022-04-17T183729D2022-04-17T190100/fault0, Eadro-SN/SN.2022-04-17T183729D2022-04-17T190100/fault4, +3 more; HR@3/HR@5/NDCG are lower bounds where only top-1 is available |

## 7. Published Baselines

### RCAEval

| dataset | suite | system | fault_subset | method | AC@1 | AC@3 | Avg@5 | source_paper | source_table_or_section | notes | comparable_to_CAPERCA |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RE2-TT | RE2 | Train Ticket | CPU; MEM; DISK; SOCKET; DELAY; LOSS | Metric BARO | 0.67 | 0.82 | 0.80 | RCAEval: A Benchmark for Root Cause Analysis of Microservice Systems with Telemetry Data, WWW Companion 2025 / arXiv:2412.17015v5 | Table 6 | Official published coarse-grained service-level RCA result on RE2 Train Ticket average across six fault types. | true |
| RE2-TT | RE2 | Train Ticket | CPU; MEM; DISK; SOCKET; DELAY; LOSS | CausalRCA | 0.22 | 0.47 | 0.43 | RCAEval: A Benchmark for Root Cause Analysis of Microservice Systems with Telemetry Data, WWW Companion 2025 / arXiv:2412.17015v5 | Table 6 | Official published coarse-grained service-level RCA result on RE2 Train Ticket average across six fault types. | true |
| RE2-TT | RE2 | Train Ticket | CPU; MEM; DISK; SOCKET; DELAY; LOSS | Metric CIRCA | 0.32 | 0.47 | 0.46 | RCAEval: A Benchmark for Root Cause Analysis of Microservice Systems with Telemetry Data, WWW Companion 2025 / arXiv:2412.17015v5 | Table 6 | Official published coarse-grained service-level RCA result on RE2 Train Ticket average across six fault types. | true |
| RE2-TT | RE2 | Train Ticket | CPU; MEM; DISK; SOCKET; DELAY; LOSS | MicroCause | 0.10 | 0.22 | 0.20 | RCAEval: A Benchmark for Root Cause Analysis of Microservice Systems with Telemetry Data, WWW Companion 2025 / arXiv:2412.17015v5 | Table 6 | Official published coarse-grained service-level RCA result on RE2 Train Ticket average across six fault types. | true |
| RE2-TT | RE2 | Train Ticket | CPU; MEM; DISK; SOCKET; DELAY; LOSS | Metric RCD | 0.09 | 0.13 | 0.13 | RCAEval: A Benchmark for Root Cause Analysis of Microservice Systems with Telemetry Data, WWW Companion 2025 / arXiv:2412.17015v5 | Table 6 | Official published coarse-grained service-level RCA result on RE2 Train Ticket average across six fault types. | true |
| RE2-TT | RE2 | Train Ticket | CPU; MEM; DISK; SOCKET; DELAY; LOSS | MicroRank | 0.16 | 0.37 | 0.31 | RCAEval: A Benchmark for Root Cause Analysis of Microservice Systems with Telemetry Data, WWW Companion 2025 / arXiv:2412.17015v5 | Table 6 | Official published trace-based coarse-grained service-level RCA result on RE2 Train Ticket average across six fault types. | true |
| RE2-TT | RE2 | Train Ticket | CPU; MEM; DISK; SOCKET; DELAY; LOSS | TraceRCA | 0.66 | 0.79 | 0.77 | RCAEval: A Benchmark for Root Cause Analysis of Microservice Systems with Telemetry Data, WWW Companion 2025 / arXiv:2412.17015v5 | Table 6 | Official published trace-based coarse-grained service-level RCA result on RE2 Train Ticket average across six fault types. | true |
| RE2-TT | RE2 | Train Ticket | CPU; MEM; DISK; SOCKET; DELAY; LOSS | Multi-source BARO | 0.69 | 0.82 | 0.81 | RCAEval: A Benchmark for Root Cause Analysis of Microservice Systems with Telemetry Data, WWW Companion 2025 / arXiv:2412.17015v5 | Table 6 | Official published multi-source coarse-grained service-level RCA result on RE2 Train Ticket average across six fault types. | true |
| RE2-TT | RE2 | Train Ticket | CPU; MEM; DISK; SOCKET; DELAY; LOSS | Multi-source CIRCA | 0.06 | 0.11 | 0.13 | RCAEval: A Benchmark for Root Cause Analysis of Microservice Systems with Telemetry Data, WWW Companion 2025 / arXiv:2412.17015v5 | Table 6 | Official published multi-source coarse-grained service-level RCA result on RE2 Train Ticket average across six fault types. | true |
| RE2-TT | RE2 | Train Ticket | CPU; MEM; DISK; SOCKET; DELAY; LOSS | PDiagnose | 0.48 | 0.70 | 0.67 | RCAEval: A Benchmark for Root Cause Analysis of Microservice Systems with Telemetry Data, WWW Companion 2025 / arXiv:2412.17015v5 | Table 6 | Official published multi-source coarse-grained service-level RCA result on RE2 Train Ticket average across six fault types. | true |
| RE2-TT | RE2 | Train Ticket | CPU; MEM; DISK; SOCKET; DELAY; LOSS | Multi-source RCD | 0.10 | 0.64 | 0.54 | RCAEval: A Benchmark for Root Cause Analysis of Microservice Systems with Telemetry Data, WWW Companion 2025 / arXiv:2412.17015v5 | Table 6 | Official published multi-source coarse-grained service-level RCA result on RE2 Train Ticket average across six fault types. | true |

### Eadro

| dataset | method | HR@1 | HR@3 | HR@5 | NDCG@3 | NDCG@5 | source_paper | source_table_or_section | notes | comparable_to_CAPERCA |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Eadro-TT | TBAC | 0.037 | 0.111 | 0.185 | 0.079 | 0.109 | Eadro: An End-to-End Troubleshooting Framework for Microservices on Multi-source Data, ICSE 2023 / arXiv:2302.05092 | Table III | Published root cause localization metric on TT. Protocol differs from CAPE-RCA because CAPE-RCA is evaluated with known fault cases/localization only. | true |
| Eadro-TT | NetMedic | 0.094 | 0.257 | 0.425 | 0.195 | 0.209 | Eadro: An End-to-End Troubleshooting Framework for Microservices on Multi-source Data, ICSE 2023 / arXiv:2302.05092 | Table III | Published root cause localization metric on TT. Protocol differs from CAPE-RCA because CAPE-RCA is evaluated with known fault cases/localization only. | true |
| Eadro-TT | MonitorRank | 0.086 | 0.199 | 0.331 | 0.142 | 0.196 | Eadro: An End-to-End Troubleshooting Framework for Microservices on Multi-source Data, ICSE 2023 / arXiv:2302.05092 | Table III | Published root cause localization metric on TT. Protocol differs from CAPE-RCA because CAPE-RCA is evaluated with known fault cases/localization only. | true |
| Eadro-TT | CloudRanger | 0.101 | 0.306 | 0.509 | 0.218 | 0.301 | Eadro: An End-to-End Troubleshooting Framework for Microservices on Multi-source Data, ICSE 2023 / arXiv:2302.05092 | Table III | Published root cause localization metric on TT. Protocol differs from CAPE-RCA because CAPE-RCA is evaluated with known fault cases/localization only. | true |
| Eadro-TT | DyCause | 0.231 | 0.615 | 0.808 | 0.448 | 0.607 | Eadro: An End-to-End Troubleshooting Framework for Microservices on Multi-source Data, ICSE 2023 / arXiv:2302.05092 | Table III | Published root cause localization metric on TT. Protocol differs from CAPE-RCA because CAPE-RCA is evaluated with known fault cases/localization only. | true |
| Eadro-TT | MS-RF-RCL | 0.637 | 0.922 | 0.970 | 0.807 | 0.827 | Eadro: An End-to-End Troubleshooting Framework for Microservices on Multi-source Data, ICSE 2023 / arXiv:2302.05092 | Table III | Published root cause localization metric on TT. Protocol differs from CAPE-RCA because CAPE-RCA is evaluated with known fault cases/localization only. | true |
| Eadro-TT | MS-SVM-RCL | 0.541 | 0.908 | 0.944 | 0.814 | 0.820 | Eadro: An End-to-End Troubleshooting Framework for Microservices on Multi-source Data, ICSE 2023 / arXiv:2302.05092 | Table III | Published root cause localization metric on TT. Protocol differs from CAPE-RCA because CAPE-RCA is evaluated with known fault cases/localization only. | true |
| Eadro-TT | MS-LSTM | 0.756 | 0.930 | 0.969 | 0.859 | 0.877 | Eadro: An End-to-End Troubleshooting Framework for Microservices on Multi-source Data, ICSE 2023 / arXiv:2302.05092 | Table III | Published root cause localization metric on TT. Protocol differs from CAPE-RCA because CAPE-RCA is evaluated with known fault cases/localization only; Eadro paper treats MS-LSTM as end-to-end. | true |
| Eadro-TT | MS-DCC | 0.767 | 0.938 | 0.972 | 0.870 | 0.882 | Eadro: An End-to-End Troubleshooting Framework for Microservices on Multi-source Data, ICSE 2023 / arXiv:2302.05092 | Table III | Published root cause localization metric on TT. Protocol differs from CAPE-RCA because CAPE-RCA is evaluated with known fault cases/localization only; Eadro paper treats MS-DCC as end-to-end. | true |
| Eadro-TT | Eadro | 0.990 | 0.992 | 0.993 | 0.994 | 0.994 | Eadro: An End-to-End Troubleshooting Framework for Microservices on Multi-source Data, ICSE 2023 / arXiv:2302.05092 | Table III | Published Eadro result on TT; end-to-end model uses predicted detector output, so not a fully identical protocol to CAPE-RCA localization-only evaluation. | true |
| Eadro-SN | TBAC | 0.001 | 0.085 | 0.181 | 0.048 | 0.087 | Eadro: An End-to-End Troubleshooting Framework for Microservices on Multi-source Data, ICSE 2023 / arXiv:2302.05092 | Table III | Published root cause localization metric on SN. Protocol differs from CAPE-RCA because CAPE-RCA is evaluated with known fault cases/localization only. | true |
| Eadro-SN | NetMedic | 0.069 | 0.187 | 0.373 | 0.146 | 0.218 | Eadro: An End-to-End Troubleshooting Framework for Microservices on Multi-source Data, ICSE 2023 / arXiv:2302.05092 | Table III | Published root cause localization metric on SN. Protocol differs from CAPE-RCA because CAPE-RCA is evaluated with known fault cases/localization only. | true |
| ... |  |  |  |  |  |  |  |  |  |  |

## 8. Figure Files

- `figures/fig4_rcaeval_all_subsets_heatmap.pdf`
- `figures/fig4_rcaeval_all_subsets_heatmap.png`
- `figures/fig5_rcaeval_published_baseline_comparison.pdf`
- `figures/fig5_rcaeval_published_baseline_comparison.png`
- `figures/fig6_eadro_baseline_comparison.pdf`
- `figures/fig6_eadro_baseline_comparison.png`
- `figures/fig7_cost_and_path_distribution.pdf`
- `figures/fig7_cost_and_path_distribution.png`

## 9. Main Observations

- RCAEval RE2/RE3 existing results are available for all six multi-source subsets.
- RE1 is metrics-only and currently missing; it should be interpreted as compatibility coverage after rerun, not as multi-source causal diagnosis evidence.
- CAPE-RCA-Full is reported as an adaptive path: shortcut statuses and continuous/EG-CDA statuses belong to the same main method, with path distribution reported separately.
- Eadro results are localization-only under known fault cases; they should not be described as matching Eadro's full end-to-end anomaly detection protocol.
- Published baseline comparisons remain conservative: unresolved or incompatible values are marked TODO rather than filled from local reproductions.

## 10. Error Analysis Summary

| dataset | case_id | ground_truth | prediction | diagnostic_path | error_type | short_reason |
| --- | --- | --- | --- | --- | --- | --- |
| RCAEval | RE1-OB/adservice_cpu/3 | adservice | cartservice | shortcut | metric_only_evidence_insufficient | metrics-only evidence could not disambiguate top-1 |
| RCAEval | RE1-OB/adservice_delay/2 | adservice | checkoutservice | shortcut | metric_only_evidence_insufficient | metrics-only evidence could not disambiguate top-1 |
| RCAEval | RE1-OB/adservice_delay/4 | adservice | checkoutservice | shortcut | metric_only_evidence_insufficient | metrics-only evidence could not disambiguate top-1 |
| RCAEval | RE1-OB/adservice_loss/1 | adservice | checkoutservice | shortcut | metric_only_evidence_insufficient | metrics-only evidence could not disambiguate top-1 |
| RCAEval | RE1-OB/adservice_loss/2 | adservice | checkoutservice | shortcut | metric_only_evidence_insufficient | metrics-only evidence could not disambiguate top-1 |
| RCAEval | RE1-OB/cartservice_delay/1 | cartservice | frontend | shortcut | metric_only_evidence_insufficient | metrics-only evidence could not disambiguate top-1 |
| RCAEval | RE1-OB/cartservice_delay/2 | cartservice | checkoutservice | shortcut | metric_only_evidence_insufficient | metrics-only evidence could not disambiguate top-1 |
| RCAEval | RE1-OB/cartservice_delay/3 | cartservice | frontend | shortcut | metric_only_evidence_insufficient | metrics-only evidence could not disambiguate top-1 |
| RCAEval | RE1-OB/cartservice_delay/4 | cartservice | frontend | shortcut | metric_only_evidence_insufficient | metrics-only evidence could not disambiguate top-1 |
| RCAEval | RE1-OB/cartservice_delay/5 | cartservice | frontend | shortcut | metric_only_evidence_insufficient | metrics-only evidence could not disambiguate top-1 |
| RCAEval | RE1-OB/cartservice_loss/4 | cartservice | recommendationservice | shortcut | metric_only_evidence_insufficient | metrics-only evidence could not disambiguate top-1 |
| RCAEval | RE1-OB/cartservice_loss/5 | cartservice | frontend | shortcut | metric_only_evidence_insufficient | metrics-only evidence could not disambiguate top-1 |
| RCAEval | RE1-OB/checkoutservice_delay/2 | checkoutservice | cartservice | shortcut | metric_only_evidence_insufficient | metrics-only evidence could not disambiguate top-1 |
| RCAEval | RE1-OB/checkoutservice_loss/1 | checkoutservice | paymentservice | shortcut | metric_only_evidence_insufficient | metrics-only evidence could not disambiguate top-1 |
| RCAEval | RE1-OB/checkoutservice_loss/2 | checkoutservice | emailservice | shortcut | metric_only_evidence_insufficient | metrics-only evidence could not disambiguate top-1 |
| RCAEval | RE1-OB/checkoutservice_loss/3 | checkoutservice | adservice | shortcut | metric_only_evidence_insufficient | metrics-only evidence could not disambiguate top-1 |
| RCAEval | RE1-OB/checkoutservice_loss/4 | checkoutservice | emailservice | shortcut | metric_only_evidence_insufficient | metrics-only evidence could not disambiguate top-1 |
| RCAEval | RE1-OB/checkoutservice_loss/5 | checkoutservice | emailservice | shortcut | metric_only_evidence_insufficient | metrics-only evidence could not disambiguate top-1 |
| RCAEval | RE1-OB/currencyservice_cpu/1 | currencyservice | recommendationservice | shortcut | metric_only_evidence_insufficient | metrics-only evidence could not disambiguate top-1 |
| RCAEval | RE1-OB/currencyservice_delay/1 | currencyservice | frontend | shortcut | metric_only_evidence_insufficient | metrics-only evidence could not disambiguate top-1 |
| ... |  |  |  |  |  |  |

## 11. Reproduction Commands

```bash
python3 paper_artifacts/scripts/collect_existing_results.py
python3 paper_artifacts/scripts/compute_metrics.py
python3 paper_artifacts/scripts/make_figures.py
python3 paper_artifacts/scripts/extract_case_study.py
python3 paper_artifacts/scripts/build_experiment_manifest.py
```
