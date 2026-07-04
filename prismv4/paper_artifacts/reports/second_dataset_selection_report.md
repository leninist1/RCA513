# Second Dataset Selection Report

## Decision

No second dataset is admitted to the main paper experiments in this artifact revision.

The artifact set keeps the candidate audit table but does not generate a misleading second-dataset comparison figure.

## Candidate Audit

| dataset | decision | protocol_match | reason |
|---|---|---|---|
| Eadro | rejected | no | Protocol mismatch: CAPE-RCA result is known-fault localization, while the published Eadro task evaluates an end-to-end RCA pipeline. |
| OpenRCA | rejected | no | Evaluation input/output protocol differs: OpenRCA asks agents to answer natural-language incident queries over long telemetry, not rank RCAEval-style service candidates. |
| AIOps2021 | rejected | no | Current adapter omits service-level traces/logs and is not comparable to CAPE-RCA's multi-source service-level RCAEval setting. |

## Source Notes

- RCAEval remains the admitted main dataset because it provides service-level top-k RCA baselines under a matching multi-source telemetry protocol: https://github.com/phamquiluan/RCAEval and https://arxiv.org/html/2412.17015v5.
- OpenRCA is valuable but targets LLM agents answering natural-language incident queries over long telemetry; that output protocol is not directly comparable to RCAEval-style service ranking: https://github.com/microsoft/OpenRCA and https://openreview.net/forum?id=M4qNIzQYpd.
- Eadro remains archived as exploratory evidence only because its published protocol includes anomaly detection and localization rather than CAPE-RCA's known-fault localization setting: https://arxiv.org/pdf/2302.05092.
- AIOps2021 is not admitted because the current local adapter is host-level/metrics-only and no compatible published multi-source service-level baseline was admitted.

## Figure Policy

Because no second dataset is accepted, `fig6` is intentionally not generated. See `paper_artifacts/notes/second_dataset_figure_not_generated.md`.
