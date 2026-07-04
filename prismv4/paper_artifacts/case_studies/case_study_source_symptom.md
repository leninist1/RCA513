# Case Study: Source-vs-Symptom Disambiguation

## Case

- Case id: `RE3-TT/ts-auth-service_f3/3`
- Dataset / suite / system: RCAEval / RE3 / Train Ticket
- Ground truth: `ts-auth-service`
- Top-1 prediction: `ts-auth-service`
- Diagnostic path: `continuous_final`

## Why This Case Matters

`ts-inside-payment-service` is a confusing symptom candidate, but CAPE-RCA keeps `ts-auth-service` as the root.

## Source Evidence

- near-onset multi-signal internal anomalies on ts-auth-service
- IVD marks ts-auth-service as the initiator
- downstream/counterfactual checks explain symptom propagation

## Symptom Evidence

- ts-inside-payment-service is affected but lacks stronger independent source evidence
- visible downstream errors are interpreted as propagation evidence

## Final Interpretation

CAPE-RCA selects ts-auth-service because its local mechanism and IVD role explain the downstream symptoms better than choosing the visible symptom component.

## Notes

Extracted from compact result fields only; raw telemetry and full transcripts are not copied.
