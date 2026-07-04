# Case Study: Source-vs-Symptom Disambiguation

- Case id: `RE3-TT/ts-auth-service_f3/3`
- Dataset / suite / system: RCAEval / RE3 / Train Ticket
- Ground truth: `ts-auth-service`
- Top-1 prediction: `ts-auth-service`
- Confusing symptom candidate: `ts-inside-payment-service`
- Diagnostic path: `continuous_final`

## Candidate List

`ts-admin-basic-info-service`, `ts-travel2-service`, `ts-admin-travel-service`, `ts-assurance-service`, `ts-contacts-service`, `ts-preserve-service`, `ts-payment-service`, `ts-auth-service`, `ts-notification-mongo`, `ts-travel-service`, `ts-food-service`, `ts-order-service`, `ts-preserve-other-service`, `ts-inside-payment-service`, `ts-consign-service`

## Local Evidence For Source

{"contradicting_factors": ["no direct request paths to symptom components in trace, but side-channel auth call (e.g., Spring Security filter) explains absence; counterfactual still supports propagation"], "hypothesis_id": "H8-ts-auth-service-emitter-exception", "missing_information": [], "position": "leading", "root_component": "ts-auth-service", "source_vs_symptom_judgment": "strong source", "supporting_factors": ["source_likelihood_score 6.0, highest among compared candidates; near-onset multi-signal anomaly (memory +672%, CPU +245%, socket +26%, anomalous INFO logs)", "IVD initiator with Copeland 6 and own-code stack trace; ranking first with exogeneity and coverage", "test_downstream_explanation and counterfactual_remove confirm ts-auth-service observed before all symptom components; widespread TokenException errors consistent with auth failure propagation; no trace paths to sympt...

## Local Evidence For Symptom Candidate

 trace, but side-channel auth call (e.g., Spring Security filter) explains absence; counterfactual still supports propagation", "recall_pool": ["ts-admin-basic-info-service", "ts-travel2-service", "ts-admin-travel-service", "ts-assurance-service", "ts-contacts-service", "ts-preserve-service", "ts-payment-service", "ts-auth-service", "ts-notification-mongo", "ts-travel-service", "ts-food-service", "ts-order-service", "ts-preserve-other-service", "ts-inside-payment-service", "ts-consign-service"], "transcript": [{"agent_response": {"agent": "continuous_noise_native", "belief_state": [{"contradicting_factors": [], "hypothesis_id": "H8-ts-auth-service-emitter-exception", "missing_information": ["source-vs-symptom comparison scores", "confirmation that ts-auth-service is called synchronously on every request", "whether its dependency (ts-auth-mongo) is healthy or reactive", "downstream explanation facts to ensure no other candidate explains more symptoms"], "position": "leading", "root_component": "ts-auth-service", "source_vs_symptom_judgment": "likely source", "supporting_factors": ["IVD initiator with highest Copeland score (6) and own-code stack trace", "extreme memory spike (+672%) and CPU spike (+245%) near onset at 1733649309", "anomalous INFO logs suggest internal reconfiguration", "TokenException errors appear simultaneously

## CPSI / Belief Assessment

[{"contradicting_factors": ["no direct request paths to symptom components in trace, but side-channel auth call (e.g., Spring Security filter) explains absence; counterfactual still supports propagation"], "hypothesis_id": "H8-ts-auth-service-emitter-exception", "missing_information": [], "position": "leading", "root_component": "ts-auth-service", "source_vs_symptom_judgment": "strong source", "supporting_factors": ["source_likelihood_score 6.0, highest among compared candidates; near-onset multi-signal anomaly (memory +672%, CPU +245%, socket +26%, anomalous INFO logs)", "IVD initiator with Copeland 6 and own-code stack trace; ranking first with exogeneity and coverage", "test_downstream_explanation and counterfactual_remove confirm ts-auth-service observed before all symptom components; widespread TokenException errors consistent with auth failure propagation; no trace paths to symptoms but likely side-channel invocation (Spring Security filter)", "counterfactual facts show all symptom components emit TokenException errors identical to ts-auth-service pattern; removal of ts-auth-service would likely eliminate symptom cascade", "anomalous INFO logs indicate internal reconfiguration, not dependency failure; ts-auth-mongo resource check not needed as emitter exception dominates"], "why_not_others": "ts-travel2-service is IVD ambiguous despite high Copeland; dominant error spi...

## EG-CDA Selected Actions

`compare_source_symptom`, `test_downstream_explanation`, `counterfactual_remove`, `inspect_noise_features`

## Final Explanation

strong source Support: source_likelihood_score 6.0, highest among compared candidates; near-onset multi-signal anomaly (memory +672%, CPU +245%, socket +26%, anomalous INFO logs); IVD initiator with Copeland 6 and own-code stack trace; ranking first with exogeneity and coverage; test_downstream_explanation and counterfactual_remove confirm ts-auth-service observed before all symptom components; widespread TokenException errors consistent with auth failure propagation; no trace paths to symptoms but likely side-channel invocation (Spring Security filter) Caveats: no direct request paths to symptom components in trace, but side-channel auth call (e.g., Spring Security filter) explains absence; counterfactual still supports propagation

## Rejected Alternatives

`ts-admin-basic-info-service`, `ts-travel2-service`, `ts-admin-travel-service`, `ts-assurance-service`, `ts-contacts-service`, `ts-preserve-service`, `ts-payment-service`, `ts-notification-mongo`

## Remaining Uncertainty

[]

## Notes

Extracted from compact result fields only. Full raw telemetry and full transcript are not copied.
