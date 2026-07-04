#!/usr/bin/env python3
"""Write a compact source-vs-symptom case study artifact."""

from __future__ import annotations

import json

from paper_utils import CASE_STUDY_DIR


CASE_STUDY = {
    "case_id": "RE3-TT/ts-auth-service_f3/3",
    "dataset": "RCAEval",
    "suite": "RE3",
    "system": "Train Ticket",
    "ground_truth": "ts-auth-service",
    "top_1_prediction": "ts-auth-service",
    "diagnostic_path": "continuous_final",
    "confusing_symptom_candidate": "ts-inside-payment-service",
    "source_evidence": [
        "near-onset multi-signal internal anomalies on ts-auth-service",
        "IVD marks ts-auth-service as the initiator",
        "downstream/counterfactual checks explain symptom propagation",
    ],
    "symptom_evidence": [
        "ts-inside-payment-service is affected but lacks stronger independent source evidence",
        "visible downstream errors are interpreted as propagation evidence",
    ],
    "final_interpretation": (
        "CAPE-RCA selects ts-auth-service because its local mechanism and IVD role "
        "explain the downstream symptoms better than choosing the visible symptom component."
    ),
}


def main() -> int:
    CASE_STUDY_DIR.mkdir(parents=True, exist_ok=True)
    (CASE_STUDY_DIR / "case_study_source_symptom.json").write_text(
        json.dumps(CASE_STUDY, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    md = [
        "# Case Study: Source-vs-Symptom Disambiguation",
        "",
        "## Case",
        "",
        f"- Case id: `{CASE_STUDY['case_id']}`",
        f"- Dataset / suite / system: {CASE_STUDY['dataset']} / {CASE_STUDY['suite']} / {CASE_STUDY['system']}",
        f"- Ground truth: `{CASE_STUDY['ground_truth']}`",
        f"- Top-1 prediction: `{CASE_STUDY['top_1_prediction']}`",
        f"- Diagnostic path: `{CASE_STUDY['diagnostic_path']}`",
        "",
        "## Why This Case Matters",
        "",
        "`ts-inside-payment-service` is a confusing symptom candidate, but CAPE-RCA keeps `ts-auth-service` as the root.",
        "",
        "## Source Evidence",
        "",
    ]
    md.extend(f"- {item}" for item in CASE_STUDY["source_evidence"])
    md.extend(["", "## Symptom Evidence", ""])
    md.extend(f"- {item}" for item in CASE_STUDY["symptom_evidence"])
    md.extend(["", "## Final Interpretation", "", CASE_STUDY["final_interpretation"], "", "## Notes", "", "Extracted from compact result fields only; raw telemetry and full transcripts are not copied."])
    (CASE_STUDY_DIR / "case_study_source_symptom.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"wrote {CASE_STUDY_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
