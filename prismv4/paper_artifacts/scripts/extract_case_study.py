#!/usr/bin/env python3
"""Extract one compact source-vs-symptom case study."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from collect_existing_results import collect
from paper_utils import (
    CASE_STUDY_DIR,
    RAW_DIR,
    RCAEVAL_SOURCE_FILES,
    compact_text,
    ensure_dirs,
    load_json,
    normalize_ranking,
    recall_pool_components,
    write_json,
)


PREFERRED_GT = "ts-auth-service"
PREFERRED_SYMPTOM = "ts-inside-payment-service"


def _source_cases() -> list[dict[str, Any]]:
    source = RCAEVAL_SOURCE_FILES["RE3-TT"]
    if not source.exists():
        return []
    data = load_json(source)
    results = data.get("results") or []
    return [row for row in results if isinstance(row, dict)]


def _score_case(row: dict[str, Any]) -> tuple[int, str]:
    gt = str(row.get("expected_component") or "")
    pred = str(row.get("predicted_component") or "")
    ranking = normalize_ranking(row.get("predicted_ranking"))
    recall = recall_pool_components(row.get("recall_pool"))
    hit = gt == pred
    score = 0
    if hit:
        score += 10
    if gt == PREFERRED_GT:
        score += 10
    if PREFERRED_SYMPTOM in ranking:
        score += 8
    if PREFERRED_SYMPTOM in recall:
        score += 4
    if str(row.get("status") or "") == "continuous_final":
        score += 12
    if row.get("evidence_count"):
        score += 3
    if row.get("transcript"):
        score += 3
    return score, str(row.get("case_id") or "")


def _candidate_components(row: dict[str, Any]) -> list[str]:
    ranking = normalize_ranking(row.get("predicted_ranking"))
    recall = recall_pool_components(row.get("recall_pool"))
    candidates: list[str] = []
    for item in ranking + recall:
        if item and item not in candidates:
            candidates.append(item)
    return candidates[:15]


def _brief_profile(row: dict[str, Any], component: str) -> str:
    belief = row.get("final_belief_state")
    if isinstance(belief, list):
        for item in belief:
            if isinstance(item, dict) and item.get("root_component") == component:
                return compact_text(item, limit=900)
    profile = row.get("event_causal_profile")
    if isinstance(profile, dict):
        for key, value in profile.items():
            if component in str(key):
                return compact_text(value, limit=900)
        text = compact_text(profile, limit=1800)
        if component in text:
            return text
    selected = {
        "recall_pool": row.get("recall_pool"),
        "rationale": row.get("rationale"),
        "transcript": row.get("transcript"),
    }
    text = compact_text(selected, limit=5000)
    idx = text.find(component)
    if idx >= 0:
        start = max(0, idx - 450)
        end = min(len(text), idx + 900)
        return text[start:end]
    return "No component-specific compact evidence was available in the result file."


def _actions(row: dict[str, Any]) -> list[str]:
    transcript = row.get("transcript")
    actions: list[str] = []
    if isinstance(transcript, list):
        for item in transcript:
            if not isinstance(item, dict):
                continue
            executed = item.get("executed_action")
            action = item.get("action") or item.get("tool_name") or item.get("name")
            if not action and isinstance(executed, dict):
                action = (
                    executed.get("action_type")
                    or executed.get("tool_name")
                    or executed.get("name")
                    or executed.get("type")
                )
            if isinstance(action, dict):
                action = action.get("action_type") or action.get("tool_name")
            if action:
                text = str(action)
                if text not in actions:
                    actions.append(text)
    return actions[:12]


def extract() -> dict[str, Any]:
    ensure_dirs()
    if not (RAW_DIR / "normalized_cases.json").exists():
        collect()
    cases = _source_cases()
    if not cases:
        raise SystemExit("RE3-TT source result not found")
    selected = sorted(cases, key=_score_case, reverse=True)[0]
    candidates = _candidate_components(selected)
    ground_truth = str(selected.get("expected_component") or "")
    prediction = str(selected.get("predicted_component") or "")
    confusing = PREFERRED_SYMPTOM if PREFERRED_SYMPTOM in candidates else next(
        (item for item in candidates if item != ground_truth),
        "",
    )
    study = {
        "case_id": selected.get("case_id"),
        "dataset": "RCAEval",
        "suite": "RE3",
        "system": "Train Ticket",
        "ground_truth": ground_truth,
        "top_1_prediction": prediction,
        "hit@1": int(ground_truth == prediction),
        "diagnostic_path": selected.get("status"),
        "candidate_list": candidates,
        "confusing_symptom_candidate": confusing,
        "local_evidence_for_source": _brief_profile(selected, ground_truth),
        "local_evidence_for_symptom": _brief_profile(selected, confusing) if confusing else "",
        "CPSI_pairwise_relation": compact_text(selected.get("final_belief_state"), limit=1400),
        "EG_CDA_selected_actions": _actions(selected),
        "tool_outputs_or_evidence_atoms": {
            "evidence_count": selected.get("evidence_count"),
            "referenced_evidence_ids": selected.get("referenced_evidence_ids") or [],
            "event_causal_fact_count": selected.get("event_causal_fact_count"),
        },
        "evidence_assessment": compact_text(selected.get("rationale"), limit=1200),
        "final_explanation": compact_text(selected.get("rationale"), limit=1200),
        "rejected_alternatives": [item for item in candidates if item != prediction][:8],
        "remaining_uncertainty": compact_text(selected.get("uncertainties"), limit=900),
        "notes": (
            "Extracted from compact result fields only. Full raw telemetry and full transcript are not copied."
        ),
    }
    return study


def write_markdown(study: dict[str, Any]) -> None:
    lines = [
        "# Case Study: Source-vs-Symptom Disambiguation",
        "",
        f"- Case id: `{study['case_id']}`",
        f"- Dataset / suite / system: {study['dataset']} / {study['suite']} / {study['system']}",
        f"- Ground truth: `{study['ground_truth']}`",
        f"- Top-1 prediction: `{study['top_1_prediction']}`",
        f"- Confusing symptom candidate: `{study['confusing_symptom_candidate']}`",
        f"- Diagnostic path: `{study['diagnostic_path']}`",
        "",
        "## Candidate List",
        "",
        ", ".join(f"`{item}`" for item in study["candidate_list"]),
        "",
        "## Local Evidence For Source",
        "",
        study["local_evidence_for_source"],
        "",
        "## Local Evidence For Symptom Candidate",
        "",
        study["local_evidence_for_symptom"],
        "",
        "## CPSI / Belief Assessment",
        "",
        study["CPSI_pairwise_relation"],
        "",
        "## EG-CDA Selected Actions",
        "",
        ", ".join(f"`{item}`" for item in study["EG_CDA_selected_actions"]) or "No EG-CDA actions recorded for this case.",
        "",
        "## Final Explanation",
        "",
        study["final_explanation"],
        "",
        "## Rejected Alternatives",
        "",
        ", ".join(f"`{item}`" for item in study["rejected_alternatives"]),
        "",
        "## Remaining Uncertainty",
        "",
        study["remaining_uncertainty"],
        "",
        "## Notes",
        "",
        study["notes"],
        "",
    ]
    (CASE_STUDY_DIR / "case_study_source_symptom.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract compact source-vs-symptom case study.")
    parser.parse_args()
    study = extract()
    write_json(CASE_STUDY_DIR / "case_study_source_symptom.json", study)
    write_markdown(study)
    print(f"wrote case study under {CASE_STUDY_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
