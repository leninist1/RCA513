#!/usr/bin/env python3
"""Assess candidate second datasets for the CAPE-RCA paper artifacts."""

from __future__ import annotations

import csv
from pathlib import Path

from paper_utils import REPO_ROOT, TABLE_DIR


REPORT_DIR = REPO_ROOT / "paper_artifacts/reports"
NOTES_DIR = REPO_ROOT / "paper_artifacts/notes"

CANDIDATES = [
    {
        "dataset": "Eadro",
        "local_path": "/home/dell2/RCA513/syh/datasets/Eadro",
        "local_adapter": "yes",
        "telemetry": "metrics/logs/traces",
        "published_baseline_status": "end-to-end anomaly detection + localization, not known-fault RCA-only",
        "protocol_match": "no",
        "decision": "rejected",
        "reason": "Protocol mismatch: CAPE-RCA result is known-fault localization, while the published Eadro task evaluates an end-to-end RCA pipeline.",
    },
    {
        "dataset": "OpenRCA",
        "local_path": "/home/dell2/RCA513/yyx/OpenRCA",
        "local_adapter": "yes",
        "telemetry": "metrics/logs/traces",
        "published_baseline_status": "LLM natural-language query benchmark; root-cause element scoring differs from RCAEval service-level top-k",
        "protocol_match": "no",
        "decision": "rejected",
        "reason": "Evaluation input/output protocol differs: OpenRCA asks agents to answer natural-language incident queries over long telemetry, not rank RCAEval-style service candidates.",
    },
    {
        "dataset": "AIOps2021",
        "local_path": "adapter-discovered host-level split",
        "local_adapter": "yes",
        "telemetry": "metrics only / host-level in current adapter",
        "published_baseline_status": "no verified multi-source service-level RCA baseline table in admitted sources",
        "protocol_match": "no",
        "decision": "rejected",
        "reason": "Current adapter omits service-level traces/logs and is not comparable to CAPE-RCA's multi-source service-level RCAEval setting.",
    },
]

FIELDS = [
    "dataset",
    "local_path",
    "local_path_exists",
    "local_adapter",
    "telemetry",
    "published_baseline_status",
    "protocol_match",
    "decision",
    "reason",
]


def main() -> int:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    NOTES_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for item in CANDIDATES:
        row = dict(item)
        path = Path(str(row["local_path"]))
        row["local_path_exists"] = "yes" if path.exists() else "not_applicable" if "adapter-discovered" in str(path) else "no"
        rows.append(row)

    out_csv = TABLE_DIR / "second_dataset_admission.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in FIELDS})

    accepted = [row for row in rows if row["decision"] == "accepted"]
    report = [
        "# Second Dataset Selection Report",
        "",
        "## Decision",
        "",
        "No second dataset is admitted to the main paper experiments in this artifact revision.",
        "",
        "The artifact set keeps the candidate audit table but does not generate a misleading second-dataset comparison figure.",
        "",
        "## Candidate Audit",
        "",
        "| dataset | decision | protocol_match | reason |",
        "|---|---|---|---|",
    ]
    for row in rows:
        report.append(
            f"| {row['dataset']} | {row['decision']} | {row['protocol_match']} | {row['reason']} |"
        )
    report.extend(
        [
            "",
            "## Source Notes",
            "",
            "- RCAEval remains the admitted main dataset because it provides service-level top-k RCA baselines under a matching multi-source telemetry protocol.",
            "- OpenRCA is valuable but targets LLM agents answering natural-language incident queries over long telemetry; that output protocol is not directly comparable to RCAEval-style service ranking.",
            "- Eadro remains archived as exploratory evidence only because its published protocol includes anomaly detection and localization rather than CAPE-RCA's known-fault localization setting.",
            "- AIOps2021 is not admitted because the current local adapter is host-level/metrics-only and no compatible published multi-source service-level baseline was admitted.",
            "",
            "## Figure Policy",
            "",
            "Because no second dataset is accepted, `fig6` is intentionally not generated. See `paper_artifacts/notes/second_dataset_figure_not_generated.md`.",
        ]
    )
    if accepted:
        report.append("")
        report.append("Accepted datasets: " + ", ".join(row["dataset"] for row in accepted))
    (REPORT_DIR / "second_dataset_selection_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    (NOTES_DIR / "second_dataset_figure_not_generated.md").write_text(
        "# Second Dataset Figure Not Generated\n\n"
        "No candidate second dataset passed the protocol and baseline comparability gate. "
        "The paper artifact set therefore omits a second-dataset comparison figure instead of plotting non-comparable numbers.\n",
        encoding="utf-8",
    )
    print(f"wrote {out_csv}")
    print(f"wrote {REPORT_DIR / 'second_dataset_selection_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
