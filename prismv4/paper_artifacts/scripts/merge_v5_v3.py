#!/usr/bin/env python3
"""Merge v5 IVD-only results with v3 results for skipped (no-consensus) cases.

For each case with status 'ivd_no_consensus_skipped', reuse the prediction
from the v3 result (which may include LLM agent records).
"""
import json
from pathlib import Path

V5_DIR = Path("prismv4/results/prism_cht/v5_optimized")
V3_DIR = Path("prismv4/results/prism_cht/v3_full")
OUT_DIR = V5_DIR  # overwrite in place

DATASETS = ["RE2-OB", "RE2-SS", "RE2-TT", "RE3-OB", "RE3-SS", "RE3-TT"]

for ds in DATASETS:
    v5_path = V5_DIR / f"{ds}_v5.json"
    v3_path = V3_DIR / f"{ds}_v3.json"
    if not v5_path.exists() or not v3_path.exists():
        print(f"SKIP {ds}: missing file")
        continue

    v5 = json.loads(v5_path.read_text())
    v3 = json.loads(v3_path.read_text())
    v3_map = {r["case_id"]: r for r in v3.get("results", [])}

    reused = 0
    for r in v5["results"]:
        if r.get("status") == "ivd_no_consensus_skipped":
            v3r = v3_map.get(r["case_id"])
            if v3r:
                r["predicted_component"] = v3r.get("predicted_component")
                r["hit"] = v3r.get("hit", False)
                r["status"] = "reused_from_v3"
                r["reason_family"] = v3r.get("reason_family", "v3_reuse")
                r["rationale"] = f"Reused from v3: {v3r.get('rationale','')}"
                r["predicted_ranking"] = v3r.get("predicted_ranking", r.get("predicted_ranking", []))
                reused += 1

    hits = sum(1 for r in v5["results"] if r.get("hit"))
    total = len(v5["results"])
    v5["top1_hits"] = hits
    v5["top1_accuracy"] = hits / total if total else 0.0

    v5_path.write_text(json.dumps(v5, indent=2, ensure_ascii=False))
    print(f"{ds}: AC@1={v5['top1_accuracy']:.4f} ({hits}/{total}) reused={reused}")

print("Merge done.")
