#!/usr/bin/env python3
"""Generate per-fault-type comparison table: CAPE-RCA v5 vs 4 multi-source baselines."""
import csv
import json
from collections import defaultdict
from pathlib import Path

REPO = Path("/home/dell2/RCA513/yyx")
V5_DIR = REPO / "prismv4/results/prism_cht/v5_optimized"
BASELINE_CSV = REPO / "prismv4/results/rcaeval_official_multisource_re2_re3/case_records.csv"
PUB_BASELINE = REPO / "prismv4/paper_artifacts/tables/published_baselines_rcaeval.csv"
OUT_CSV = REPO / "prismv4/paper_artifacts/tables/cape_rca_vs_baselines_per_fault.csv"
REPORT_FILE = REPO / "prismv4/paper_artifacts/tables/per_fault_comparison.md"

RE2_FAULTS = ["cpu", "mem", "disk", "socket", "delay", "loss"]
RE3_FAULTS = ["f1", "f2", "f3", "f4", "f5"]
ALL_FAULTS = RE2_FAULTS + RE3_FAULTS
DATASETS = ["RE2-OB", "RE2-SS", "RE2-TT", "RE3-OB", "RE3-SS", "RE3-TT"]
METHODS_ORDER = ["cape-rca", "baro", "circa", "pdiagnose", "rcd"]

def faults_for(ds: str) -> list[str]:
    return RE2_FAULTS if ds.startswith("RE2") else RE3_FAULTS

def fault_from_case_id(case_id: str) -> str:
    parts = case_id.split("/")
    if len(parts) >= 2:
        sp = parts[1]
        # RE2: "checkoutservice_cpu" → "cpu"
        for f in RE2_FAULTS:
            if sp.endswith(f"_{f}"):
                return f
        # RE3: "cartservice_f1" or "ts-route-service_f3_1"
        for f in RE3_FAULTS:
            if f"_{f}" in sp:
                return f
    return "unknown"

def rank_of_gt(gt: str, ranking: list[str]) -> int | None:
    for i, c in enumerate(ranking):
        if c == gt:
            return i + 1
    return None

# --- 1. CAPE-RCA v5 per-fault metrics ---
cape = {}
for ds in DATASETS:
    path = V5_DIR / f"{ds}_v5.json"
    if not path.exists():
        continue
    data = json.loads(path.read_text())
    flist = faults_for(ds)
    aggs = {ft: {"cases": 0, "hits1": 0, "hits3": 0, "sum_avg5": 0.0} for ft in flist}
    aggs["all"] = {"cases": 0, "hits1": 0, "hits3": 0, "sum_avg5": 0.0}

    for r in data["results"]:
        ft = fault_from_case_id(r["case_id"])
        gt = r.get("expected_component", "")
        pred = r.get("predicted_component")
        ranking = r.get("predicted_ranking", [pred] if pred else [])
        if isinstance(ranking, str):
            try: ranking = json.loads(ranking)
            except Exception: ranking = [pred] if pred else []
        if not isinstance(ranking, list):
            ranking = [pred] if pred else []

        hit1 = bool(r.get("hit"))
        rank = rank_of_gt(gt, ranking)
        hit3 = rank is not None and rank <= 3
        avg5 = 1.0 / max(rank, 1) if rank else 0.0

        # Always count in "all" aggregate
        aggs["all"]["cases"] += 1
        aggs["all"]["hits1"] += int(hit1)
        aggs["all"]["hits3"] += int(hit3)
        aggs["all"]["sum_avg5"] += avg5

        # Count in per-fault aggregate if fault type is recognized
        if ft in flist:
            aggs[ft]["cases"] += 1
            aggs[ft]["hits1"] += int(hit1)
            aggs[ft]["hits3"] += int(hit3)
            aggs[ft]["sum_avg5"] += avg5

    for ft, rec in aggs.items():
        n = rec["cases"]
        rec["ac1"] = rec["hits1"] / n if n else 0
        rec["ac3"] = rec["hits3"] / n if n else 0
        rec["avg5"] = rec["sum_avg5"] / n if n else 0
    cape[ds] = aggs

# --- 2. Baseline metrics ---
baseline = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {"cases": 0, "hits1": 0, "hits3": 0, "sum_avg5": 0.0})))
with open(BASELINE_CSV, newline="") as f:
    for row in csv.DictReader(f):
        method = row["method"]
        ds = row["subset"]
        ft = row["expected_fault"].lower()
        flist = faults_for(ds)
        if ft not in flist:
            continue
        hit1 = int(row.get("hit_at_1", 0))
        hit3 = int(row.get("hit_at_3", 0))
        avg5 = float(row.get("case_avg_at_5", 0.0))
        baseline[ds][method][ft]["cases"] += 1
        baseline[ds][method][ft]["hits1"] += hit1
        baseline[ds][method][ft]["hits3"] += hit3
        baseline[ds][method][ft]["sum_avg5"] += avg5

# Compute baseline aggregates
baseline_aggs = {}
for ds in baseline:
    baseline_aggs[ds] = {}
    flist = faults_for(ds)
    for method in baseline[ds]:
        raw = baseline[ds][method]
        aggs = {}
        aggs["all"] = {"cases": 0, "hits1": 0, "hits3": 0, "sum_avg5": 0.0}
        for ft in flist:
            rec = raw.get(ft, {})
            n = rec.get("cases", 0)
            ac1 = rec["hits1"] / n if n else 0
            ac3 = rec["hits3"] / n if n else 0
            avg5 = rec["sum_avg5"] / n if n else 0
            aggs[ft] = {"cases": n, "ac1": ac1, "ac3": ac3, "avg5": avg5}
            aggs["all"]["cases"] += n
            aggs["all"]["hits1"] += rec.get("hits1", 0)
            aggs["all"]["hits3"] += rec.get("hits3", 0)
            aggs["all"]["sum_avg5"] += rec.get("sum_avg5", 0.0)
        t = aggs["all"]
        t["ac1"] = t["hits1"] / t["cases"] if t["cases"] else 0
        t["ac3"] = t["hits3"] / t["cases"] if t["cases"] else 0
        t["avg5"] = t["sum_avg5"] / t["cases"] if t["cases"] else 0
        baseline_aggs[ds][method] = aggs

# --- Published baselines for RE2-TT ---
pub_tt = {}
with open(PUB_BASELINE, newline="") as f:
    for row in csv.DictReader(f):
        if row["dataset"] == "RE2-TT":
            pub_tt[row["method"]] = {
                "ac1": float(row["AC@1"]), "ac3": float(row["AC@3"]), "avg5": float(row["Avg@5"]),
            }

# --- CSV output ---
with open(OUT_CSV, "w", newline="") as f:
    header = ["dataset", "method"]
    for ft in ALL_FAULTS + ["average"]:
        header.append(f"{ft}_AC@1")
    for ft in ALL_FAULTS + ["average"]:
        header.append(f"{ft}_AC@3")
    for ft in ALL_FAULTS + ["average"]:
        header.append(f"{ft}_Avg@5")
    header.append("cases")
    writer = csv.DictWriter(f, fieldnames=header)
    writer.writeheader()

    for ds in DATASETS:
        for method in METHODS_ORDER:
            if method == "cape-rca":
                aggs = cape.get(ds, {})
            elif ds == "RE2-TT" and method in pub_tt:
                aggs = {"all": pub_tt[method]}
            else:
                aggs = baseline_aggs.get(ds, {}).get(method, {})
            if not aggs:
                continue
            row_data = {"dataset": ds, "method": method}
            for ft in faults_for(ds) + ["all"]:
                key = ft if ft != "all" else "average"
                rec = aggs.get("all" if ft == "all" else ft, {})
                row_data[f"{key}_AC@1"] = f"{rec.get('ac1', 0):.4f}"
                row_data[f"{key}_AC@3"] = f"{rec.get('ac3', 0):.4f}"
                row_data[f"{key}_Avg@5"] = f"{rec.get('avg5', 0):.4f}"
            row_data["cases"] = aggs.get("all", {}).get("cases", 0)
            writer.writerow(row_data)
print(f"CSV: {OUT_CSV}")

# --- Markdown report ---
lines = ["# CAPE-RCA vs Multi-source Baselines: Per-Fault-Type Comparison", ""]

for section in [
    ("RE2 Online Boutique", "RE2-OB", RE2_FAULTS, ["CPU", "MEM", "DISK", "SOCKET", "DELAY", "LOSS"]),
    ("RE2 Sock Shop", "RE2-SS", RE2_FAULTS, ["CPU", "MEM", "DISK", "SOCKET", "DELAY", "LOSS"]),
]:
    title, ds, flist, flabels = section
    lines.append(f"## {title}")
    lines.append("")
    lines.append("| Method | Metric | " + " | ".join(flabels) + " | AVERAGE |")
    lines.append("|---:|---:" + "|---:" * (len(flist) + 1) + "|")
    for method in METHODS_ORDER:
        aggs = cape.get(ds, {}) if method == "cape-rca" else baseline_aggs.get(ds, {}).get(method, {})
        if not aggs:
            continue
        for metric, mkey in [("AC@1", "ac1"), ("AC@3", "ac3"), ("Avg@5", "avg5")]:
            vals = [f"{aggs.get(ft, {}).get(mkey, 0):.3f}" for ft in flist]
            overall = f"{aggs.get('all', {}).get(mkey, 0):.3f}"
            lines.append(f"| {method} | {metric} | {' | '.join(vals)} | {overall} |")
        lines.append("")

# RE2-TT (published baselines only)
lines.append("## RE2 Train Ticket")
lines.append("")
lines.append("| Method | AC@1 | AC@3 | Avg@5 |")
lines.append("|---|---:|---:|---:|")
cape_tt = cape.get("RE2-TT", {}).get("all", {})
lines.append(f"| CAPE-RCA v5 | {cape_tt.get('ac1', 0):.3f} | {cape_tt.get('ac3', 0):.3f} | {cape_tt.get('avg5', 0):.3f} |")
for m, v in pub_tt.items():
    lines.append(f"| {m} (published) | {v['ac1']:.3f} | {v['ac3']:.3f} | {v['avg5']:.3f} |")

# RE3 sections
for section in [
    ("RE3 Online Boutique", "RE3-OB"),
    ("RE3 Sock Shop", "RE3-SS"),
    ("RE3 Train Ticket", "RE3-TT"),
]:
    title, ds = section
    lines.append(f"## {title}")
    flist = RE3_FAULTS
    flabels = [f.upper() for f in flist]
    lines.append("")
    lines.append("| Method | Metric | " + " | ".join(flabels) + " | AVERAGE |")
    lines.append("|---:|---:" + "|---:" * (len(flist) + 1) + "|")
    for method in METHODS_ORDER:
        aggs = cape.get(ds, {}) if method == "cape-rca" else baseline_aggs.get(ds, {}).get(method, {})
        if not aggs:
            continue
        for metric, mkey in [("AC@1", "ac1"), ("AC@3", "ac3"), ("Avg@5", "avg5")]:
            vals = [f"{aggs.get(ft, {}).get(mkey, 0):.3f}" for ft in flist]
            overall = f"{aggs.get('all', {}).get(mkey, 0):.3f}"
            lines.append(f"| {method} | {metric} | {' | '.join(vals)} | {overall} |")
        lines.append("")

REPORT_FILE.write_text("\n".join(lines) + "\n")
print(f"Report: {REPORT_FILE}")
