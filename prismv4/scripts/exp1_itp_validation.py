#!/usr/bin/env python3
"""Experiment 1: ITP validation — minimal onset comparison.
Skips trace dependency (too slow per-case for 174MB traces.csv).
Only checks: is expected_component.onset <= predicted_component.onset?
And: does expected have a valid onset?
"""
import json, os, sys
from pathlib import Path
from collections import defaultdict
import pandas as pd, numpy as np

sys.path.insert(0, "/home/dell2/RCA513/yyx")

RE2_DATA = "/home/dell2/RCA513/ysj/dataset/RCAEval/RE2/RE2-TT"
RESULT = "/home/dell2/RCA513/yyx/prismv4/results/prism_cht/RE2-TT_lw_v2.json"
BASELINE_SEC = 300

print("Loading data...", flush=True)
v2 = json.load(open(RESULT))
fails = [c for c in v2["results"] if not c.get("hit")]
hits  = [c for c in v2["results"] if c.get("hit")]
print(f"Hits={len(hits)} Fails={len(fails)}", flush=True)

def find_onset(df, component, inject, baseline_sec=300):
    """Earliest time >= inject where component metric deviates >3sigma."""
    cols = [c for c in df.columns if c.startswith(component + "_")]
    if not cols: return None
    pre = df[df["time"] < inject].tail(baseline_sec)
    post = df[df["time"] >= inject]
    if pre.empty or post.empty: return None
    best = None
    for col in cols:
        mu, sd = pre[col].mean(), pre[col].std()
        if pd.isna(sd) or sd == 0: continue
        upper = mu + 3*sd; lower = mu - 3*sd
        anom = post[(post[col] > upper) | (post[col] < lower)]
        if not anom.empty:
            t = float(anom["time"].iloc[0])
            if best is None or t < best: best = t
    return best

# Process: cache each unique case dir to avoid re-reading
case_cache = {}
results = []
all_cases = fails + hits[:10]  # include 10 hit cases for baseline comparison

for idx, fc in enumerate(all_cases):
    if idx % 10 == 0:
        print(f"  [{idx}/{len(all_cases)}]", flush=True)
    case_id = fc["case_id"]
    parts = case_id.split("/")
    if len(parts) != 3: continue
    cd = Path(RE2_DATA) / parts[1] / parts[2]
    if not cd.exists(): continue

    # Cache
    cache_key = str(cd)
    if cache_key not in case_cache:
        # Prefer simple_metrics for speed
        for fname in ["simple_metrics.csv", "metrics.csv"]:
            fp = cd / fname
            if fp.exists(): break
        df = pd.read_csv(fp).replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
        df["time"] = pd.to_numeric(df["time"], errors="coerce")
        inject = float((cd / "inject_time.txt").read_text().strip())
        # Pre-compute onsets for all components in one pass
        comp_cols = defaultdict(list)
        for c in df.columns:
            if c == "time": continue
            comp = c.split("_", 1)[0]
            comp_cols[comp].append(c)
        pre = df[df["time"] < inject].tail(BASELINE_SEC)
        post = df[df["time"] >= inject]
        comp_onset = {}
        for comp, cols in comp_cols.items():
            if len(cols) > 50: cols = cols[:50]  # cap
            best = None
            for col in cols:
                if col not in pre.columns or col not in post.columns: continue
                mu, sd = pre[col].mean(), pre[col].std()
                if pd.isna(sd) or sd == 0: continue
                upper = mu + 3*sd; lower = mu - 3*sd
                anom = post[(post[col] > upper) | (post[col] < lower)]
                if not anom.empty:
                    t = float(anom["time"].iloc[0])
                    if best is None or t < best: best = t
            comp_onset[comp] = best
        case_cache[cache_key] = comp_onset

    onset_map = case_cache[cache_key]
    expected = fc["expected_component"]
    predicted = fc.get("predicted_component", "")
    eo = onset_map.get(expected)
    po = onset_map.get(predicted) if predicted else None
    is_sc = "shortcut" in fc.get("status", "")
    is_hit = fc.get("hit", False)

    results.append({
        "id": parts[1] + "/" + parts[2],
        "exp": expected, "pred": predicted,
        "eo": eo, "po": po,
        "itp_first": (eo is not None and po is not None and eo <= po),
        "sc": is_sc, "hit": is_hit,
    })

# Summary
fails_res = [r for r in results if not r["hit"]]
hits_res = [r for r in results if r["hit"]]

print(f"\n{'='*60}")
print(f"ITP Experiment: onset ordering analysis")
print(f"{'='*60}")

for label, subset in [("Fails", fails_res), ("Hits (10)", hits_res)]:
    n = len(subset)
    n_eo = sum(1 for r in subset if r["eo"] is not None)
    n_po = sum(1 for r in subset if r["po"] is not None)
    n_both = sum(1 for r in subset if r["eo"] is not None and r["po"] is not None)
    n_first = sum(1 for r in subset if r["itp_first"])
    print(f"\n{label} (n={n}):")
    print(f"  has expected_onset: {n_eo}/{n} = {100*n_eo/max(n,1):.0f}%")
    print(f"  has predicted_onset: {n_po}/{n} = {100*n_po/max(n,1):.0f}%")
    print(f"  both have onset: {n_both}")
    print(f"  exp.onset <= pred.onset: {n_first}/{n_both} = {100*n_first/max(n_both,1):.0f}%")

# Show a random sample from fails where eo exists
fails_with_eo = [r for r in fails_res if r["eo"] is not None]
print(f"\n=== Sample fails with onset ({len(fails_with_eo)} cases) ===")
for r in fails_with_eo[:10]:
    if r["eo"] and r["po"]: dt = f"Delta={r['po']-r['eo']:.0f}s"
    else: dt = "no pred onset"
    p_str = f"{r['po']:.0f}s" if r["po"] is not None else "na"
    print(f"  {r['id'][:40]}: exp={r['exp']}({r['eo']:.0f}s after inj) pred={r['pred']}({p_str}) {dt} sc={r['sc']}")

print(f"\nDone!")
