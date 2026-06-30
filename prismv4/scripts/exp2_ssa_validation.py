#!/usr/bin/env python3
"""Experiment 2: SSA validation (inline-safe version)."""
import json, sys, os, numpy as np, pandas as pd
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, "/home/dell2/RCA513/yyx")

SIG_MAP = {
    "cpu": ["cpu", "cpuutil", "cpuload", "singlecpu", "jvm_cpuload", "oslinux_cpu"],
    "memory": ["mem", "memory", "memused", "memperc", "memfree", "heap", "jvm_memory"],
    "disk": ["disk", "diskio", "blkio", "dskread", "dskwrite", "dskbps", "dsktps", "iops"],
    "network": ["net", "tcp", "packet", "retransmit", "socket", "netkbtotalpersec"],
}


def sig_of(kpi):
    low = kpi.lower()
    for s, pats in SIG_MAP.items():
        for p in pats:
            if p in low:
                return s
    return "other"


def sparse_metrics(df, comp, inject):
    try:
        cols = [c for c in df.columns if c.startswith(comp + "_")]
        if not cols:
            return None
        pre = df[df["time"] < inject].tail(300)
        post = df[df["time"] >= inject]
        if pre.empty or post.empty:
            return None
        full_mag = 0.0
        top_per_sig = defaultdict(list)
        tot_per_sig = defaultdict(int)
        anom_per_sig = defaultdict(int)
        for col in cols:
            kpi = col[len(comp) + 1:]
            sig = sig_of(kpi)
            mu = float(pre[col].mean())
            sd = float(pre[col].std())
            tot_per_sig[sig] += 1
            if sd == 0 or np.isnan(sd):
                continue
            u = mu + 3 * sd
            lo = mu - 3 * sd
            anom = post[(post[col] > u) | (post[col] < lo)]
            if not anom.empty:
                dev = float(np.max(np.abs(post[col].values - mu) / max(sd, 0.01)))
                full_mag += dev
                anom_per_sig[sig] += 1
                top_per_sig[sig].append(dev)
        sparse_mag = 0.0
        for sig, vals in top_per_sig.items():
            top3 = sorted(vals, reverse=True)[:3]
            sparse_mag += sum(top3)
        tot_kpi = sum(tot_per_sig.values())
        tot_anom = sum(anom_per_sig.values())
        den = tot_anom / max(tot_kpi, 1)
        return {
            "full": round(full_mag, 1),
            "sparse": round(sparse_mag, 1),
            "density": round(den, 4),
            "n_kpi": tot_kpi,
            "n_anom": tot_anom,
        }
    except Exception:
        return None


from prismv4.experiments.rcaeval_adapter import discover_aiops2021_cases, load_aiops2021_case

print("Loading...", flush=True)
cases = discover_aiops2021_cases(split="test")
v1 = json.load(open("prismv4/results/prism_cht/AIOps2021-test_lw_v1.json"))
results = []
for idx, cs in enumerate(cases):
    if (idx + 1) % 10 == 0:
        print("  [%d/%d]" % (idx + 1, len(cases)), flush=True)
    gt_id = cs["groundtruth_id"]
    expected = cs["expected_component"]
    ivd_c = next((c for c in v1["results"] if str(gt_id) in c.get("case_id", "")), None)
    pred = ivd_c.get("predicted_component") if ivd_c else None
    try:
        loaded = load_aiops2021_case(cs, top_k=5, window_pre_min=30)
    except Exception:
        continue
    df = loaded.store.metrics
    et = loaded.store.event_time
    all_c = list(loaded.store.components)
    scored = []
    for comp in all_c:
        m = sparse_metrics(df, comp, et)
        if m:
            scored.append((comp, m))
    if not scored:
        continue
    by_f = sorted(scored, key=lambda x: -x[1]["full"])
    by_s = sorted(scored, key=lambda x: -x[1]["sparse"])

    def rank_of(c, lst):
        for i, (cc, _) in enumerate(lst):
            if cc == c:
                return i
        return len(lst)

    exp_data = next((m for c, m in scored if c == expected), {})
    results.append({
        "id": gt_id, "exp": expected,
        "exp_fr": rank_of(expected, by_f),
        "exp_sr": rank_of(expected, by_s),
        "exp_fm": exp_data.get("full", 0),
        "exp_sm": exp_data.get("sparse", 0),
        "exp_dn": exp_data.get("density", 0),
        "exp_nk": exp_data.get("n_kpi", 0),
        "exp_na": exp_data.get("n_anom", 0),
    })

n = len(results)
n_better = sum(1 for r in results if r["exp_sr"] < r["exp_fr"])
n_worse = sum(1 for r in results if r["exp_sr"] > r["exp_fr"])
print("\nSSA Report (n=%d cases):" % n)
print("  Sparse better rank: %d (%d%%)" % (n_better, 100 * n_better // n))
print("  Sparse worse rank: %d (%d%%)" % (n_worse, 100 * n_worse // n))
print("  Full TOP1: %d/%d=%d%%" % (sum(1 for r in results if r["exp_fr"] == 0), n, 100 * sum(1 for r in results if r["exp_fr"] == 0) // n))
print("  Sparse TOP1: %d/%d=%d%%" % (sum(1 for r in results if r["exp_sr"] == 0), n, 100 * sum(1 for r in results if r["exp_sr"] == 0) // n))
print("  Full TOP3: %d/%d=%d%%" % (sum(1 for r in results if r["exp_fr"] < 3), n, 100 * sum(1 for r in results if r["exp_fr"] < 3) // n))
print("  Sparse TOP3: %d/%d=%d%%" % (sum(1 for r in results if r["exp_sr"] < 3), n, 100 * sum(1 for r in results if r["exp_sr"] < 3) // n))
avg_kpi = sum(r["exp_nk"] for r in results) / max(n, 1)
avg_anom = sum(r["exp_na"] for r in results) / max(n, 1)
print("  Avg KPIs/comp: %.0f  Avg anomalous KPIs: %.1f  Avg density: %.3f" % (avg_kpi, avg_anom, avg_anom / max(avg_kpi, 1)))
print("\nSample:")
for r in results[:7]:
    print("  %s: exp=%s fm=%.1f(r%d) sm=%.1f(r%d) nk=%d na=%d den=%.3f" % (r["id"], r["exp"], r["exp_fm"], r["exp_fr"], r["exp_sm"], r["exp_sr"], r["exp_nk"], r["exp_na"], r["exp_dn"]))
print("Done!")
