#!/usr/bin/env bash
set -euo pipefail

BASE=/home/dell2/RCA513/ysj/trace_summary/confidence_aware_rca_v2_drift_focus_2026-06-02_resend/source
MARKET_ROOT=/home/dell2/RCA513/ysj/dataset/openrca/Market
SUBSETS=("cloudbed-1" "cloudbed-2")

cd "$BASE"
export PYTHONPATH=".:..:${PYTHONPATH:-}"

mkdir -p knowledge logs

for subset in "${SUBSETS[@]}"; do
  DATA_ROOT=$MARKET_ROOT/$subset
  safe=${subset//-/_}

  echo "=== build baseline + node graph ${subset} ==="

  python3 - "$DATA_ROOT" "$safe" <<'PY' 2>&1 | tee "logs/build_market_knowledge_${safe}.log"
import json
import sys
from pathlib import Path

import pandas as pd

from refute.src.baseline_distributions import build_baseline_from_metric_csvs


root = Path(sys.argv[1])
safe = sys.argv[2]

required_metric_cols = {"timestamp", "cmdb_id", "kpi_name", "value"}
all_metric_paths = sorted((root / "telemetry").glob("*/metric/metric_*.csv"))
metric_paths = []
skipped_paths = []
for path in all_metric_paths:
    header = pd.read_csv(path, nrows=0)
    if required_metric_cols <= set(header.columns):
        metric_paths.append(path)
    else:
        skipped_paths.append((path, list(header.columns)))

if not metric_paths:
    raise SystemExit(f"no compatible metric csvs under {root}")

for path, columns in skipped_paths:
    print(f"skip incompatible metric csv: {path} columns={columns}")

store = build_baseline_from_metric_csvs(
    metric_paths,
    dataset="Market",
    include_node_level=True,
    metadata={"source_root": str(root), "metric_csvs": [str(p) for p in metric_paths]},
)
baseline_path = Path(f"knowledge/market_{safe}_baseline_distributions_all_metric_dates.json")
store.save_json(baseline_path)


def split_cmdb(raw):
    raw = str(raw)
    if raw.startswith("node-") and "." in raw:
        node, service = raw.split(".", 1)
        return node, service
    if raw.startswith("node-"):
        return raw, raw
    return f"node::{raw}", raw


containers = {}
nodes = {}

for path in metric_paths:
    for chunk in pd.read_csv(path, usecols=["cmdb_id", "kpi_name"], chunksize=300000):
        for raw, kpi in chunk.dropna().drop_duplicates().itertuples(index=False, name=None):
            node, service = split_cmdb(raw)
            kpi = str(kpi)

            container = containers.setdefault(
                service,
                {
                    "node_proxy": node,
                    "container_kpis": set(),
                    "node_kpis": set(),
                    "abstract_buckets": {},
                },
            )
            container["container_kpis"].add(kpi)

            node_row = nodes.setdefault(node, {"hosted_containers": set(), "node_kpis": set()})
            node_row["hosted_containers"].add(service)
            if service == node:
                node_row["node_kpis"].add(kpi)

graph = {
    "dataset": "Market",
    "containers": {
        key: {
            **value,
            "container_kpis": sorted(value["container_kpis"]),
            "node_kpis": sorted(value["node_kpis"]),
        }
        for key, value in sorted(containers.items())
    },
    "nodes": {
        key: {
            "hosted_containers": sorted(value["hosted_containers"]),
            "node_kpis": sorted(value["node_kpis"]),
        }
        for key, value in sorted(nodes.items())
    },
}

graph_path = Path(f"knowledge/market_{safe}_node_container_graph.json")
graph_path.write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")

print(f"baseline={baseline_path} stats={len(store)}")
print(f"graph={graph_path} containers={len(containers)} nodes={len(nodes)}")
PY
done

echo "=== generated files ==="
ls -lh \
  knowledge/market_cloudbed_1_baseline_distributions_all_metric_dates.json \
  knowledge/market_cloudbed_1_node_container_graph.json \
  knowledge/market_cloudbed_2_baseline_distributions_all_metric_dates.json \
  knowledge/market_cloudbed_2_node_container_graph.json
