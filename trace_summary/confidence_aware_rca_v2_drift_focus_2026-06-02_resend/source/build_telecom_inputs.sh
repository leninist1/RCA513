#!/usr/bin/env bash
set -euo pipefail

BASE=/home/dell2/RCA513/yyx/trace_summary/confidence_aware_rca_v2_drift_focus_2026-06-02_resend/source
DATA_ROOT=/home/dell2/RCA513/ysj/dataset/openrca/Telecom
QUERY_CSV=$DATA_ROOT/query.csv
RECORD_CSV=$DATA_ROOT/record.csv

TRACE_DIR=trace_summaries/telecom_queries_full
BASELINE=knowledge/telecom_baseline_distributions_all_metric_dates.json
GRAPH=knowledge/telecom_node_container_graph.json

cd "$BASE"
export PYTHONPATH=".:..:${PYTHONPATH:-}"
export TZ=Asia/Shanghai

mkdir -p logs knowledge trace_summaries

test -d "$DATA_ROOT"
test -f "$QUERY_CSV"
test -f "$RECORD_CSV"

if [[ "${RUN_COMPAT_PATCH:-1}" == "1" && -x ./patch_telecom_compat.sh ]]; then
  echo "=== apply Telecom compat patch ==="
  ./patch_telecom_compat.sh
fi

echo "=== Telecom build preflight ==="
python3 - "$DATA_ROOT" <<'PY'
import sys
from pathlib import Path

from refute.src.data_loader import BankDataPaths, load_log_day, load_metric_day
from refute_b_v2.query_windows import parse_query_window
from refute_b_v2_d32.schema import reason_bucket

root = Path(sys.argv[1])
window = parse_query_window(
    "During the specified time range of April 11, 2020, from 00:00 to 00:30, there was one failure reported."
)
assert window.start.year == 2020 and window.start.month == 4
assert reason_bucket("CPU fault") == "cpu"
assert reason_bucket("network delay") == "network_latency"

paths = BankDataPaths.from_root(root)
metric = load_metric_day(paths, "2020_04_11")
log = load_log_day(paths, "2020_04_11")
assert not metric.empty
assert {"timestamp", "cmdb_id", "kpi_name", "value"} <= set(metric.columns)
assert int(metric["timestamp"].max()) < 10_000_000_000
print(f"metric_rows={len(metric)} log_rows={len(log)} preflight_ok")
PY

echo "=== build Telecom trace summary ==="
python3 eval/build_query_trace_summaries.py \
  --data-root "$DATA_ROOT" \
  --query-csv "$QUERY_CSV" \
  --out-dir "$TRACE_DIR" \
  2>&1 | tee logs/build_trace_telecom.log

echo "=== build Telecom baseline + node graph ==="
python3 - "$DATA_ROOT" "$BASELINE" "$GRAPH" <<'PY' 2>&1 | tee logs/build_telecom_knowledge.log
import json
import sys
from pathlib import Path

import pandas as pd

from refute.src.baseline_distributions import build_baseline_from_metric_csvs


root = Path(sys.argv[1])
baseline_path = Path(sys.argv[2])
graph_path = Path(sys.argv[3])

required_base_cols = {"timestamp", "cmdb_id", "value"}
metric_paths = []
skipped = []

for path in sorted((root / "telemetry").glob("*/metric/metric_*.csv")):
    header = pd.read_csv(path, nrows=0)
    cols = set(header.columns)
    if required_base_cols <= cols and ("kpi_name" in cols or "name" in cols):
        metric_paths.append(path)
    else:
        skipped.append((path, sorted(cols)))

if not metric_paths:
    raise SystemExit(f"no compatible Telecom metric csvs under {root}")

for path, cols in skipped:
    print(f"skip incompatible metric csv: {path} columns={cols}")

store = build_baseline_from_metric_csvs(
    metric_paths,
    dataset="Telecom",
    include_node_level=True,
    metadata={"source_root": str(root), "metric_csvs": [str(path) for path in metric_paths]},
)
store.save_json(baseline_path)

components = {}
for path in metric_paths:
    header = pd.read_csv(path, nrows=0)
    if "kpi_name" in header.columns:
        usecols = ["cmdb_id", "kpi_name"]
        rename = {}
    else:
        usecols = ["cmdb_id", "name"]
        rename = {"name": "kpi_name"}

    for chunk in pd.read_csv(path, usecols=usecols, chunksize=300000):
        if rename:
            chunk = chunk.rename(columns=rename)
        chunk = chunk[["cmdb_id", "kpi_name"]]  # force column order (usecols does not guarantee order)
        for row_tup in chunk.dropna().drop_duplicates().itertuples(index=False):
            cmdb_id = str(row_tup.cmdb_id)
            kpi_name = str(row_tup.kpi_name)
            row = components.setdefault(
                cmdb_id,
                {
                    "node_proxy": cmdb_id,
                    "container_kpis": set(),
                    "node_kpis": set(),
                    "abstract_buckets": {},
                },
            )
            row["container_kpis"].add(kpi_name)

nodes = {
    cmdb_id: {
        "hosted_containers": [cmdb_id],
        "node_kpis": [],
    }
    for cmdb_id in sorted(components)
}

graph = {
    "dataset": "Telecom",
    "containers": {
        cmdb_id: {
            **row,
            "container_kpis": sorted(row["container_kpis"]),
            "node_kpis": sorted(row["node_kpis"]),
        }
        for cmdb_id, row in sorted(components.items())
    },
    "nodes": nodes,
}
graph_path.write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")

print(f"baseline={baseline_path} stats={len(store)}")
print(f"graph={graph_path} containers={len(components)} nodes={len(nodes)}")
PY

echo "=== verify Telecom derived inputs ==="
python3 - "$DATA_ROOT" "$TRACE_DIR" "$BASELINE" "$GRAPH" <<'PY'
import json
import sys
from pathlib import Path

import pandas as pd

root = Path(sys.argv[1])
trace_dir = Path(sys.argv[2])
baseline_path = Path(sys.argv[3])
graph_path = Path(sys.argv[4])

query_n = len(pd.read_csv(root / "query.csv"))
manifest = json.loads((trace_dir / "manifest.json").read_text(encoding="utf-8"))
baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
graph = json.loads(graph_path.read_text(encoding="utf-8"))

statuses = {
    status: sum(1 for row in manifest["rows"] if row.get("trace_status") == status)
    for status in sorted({row.get("trace_status") for row in manifest["rows"]})
}
eligible = sum(1 for row in baseline.get("stats", []) if row.get("eligible"))

print(f"query_n={query_n}")
print(f"trace_manifest_n={manifest.get('n')} rows={len(manifest.get('rows', []))} status={statuses}")
print(f"baseline_stats={len(baseline.get('stats', []))} eligible={eligible}")
print(f"graph_containers={len(graph.get('containers', {}))} nodes={len(graph.get('nodes', {}))}")

assert manifest.get("n") == query_n
assert len(manifest.get("rows", [])) == query_n
assert len(baseline.get("stats", [])) > 0
assert eligible > 0
assert len(graph.get("containers", {})) > 0
assert len(graph.get("nodes", {})) > 0
print("Telecom build OK")
PY

echo "=== generated files ==="
ls -lh "$TRACE_DIR/manifest.json" "$BASELINE" "$GRAPH"
