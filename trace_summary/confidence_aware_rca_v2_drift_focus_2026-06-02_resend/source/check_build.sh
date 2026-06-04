set -euo pipefail

YSJ=/home/dell2/RCA513/ysj
PKG=$YSJ/trace_summary/confidence_aware_rca_v2_drift_focus_2026-06-02_resend
BASE=$PKG/source
MARKET_ROOT=$YSJ/dataset/openrca/Market
SUBSETS=("cloudbed-1" "cloudbed-2")

cd "$BASE"

for subset in "${SUBSETS[@]}"; do
  safe=${subset//-/_}
  DATA_ROOT=$MARKET_ROOT/$subset
  TRACE_DIR="trace_summaries/market_${safe}_queries_full"
  BASELINE="knowledge/market_${safe}_baseline_distributions_all_metric_dates.json"
  GRAPH="knowledge/market_${safe}_node_container_graph.json"

  echo "=== check ${subset} ==="

  test -f "$DATA_ROOT/query.csv"
  test -f "$DATA_ROOT/record.csv"
  test -f "$TRACE_DIR/manifest.json"
  test -f "$BASELINE"
  test -f "$GRAPH"

  python3 - "$DATA_ROOT" "$TRACE_DIR" "$BASELINE" "$GRAPH" <<'PY'
import json
import sys
from pathlib import Path
from collections import Counter

import pandas as pd

data_root = Path(sys.argv[1])
trace_dir = Path(sys.argv[2])
baseline_path = Path(sys.argv[3])
graph_path = Path(sys.argv[4])

query_n = len(pd.read_csv(data_root / "query.csv"))
record = pd.read_csv(data_root / "record.csv")

manifest = json.loads((trace_dir / "manifest.json").read_text(encoding="utf-8"))
rows = manifest.get("rows", [])
trace_status = Counter(row.get("trace_status") for row in rows)

baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
stats = baseline.get("stats", [])
eligible = sum(1 for row in stats if row.get("eligible"))

graph = json.loads(graph_path.read_text(encoding="utf-8"))
containers = graph.get("containers", {})
nodes = graph.get("nodes", {})

missing_trace_files = [
    row["row_id"] for row in rows
    if not (trace_dir / f"query_{int(row['row_id']):03d}.json").exists()
]
prefixed_components = [
    row.get("cmdb_id", "") for row in stats[:5000]
    if str(row.get("cmdb_id", "")).startswith("node-")
    and "." in str(row.get("cmdb_id", ""))
]

record_components = {str(x).strip() for x in record["component"].dropna()}
graph_components = set(containers)
missing_record_components = sorted(record_components - graph_components)

print(f"query_n={query_n}")
print(f"trace_manifest_n={manifest.get('n')} rows={len(rows)} status={dict(trace_status)}")
print(f"trace_missing_json_files={len(missing_trace_files)}")
print(f"baseline_stats={len(stats)} eligible={eligible}")
print(f"graph_containers={len(containers)} nodes={len(nodes)}")
print(f"baseline_prefixed_node_dot_examples={prefixed_components[:5]}")
print(f"record_components_missing_in_graph={missing_record_components[:20]} count={len(missing_record_components)}")

assert len(rows) == query_n, "trace manifest rows != query rows"
assert manifest.get("n") == query_n, "trace manifest n != query rows"
assert not missing_trace_files, "some query_XXX trace summary files are missing"
assert len(stats) > 0, "baseline has no stats"
assert eligible > 0, "baseline has no eligible stats"
assert len(containers) > 0 and len(nodes) > 0, "node graph is empty"
assert not prefixed_components, "baseline cmdb_id still contains node-X.service prefix"
print("OK")
PY

done