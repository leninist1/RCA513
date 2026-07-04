#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/dell2/RCA513/yyx/trace_summary/.venv_d32/bin/python}"

PYTHONPATH="$(dirname "$REPO_ROOT")${PYTHONPATH:+:$PYTHONPATH}" \
  "$PYTHON_BIN" "$SCRIPT_DIR/select_second_dataset.py"

echo "No second dataset is admitted for main experiments. See paper_artifacts/reports/second_dataset_selection_report.md"
exit 0
