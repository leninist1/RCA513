# Confidence-aware RCA v2 drift-focus 完整源码依赖包

本包用于补发可复现实验所需的源码依赖，重点补齐上一版遗漏的 `refute/` 旧主线依赖。

## 包内目录

- `source/refute/src/`: d32/PoRCA 底座仍依赖的旧主线源码，包括 `baseline_distributions.py`、`data_loader.py`、`node_container_split.py`。
- `source/refute/knowledge/`: d32 默认使用的 baseline distribution 和 node-container graph。
- `source/refute_b_v2/`: Scheme B v2 / PoRCA 共享工具，包括 `query_windows.py`、`trace_propagation.py`。
- `source/refute_b_v2_d32/`: d32 time-vote / trace-summary 底座源码。
- `source/confidence_aware_rca/`: confidence-aware selective prediction 代码。
- `source/eval/`: d32 LODO 与 knowledge build 入口脚本。
- `source/tests/`: 当前单元测试。
- `source/knowledge/`: d32/refutation rules，以及预构建的 `d32_lodo`、`d32_lodo_shape_reasonprior`、`d32_lodo_shape_reasonprior_innerlodo` knowledge 子目录。
- `source/trace_summaries/openrca_queries_full_batch18/`: 当前 trace-summary 路线的默认派生输入。
- `docs/`: 交接文档。

## 未包含内容

本包不包含原始 Bank telemetry 数据。远端默认原始数据路径为：

```text
/home/yan/workspace/data/openrca/Bank
```

该目录约 20G，包含 `query.csv`、`record.csv` 和 `telemetry/` 下各日期的 log/metric/trace 原始文件。

## 运行提示

在包的 `source/` 目录作为工作目录运行时，建议设置：

```bash
export PYTHONPATH=.:..
```

d32 LODO 默认入口：

```bash
python3 eval/run_openrca_d32_lodo.py \
  --data-root /home/yan/workspace/data/openrca/Bank \
  --baseline refute/knowledge/baseline_distributions_all_metric_dates.json \
  --node-graph refute/knowledge/node_container_graph_2021_03_10.json \
  --rules knowledge/refutation_rules_v2.json \
  --trace-summary-dir trace_summaries/openrca_queries_full_batch18
```

如果在原远端项目目录 `/home/yan/workspace/projects/rca-bank-adaptive/refute_b_v2` 中运行，也可以继续使用脚本原有默认相对路径。
