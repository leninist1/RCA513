"""Counterfactual RCA engine adapted from Phase 1 for OpenRCA unified telemetry."""

import copy
import threading
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from ..config import (
    BASELINE_WINDOW_SECONDS, TOP_K, ANOMALY_Z_THRESHOLD,
    TIME_TOLERANCE_SECONDS, DEFAULT_METRIC_WEIGHTS, LOG_KEYWORD_TIERS, Candidate,
)
from ..data.schema import SYSTEM_SCHEMAS
from .graph import build_graph_from_traces, infer_graph_from_metrics, expand_candidates_upstream


class CounterfactualEngine:
    _analysis_cache: Dict[str, Any] = {}
    _cache_lock = threading.Lock()

    def __init__(self, system_name: str = "Bank"):
        self.system_name = system_name
        self.schema = SYSTEM_SCHEMAS.get(system_name, {})
        self.baseline_window = BASELINE_WINDOW_SECONDS
        self.top_k = TOP_K
        self.z_threshold = ANOMALY_Z_THRESHOLD
        self.time_tolerance = TIME_TOLERANCE_SECONDS
        self._templates = None  # Discovered from data at runtime

    @property
    def metric_templates(self) -> List[Dict]:
        """Safe accessor - returns current templates or generic default."""
        if self._templates is not None:
            return self._templates
        return self._discover_metric_templates(None)

    def _discover_metric_templates(self, df: pd.DataFrame = None) -> List[Dict]:
        """Auto-discover metric templates, filtering to relevant signal metrics."""
        templates = []
        seen = set()

        for key, cfg in self.schema.items():
            if key.startswith("metric_") and "value_cols" in cfg:
                for vcol in cfg["value_cols"]:
                    if vcol not in seen:
                        seen.add(vcol)
                        templates.append({"name": vcol, "weight": DEFAULT_METRIC_WEIGHTS.get(vcol, 1.0)})

        if df is not None and not df.empty and "metric_name" in df.columns:
            for mn in df["metric_name"].dropna().unique():
                mn_str = str(mn)
                if mn_str in seen:
                    continue
                # Only include metrics likely to carry RCA signal
                if self._is_relevant_metric(mn_str):
                    seen.add(mn_str)
                    templates.append({"name": mn_str, "weight": 1.0})

        if not templates:
            templates.append({"name": "value", "weight": 1.0})
        return templates

    @staticmethod
    def _is_relevant_metric(metric_name: str) -> bool:
        """Tight signal metric filter: CPU, memory, disk, network, request, error, DB, app.

        Excludes config/uptime/filesystem metrics that don't carry fault signal.
        """
        name_lower = metric_name.lower()
        exclude = [
            "uptime", "hostname", "check-hostname", "check-defaultroute",
            "filesystem", "fscapacity", "fsinode", "fsusedspace", "fsavailablespace",
            "dirsize", "filesize", "file_",
            "swap", "swptot", "zabbix",
            "open_files", "open_tables", "opened_table", "table_open_cache",
            "binlog", "created_tmp", "sort_merge",
            "bytes_sent", "bytes_received", "slow_launch",
            "slave_open", "tc_log",
            "aborted", "maxconnections", "max_used", "max_trx",
            "longesttrx", "maxTrxRows",
            "getresponsetime", "getconnectedstate", "currentsqlmax",
            "qcache", "questions", "rows_read",
            "handler_write", "handler_update", "handler_savepoint",
            "handler_rollback", "handler_read_rnd", "handler_read_prev",
            "handler_read_next", "handler_delete", "handler_commit",
            "handler_read_key", "handler_read_first",
            "key_writes", "key_write", "key_reads", "key_read",
            "table_locks_immediate", "table_open_cache_misses", "table_open_cache_hits",
            "table_open_cache_overflows",
            "innodb_pages_created", "innodb_pages_read", "innodb_pages_written",
            "innodb_data_pending_writes", "innodb_data_pending_reads", "innodb_data_pending_fsyncs",
            "innodb_open_files", "innodb_dblwr",
            "innodb_row_lock_time_avg", "innodb_row_lock_time_max",
            "innodb_os_log_pending_writes", "innodb_os_log_pending_fsyncs",
            "innodb_log_waits", "innodb_log_writes", "innodb_log_write_requests",
            "mysql_queries", "slave_",
            "process_", "procpp", "procp",
            "localdisk-dskrtps", "localdisk-dskwtps", "localdisk-dskread",
            "localdisk-dskwrite", "localdisk-dskreadwrite", "localdisk-dskavgserv",
            "localdisk-dsktps",
            "com_update", "com_select", "com_replace", "com_load",
            "com_insert", "com_delete",
        ]
        for ex in exclude:
            if ex in name_lower:
                return False

        include = [
            "cpu", "cpuload", "cpuutil", "cpuwio", "singlecpu",
            "memory", "mem", "memfree", "memused", "memperc",
            "jvm", "heap", "gc", "tomcat",
            "disk_io", "diskio",
            "network_", "tcp", "latency", "packet_loss", "packetloss",
            "request", "error", "timeout", "fail",
            "rr", "sr", "mrt", "count",
            "thread", "lock", "connection",
            "load", "busy",
            "dskpercentbusy", "dskbps",
            # DB signal metrics
            "innodb_row_lock_waits", "innodb_row_lock_current_waits",
            "innodb_buffer_pool_wait_free",
            "innodb_buffer_pool_reads", "innodb_buffer_pool_read_requests",
            "innodb_data_reads", "innodb_data_writes", "innodb_data_fsyncs",
            "innodb_data_read", "innodb_data_written",
            "threadsrunning", "threadsconnected", "threads_created",
            "innodb_row_lock_time",
            "innodb_buffer_pool_pages_dirty", "innodb_buffer_pool_pages_flushed",
            "innodb_buffer_pool_pages_free", "innodb_buffer_pool_pages_total",
            "table_locks_waited",
            "innodb_log_fsyncs", "innodb_os_log_fsyncs",
            "innodb_data_pending_fsyncs",
            # App signal
            "errorcount", "requestcount",
        ]
        for inc in include:
            if inc in name_lower:
                return True
        return False

    # ---------- Temporal split ----------

    def split_temporal(self, telemetry, inject_time: float):
        """Split telemetry into baseline and fault windows using time ranges."""
        if telemetry.metrics is not None and not telemetry.metrics.empty:
            metrics = telemetry.metrics
            # Filter by time window, not row count
            baseline_start = inject_time - self.baseline_window
            fault_end = inject_time + self.baseline_window

            baseline_metrics = metrics[
                (metrics["timestamp"] >= baseline_start) & (metrics["timestamp"] < inject_time)
            ]
            fault_metrics = metrics[
                (metrics["timestamp"] >= inject_time) & (metrics["timestamp"] <= fault_end)
            ]
        else:
            baseline_metrics = pd.DataFrame()
            fault_metrics = pd.DataFrame()

        return baseline_metrics, fault_metrics

    # ---------- Anomaly detection ----------

    def detect_anomalies(
        self, baseline_df: pd.DataFrame, fault_df: pd.DataFrame,
        entities: List[str], edge_graph: Optional[Dict] = None
    ) -> List[Candidate]:
        """Z-score anomaly detection across all entities."""
        if fault_df.empty or baseline_df.empty:
            return []

        scores = []
        for entity in entities:
            score = self._entity_anomaly_score(baseline_df, fault_df, entity)

            if edge_graph and entity in edge_graph:
                out_weight = sum(edge_graph.get(entity, {}).values())
                score = max(0.1, score - out_weight * 0.5)

            if score > 0:
                scores.append(Candidate(entity=entity, anomaly_score=score))

        scores.sort(key=lambda c: c.anomaly_score, reverse=True)
        return scores[:self.top_k]

    def _entity_anomaly_score(self, baseline_df: pd.DataFrame, fault_df: pd.DataFrame, entity: str) -> float:
        """Compute anomaly score for one entity, averaged across metric dimensions."""
        entity_data_b = baseline_df[baseline_df["entity"] == entity]
        entity_data_f = fault_df[fault_df["entity"] == entity]

        if entity_data_b.empty or entity_data_f.empty:
            return 0.0

        scores = []
        for template in self.metric_templates:
            metric_name = template["name"]
            weight = template["weight"]

            b_vals = entity_data_b[entity_data_b["metric_name"] == metric_name]["value"]
            f_vals = entity_data_f[entity_data_f["metric_name"] == metric_name]["value"]

            if b_vals.empty or f_vals.empty:
                continue

            b_vals = b_vals.dropna().values
            f_vals = f_vals.dropna().values
            if len(b_vals) == 0 or len(f_vals) == 0:
                continue

            b_med = np.median(b_vals)
            mad = np.median(np.abs(b_vals - b_med)) + 1e-9
            f_peak = np.percentile(f_vals, 95)
            z = max(0, f_peak - b_med) / mad
            if z > self.z_threshold:
                # Cap individual metric contribution via sigmoid
                capped = 1.0 / (1.0 + np.exp(-z / 10.0)) * weight
                scores.append(capped)

        if not scores:
            return 0.0
        # Average across metrics, then scale by count (more anomalous metrics = higher score)
        return np.mean(scores) * min(len(scores), 20)

    # ---------- Counterfactual ----------

    def apply_counterfactual(
        self, fault_df: pd.DataFrame, baseline_df: pd.DataFrame,
        target_entity: str, graph: Optional[Dict] = None
    ) -> pd.DataFrame:
        """Restore target entity (and optionally downstream) to baseline values."""
        cf_df = fault_df.copy()

        entities_to_restore = {target_entity}
        if graph and target_entity in graph:
            for child in graph[target_entity]:
                entities_to_restore.add(child)

        for entity in entities_to_restore:
            entity_mask_f = cf_df["entity"] == entity
            entity_mask_b = baseline_df["entity"] == entity

            if not entity_mask_f.any() or not entity_mask_b.any():
                continue

            for template in self.metric_templates:
                metric_name = template["name"]
                b_vals = baseline_df.loc[
                    entity_mask_b & (baseline_df["metric_name"] == metric_name), "value"
                ]
                if b_vals.empty:
                    continue
                median_val = np.median(b_vals.dropna().values)
                cf_df.loc[
                    entity_mask_f & (cf_df["metric_name"] == metric_name), "value"
                ] = median_val

        return cf_df

    # ---------- Recovery score ----------

    def compute_recovery(
        self, orig_df: pd.DataFrame, cf_df: pd.DataFrame,
        baseline_df: pd.DataFrame, target_entity: str,
        graph: Optional[Dict] = None, entities: Optional[List[str]] = None,
        logs_df: Optional[pd.DataFrame] = None,
        top_metrics: Optional[List[str]] = None,
    ) -> float:
        """Compute system-wide recovery: measure ALL entities' degradation before vs after.

        Phase 1 insight: when true root cause is restored, downstream improves too.
        When non-root-cause is restored, most system degradation remains.
        """
        check_entities = entities if entities else [target_entity]

        orig_total = 0.0
        cf_total = 0.0
        for ent in check_entities:
            if top_metrics:
                orig_total += self._entity_degradation(orig_df, baseline_df, ent, logs_df, top_metrics)
                cf_total += self._entity_degradation(cf_df, baseline_df, ent, logs_df, top_metrics)
            else:
                orig_total += self._entity_degradation(orig_df, baseline_df, ent, logs_df)
                cf_total += self._entity_degradation(cf_df, baseline_df, ent, logs_df)

        if orig_total < 1e-9:
            return 0.0
        improvement = (orig_total - cf_total) / orig_total
        return float(max(0.0, min(1.0, improvement)))

    def _entity_degradation(
        self, df: pd.DataFrame, baseline_df: pd.DataFrame,
        entity: str, logs_df: Optional[pd.DataFrame] = None,
        metric_filter: Optional[List[str]] = None,
    ) -> float:
        """Compute degradation for one entity (delegates to batch version)."""
        return self._batch_entity_degradation(
            df, baseline_df, [entity], logs_df, metric_filter,
        ).get(entity, 0.0)

    def _batch_entity_degradation(
        self, df: pd.DataFrame, baseline_df: pd.DataFrame,
        entities: List[str], logs_df: Optional[pd.DataFrame] = None,
        metric_filter: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """Batch-compute degradation for multiple entities in one vectorised pass.

        Uses pandas groupby aggregations for median/percentile, which are
        orders of magnitude faster than per-entity per-metric loops.
        """
        if df.empty or baseline_df.empty or not entities:
            return {e: 0.0 for e in entities}

        entity_set = set(entities)
        template_names = [t["name"] for t in self.metric_templates
                          if not metric_filter or t["name"] in metric_filter]
        weights = {t["name"]: t["weight"] for t in self.metric_templates}

        # Filter to relevant entities and metrics
        f_sub = df[df["entity"].isin(entity_set) & df["metric_name"].isin(template_names)]
        b_sub = baseline_df[baseline_df["entity"].isin(entity_set) & baseline_df["metric_name"].isin(template_names)]

        if f_sub.empty or b_sub.empty:
            return {e: 0.0 for e in entities}

        # Baseline: median per (entity, metric)
        b_medians = b_sub.groupby(["entity", "metric_name"])["value"].median()

        # MAD: median of absolute deviations, computed via transform
        b_sub_copy = b_sub.copy()
        b_sub_copy["b_median"] = b_sub_copy.groupby(["entity", "metric_name"])["value"].transform("median")
        b_sub_copy["deviation"] = (b_sub_copy["value"] - b_sub_copy["b_median"]).abs()
        b_mad = b_sub_copy.groupby(["entity", "metric_name"])["deviation"].median() + 1e-9

        # Fault: 95th percentile per (entity, metric)
        f_peak = f_sub.groupby(["entity", "metric_name"])["value"].quantile(0.95)

        # Compute degradation per (entity, metric)
        deg_map: Dict[str, float] = {e: 0.0 for e in entities}
        for (entity, metric), f_p in f_peak.items():
            if entity not in entity_set:
                continue
            b_med = b_medians.get((entity, metric))
            mad_val = b_mad.get((entity, metric))
            if b_med is None or mad_val is None or mad_val <= 0:
                continue
            w = weights.get(metric, 1.0)
            deg_map[entity] += max(0.0, float(f_p - b_med)) / float(mad_val) * w

        # Log contribution
        if logs_df is not None and not logs_df.empty:
            for entity in entities:
                log_score = self._log_semantic_score(logs_df, entity)
                if log_score > 0:
                    deg_map[entity] += log_score * 100

        return deg_map

    def _log_semantic_score(self, logs_df: pd.DataFrame, entity: str) -> float:
        """Log keyword tier scoring for an entity."""
        entity_logs = logs_df[logs_df["entity"] == entity]
        if entity_logs.empty:
            return 0.0
        score = 0.0
        messages = entity_logs["message"].astype(str)
        for keyword, weight in LOG_KEYWORD_TIERS.items():
            if messages.str.contains(keyword, case=False).any():
                score += weight
        return score

    # ---------- Orchestration ----------

    def run_initial_analysis(
        self, telemetry, inject_time: float, entities: Optional[List[str]] = None
    ) -> Tuple[List[Candidate], Dict, pd.DataFrame, pd.DataFrame]:
        """Run full initial analysis with caching by (system, date, inject_time)."""
        # Build cache key (thread-safe)
        cache_key = f"{self.system_name}:{inject_time}"
        with self._cache_lock:
            if cache_key in self._analysis_cache:
                cached = self._analysis_cache[cache_key]
                return cached["candidates"], cached["graph"], cached["baseline"], cached["fault"]

        baseline_df, fault_df = self.split_temporal(telemetry, inject_time)

        if entities is None:
            entities = telemetry.entities

        # Lazily discover metric templates from actual data
        all_data = pd.concat([baseline_df, fault_df], ignore_index=True) if not baseline_df.empty else fault_df
        self._templates = self._discover_metric_templates(all_data)

        # Build graph
        edge_graph = {}
        if telemetry.traces is not None and not telemetry.traces.empty:
            edge_graph = build_graph_from_traces(telemetry.traces, inject_time=inject_time, max_traces=3000)
        if not edge_graph:
            edge_graph = infer_graph_from_metrics(baseline_df, fault_df, entities, self.time_tolerance)

        # Detect anomalies on all entities (capped at 50 for broader coverage)
        entities_subset = entities[:50] if len(entities) > 50 else entities
        candidates = self.detect_anomalies(baseline_df, fault_df, entities_subset, edge_graph)

        # Expand with upstream (from full entity list)
        candidate_entities = [c.entity for c in candidates]
        expanded = expand_candidates_upstream(candidate_entities, edge_graph, entities)
        for e in expanded:
            if e not in candidate_entities:
                candidates.append(Candidate(entity=e, anomaly_score=0.1))

        # Cap at TOP_K
        candidates = candidates[:self.top_k]

        # Compute log counts for influence scoring
        log_counts = {}
        if telemetry.logs is not None and not telemetry.logs.empty:
            error_mask = telemetry.logs["message"].astype(str).str.contains(
                "ERROR|Exception|Timeout|Fail|error", case=False, na=False
            )
            log_counts = telemetry.logs[error_mask].groupby("entity").size().to_dict()

        # Adjust scores with graph centrality + log boost
        for c in candidates:
            out_rad = sum(edge_graph.get(c.entity, {}).values())
            in_ctrl = sum(children.get(c.entity, 0) for children in edge_graph.values())
            log_boost = np.log1p(log_counts.get(c.entity, 0)) * 5.0
            c.anomaly_score = c.anomaly_score + 2.0 * in_ctrl - 1.5 * out_rad + log_boost

        # System-wide recovery for ALL candidates.
        # Build health-check entity set: candidate entities + top degraded entities.
        candidate_entities = {c.entity for c in candidates}
        entities_for_deg = entities[:25] if len(entities) > 25 else entities

        # Batch-compute degradation for all entities in one pass (vectorised)
        all_deg_cache = self._batch_entity_degradation(
            fault_df, baseline_df, entities_for_deg, telemetry.logs,
        )

        # Health check: candidate entities + top 12 degraded non-candidate entities
        non_candidate_degs = [(e, d) for e, d in all_deg_cache.items()
                              if e not in candidate_entities]
        non_candidate_degs.sort(key=lambda x: x[1], reverse=True)
        top_non_candidates = [e for e, _ in non_candidate_degs[:12]]

        entities_for_health = list(candidate_entities) + top_non_candidates
        seen = set()
        entities_for_health = [e for e in entities_for_health
                               if not (e in seen or seen.add(e))]

        orig_deg_cache = {ent: all_deg_cache.get(ent, 0.0)
                          for ent in entities_for_health}

        for c in candidates:
            cf_df = self.apply_counterfactual(fault_df, baseline_df, c.entity, edge_graph)
            orig_total = sum(orig_deg_cache.values())
            # Batch-compute counterfactual degradations
            cf_deg_cache = self._batch_entity_degradation(
                cf_df, baseline_df, entities_for_health, telemetry.logs,
            )
            cf_total = sum(cf_deg_cache.values())
            c.recovery_score = float(max(0.0, min(1.0, (orig_total - cf_total) / max(orig_total, 1e-9))))

        # Sort by recovery then influence
        candidates.sort(key=lambda c: (round(c.recovery_score if c.recovery_score is not None else 0.0, 2), c.anomaly_score), reverse=True)

        # Cache results (thread-safe)
        with self._cache_lock:
            if len(self._analysis_cache) >= 5:
                oldest = next(iter(self._analysis_cache))
                del self._analysis_cache[oldest]
            self._analysis_cache[cache_key] = {
                "candidates": candidates, "graph": edge_graph,
                "baseline": baseline_df, "fault": fault_df,
            }

        return candidates, edge_graph, baseline_df, fault_df
