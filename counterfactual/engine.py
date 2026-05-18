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
        """Filter to metrics likely to carry fault signal.

        Excludes uptime, hostname checks, config constants, etc.
        Includes CPU, memory, disk, network, request, error, latency metrics.
        """
        name_lower = metric_name.lower()
        # Exclude non-signal metrics
        exclude = [
            "uptime", "hostname", "host_uptime", "check-hostname",
            "filesystem", "fscapacity", "fsinode", "fsavailable",
            "dirsize", "filesize",
        ]
        for ex in exclude:
            if ex in name_lower:
                return False
        # Include signal metrics
        include = [
            "cpu", "mem", "memory", "disk", "io", "network",
            "request", "error", "latency", "timeout", "fail",
            "rr", "sr", "cnt", "mrt", "count", "rate",
            "swap", "tcp", "connection", "thread", "lock",
            "innodb", "handler", "qcache", "query",
            "jvm", "gc", "heap", "tomcat",
            "load", "util", "usage", "busy",
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
        """Compute recovery: target entity degradation before vs after restoration.

        Uses capped top-metrics for efficiency and discrimination.
        """
        if top_metrics:
            orig_deg = self._entity_degradation(orig_df, baseline_df, target_entity, logs_df, top_metrics)
            cf_deg = self._entity_degradation(cf_df, baseline_df, target_entity, logs_df, top_metrics)
        else:
            orig_deg = self._entity_degradation(orig_df, baseline_df, target_entity, logs_df)
            cf_deg = self._entity_degradation(cf_df, baseline_df, target_entity, logs_df)

        if orig_deg < 1e-9:
            return 0.0
        improvement = (orig_deg - cf_deg) / orig_deg
        return float(max(0.0, min(1.0, improvement)))

    def _entity_degradation(
        self, df: pd.DataFrame, baseline_df: pd.DataFrame,
        entity: str, logs_df: Optional[pd.DataFrame] = None,
        metric_filter: Optional[List[str]] = None,
    ) -> float:
        """Compute degradation for one entity, optionally filtered to specific metrics."""
        entity_data = df[df["entity"] == entity]
        entity_baseline = baseline_df[baseline_df["entity"] == entity]

        if entity_data.empty:
            return 0.0

        deg = 0.0
        for template in self.metric_templates:
            metric_name = template["name"]
            if metric_filter and metric_name not in metric_filter:
                continue
            weight = template["weight"]

            f_vals = entity_data[entity_data["metric_name"] == metric_name]["value"]
            b_vals = entity_baseline[entity_baseline["metric_name"] == metric_name]["value"]

            if f_vals.empty or b_vals.empty:
                continue

            f_vals = f_vals.dropna().values
            b_vals = b_vals.dropna().values
            if len(f_vals) == 0 or len(b_vals) == 0:
                continue

            b_med = np.median(b_vals)
            mad = np.median(np.abs(b_vals - b_med)) + 1e-9
            f_peak = np.percentile(f_vals, 95)
            deg += (max(0, f_peak - b_med) / mad) * weight

        if logs_df is not None and not logs_df.empty:
            deg += self._log_semantic_score(logs_df, entity) * 10000

        return deg

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

        # Detect anomalies (cap entities to avoid slow computation)
        entities_subset = entities[:20] if len(entities) > 20 else entities
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

        # Quick recovery for top candidates only
        candidates.sort(key=lambda c: c.anomaly_score, reverse=True)
        for i, c in enumerate(candidates):
            if i < 3:  # Only top 3
                cf_df = self.apply_counterfactual(fault_df, baseline_df, c.entity, edge_graph)
                c.recovery_score = self.compute_recovery(
                    fault_df, cf_df, baseline_df, c.entity, edge_graph, entities, telemetry.logs
                )
            else:
                c.recovery_score = 0.0

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
