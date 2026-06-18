"""RCAEval adapter for PRISM-CHT experiments.

This module is intentionally outside ``prism_cht``.  It knows about
RCAEval directory names, CSV columns, and answer labels; the CHT core
only sees a generic case and a telemetry store.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from prismv4.prism_cht.case_types import GenericRCACase, ObservedComponent
from prismv4.prism_cht.telemetry_store import (
    OnsetObservation,
    TraceHop,
    TracePath,
)


BASE_WINDOW_SECONDS = 300
NEAR_ONSET_WINDOW_SECONDS = 30.0
DEFAULT_METRIC_ALIASES = {
    "cpu": ("cpu", "container-cpu-usage-seconds-total"),
    "memory": ("mem", "memory", "container-memory-working-set-bytes"),
    "latency": ("latency-90", "latency-99", "istio-latency-99"),
    "error": ("error", "istio-error-total"),
    "disk": ("diskio", "blkio", "fs-writes", "fs-reads"),
    "socket": ("socket", "sockets"),
    "workload": ("workload", "request-total"),
}
VALID_SERVICE_EXCLUDES = (
    "ip-",
    "compute.internal",
    "frontend-external",
    "loadgenerator",
)


@dataclass(frozen=True)
class RCAEvalLoadedCase:
    case: GenericRCACase
    store: "RCAEvalTelemetryStore"
    expected_component: str
    case_dir: Path


def discover_re3_cases(
    root: str | Path,
    *,
    system: str = "RE3-OB",
    limit: int | None = None,
) -> tuple[Path, ...]:
    root_path = Path(root)
    system_dir = root_path / system
    if not system_dir.exists():
        raise FileNotFoundError(f"RCAEval system directory not found: {system_dir}")

    cases: list[Path] = []
    for fault_dir in sorted(p for p in system_dir.iterdir() if p.is_dir()):
        for rep_dir in sorted(p for p in fault_dir.iterdir() if p.is_dir()):
            if (rep_dir / "metrics.csv").exists() and (rep_dir / "inject_time.txt").exists():
                cases.append(rep_dir)
                if limit is not None and len(cases) >= limit:
                    return tuple(cases)
    return tuple(cases)


def load_re3_case(case_dir: str | Path, *, top_k: int = 5) -> RCAEvalLoadedCase:
    case_path = Path(case_dir)
    event_time = float((case_path / "inject_time.txt").read_text().strip())
    expected_component = _expected_component_from_path(case_path)

    metrics = _read_metrics(case_path)
    logs = _read_optional_csv(case_path / "logs.csv")
    traces = _read_optional_csv(case_path / "traces.csv")

    store = RCAEvalTelemetryStore(
        metrics=metrics,
        logs=logs,
        traces=traces,
        event_time=event_time,
    )
    observations = store.build_observations(top_k=top_k)
    components = tuple(sorted(store.components))
    entries = tuple(c for c in components if "frontend" in c or "gateway" in c)
    if not entries and components:
        entries = (components[0],)

    system_name = case_path.parent.parent.name
    fault_name = case_path.parent.name
    case = GenericRCACase(
        case_id=f"{system_name}/{fault_name}/{case_path.name}",
        dataset_name="RCAEval",
        system_name=system_name,
        event_time=event_time,
        components=components,
        entry_components=entries,
        observations=observations,
        metadata={
            "case_dir": str(case_path),
            "fault_name": fault_name,
        },
    )
    return RCAEvalLoadedCase(
        case=case,
        store=store,
        expected_component=expected_component,
        case_dir=case_path,
    )


class RCAEvalTelemetryStore:
    """TelemetryStore backed by RCAEval metric/log/trace CSVs."""

    def __init__(
        self,
        *,
        metrics: pd.DataFrame,
        logs: pd.DataFrame | None,
        traces: pd.DataFrame | None,
        event_time: float,
    ) -> None:
        self.metrics = metrics
        self.logs = logs
        self.traces = _normalize_traces(traces)
        self.event_time = float(event_time)
        self.components = _extract_components(metrics)
        self._baseline = metrics[metrics["time"] < self.event_time].tail(
            max(1, BASE_WINDOW_SECONDS)
        )
        self._window = metrics[metrics["time"] >= self.event_time]
        self._observations: tuple[ObservedComponent, ...] | None = None

    def build_observations(self, *, top_k: int) -> tuple[ObservedComponent, ...]:
        if self._observations is not None:
            return self._observations[:top_k]

        observed: list[ObservedComponent] = []
        for component in self.components:
            metrics = self._component_metric_observations(component)
            log_items = self._component_log_observations(component)
            if not metrics and not log_items:
                continue

            signals = tuple(sorted({m["signal"] for m in metrics} | {"log"} if log_items else {m["signal"] for m in metrics}))
            first_seen_values = [float(m["first_seen"]) for m in metrics]
            first_seen_values.extend(float(item["timestamp"]) for item in log_items)
            first_seen = min(first_seen_values) if first_seen_values else self.event_time
            magnitude = sum(float(m["magnitude"]) for m in metrics) + float(len(log_items))
            reason = _reason_from_signals(signals)
            symptoms = tuple(
                f"{component} {item['signal']} changed near {item['first_seen']:.3f}"
                for item in metrics[:3]
            ) or (f"{component} has anomalous telemetry",)
            observed.append(
                ObservedComponent(
                    component=component,
                    reason_family=reason,
                    first_seen=first_seen,
                    magnitude=magnitude,
                    signals=signals or ("unknown",),
                    symptoms=symptoms,
                )
            )

        observed.sort(key=lambda o: (-o.magnitude, o.first_seen, o.component))
        self._observations = tuple(observed)
        return self._observations[:top_k]

    def get_onset_observations(
        self,
        *,
        component_scope: Sequence[str],
        signal_scope: Sequence[str],
        time_window: tuple[float, float],
    ) -> tuple[OnsetObservation, ...]:
        signal_set = set(signal_scope)
        rows: list[OnsetObservation] = []
        for component in component_scope:
            for item in self._component_metric_observations(component):
                if item["signal"] not in signal_set:
                    continue
                onset = float(item["first_seen"])
                if time_window[0] <= onset <= time_window[1]:
                    rows.append(
                        OnsetObservation(
                            component=component,
                            signal=str(item["signal"]),
                            onset_time=onset,
                            source="rcaeval_metrics",
                        )
                    )
        rows.sort(key=lambda o: (o.onset_time, o.component, o.signal))
        return tuple(rows)

    def inspect_reason_signatures(
        self,
        *,
        component_scope: Sequence[str],
        reason_by_component: Mapping[str, str],
        time_window: tuple[float, float],
    ) -> tuple[Mapping[str, Any], ...]:
        rows = []
        for component in component_scope:
            metrics = [
                item
                for item in self._component_metric_observations(component)
                if time_window[0] <= float(item["first_seen"]) <= time_window[1]
            ]
            logs = [
                item
                for item in self._component_log_observations(component)
                if time_window[0] <= float(item["timestamp"]) <= time_window[1]
            ]
            magnitude = sum(float(m["magnitude"]) for m in metrics) + len(logs)
            rows.append(
                {
                    "component": component,
                    "reason_family": reason_by_component.get(component, "unspecified anomaly"),
                    "magnitude": magnitude,
                    "signals": [m["signal"] for m in metrics[:5]],
                    "log_examples": [item["message"][:160] for item in logs[:3]],
                }
            )
        rows.sort(key=lambda r: (-float(r["magnitude"]), str(r["component"])))
        return tuple(rows)

    def inspect_noise_features(
        self,
        *,
        component_scope: Sequence[str],
        time_window: tuple[float, float],
        feature_groups: Sequence[str] = (),
    ) -> Mapping[str, Any]:
        rows = [
            self._noise_feature_row(component=component, time_window=time_window)
            for component in component_scope
        ]
        rows.sort(
            key=lambda row: (
                row["earliest_observed_time"] is None,
                row["earliest_observed_time"] or float("inf"),
                -float(row["local_anomaly_magnitude"]),
                row["component"],
            )
        )
        return {
            "feature_groups": list(feature_groups),
            "component_features": rows,
            "window": list(time_window),
            "feature_semantics": [
                "earliest_observed_time is a telemetry observation, not proof of root cause",
                "local_anomaly_magnitude measures local abnormality and can be high for downstream symptoms",
                "near_onset_anomaly_magnitude is usually more relevant than a late dominant metric spike",
                "near_onset_internal_signals are stronger source-mechanism cues than latency-only symptoms",
                "latency_only_near_onset components need corroboration before being selected as root cause",
                "late_dominant_metric means the largest metric spike happens well after the component's first anomaly",
                "log_examples are mechanism clues and should be interpreted with propagation context",
                "log_features include the emitter component; dependency terms in an emitter log are not root-cause labels",
            ],
        }

    def compare_source_symptom(
        self,
        *,
        component_scope: Sequence[str],
        time_window: tuple[float, float],
    ) -> Mapping[str, Any]:
        rows = [
            self._noise_feature_row(component=component, time_window=time_window)
            for component in component_scope
        ]
        rows = [dict(row, source_symptom_cues=_source_symptom_cues(row)) for row in rows]
        rows.sort(
            key=lambda row: (
                row["earliest_observed_time"] is None,
                row["earliest_observed_time"] or float("inf"),
                -float(row["near_onset_anomaly_magnitude"]),
                row["component"],
            )
        )
        earliest_time = next(
            (row["earliest_observed_time"] for row in rows if row["earliest_observed_time"] is not None),
            None,
        )
        earliest_components = (
            [
                row["component"]
                for row in rows
                if row["earliest_observed_time"] == earliest_time
            ]
            if earliest_time is not None
            else []
        )
        return {
            "component_features": rows,
            "relative_cues": {
                "earliest_observed_time": earliest_time,
                "earliest_components": earliest_components,
                "strongest_local_components": [
                    row["component"]
                    for row in sorted(
                        rows,
                        key=lambda item: (-float(item["local_anomaly_magnitude"]), item["component"]),
                    )[:3]
                ],
                "strongest_near_onset_components": [
                    row["component"]
                    for row in sorted(
                        rows,
                        key=lambda item: (-float(item["near_onset_anomaly_magnitude"]), item["component"]),
                    )[:3]
                ],
            },
            "interpretation_rules": [
                "Treat earliest_components as candidates needing corroboration",
                "Treat strongest_local_components as possible symptoms unless mechanism and propagation agree",
                "Prefer strongest_near_onset_components over strongest_local_components when the local maximum is late",
                "Entry component request errors are often upstream symptoms of backend dependency failures",
                "A dependency/storage term in one component's log is not sufficient by itself to choose the dependency",
                "A latency-only earliest component is usually a symptom unless additional internal mechanism evidence supports it",
                "Prefer a component whose local mechanism can explain other affected components",
            ],
        }

    def counterfactual_remove(
        self,
        *,
        component: str,
        symptom_components: Sequence[str],
        time_window: tuple[float, float],
    ) -> Mapping[str, Any]:
        candidate = self._noise_feature_row(component=component, time_window=time_window)
        symptoms = [
            self._noise_feature_row(component=symptom, time_window=time_window)
            for symptom in symptom_components
        ]
        trace_paths: dict[str, int] = {}
        reverse_trace_paths: dict[str, int] = {}
        dependency_direction: dict[str, str] = {}
        for symptom in symptom_components:
            forward = self.find_trace_paths(
                source_component=component,
                target_component=symptom,
                time_window=time_window,
                max_hops=6,
                max_paths=20,
            )
            reverse = self.find_trace_paths(
                source_component=symptom,
                target_component=component,
                time_window=time_window,
                max_hops=6,
                max_paths=20,
            )
            trace_paths[symptom] = len(forward)
            reverse_trace_paths[symptom] = len(reverse)
            dependency_direction[symptom] = _dependency_direction_interpretation(
                candidate=component,
                symptom=symptom,
                forward_count=len(forward),
                reverse_count=len(reverse),
            )
        return {
            "removed_component": component,
            "candidate_features": candidate,
            "symptom_features": symptoms,
            "candidate_to_symptom_trace_path_counts": trace_paths,
            "symptom_to_candidate_trace_path_counts": reverse_trace_paths,
            "request_direction_interpretation": dependency_direction,
            "counterfactual_limits": [
                "RCAEval adapter cannot simulate removal; this exposes observable dependency and feature facts only",
                "If candidate has no path to symptoms, absence is weak evidence unless tracing is complete",
                "Trace paths follow request direction; reverse caller paths can still support a callee/dependency root cause",
            ],
        }

    def test_downstream_explanation(
        self,
        *,
        candidate_component: str,
        symptom_components: Sequence[str],
        time_window: tuple[float, float],
    ) -> Mapping[str, Any]:
        candidate = self._noise_feature_row(
            component=candidate_component,
            time_window=time_window,
        )
        facts = []
        for symptom in symptom_components:
            symptom_row = self._noise_feature_row(
                component=symptom,
                time_window=time_window,
            )
            paths = self.find_trace_paths(
                source_component=candidate_component,
                target_component=symptom,
                time_window=time_window,
                max_hops=6,
                max_paths=20,
            )
            reverse_paths = self.find_trace_paths(
                source_component=symptom,
                target_component=candidate_component,
                time_window=time_window,
                max_hops=6,
                max_paths=20,
            )
            facts.append(
                {
                    "symptom_component": symptom,
                    "symptom_earliest_observed_time": symptom_row["earliest_observed_time"],
                    "candidate_to_symptom_path_count": len(paths),
                    "symptom_to_candidate_path_count": len(reverse_paths),
                    "candidate_observed_before_symptom": (
                        candidate["earliest_observed_time"] is not None
                        and symptom_row["earliest_observed_time"] is not None
                        and candidate["earliest_observed_time"] <= symptom_row["earliest_observed_time"]
                    ),
                    "request_direction_interpretation": _dependency_direction_interpretation(
                        candidate=candidate_component,
                        symptom=symptom,
                        forward_count=len(paths),
                        reverse_count=len(reverse_paths),
                    ),
                    "symptom_signals": symptom_row["signals"],
                    "path_examples": [
                        _trace_path_json(path)
                        for path in paths[:3]
                    ],
                    "reverse_path_examples": [
                        _trace_path_json(path)
                        for path in reverse_paths[:3]
                    ],
                }
            )
        return {
            "candidate_component": candidate_component,
            "candidate_features": candidate,
            "downstream_facts": facts,
            "interpretation_rules": [
                "A candidate explains a symptom better when it has local mechanism evidence and observed-before-symptom timing",
                "Path absence weakens propagation only when trace coverage is expected to be complete",
                "candidate_to_symptom_path_count follows request direction and can mean the candidate calls/depends on the symptom",
                "symptom_to_candidate_path_count can support the candidate as a callee/dependency whose failure surfaced upstream",
                "A symptom with much stronger local mechanism evidence may itself remain a candidate",
            ],
        }

    def build_event_causal_observations(
        self,
        *,
        component_scope: Sequence[str],
        time_window: tuple[float, float],
        max_events: int = 32,
    ) -> Mapping[str, Any]:
        """Build deterministic event-level causal feature candidates.

        This is a feature layer, not a root-cause judge. It fuses metric,
        log, and trace context into event records that an LLM sub-agent can
        further organize into a causal timeline.
        """
        trace_context = self._trace_dependency_context(
            component_scope=component_scope,
            time_window=time_window,
        )
        rows = [
            self._noise_feature_row(component=component, time_window=time_window)
            for component in component_scope
        ]
        rows.sort(
            key=lambda row: (
                row["earliest_observed_time"] is None,
                row["earliest_observed_time"] or float("inf"),
                -float(row["near_onset_anomaly_magnitude"]),
                row["component"],
            )
        )

        events: list[Mapping[str, Any]] = []
        for index, row in enumerate(rows, start=1):
            component = str(row["component"])
            component_trace = trace_context.get(component, _empty_trace_context(component))
            mechanism_strength = _mechanism_strength(row, component_trace)
            primary_event = {
                "event_id": f"event:{index}:{component}:primary",
                "component": component,
                "event_time": row["earliest_observed_time"],
                "event_kind": _event_kind_from_feature_row(row, component_trace),
                "temporal_anchor": "first observed metric/log anomaly for this component",
                "multimodal_bundle": {
                    "metric_features": row["metric_features"][:6],
                    "log_features": row["log_features"][:3],
                    "trace_context": component_trace,
                },
                "causalized_features": {
                    "near_onset_signals": row["near_onset_signals"],
                    "near_onset_internal_signals": row["near_onset_internal_signals"],
                    "near_onset_anomaly_magnitude": row["near_onset_anomaly_magnitude"],
                    "latency_only_near_onset": row["latency_only_near_onset"],
                    "late_dominant_metric": row["late_dominant_metric"],
                    "dominant_metric": row["dominant_metric"],
                    "is_entry_like_component": row["is_entry_like_component"],
                    "mechanism_strength": mechanism_strength,
                    "source_symptom_cues": _source_symptom_cues(row),
                    "event_causal_cues": _event_causal_cues(
                        row,
                        component_trace,
                        mechanism_strength,
                    ),
                },
            }
            events.append(primary_event)
            if row["late_dominant_metric"] and row["dominant_metric"] is not None:
                dominant = row["dominant_metric"]
                events.append(
                    {
                        "event_id": f"event:{index}:{component}:late_dominant_metric",
                        "component": component,
                        "event_time": dominant["first_seen"],
                        "event_kind": "late_dominant_metric_followup",
                        "temporal_anchor": "largest local metric spike, not the component onset",
                        "multimodal_bundle": {
                            "metric_features": [dominant],
                            "log_features": [],
                            "trace_context": component_trace,
                        },
                        "causalized_features": {
                            "late_dominant_metric": True,
                            "seconds_after_primary_event": dominant["seconds_after_earliest"],
                            "interpretation": (
                                "late dominant metric can be a consequence or secondary event; "
                                "do not use it as onset evidence"
                            ),
                        },
                    }
                )

        events.sort(
            key=lambda event: (
                event["event_time"] is None,
                event["event_time"] or float("inf"),
                str(event["event_id"]),
            )
        )
        return {
            "window": list(time_window),
            "component_scope": list(component_scope),
            "event_causal_observations": events[:max_events],
            "component_feature_rows": rows,
            "feature_semantics": [
                "Each event is a causalized feature candidate, not a root-cause verdict.",
                "event_kind summarizes how multimodal evidence should be interpreted before final reasoning.",
                "caller->callee trace direction is request topology, not automatic fault direction.",
                "near_onset_internal_signals are stronger source cues than latency-only near-onset observations.",
                "mechanism_strength weak means single-signal or caller-side evidence that needs corroboration.",
                "Caller-side memory plus callee dependencies can be queueing/blocking symptom, not independent root proof.",
                "Trace error status on a caller->callee edge identifies the failure boundary, not necessarily the initiating component.",
                "late_dominant_metric events should be treated as follow-up evidence unless corroborated by onset facts.",
            ],
        }

    def find_trace_paths(
        self,
        *,
        source_component: str,
        target_component: str,
        time_window: tuple[float, float],
        max_hops: int,
        max_paths: int,
    ) -> tuple[TracePath, ...]:
        if self.traces is None or self.traces.empty:
            return ()
        required = {"trace_id", "span_id", "service", "timestamp"}
        if not required.issubset(self.traces.columns):
            return ()

        paths: list[TracePath] = []
        parent_col = "parent_id" if "parent_id" in self.traces.columns else None
        if parent_col is None:
            return ()

        scoped = self.traces[
            (self.traces["timestamp"] >= time_window[0])
            & (self.traces["timestamp"] <= time_window[1])
        ]
        for _, group in scoped.groupby("trace_id"):
            id_to_row = {
                str(row["span_id"]): row
                for _, row in group.iterrows()
                if pd.notna(row.get("span_id"))
            }
            edges = {}
            for _, row in group.iterrows():
                parent_id = row.get(parent_col)
                if pd.isna(parent_id) or str(parent_id) not in id_to_row:
                    continue
                parent = id_to_row[str(parent_id)]
                src = str(parent.get("service"))
                dst = str(row.get("service"))
                edges.setdefault(src, []).append((dst, row))

            for hop_rows in _find_paths(edges, source_component, target_component, max_hops):
                hops = []
                for dst, row in hop_rows:
                    parent_service = str(row.get("_parent_service", source_component))
                    hops.append(
                        TraceHop(
                            source_component=parent_service,
                            target_component=dst,
                            timestamp=float(row.get("timestamp", self.event_time)),
                            latency_ms=float(row.get("duration", 0.0) or 0.0),
                            status=str(row.get("status", "")),
                        )
                    )
                if hops:
                    paths.append(TracePath(hops=tuple(hops)))
                    if len(paths) >= max_paths:
                        return tuple(paths)
        return tuple(paths)

    def _trace_dependency_context(
        self,
        *,
        component_scope: Sequence[str],
        time_window: tuple[float, float],
    ) -> Mapping[str, Mapping[str, Any]]:
        components = set(component_scope)
        context: dict[str, dict[str, Any]] = {
            component: dict(_empty_trace_context(component))
            for component in component_scope
        }
        if self.traces is None or self.traces.empty:
            return _finalize_trace_context(context)
        required = {"trace_id", "span_id", "service", "timestamp"}
        if not required.issubset(self.traces.columns):
            return _finalize_trace_context(context)
        parent_col = "parent_id" if "parent_id" in self.traces.columns else None
        if parent_col is None:
            return _finalize_trace_context(context)

        scoped = self.traces[
            (self.traces["timestamp"] >= time_window[0])
            & (self.traces["timestamp"] <= time_window[1])
        ]
        for _, group in scoped.groupby("trace_id"):
            id_to_row = {
                str(row["span_id"]): row
                for _, row in group.iterrows()
                if pd.notna(row.get("span_id"))
            }
            for _, row in group.iterrows():
                parent_id = row.get(parent_col)
                if pd.isna(parent_id) or str(parent_id) not in id_to_row:
                    continue
                parent = id_to_row[str(parent_id)]
                caller = str(parent.get("service"))
                callee = str(row.get("service"))
                if caller == callee:
                    continue
                if caller in components:
                    _add_trace_edge_feature(
                        context[caller]["direct_callees"],
                        peer=callee,
                        row=row,
                    )
                if callee in components:
                    _add_trace_edge_feature(
                        context[callee]["direct_callers"],
                        peer=caller,
                        row=row,
                    )

        return _finalize_trace_context(context)

    def retrieve_records(
        self,
        *,
        modality: str,
        component_scope: Sequence[str],
        time_window: tuple[float, float],
        limit: int,
    ) -> tuple[Mapping[str, Any], ...]:
        comp_set = set(component_scope)
        records: list[Mapping[str, Any]] = []
        if modality == "metric":
            for component in component_scope:
                for item in self._component_metric_observations(component):
                    ts = float(item["first_seen"])
                    if time_window[0] <= ts <= time_window[1]:
                        records.append(
                            {
                                "modality": "metric",
                                "component": component,
                                "timestamp": ts,
                                "payload": dict(item),
                            }
                        )
        elif modality == "log":
            for item in self._all_log_observations():
                if item["component"] in comp_set and time_window[0] <= float(item["timestamp"]) <= time_window[1]:
                    records.append(
                        {
                            "modality": "log",
                            "component": item["component"],
                            "timestamp": float(item["timestamp"]),
                            "payload": dict(item),
                        }
                    )
        records.sort(key=lambda r: (float(r["timestamp"]), str(r["component"])))
        return tuple(records[:limit])

    def _noise_feature_row(
        self,
        *,
        component: str,
        time_window: tuple[float, float],
    ) -> Mapping[str, Any]:
        metrics = [
            item
            for item in self._component_metric_observations(component)
            if time_window[0] <= float(item["first_seen"]) <= time_window[1]
        ]
        logs = [
            item
            for item in self._component_log_observations(component)
            if time_window[0] <= float(item["timestamp"]) <= time_window[1]
        ]
        observed_times = [float(item["first_seen"]) for item in metrics]
        observed_times.extend(float(item["timestamp"]) for item in logs)
        signals = sorted({str(item["signal"]) for item in metrics} | ({"log"} if logs else set()))
        earliest = min(observed_times) if observed_times else None
        near_deadline = (
            earliest + NEAR_ONSET_WINDOW_SECONDS
            if earliest is not None
            else None
        )
        near_metrics = [
            item
            for item in metrics
            if near_deadline is not None and float(item["first_seen"]) <= near_deadline
        ]
        near_logs = [
            item
            for item in logs
            if near_deadline is not None and float(item["timestamp"]) <= near_deadline
        ]
        near_onset_signals = sorted(
            {str(item["signal"]) for item in near_metrics}
            | ({"log"} if near_logs else set())
        )
        near_onset_internal_signals = [
            signal
            for signal in near_onset_signals
            if signal in {"cpu", "disk", "error", "log", "memory", "socket"}
        ]
        latency_only_near_onset = bool(near_onset_signals) and all(
            signal in {"latency", "workload"} for signal in near_onset_signals
        )
        dominant_metric = max(
            metrics,
            key=lambda item: float(item["magnitude"]),
            default=None,
        )
        dominant_metric_info = (
            {
                "signal": dominant_metric["signal"],
                "raw_metric": dominant_metric["raw_metric"],
                "first_seen": float(dominant_metric["first_seen"]),
                "magnitude": float(dominant_metric["magnitude"]),
                "direction": dominant_metric["direction"],
                "seconds_after_earliest": (
                    float(dominant_metric["first_seen"]) - earliest
                    if earliest is not None
                    else None
                ),
            }
            if dominant_metric is not None
            else None
        )
        late_dominant_metric = (
            bool(
                dominant_metric_info
                and dominant_metric_info["seconds_after_earliest"] is not None
                and float(dominant_metric_info["seconds_after_earliest"]) > NEAR_ONSET_WINDOW_SECONDS
            )
        )
        return {
            "component": component,
            "earliest_observed_time": earliest,
            "local_anomaly_magnitude": sum(float(item["magnitude"]) for item in metrics) + float(len(logs)),
            "near_onset_window_seconds": NEAR_ONSET_WINDOW_SECONDS,
            "near_onset_signals": near_onset_signals,
            "near_onset_internal_signals": near_onset_internal_signals,
            "latency_only_near_onset": latency_only_near_onset,
            "mechanism_timing_caution": (
                "near-onset evidence is latency/workload only; treat as symptom-like until corroborated"
                if latency_only_near_onset
                else "near-onset evidence includes internal mechanism signals or no near-onset signals were observed"
            ),
            "near_onset_anomaly_magnitude": (
                sum(float(item["magnitude"]) for item in near_metrics) + float(len(near_logs))
            ),
            "dominant_metric": dominant_metric_info,
            "late_dominant_metric": late_dominant_metric,
            "magnitude_timing_caution": (
                "dominant metric spike is late; do not use total local magnitude as onset evidence"
                if late_dominant_metric
                else "dominant metric is near the component's first observed anomaly or no dominant metric exists"
            ),
            "reason_family_from_signals": _reason_from_signals(signals or ("unknown",)),
            "is_entry_like_component": _is_entry_like_component(component),
            "signals": signals,
            "metric_features": [
                {
                    "signal": item["signal"],
                    "raw_metric": item["raw_metric"],
                    "first_seen": float(item["first_seen"]),
                    "magnitude": float(item["magnitude"]),
                    "direction": item["direction"],
                }
                for item in metrics[:10]
            ],
            "log_count": len(logs),
            "log_examples": [item["message"][:180] for item in logs[:5]],
            "log_features": [
                _log_feature(
                    emitter_component=component,
                    message=str(item["message"]),
                    timestamp=float(item["timestamp"]),
                    known_components=self.components,
                )
                for item in logs[:5]
            ],
        }

    def find_symptom_records(
        self,
        *,
        explained_components: Sequence[str],
        time_window: tuple[float, float],
        limit: int,
    ) -> tuple[Mapping[str, Any], ...]:
        explained = set(explained_components)
        symptoms = []
        for obs in self.build_observations(top_k=len(self.components)):
            if obs.component in explained:
                continue
            if time_window[0] <= obs.first_seen <= time_window[1]:
                symptoms.append(
                    {
                        "component": obs.component,
                        "timestamp": obs.first_seen,
                        "signals": list(obs.signals),
                        "magnitude": obs.magnitude,
                    }
                )
        return tuple(symptoms[:limit])

    def _component_metric_observations(self, component: str) -> list[dict[str, Any]]:
        if self._baseline.empty or self._window.empty:
            return []
        rows = []
        prefix = component + "_"
        for column in self.metrics.columns:
            if not column.startswith(prefix):
                continue
            signal = column[len(prefix):]
            baseline_values = pd.to_numeric(self._baseline[column], errors="coerce").dropna()
            window_values = pd.to_numeric(self._window[column], errors="coerce").dropna()
            if baseline_values.empty or window_values.empty:
                continue
            base_med = float(baseline_values.median())
            mad = float((baseline_values - base_med).abs().median())
            if mad <= 1e-9:
                mad = max(abs(base_med) * 0.05, 1e-9)
            deviations = (window_values - base_med).abs() / mad
            max_dev = float(deviations.max()) if not deviations.empty else 0.0
            if max_dev < 3.0:
                continue
            first_idx = deviations[deviations >= 3.0].index[0]
            rows.append(
                {
                    "signal": _canonical_signal(signal),
                    "raw_metric": signal,
                    "first_seen": float(self.metrics.loc[first_idx, "time"]),
                    "magnitude": max_dev,
                    "direction": (
                        "up"
                        if float(self.metrics.loc[first_idx, column]) >= base_med
                        else "down"
                    ),
                }
            )
        rows.sort(key=lambda r: (-float(r["magnitude"]), float(r["first_seen"])))
        return rows

    def _component_log_observations(self, component: str) -> list[dict[str, Any]]:
        return [item for item in self._all_log_observations() if item["component"] == component]

    def _all_log_observations(self) -> list[dict[str, Any]]:
        if self.logs is None or self.logs.empty:
            return []
        if "container_name" not in self.logs.columns or "message" not in self.logs.columns:
            return []
        logs = self.logs.copy()
        if "timestamp" in logs.columns:
            logs["_ts"] = pd.to_numeric(logs["timestamp"], errors="coerce") / 1_000_000_000.0
        elif "time" in logs.columns:
            logs["_ts"] = pd.to_numeric(logs["time"], errors="coerce")
        else:
            logs["_ts"] = self.event_time
        logs = logs[logs["_ts"] >= self.event_time]
        if logs.empty:
            return []
        mask = logs["message"].astype(str).str.contains(
            "error|exception|timeout|fail|oom|killed|refused",
            case=False,
            na=False,
        )
        rows = []
        for _, row in logs[mask].head(500).iterrows():
            rows.append(
                {
                    "component": str(row.get("container_name")),
                    "timestamp": float(row.get("_ts", self.event_time)),
                    "message": str(row.get("message", "")),
                }
            )
        return rows


def _read_metrics(case_path: Path) -> pd.DataFrame:
    preferred = case_path / "simple_metrics.csv"
    path = preferred if preferred.exists() else case_path / "metrics.csv"
    df = pd.read_csv(path)
    if "time.1" in df.columns:
        df = df.drop(columns=["time.1"])
    df = df.replace([float("inf"), float("-inf")], pd.NA).ffill().fillna(0)
    if "time" not in df.columns:
        raise ValueError(f"metrics file has no time column: {path}")
    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    return df.dropna(subset=["time"])


def _read_optional_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path, on_bad_lines="skip")
    except Exception:
        return None


def _normalize_traces(traces: pd.DataFrame | None) -> pd.DataFrame | None:
    if traces is None or traces.empty:
        return traces
    out = traces.copy()
    rename = {}
    for col in out.columns:
        low = col.lower()
        if low in ("traceid", "trace_id"):
            rename[col] = "trace_id"
        elif low in ("spanid", "span_id"):
            rename[col] = "span_id"
        elif low in ("parentspanid", "parent_span_id", "parent_id", "parentid"):
            rename[col] = "parent_id"
        elif low in ("servicename", "service"):
            rename[col] = "service"
        elif low in ("statuscode", "status_code"):
            rename[col] = "status"
    out = out.rename(columns=rename)
    if "startTimeMillis" in out.columns:
        out["timestamp"] = pd.to_numeric(out["startTimeMillis"], errors="coerce") / 1000.0
    elif "startTime" in out.columns:
        out["timestamp"] = pd.to_numeric(out["startTime"], errors="coerce") / 1_000_000.0
    elif "timestamp" not in out.columns and "time" in out.columns:
        out["timestamp"] = pd.to_numeric(out["time"], errors="coerce")
    if "duration" not in out.columns:
        out["duration"] = 0.0
    return out


def _extract_components(metrics: pd.DataFrame) -> tuple[str, ...]:
    components = set()
    for column in metrics.columns:
        if column == "time" or "_" not in column:
            continue
        component = column.split("_", 1)[0]
        if any(pattern in component for pattern in VALID_SERVICE_EXCLUDES):
            continue
        components.add(component)
    return tuple(sorted(components))


def _canonical_signal(metric_suffix: str) -> str:
    low = metric_suffix.lower()
    for name, aliases in DEFAULT_METRIC_ALIASES.items():
        if any(alias in low for alias in aliases):
            return name
    return low.split("-", 1)[0]


def _reason_from_signals(signals: Sequence[str]) -> str:
    signal_set = set(signals)
    if "error" in signal_set:
        return "service error"
    if "latency" in signal_set:
        return "latency degradation"
    if "cpu" in signal_set:
        return "cpu saturation"
    if "memory" in signal_set:
        return "memory pressure"
    if "disk" in signal_set:
        return "disk io saturation"
    if "socket" in signal_set:
        return "connection saturation"
    return "unspecified anomaly"


def _is_entry_like_component(component: str) -> bool:
    low = component.lower()
    return "frontend" in low or "gateway" in low or "ingress" in low


def _source_symptom_cues(row: Mapping[str, Any]) -> list[str]:
    cues: list[str] = []
    if row.get("late_dominant_metric"):
        cues.append(
            "local magnitude is dominated by a late metric spike; compare near_onset_anomaly_magnitude before treating it as a source"
        )
    if row.get("is_entry_like_component") and (
        "error" in set(row.get("signals", [])) or int(row.get("log_count", 0) or 0) > 0
    ):
        cues.append(
            "entry component errors are often upstream symptoms of backend dependency failures"
        )
    if float(row.get("near_onset_anomaly_magnitude", 0.0) or 0.0) <= 0.0 and float(
        row.get("local_anomaly_magnitude", 0.0) or 0.0
    ) > 0.0:
        cues.append("no near-onset local anomaly; later symptoms may dominate total magnitude")
    if row.get("latency_only_near_onset"):
        cues.append(
            "near-onset evidence is latency/workload only; do not choose it over a slightly later internal mechanism without corroboration"
        )
    for feature in row.get("log_features", []) or []:
        if isinstance(feature, Mapping) and feature.get("diagnostic_role") == "dependency_failure_reported_by_emitter":
            cues.append(
                "log reports a dependency/storage failure from the emitter; this is dependency evidence, not a dependency root-cause label"
            )
            if feature.get("emitter_exception_observed"):
                cues.append(
                    "same log also contains emitter-side exception evidence; keep the emitter as a root candidate"
                )
            break
        if isinstance(feature, Mapping) and feature.get("diagnostic_role") == "generic_request_error_observed_by_emitter":
            cues.append(
                "generic request errors identify an affected caller, but not necessarily the initiating component"
            )
            break
    if not cues:
        cues.append("no special caution beyond normal onset, mechanism, and propagation checks")
    return cues


def _event_kind_from_feature_row(
    row: Mapping[str, Any],
    trace_context: Mapping[str, Any],
) -> str:
    log_roles = [
        str(item.get("diagnostic_role"))
        for item in row.get("log_features", []) or []
        if isinstance(item, Mapping)
    ]
    if "dependency_failure_reported_by_emitter" in log_roles:
        return "emitter_dependency_failure_event"
    if row.get("is_entry_like_component") and (
        "error" in set(row.get("signals", [])) or "log" in set(row.get("signals", []))
    ):
        return "entry_error_observation_event"
    if row.get("latency_only_near_onset"):
        return "latency_symptom_candidate_event"
    mechanism = _mechanism_strength(row, trace_context)
    if mechanism["level"] == "weak" and mechanism["cautions"]:
        return "weak_internal_or_queueing_candidate_event"
    if row.get("near_onset_internal_signals"):
        return "internal_mechanism_candidate_event"
    if row.get("late_dominant_metric"):
        return "late_metric_dominated_component_event"
    return "ambiguous_component_anomaly_event"


def _mechanism_strength(
    row: Mapping[str, Any],
    trace_context: Mapping[str, Any],
) -> Mapping[str, Any]:
    internal = set(row.get("near_onset_internal_signals", []) or [])
    signals = set(row.get("near_onset_signals", []) or [])
    log_features = [
        feature
        for feature in row.get("log_features", []) or []
        if isinstance(feature, Mapping)
    ]
    direct_callees = trace_context.get("direct_callees", []) or []
    direct_callers = trace_context.get("direct_callers", []) or []
    caller_error_boundary = any(
        int(item.get("error_status_count", 0) or 0) > 0
        for item in direct_callers
        if isinstance(item, Mapping)
    )
    has_emitter_exception = any(
        feature.get("emitter_exception_observed")
        or feature.get("diagnostic_role") == "emitter_error_or_internal_exception"
        for feature in log_features
    )
    has_error_or_log = bool({"error", "log"} & internal) or has_emitter_exception
    has_multi_mechanism = len(internal - {"log"}) >= 2
    single_internal = len(internal - {"log"}) == 1 and not has_error_or_log
    caller_side_memory_caution = (
        internal <= {"memory"}
        and "memory" in internal
        and bool(direct_callees)
        and not has_error_or_log
    )

    reasons: list[str] = []
    cautions: list[str] = []
    if has_emitter_exception:
        reasons.append("emitter-side exception/log evidence")
    if has_multi_mechanism:
        reasons.append(
            "multiple near-onset internal mechanism signals: "
            + ", ".join(sorted(internal - {"log"}))
        )
    elif internal:
        reasons.append(
            "single near-onset internal mechanism signal: "
            + ", ".join(sorted(internal))
        )
    if row.get("latency_only_near_onset"):
        cautions.append("near-onset evidence is latency/workload only")
    if caller_side_memory_caution:
        cautions.append(
            "single caller-side memory signal with callees can be queueing/blocking symptom"
        )
    if row.get("late_dominant_metric"):
        cautions.append("dominant metric spike is late relative to primary event")
    if caller_error_boundary:
        cautions.append(
            "caller trace errors identify a failure boundary, not independent root proof"
        )

    if has_error_or_log or has_multi_mechanism:
        level = "strong"
    elif row.get("latency_only_near_onset"):
        level = "symptom_like"
    elif caller_side_memory_caution:
        level = "weak"
    elif single_internal:
        level = "moderate" if direct_callers and not direct_callees else "weak"
    elif signals:
        level = "weak"
    else:
        level = "unknown"

    if not reasons:
        reasons.append("no strong near-onset internal mechanism cue")
    return {
        "level": level,
        "reasons": reasons[:4],
        "cautions": cautions[:5],
        "caller_error_boundary": caller_error_boundary,
        "caller_side_memory_caution": caller_side_memory_caution,
    }


def _event_causal_cues(
    row: Mapping[str, Any],
    trace_context: Mapping[str, Any],
    mechanism_strength: Mapping[str, Any],
) -> Mapping[str, list[str]]:
    source_like: list[str] = []
    symptom_like: list[str] = []
    ambiguity: list[str] = []

    internal = list(row.get("near_onset_internal_signals", []) or [])
    if internal:
        if mechanism_strength.get("level") in {"strong", "moderate"}:
            source_like.append(
                f"near-onset internal mechanism signals observed: {', '.join(internal)}"
            )
        else:
            ambiguity.append(
                "near-onset internal signal is weak or single-signal; corroborate before selecting as root"
            )
    if row.get("latency_only_near_onset"):
        symptom_like.append(
            "near-onset evidence is latency/workload only, which is often a downstream symptom"
        )
    if row.get("is_entry_like_component") and (
        "error" in set(row.get("signals", [])) or int(row.get("log_count", 0) or 0) > 0
    ):
        symptom_like.append(
            "entry component errors are often where backend failures become visible"
        )
    if row.get("late_dominant_metric"):
        ambiguity.append(
            "largest local magnitude occurs late; separate follow-up event from initiating event"
        )

    direct_callers = trace_context.get("direct_callers", [])
    direct_callees = trace_context.get("direct_callees", [])
    if direct_callers:
        if row.get("latency_only_near_onset") or mechanism_strength.get("level") == "weak":
            ambiguity.append(
                "has direct callers, but mechanism evidence is weak; caller errors may be failure-boundary evidence"
            )
        else:
            source_like.append(
                "has direct callers; a fault here can surface upstream in caller errors or latency"
            )
    if direct_callees:
        ambiguity.append(
            "has direct callees/dependencies; request path to a callee does not prove caller-root causality"
        )
    if mechanism_strength.get("caller_side_memory_caution"):
        symptom_like.append(
            "single memory increase in a caller can reflect queueing/blocking on a slower callee"
        )
    if mechanism_strength.get("caller_error_boundary"):
        ambiguity.append(
            "trace error status on caller edge marks where failure was observed, not necessarily where it started"
        )
    for caution in mechanism_strength.get("cautions", []) or []:
        if isinstance(caution, str) and caution not in ambiguity:
            ambiguity.append(caution)

    for feature in row.get("log_features", []) or []:
        if not isinstance(feature, Mapping):
            continue
        if feature.get("diagnostic_role") == "dependency_failure_reported_by_emitter":
            ambiguity.append(
                "emitter log mentions dependency/storage failure; mentioned dependency is evidence, not a label"
            )
            if feature.get("emitter_exception_observed"):
                source_like.append(
                    "same log contains emitter-side exception/status evidence"
                )
        elif feature.get("diagnostic_role") == "generic_request_error_observed_by_emitter":
            symptom_like.append("generic request error is visibility evidence, not source proof")

    if not source_like:
        source_like.append("no strong near-onset source cue detected in deterministic features")
    if not symptom_like:
        symptom_like.append("no strong symptom-only cue detected in deterministic features")
    if not ambiguity:
        ambiguity.append("no special deterministic ambiguity cue")
    return {
        "source_like": source_like,
        "symptom_like": symptom_like,
        "ambiguity": ambiguity,
    }


def _dependency_direction_interpretation(
    *,
    candidate: str,
    symptom: str,
    forward_count: int,
    reverse_count: int,
) -> str:
    if forward_count and reverse_count:
        return (
            f"request traces exist both {candidate}->{symptom} and {symptom}->{candidate}; "
            "use local mechanism and timing to infer fault direction"
        )
    if forward_count:
        return (
            f"request path {candidate}->{symptom} means {candidate} calls or reaches {symptom}; "
            "this is dependency/request direction, not proof that the caller caused the callee anomaly"
        )
    if reverse_count:
        return (
            f"request path {symptom}->{candidate} means {candidate} is a callee/dependency; "
            f"a {candidate} fault can surface upstream as {symptom} errors or latency"
        )
    return (
        "no request path observed in either direction; absence weakens dependency evidence only if trace coverage is complete"
    )


def _empty_trace_context(component: str) -> Mapping[str, Any]:
    return {
        "component": component,
        "direct_callers": {},
        "direct_callees": {},
        "request_direction_semantics": (
            "trace context unavailable or no direct edges observed in the event window"
        ),
    }


def _add_trace_edge_feature(
    edge_map: dict[str, dict[str, Any]],
    *,
    peer: str,
    row,
) -> None:
    item = edge_map.setdefault(
        peer,
        {
            "component": peer,
            "call_count": 0,
            "error_status_count": 0,
            "latency_examples": [],
            "status_examples": [],
        },
    )
    item["call_count"] += 1
    status = str(row.get("status", ""))
    if status and status.lower() != "nan" and status not in ("0", "0.0", "OK", "ok"):
        item["error_status_count"] += 1
    if len(item["latency_examples"]) < 3:
        item["latency_examples"].append(float(row.get("duration", 0.0) or 0.0))
    if status and status.lower() != "nan" and len(item["status_examples"]) < 3:
        item["status_examples"].append(status)


def _finalize_trace_context(
    context: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Mapping[str, Any]]:
    finalized: dict[str, Mapping[str, Any]] = {}
    for component, item in context.items():
        callers = item.get("direct_callers", {})
        callees = item.get("direct_callees", {})
        finalized[component] = {
            "component": component,
            "direct_callers": _trace_edge_summary_list(callers if isinstance(callers, Mapping) else {}),
            "direct_callees": _trace_edge_summary_list(callees if isinstance(callees, Mapping) else {}),
            "request_direction_semantics": (
                "direct_callers can be upstream components where this component's fault may surface; "
                "direct_callees are dependencies or downstream calls and do not prove this component caused their anomaly"
            ),
        }
    return finalized


def _trace_edge_summary_list(edge_map: Mapping[str, Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    rows = list(edge_map.values())
    rows.sort(
        key=lambda item: (
            -int(item.get("error_status_count", 0)),
            -int(item.get("call_count", 0)),
            str(item.get("component", "")),
        )
    )
    return rows[:8]


def _trace_path_json(path: TracePath) -> list[Mapping[str, Any]]:
    return [
        {
            "source": hop.source_component,
            "target": hop.target_component,
            "timestamp": hop.timestamp,
            "latency_ms": hop.latency_ms,
            "status": hop.status,
        }
        for hop in path.hops
    ]


def _log_feature(
    *,
    emitter_component: str,
    message: str,
    timestamp: float,
    known_components: Sequence[str],
) -> Mapping[str, Any]:
    low = message.lower()
    dependency_terms = [
        component
        for component in known_components
        if component != emitter_component and component.lower() in low
    ]
    for term in ("redis", "storage", "cart storage", "database", "db", "cache"):
        if term in low and term not in dependency_terms:
            dependency_terms.append(term)

    has_dependency_failure = bool(dependency_terms) and any(
        token in low
        for token in (
            "can't access",
            "cannot access",
            "failed",
            "fail",
            "timeout",
            "refused",
            "unavailable",
            "precondition",
            "connection",
        )
    )
    emitter_exception_observed = any(
        token in low
        for token in (
            "exception",
            "overflowexception",
            "nullpointerexception",
            "panic",
            "stacktrace",
            "error status",
        )
    )
    if has_dependency_failure:
        role = "dependency_failure_reported_by_emitter"
        caution = (
            "The log is emitted by the component under analysis; mentioned dependencies are clues, not labels."
        )
    elif "request error" in low:
        role = "generic_request_error_observed_by_emitter"
        caution = "Generic request errors often reflect upstream visibility of backend failures."
    elif any(token in low for token in ("exception", "error", "fail", "timeout", "oom", "killed")):
        role = "emitter_error_or_internal_exception"
        caution = "May support an emitter-side mechanism when consistent with metrics and propagation."
    else:
        role = "log_anomaly_observed_by_emitter"
        caution = "Interpret with metrics, timing, and dependency context."

    return {
        "emitter_component": emitter_component,
        "timestamp": timestamp,
        "message_preview": message[:180],
        "dependency_terms": dependency_terms[:8],
        "emitter_exception_observed": emitter_exception_observed,
        "diagnostic_role": role,
        "interpretation_caution": caution,
    }


def _expected_component_from_path(case_path: Path) -> str:
    fault_name = case_path.parent.name
    return fault_name.rsplit("_f", 1)[0]


def _find_paths(edges, source, target, max_hops):
    found = []

    def dfs(node, path, visited):
        if len(path) >= max_hops:
            return
        for child, row in edges.get(node, []):
            if child in visited:
                continue
            row = row.copy()
            row["_parent_service"] = node
            next_path = path + [(child, row)]
            if child == target:
                found.append(next_path)
            else:
                dfs(child, next_path, visited | {child})

    dfs(source, [], {source})
    return found
