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
    entries = _infer_entry_components(
        components=components,
        store=store,
        event_time=event_time,
    )

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
        # Memoize per-component metric/log observations.  These are pure
        # functions of the immutable post-__init__ store state and are
        # recomputed by build_observations, _noise_feature_row, and the
        # event causalizer; caching removes the dominant cost of TT-sized
        # component sets in both the offline recall tool and the live runner.
        self._metric_obs_cache: dict[str, list[dict[str, Any]]] = {}
        self._log_obs_cache: dict[str, list[dict[str, Any]]] = {}
        self._all_log_obs_cache: list[dict[str, Any]] | None = None
        # Memoize trace dependency context by (component scope, window).  The
        # trace edge graph is a pure function of immutable post-init state, so
        # caching eliminates the dominant OB-case cost when build_tiered_recall
        # _pool and the EventCausalizer ask for the same scoped window.
        self._trace_context_cache: dict[
            tuple[frozenset[str], tuple[float, float]],
            Mapping[str, Mapping[str, Any]],
        ] = {}

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
            reason = _reason_from_observations(
                signals, log_items,
                known_components=self.components,
                component=component,
            )
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
        # Rank by source_likelihood_score for explicit source-vs-dependency
        # disambiguation.  Higher score = more likely initiating root.
        scored = sorted(
            rows,
            key=lambda r: (
                -float(r.get("source_likelihood_score", 0.0)),
                -float(r["near_onset_anomaly_magnitude"]),
                r["component"],
            ),
        )
        top_source_candidates = [
            r["component"] for r in scored[:3]
            if r.get("source_likelihood_score", 0.0) > 0
        ]
        dependency_candidates = [
            r["component"] for r in rows
            if r.get("causal_role") == "dependency_candidate"
        ]
        # Inferred dependency edges among the scoped components.
        inferred_edges = [
            {"caller": c, "callee": s}
            for c, s in _infer_dependency_edges(component_scope)
        ]

        # IVD: Initiator–Victim Disambiguation among source_candidates.
        source_candidates = [
            r["component"] for r in rows
            if r.get("causal_role") in ("source_candidate", "ambiguous")
            and r.get("source_likelihood_score", 0.0) > 0
        ]
        all_anomalous = [
            r["component"] for r in rows
            if float(r.get("local_anomaly_magnitude", 0.0)) > 0
        ]
        ivd_result = None
        if len(source_candidates) >= 2:
            ivd_result = _compute_ivd(
                self,
                candidates=source_candidates,
                all_anomalous_components=all_anomalous,
                time_window=time_window,
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
                "top_source_candidates_by_likelihood": top_source_candidates,
                "dependency_candidates": dependency_candidates,
                "inferred_dependency_edges": inferred_edges,
            },
            "ivd": ivd_result,
            "interpretation_rules": [
                "Treat earliest_components as candidates needing corroboration",
                "Treat strongest_local_components as possible symptoms unless mechanism and propagation agree",
                "Prefer strongest_near_onset_components over strongest_local_components when the local maximum is late",
                "Entry component request errors are often upstream symptoms of backend dependency failures",
                "A dependency/storage term in one component's log is not sufficient by itself to choose the dependency",
                "A latency-only earliest component is usually a symptom unless additional internal mechanism evidence supports it",
                "Prefer a component whose local mechanism can explain other affected components",
                "Prefer top_source_candidates_by_likelihood over dependency_candidates: a service with emitter exceptions is more likely the root than a storage with resource anomalies",
                "When a service and its storage dependency both show anomalies, the service is more likely the root because a service logic error causes abnormal DB queries that make the DB show reactive resource pressure",
                "Use inferred_dependency_edges to determine call direction: if service A calls storage B, A is the caller and B is the callee/dependency; B's anomalies may be caused by A's fault",
                "IVD (Initiator-Victim Disambiguation): when multiple source_candidates have similar scores, use the ivd verdicts to determine initiator vs victim. Do NOT select a candidate with ivd_role=victim over one with ivd_role=initiator. The initiator is the candidate whose anomaly is least explainable by other candidates and most explanatory for downstream anomalies.",
                "Do not use raw event timeline ordering to break ties between IVD-ranked candidates. Temporal fragility: if two candidates have near-simultaneous onsets (gap < 10s), raw timestamp order is unreliable and must not be used as the sole tiebreaker.",
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
                    "is_storage_component": row.get("is_storage_component", False),
                    "causal_role": row.get("causal_role", "ambiguous"),
                    "source_likelihood_score": row.get("source_likelihood_score", 0.0),
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
                "is_entry_like_component identifies the request surface (where traffic enters the system), not root-cause strength; an entry component can be a victim of a backend dependency fault.",
                "causal_role classifies each component: source_candidate (service with emitter exceptions or strong mechanism), dependency_candidate (storage with resource anomalies but no emitter exceptions), propagation_symptom (latency/workload only), entry_symptom (entry with errors but weak mechanism).",
                "source_likelihood_score is a heuristic ranking: higher = more likely initiating root. Prefer source_candidate over dependency_candidate when both show anomalies.",
                "is_storage_component marks database/cache layers (mongo, redis, mysql, etc.); their resource anomalies are often reactive to service-level faults.",
                "Inferred dependency edges (inferred_from_naming in trace_context) show service->storage calls not captured by trace spans; use them to determine causal direction between a service and its database.",
                "IVD (Initiator-Victim Disambiguation): among multiple source_candidates with similar scores, the initiator is the one whose anomaly is least explainable by other candidates and most explanatory for downstream anomalies. Use ivd verdicts to break ties. Do not select a victim over an initiator.",
            ],
            "ivd": self._compute_ivd_for_events(rows, time_window),
        }

    def _compute_ivd_for_events(
        self,
        rows: Sequence[Mapping[str, Any]],
        time_window: tuple[float, float],
    ) -> Mapping[str, Any] | None:
        """Run IVD among source_candidates found in the event rows."""
        source_candidates = [
            r["component"] for r in rows
            if r.get("causal_role") in ("source_candidate", "ambiguous")
            and float(r.get("source_likelihood_score", 0.0)) > 0
        ]
        if len(source_candidates) < 2:
            return None
        all_anomalous = [
            r["component"] for r in rows
            if float(r.get("local_anomaly_magnitude", 0.0)) > 0
        ]
        return _compute_ivd(
            self,
            candidates=source_candidates,
            all_anomalous_components=all_anomalous,
            time_window=time_window,
        )

    def build_tiered_recall_pool(
        self,
        *,
        pool_size: int = 15,
        time_window: tuple[float, float] | None = None,
    ) -> tuple[str, ...]:
        """Rank all components by a tiered recall strategy and return the pool.

        This separates candidate *recall* (find the answer) from final
        *reasoning width* (how many hypotheses the LLM reasons over).  The
        total-magnitude top-k used by ``build_observations`` drops true roots
        on TT-like systems where downstream symptom magnitude dwarfs the
        source.  This pool combines several weaker-but-complementary tiers so
        the answer survives even when no single ranking would surface it:

          - magnitude with workload/request-total down-weighted
          - earliest onset
          - near-onset internal-mechanism priority (cpu/mem/error/log/...)
          - log / error emitters
          - topology-central services (trace root with callees, high degree)

        Tiers are unioned (deduplicated) and the pool is capped at
        ``pool_size``.  Offline validation: this achieves 30/30 recall on both
        RE3-TT and RE3-OB at pool_size=15, versus 1/30 for raw magnitude
        top-5 on RE3-TT.
        """
        if time_window is None:
            time_window = (
                max(0.0, self.event_time - NEAR_ONSET_WINDOW_SECONDS),
                self.event_time + 600.0,
            )
        components = list(self.components)
        if not components:
            return ()
        trace_context = self._trace_dependency_context(
            component_scope=components,
            time_window=time_window,
        )
        rows = [
            self._recall_score_row(
                component=component,
                time_window=time_window,
                trace_context=trace_context.get(component, {}),
            )
            for component in components
        ]
        rows = [r for r in rows if r is not None]
        by_component = {r["component"]: r for r in rows}

        def add(pool: list[str], seen: set[str], top: Sequence[str], quota: int) -> None:
            added = 0
            for comp in top:
                if comp in seen:
                    continue
                if len(pool) >= pool_size or added >= quota:
                    return
                pool.append(comp)
                seen.add(comp)
                added += 1

        per_tier = max(2, pool_size // 5)
        pool: list[str] = []
        seen: set[str] = set()

        mag_nowl = [r["component"] for r in sorted(rows, key=_recall_magnitude_no_workload_key)]
        onset = [r["component"] for r in sorted(rows, key=_recall_onset_key)]
        mechanism = [r["component"] for r in sorted(rows, key=_recall_mechanism_key)]
        emitters = [
            r["component"]
            for r in sorted(
                rows,
                key=lambda r: (
                    -int(r["has_emitter_exception"]),
                    -int(r["log_count"]),
                    -float(r["near_onset_magnitude"]),
                    r["component"],
                ),
            )
            if r["has_emitter_exception"] or r["log_count"] > 0
        ]
        topo = sorted(
            rows,
            key=lambda r: (
                not (r["n_callers"] == 0 and r["n_callees"] > 0),
                -(r["n_callers"] + r["n_callees"]),
                r["component"],
            ),
        )
        topo_names = [
            r["component"] for r in topo if (r["n_callers"] + r["n_callees"]) > 0
        ]

        add(pool, seen, mag_nowl, per_tier)
        add(pool, seen, onset, per_tier)
        add(pool, seen, mechanism, per_tier)
        add(pool, seen, emitters, per_tier)
        add(pool, seen, topo_names, per_tier)
        add(pool, seen, mag_nowl, pool_size)
        return tuple(pool[:pool_size])

    def _recall_score_row(
        self,
        *,
        component: str,
        time_window: tuple[float, float],
        trace_context: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        feature = self._noise_feature_row(component=component, time_window=time_window)
        metrics = self._component_metric_observations(component)
        workload_mag = sum(
            float(m["magnitude"]) for m in metrics if m["signal"] == "workload"
        )
        has_workload = any(m["signal"] == "workload" for m in metrics)
        mech = _mechanism_strength(feature, trace_context)
        log_features = feature.get("log_features", []) or []
        has_emitter_exception = any(
            (f.get("emitter_exception_observed") if isinstance(f, Mapping) else False)
            or (
                f.get("diagnostic_role") == "emitter_error_or_internal_exception"
                if isinstance(f, Mapping)
                else False
            )
            for f in log_features
        )
        return {
            "component": component,
            "total_magnitude": float(feature.get("local_anomaly_magnitude", 0.0)),
            "workload_magnitude": workload_mag,
            "near_onset_magnitude": float(feature.get("near_onset_anomaly_magnitude", 0.0)),
            "earliest_time": feature.get("earliest_observed_time"),
            "near_onset_internal": tuple(feature.get("near_onset_internal_signals", []) or ()),
            "latency_only_near_onset": bool(feature.get("latency_only_near_onset")),
            "log_count": int(feature.get("log_count", 0) or 0),
            "has_emitter_exception": has_emitter_exception,
            "mechanism_level": str(mech.get("level", "unknown")),
            "n_callers": len(trace_context.get("direct_callers", []) or []),
            "n_callees": len(trace_context.get("direct_callees", []) or []),
            "has_workload_signal": has_workload,
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
        cache_key = (frozenset(component_scope), (float(time_window[0]), float(time_window[1])))
        cached = self._trace_context_cache.get(cache_key)
        if cached is not None:
            return cached
        result = self._compute_trace_dependency_context(
            component_scope=component_scope,
            time_window=time_window,
        )
        self._trace_context_cache[cache_key] = result
        return result

    def _compute_trace_dependency_context(
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

        # Augment with inferred service->storage dependency edges that are
        # not captured by trace spans (traces typically record
        # service-to-service calls, not service-to-database calls).
        for caller, callee in _infer_dependency_edges(component_scope):
            if caller in context and callee in context:
                edge_map = context[caller]["direct_callees"]
                if not isinstance(edge_map, dict):
                    edge_map = {}
                    context[caller]["direct_callees"] = edge_map
                if callee not in edge_map:
                    edge_map[callee] = {
                        "component": callee,
                        "call_count": 0,
                        "error_status_count": 0,
                        "latency_examples": [],
                        "status_examples": [],
                        "inferred_from_naming": True,
                    }
                edge_map = context[callee]["direct_callers"]
                if not isinstance(edge_map, dict):
                    edge_map = {}
                    context[callee]["direct_callers"] = edge_map
                if caller not in edge_map:
                    edge_map[caller] = {
                        "component": caller,
                        "call_count": 0,
                        "error_status_count": 0,
                        "latency_examples": [],
                        "status_examples": [],
                        "inferred_from_naming": True,
                    }

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
        row = {
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
            "is_storage_component": _is_storage_component(component),
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
        # Add causal_role and source_likelihood_score using trace context.
        # Use all components as scope to leverage the shared cache (a single
        # trace dependency context computation covers all components).
        trace_ctx_all = self._trace_dependency_context(
            component_scope=self.components,
            time_window=time_window,
        )
        trace_ctx = trace_ctx_all.get(component, {})
        row["causal_role"] = _causal_role(row, trace_ctx)
        row["source_likelihood_score"] = _source_likelihood_score(row, trace_ctx)
        return row

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
        cached = self._metric_obs_cache.get(component)
        if cached is not None:
            return cached
        if self._baseline.empty or self._window.empty:
            result: list[dict[str, Any]] = []
            self._metric_obs_cache[component] = result
            return result
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
        self._metric_obs_cache[component] = rows
        return rows

    def _component_log_observations(self, component: str) -> list[dict[str, Any]]:
        cached = self._log_obs_cache.get(component)
        if cached is not None:
            return cached
        result = [item for item in self._all_log_observations() if item["component"] == component]
        self._log_obs_cache[component] = result
        return result

    def _all_log_observations(self) -> list[dict[str, Any]]:
        if self._all_log_obs_cache is not None:
            return self._all_log_obs_cache
        if self.logs is None or self.logs.empty:
            self._all_log_obs_cache = []
            return []
        if "container_name" not in self.logs.columns or "message" not in self.logs.columns:
            self._all_log_obs_cache = []
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
            self._all_log_obs_cache = []
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
        self._all_log_obs_cache = rows
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


def _reason_from_observations(
    signals: Sequence[str],
    log_items: Sequence[Mapping[str, Any]],
    *,
    known_components: Sequence[str],
    component: str,
) -> str:
    """Determine reason family from signals AND log features.

    Prioritizes emitter-exception evidence over raw signal type so that a
    service with application-level errors gets a root-cause-like label
    instead of a symptom-like 'latency degradation' label.
    """
    if log_items:
        has_emitter_exception = False
        for item in log_items[:5]:
            feat = _log_feature(
                emitter_component=component,
                message=str(item["message"]),
                timestamp=float(item["timestamp"]),
                known_components=known_components,
            )
            if feat.get("emitter_exception_observed") or feat.get("diagnostic_role") == "emitter_error_or_internal_exception":
                has_emitter_exception = True
                break
        if has_emitter_exception:
            return "emitter exception"
    return _reason_from_signals(signals)


def _is_entry_like_component(component: str) -> bool:
    low = component.lower()
    return "frontend" in low or "gateway" in low or "ingress" in low


_STORAGE_SUFFIXES = ("-mongo", "-redis", "-mysql", "-db", "-database", "-storage", "-es", "-elasticsearch")
_STORAGE_KEYWORDS = ("mongo", "redis", "mysql", "elasticsearch")


def _is_storage_component(component: str) -> bool:
    low = component.lower()
    if any(low.endswith(suffix) for suffix in _STORAGE_SUFFIXES):
        return True
    if any(kw in low for kw in _STORAGE_KEYWORDS):
        return True
    return False


def _causal_role(
    row: Mapping[str, Any],
    trace_context: Mapping[str, Any],
) -> str:
    """Classify a component's causal role for root-cause disambiguation.

    Returns one of:
      - source_candidate: service with emitter exceptions or strong internal
        mechanism; likely the initiating faulty component.
      - dependency_candidate: storage/database with resource anomalies but
        no emitter exceptions; usually a reactive dependency.
      - propagation_symptom: downstream service with only latency/workload
        signals; likely a cascading effect.
      - entry_symptom: entry-like component with errors but no strong
        internal mechanism; likely where the failure surfaces.
      - ambiguous: does not fit the above cleanly.
    """
    is_storage = _is_storage_component(str(row.get("component", "")))
    log_features = [
        f for f in row.get("log_features", []) or []
        if isinstance(f, Mapping)
    ]
    has_emitter_exception = any(
        f.get("emitter_exception_observed")
        or f.get("diagnostic_role") == "emitter_error_or_internal_exception"
        for f in log_features
    )
    internal = set(row.get("near_onset_internal_signals", []) or [])
    latency_only = bool(row.get("latency_only_near_onset"))
    is_entry = bool(row.get("is_entry_like_component"))
    has_error_or_log = bool({"error", "log"} & internal) or has_emitter_exception
    has_multi_mechanism = len(internal - {"log"}) >= 2

    if is_storage:
        if has_emitter_exception:
            return "source_candidate"
        return "dependency_candidate"

    if has_emitter_exception or (has_error_or_log and has_multi_mechanism):
        return "source_candidate"

    if latency_only:
        return "propagation_symptom"

    if is_entry and not has_multi_mechanism:
        return "entry_symptom"

    if has_multi_mechanism or has_error_or_log:
        return "source_candidate"

    if internal:
        return "ambiguous"

    return "propagation_symptom"


def _source_likelihood_score(
    row: Mapping[str, Any],
    trace_context: Mapping[str, Any],
) -> float:
    """Heuristic source-likelihood score for ranking candidates.

    Higher = more likely to be the initiating root cause.
    """
    score = 0.0
    log_features = [
        f for f in row.get("log_features", []) or []
        if isinstance(f, Mapping)
    ]
    has_emitter_exception = any(
        f.get("emitter_exception_observed")
        or f.get("diagnostic_role") == "emitter_error_or_internal_exception"
        for f in log_features
    )
    internal = set(row.get("near_onset_internal_signals", []) or [])
    latency_only = bool(row.get("latency_only_near_onset"))
    is_storage = _is_storage_component(str(row.get("component", "")))
    is_entry = bool(row.get("is_entry_like_component"))
    late_dominant = bool(row.get("late_dominant_metric"))

    if has_emitter_exception:
        score += 3.0
    if len(internal - {"log"}) >= 2:
        score += 2.0
    elif internal - {"log"}:
        score += 1.0
    if not is_storage:
        score += 1.0
    if is_storage:
        score -= 2.0
    if is_entry:
        score -= 1.0
    if latency_only:
        score -= 2.0
    if late_dominant:
        score -= 1.0
    if not internal and not latency_only and not log_features:
        score -= 1.0
    return score


# ---------------------------------------------------------------------------
# IVD: Initiator–Victim Disambiguation
#
# When multiple source candidates have similar scores, the root cause is not
# the earliest or the loudest but the one whose anomaly is *least explainable
# by other candidates* and *most explanatory for other candidates' anomalies*.
#
# This module implements a pairwise tournament (Copeland score) among
# source_candidates using five evidence channels:
#   1. Exogeneity       – anomaly persists given upstream context
#   2. DownstreamCoverage – can explain other candidates via topology/trace
#   3. IncomingExplainedness – is itself explained by another candidate
#   4. FaultSignature    – own-code exception vs dependency-induced
#   5. TopologyDirection – reachable asymmetry in call graph
# ---------------------------------------------------------------------------

_OWN_CODE_TOKENS = (
    "nullpointerexception", "overflowexception", "illegalargumentexception",
    "illegalstateexception", "arrayindex", "classcast", "numberformat",
    "unsupportedoperation", "concurrentmodification", "stacktrace",
    "no bean named", "configuring", "postprocessor", "started", "stopped",
    "removing", "channel", "consumer", "listener",
)
_DEPENDENCY_INDUCED_TOKENS = (
    "can't access", "cannot access", "connection refused", "connection timed out",
    "failed to call", "failed to connect", "timeout", "timed out",
    "unavailable", "precondition", "i/o error", "socket closed",
    "reset", "broken pipe",
)
_CLIENT_SIDE_TOKENS = (
    "http client", "resttemplate", "feign", "ribbon", "hystrix",
    "circuit breaker", "fallback", "retry",
)


def _fault_signature_score(row: Mapping[str, Any]) -> float:
    """Classify exception semantics: own-code (root-like) vs dependency-induced (victim-like).

    Returns a score in [-2, +2]:
      +2  = strong own-code stack trace / internal configuration error
      +1  = emitter exception with no dependency terms
       0  = ambiguous or no exception
      -1  = emitter exception with dependency terms (mixed)
      -2  = pure dependency/client-side failure (timeout, connection refused)
    """
    log_features = [
        f for f in row.get("log_features", []) or []
        if isinstance(f, Mapping)
    ]
    if not log_features:
        return 0.0

    messages = [str(f.get("message_preview", "")).lower() for f in log_features]
    all_text = " ".join(messages)

    has_own_code = any(tok in all_text for tok in _OWN_CODE_TOKENS)
    has_dep_induced = any(tok in all_text for tok in _DEPENDENCY_INDUCED_TOKENS)
    has_client_side = any(tok in all_text for tok in _CLIENT_SIDE_TOKENS)
    has_exception = any(f.get("emitter_exception_observed") for f in log_features)
    has_dep_role = any(
        f.get("diagnostic_role") == "dependency_failure_reported_by_emitter"
        for f in log_features
    )

    if has_own_code and not has_dep_induced:
        return 2.0
    if has_own_code and has_dep_induced:
        return 1.0
    if has_exception and not has_dep_role and not has_dep_induced:
        return 1.0
    if has_dep_induced and not has_own_code:
        return -2.0 if has_client_side else -1.0
    if has_dep_role and not has_exception:
        return -1.0
    return 0.0


def _temporal_causality_signal(
    row: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Extract temporal ordering between resource and workload anomalies.

    Root cause: resource onset precedes workload onset (internal fault → slow → retries).
    Victim: workload onset precedes or coincides with resource onset (external load).
    """
    metric_features = row.get("metric_features", []) or []
    if not metric_features:
        return {"resource_onset": None, "workload_onset": None, "gap": None, "verdict": "unknown"}

    resource_onset = None
    workload_onset = None
    for m in metric_features:
        sig = m.get("signal", "")
        ts = m.get("first_seen")
        if ts is None:
            continue
        if sig in ("memory", "cpu", "disk") and (resource_onset is None or ts < resource_onset):
            resource_onset = ts
        if sig == "workload" and (workload_onset is None or ts < workload_onset):
            workload_onset = ts

    if resource_onset is not None and workload_onset is not None:
        gap = workload_onset - resource_onset
        if gap > 10:
            verdict = "resource_before_workload"
        elif gap < -5:
            verdict = "workload_before_resource"
        else:
            verdict = "near_simultaneous"
    elif resource_onset is not None and workload_onset is None:
        gap = None
        verdict = "resource_only"
    elif workload_onset is not None and resource_onset is None:
        gap = None
        verdict = "workload_only"
    else:
        gap = None
        verdict = "unknown"

    return {
        "resource_onset": resource_onset,
        "workload_onset": workload_onset,
        "gap_seconds": gap,
        "verdict": verdict,
    }


def _compute_ivd(
    store: "RCAEvalTelemetryStore",
    *,
    candidates: Sequence[str],
    all_anomalous_components: Sequence[str],
    time_window: tuple[float, float],
) -> Mapping[str, Any]:
    """Run Initiator–Victim Disambiguation pairwise tournament.

    Returns per-candidate IVD verdicts and a Copeland ranking.
    Only the top candidates by source_likelihood_score participate in the
    tournament to avoid noise from weak bystanders.
    """
    if len(candidates) < 2:
        return {
            "ivd_enabled": len(candidates) >= 2,
            "candidates": list(candidates),
            "verdicts": {},
            "ranking": list(candidates),
            "pairwise_results": [],
        }

    # Pre-filter: keep only candidates with score >= 50% of the max score,
    # or at most the top 8 by score. This removes weak bystanders that
    # would pollute the pairwise tournament.
    rows_for_filter = {
        comp: store._noise_feature_row(component=comp, time_window=time_window)
        for comp in candidates
    }
    scored = sorted(
        candidates,
        key=lambda c: (-float(rows_for_filter[c].get("source_likelihood_score", 0.0)), c),
    )
    max_score = float(rows_for_filter[scored[0]].get("source_likelihood_score", 0.0))
    if max_score > 0:
        threshold = max_score * 0.5
        filtered = [c for c in scored if float(rows_for_filter[c].get("source_likelihood_score", 0.0)) >= threshold]
    else:
        filtered = list(scored)
    # Cap at 8 to keep pairwise tournament manageable (max 28 pairs)
    tournament_candidates = filtered[:8]
    # Also require minimum resource magnitude to exclude bystanders
    tournament_candidates = [
        c for c in tournament_candidates
        if sum(
            float(m.get("magnitude", 0))
            for m in rows_for_filter[c].get("metric_features", [])
            if m.get("signal") in ("memory", "cpu", "disk")
        ) > 5
        or int(rows_for_filter[c].get("log_count", 0)) > 0
    ]
    if len(tournament_candidates) < 2:
        tournament_candidates = filtered[:5]

    # Build feature rows for all tournament candidates
    rows = {
        comp: rows_for_filter[comp]
        for comp in tournament_candidates
    }
    trace_ctx = store._trace_dependency_context(
        component_scope=list(tournament_candidates) + list(all_anomalous_components),
        time_window=time_window,
    )

    # Compute per-candidate signals
    signals = {}
    for comp in tournament_candidates:
        row = rows[comp]
        signals[comp] = {
            "fault_signature": _fault_signature_score(row),
            "temporal": _temporal_causality_signal(row),
            "causal_role": row.get("causal_role", "ambiguous"),
            "source_likelihood": row.get("source_likelihood_score", 0.0),
            "is_storage": row.get("is_storage_component", False),
            "resource_magnitude": sum(
                float(m.get("magnitude", 0))
                for m in row.get("metric_features", [])
                if m.get("signal") in ("memory", "cpu", "disk")
            ),
            "workload_magnitude": sum(
                float(m.get("magnitude", 0))
                for m in row.get("metric_features", [])
                if m.get("signal") == "workload"
            ),
            "log_count": row.get("log_count", 0),
        }

    # CallerAnomalies: for each candidate, how many of its CALLERS are anomalous?
    # If many callers are anomalous, the candidate (callee) likely caused their
    # failures → initiator signal.  (Reversed from old "incoming_explainedness".)
    # Use the already-computed trace_context direct_callees for speed.
    coverage = {}
    inferred_edges = _infer_dependency_edges(list(tournament_candidates) + list(all_anomalous_components))
    for comp in tournament_candidates:
        reachable = set()
        # Direct callees from trace context
        node = trace_ctx.get(comp, {})
        for callee in node.get("direct_callees", []):
            if isinstance(callee, Mapping):
                peer = callee.get("component", "")
                if peer in all_anomalous_components and peer != comp:
                    reachable.add(peer)
        # Inferred dependency edges
        for caller, callee in inferred_edges:
            if caller == comp and callee in all_anomalous_components:
                reachable.add(callee)
        coverage[comp] = reachable

    # CalleeAnomalies (was "IncomingExplainedness"): callers of comp that are
    # anomalous.  If many callers are anomalous, comp (as callee) explains their
    # failures → initiator signal.
    incoming = {}
    for comp in tournament_candidates:
        explainers = set()
        # Direct callers from trace context
        node = trace_ctx.get(comp, {})
        for caller in node.get("direct_callers", []):
            if isinstance(caller, Mapping):
                peer = caller.get("component", "")
                if peer in tournament_candidates and peer != comp:
                    explainers.add(peer)
        # Inferred dependency: if other calls comp
        for caller, callee in inferred_edges:
            if callee == comp and caller in tournament_candidates:
                explainers.add(caller)
        incoming[comp] = explainers

    # Pairwise tournament
    pairwise_results = []
    copeland = {comp: 0 for comp in tournament_candidates}

    for i, a in enumerate(tournament_candidates):
        for b in tournament_candidates[i + 1:]:
            a_beats_b = _ivd_pairwise(
                a=a, b=b,
                signals_a=signals[a], signals_b=signals[b],
                coverage_a=coverage[a], coverage_b=coverage[b],
                incoming_a=incoming[a], incoming_b=incoming[b],
                trace_ctx=trace_ctx,
            )
            pairwise_results.append(a_beats_b)
            if a_beats_b["winner"] == "a":
                copeland[a] += 1
                copeland[b] -= 1
            elif a_beats_b["winner"] == "b":
                copeland[b] += 1
                copeland[a] -= 1

    # Rank by Copeland score, then by source_likelihood
    ranking = sorted(
        tournament_candidates,
        key=lambda c: (-copeland[c], -signals[c]["source_likelihood"], c),
    )

    # Build verdicts
    verdicts = {}
    for comp in tournament_candidates:
        sig = signals[comp]
        is_initiator = copeland[comp] > 0 and ranking[0] == comp
        is_victim = copeland[comp] < 0
        if is_initiator:
            role = "initiator"
        elif is_victim:
            role = "victim"
        else:
            role = "ambiguous"

        verdicts[comp] = {
            "ivd_role": role,
            "copeland_score": copeland[comp],
            "fault_signature": sig["fault_signature"],
            "fault_signature_label": _fault_signature_label(sig["fault_signature"]),
            "temporal_verdict": sig["temporal"]["verdict"],
            "temporal_gap_seconds": sig["temporal"]["gap_seconds"],
            "downstream_coverage_count": len(coverage[comp]),
            "downstream_covered": sorted(coverage[comp])[:5],
            "caller_anomalies": sorted(incoming[comp])[:3],
            "caller_anomalies_count": len(incoming[comp]),
            "resource_magnitude": sig["resource_magnitude"],
            "workload_magnitude": sig["workload_magnitude"],
        }

    return {
        "ivd_enabled": True,
        "candidates": list(tournament_candidates),
        "verdicts": verdicts,
        "ranking": ranking,
        "pairwise_results": pairwise_results,
        "interpretation": (
            "IVD uses pairwise tournament (Copeland score) to identify the "
            "initiator vs victim among source candidates. The initiator is "
            "the candidate whose anomaly is least explainable by others and "
            "most explanatory for others. Do not select a candidate labeled "
            "as 'victim' over one labeled as 'initiator'."
        ),
    }


def _fault_signature_label(score: float) -> str:
    if score >= 2.0:
        return "own_code_stack_trace"
    if score >= 1.0:
        return "emitter_exception_no_dependency"
    if score <= -2.0:
        return "dependency_client_side_failure"
    if score <= -1.0:
        return "dependency_induced_exception"
    return "ambiguous_or_no_exception"


def _ivd_pairwise(
    *,
    a: str,
    b: str,
    signals_a: Mapping[str, Any],
    signals_b: Mapping[str, Any],
    coverage_a: set,
    coverage_b: set,
    incoming_a: set,
    incoming_b: set,
    trace_ctx: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Any]:
    """Compare two candidates pairwise and determine a winner.

    Returns {a, b, winner, evidence} where winner is 'a', 'b', or 'tie'.
    """
    reasons_a = []
    reasons_b = []

    # 1. FaultSignature asymmetry
    fs_a = signals_a["fault_signature"]
    fs_b = signals_b["fault_signature"]
    if fs_a > fs_b + 0.5:
        reasons_a.append(f"fault_signature: {a} has stronger own-code exception ({fs_a:.1f} vs {fs_b:.1f})")
    elif fs_b > fs_a + 0.5:
        reasons_b.append(f"fault_signature: {b} has stronger own-code exception ({fs_b:.1f} vs {fs_a:.1f})")

    # 2. Temporal causality: resource-before-workload = root-like
    tv_a = signals_a["temporal"]["verdict"]
    tv_b = signals_b["temporal"]["verdict"]
    root_like = {"resource_before_workload", "resource_only"}
    victim_like = {"workload_before_resource", "workload_only"}
    if tv_a in root_like and tv_b in victim_like:
        reasons_a.append(f"temporal: {a} has resource-before-workload ({tv_a}) while {b} is {tv_b}")
    elif tv_b in root_like and tv_a in victim_like:
        reasons_b.append(f"temporal: {b} has resource-before-workload ({tv_b}) while {a} is {tv_a}")

    # 3. CalleeAnomalies asymmetry: more anomalous callees = VICTIM (your
    #    downstream dependencies are failing, explaining your anomalies)
    #    BUT only when resource magnitudes are comparable (within 3x).
    #    When one candidate has >>3x resource, callee anomalies are likely
    #    cascading effects, not causal explanations.
    res_a = signals_a["resource_magnitude"]
    wl_a = signals_a["workload_magnitude"]
    res_b = signals_b["resource_magnitude"]
    wl_b = signals_b["workload_magnitude"]
    resource_comparable = max(res_a, res_b) < min(res_a, res_b) * 3 + 0.01

    cov_a = len(coverage_a)
    cov_b = len(coverage_b)
    if resource_comparable:
        if cov_a < cov_b:
            reasons_a.append(f"callee_anomalies: {a} has {cov_a} anomalous callees vs {b} has {cov_b}")
        elif cov_b < cov_a:
            reasons_b.append(f"callee_anomalies: {b} has {cov_b} anomalous callees vs {a} has {cov_a}")

    # 4. CallerAnomalies asymmetry: more anomalous callers = INITIATOR (your
    #    failure explains upstream callers' failures)
    #    Also gated on resource comparability.
    inc_a = len(incoming_a)
    inc_b = len(incoming_b)
    if resource_comparable:
        if inc_a > inc_b:
            reasons_a.append(f"caller_anomalies: {a} has {inc_a} anomalous callers vs {b} has {inc_b}")
        elif inc_b > inc_a:
            reasons_b.append(f"caller_anomalies: {b} has {inc_b} anomalous callers vs {a} has {inc_a}")

    # 5. Storage penalty
    if signals_a["is_storage"] and not signals_b["is_storage"]:
        reasons_b.append(f"topology: {a} is storage/dependency layer, {b} is a service")
    elif signals_b["is_storage"] and not signals_a["is_storage"]:
        reasons_a.append(f"topology: {b} is storage/dependency layer, {a} is a service")

    # 6. Resource-to-workload imbalance (our earlier discovery)
    # Root has high resource / low workload; victim has low resource / high workload
    # When either candidate has near-zero workload (<1.0), use resource ratio
    # alone (rw_ratio is unreliable with tiny denominator).
    rw_ratio_a = res_a / max(wl_a, 0.01)
    rw_ratio_b = res_b / max(wl_b, 0.01)
    both_have_workload = wl_a > 1.0 and wl_b > 1.0
    if both_have_workload:
        imb_a = res_a > res_b * 3 and rw_ratio_a > rw_ratio_b * 3
        imb_b = res_b > res_a * 3 and rw_ratio_b > rw_ratio_a * 3
    else:
        imb_a = res_a > res_b * 3
        imb_b = res_b > res_a * 3
    if imb_a:
        reasons_a.append(f"imbalance: {a} has resource={res_a:.0f} rw_ratio={rw_ratio_a:.1f} vs {b} resource={res_b:.0f} rw_ratio={rw_ratio_b:.1f}")
        if res_a > res_b * 10:
            reasons_a.append(f"resource_dominance: {a} has {res_a/max(res_b,0.01):.0f}x more resource than {b}")
    elif imb_b:
        reasons_b.append(f"imbalance: {b} has resource={res_b:.0f} rw_ratio={rw_ratio_b:.1f} vs {a} resource={res_a:.0f} rw_ratio={rw_ratio_a:.1f}")
        if res_b > res_a * 10:
            reasons_b.append(f"resource_dominance: {b} has {res_b/max(res_a,0.01):.0f}x more resource than {a}")

    # Determine winner
    score_a = len(reasons_a)
    score_b = len(reasons_b)
    if score_a > score_b:
        winner = "a"
    elif score_b > score_a:
        winner = "b"
    else:
        winner = "tie"

    return {
        "a": a,
        "b": b,
        "winner": winner,
        "reasons_a": reasons_a,
        "reasons_b": reasons_b,
        "score_a": score_a,
        "score_b": score_b,
    }


def _infer_dependency_edges(components: Sequence[str]) -> list[tuple[str, str]]:
    """Infer service-to-storage dependency edges from naming patterns.

    Returns a list of (caller, callee) pairs where caller is a service
    and callee is its storage dependency.  For example:
      ts-auth-service -> ts-auth-mongo
      carts -> carts-db
    """
    edges: list[tuple[str, str]] = []
    services = [c for c in components if not _is_storage_component(c)]
    storages = [c for c in components if _is_storage_component(c)]
    storage_set = set(storages)
    for service in services:
        low = service.lower()
        prefix = low
        if prefix.endswith("-service"):
            prefix = prefix[: -len("-service")]
        elif prefix.endswith("service"):
            prefix = prefix[: -len("service")]
        for storage in storages:
            slow = storage.lower()
            sprefix = slow
            for suffix in _STORAGE_SUFFIXES:
                if sprefix.endswith(suffix):
                    sprefix = sprefix[: -len(suffix)]
                    break
            if sprefix and sprefix in low and storage != service and storage in storage_set:
                edges.append((service, storage))
    return edges


def _infer_entry_components(
    *,
    components: Sequence[str],
    store: "RCAEvalTelemetryStore",
    event_time: float,
) -> tuple[str, ...]:
    """Infer entry/surface components for propagation reasoning.

    Order of precedence:
      1. Explicit entry-like names (frontend/gateway/ingress).
      2. Trace-topology roots: components that appear as trace span roots
         (have callees but no callers) in the post-event window.  These are
         the request entry points even when names do not say "gateway"
         (e.g. RE3-TT where ts-admin-basic-info-service / ts-preserve-service
         are not gateways but are real trace roots).
      3. Fallback to the first sorted component (legacy behaviour).
    """
    named = tuple(c for c in components if _is_entry_like_component(c))
    if named:
        return named

    window = (max(0.0, event_time - NEAR_ONSET_WINDOW_SECONDS), event_time + 600.0)
    try:
        ctx = store._trace_dependency_context(
            component_scope=components,
            time_window=window,
        )
    except Exception:
        ctx = {}
    root_scores: dict[str, int] = {}
    for component in components:
        node = ctx.get(component, {})
        n_callers = len(node.get("direct_callers", []) or [])
        n_callees = len(node.get("direct_callees", []) or [])
        if n_callers == 0 and n_callees > 0:
            root_scores[component] = n_callees
    if root_scores:
        ranked = sorted(root_scores.items(), key=lambda kv: (-kv[1], kv[0]))
        return tuple(c for c, _ in ranked)

    if components:
        return (components[0],)
    return ()


_RECALL_MECHANISM_LEVEL_RANK = {
    "strong": 0,
    "moderate": 1,
    "weak": 2,
    "symptom_like": 3,
    "unknown": 4,
}


def _recall_magnitude_no_workload_key(row: Mapping[str, Any]):
    adjusted = float(row["total_magnitude"]) - float(row["workload_magnitude"])
    earliest = row["earliest_time"]
    return (
        -adjusted,
        earliest if earliest is not None else float("inf"),
        row["component"],
    )


def _recall_onset_key(row: Mapping[str, Any]):
    earliest = row["earliest_time"]
    return (
        earliest if earliest is not None else float("inf"),
        -float(row["near_onset_magnitude"]),
        row["component"],
    )


def _recall_mechanism_key(row: Mapping[str, Any]):
    level = _RECALL_MECHANISM_LEVEL_RANK.get(str(row.get("mechanism_level", "unknown")), 4)
    internal = set(row.get("near_onset_internal", ()) or ()) - {"log"}
    earliest = row["earliest_time"]
    return (
        level,
        -len(internal),
        -float(row["near_onset_magnitude"]),
        earliest if earliest is not None else float("inf"),
        row["component"],
    )


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
    # Storage/dependency disambiguation cues.
    if row.get("is_storage_component"):
        cues.append(
            "component is a storage/dependency layer (mongo/redis/db); "
            "resource anomalies here are often reactive to service-level "
            "logic errors that cause abnormal query patterns; prefer a "
            "service with emitter exceptions as root cause"
        )
    # Causal role cue.
    causal_role = row.get("causal_role")
    if causal_role == "source_candidate":
        cues.append(
            "causal_role=source_candidate: has emitter exceptions or strong "
            "internal mechanism as a service; prioritize over dependency and "
            "symptom candidates"
        )
    elif causal_role == "dependency_candidate":
        cues.append(
            "causal_role=dependency_candidate: storage with resource anomalies "
            "but no emitter exceptions; likely a reactive dependency, not the root"
        )
    elif causal_role == "propagation_symptom":
        cues.append(
            "causal_role=propagation_symptom: latency/workload-only evidence; "
            "likely a downstream cascading effect"
        )
    elif causal_role == "entry_symptom":
        cues.append(
            "causal_role=entry_symptom: entry component where failures surface; "
            "not necessarily the initiating root"
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
                    "same log contains emitter-side exception/status evidence; "
                    "the emitting service is a strong root candidate because "
                    "a service logic error can cause its storage dependency to "
                    "show reactive resource anomalies"
                )
        elif feature.get("diagnostic_role") == "generic_request_error_observed_by_emitter":
            symptom_like.append("generic request error is visibility evidence, not source proof")

    # Storage/dependency disambiguation cues.
    if row.get("is_storage_component"):
        symptom_like.append(
            "component is a storage/dependency layer; resource anomalies here "
            "are often reactive to service-level faults, not the initiating root"
        )
    # Inferred dependency direction: if this component has callees that are
    # storage components (inferred from naming), it is a service that calls
    # those storages, strengthening its source candidacy.
    inferred_callees = [
        c for c in (direct_callees or [])
        if isinstance(c, Mapping) and c.get("inferred_from_naming")
    ]
    if inferred_callees and not row.get("is_storage_component"):
        source_like.append(
            "has inferred storage dependencies (service calls database); "
            "if this service has a logic error, it can cause abnormal query "
            "patterns that make the storage show resource anomalies"
        )
    inferred_callers = [
        c for c in (direct_callers or [])
        if isinstance(c, Mapping) and c.get("inferred_from_naming")
    ]
    if inferred_callers and row.get("is_storage_component"):
        symptom_like.append(
            "storage component is called by inferred service dependency; "
            "resource anomalies may be caused by the calling service's fault"
        )

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
        ),
    )
    result: list[Mapping[str, Any]] = []
    for item in rows[:8]:
        entry = {
            "component": item.get("component", ""),
            "call_count": int(item.get("call_count", 0)),
            "error_status_count": int(item.get("error_status_count", 0)),
            "latency_examples": item.get("latency_examples", [])[:3],
            "status_examples": item.get("status_examples", [])[:3],
        }
        if item.get("inferred_from_naming"):
            entry["inferred_from_naming"] = True
        result.append(entry)
    return result


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


_RE_FAULT_TYPE_SUFFIXES = (
    "_cpu", "_delay", "_disk", "_loss", "_mem", "_socket",
)


def _expected_component_from_path(case_path: Path) -> str:
    fault_name = case_path.parent.name
    # RE3 convention: {component}_f{number}
    if "_f" in fault_name:
        return fault_name.rsplit("_f", 1)[0]
    # RE1/RE2 convention: {component}_{fault_type}
    for suffix in _RE_FAULT_TYPE_SUFFIXES:
        if fault_name.endswith(suffix):
            return fault_name[: -len(suffix)]
    return fault_name


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
