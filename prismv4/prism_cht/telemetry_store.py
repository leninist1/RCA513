"""Telemetry store protocol and mock implementation for PRISM-CHT.

Provides data-classes for onset observations and trace paths, a
``TelemetryStore`` protocol that tools depend on, and a
deterministic ``MockTelemetryStore`` for testing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence, Tuple

from .canonical import deep_freeze


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OnsetObservation:
    """A single observed signal onset for one component."""

    component: str
    signal: str
    onset_time: float
    source: str


@dataclass(frozen=True)
class TraceHop:
    """A single hop (edge) in a trace path."""

    source_component: str
    target_component: str
    timestamp: float
    latency_ms: float
    status: str


@dataclass(frozen=True)
class TracePath:
    """An ordered sequence of trace hops forming a path."""

    hops: Tuple[TraceHop, ...]


# ---------------------------------------------------------------------------
# TelemetryStore protocol
# ---------------------------------------------------------------------------


class TelemetryStore(Protocol):
    """Protocol for reading raw telemetry observations.

    Tools depend on this protocol, not on concrete store
    implementations.
    """

    def get_onset_observations(
        self,
        *,
        component_scope: Sequence[str],
        signal_scope: Sequence[str],
        time_window: Tuple[float, float],
    ) -> Tuple[OnsetObservation, ...]:
        ...

    def find_trace_paths(
        self,
        *,
        source_component: str,
        target_component: str,
        time_window: Tuple[float, float],
        max_hops: int,
        max_paths: int,
    ) -> Tuple[TracePath, ...]:
        ...

    def retrieve_records(
        self,
        *,
        modality: str,
        component_scope: Sequence[str],
        time_window: Tuple[float, float],
        limit: int,
    ) -> Tuple[Mapping[str, Any], ...]:
        ...


# ---------------------------------------------------------------------------
# MockTelemetryStore
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MockTelemetryStore:
    """Deterministic, in-memory telemetry store for testing.

    Parameters
    ----------
    onset_observations:
        Pre-loaded onset observations.
    trace_paths:
        Pre-loaded trace paths.
    raw_records:
        Pre-loaded raw records.  Each must contain at least the keys
        ``modality``, ``component``, ``timestamp``, and ``payload``.
    """

    onset_observations: Sequence[OnsetObservation] = ()
    trace_paths: Sequence[TracePath] = ()
    raw_records: Sequence[Mapping[str, Any]] = ()

    def __post_init__(self):
        # Deep-freeze all construction data
        object.__setattr__(
            self,
            "onset_observations",
            tuple(deep_freeze(o) for o in self.onset_observations),
        )
        object.__setattr__(
            self,
            "trace_paths",
            tuple(deep_freeze(p) for p in self.trace_paths),
        )

        frozen_records = []
        for r in self.raw_records:
            _validate_raw_record(r)
            frozen_records.append(deep_freeze(r))
        object.__setattr__(self, "raw_records", tuple(frozen_records))

        # Call counters for testing repeat-query protection
        object.__setattr__(self, "_onset_call_count", 0)
        object.__setattr__(self, "_trace_call_count", 0)
        object.__setattr__(self, "_records_call_count", 0)

    # -- query methods ------------------------------------------------------

    def get_onset_observations(
        self,
        *,
        component_scope: Sequence[str],
        signal_scope: Sequence[str],
        time_window: Tuple[float, float],
    ) -> Tuple[OnsetObservation, ...]:
        object.__setattr__(
            self, "_onset_call_count", self._onset_call_count + 1
        )
        comp_set = set(component_scope)
        sig_set = set(signal_scope)
        t0, t1 = time_window

        results = [
            o
            for o in self.onset_observations
            if o.component in comp_set
            and o.signal in sig_set
            and t0 <= o.onset_time <= t1
        ]
        results.sort(key=lambda o: (o.onset_time, o.component, o.signal))
        return tuple(results)

    def find_trace_paths(
        self,
        *,
        source_component: str,
        target_component: str,
        time_window: Tuple[float, float],
        max_hops: int,
        max_paths: int,
    ) -> Tuple[TracePath, ...]:
        object.__setattr__(
            self, "_trace_call_count", self._trace_call_count + 1
        )
        t0, t1 = time_window

        matching = []
        for path in self.trace_paths:
            if len(path.hops) > max_hops:
                continue
            if len(matching) >= max_paths:
                break

            # Path must start with source_component and end with target_component
            if (
                path.hops
                and path.hops[0].source_component == source_component
                and path.hops[-1].target_component == target_component
            ):
                # All hop timestamps must be within the window
                if all(t0 <= h.timestamp <= t1 for h in path.hops):
                    matching.append(path)

        # Deterministic sort: by number of hops, then by first hop timestamp
        matching.sort(
            key=lambda p: (len(p.hops), p.hops[0].timestamp if p.hops else 0.0)
        )
        return tuple(matching)

    def retrieve_records(
        self,
        *,
        modality: str,
        component_scope: Sequence[str],
        time_window: Tuple[float, float],
        limit: int,
    ) -> Tuple[Mapping[str, Any], ...]:
        object.__setattr__(
            self, "_records_call_count", self._records_call_count + 1
        )
        comp_set = set(component_scope)
        t0, t1 = time_window

        results = [
            r
            for r in self.raw_records
            if r.get("modality") == modality
            and r.get("component") in comp_set
            and t0 <= r.get("timestamp", float("-inf")) <= t1
        ]
        results.sort(key=lambda r: (r.get("timestamp", 0.0), r.get("component", "")))
        return tuple(results[:limit])

    # -- accessors for testing ----------------------------------------------

    def onset_call_count(self) -> int:
        return self._onset_call_count

    def trace_call_count(self) -> int:
        return self._trace_call_count

    def records_call_count(self) -> int:
        return self._records_call_count


def _validate_raw_record(record: Mapping[str, Any]) -> None:
    """Validate that *record* contains required keys."""
    required = {"modality", "component", "timestamp", "payload"}
    missing = required - set(record.keys())
    if missing:
        raise ValueError(
            f"raw_record missing required keys: {sorted(missing)}"
        )
    if not isinstance(record["payload"], Mapping):
        raise ValueError(
            f"raw_record 'payload' must be a Mapping, "
            f"got {type(record['payload']).__name__}"
        )
