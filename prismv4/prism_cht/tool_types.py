"""Fact-only tool types for PRISM-CHT.

Tools return immutable *FactToolResult* records that describe
observable telemetry — never root-cause judgements, scores, or
probabilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence, Tuple

from .canonical import deep_freeze

_FORBIDDEN_KEYS: set[str] = {
    "root_cause",
    "winner",
    "verdict",
    "posterior",
    "probability",
    "confidence",
    "confidence_score",
    "likelihood",
    "rank",
    "ranking",
    "score",
    "root_score",
    "support",
    "support_score",
    "against",
    "against_score",
    "net_score",
    "component_score",
}


def validate_fact_only_payload(value: Any) -> None:
    """Recursively scan *value* and raise ValueError if any forbidden key appears.

    Forbidden keys are matched case-insensitively after stripping
    whitespace.  Nested ``Mapping`` and sequences (list, tuple) are
    scanned depth-first.  The intent is to guard ``observation`` and
    ``provenance`` dicts so tools cannot leak root-cause scores.
    """
    _validate_recursive(value, path="")


def _validate_recursive(value: Any, path: str) -> None:
    if isinstance(value, Mapping):
        for key, val in value.items():
            key_str = str(key).strip().lower()
            if key_str in _FORBIDDEN_KEYS:
                raise ValueError(
                    f"Forbidden key '{key}' found at {path or '<root>'}; "
                    f"fact-only payloads must not contain root-cause scores"
                )
            _validate_recursive(val, f"{path}.{key}" if path else str(key))
    elif isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            _validate_recursive(item, f"{path}[{i}]" if path else f"[{i}]")


class FactTool(Protocol):
    """Protocol for a fact-only investigation tool.

    Tools must have a unique *name* and an ``execute`` method that
    returns a ``FactToolResult``.  They may only observe facts; they
    must not compute scores or judge root causes.
    """

    name: str

    def execute(
        self,
        *,
        args: Mapping[str, Any],
        store: "TelemetryStore",
    ) -> "FactToolResult":
        ...


@dataclass(frozen=True)
class FactToolResult:
    """Immutable factual observation returned by a PRISM-CHT tool.

    Carries no evidence_id, no query_signature, and no root-cause
    scores.  Those identifiers are assigned by the executor when the
    result is persisted into an ``EvidenceGraph``.
    """

    modality: str
    component_scope: Tuple[str, ...]
    time_window: Tuple[float, float]
    observation: Mapping[str, Any]
    provenance: Mapping[str, Any]
    missing_fields: Tuple[str, ...] = ()
    reliability_note: str = ""

    def __post_init__(self):
        if not self.modality or not self.modality.strip():
            raise ValueError("modality must be non-empty")

        # Normalize and validate component_scope
        scope = tuple(sorted(set(self.component_scope)))
        if not scope:
            raise ValueError("component_scope must not be empty")
        object.__setattr__(self, "component_scope", scope)

        # Validate time_window
        if len(self.time_window) != 2:
            raise ValueError(
                f"time_window must have exactly 2 elements, "
                f"got {len(self.time_window)}"
            )
        start, end = self.time_window
        if start > end:
            raise ValueError(
                f"time_window start ({start}) must be <= end ({end})"
            )

        # observation must be Mapping
        if self.observation is None:
            raise ValueError("observation must not be None")
        if not isinstance(self.observation, Mapping):
            raise ValueError(
                f"observation must be a Mapping, "
                f"got {type(self.observation).__name__}"
            )

        # provenance must be Mapping
        if self.provenance is None:
            raise ValueError("provenance must not be None")
        if not isinstance(self.provenance, Mapping):
            raise ValueError(
                f"provenance must be a Mapping, "
                f"got {type(self.provenance).__name__}"
            )

        # Normalize missing_fields
        object.__setattr__(
            self,
            "missing_fields",
            tuple(sorted(set(self.missing_fields))),
        )

        # Guard against forbidden keys in observation and provenance
        validate_fact_only_payload(self.observation)
        validate_fact_only_payload(self.provenance)

        # Deep-freeze nested containers
        object.__setattr__(self, "observation", deep_freeze(self.observation))
        object.__setattr__(self, "provenance", deep_freeze(self.provenance))
