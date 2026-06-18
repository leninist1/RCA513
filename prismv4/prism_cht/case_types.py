"""Dataset-neutral case model for PRISM-CHT.

These types describe an RCA case without tying the tournament to a
particular dataset layout.  Evaluation labels are intentionally kept
outside ``prism_cht``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class ObservedComponent:
    """Perceived component-level observation used to seed hypotheses."""

    component: str
    reason_family: str
    first_seen: float
    magnitude: float
    signals: tuple[str, ...]
    symptoms: tuple[str, ...]

    def __post_init__(self):
        if not self.component or not self.component.strip():
            raise ValueError("component must be non-empty")
        if not self.reason_family or not self.reason_family.strip():
            raise ValueError("reason_family must be non-empty")
        if self.magnitude < 0:
            raise ValueError("magnitude must be non-negative")
        if not self.signals:
            raise ValueError("signals must contain at least one item")


@dataclass(frozen=True)
class GenericRCACase:
    """Dataset-neutral input bundle for a CHT diagnosis run."""

    case_id: str
    dataset_name: str
    system_name: str
    event_time: float
    components: tuple[str, ...]
    entry_components: tuple[str, ...]
    observations: tuple[ObservedComponent, ...]
    metadata: Mapping[str, str] | None = None

    def __post_init__(self):
        if not self.case_id or not self.case_id.strip():
            raise ValueError("case_id must be non-empty")
        if not self.dataset_name or not self.dataset_name.strip():
            raise ValueError("dataset_name must be non-empty")
        if not self.system_name or not self.system_name.strip():
            raise ValueError("system_name must be non-empty")
        if not self.components:
            raise ValueError("components must contain at least one item")
        if len(set(self.components)) != len(self.components):
            raise ValueError("components contains duplicate entries")
        component_set = set(self.components)
        for entry in self.entry_components:
            if entry not in component_set:
                raise ValueError(
                    f"entry component '{entry}' is not present in components"
                )
        for obs in self.observations:
            if obs.component not in component_set:
                raise ValueError(
                    f"observation component '{obs.component}' is not present "
                    "in components"
                )
