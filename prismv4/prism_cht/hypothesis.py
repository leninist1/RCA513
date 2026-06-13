"""Falsifiable causal hypothesis for PRISM-CHT tournament.

Each hypothesis declares predicted observations and falsifiers;
only hypotheses with falsifiers may be activated.  State transitions
are governed by a strict lifecycle machine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple


class HypothesisStatus(str, Enum):
    DRAFT = "draft"
    ACTIVE = "active"
    SUPPORTED = "supported"
    WEAKENED = "weakened"
    REFUTED = "refuted"
    SURVIVED = "survived"
    FINAL = "final"


_ALLOWED_TRANSITIONS: dict[HypothesisStatus, set[HypothesisStatus]] = {
    HypothesisStatus.DRAFT: {
        HypothesisStatus.ACTIVE,
    },
    HypothesisStatus.ACTIVE: {
        HypothesisStatus.SUPPORTED,
        HypothesisStatus.WEAKENED,
        HypothesisStatus.REFUTED,
    },
    HypothesisStatus.SUPPORTED: {
        HypothesisStatus.WEAKENED,
        HypothesisStatus.REFUTED,
        HypothesisStatus.SURVIVED,
    },
    HypothesisStatus.WEAKENED: {
        HypothesisStatus.SUPPORTED,
        HypothesisStatus.REFUTED,
    },
    HypothesisStatus.SURVIVED: {
        HypothesisStatus.FINAL,
    },
    HypothesisStatus.REFUTED: set(),
    HypothesisStatus.FINAL: set(),
}


@dataclass
class CausalHypothesis:
    hypothesis_id: str
    root_component: str
    reason_family: str
    onset_interval: Tuple[float, float]

    local_trigger: str
    propagation_path: List[str]
    explained_symptoms: List[str]

    predicted_observations: List[str]
    falsifiers: List[str]

    supporting_evidence_ids: List[str] = field(default_factory=list)
    contradicting_evidence_ids: List[str] = field(default_factory=list)
    unresolved_questions: List[str] = field(default_factory=list)

    status: HypothesisStatus = HypothesisStatus.DRAFT

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_for_activation(self) -> List[str]:
        """Return a list of validation errors (empty = valid)."""
        errors: List[str] = []

        if not self.hypothesis_id or not self.hypothesis_id.strip():
            errors.append("hypothesis_id is empty")
        if not self.root_component or not self.root_component.strip():
            errors.append("root_component is empty")
        if not self.reason_family or not self.reason_family.strip():
            errors.append("reason_family is empty")

        if len(self.onset_interval) != 2:
            errors.append(
                f"onset_interval must have exactly 2 elements, got {len(self.onset_interval)}"
            )
        else:
            start, end = self.onset_interval
            if start > end:
                errors.append(
                    f"onset_interval start ({start}) must be <= end ({end})"
                )

        if not self.local_trigger or not self.local_trigger.strip():
            errors.append("local_trigger is empty")

        if not self.propagation_path:
            errors.append("propagation_path must contain at least one component")
        elif any(not c or not c.strip() for c in self.propagation_path):
            errors.append("propagation_path contains empty component name")

        if not self.predicted_observations:
            errors.append("predicted_observations must contain at least one item")

        if not self.falsifiers:
            errors.append(
                "falsifiers must contain at least one item; "
                "hypothesis without falsifiers is untestable"
            )
        elif all(not item or not item.strip() for item in self.falsifiers):
            errors.append(
                "falsifiers must contain at least one non-empty item; "
                "hypothesis without falsifiers is untestable"
            )

        return errors

    # ------------------------------------------------------------------
    # State machine
    # ------------------------------------------------------------------

    def activate(self) -> None:
        errors = self.validate_for_activation()
        if errors:
            raise ValueError(
                f"Cannot activate hypothesis '{self.hypothesis_id}': "
                + "; ".join(errors)
            )
        if self.status != HypothesisStatus.DRAFT:
            raise ValueError(
                f"Cannot activate hypothesis '{self.hypothesis_id}': "
                f"current status is {self.status.value}, expected DRAFT"
            )
        self.status = HypothesisStatus.ACTIVE

    def validate_transition_to(self, new_status: HypothesisStatus) -> None:
        """Validate that a transition to *new_status* is legal, without mutating."""
        if not isinstance(new_status, HypothesisStatus):
            raise ValueError(
                f"new_status must be a HypothesisStatus, got {type(new_status).__name__}"
            )
        allowed = _ALLOWED_TRANSITIONS.get(self.status, set())
        if new_status not in allowed:
            raise ValueError(
                f"Cannot transition hypothesis '{self.hypothesis_id}' "
                f"from {self.status.value} to {new_status.value}; "
                f"allowed transitions: {sorted(s.value for s in allowed)}"
            )

    def transition_to(self, new_status: HypothesisStatus) -> None:
        self.validate_transition_to(new_status)
        self.status = new_status

    # ------------------------------------------------------------------
    # Evidence tracking
    # ------------------------------------------------------------------

    def attach_support(self, evidence_id: str) -> None:
        if not evidence_id or not evidence_id.strip():
            raise ValueError("evidence_id must be non-empty")
        if evidence_id in self.supporting_evidence_ids:
            raise ValueError(
                f"Duplicate supporting evidence_id '{evidence_id}' "
                f"for hypothesis '{self.hypothesis_id}'"
            )
        self.supporting_evidence_ids.append(evidence_id)

    def attach_contradiction(self, evidence_id: str) -> None:
        if not evidence_id or not evidence_id.strip():
            raise ValueError("evidence_id must be non-empty")
        if evidence_id in self.contradicting_evidence_ids:
            raise ValueError(
                f"Duplicate contradicting evidence_id '{evidence_id}' "
                f"for hypothesis '{self.hypothesis_id}'"
            )
        self.contradicting_evidence_ids.append(evidence_id)
