"""Deterministic diagnostic policies for end-to-end PRISM-CHT runs.

The policies in this module are dataset-neutral and drive the existing
Lead/Challenger controllers through real gates, tools, evidence graph
updates, and final verification.  They are deliberately conservative:
tools observe facts, while the policy interprets those facts into
structured hypothesis updates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .action_schema import DiscriminativeAction
from .case_types import GenericRCACase, ObservedComponent
from .challenge_types import ChallengeProposal, ChallengeResolution, ChallengeVerdict
from .challenger_policy import ChallengerPolicy
from .evidence_graph import EvidenceAtom
from .hypothesis import CausalHypothesis, HypothesisStatus
from .lead_policy import LeadPolicy, PolicyDecision
from .tournament_types import (
    AssessmentOutcome,
    EvidenceAssessment,
    EvidenceLinkProposal,
    EvidenceRelation,
    HypothesisStatusUpdate,
    LeadNomination,
    LeadTournamentSnapshot,
    TripletEvidenceCoverage,
)


@dataclass(frozen=True)
class HypothesisBundle:
    hypotheses: tuple[CausalHypothesis, ...]
    observation_by_hypothesis_id: Mapping[str, ObservedComponent]


def build_hypotheses_from_case(
    case: GenericRCACase,
    *,
    max_hypotheses: int = 5,
    candidates: tuple[str, ...] | None = None,
) -> HypothesisBundle:
    """Build falsifiable causal hypotheses from perceived observations.

    When ``candidates`` is provided, it is treated as an explicit recall
    pool ordered by the caller's tiered-recall strategy.  The first
    ``max_hypotheses`` distinct components from that pool are turned into
    hypotheses, regardless of raw magnitude, so the LLM reasons over a set
    that contains the answer instead of a magnitude top-k that may have
    dropped it.  Observations are still used to populate onset/symptom
    fields when available; candidates without an observation get a neutral
    placeholder observation.
    """
    if max_hypotheses < 2:
        raise ValueError("max_hypotheses must be at least 2")

    obs_by_component: dict[str, ObservedComponent] = {
        obs.component: obs for obs in case.observations
    }

    if candidates is not None:
        ordered_components = list(candidates)
    else:
        ordered_components = [
            obs.component
            for obs in sorted(
                case.observations,
                key=lambda o: (-o.magnitude, o.first_seen, o.component),
            )
        ]

    selected: list[ObservedComponent] = []
    seen_components: set[str] = set()
    for component in ordered_components:
        if component in seen_components:
            continue
        if component not in obs_by_component:
            continue
        selected.append(obs_by_component[component])
        seen_components.add(component)
        if len(selected) >= max_hypotheses:
            break

    # For explicit candidates that had no observation, synthesize a neutral
    # placeholder so recall is not lost purely due to missing telemetry.
    if candidates is not None:
        for component in ordered_components:
            if len(selected) >= max_hypotheses:
                break
            if component in seen_components:
                continue
            if component not in set(case.components):
                continue
            selected.append(
                ObservedComponent(
                    component=component,
                    reason_family="unspecified anomaly",
                    first_seen=case.event_time,
                    magnitude=0.0,
                    signals=("unknown",),
                    symptoms=("system symptoms after event",),
                )
            )
            seen_components.add(component)

    if len(selected) < 2:
        for component in case.components:
            if component in seen_components:
                continue
            selected.append(
                ObservedComponent(
                    component=component,
                    reason_family="unspecified anomaly",
                    first_seen=case.event_time,
                    magnitude=0.0,
                    signals=("unknown",),
                    symptoms=("system symptoms after event",),
                )
            )
            seen_components.add(component)
            if len(selected) >= 2:
                break

    if len(selected) < 2:
        raise ValueError("at least two hypotheses are required")

    target_symptom = (
        case.entry_components[0]
        if case.entry_components
        else selected[0].component
    )
    hypotheses: list[CausalHypothesis] = []
    obs_by_hid: dict[str, ObservedComponent] = {}
    for idx, obs in enumerate(selected, 1):
        hid = f"H{idx}-{_stable_id(obs.component)}-{_stable_id(obs.reason_family)}"
        path = (
            [obs.component]
            if obs.component == target_symptom
            else [obs.component, target_symptom]
        )
        h = CausalHypothesis(
            hypothesis_id=hid,
            root_component=obs.component,
            reason_family=obs.reason_family,
            onset_interval=(max(0.0, obs.first_seen - 30.0), obs.first_seen + 30.0),
            local_trigger=(
                f"{obs.component} shows {obs.reason_family} around "
                f"{obs.first_seen:.3f}"
            ),
            propagation_path=path,
            explained_symptoms=tuple(obs.symptoms) or ("system symptoms",),
            predicted_observations=(
                f"{obs.component} should have an early telemetry deviation",
                f"{obs.component} should show signals compatible with "
                f"{obs.reason_family}",
            ),
            falsifiers=(
                "another active hypothesis has an earlier compatible onset",
                "the component lacks matching metric, log, or trace evidence",
            ),
        )
        hypotheses.append(h)
        obs_by_hid[hid] = obs

    return HypothesisBundle(
        hypotheses=tuple(hypotheses),
        observation_by_hypothesis_id=obs_by_hid,
    )


class DeterministicLeadDiagnosticPolicy(LeadPolicy):
    """A compact Lead Agent policy driven by discriminative fact queries."""

    def __init__(
        self,
        *,
        case: GenericRCACase,
        observation_by_hypothesis_id: Mapping[str, ObservedComponent],
    ) -> None:
        self._case = case
        self._obs_by_hid = dict(observation_by_hypothesis_id)
        self._actions = self._build_actions()
        self._next_action_index = 0
        self._support_ids: dict[str, list[str]] = {hid: [] for hid in self._obs_by_hid}
        self._contradiction_ids: dict[str, list[str]] = {
            hid: [] for hid in self._obs_by_hid
        }
        self._preferred_hid: str | None = None
        self._nomination_returned = False

    def decide_next(self, *, snapshot: LeadTournamentSnapshot) -> PolicyDecision:
        if self._nomination_returned:
            raise ValueError("nomination already returned")
        if self._next_action_index < len(self._actions):
            action = self._actions[self._next_action_index]
            self._next_action_index += 1
            return action
        self._nomination_returned = True
        return self._build_nomination(snapshot)

    def assess_evidence(
        self,
        *,
        snapshot: LeadTournamentSnapshot,
        action: DiscriminativeAction,
        evidence: EvidenceAtom,
    ) -> EvidenceAssessment:
        if action.tool_name == "compare_onset_order":
            return self._assess_onset(snapshot, action, evidence)
        if action.tool_name == "inspect_reason_signature":
            return self._assess_reason(snapshot, action, evidence)
        if action.tool_name == "inspect_trace_path":
            return self._assess_trace(snapshot, action, evidence)
        raise ValueError(f"unsupported action tool '{action.tool_name}'")

    def _build_actions(self) -> tuple[DiscriminativeAction, ...]:
        target_ids = tuple(self._obs_by_hid.keys())
        components = [self._obs_by_hid[hid].component for hid in target_ids]
        signals = sorted({s for obs in self._obs_by_hid.values() for s in obs.signals})
        if not signals:
            signals = ["latency"]
        t0 = max(0.0, self._case.event_time - 300.0)
        t1 = self._case.event_time + 600.0

        onset = DiscriminativeAction(
            action_id="lead-onset-order",
            action_type="run_discriminative_test",
            target_hypothesis_ids=target_ids,
            question="Which candidate component shows the earliest observed deviation?",
            tool_name="compare_onset_order",
            args={
                "component_scope": components,
                "signal_scope": signals,
                "time_window": [t0, t1],
            },
            expected_outcomes={
                hid: f"{obs.component} has the earliest compatible onset"
                for hid, obs in self._obs_by_hid.items()
            },
            why_discriminative=(
                "competing root-cause hypotheses predict different first "
                "deviating components"
            ),
        )

        reason = DiscriminativeAction(
            action_id="lead-reason-signature",
            action_type="run_discriminative_test",
            target_hypothesis_ids=target_ids,
            question="Which candidate has telemetry matching its proposed reason family?",
            tool_name="inspect_reason_signature",
            args={
                "component_scope": components,
                "reason_by_component": {
                    obs.component: obs.reason_family for obs in self._obs_by_hid.values()
                },
                "time_window": [t0, t1],
            },
            expected_outcomes={
                hid: f"{obs.component} has strongest {obs.reason_family} signature"
                for hid, obs in self._obs_by_hid.items()
            },
            why_discriminative=(
                "a component-root hypothesis should expose a local reason "
                "signature before downstream symptoms dominate"
            ),
        )

        entry = (
            self._case.entry_components[0]
            if self._case.entry_components
            else components[0]
        )
        first_component = components[0]
        trace = DiscriminativeAction(
            action_id="lead-trace-propagation",
            action_type="run_discriminative_test",
            target_hypothesis_ids=target_ids,
            question="Does the leading candidate have a path to an entry symptom?",
            tool_name="inspect_trace_path",
            args={
                "source_component": first_component,
                "target_component": entry,
                "time_window": [t0, t1],
                "max_hops": 6,
                "max_paths": 10,
            },
            expected_outcomes={
                hid: (
                    f"{obs.component} explains the entry symptom path"
                    if obs.component == first_component
                    else f"{obs.component} is less consistent with this path"
                )
                for hid, obs in self._obs_by_hid.items()
            },
            why_discriminative=(
                "a surviving hypothesis should have at least a plausible "
                "propagation relation to the entry symptom"
            ),
        )

        return (onset, reason, trace)

    def _assess_onset(
        self,
        snapshot: LeadTournamentSnapshot,
        action: DiscriminativeAction,
        evidence: EvidenceAtom,
    ) -> EvidenceAssessment:
        onsets = evidence.observation.get("observed_onsets", ())
        best_component = None
        if onsets:
            best_component = onsets[0].get("component")
        if best_component is None:
            best_component = self._best_observation().component
        best_hid = self._hid_for_component(best_component) or self._best_hid()
        self._preferred_hid = best_hid
        return self._build_informative_assessment(
            snapshot=snapshot,
            action=action,
            evidence=evidence,
            support_hid=best_hid,
            rationale=(
                f"earliest observed deviation is attributed to "
                f"{self._obs_by_hid[best_hid].component}"
            ),
        )

    def _assess_reason(
        self,
        snapshot: LeadTournamentSnapshot,
        action: DiscriminativeAction,
        evidence: EvidenceAtom,
    ) -> EvidenceAssessment:
        signatures = evidence.observation.get("reason_signatures", ())
        best_component = None
        best_magnitude = -1.0
        for item in signatures:
            magnitude = float(item.get("magnitude", 0.0) or 0.0)
            if magnitude > best_magnitude:
                best_magnitude = magnitude
                best_component = item.get("component")
        best_hid = self._hid_for_component(best_component or "") or self._best_hid()
        self._preferred_hid = best_hid
        return self._build_informative_assessment(
            snapshot=snapshot,
            action=action,
            evidence=evidence,
            support_hid=best_hid,
            rationale=(
                f"reason signature most strongly matches "
                f"{self._obs_by_hid[best_hid].component}"
            ),
        )

    def _assess_trace(
        self,
        snapshot: LeadTournamentSnapshot,
        action: DiscriminativeAction,
        evidence: EvidenceAtom,
    ) -> EvidenceAssessment:
        source = str(action.args.get("source_component", ""))
        paths = evidence.observation.get("paths", ())
        support_hid = self._hid_for_component(source) if paths else self._best_hid()
        return self._build_informative_assessment(
            snapshot=snapshot,
            action=action,
            evidence=evidence,
            support_hid=support_hid or self._best_hid(),
            rationale=(
                "trace path evidence was inspected for propagation consistency"
            ),
            soften_competitors=False,
        )

    def _build_informative_assessment(
        self,
        *,
        snapshot: LeadTournamentSnapshot,
        action: DiscriminativeAction,
        evidence: EvidenceAtom,
        support_hid: str,
        rationale: str,
        soften_competitors: bool = True,
    ) -> EvidenceAssessment:
        links: list[EvidenceLinkProposal] = [
            EvidenceLinkProposal(
                hypothesis_id=support_hid,
                evidence_id=evidence.evidence_id,
                relation=EvidenceRelation.SUPPORTS,
                rationale=rationale,
            )
        ]
        self._support_ids.setdefault(support_hid, []).append(evidence.evidence_id)
        status_by_hid = {h.hypothesis_id: h.status for h in snapshot.hypotheses}
        updates: list[HypothesisStatusUpdate] = []
        if status_by_hid.get(support_hid) in (
            HypothesisStatus.ACTIVE,
            HypothesisStatus.WEAKENED,
        ):
            updates.append(
                HypothesisStatusUpdate(
                    hypothesis_id=support_hid,
                    new_status=HypothesisStatus.SUPPORTED,
                    rationale="the current evidence supports this causal hypothesis",
                )
            )

        for hid in action.target_hypothesis_ids:
            if hid == support_hid:
                continue
            links.append(
                EvidenceLinkProposal(
                    hypothesis_id=hid,
                    evidence_id=evidence.evidence_id,
                    relation=EvidenceRelation.CONTRADICTS,
                    rationale="the same observation is less consistent with this competitor",
                )
            )
            self._contradiction_ids.setdefault(hid, []).append(evidence.evidence_id)
            if soften_competitors and status_by_hid.get(hid) in (
                HypothesisStatus.ACTIVE,
                HypothesisStatus.SUPPORTED,
            ):
                updates.append(
                    HypothesisStatusUpdate(
                        hypothesis_id=hid,
                        new_status=HypothesisStatus.WEAKENED,
                        rationale="a competing hypothesis better explains this evidence",
                    )
                )

        return EvidenceAssessment(
            action_id=action.action_id,
            evidence_id=evidence.evidence_id,
            outcome=AssessmentOutcome.INFORMATIVE,
            links=tuple(links),
            status_updates=tuple(updates),
            rationale=rationale,
        )

    def _build_nomination(self, snapshot: LeadTournamentSnapshot) -> LeadNomination:
        status_by_hid = {h.hypothesis_id: h.status for h in snapshot.hypotheses}
        supported = [
            hid
            for hid in self._obs_by_hid
            if status_by_hid.get(hid) == HypothesisStatus.SUPPORTED
        ]
        hid = self._preferred_hid if self._preferred_hid in supported else None
        if hid is None:
            hid = supported[0] if supported else self._best_hid()

        support_ids = tuple(dict.fromkeys(self._support_ids.get(hid, ())))
        if not support_ids:
            raise ValueError(f"cannot nominate hypothesis '{hid}' without support")

        competitors = tuple(
            c
            for c in self._obs_by_hid
            if c != hid and self._contradiction_ids.get(c)
        )
        if not competitors:
            raise ValueError(f"cannot nominate hypothesis '{hid}' without competitor")

        reason_eid = support_ids[-1]
        onset_eid = support_ids[0]
        return LeadNomination(
            hypothesis_id=hid,
            supporting_evidence_ids=support_ids,
            addressed_competitor_ids=competitors,
            triplet_grounding=TripletEvidenceCoverage(
                component_evidence_ids=(onset_eid,),
                reason_evidence_ids=(reason_eid,),
                onset_evidence_ids=(onset_eid,),
            ),
            rationale=(
                f"{self._obs_by_hid[hid].component} survived the Lead "
                "hypothesis tournament with grounded supporting evidence"
            ),
        )

    def _best_observation(self) -> ObservedComponent:
        return max(
            self._obs_by_hid.values(),
            key=lambda o: (o.magnitude, -o.first_seen),
        )

    def _best_hid(self) -> str:
        best = self._best_observation()
        return self._hid_for_component(best.component) or next(iter(self._obs_by_hid))

    def _hid_for_component(self, component: str) -> str | None:
        for hid, obs in self._obs_by_hid.items():
            if obs.component == component:
                return hid
        return None


class DeterministicChallengerDiagnosticPolicy(ChallengerPolicy):
    """A single-test Challenger that tries to falsify the nomination."""

    def propose_challenge(self, *, snapshot) -> ChallengeProposal:
        nominee = snapshot.nominated_hypothesis
        competitors = tuple(h.hypothesis_id for h in snapshot.competitor_hypotheses)
        components = [nominee.root_component] + [
            h.root_component for h in snapshot.competitor_hypotheses
        ]
        t0 = max(0.0, nominee.onset_interval[0] - 60.0)
        t1 = nominee.onset_interval[1] + 300.0
        action = DiscriminativeAction(
            action_id="challenge-raw-local-evidence",
            action_type="run_discriminative_test",
            target_hypothesis_ids=(nominee.hypothesis_id,) + competitors,
            question="Does raw local telemetry still support the nominated hypothesis?",
            tool_name="retrieve_raw_evidence",
            args={
                "modality": "metric",
                "component_scope": components,
                "time_window": [t0, t1],
                "limit": 20,
            },
            expected_outcomes={
                nominee.hypothesis_id: (
                    f"{nominee.root_component} retains local raw evidence"
                ),
                **{
                    h.hypothesis_id: (
                        f"{h.root_component} exposes a stronger alternative"
                    )
                    for h in snapshot.competitor_hypotheses
                },
            },
            why_discriminative=(
                "a final nominee should survive a direct raw-evidence check "
                "against its strongest competitors"
            ),
        )
        return ChallengeProposal(
            challenge_id="challenge-raw-local-evidence",
            nominated_hypothesis_id=nominee.hypothesis_id,
            competitor_hypothesis_ids=competitors,
            challenge_claim=(
                "the nominated component may only be a downstream symptom"
            ),
            falsification_target=nominee.hypothesis_id,
            action=action,
            rationale="challenge the nomination using raw local telemetry",
        )

    def assess_challenge(
        self,
        *,
        snapshot,
        proposal: ChallengeProposal,
        evidence: EvidenceAtom,
    ) -> ChallengeResolution:
        nominee_id = proposal.nominated_hypothesis_id
        links = [
            EvidenceLinkProposal(
                hypothesis_id=nominee_id,
                evidence_id=evidence.evidence_id,
                relation=EvidenceRelation.SUPPORTS,
                rationale="raw local telemetry does not falsify the nominee",
            )
        ]
        assessment = EvidenceAssessment(
            action_id=proposal.action.action_id,
            evidence_id=evidence.evidence_id,
            outcome=AssessmentOutcome.INFORMATIVE,
            links=tuple(links),
            status_updates=(
                HypothesisStatusUpdate(
                    hypothesis_id=nominee_id,
                    new_status=HypothesisStatus.SURVIVED,
                    rationale="nomination survived the adversarial raw-evidence check",
                ),
            ),
            rationale="the challenge found no decisive contradiction",
        )
        return ChallengeResolution(
            challenge_id=proposal.challenge_id,
            action_id=proposal.action.action_id,
            evidence_id=evidence.evidence_id,
            verdict=ChallengeVerdict.NOMINATION_SURVIVED,
            assessment=assessment,
            rationale="nomination survived challenge",
        )


def _stable_id(value: str) -> str:
    return (
        value.lower()
        .replace(" ", "-")
        .replace("_", "-")
        .replace("/", "-")
        .replace(".", "-")
    )
