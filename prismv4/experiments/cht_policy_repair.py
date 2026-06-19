"""Experiment-local LLM action shape repair for PRISM-CHT."""

from __future__ import annotations

from typing import Mapping

from prismv4.prism_cht.action_schema import DiscriminativeAction
from prismv4.prism_cht.case_types import GenericRCACase
from prismv4.prism_cht.challenge_types import (
    ChallengeProposal,
    ChallengeResolution,
    ChallengeVerdict,
)
from prismv4.prism_cht.challenger_policy import ChallengerPolicy
from prismv4.prism_cht.hypothesis import HypothesisStatus
from prismv4.prism_cht.lead_policy import LeadPolicy, PolicyDecision
from prismv4.prism_cht.tournament_types import (
    AssessmentOutcome,
    EvidenceAssessment,
    EvidenceRelation,
    HypothesisStatusUpdate,
    LeadNomination,
)


class RepairingLeadPolicy:
    """Repair only action argument shapes before tool execution."""

    def __init__(self, *, inner: LeadPolicy, case: GenericRCACase) -> None:
        self._inner = inner
        self._case = case

    def decide_next(self, *, snapshot) -> PolicyDecision:
        try:
            decision = self._inner.decide_next(snapshot=snapshot)
        except ValueError as exc:
            if not (
                _is_repairable_action_error(exc)
                or _is_repairable_nomination_error(exc)
            ):
                raise
            if getattr(snapshot, "decision_mode", "") == "nomination_only":
                nomination = _best_guardrail_nomination(snapshot=snapshot)
                if nomination is not None:
                    return nomination
            return _fallback_action(snapshot=snapshot, case=self._case)
        if isinstance(decision, DiscriminativeAction):
            if getattr(snapshot, "decision_mode", "") == "nomination_only":
                nomination = _best_guardrail_nomination(snapshot=snapshot)
                if nomination is not None:
                    return nomination
            return repair_action(decision, snapshot=snapshot, case=self._case)
        if isinstance(decision, LeadNomination):
            return repair_nomination_or_request_more_evidence(
                decision,
                snapshot=snapshot,
                case=self._case,
            )
        return decision

    def assess_evidence(self, *, snapshot, action, evidence):
        assessment = self._inner.assess_evidence(
            snapshot=snapshot,
            action=action,
            evidence=evidence,
        )
        return repair_lead_assessment(
            assessment,
            snapshot=snapshot,
            action=action,
            evidence=evidence,
        )


class RepairingChallengerPolicy:
    """Repair only challenge action argument shapes before tool execution."""

    def __init__(self, *, inner: ChallengerPolicy, case: GenericRCACase) -> None:
        self._inner = inner
        self._case = case

    def propose_challenge(self, *, snapshot):
        try:
            proposal = self._inner.propose_challenge(snapshot=snapshot)
        except ValueError as exc:
            if not _is_repairable_action_error(exc):
                raise
            action = _fallback_action(snapshot=snapshot, case=self._case)
            return ChallengeProposal(
                challenge_id="repair-challenge-single-target",
                nominated_hypothesis_id=snapshot.nominated_hypothesis.hypothesis_id,
                competitor_hypothesis_ids=(
                    snapshot.competitor_hypotheses[0].hypothesis_id,
                ),
                challenge_claim="LLM challenge action targeted only one hypothesis.",
                falsification_target="Run a two-hypothesis contrast instead.",
                action=action,
                rationale="repair single-target challenge into a contrastive action",
            )
        repaired = repair_action(
            proposal.action,
            snapshot=snapshot,
            case=self._case,
        )
        if repaired is proposal.action:
            return proposal
        return ChallengeProposal(
            challenge_id=proposal.challenge_id,
            nominated_hypothesis_id=proposal.nominated_hypothesis_id,
            competitor_hypothesis_ids=proposal.competitor_hypothesis_ids,
            challenge_claim=proposal.challenge_claim,
            falsification_target=proposal.falsification_target,
            action=repaired,
            rationale=proposal.rationale,
        )

    def assess_challenge(self, *, snapshot, proposal, evidence):
        resolution = self._inner.assess_challenge(
            snapshot=snapshot,
            proposal=proposal,
            evidence=evidence,
        )
        return repair_challenge_resolution(
            resolution,
            snapshot=snapshot,
            proposal=proposal,
            evidence=evidence,
        )


def repair_lead_assessment(
    assessment: EvidenceAssessment,
    *,
    snapshot,
    action=None,
    evidence=None,
) -> EvidenceAssessment:
    repair_notes: list[str] = []
    outcome = assessment.outcome
    if assessment.outcome == AssessmentOutcome.INCONCLUSIVE and (
        assessment.links or assessment.status_updates
    ):
        outcome = AssessmentOutcome.INFORMATIVE
        repair_notes.append("repaired inconclusive-with-links to informative")

    if _single_missing_propagation_path(action=action, evidence=evidence):
        return EvidenceAssessment(
            action_id=assessment.action_id,
            evidence_id=assessment.evidence_id,
            outcome=AssessmentOutcome.INCONCLUSIVE,
            links=(),
            status_updates=(),
            rationale=(
                assessment.rationale
                + " Repair note: single missing propagation path treated as inconclusive."
            ),
        )
    reason_signature_only = getattr(action, "tool_name", "") == "inspect_reason_signature"

    if outcome == AssessmentOutcome.INCONCLUSIVE:
        return EvidenceAssessment(
            action_id=assessment.action_id,
            evidence_id=assessment.evidence_id,
            outcome=outcome,
            links=(),
            status_updates=(),
            rationale=assessment.rationale,
        )

    status_by_hid = {h.hypothesis_id: h.status for h in snapshot.hypotheses}
    links = []
    seen_links: set[tuple[str, str]] = set()
    for link in assessment.links:
        key = (link.hypothesis_id, link.evidence_id)
        if key in seen_links:
            continue
        links.append(link)
        seen_links.add(key)

    updates: list[HypothesisStatusUpdate] = []
    seen: set[str] = set()
    if reason_signature_only:
        repair_notes.append("reason-signature evidence kept as links without status promotion")
    else:
        for link in links:
            if link.hypothesis_id in seen:
                continue
            current = status_by_hid.get(link.hypothesis_id)
            new_status = None
            if link.relation == EvidenceRelation.SUPPORTS and current in (
                HypothesisStatus.ACTIVE,
                HypothesisStatus.WEAKENED,
            ):
                new_status = HypothesisStatus.SUPPORTED
            elif link.relation == EvidenceRelation.CONTRADICTS and current in (
                HypothesisStatus.ACTIVE,
                HypothesisStatus.SUPPORTED,
            ):
                new_status = HypothesisStatus.WEAKENED
            elif link.relation == EvidenceRelation.CONTRADICTS and current == HypothesisStatus.WEAKENED:
                new_status = HypothesisStatus.REFUTED
            if new_status is not None:
                updates.append(
                    HypothesisStatusUpdate(
                        hypothesis_id=link.hypothesis_id,
                        new_status=new_status,
                        rationale="status repaired from evidence relation",
                    )
                )
                seen.add(link.hypothesis_id)
    if not links:
        outcome = AssessmentOutcome.INCONCLUSIVE
        repair_notes.append("repaired informative-without-links to inconclusive")
    rationale = assessment.rationale
    if repair_notes:
        rationale = rationale + " Repair note: " + "; ".join(repair_notes) + "."
    return EvidenceAssessment(
        action_id=assessment.action_id,
        evidence_id=assessment.evidence_id,
        outcome=outcome,
        links=tuple(links),
        status_updates=tuple(updates),
        rationale=rationale,
    )


def repair_challenge_resolution(
    resolution: ChallengeResolution,
    *,
    snapshot,
    proposal: ChallengeProposal,
    evidence=None,
) -> ChallengeResolution:
    assessment = resolution.assessment
    if _single_missing_propagation_path(action=proposal.action, evidence=evidence):
        fixed_assessment = EvidenceAssessment(
            action_id=assessment.action_id,
            evidence_id=assessment.evidence_id,
            outcome=AssessmentOutcome.INCONCLUSIVE,
            links=(),
            status_updates=(),
            rationale=(
                assessment.rationale
                + " Repair note: single missing propagation path treated as inconclusive."
            ),
        )
        return ChallengeResolution(
            challenge_id=resolution.challenge_id,
            action_id=resolution.action_id,
            evidence_id=resolution.evidence_id,
            verdict=ChallengeVerdict.INCONCLUSIVE,
            assessment=fixed_assessment,
            rationale=(
                resolution.rationale
                + " Repair note: single missing propagation path cannot decide challenge."
            ),
        )
    if resolution.verdict == ChallengeVerdict.INCONCLUSIVE:
        fixed_assessment = EvidenceAssessment(
            action_id=assessment.action_id,
            evidence_id=assessment.evidence_id,
            outcome=AssessmentOutcome.INCONCLUSIVE,
            links=(),
            status_updates=(),
            rationale=assessment.rationale,
        )
    else:
        nominee_id = proposal.nominated_hypothesis_id
        updates: tuple[HypothesisStatusUpdate, ...]
        if resolution.verdict == ChallengeVerdict.NOMINATION_SURVIVED:
            updates = (
                HypothesisStatusUpdate(
                    hypothesis_id=nominee_id,
                    new_status=HypothesisStatus.SURVIVED,
                    rationale="nomination survived challenge evidence",
                ),
            )
        else:
            updates = (
                HypothesisStatusUpdate(
                    hypothesis_id=nominee_id,
                    new_status=HypothesisStatus.WEAKENED,
                    rationale="challenge evidence weakened the nomination",
                ),
            )
        fixed_assessment = EvidenceAssessment(
            action_id=assessment.action_id,
            evidence_id=assessment.evidence_id,
            outcome=AssessmentOutcome.INFORMATIVE,
            links=assessment.links,
            status_updates=updates,
            rationale=assessment.rationale,
        )
    return ChallengeResolution(
        challenge_id=resolution.challenge_id,
        action_id=resolution.action_id,
        evidence_id=resolution.evidence_id,
        verdict=resolution.verdict,
        assessment=fixed_assessment,
        rationale=resolution.rationale,
    )


def _single_missing_propagation_path(*, action, evidence) -> bool:
    if action is None or evidence is None:
        return False
    if getattr(action, "tool_name", "") != "check_propagation_consistency":
        return False
    symptoms = action.args.get("symptom_components", ())
    if not isinstance(symptoms, (list, tuple)) or len(symptoms) != 1:
        return False
    if getattr(evidence, "modality", "") != "propagation":
        return False
    facts = dict(evidence.observation).get("propagation_facts", ())
    if not isinstance(facts, (list, tuple)) or len(facts) != 1:
        return False
    fact = facts[0]
    if not isinstance(fact, Mapping):
        return False
    return (
        bool(fact.get("has_observed_path")) is False
        and int(fact.get("path_count", 0) or 0) == 0
    )


def repair_action(
    action: DiscriminativeAction,
    *,
    snapshot,
    case: GenericRCACase,
) -> DiscriminativeAction:
    args = dict(action.args)
    hypotheses = _hypotheses_from_snapshot(snapshot)
    target_ids = tuple(action.target_hypothesis_ids)
    components = [
        hypotheses[hid]["component"]
        for hid in target_ids
        if hid in hypotheses
    ]
    if len(components) < 2:
        components = list(case.components[:2])

    t0, t1 = _time_window(args.get("time_window"), case)

    if action.tool_name == "compare_onset_order":
        args["component_scope"] = _list_arg(args.get("component_scope"), fallback=components)
        if len(args["component_scope"]) < 2:
            args["component_scope"] = components
        args["signal_scope"] = _list_arg(
            args.get("signal_scope"),
            fallback=_signals_from_hypotheses(hypotheses, target_ids),
        )
        args["time_window"] = [t0, t1]

    elif action.tool_name == "inspect_reason_signature":
        args["component_scope"] = _list_arg(args.get("component_scope"), fallback=components)
        if len(args["component_scope"]) < 2:
            args["component_scope"] = components
        reason_by_component = args.get("reason_by_component")
        if not isinstance(reason_by_component, Mapping):
            reason_by_component = {}
        args["reason_by_component"] = {
            component: str(
                reason_by_component.get(component)
                or _reason_for_component(hypotheses, component)
            )
            for component in args["component_scope"]
        }
        args["time_window"] = [t0, t1]

    elif action.tool_name == "inspect_trace_path":
        source = str(args.get("source_component") or components[0])
        target = str(
            args.get("target_component")
            or (case.entry_components[0] if case.entry_components else components[-1])
        )
        if source == target and len(components) > 1:
            target = components[1]
        args["source_component"] = source
        args["target_component"] = target
        args["time_window"] = [t0, t1]
        args["max_hops"] = _positive_int(args.get("max_hops"), 6)
        args["max_paths"] = _positive_int(args.get("max_paths"), 10)

    elif action.tool_name == "retrieve_raw_evidence":
        args["modality"] = str(args.get("modality") or "metric")
        args["component_scope"] = _list_arg(args.get("component_scope"), fallback=components)
        args["time_window"] = [t0, t1]
        args["limit"] = min(_positive_int(args.get("limit"), 20), 50)

    elif action.tool_name == "check_propagation_consistency":
        args["source_component"] = str(args.get("source_component") or components[0])
        args["symptom_components"] = _list_arg(
            args.get("symptom_components"),
            fallback=list(case.entry_components or components[1:]),
        )
        args["time_window"] = [t0, t1]

    elif action.tool_name == "find_unexplained_symptoms":
        args["explained_components"] = _list_arg(
            args.get("explained_components"),
            fallback=components[:1],
        )
        args["time_window"] = [t0, t1]
        args["limit"] = min(_positive_int(args.get("limit"), 20), 50)

    else:
        return action

    expected_outcomes = _repair_expected_outcomes(
        action=action,
        hypotheses=hypotheses,
    )

    if args == dict(action.args) and expected_outcomes == dict(action.expected_outcomes):
        return action
    return DiscriminativeAction(
        action_id=action.action_id,
        action_type=action.action_type,
        target_hypothesis_ids=action.target_hypothesis_ids,
        question=action.question,
        tool_name=action.tool_name,
        args=args,
        expected_outcomes=expected_outcomes,
        why_discriminative=action.why_discriminative,
    )


def _repair_expected_outcomes(
    *,
    action: DiscriminativeAction,
    hypotheses: Mapping[str, Mapping[str, str]],
) -> dict[str, str]:
    expected = {
        hid: str(action.expected_outcomes.get(hid, "")).strip()
        for hid in action.target_hypothesis_ids
    }
    normalized = {" ".join(text.lower().split()) for text in expected.values()}
    if len(normalized) > 1 and all(expected.values()):
        return expected
    repaired: dict[str, str] = {}
    for hid in action.target_hypothesis_ids:
        info = hypotheses.get(hid, {})
        component = str(info.get("component") or hid)
        reason = str(info.get("reason") or info.get("reason_family") or "local anomaly")
        repaired[hid] = (
            f"{hid}: {component} shows the strongest local {reason} evidence "
            f"relative to the other targeted hypotheses"
        )
    return repaired


def repair_nomination_or_request_more_evidence(
    nomination: LeadNomination,
    *,
    snapshot,
    case: GenericRCACase,
) -> PolicyDecision:
    status_by_hid = {h.hypothesis_id: h.status for h in snapshot.hypotheses}
    hypothesis_by_hid = {h.hypothesis_id: h for h in snapshot.hypotheses}
    contradiction_by_hid = {
        h.hypothesis_id: tuple(h.contradicting_evidence_ids)
        for h in snapshot.hypotheses
    }
    nominee_snapshot = hypothesis_by_hid.get(nomination.hypothesis_id)
    linked_support = set(nominee_snapshot.supporting_evidence_ids if nominee_snapshot else ())
    fixed_support = tuple(
        eid for eid in nomination.supporting_evidence_ids if eid in linked_support
    )
    if getattr(snapshot, "decision_mode", "") == "nomination_only":
        return _repair_nomination_best_effort(
            nomination,
            snapshot=snapshot,
            fixed_support=fixed_support,
        )
    if not fixed_support:
        nominee = nominee_snapshot
        supported_competitors = [
            h
            for h in snapshot.hypotheses
            if h.hypothesis_id != nomination.hypothesis_id
            and h.status in (HypothesisStatus.ACTIVE, HypothesisStatus.SUPPORTED)
        ]
        if nominee is not None and supported_competitors:
            return _competitor_resolution_action(
                nominee=nominee,
                competitor=supported_competitors[0],
                snapshot=snapshot,
                case=case,
            )
        return nomination

    valid_competitors = tuple(
        hid
        for hid in nomination.addressed_competitor_ids
        if status_by_hid.get(hid) in (HypothesisStatus.WEAKENED, HypothesisStatus.REFUTED)
        and contradiction_by_hid.get(hid)
    )
    all_competitor_ids = tuple(
        h.hypothesis_id
        for h in snapshot.hypotheses
        if h.hypothesis_id != nomination.hypothesis_id
    )
    if set(valid_competitors) == set(all_competitor_ids):
        if valid_competitors == nomination.addressed_competitor_ids:
            competitors = nomination.addressed_competitor_ids
        else:
            competitors = valid_competitors
        return LeadNomination(
            hypothesis_id=nomination.hypothesis_id,
            supporting_evidence_ids=fixed_support,
            addressed_competitor_ids=competitors,
            triplet_grounding=_repair_triplet_grounding(nomination, fixed_support),
            rationale=nomination.rationale,
        )

    nominee = next(
        (h for h in snapshot.hypotheses if h.hypothesis_id == nomination.hypothesis_id),
        None,
    )
    unresolved_competitors = [
        h
        for h in snapshot.hypotheses
        if h.hypothesis_id != nomination.hypothesis_id
        and h.hypothesis_id not in valid_competitors
    ]
    if nominee is None or not unresolved_competitors:
        return nomination
    competitor = unresolved_competitors[0]
    components = [nominee.root_component, competitor.root_component]
    t0 = max(0.0, case.event_time - 300.0)
    t1 = case.event_time + 600.0
    return _competitor_resolution_action(
        nominee=nominee,
        competitor=competitor,
        snapshot=snapshot,
        case=case,
    )


def _repair_nomination_best_effort(
    nomination: LeadNomination,
    *,
    snapshot,
    fixed_support: tuple[str, ...],
) -> LeadNomination:
    hypothesis_by_hid = {h.hypothesis_id: h for h in snapshot.hypotheses}
    nominee = hypothesis_by_hid.get(nomination.hypothesis_id)
    if nominee is None:
        fallback = _best_guardrail_nomination(snapshot=snapshot)
        return fallback if fallback is not None else nomination

    support_ids = fixed_support or tuple(nominee.supporting_evidence_ids)
    if not support_ids:
        fallback = _best_guardrail_nomination(snapshot=snapshot)
        return fallback if fallback is not None else nomination

    addressed = tuple(
        hid
        for hid in nomination.addressed_competitor_ids
        if hid in hypothesis_by_hid
        and hid != nomination.hypothesis_id
        and hypothesis_by_hid[hid].status in (HypothesisStatus.WEAKENED, HypothesisStatus.REFUTED)
        and hypothesis_by_hid[hid].contradicting_evidence_ids
    )
    all_competitors = tuple(
        h.hypothesis_id
        for h in snapshot.hypotheses
        if h.hypothesis_id != nomination.hypothesis_id
    )
    if set(addressed) != set(all_competitors):
        fallback = _best_guardrail_nomination(snapshot=snapshot)
        return fallback if fallback is not None else nomination

    return LeadNomination(
        hypothesis_id=nomination.hypothesis_id,
        supporting_evidence_ids=support_ids,
        addressed_competitor_ids=addressed,
        triplet_grounding=_repair_triplet_grounding(nomination, support_ids),
        rationale=nomination.rationale + " Repair note: final-budget nomination repaired.",
    )


def _best_guardrail_nomination(*, snapshot) -> LeadNomination | None:
    from prismv4.prism_cht.tournament_types import TripletEvidenceCoverage

    candidates = sorted(
        snapshot.hypotheses,
        key=lambda h: (
            h.status != HypothesisStatus.SUPPORTED,
            -len(h.supporting_evidence_ids),
            len(h.contradicting_evidence_ids),
            h.hypothesis_id,
        ),
    )
    for h in candidates:
        if h.status != HypothesisStatus.SUPPORTED or not h.supporting_evidence_ids:
            continue
        addressed = tuple(
            c.hypothesis_id
            for c in snapshot.hypotheses
            if c.hypothesis_id != h.hypothesis_id
            and c.status in (HypothesisStatus.WEAKENED, HypothesisStatus.REFUTED)
            and c.contradicting_evidence_ids
        )
        all_competitors = tuple(
            c.hypothesis_id
            for c in snapshot.hypotheses
            if c.hypothesis_id != h.hypothesis_id
        )
        if set(addressed) != set(all_competitors):
            continue
        support = tuple(h.supporting_evidence_ids)
        return LeadNomination(
            hypothesis_id=h.hypothesis_id,
            supporting_evidence_ids=support,
            addressed_competitor_ids=addressed,
            triplet_grounding=TripletEvidenceCoverage(
                component_evidence_ids=(support[0],),
                reason_evidence_ids=(support[0],),
                onset_evidence_ids=(support[0],),
            ),
            rationale="Final-budget guardrail-ready nomination from applied evidence.",
        )
    return None


def _competitor_resolution_action(*, nominee, competitor, snapshot, case):
    components = [nominee.root_component, competitor.root_component]
    t0 = max(0.0, case.event_time - 300.0)
    t1 = case.event_time + 600.0
    return DiscriminativeAction(
        action_id=f"repair-competitor-test-{snapshot.round_index}",
        action_type="run_discriminative_test",
        target_hypothesis_ids=(nominee.hypothesis_id, competitor.hypothesis_id),
        question="Resolve the still-supported competitor before nomination.",
        tool_name="compare_onset_order",
        args={
            "component_scope": components,
            "signal_scope": ["cpu", "memory", "latency", "error"],
            "time_window": [t0, t1],
        },
        expected_outcomes={
            nominee.hypothesis_id: f"{nominee.root_component} remains earlier",
            competitor.hypothesis_id: f"{competitor.root_component} is earlier",
        },
        why_discriminative="nomination requires addressed competitors to be weakened or refuted",
    )


def _repair_triplet_grounding(nomination: LeadNomination, support_ids: tuple[str, ...]):
    def keep_or_first(values):
        kept = tuple(eid for eid in values if eid in support_ids)
        return kept or (support_ids[0],)

    from prismv4.prism_cht.tournament_types import TripletEvidenceCoverage

    return TripletEvidenceCoverage(
        component_evidence_ids=keep_or_first(nomination.triplet_grounding.component_evidence_ids),
        reason_evidence_ids=keep_or_first(nomination.triplet_grounding.reason_evidence_ids),
        onset_evidence_ids=keep_or_first(nomination.triplet_grounding.onset_evidence_ids),
    )


def _is_repairable_action_error(exc: ValueError) -> bool:
    text = str(exc)
    return (
        "target_hypothesis_ids must contain at least two" in text
        or "target_hypothesis_ids contains duplicate" in text
        or "expected_outcomes must cover hypothesis" in text
        or "all expected_outcomes are identical" in text
    )


def _is_repairable_nomination_error(exc: ValueError) -> bool:
    text = str(exc)
    return (
        "component_evidence_ids must contain at least one evidence ID" in text
        or "reason_evidence_ids must contain at least one evidence ID" in text
        or "onset_evidence_ids must contain at least one evidence ID" in text
        or "supporting_evidence_ids contains duplicate entries" in text
        or "addressed_competitor_ids contains duplicate entries" in text
    )


def _fallback_action(*, snapshot, case: GenericRCACase) -> DiscriminativeAction:
    hypothesis_items = _ordered_hypothesis_items(snapshot)
    if len(hypothesis_items) < 2:
        raise ValueError("cannot repair single-target action without two hypotheses")

    primary, secondary = hypothesis_items[0], hypothesis_items[1]
    target_ids = (primary.hypothesis_id, secondary.hypothesis_id)
    components = [primary.root_component, secondary.root_component]
    t0, t1 = _time_window(None, case)
    expected = {
        primary.hypothesis_id: f"{primary.root_component} has stronger early evidence",
        secondary.hypothesis_id: f"{secondary.root_component} has stronger early evidence",
    }
    candidates = (
        DiscriminativeAction(
            action_id=f"repair-single-target-reason-{_round_index(snapshot)}",
            action_type="run_discriminative_test",
            target_hypothesis_ids=target_ids,
            question="Compare candidate reason signatures after a single-target LLM action.",
            tool_name="inspect_reason_signature",
            args={
                "component_scope": components,
                "reason_by_component": {
                    primary.root_component: primary.reason_family,
                    secondary.root_component: secondary.reason_family,
                },
                "time_window": [t0, t1],
            },
            expected_outcomes=expected,
            why_discriminative="repairs a single-target LLM action into a two-candidate contrast",
        ),
        DiscriminativeAction(
            action_id=f"repair-single-target-propagation-{_round_index(snapshot)}",
            action_type="run_discriminative_test",
            target_hypothesis_ids=target_ids,
            question="Check whether one candidate can explain observed propagation.",
            tool_name="check_propagation_consistency",
            args={
                "source_component": primary.root_component,
                "symptom_components": [secondary.root_component],
                "time_window": [t0, t1],
            },
            expected_outcomes=expected,
            why_discriminative="observed propagation should favor one candidate over the other",
        ),
        DiscriminativeAction(
            action_id=f"repair-single-target-raw-{_round_index(snapshot)}",
            action_type="run_discriminative_test",
            target_hypothesis_ids=target_ids,
            question="Retrieve raw evidence for the two candidates after malformed output.",
            tool_name="retrieve_raw_evidence",
            args={
                "modality": "metric",
                "component_scope": components,
                "time_window": [t0, t1],
                "limit": 20,
            },
            expected_outcomes=expected,
            why_discriminative="raw records can separate the candidate with stronger anomaly evidence",
        ),
    )
    used_signatures = {
        step.action.query_signature()
        for step in getattr(snapshot, "audit_steps", getattr(snapshot, "lead_audit_steps", ()))
    }
    for action in candidates:
        if action.query_signature() not in used_signatures:
            return action
    return candidates[_round_index(snapshot) % len(candidates)]


def _ordered_hypothesis_items(snapshot):
    if hasattr(snapshot, "hypotheses"):
        items = list(snapshot.hypotheses)
    else:
        items = [snapshot.nominated_hypothesis] + list(snapshot.competitor_hypotheses)
    eligible = [
        h
        for h in items
        if h.status not in (HypothesisStatus.REFUTED, HypothesisStatus.FINAL)
    ]
    return eligible or items


def _round_index(snapshot) -> int:
    return int(getattr(snapshot, "round_index", len(getattr(snapshot, "lead_audit_steps", ()))))


def _hypotheses_from_snapshot(snapshot):
    if hasattr(snapshot, "hypotheses"):
        items = snapshot.hypotheses
    else:
        items = (snapshot.nominated_hypothesis,) + tuple(snapshot.competitor_hypotheses)
    return {
        h.hypothesis_id: {
            "component": h.root_component,
            "reason": h.reason_family,
            "signals": tuple(h.predicted_observations),
        }
        for h in items
    }


def _list_arg(value, *, fallback):
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value if str(v).strip()]
    return list(fallback)


def _time_window(value, case: GenericRCACase) -> tuple[float, float]:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        try:
            t0 = float(value[0])
            t1 = float(value[1])
            if t0 <= t1:
                return t0, t1
        except Exception:
            pass
    return max(0.0, case.event_time - 300.0), case.event_time + 600.0


def _signals_from_hypotheses(hypotheses, target_ids):
    signals = []
    for hid in target_ids:
        reason = str(hypotheses.get(hid, {}).get("reason", "")).lower()
        if "cpu" in reason:
            signals.append("cpu")
        elif "memory" in reason:
            signals.append("memory")
        elif "latency" in reason:
            signals.append("latency")
        elif "error" in reason:
            signals.append("error")
    return signals or ["cpu", "memory", "latency", "error"]


def _reason_for_component(hypotheses, component):
    for data in hypotheses.values():
        if data["component"] == component:
            return data["reason"]
    return "unspecified anomaly"


def _positive_int(value, fallback):
    try:
        parsed = int(value)
    except Exception:
        return fallback
    return parsed if parsed > 0 else fallback
