"""Noise-native observe-act-reason RCA agent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import math

import numpy as np

from .evidence_frame import CandidateFrame, EvidenceFrame, canonical_entity_name
from .fault_event import EvidenceObservation, FaultEvent, NoiseNativeAgentState
from .posterior import (
    PosteriorWeights,
    posterior_vector,
    root_role_features,
    root_selection_score,
    score_events,
)
from .tools import NoiseNativeToolbox, ToolContext


ReasonInferer = Callable[[str], str]


@dataclass
class NoiseNativeAgentResult:
    applied: bool
    state: NoiseNativeAgentState
    posterior: Optional[np.ndarray]
    debug: Dict[str, Any]


@dataclass
class ActionProposal:
    action_name: str
    utility: float
    event: Optional[FaultEvent] = None
    other_event: Optional[FaultEvent] = None
    reason: str = ""

    def to_debug(self) -> Dict[str, Any]:
        return {
            "action_name": self.action_name,
            "utility": round(float(self.utility), 6),
            "event_id": self.event.event_id if self.event is not None else "",
            "component": self.event.component if self.event is not None else "",
            "other_event_id": self.other_event.event_id if self.other_event is not None else "",
            "other_component": self.other_event.component if self.other_event is not None else "",
            "reason": self.reason,
        }


class NoiseNativePRISMAgent:
    def __init__(
        self,
        evidence_frame: Optional[EvidenceFrame],
        tool_context: ToolContext,
        weights: PosteriorWeights,
        reason_inferer: Optional[ReasonInferer] = None,
        max_events: int = 10,
        max_rounds: int = 2,
    ) -> None:
        self.evidence_frame = evidence_frame
        self.tool_context = tool_context
        self.weights = weights
        self.reason_inferer = reason_inferer
        self.max_events = max(1, int(max_events))
        self.max_rounds = max(1, int(max_rounds))
        self.toolbox = NoiseNativeToolbox(tool_context)

    def run(self) -> NoiseNativeAgentResult:
        agent_state = NoiseNativeAgentState()
        agent_state.events = self._initialize_events()
        if not agent_state.events:
            agent_state.stop_reason = "no_noise_native_events"
            return NoiseNativeAgentResult(
                applied=False,
                state=agent_state,
                posterior=None,
                debug={"enabled": True, "applied": False, "reason": agent_state.stop_reason},
            )

        initial_attrib = score_events(agent_state.events, self.weights)
        agent_state.posterior_attributions.extend(initial_attrib)
        for round_idx in range(self.max_rounds):
            before_top = self._top_event_id(agent_state.events)
            before_uncertainty = self._uncertainty_snapshot(agent_state.events)
            observations = self._act(agent_state, round_idx)
            if not observations:
                agent_state.stop_reason = "no_useful_observations"
                break
            self._observe(agent_state, observations)
            agent_state.posterior_attributions.extend(
                score_events(agent_state.events, self.weights)
            )
            after_top = self._top_event_id(agent_state.events)
            after_uncertainty = self._uncertainty_snapshot(agent_state.events)
            actual_ig = (
                before_uncertainty["event_entropy"]
                + before_uncertainty["mean_status_entropy"]
                + before_uncertainty["coverage_uncertainty"]
                - after_uncertainty["event_entropy"]
                - after_uncertainty["mean_status_entropy"]
                - after_uncertainty["coverage_uncertainty"]
            )
            uncertainty_record = {
                "round": round_idx,
                "before": before_uncertainty,
                "after": after_uncertainty,
                "actual_information_gain": round(float(actual_ig), 6),
            }
            agent_state.uncertainty_history.append(uncertainty_record)
            agent_state.action_history.append(
                {
                    "round": round_idx,
                    "before_top": before_top,
                    "after_top": after_top,
                    "observation_count": len(observations),
                    "actions": list(getattr(self, "_last_action_debug", [])),
                    "uncertainty": uncertainty_record,
                    "active_events": [event.event_id for event in agent_state.active_events[:5]],
                }
            )
            if self._should_stop(agent_state.events, after_uncertainty):
                agent_state.stop_reason = "uncertainty_reduced"
                break
        if not agent_state.stop_reason:
            agent_state.stop_reason = "max_rounds"
        posterior = posterior_vector(agent_state.events, self.tool_context.entities)
        return NoiseNativeAgentResult(
            applied=True,
            state=agent_state,
            posterior=posterior,
            debug={
                "enabled": True,
                "applied": True,
                "agent_cycle": "observe_act_reason",
                "posterior_factorization": {
                    "NoiseLab_logit": self.weights.noise,
                    "metric_likelihood": self.weights.metric,
                    "log_likelihood": self.weights.log,
                    "trace_direction_likelihood": self.weights.trace,
                    "counterfactual_likelihood": self.weights.counterfactual,
                    "pairwise_verdict": self.weights.pairwise,
                    "source_isolation": self.weights.isolation,
                    "residual_collapse": self.weights.collapse,
                    "mechanism_break_likelihood": self.weights.mechanism,
                    "mechanism_parent_refutation": self.weights.parent_refutation,
                    "intervention_uniqueness": self.weights.intervention,
                    "symptom_conflict=max(0,symptomness-source_likelihood)": -self.weights.symptom,
                    "hotspot_symptom": -self.weights.hotspot,
                    "broad_explainer": -self.weights.broad,
                },
                "state": agent_state.to_debug(),
            },
        )

    def _initialize_events(self) -> List[FaultEvent]:
        candidates = self._candidate_frames()
        events: List[FaultEvent] = []
        seen = set()
        for idx, candidate in enumerate(candidates[: self.max_events], start=1):
            component = self._component_for_candidate(candidate)
            if not component or component in seen:
                continue
            seen.add(component)
            event = FaultEvent(
                event_id=f"event_{idx}:{component}",
                component=component,
                reason=self._candidate_reason(component, candidate),
                time=self._candidate_time(component, candidate),
                posterior=float(candidate.prior_mass),
                candidate_id=candidate.candidate_id,
                reason_candidates=list(candidate.reason_candidates),
            )
            self._add_initial_observations(event, candidate)
            events.append(event)
        return events

    def _candidate_frames(self) -> List[CandidateFrame]:
        if self.evidence_frame is not None and self.evidence_frame.candidates:
            return self.evidence_frame.top_candidates(self.max_events)
        return []

    def _component_for_candidate(self, candidate: CandidateFrame) -> str:
        key = candidate.canonical_component
        for entity in self.tool_context.entities:
            if canonical_entity_name(entity) == key:
                return str(entity)
        return str(candidate.component_id or candidate.object_id)

    def _candidate_reason(self, component: str, candidate: CandidateFrame) -> str:
        if candidate.reason_candidates:
            best = max(
                candidate.reason_candidates,
                key=lambda item: float(item.get("score", 0.0) or 0.0),
            )
            reason = str(best.get("reason", "") or "")
            if reason:
                return reason
        if self.reason_inferer is not None:
            inferred = self.reason_inferer(component)
            if inferred:
                return inferred
        return ""

    def _candidate_time(self, component: str, candidate: CandidateFrame) -> Optional[float]:
        for item in candidate.time_candidates:
            if "timestamp" in item:
                try:
                    return float(item["timestamp"])
                except (TypeError, ValueError):
                    pass
        return self.tool_context.anomaly_times.get(component)

    def _add_initial_observations(
        self, event: FaultEvent, candidate: CandidateFrame
    ) -> None:
        prior_mass = max(float(candidate.prior_mass), 1e-9)
        event.add_observation(
            EvidenceObservation(
                evidence_id=f"init:noise:{candidate.candidate_id}",
                tool_name="observe_noiselab",
                event_id=event.event_id,
                component=event.component,
                factor_name="NoiseLab_logit",
                factor_delta=math.log(prior_mass * self.max_events + 1.0),
                payload={
                    "noise_score": round(float(candidate.noise_score), 6),
                    "calibrated_logit": round(float(candidate.calibrated_logit), 6),
                    "prior_mass": round(prior_mass, 6),
                },
            )
        )
        event.add_observation(
            EvidenceObservation(
                evidence_id=f"init:source:{candidate.candidate_id}",
                tool_name="observe_noiselab",
                event_id=event.event_id,
                component=event.component,
                factor_name="source_likelihood",
                factor_delta=float(candidate.source_likelihood),
                payload={"source_likelihood": round(float(candidate.source_likelihood), 6)},
            )
        )
        event.add_observation(
            EvidenceObservation(
                evidence_id=f"init:symptom:{candidate.candidate_id}",
                tool_name="observe_noiselab",
                event_id=event.event_id,
                component=event.component,
                factor_name="symptomness",
                factor_delta=float(candidate.symptomness),
                payload={"symptomness": round(float(candidate.symptomness), 6)},
            )
        )
        isolation, isolation_payload = self._source_isolation(candidate)
        event.add_observation(
            EvidenceObservation(
                evidence_id=f"init:isolation:{candidate.candidate_id}",
                tool_name="observe_noiselab",
                event_id=event.event_id,
                component=event.component,
                factor_name="source_isolation",
                factor_delta=isolation,
                payload=isolation_payload,
            )
        )
        hotspot, hotspot_payload = self._hotspot_symptom(candidate)
        event.add_observation(
            EvidenceObservation(
                evidence_id=f"init:hotspot:{candidate.candidate_id}",
                tool_name="observe_noiselab",
                event_id=event.event_id,
                component=event.component,
                factor_name="hotspot_symptom",
                factor_delta=hotspot,
                payload=hotspot_payload,
            )
        )
        if hotspot >= 0.55 and isolation < hotspot:
            event.conflict_notes.append("hotspot_symptom_exceeds_source_isolation")

    def _act(
        self, agent_state: NoiseNativeAgentState, round_idx: int
    ) -> List[EvidenceObservation]:
        observations: List[EvidenceObservation] = []
        proposals = self._propose_actions(agent_state.events, round_idx)
        max_actions = max(8, min(24, len(agent_state.events) * 2))
        selected = proposals[:max_actions]
        self._last_action_debug = [proposal.to_debug() for proposal in selected]
        for proposal in selected:
            event = proposal.event
            other = proposal.other_event
            if event is None:
                continue
            if proposal.action_name == "inspect_metric":
                observations.append(self.toolbox.inspect_metric(event))
            elif proposal.action_name == "inspect_log":
                observations.append(self.toolbox.inspect_log(event))
            elif proposal.action_name == "verify_trace":
                observations.append(self.toolbox.verify_trace(event))
            elif proposal.action_name == "run_counterfactual":
                observations.append(self.toolbox.run_counterfactual(event))
            elif proposal.action_name == "mechanism_intervention":
                observations.extend(self.toolbox.mechanism_intervention(event))
            elif proposal.action_name == "residual_collapse":
                observations.append(self.toolbox.residual_collapse(event))
            elif proposal.action_name == "split_event":
                observations.append(self.toolbox.split_event(event))
            elif proposal.action_name == "compare_pair" and other is not None:
                observations.extend(self.toolbox.compare_pair(event, other))
            elif proposal.action_name == "merge_events" and other is not None:
                observations.extend(self.toolbox.merge_events(event, other))
        return observations

    def _observe(
        self,
        agent_state: NoiseNativeAgentState,
        observations: Sequence[EvidenceObservation],
    ) -> None:
        for observation in observations:
            event = agent_state.event_by_id(observation.event_id)
            if event is None:
                continue
            event.add_observation(
                observation,
                step=len(agent_state.uncertainty_history),
            )
            agent_state.observations.append(observation)
            if observation.factor_name == "merge_duplicate" and observation.factor_delta < 0:
                event.merged_into = str(observation.payload.get("merge_into", ""))

    def _rank_events(self, events: Sequence[FaultEvent]) -> List[FaultEvent]:
        return sorted(events, key=lambda event: event.posterior, reverse=True)

    def _top_event_id(self, events: Sequence[FaultEvent]) -> str:
        ranked = self._rank_events(events)
        return ranked[0].event_id if ranked else ""

    def _top_gap(self, events: Sequence[FaultEvent]) -> float:
        ranked = self._rank_events(events)
        if len(ranked) < 2:
            return 1.0
        return float(ranked[0].posterior - ranked[1].posterior)

    def _propose_actions(
        self, events: Sequence[FaultEvent], round_idx: int
    ) -> List[ActionProposal]:
        ranked = self._rank_events(events)
        proposals: List[ActionProposal] = []
        has_logs = bool(np.max(self.tool_context.log_signal) > 0.0) if self.tool_context.log_signal.size else False
        for event in ranked:
            base = self._event_uncertainty_value(event)
            missing_tools = self._missing_required_tools(event, has_logs)
            for tool_name in missing_tools:
                proposals.append(
                    ActionProposal(
                        action_name=tool_name,
                        utility=base + 0.75,
                        event=event,
                        reason="minimum_modality_coverage",
                    )
                )
            source = max(0.0, event.factor_value("source_likelihood"))
            symptom = max(0.0, event.factor_value("symptomness"))
            role_conflict = min(source, symptom)
            if role_conflict > 0.35 and not event.has_tool("verify_trace"):
                proposals.append(
                    ActionProposal(
                        action_name="verify_trace",
                        utility=base + role_conflict + 0.35,
                        event=event,
                        reason="root_symptom_role_conflict",
                    )
                )
            if (
                event.posterior >= 0.04
                and event.status_entropy() >= 0.45
                and not event.has_tool("run_counterfactual")
            ):
                proposals.append(
                    ActionProposal(
                        action_name="run_counterfactual",
                        utility=base + 0.25 * event.posterior,
                        event=event,
                        reason="high_status_entropy_verification",
                    )
                )
            cmi_profile = self.tool_context.cmi_profiles.get(event.component) or {}
            if (
                cmi_profile
                and event.posterior >= 0.025
                and not event.has_tool("mechanism_intervention")
            ):
                proposals.append(
                    ActionProposal(
                        action_name="mechanism_intervention",
                        utility=base
                        + 1.10
                        + 0.30 * max(0.0, float(cmi_profile.get("cmi_score", 0.0) or 0.0))
                        + 0.25
                        * max(
                            0.0,
                            float(cmi_profile.get("conditional_residual_z_norm", 0.0) or 0.0),
                        ),
                        event=event,
                        reason="causal_mechanism_uncertainty_reduction",
                    )
                )
            role = root_role_features(event)
            hotspot_gap = role["hotspot_symptom"] - role["source_isolation"]
            margin_abs = abs(role["source_margin"])
            if (
                hotspot_gap >= 0.20
                and event.posterior >= 0.03
                and not event.has_tool("run_counterfactual")
            ):
                proposals.append(
                    ActionProposal(
                        action_name="run_counterfactual",
                        utility=base + 0.60 + hotspot_gap,
                        event=event,
                        reason="hotspot_vs_root_disambiguation",
                    )
                )
            if (
                event.posterior >= 0.03
                and not event.has_tool("residual_collapse")
                and (
                    hotspot_gap >= 0.12
                    or margin_abs <= 0.35
                    or event.status_entropy() >= 0.42
                )
            ):
                proposals.append(
                    ActionProposal(
                        action_name="residual_collapse",
                        utility=base
                        + 0.85
                        + max(0.0, hotspot_gap)
                        + 0.25 * (1.0 - min(1.0, margin_abs)),
                        event=event,
                        reason="residual_collapse_root_test",
                    )
                )
            if round_idx == 0 and not event.has_tool("split_event"):
                proposals.append(
                    ActionProposal(
                        action_name="split_event",
                        utility=0.10 + 0.15 * event.status_entropy(),
                        event=event,
                        reason="event_cardinality_check",
                    )
                )

        for left, right in zip(ranked[:4], ranked[1:5]):
            margin = abs(float(left.posterior) - float(right.posterior))
            if margin <= 0.16 or left.status_entropy() > 0.50 or right.status_entropy() > 0.50:
                utility = 0.70 * (1.0 - min(1.0, margin / 0.25))
                proposals.append(
                    ActionProposal(
                        action_name="compare_pair",
                        utility=utility + 0.20 * max(left.posterior, right.posterior),
                        event=left,
                        other_event=right,
                        reason="pairwise_margin_uncertainty",
                    )
                )
            if left.component == right.component and left.reason == right.reason:
                proposals.append(
                    ActionProposal(
                        action_name="merge_events",
                        utility=0.60,
                        event=left,
                        other_event=right,
                        reason="duplicate_event_check",
                    )
                )

        proposals.sort(key=lambda proposal: proposal.utility, reverse=True)
        deduped: List[ActionProposal] = []
        seen = set()
        for proposal in proposals:
            key = (
                proposal.action_name,
                proposal.event.event_id if proposal.event is not None else "",
                proposal.other_event.event_id if proposal.other_event is not None else "",
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(proposal)
        return deduped

    def _missing_required_tools(self, event: FaultEvent, has_logs: bool) -> List[str]:
        required = ["inspect_metric", "verify_trace"]
        if has_logs:
            required.append("inspect_log")
        return [tool_name for tool_name in required if not event.has_tool(tool_name)]

    def _event_uncertainty_value(self, event: FaultEvent) -> float:
        posterior_uncertainty = 1.0 - abs(2.0 * float(event.posterior) - 1.0)
        role = root_role_features(event)
        role_conflict = max(0.0, role["hotspot_symptom"] - role["source_isolation"])
        return (
            0.50 * event.status_entropy()
            + 0.35 * posterior_uncertainty
            + 0.15 * float(event.status_probs.get("unresolved", 0.0))
            + 0.25 * role_conflict
        )

    def _uncertainty_snapshot(self, events: Sequence[FaultEvent]) -> Dict[str, Any]:
        p = np.array([max(float(event.posterior), 1e-9) for event in events], dtype=float)
        if p.size:
            p = p / max(float(np.sum(p)), 1e-9)
            event_entropy = -float(np.sum(p * np.log(p))) / math.log(max(2, p.size))
        else:
            event_entropy = 0.0
        status_entropies = [event.status_entropy() for event in events]
        ranked = self._rank_events(events)
        selection_ranked = sorted(events, key=root_selection_score, reverse=True)
        return {
            "event_entropy": round(float(event_entropy), 6),
            "mean_status_entropy": round(
                float(np.mean(status_entropies)) if status_entropies else 0.0, 6
            ),
            "coverage_uncertainty": round(
                1.0
                - sum(
                    1
                    for event in events
                    if event.has_tool("inspect_metric") and event.has_tool("verify_trace")
                )
                / max(1, len(events)),
                6,
            ),
            "top_gap": round(float(self._top_gap(events)), 6),
            "top_event": ranked[0].event_id if ranked else "",
            "top_selection_event": selection_ranked[0].event_id if selection_ranked else "",
            "top_selection_score": round(
                float(root_selection_score(selection_ranked[0])) if selection_ranked else 0.0,
                6,
            ),
            "covered_events": sum(
                1
                for event in events
                if event.has_tool("inspect_metric") and event.has_tool("verify_trace")
            ),
            "event_count": len(events),
        }

    def _should_stop(
        self, events: Sequence[FaultEvent], uncertainty: Dict[str, Any]
    ) -> bool:
        if not events:
            return True
        top = self._rank_events(events)[0]
        coverage_ratio = uncertainty["covered_events"] / max(1, uncertainty["event_count"])
        return (
            float(uncertainty["event_entropy"]) <= 0.45
            and float(uncertainty["mean_status_entropy"]) <= 0.45
            and float(uncertainty["top_gap"]) >= 0.20
            and top.root_probability() >= 0.40
            and coverage_ratio >= 0.60
        )

    def _source_isolation(self, candidate: CandidateFrame) -> Tuple[float, Dict[str, Any]]:
        features = candidate.structural_features
        source_protection = self._feature(features, "mask_source_protection")
        uniqueness = self._feature(features, "structure_structural_uniqueness")
        hard_negative = self._feature(features, "structure_hard_negative_resistance")
        root_source = self._feature(features, "noise_root_source_score")
        temporal_source = self._feature(features, "noise_temporal_source_score", "delay_source_time_consistency")
        multi_view = self._feature(features, "structure_multi_view_consistency")
        source_margin = max(0.0, float(candidate.source_likelihood) - float(candidate.symptomness))
        value = _clip01(
            0.22 * source_protection
            + 0.18 * uniqueness
            + 0.18 * hard_negative
            + 0.18 * root_source
            + 0.12 * temporal_source
            + 0.08 * multi_view
            + 0.18 * source_margin
        )
        payload = {
            "source_isolation": round(value, 6),
            "source_protection": round(source_protection, 6),
            "structural_uniqueness": round(uniqueness, 6),
            "hard_negative_resistance": round(hard_negative, 6),
            "root_source_score": round(root_source, 6),
            "temporal_source_score": round(temporal_source, 6),
            "multi_view_consistency": round(multi_view, 6),
            "source_margin": round(source_margin, 6),
        }
        return value, payload

    def _hotspot_symptom(self, candidate: CandidateFrame) -> Tuple[float, Dict[str, Any]]:
        features = candidate.structural_features
        hotspot_bias = self._feature(features, "noise_hotspot_bias")
        local_hub = self._feature(features, "mask_local_hub_pressure")
        replaceability = max(
            self._feature(features, "subspace_replaceability"),
            self._feature(features, "mask_replaceability_penalty"),
        )
        reverb = max(
            self._feature(features, "mask_gated_reverb_penalty"),
            self._feature(features, "noise_reverb_mass"),
            self._feature(features, "subspace_sector_reverb_ratio"),
        )
        common_mode = self._feature(features, "subspace_local_common_mode_alignment")
        source_protection = self._feature(features, "mask_source_protection")
        root_source = self._feature(features, "noise_root_source_score")
        symptom_margin = max(0.0, float(candidate.symptomness) - float(candidate.source_likelihood))
        value = _clip01(
            0.20 * hotspot_bias
            + 0.20 * local_hub
            + 0.18 * replaceability
            + 0.18 * reverb
            + 0.12 * common_mode
            + 0.22 * symptom_margin
            - 0.12 * source_protection
            - 0.10 * root_source
        )
        payload = {
            "hotspot_symptom": round(value, 6),
            "hotspot_bias": round(hotspot_bias, 6),
            "local_hub_pressure": round(local_hub, 6),
            "replaceability": round(replaceability, 6),
            "reverb_pressure": round(reverb, 6),
            "common_mode_alignment": round(common_mode, 6),
            "symptom_margin": round(symptom_margin, 6),
            "source_protection": round(source_protection, 6),
            "root_source_score": round(root_source, 6),
        }
        return value, payload

    def _feature(self, features: Dict[str, Any], *names: str) -> float:
        for name in names:
            if name in features:
                try:
                    return float(features.get(name, 0.0) or 0.0)
                except (TypeError, ValueError):
                    return 0.0
            for prefix in ("noise", "structure", "delay", "beam", "subspace", "mask", "reverb"):
                marker = f"{prefix}_"
                if not name.startswith(marker):
                    continue
                group = features.get(prefix)
                if group is None and prefix == "mask":
                    group = features.get("reverb")
                if isinstance(group, dict):
                    nested_name = name[len(marker):]
                    try:
                        return float(group.get(nested_name, 0.0) or 0.0)
                    except (TypeError, ValueError):
                        return 0.0
        return 0.0


def _clip01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))
