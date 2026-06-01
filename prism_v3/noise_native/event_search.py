"""Beam search over fault-event sets.

This module keeps event-set search separate from label/evaluation code. It
operates only on runtime EvidenceFrame candidates and can be used by later
synthesis stages without reading GT.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from .evidence_frame import CandidateFrame, EvidenceFrame, canonical_entity_name
from .fault_event import FaultEvent


@dataclass
class EventSearchConfig:
    beam_width: int = 6
    max_events: int = 3
    candidate_limit: int = 12
    diversity_bonus: float = 0.08
    symptom_penalty: float = 0.20
    duplicate_penalty: float = 0.60


@dataclass
class EventSet:
    events: List[FaultEvent]
    score: float
    debug: Dict[str, Any] = field(default_factory=dict)

    def components(self) -> List[str]:
        return [event.component for event in self.events]

    def key(self) -> Tuple[str, ...]:
        return tuple(sorted(canonical_entity_name(component) for component in self.components()))


class EventSetSearcher:
    def __init__(self, config: EventSearchConfig | None = None) -> None:
        self.config = config or EventSearchConfig()

    def search(self, evidence_frame: EvidenceFrame, *, fault_count: int = 1) -> EventSet:
        candidates = evidence_frame.top_candidates(self.config.candidate_limit)
        if not candidates:
            return EventSet(events=[], score=0.0, debug={"reason": "no_candidates"})
        target = max(1, min(int(fault_count or 1), int(self.config.max_events)))
        beam = [EventSet(events=[self._event_from_candidate(candidates[0], 1)], score=0.0)]
        beam[0].score = self._score(beam[0])
        for cand_idx, candidate in enumerate(candidates[1:], start=2):
            expanded: List[EventSet] = []
            for state in beam:
                expanded.append(state)
                if len(state.events) < target:
                    expanded.append(
                        EventSet(
                            events=list(state.events) + [self._event_from_candidate(candidate, cand_idx)],
                            score=0.0,
                            debug={"op": "add", "candidate": candidate.object_id},
                        )
                    )
                expanded.append(self._replace_weakest(state, candidate, cand_idx))
            dedup: Dict[Tuple[str, ...], EventSet] = {}
            for state in expanded:
                merged = self._merge_duplicates(state)
                merged.score = self._score(merged)
                key = merged.key()
                if key not in dedup or merged.score > dedup[key].score:
                    dedup[key] = merged
            beam = sorted(dedup.values(), key=lambda item: item.score, reverse=True)[: self.config.beam_width]
        best = max(beam, key=lambda item: item.score)
        best.debug = {
            **dict(best.debug),
            "beam_width": self.config.beam_width,
            "candidate_count": len(candidates),
            "target_fault_count": target,
            "selected_components": best.components(),
        }
        return best

    def _event_from_candidate(self, candidate: CandidateFrame, rank: int) -> FaultEvent:
        reason = ""
        if candidate.reason_candidates:
            reason = str(max(candidate.reason_candidates, key=lambda item: float(item.get("score", 0.0))).get("reason", ""))
        event = FaultEvent(
            event_id=f"search_event_{rank}:{candidate.object_id}",
            component=str(candidate.component_id or candidate.object_id),
            reason=reason,
            time=_candidate_time(candidate),
            posterior=float(candidate.prior_mass),
            candidate_id=candidate.candidate_id,
            reason_candidates=list(candidate.reason_candidates),
        )
        event.factors.update(
            {
                "NoiseLab_logit": float(candidate.calibrated_logit),
                "source_likelihood": float(candidate.source_likelihood),
                "symptomness": float(candidate.symptomness),
            }
        )
        return event

    def _replace_weakest(self, state: EventSet, candidate: CandidateFrame, rank: int) -> EventSet:
        if not state.events:
            return EventSet(events=[self._event_from_candidate(candidate, rank)], score=0.0, debug={"op": "replace_empty"})
        replacement = self._event_from_candidate(candidate, rank)
        events = list(state.events)
        weakest_idx = min(range(len(events)), key=lambda idx: self._event_score(events[idx]))
        events[weakest_idx] = replacement
        return EventSet(events=events, score=0.0, debug={"op": "replace", "candidate": candidate.object_id})

    def _merge_duplicates(self, state: EventSet) -> EventSet:
        seen: Dict[str, FaultEvent] = {}
        for event in state.events:
            key = canonical_entity_name(event.component)
            existing = seen.get(key)
            if existing is None or self._event_score(event) > self._event_score(existing):
                seen[key] = event
        merged = list(seen.values())
        debug = dict(state.debug)
        if len(merged) != len(state.events):
            debug["merged_duplicates"] = len(state.events) - len(merged)
        return EventSet(events=merged, score=state.score, debug=debug)

    def _score(self, state: EventSet) -> float:
        if not state.events:
            return 0.0
        score = sum(self._event_score(event) for event in state.events)
        families = {canonical_entity_name(event.component)[:4] for event in state.events}
        score += self.config.diversity_bonus * max(0, len(families) - 1)
        duplicates = len(state.events) - len(state.key())
        score -= self.config.duplicate_penalty * max(0, duplicates)
        return float(score)

    def _event_score(self, event: FaultEvent) -> float:
        source = float(event.factors.get("source_likelihood", 0.0))
        symptom = float(event.factors.get("symptomness", 0.0))
        return float(event.posterior + 0.25 * source - self.config.symptom_penalty * max(0.0, symptom - source))


def search_event_sets(
    evidence_frame: EvidenceFrame,
    *,
    fault_count: int = 1,
    config: EventSearchConfig | None = None,
) -> EventSet:
    return EventSetSearcher(config).search(evidence_frame, fault_count=fault_count)


def _candidate_time(candidate: CandidateFrame) -> float | None:
    for item in candidate.time_candidates:
        if "timestamp" in item:
            try:
                return float(item["timestamp"])
            except (TypeError, ValueError):
                return None
    return None
