"""Candidate-space generation for Scheme B v2."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from refute_b_v2.background_suppression import AnomalyEvent, BackgroundCalibrator, ComponentScore, component_scores
from refute_b_v2.rules import Candidate


@dataclass(frozen=True)
class CandidateHypothesis:
    candidate: Candidate
    component_score: ComponentScore
    seed_sources: tuple[str, ...] = ()
    candidate_times: tuple[int, ...] = ()

    def to_dict(self) -> dict:
        return {
            "candidate": self.candidate.to_dict(),
            "component_score": self.component_score.to_dict(),
            "seed_sources": list(self.seed_sources),
            "candidate_times": list(self.candidate_times),
        }


@dataclass
class CandidateGenerator:
    top_k: int = 5
    per_reason_k: int = 2
    calibrator: BackgroundCalibrator | None = None

    def generate(self, events: Iterable[AnomalyEvent], fallback_services: Iterable[str] = (),
                 fallback_reasons: Iterable[str] = ("network latency",)) -> list[CandidateHypothesis]:
        events = list(events)
        rows = component_scores(events, self.calibrator)
        selected = []
        per_reason_counts: dict[str, int] = {}
        source_map = _source_map(events)
        for row in rows:
            count = per_reason_counts.get(row.reason, 0)
            if count >= self.per_reason_k:
                continue
            per_reason_counts[row.reason] = count + 1
            selected.append(CandidateHypothesis(
                candidate=Candidate(row.component, row.reason),
                component_score=row,
                seed_sources=tuple(sorted(source_map.get((row.component, row.reason), ()))),
                candidate_times=row.timestamps,
            ))
            if len(selected) >= self.top_k:
                break
        if selected:
            return selected
        # Explicit fallback prevents silent empty candidate sets and marks low evidence.
        fallback = []
        for service in fallback_services:
            for reason in fallback_reasons:
                score = ComponentScore(str(service), str(reason), 0.0, 0.0, 0.0, None, 0, 0.0, 0.0, 0.0, ())
                fallback.append(CandidateHypothesis(Candidate(str(service), str(reason)), score, ("fallback",), ()))
                if len(fallback) >= self.top_k:
                    return fallback
        return fallback


def _source_map(events: list[AnomalyEvent]) -> dict[tuple[str, str], set[str]]:
    out: dict[tuple[str, str], set[str]] = {}
    for event in events:
        out.setdefault((str(event.component), str(event.reason)), set()).add(str(event.source))
    return out
