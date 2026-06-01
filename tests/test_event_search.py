from __future__ import annotations

from prism_v3.noise_native.event_search import EventSearchConfig, search_event_sets
from prism_v3.noise_native.evidence_frame import CandidateFrame, EvidenceFrame


def test_event_search_selects_high_source_candidates() -> None:
    frame = EvidenceFrame(
        source="test",
        applied=True,
        candidates=[
            CandidateFrame("c1", "a", "a", prior_mass=0.6, source_likelihood=0.8, symptomness=0.1),
            CandidateFrame("c2", "b", "b", prior_mass=0.5, source_likelihood=0.7, symptomness=0.1),
            CandidateFrame("c3", "c", "c", prior_mass=0.7, source_likelihood=0.1, symptomness=0.9),
        ],
    )
    result = search_event_sets(frame, fault_count=2, config=EventSearchConfig(beam_width=4, max_events=2))
    assert len(result.events) == 2
    assert "a" in result.components()
    assert result.score > 0
