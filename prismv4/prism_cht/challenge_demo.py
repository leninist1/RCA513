"""Deterministic challenge demo scenarios for PRISM-CHT Phase 3.

Provides three pre-configured factories that build complete
single-challenge adversarial review scenarios:

- ``build_demo_survival_challenge()`` — nomination survives challenge
- ``build_demo_refutation_challenge()`` — challenge refutes nomination
- ``build_demo_inconclusive_challenge()`` — challenge is inconclusive

Each factory creates fresh, independent infrastructure — no shared
mutable state between scenarios.
"""

from __future__ import annotations

from prismv4.prism_cht.action_gate import ActionGate
from prismv4.prism_cht.action_schema import DiscriminativeAction
from prismv4.prism_cht.assessment_gate import EvidenceAssessmentGate
from prismv4.prism_cht.challenge_gate import ChallengeReviewGate
from prismv4.prism_cht.challenge_types import (
    ChallengeProposal,
    ChallengeResolution,
    ChallengeVerdict,
)
from prismv4.prism_cht.challenger_controller import ChallengerController
from prismv4.prism_cht.challenger_policy import ScriptedChallengerPolicy
from prismv4.prism_cht.evidence_graph import EvidenceGraph
from prismv4.prism_cht.executor import InvestigationExecutor
from prismv4.prism_cht.hypothesis import CausalHypothesis, HypothesisStatus
from prismv4.prism_cht.lead_controller import LeadTournamentController
from prismv4.prism_cht.lead_policy import ScriptedInvestigationTurn, ScriptedLeadPolicy
from prismv4.prism_cht.telemetry_store import (
    MockTelemetryStore,
    OnsetObservation,
    TraceHop,
    TracePath,
)
from prismv4.prism_cht.tool_registry import build_default_tool_registry
from prismv4.prism_cht.tournament_types import (
    AssessmentOutcome,
    EvidenceAssessment,
    EvidenceLinkProposal,
    EvidenceRelation,
    HypothesisStatusUpdate,
    LeadNomination,
)


# ===========================================================================
# Shared helpers
# ===========================================================================


def _build_base_hypotheses():
    """Build the standard H1 (os_009 CPU) and H2 (db_002 pool) hypotheses."""
    h1 = CausalHypothesis(
        hypothesis_id="H-os_009-cpu",
        root_component="os_009",
        reason_family="high cpu usage",
        onset_interval=(1000.0, 2000.0),
        local_trigger="os_009 CPU utilization exceeds 95% threshold",
        propagation_path=["os_009", "app_003"],
        explained_symptoms=[
            "app_003 latency spike",
            "app_003 error rate increase",
        ],
        predicted_observations=[
            "os_009 CPU peak is the earliest anomaly in the system",
            "app_003 symptoms are downstream of os_009",
        ],
        falsifiers=[
            "A different component (e.g. db_002) has an earlier anomaly onset",
            "No trace path exists from os_009 to app_003",
        ],
    )

    h2 = CausalHypothesis(
        hypothesis_id="H-db_002-pool",
        root_component="db_002",
        reason_family="connection pool exhaustion",
        onset_interval=(1000.0, 2000.0),
        local_trigger="db_002 connection pool exhausted, queued requests > 100",
        propagation_path=["db_002", "app_003"],
        explained_symptoms=[
            "app_003 latency spike",
            "app_003 error rate increase",
        ],
        predicted_observations=[
            "db_002 connection pool anomaly onset precedes os_009 CPU spike",
            "Explicit trace path exists from db_002 to app_003",
        ],
        falsifiers=[
            "os_009 CPU anomaly precedes db_002 pool exhaustion",
            "No trace path exists from db_002 to app_003",
        ],
    )
    return h1, h2


def _build_base_onset_observations():
    """Standard onset observations: db_002 before os_009."""
    return [
        OnsetObservation(
            component="db_002",
            signal="connection_pool_queued",
            onset_time=1100.0,
            source="prom_db",
        ),
        OnsetObservation(
            component="db_002",
            signal="connection_pool_errors",
            onset_time=1110.0,
            source="prom_db",
        ),
        OnsetObservation(
            component="os_009",
            signal="cpu",
            onset_time=1250.0,
            source="prom",
        ),
        OnsetObservation(
            component="os_009",
            signal="load_average",
            onset_time=1260.0,
            source="prom",
        ),
        OnsetObservation(
            component="app_003",
            signal="latency_p99",
            onset_time=1300.0,
            source="prom_app",
        ),
    ]


def _build_base_trace_paths():
    """Standard trace path: db_002 -> app_003."""
    return [
        TracePath(
            hops=(
                TraceHop(
                    source_component="db_002",
                    target_component="app_003",
                    timestamp=1120.0,
                    latency_ms=45.0,
                    status="error",
                ),
            )
        )
    ]


def _build_lead_phase(*, store, graph, gate, registry, assessment_gate):
    """Build Lead tournament infrastructure and scripted policy."""
    executor = InvestigationExecutor(
        gate=gate,
        registry=registry,
        graph=graph,
        store=store,
    )

    controller = LeadTournamentController(
        executor=executor,
        assessment_gate=assessment_gate,
        graph=graph,
        max_rounds=4,
    )

    # Round 1: compare_onset_order
    onset_component_scope = ["db_002", "os_009", "app_003"]
    onset_signal_scope = [
        "connection_pool_queued",
        "connection_pool_errors",
        "cpu",
        "load_average",
        "latency_p99",
    ]

    action_r1 = DiscriminativeAction(
        action_id="R1-compare-onset",
        action_type="run_discriminative_test",
        target_hypothesis_ids=("H-os_009-cpu", "H-db_002-pool"),
        question=(
            "Which root-cause candidate exhibits the earliest observable "
            "anomaly onset — os_009 CPU or db_002 connection pool?"
        ),
        tool_name="compare_onset_order",
        args={
            "component_scope": onset_component_scope,
            "signal_scope": onset_signal_scope,
            "time_window": [1000.0, 2000.0],
        },
        expected_outcomes={
            "H-os_009-cpu": "os_009 CPU spike is the earliest anomaly",
            "H-db_002-pool": "db_002 connection pool exhaustion is the earliest anomaly",
        },
        why_discriminative=(
            "os_009 and db_002 predict opposite onset order; "
            "the earlier failure is more likely the root cause"
        ),
    )

    sig_r1 = action_r1.query_signature()
    evidence_id_r1 = "evidence:" + sig_r1

    assessment_r1 = EvidenceAssessment(
        action_id=action_r1.action_id,
        evidence_id=evidence_id_r1,
        outcome=AssessmentOutcome.INFORMATIVE,
        links=(
            EvidenceLinkProposal(
                hypothesis_id="H-db_002-pool",
                evidence_id=evidence_id_r1,
                relation=EvidenceRelation.SUPPORTS,
                rationale=(
                    "db_002 connection pool onset at 1100.0 precedes "
                    "os_009 CPU onset at 1250.0 by 150.0 seconds; "
                    "earlier anomaly supports db_002 as root cause"
                ),
            ),
            EvidenceLinkProposal(
                hypothesis_id="H-os_009-cpu",
                evidence_id=evidence_id_r1,
                relation=EvidenceRelation.CONTRADICTS,
                rationale=(
                    "os_009 predicted to be the earliest anomaly, but "
                    "db_002 onset is earlier; this contradicts the "
                    "hypothesis that os_009 CPU is the root cause"
                ),
            ),
        ),
        status_updates=(
            HypothesisStatusUpdate(
                hypothesis_id="H-db_002-pool",
                new_status=HypothesisStatus.SUPPORTED,
                rationale=(
                    "db_002 connection pool exhaustion is the earliest "
                    "observable anomaly, consistent with the hypothesis"
                ),
            ),
            HypothesisStatusUpdate(
                hypothesis_id="H-os_009-cpu",
                new_status=HypothesisStatus.WEAKENED,
                rationale=(
                    "os_009 predicted earliest anomaly is contradicted; "
                    "db_002 onset precedes os_009 CPU"
                ),
            ),
        ),
        rationale=(
            "Onset order comparison reveals db_002 connection pool "
            "exhaustion at 1100.0 precedes os_009 CPU spike at 1250.0, "
            "weakening the os_009 CPU hypothesis and supporting the "
            "db_002 connection pool hypothesis"
        ),
    )

    # Round 2: inspect_trace_path
    action_r2 = DiscriminativeAction(
        action_id="R2-inspect-trace",
        action_type="run_discriminative_test",
        target_hypothesis_ids=("H-os_009-cpu", "H-db_002-pool"),
        question=(
            "Is there a trace path from db_002 to app_003 "
            "that explains the observed symptoms?"
        ),
        tool_name="inspect_trace_path",
        args={
            "source_component": "db_002",
            "target_component": "app_003",
            "time_window": [1000.0, 2000.0],
            "max_hops": 4,
            "max_paths": 10,
        },
        expected_outcomes={
            "H-os_009-cpu": "No trace path from db_002 to app_003",
            "H-db_002-pool": "Trace path exists from db_002 to app_003",
        },
        why_discriminative=(
            "db_002 hypothesis requires an explicit trace path to "
            "app_003 to explain downstream symptoms"
        ),
    )

    sig_r2 = action_r2.query_signature()
    evidence_id_r2 = "evidence:" + sig_r2

    assessment_r2 = EvidenceAssessment(
        action_id=action_r2.action_id,
        evidence_id=evidence_id_r2,
        outcome=AssessmentOutcome.INFORMATIVE,
        links=(
            EvidenceLinkProposal(
                hypothesis_id="H-db_002-pool",
                evidence_id=evidence_id_r2,
                relation=EvidenceRelation.SUPPORTS,
                rationale=(
                    "Explicit trace path db_002 -> app_003 found with "
                    "latency 45ms, status=error; confirms downstream "
                    "propagation from db_002 to app_003"
                ),
            ),
        ),
        status_updates=(),
        rationale=(
            "Trace inspection confirms a direct path from db_002 to "
            "app_003, providing propagation evidence for the connection "
            "pool exhaustion hypothesis"
        ),
    )

    # Nomination: H2 wins, H1 is the addressed competitor
    nomination = LeadNomination(
        hypothesis_id="H-db_002-pool",
        supporting_evidence_ids=(evidence_id_r1, evidence_id_r2),
        addressed_competitor_ids=("H-os_009-cpu",),
        rationale=(
            "db_002 connection pool exhaustion is nominated because: "
            "(1) onset order shows db_002 anomalies (1100.0s) precede "
            "os_009 CPU spike (1250.0s); "
            "(2) explicit trace path from db_002 to app_003 confirms "
            "propagation; "
            "competitor H1 (os_009 CPU) is WEAKENED by earlier db_002 onset. "
            "Challenge required to rule out confounders."
        ),
    )

    lead_policy = ScriptedLeadPolicy(
        turns=[
            ScriptedInvestigationTurn(action=action_r1, assessment=assessment_r1),
            ScriptedInvestigationTurn(action=action_r2, assessment=assessment_r2),
        ],
        nomination=nomination,
    )

    return controller, lead_policy, action_r1, action_r2


def _build_challenger_infra(*, store, graph, gate, registry, assessment_gate):
    """Build Challenger infrastructure sharing the same graph/gate/store."""
    executor = InvestigationExecutor(
        gate=gate,
        registry=registry,
        graph=graph,
        store=store,
    )

    challenge_gate = ChallengeReviewGate()

    challenge_controller = ChallengerController(
        executor=executor,
        assessment_gate=assessment_gate,
        challenge_gate=challenge_gate,
        graph=graph,
    )

    return challenge_controller


# ===========================================================================
# 1. Survival demo
# ===========================================================================


def build_demo_survival_challenge() -> tuple[
    LeadTournamentController,
    tuple[CausalHypothesis, ...],
    ScriptedLeadPolicy,
    ChallengerController,
    ScriptedChallengerPolicy,
]:
    """Build a scenario where the nomination survives the challenge.

    Challenger investigates db_002 raw records which confirm ongoing
    pool exhaustion — the challenge is answered and H2 survives.
    """

    h1, h2 = _build_base_hypotheses()

    raw_records_survival = [
        {
            "modality": "metrics",
            "component": "db_002",
            "timestamp": 1100.0,
            "payload": {
                "connection_pool_queued": 150,
                "connection_pool_active": 0,
                "connection_pool_errors": 12,
                "status": "exhausted",
                "duration_seconds": 300,
            },
        },
        {
            "modality": "metrics",
            "component": "db_002",
            "timestamp": 1200.0,
            "payload": {
                "connection_pool_queued": 200,
                "connection_pool_active": 0,
                "connection_pool_errors": 25,
                "status": "exhausted",
                "duration_seconds": 200,
            },
        },
    ]

    store = MockTelemetryStore(
        onset_observations=_build_base_onset_observations(),
        trace_paths=_build_base_trace_paths(),
        raw_records=raw_records_survival,
    )

    gate = ActionGate()
    registry = build_default_tool_registry()
    graph = EvidenceGraph()
    assessment_gate = EvidenceAssessmentGate()

    lead_ctrl, lead_policy, _, _ = _build_lead_phase(
        store=store, graph=graph, gate=gate,
        registry=registry, assessment_gate=assessment_gate,
    )

    challenge_ctrl = _build_challenger_infra(
        store=store, graph=graph, gate=gate,
        registry=registry, assessment_gate=assessment_gate,
    )

    # Challenger action: retrieve_raw_evidence for db_002
    chall_action = DiscriminativeAction(
        action_id="C1-retrieve-raw",
        action_type="run_discriminative_test",
        target_hypothesis_ids=("H-db_002-pool", "H-os_009-cpu"),
        question=(
            "Do raw telemetry records for db_002 show sustained "
            "connection pool exhaustion during the fault window?"
        ),
        tool_name="retrieve_raw_evidence",
        args={
            "modality": "metrics",
            "component_scope": ["db_002"],
            "time_window": [1000.0, 2000.0],
            "limit": 20,
        },
        expected_outcomes={
            "H-db_002-pool": "Raw records confirm sustained pool exhaustion",
            "H-os_009-cpu": "Raw records show pool was not exhausted",
        },
        why_discriminative=(
            "If db_002 shows sustained pool exhaustion in raw records, "
            "the nomination is corroborated; if records show recovery "
            "before the symptom window, the nomination is refuted"
        ),
    )

    chall_sig = chall_action.query_signature()
    chall_evidence_id = "evidence:" + chall_sig

    chall_proposal = ChallengeProposal(
        challenge_id="C1",
        nominated_hypothesis_id="H-db_002-pool",
        competitor_hypothesis_ids=("H-os_009-cpu",),
        challenge_claim=(
            "db_002 connection pool exhaustion may be a transient spike; "
            "if pool recovered before the main symptom window, db_002 "
            "cannot explain sustained faults"
        ),
        falsification_target=(
            "Show that db_002 connection pool exhaustion was sustained "
            "throughout the symptom window, refuting the transient hypothesis"
        ),
        action=chall_action,
        rationale=(
            "Raw telemetry records for db_002 will reveal whether pool "
            "exhaustion was sustained or transient"
        ),
    )

    chall_assessment = EvidenceAssessment(
        action_id=chall_action.action_id,
        evidence_id=chall_evidence_id,
        outcome=AssessmentOutcome.INFORMATIVE,
        links=(
            EvidenceLinkProposal(
                hypothesis_id="H-db_002-pool",
                evidence_id=chall_evidence_id,
                relation=EvidenceRelation.SUPPORTS,
                rationale=(
                    "Raw records show db_002 connection pool exhausted at "
                    "1100s (queued=150, errors=12) and again at 1200s "
                    "(queued=200, errors=25); pool exhaustion is sustained, "
                    "not transient"
                ),
            ),
        ),
        status_updates=(
            HypothesisStatusUpdate(
                hypothesis_id="H-db_002-pool",
                new_status=HypothesisStatus.SURVIVED,
                rationale=(
                    "Challenge attempted to falsify via transient argument, "
                    "but raw records confirm sustained pool exhaustion; "
                    "nomination survives adversarial review"
                ),
            ),
        ),
        rationale=(
            "Raw records for db_002 demonstrate sustained connection pool "
            "exhaustion across the analysis window.  The challenger's "
            "transient hypothesis is refuted by the evidence, and "
            "the nomination survives."
        ),
    )

    chall_resolution = ChallengeResolution(
        challenge_id="C1",
        action_id=chall_action.action_id,
        evidence_id=chall_evidence_id,
        verdict=ChallengeVerdict.NOMINATION_SURVIVED,
        assessment=chall_assessment,
        rationale=(
            "Evidence confirms sustained pool exhaustion; "
            "challenge claim of transient failure is not supported"
        ),
    )

    chall_policy = ScriptedChallengerPolicy(
        proposal=chall_proposal,
        resolution=chall_resolution,
    )

    return lead_ctrl, (h1, h2), lead_policy, challenge_ctrl, chall_policy


# ===========================================================================
# 2. Refutation demo
# ===========================================================================


def build_demo_refutation_challenge() -> tuple[
    LeadTournamentController,
    tuple[CausalHypothesis, ...],
    ScriptedLeadPolicy,
    ChallengerController,
    ScriptedChallengerPolicy,
]:
    """Build a scenario where the challenge refutes the nomination.

    Challenger investigates db_002 raw records which show recovery
    before the main fault window — H2 is weakened and returned to Lead.
    """

    h1, h2 = _build_base_hypotheses()

    raw_records_refutation = [
        {
            "modality": "metrics",
            "component": "db_002",
            "timestamp": 1050.0,
            "payload": {
                "connection_pool_queued": 80,
                "connection_pool_active": 20,
                "connection_pool_errors": 2,
                "status": "normal",
                "duration_seconds": 0,
            },
        },
        {
            "modality": "metrics",
            "component": "db_002",
            "timestamp": 1080.0,
            "payload": {
                "connection_pool_queued": 0,
                "connection_pool_active": 50,
                "connection_pool_errors": 0,
                "status": "recovered",
                "duration_seconds": 0,
            },
        },
    ]

    store = MockTelemetryStore(
        onset_observations=_build_base_onset_observations(),
        trace_paths=_build_base_trace_paths(),
        raw_records=raw_records_refutation,
    )

    gate = ActionGate()
    registry = build_default_tool_registry()
    graph = EvidenceGraph()
    assessment_gate = EvidenceAssessmentGate()

    lead_ctrl, lead_policy, _, _ = _build_lead_phase(
        store=store, graph=graph, gate=gate,
        registry=registry, assessment_gate=assessment_gate,
    )

    challenge_ctrl = _build_challenger_infra(
        store=store, graph=graph, gate=gate,
        registry=registry, assessment_gate=assessment_gate,
    )

    # Challenger action: retrieve_raw_evidence for db_002
    chall_action = DiscriminativeAction(
        action_id="C1-retrieve-raw",
        action_type="run_discriminative_test",
        target_hypothesis_ids=("H-db_002-pool", "H-os_009-cpu"),
        question=(
            "Do raw telemetry records for db_002 show that the "
            "connection pool recovered before the main symptom window?"
        ),
        tool_name="retrieve_raw_evidence",
        args={
            "modality": "metrics",
            "component_scope": ["db_002"],
            "time_window": [1000.0, 2000.0],
            "limit": 20,
        },
        expected_outcomes={
            "H-db_002-pool": "Records would show sustained exhaustion",
            "H-os_009-cpu": "Records would show recovery before symptoms",
        },
        why_discriminative=(
            "If db_002 pool recovered before the main symptom window, "
            "then connection pool exhaustion cannot be the root cause"
        ),
    )

    chall_sig = chall_action.query_signature()
    chall_evidence_id = "evidence:" + chall_sig

    chall_proposal = ChallengeProposal(
        challenge_id="C1",
        nominated_hypothesis_id="H-db_002-pool",
        competitor_hypothesis_ids=("H-os_009-cpu",),
        challenge_claim=(
            "db_002 connection pool exhaustion may have recovered before "
            "the main fault window; if so, it cannot explain sustained "
            "app_003 symptoms"
        ),
        falsification_target=(
            "Show that db_002 connection pool remained exhausted throughout "
            "the symptom window"
        ),
        action=chall_action,
        rationale=(
            "Raw records for db_002 will show whether the pool recovered "
            "or remained exhausted"
        ),
    )

    chall_assessment = EvidenceAssessment(
        action_id=chall_action.action_id,
        evidence_id=chall_evidence_id,
        outcome=AssessmentOutcome.INFORMATIVE,
        links=(
            EvidenceLinkProposal(
                hypothesis_id="H-db_002-pool",
                evidence_id=chall_evidence_id,
                relation=EvidenceRelation.CONTRADICTS,
                rationale=(
                    "Raw records show db_002 pool status 'normal' at 1050s "
                    "and 'recovered' at 1080s with 0 queued and 50 active "
                    "connections; pool was not exhausted during the main "
                    "fault window, contradicting the nomination"
                ),
            ),
        ),
        status_updates=(
            HypothesisStatusUpdate(
                hypothesis_id="H-db_002-pool",
                new_status=HypothesisStatus.WEAKENED,
                rationale=(
                    "Raw records contradict the hypothesis that pool "
                    "exhaustion caused sustained symptoms; nomination "
                    "is refuted and returned to Lead phase"
                ),
            ),
        ),
        rationale=(
            "Raw records indicate db_002 pool was normal/recovered when "
            "the main symptoms appeared; connection pool exhaustion "
            "hypothesis is contradicted by evidence"
        ),
    )

    chall_resolution = ChallengeResolution(
        challenge_id="C1",
        action_id=chall_action.action_id,
        evidence_id=chall_evidence_id,
        verdict=ChallengeVerdict.NOMINATION_REFUTED,
        assessment=chall_assessment,
        rationale=(
            "Evidence shows pool recovery before symptoms; "
            "nomination is refuted"
        ),
    )

    chall_policy = ScriptedChallengerPolicy(
        proposal=chall_proposal,
        resolution=chall_resolution,
    )

    return lead_ctrl, (h1, h2), lead_policy, challenge_ctrl, chall_policy


# ===========================================================================
# 3. Inconclusive demo
# ===========================================================================


def build_demo_inconclusive_challenge() -> tuple[
    LeadTournamentController,
    tuple[CausalHypothesis, ...],
    ScriptedLeadPolicy,
    ChallengerController,
    ScriptedChallengerPolicy,
]:
    """Build a scenario where the challenge is inconclusive.

    Challenger investigates db_002 raw records but the tool returns
    insufficient data — the nomination neither survives nor is refuted.
    """

    h1, h2 = _build_base_hypotheses()

    # Store with no raw records for db_002 — tool returns empty
    store = MockTelemetryStore(
        onset_observations=_build_base_onset_observations(),
        trace_paths=_build_base_trace_paths(),
        raw_records=[],  # No records match the query
    )

    gate = ActionGate()
    registry = build_default_tool_registry()
    graph = EvidenceGraph()
    assessment_gate = EvidenceAssessmentGate()

    lead_ctrl, lead_policy, _, _ = _build_lead_phase(
        store=store, graph=graph, gate=gate,
        registry=registry, assessment_gate=assessment_gate,
    )

    challenge_ctrl = _build_challenger_infra(
        store=store, graph=graph, gate=gate,
        registry=registry, assessment_gate=assessment_gate,
    )

    # Challenger action: retrieve_raw_evidence for db_002
    chall_action = DiscriminativeAction(
        action_id="C1-retrieve-raw",
        action_type="run_discriminative_test",
        target_hypothesis_ids=("H-db_002-pool", "H-os_009-cpu"),
        question=(
            "What do raw telemetry records show about db_002 "
            "connection pool during the fault window?"
        ),
        tool_name="retrieve_raw_evidence",
        args={
            "modality": "metrics",
            "component_scope": ["db_002"],
            "time_window": [1000.0, 2000.0],
            "limit": 20,
        },
        expected_outcomes={
            "H-db_002-pool": "Records confirm pool state",
            "H-os_009-cpu": "Records show pool state",
        },
        why_discriminative=(
            "Raw records for db_002 may confirm or deny sustained "
            "pool exhaustion"
        ),
    )

    chall_sig = chall_action.query_signature()
    chall_evidence_id = "evidence:" + chall_sig

    chall_proposal = ChallengeProposal(
        challenge_id="C1",
        nominated_hypothesis_id="H-db_002-pool",
        competitor_hypothesis_ids=("H-os_009-cpu",),
        challenge_claim=(
            "db_002 connection pool status during the fault window "
            "is unknown from current evidence"
        ),
        falsification_target=(
            "Determine db_002 pool status during the fault window"
        ),
        action=chall_action,
        rationale=(
            "Raw records may reveal pool state not visible in onset data"
        ),
    )

    chall_assessment = EvidenceAssessment(
        action_id=chall_action.action_id,
        evidence_id=chall_evidence_id,
        outcome=AssessmentOutcome.INCONCLUSIVE,
        links=(),
        status_updates=(),
        rationale=(
            "No raw records found for db_002 in the query window; "
            "the challenge investigation is inconclusive"
        ),
    )

    chall_resolution = ChallengeResolution(
        challenge_id="C1",
        action_id=chall_action.action_id,
        evidence_id=chall_evidence_id,
        verdict=ChallengeVerdict.INCONCLUSIVE,
        assessment=chall_assessment,
        rationale=(
            "No records returned; evidence is insufficient to determine "
            "whether nomination survives or is refuted"
        ),
    )

    chall_policy = ScriptedChallengerPolicy(
        proposal=chall_proposal,
        resolution=chall_resolution,
    )

    return lead_ctrl, (h1, h2), lead_policy, challenge_ctrl, chall_policy
