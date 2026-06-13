"""Deterministic single-case demo scenario for the Lead tournament.

Constructs a fully-configured tournament with two competing causal
hypotheses, a MockTelemetryStore containing onset and trace data,
and a ScriptedLeadPolicy that runs through two investigation rounds
before nominating H2 (db_002 connection pool exhaustion).
"""

from __future__ import annotations

from prismv4.prism_cht.action_gate import ActionGate
from prismv4.prism_cht.action_schema import DiscriminativeAction
from prismv4.prism_cht.assessment_gate import EvidenceAssessmentGate
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


def build_demo_lead_tournament() -> tuple[
    LeadTournamentController,
    tuple[CausalHypothesis, ...],
    ScriptedLeadPolicy,
]:
    """Build a complete demo tournament scenario.

    Returns (controller, initial_hypotheses, policy).
    """

    # ------------------------------------------------------------------
    # Hypotheses
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Telemetry store — db_002 onset is earlier, and trace path exists
    # ------------------------------------------------------------------

    onset_observations = [
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

    trace_paths = [
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

    store = MockTelemetryStore(
        onset_observations=onset_observations,
        trace_paths=trace_paths,
    )

    # ------------------------------------------------------------------
    # Infrastructure
    # ------------------------------------------------------------------

    gate = ActionGate()
    registry = build_default_tool_registry()
    graph = EvidenceGraph()
    assessment_gate = EvidenceAssessmentGate()

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

    # ------------------------------------------------------------------
    # Round 1 action: compare_onset_order
    # ------------------------------------------------------------------

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
        target_hypothesis_ids=(h1.hypothesis_id, h2.hypothesis_id),
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
            h1.hypothesis_id: "os_009 CPU spike is the earliest anomaly",
            h2.hypothesis_id: "db_002 connection pool exhaustion is the earliest anomaly",
        },
        why_discriminative=(
            "os_009 and db_002 predict opposite onset order; "
            "the earlier failure is more likely the root cause"
        ),
    )

    # Assessment for Round 1:
    # - db_002 has earlier onset → contradicts H1, supports H2
    # - H1: ACTIVE -> WEAKENED
    # - H2: ACTIVE -> SUPPORTED

    # Need to predict the evidence_id that the executor will generate.
    # The evidence_id is "evidence:" + query_signature.
    # We construct it here so the assessment references the correct ID.

    # But wait — the ScriptedLeadPolicy assessment needs to reference the
    # evidence_id that the executor creates.  We can't know it before the
    # controller runs.  However, the assessment gate checks that the
    # assessment references real evidence.

    # Problem: the ScriptedLeadPolicy is pre-configured, but the evidence_id
    # is only known at runtime.  We need the policy's assess_evidence to
    # return an assessment with the correct evidence_id.

    # Solution: The ScriptedLeadPolicy receives the actual evidence in
    # assess_evidence().  We can have it return a pre-configured assessment
    # but with the evidence_id patched.  OR, we can configure the
    # ScriptedLeadPolicy with assessments that already have the correct
    # evidence IDs.

    # Let me think... The executor creates evidence_id = "evidence:" + sig.
    # The sig is deterministic from tool_name + args.  So we can pre-compute it.

    sig_r1 = action_r1.query_signature()
    evidence_id_r1 = "evidence:" + sig_r1

    assessment_r1 = EvidenceAssessment(
        action_id=action_r1.action_id,
        evidence_id=evidence_id_r1,
        outcome=AssessmentOutcome.INFORMATIVE,
        links=(
            EvidenceLinkProposal(
                hypothesis_id=h2.hypothesis_id,
                evidence_id=evidence_id_r1,
                relation=EvidenceRelation.SUPPORTS,
                rationale=(
                    "db_002 connection pool onset at 1100.0 precedes "
                    "os_009 CPU onset at 1250.0 by 150.0 seconds; "
                    "earlier anomaly supports db_002 as root cause"
                ),
            ),
            EvidenceLinkProposal(
                hypothesis_id=h1.hypothesis_id,
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
                hypothesis_id=h2.hypothesis_id,
                new_status=HypothesisStatus.SUPPORTED,
                rationale=(
                    "db_002 connection pool exhaustion is the earliest "
                    "observable anomaly, consistent with the hypothesis"
                ),
            ),
            HypothesisStatusUpdate(
                hypothesis_id=h1.hypothesis_id,
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

    # ------------------------------------------------------------------
    # Round 2 action: inspect_trace_path
    # ------------------------------------------------------------------

    action_r2 = DiscriminativeAction(
        action_id="R2-inspect-trace",
        action_type="run_discriminative_test",
        target_hypothesis_ids=(h1.hypothesis_id, h2.hypothesis_id),
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
            h1.hypothesis_id: "No trace path from db_002 to app_003",
            h2.hypothesis_id: "Trace path exists from db_002 to app_003",
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
                hypothesis_id=h2.hypothesis_id,
                evidence_id=evidence_id_r2,
                relation=EvidenceRelation.SUPPORTS,
                rationale=(
                    "Explicit trace path db_002 -> app_003 found with "
                    "latency 45ms, status=error; confirms downstream "
                    "propagation from db_002 to app_003"
                ),
            ),
        ),
        status_updates=(),  # No status change needed; H2 is already SUPPORTED
        rationale=(
            "Trace inspection confirms a direct path from db_002 to "
            "app_003, providing propagation evidence for the connection "
            "pool exhaustion hypothesis"
        ),
    )

    # ------------------------------------------------------------------
    # Nomination
    # ------------------------------------------------------------------

    nomination = LeadNomination(
        hypothesis_id=h2.hypothesis_id,
        supporting_evidence_ids=(evidence_id_r1, evidence_id_r2),
        addressed_competitor_ids=(h1.hypothesis_id,),
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

    # ------------------------------------------------------------------
    # Policy
    # ------------------------------------------------------------------

    policy_script = ScriptedLeadPolicy(
        turns=[
            ScriptedInvestigationTurn(action=action_r1, assessment=assessment_r1),
            ScriptedInvestigationTurn(action=action_r2, assessment=assessment_r2),
        ],
        nomination=nomination,
    )

    return controller, (h1, h2), policy_script
