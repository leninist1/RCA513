"""Background-suppressed joint answer selection for Scheme B.

This is the single assembly point for the modules that used to be implemented
but bypassed by the official runner:

CandidateGenerator -> IterativeRefutation -> EvidenceMatrix ->
ReasonCompetition -> ComponentConditionalTimeAnchor -> Confidence -> optional LLM.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import json
from math import isfinite
from typing import Any, Callable, Iterable, Mapping, Sequence

import pandas as pd

from refute_b_v2.background_suppression import (
    AnomalyEvent,
    BackgroundCalibrator,
    component_scores,
)
from refute_b_v2.candidate_generation import CandidateGenerator, CandidateHypothesis
from refute_b_v2.confidence_calibration import RuntimeConfidence, RuntimeConfidenceCalibrator
from refute_b_v2.confidence_semantics import explain_confidence
from refute_b_v2.evidence_adapter import SummaryBackedEvidence
from refute_b_v2.evidence_matrix import MatrixRow
from refute_b_v2.evidence_signature import case_signature_from_results
from refute_b_v2.iterative_refutation import final_decision, run_iterative_refutation
from refute_b_v2.llm_arbitration import run_llm_arbitration
from refute_b_v2.reason_competition import ReasonAdjustmentConfig, ReasonScoreCalibrator, compete_reasons
from refute_b_v2.rule_engine import RuleEngine
from refute_b_v2.rules import Candidate, normalize_reason_bucket
from refute_b_v2.time_anchor import TimeAnchor, component_time_anchors


try:  # pragma: no cover - exercised in integration on the server project.
    from refute.src.evidence_query import EvidenceQuery, kpi_in_bucket
except Exception:  # pragma: no cover
    EvidenceQuery = None

    def kpi_in_bucket(kpi_name: str, bucket: str) -> bool:
        text = str(kpi_name)
        patterns = {
            "cpu": ("CPU", "Cpu", "CPULoad"),
            "memory": ("MEMORY", "Memory", "Mem", "Heap", "used_memory", "Qcache"),
            "filesystem": ("FILESYSTEM", "FSAvailable", "FSCapacity", "FSInode"),
            "disk": ("DSK", "Disk", "Read", "Write"),
            "network": ("Network", "TCP", "Packet", "rejected", "Aborted"),
        }
        return any(token in text for token in patterns.get(bucket, ()))


EvidenceFactory = Callable[..., Any]


@dataclass(frozen=True)
class JointSelectorConfig:
    top_k_components: int = 8
    per_reason_k: int = 3
    max_reasons: int = 5
    max_candidates: int = 120
    max_llm_calls: int = 8
    time_anchor_top_k: int = 3
    min_high_gap: float = 3.0
    llm_enabled: bool = False
    force_llm: bool = False


@dataclass(frozen=True)
class StructuredRCADecision:
    occurrence_datetime: str
    component: str
    reason: str
    confidence: str
    source: str
    evidence_refs: tuple[str, ...] = ()

    def to_openrca_dict(self) -> dict[str, str]:
        return {
            "root cause occurrence datetime": self.occurrence_datetime,
            "root cause component": self.component,
            "root cause reason": self.reason,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.to_openrca_dict(),
            "confidence": self.confidence,
            "source": self.source,
            "evidence_refs": list(self.evidence_refs),
        }


@dataclass
class JointCandidateDecision:
    candidate: Candidate
    matrix_row: MatrixRow
    time_anchor: TimeAnchor
    confidence: RuntimeConfidence
    semantic_confidence: dict
    reason_competition: dict
    background: dict
    score: float
    structured: StructuredRCADecision

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate.to_dict(),
            "time_anchor": self.time_anchor.to_dict(),
            "confidence": self.confidence.to_dict(),
            "semantic_confidence": self.semantic_confidence,
            "reason_competition": self.reason_competition,
            "background": self.background,
            "score": self.score,
            "structured": self.structured.to_dict(),
            "matrix_row": self.matrix_row.to_dict(),
        }


@dataclass
class JointSelectionResult:
    decisions: list[JointCandidateDecision]
    prediction: dict[str, dict[str, str]]
    debug: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "prediction": self.prediction,
            "decisions": [row.to_dict() for row in self.decisions],
            "debug": self.debug,
        }


class JointAnswerSelector:
    def __init__(
        self,
        engine: RuleEngine,
        baseline: Any,
        node_graph: Mapping,
        services: Sequence[str],
        *,
        llm_client: Any | None = None,
        similar_index: Any | None = None,
        evidence_factory: EvidenceFactory | None = None,
        confidence_calibrator: RuntimeConfidenceCalibrator | None = None,
        reason_calibrator: ReasonScoreCalibrator | None = None,
        reason_adjustments: ReasonAdjustmentConfig | None = None,
        config: JointSelectorConfig | None = None,
    ):
        self.engine = engine
        self.baseline = baseline
        self.node_graph = dict(node_graph)
        self.services = [str(s) for s in services]
        self.llm_client = llm_client
        self.similar_index = similar_index
        self.evidence_factory = evidence_factory or self._default_evidence_factory
        self.confidence_calibrator = confidence_calibrator
        self.reason_calibrator = reason_calibrator or ReasonScoreCalibrator()
        self.reason_adjustments = reason_adjustments or ReasonAdjustmentConfig()
        self.config = config or JointSelectorConfig()

    def select(
        self,
        *,
        case_id: str,
        metric_df: pd.DataFrame,
        log_df: pd.DataFrame,
        trace_summary: dict | None,
        modal_status: Mapping[str, str],
        trace_df: pd.DataFrame | None = None,
        failure_count: int = 1,
        window_start_ts: int | None = None,
        background_metric_df: pd.DataFrame | None = None,
    ) -> JointSelectionResult:
        events = (
            metric_events(metric_df, self.baseline)
            + log_events(log_df)
            + trace_events(trace_summary)
            + raw_trace_events(trace_df, window_start_ts)
        )
        background_events = metric_events(background_metric_df, self.baseline) if background_metric_df is not None else []
        calibrator = fit_component_background(background_events)
        hypotheses = CandidateGenerator(
            top_k=self.config.top_k_components,
            per_reason_k=self.config.per_reason_k,
            calibrator=calibrator,
        ).generate(events, fallback_services=self.services, fallback_reasons=_fallback_reasons(events))
        candidates = self._expanded_candidates(events, hypotheses)
        evidence = self._make_evidence(metric_df, log_df, trace_summary, trace_df, modal_status)
        states = run_iterative_refutation(
            self.engine,
            candidates,
            evidence,
            min_high_gap=self.config.min_high_gap,
        )
        matrix = final_decision(states)
        rows = _all_matrix_rows(matrix)
        signature = case_signature_from_results(case_id, modal_status, [row.result for row in rows[:10]]).to_dict()
        reason_rows = self._reason_competition(rows, hypotheses)
        decisions = self._build_joint_decisions(
            rows,
            hypotheses,
            reason_rows,
            metric_df,
            window_start_ts,
        )
        decisions = self._apply_llm_if_requested(case_id, decisions, rows, matrix.to_dict(), modal_status, trace_summary, signature)
        selected = _dedupe_decisions(decisions, max(1, int(failure_count)))
        prediction = {str(i + 1): row.structured.to_openrca_dict() for i, row in enumerate(selected)}
        return JointSelectionResult(
            decisions=selected,
            prediction=prediction,
            debug={
                "case_id": case_id,
                "event_count": len(events),
                "background_event_count": len(background_events),
                "candidate_count": len(candidates),
                "hypotheses": [h.to_dict() for h in hypotheses],
                "iterations": [state.to_dict() for state in states],
                "signature": signature,
                "all_ranked_decisions": [row.to_dict() for row in decisions[:20]],
            },
        )

    def _make_evidence(self, metric_df: pd.DataFrame, log_df: pd.DataFrame, trace_summary: dict | None,
                       trace_df: pd.DataFrame | None, modal_status: Mapping[str, str]):
        try:
            return self.evidence_factory(metric_df, log_df, trace_summary, trace_df, modal_status)
        except TypeError:
            return self.evidence_factory(metric_df, log_df, trace_summary, modal_status)

    def _default_evidence_factory(self, metric_df: pd.DataFrame, log_df: pd.DataFrame,
                                  trace_summary: dict | None, trace_df: pd.DataFrame | None,
                                  modal_status: Mapping[str, str]):
        if EvidenceQuery is None:
            raise RuntimeError("refute.src.evidence_query.EvidenceQuery is unavailable")
        base = EvidenceQuery(
            metric_df,
            self.baseline,
            node_graph=self.node_graph,
            log_df=log_df,
            trace_df=trace_df,
            modal_status=dict(modal_status),
        )
        return SummaryBackedEvidence(base, trace_summary)

    def _expanded_candidates(self, events: list[AnomalyEvent], hypotheses: list[CandidateHypothesis]) -> list[Candidate]:
        candidate_map: dict[tuple[str, str], Candidate] = {}
        for hyp in hypotheses:
            candidate_map[(hyp.candidate.service, hyp.candidate.reason)] = hyp.candidate
        reason_totals: dict[str, float] = {}
        for event in events:
            reason_totals[event.reason] = reason_totals.get(event.reason, 0.0) + abs(float(event.strength or 0.0))
        top_reasons = [
            reason for reason, _ in sorted(reason_totals.items(), key=lambda kv: (-kv[1], kv[0]))[: self.config.max_reasons]
        ]
        if not top_reasons:
            top_reasons = list(_fallback_reasons(events))
        for reason in top_reasons:
            for service in self.services:
                candidate_map.setdefault((service, reason), Candidate(service, reason))
                if len(candidate_map) >= self.config.max_candidates:
                    break
            if len(candidate_map) >= self.config.max_candidates:
                break
        return sorted(candidate_map.values(), key=lambda c: (c.reason, c.service))

    def _reason_competition(self, rows: list[MatrixRow], hypotheses: list[CandidateHypothesis]) -> dict[tuple[str, str], dict]:
        background = {(h.candidate.service, h.candidate.reason): h.component_score.score for h in hypotheses}
        by_service: dict[str, dict[str, float]] = {}
        evidence_flags: dict[str, dict[str, bool]] = {}
        for row in rows:
            cand = row.result.candidate
            by_service.setdefault(cand.service, {})[cand.reason] = (
                row.vector.support_strength
                - row.vector.refute_strength
                + background.get((cand.service, cand.reason), 0.0)
            )
            flags = evidence_flags.setdefault(cand.service, {})
            _merge_flags(flags, _evidence_flags(row))
        out = {}
        for service, reason_scores in by_service.items():
            competed = compete_reasons(
                reason_scores,
                evidence_flags.get(service, {}),
                self.reason_calibrator,
                self.reason_adjustments,
            )
            for item in competed:
                out[(service, item.reason)] = item.to_dict()
        return out

    def _build_joint_decisions(
        self,
        rows: list[MatrixRow],
        hypotheses: list[CandidateHypothesis],
        reason_rows: dict[tuple[str, str], dict],
        metric_df: pd.DataFrame,
        window_start_ts: int | None,
    ) -> list[JointCandidateDecision]:
        hyp_map = {(h.candidate.service, h.candidate.reason): h for h in hypotheses}
        support_order = sorted(rows, key=lambda r: r.vector.support_strength - r.vector.refute_strength, reverse=True)
        gap_by_candidate = _gap_features(support_order)
        decisions = []
        for row in rows:
            cand = row.result.candidate
            hyp = hyp_map.get((cand.service, cand.reason))
            candidate_times = hyp.candidate_times if hyp else ()
            anchors = component_time_anchors(
                metric_df,
                cand.service,
                self.baseline,
                candidate_timestamps=candidate_times or ([window_start_ts] if window_start_ts else []),
                top_k=self.config.time_anchor_top_k,
            )
            if not anchors and window_start_ts is not None:
                anchors = [TimeAnchor(int(window_start_ts // 60 * 60), 0.0, "window_start_fallback", 0, {})]
            if not anchors:
                continue
            semantic = explain_confidence(row).to_dict()
            features = {
                "support_strength": row.vector.support_strength,
                "top1_top2_gap": gap_by_candidate.get((cand.service, cand.reason), 0.0),
                "hard_support_count": row.vector.support_hard,
                "hard_refute_count": row.vector.refute_hard,
                "blind_count": row.vector.blind,
            }
            runtime_conf = (
                self.confidence_calibrator.label_for(cand.reason, features, semantic["label"])
                if self.confidence_calibrator
                else RuntimeConfidence(semantic["label"], "semantic_rules", "", None)
            )
            reason_info = reason_rows.get((cand.service, cand.reason), {})
            background = hyp.component_score.to_dict() if hyp else {}
            anchor = anchors[0]
            score = _joint_score(row, reason_info, runtime_conf, anchor, background)
            structured = StructuredRCADecision(
                occurrence_datetime=_fmt_ts(anchor.timestamp),
                component=cand.service,
                reason=cand.reason,
                confidence=runtime_conf.label,
                source="joint_selector",
                evidence_refs=tuple(card.rule_id for card in row.result.cards if card.polarity == "support"),
            )
            decisions.append(JointCandidateDecision(cand, row, anchor, runtime_conf, semantic, reason_info, background, score, structured))
        return sorted(decisions, key=lambda d: (-d.score, d.structured.occurrence_datetime, d.candidate.service, d.candidate.reason))

    def _apply_llm_if_requested(
        self,
        case_id: str,
        decisions: list[JointCandidateDecision],
        rows: list[MatrixRow],
        matrix_dict: dict,
        modal_status: Mapping[str, str],
        trace_summary: dict | None,
        signature: dict,
    ) -> list[JointCandidateDecision]:
        if not (self.config.llm_enabled and self.llm_client and decisions):
            return decisions
        allowed = [decision.structured.to_dict() for decision in decisions[:20]]
        context = {
            "modal_status": dict(modal_status),
            "trace_summary": trace_summary,
            "signature": signature,
            "allowed_decisions": allowed,
            "required_final_schema": {
                "root cause occurrence datetime": "<must equal one allowed decision datetime>",
                "root cause component": "<must equal one allowed component>",
                "root cause reason": "<must equal one allowed reason>",
                "confidence": "HIGH|MEDIUM|LOW|UNKNOWN",
                "evidence_refs": ["rule ids from evidence cards"],
            },
        }
        llm_info = run_llm_arbitration(
            case_id,
            [row.result for row in rows[:20]],
            self.llm_client,
            context=context,
            similar_index=self.similar_index,
            force=self.config.force_llm,
            max_calls=self.config.max_llm_calls,
        )
        chosen = _validated_llm_decision(llm_info, decisions)
        if chosen is None:
            for decision in decisions:
                decision.background = {**decision.background, "llm_arbitration": llm_info}
            return decisions
        reordered = [chosen] + [decision for decision in decisions if decision is not chosen]
        chosen.structured = StructuredRCADecision(
            chosen.structured.occurrence_datetime,
            chosen.structured.component,
            chosen.structured.reason,
            chosen.structured.confidence,
            "llm_structured",
            chosen.structured.evidence_refs,
        )
        chosen.background = {**chosen.background, "llm_arbitration": llm_info}
        return reordered


def metric_events(metric_df: pd.DataFrame | None, baseline: Any) -> list[AnomalyEvent]:
    out = []
    if metric_df is None or metric_df.empty:
        return out
    for row in metric_df.itertuples(index=False):
        reason = reason_for_metric(getattr(row, "kpi_name", ""))
        if reason is None:
            continue
        try:
            value = getattr(row, "value")
            result = baseline.is_anomalous(getattr(row, "cmdb_id"), getattr(row, "kpi_name"), value, threshold="p99")
            if not getattr(result, "is_anomalous", False):
                continue
            strength = abs(float(getattr(result, "deviation", 0.0) or 0.0))
        except Exception:
            continue
        if not isfinite(strength) or strength <= 0:
            continue
        out.append(AnomalyEvent(
            int(getattr(row, "timestamp") // 60 * 60),
            str(getattr(row, "cmdb_id")),
            reason,
            strength,
            "metric",
            {"kpi_name": str(getattr(row, "kpi_name", ""))},
        ))
    return out


def log_events(log_df: pd.DataFrame | None) -> list[AnomalyEvent]:
    out = []
    if log_df is None or log_df.empty or "value" not in log_df.columns:
        return out
    for row in log_df.itertuples(index=False):
        hit = reason_for_log(getattr(row, "value", ""))
        if not hit:
            continue
        reason, score = hit
        out.append(AnomalyEvent(int(getattr(row, "timestamp") // 60 * 60), str(getattr(row, "cmdb_id")), reason, score, "log"))
    return out


def trace_events(summary: dict | None) -> list[AnomalyEvent]:
    if not summary or summary.get("trace_status") != "present":
        return []
    out = []
    for edge in summary.get("events", {}).get("slow_edges", []):
        ts = _summary_ts_to_seconds(edge.get("first_timestamp"), summary)
        if ts is None:
            continue
        strength = float(edge.get("slow_ratio", 0.0) or 0.0)
        for component in {str(edge.get("src")), str(edge.get("dst"))}:
            if component and component != "None":
                out.append(AnomalyEvent(ts, component, "network latency", strength, "trace_slow_edge", edge))
    for edge in summary.get("events", {}).get("dropped_edges", []):
        ts = _summary_ts_to_seconds(edge.get("first_timestamp"), summary)
        if ts is None:
            continue
        strength = float(edge.get("count_drop_ratio", 0.0) or 0.0) * 10.0
        for component in {str(edge.get("src")), str(edge.get("dst"))}:
            if component and component != "None":
                out.append(AnomalyEvent(ts, component, "network packet loss", strength, "trace_edge_drop", edge))
    first = summary.get("events", {}).get("first_anomalous_service")
    ts = _summary_ts_to_seconds(None, summary)
    if first and ts is not None:
        out.append(AnomalyEvent(ts, str(first), "network latency", 2.0, "trace_first_anomaly"))
    return out


def raw_trace_events(trace_df: pd.DataFrame | None, window_start_ts: int | None = None) -> list[AnomalyEvent]:
    if trace_df is None or trace_df.empty:
        return []
    try:
        from refute.src.trace_evidence import build_trace_edges, service_duration_summary
    except Exception:
        return []
    out: list[AnomalyEvent] = []
    summary = service_duration_summary(trace_df)
    if not summary.empty:
        peers = summary[summary["count"] >= 5]["p95"].astype(float)
        if len(peers) >= 3:
            median = float(peers.median())
            mad = float((peers - median).abs().median())
            if mad > 1e-9:
                for row in summary.itertuples(index=False):
                    if int(row.count) < 5:
                        continue
                    z = (float(row.p95) - median) / mad
                    if z > 3.0:
                        ts = _raw_trace_ts_to_seconds(trace_df, getattr(row, "cmdb_id"), window_start_ts)
                        out.append(AnomalyEvent(ts, str(row.cmdb_id), "network latency", float(z), "raw_trace_service_p95", {
                            "cmdb_id": str(row.cmdb_id),
                            "count": int(row.count),
                            "p95": float(row.p95),
                            "z": float(z),
                        }))
    edges = build_trace_edges(trace_df)
    if not edges.empty:
        edge_summary = (
            edges.groupby(["src", "dst"])["child_duration"]
            .agg(count="count", p95=lambda s: float(s.quantile(0.95)))
            .reset_index()
        )
        peers = edge_summary[edge_summary["count"] >= 3]["p95"].astype(float)
        if len(peers) >= 3:
            median = float(peers.median())
            mad = float((peers - median).abs().median())
            if mad > 1e-9:
                for row in edge_summary.itertuples(index=False):
                    if int(row.count) < 3:
                        continue
                    z = (float(row.p95) - median) / mad
                    if z > 3.0:
                        details = {"src": str(row.src), "dst": str(row.dst), "count": int(row.count), "p95": float(row.p95), "z": float(z)}
                        ts = _raw_trace_ts_to_seconds(trace_df, row.dst, window_start_ts)
                        out.append(AnomalyEvent(ts, str(row.src), "network latency", float(z), "raw_trace_slow_edge", details))
                        out.append(AnomalyEvent(ts, str(row.dst), "network latency", float(z), "raw_trace_slow_edge", details))
    return out[:40]


def fit_component_background(events: list[AnomalyEvent]) -> BackgroundCalibrator:
    calibrator = BackgroundCalibrator()
    by_hour: dict[int, list[AnomalyEvent]] = {}
    for event in events:
        by_hour.setdefault(int(event.timestamp // 3600), []).append(event)
    for group in by_hour.values():
        for row in component_scores(group):
            ts = row.timestamps[0] if row.timestamps else group[0].timestamp
            from refute_b_v2.background_suppression import hour_bucket

            calibrator.add(row.component, row.reason, hour_bucket(ts), row.clipped_sum)
    return calibrator


def reason_for_metric(kpi_name: str) -> str | None:
    kpi = str(kpi_name)
    if kpi_in_bucket(kpi, "cpu"):
        return "high JVM CPU load" if "JVM" in kpi or "Tomcat" in kpi else "high CPU usage"
    if kpi_in_bucket(kpi, "memory"):
        if any(token in kpi for token in ("JVM", "Heap", "Tomcat-MEMORY")):
            return "JVM Out of Memory (OOM) Heap"
        return "high memory usage"
    if kpi_in_bucket(kpi, "filesystem"):
        return "high disk space usage"
    if kpi_in_bucket(kpi, "disk"):
        return "high disk I/O read usage"
    if kpi_in_bucket(kpi, "network"):
        packet_tokens = ("Packet", "Packets", "Err", "rejected", "Aborted", "TCP")
        return "network packet loss" if any(token in kpi for token in packet_tokens) else "network latency"
    return None


def reason_for_log(text: str) -> tuple[str, float] | None:
    s = str(text)
    low = s.lower()
    if any(token in s for token in ("OutOfMemoryError", "Full GC", "SIGKILL")) or "oom" in low:
        return "JVM Out of Memory (OOM) Heap", 8.0
    if any(token in low for token in ("reset by peer", "broken pipe", "connection reset", "retry")):
        return "network packet loss", 5.0
    if any(token in low for token in ("timeout", "timed out", "connection refused")):
        return "network latency", 5.0
    if "no space" in low or "disk" in low:
        return "high disk space usage", 4.0
    return None


def _summary_ts_to_seconds(value, summary: dict) -> int | None:
    if value is None:
        try:
            return int(datetime.strptime(summary.get("window", {})["start"], "%Y-%m-%d %H:%M:%S").timestamp() // 60 * 60)
        except (KeyError, ValueError, TypeError):
            return None
    ts = int(value)
    if ts > 10_000_000_000:
        ts = ts // 1000
    return int(ts // 60 * 60)


def _raw_trace_ts_to_seconds(trace_df: pd.DataFrame, service: str, fallback_ts: int | None) -> int:
    try:
        rows = trace_df[trace_df["cmdb_id"].astype(str) == str(service)]
        if rows.empty:
            rows = trace_df
        ts = int(rows["timestamp"].min())
        if ts > 10_000_000_000:
            ts //= 1000
        return int(ts // 60 * 60)
    except Exception:
        return int((fallback_ts or 0) // 60 * 60)


def _fallback_reasons(events: Iterable[AnomalyEvent]) -> tuple[str, ...]:
    reasons = tuple(sorted({event.reason for event in events}))
    return reasons or ("network latency", "high CPU usage", "high memory usage")


def _all_matrix_rows(matrix) -> list[MatrixRow]:
    rows = list(matrix.high_suspicion) + list(matrix.ambiguous) + list(matrix.data_blind_spots) + list(matrix.low_suspicion)
    return sorted(rows, key=lambda row: (-row.vector.support_strength, row.vector.refute_strength, row.result.candidate.service, row.result.candidate.reason))


def _gap_features(rows: list[MatrixRow]) -> dict[tuple[str, str], float]:
    nets = [row.vector.support_strength - row.vector.refute_strength for row in rows]
    out = {}
    for idx, row in enumerate(rows):
        next_net = nets[idx + 1] if idx + 1 < len(nets) else 0.0
        cand = row.result.candidate
        out[(cand.service, cand.reason)] = nets[idx] - next_net
    return out


def _evidence_flags(row: MatrixRow) -> dict[str, bool]:
    flags = {"gc_or_heap_pressure": False, "tcp_retransmit_or_reset": False, "memory_pressure": False}
    for card in row.result.cards:
        text = f"{card.rule_id} {card.text}".lower()
        if card.polarity != "support":
            continue
        if any(token in text for token in ("gc", "heap", "oom", "memory")):
            flags["gc_or_heap_pressure"] = True
            flags["memory_pressure"] = True
        if any(token in text for token in ("retry", "reset", "broken", "drop", "packet")):
            flags["tcp_retransmit_or_reset"] = True
    return flags


def _merge_flags(target: dict[str, bool], update: dict[str, bool]) -> None:
    for key, value in update.items():
        target[key] = bool(target.get(key) or value)


def _joint_score(row: MatrixRow, reason_info: Mapping, confidence: RuntimeConfidence,
                 anchor: TimeAnchor, background: Mapping) -> float:
    score = 0.0
    score += row.vector.support_strength * 2.0
    score -= row.vector.refute_strength * 3.0
    score += row.vector.support_hard * 5.0
    score -= row.vector.refute_hard * 8.0
    score -= row.vector.blind * 1.0
    score += float(reason_info.get("adjusted_score", 0.0) or 0.0)
    score += min(10.0, float(background.get("relative_score", 0.0) or 0.0))
    score += min(3.0, anchor.score * 0.25)
    if confidence.label == "HIGH":
        score += 4.0
    elif confidence.label == "LOW":
        score -= 4.0
    return score


def _dedupe_decisions(decisions: list[JointCandidateDecision], limit: int) -> list[JointCandidateDecision]:
    selected = []
    used = set()
    for decision in decisions:
        key = (decision.structured.occurrence_datetime, decision.structured.component)
        if key in used:
            continue
        selected.append(decision)
        used.add(key)
        if len(selected) >= limit:
            break
    return selected


def _validated_llm_decision(llm_info: dict, decisions: list[JointCandidateDecision]) -> JointCandidateDecision | None:
    if not llm_info or not llm_info.get("called"):
        return None
    text = str(llm_info.get("recommendation", "")).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    dt = data.get("root cause occurrence datetime") or data.get("time")
    comp = data.get("root cause component") or data.get("component")
    reason = data.get("root cause reason") or data.get("reason")
    for decision in decisions:
        if (
            decision.structured.occurrence_datetime == dt
            and decision.structured.component == comp
            and decision.structured.reason == reason
        ):
            return decision
    return None


def _fmt_ts(timestamp: int) -> str:
    return datetime.fromtimestamp(int(timestamp)).strftime("%Y-%m-%d %H:%M:%S")
