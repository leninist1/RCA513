"""Candidate-local time anchor voting for d32."""
from __future__ import annotations

from dataclasses import dataclass
from math import log1p
from typing import Any, Mapping

import pandas as pd

from refute_b_v2_d32.evidence import kpi_in_bucket
from refute_b_v2_d32.schema import RootCandidate
from refute_b_v2_d32.signature import reason_for_log


@dataclass(frozen=True)
class TimeAnchorVote:
    timestamp: int
    score: float
    source: str
    kind: str
    strength: float
    details: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "score": self.score,
            "source": self.source,
            "kind": self.kind,
            "strength": self.strength,
            "details": dict(self.details),
        }


@dataclass(frozen=True)
class CandidateTimeAnchor:
    timestamp: int
    score: float
    policy: str
    votes: tuple[TimeAnchorVote, ...]
    rejected_votes: tuple[TimeAnchorVote, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "score": self.score,
            "policy": self.policy,
            "votes": [vote.to_dict() for vote in self.votes],
            "rejected_votes": [vote.to_dict() for vote in self.rejected_votes],
        }


@dataclass(frozen=True)
class CandidateTimeAnchorPolicy:
    bucket_seconds: int = 60
    metric_cross_vote_weight: float = 1.4
    metric_sustained_bonus: float = 2.0
    metric_peak_bonus: float = 0.45
    trace_first_bonus: float = 3.0
    trace_role_bonus: float = 0.8
    log_match_bonus: float = 3.2
    early_vote_bonus: float = 0.25
    top_k_votes: int = 10


class CandidateTimeAnchorer:
    def __init__(self, baseline, node_graph: Mapping[str, Any] | None = None,
                 policy: CandidateTimeAnchorPolicy | None = None):
        self.baseline = baseline
        self.node_graph = dict(node_graph or {"containers": {}, "nodes": {}})
        self.policy = policy or CandidateTimeAnchorPolicy()

    def anchor(
        self,
        candidate: RootCandidate,
        metric_df: pd.DataFrame,
        log_df: pd.DataFrame,
        trace_summary: Mapping[str, Any] | None,
        fallback_ts: int,
    ) -> CandidateTimeAnchor:
        votes = []
        bucket = candidate.reason_bucket
        votes.extend(self._metric_votes(candidate, metric_df))
        votes.extend(self._log_votes(candidate, log_df))
        votes.extend(self._trace_votes(candidate, trace_summary))
        if not votes:
            ts = self._bucket(int(fallback_ts))
            fallback = TimeAnchorVote(ts, 0.0, "fallback", "window_start", 0.0, {"reason": "no candidate-local time evidence"})
            return CandidateTimeAnchor(ts, 0.0, "fallback_window_start", (fallback,), ())

        scored = self._aggregate_votes(votes, bucket)
        selected_ts, selected_score = sorted(scored.items(), key=lambda item: (-item[1], item[0]))[0]
        selected = tuple(vote for vote in votes if vote.timestamp == selected_ts)
        rejected = tuple(sorted((vote for vote in votes if vote.timestamp != selected_ts), key=lambda vote: (-vote.score, vote.timestamp))[: self.policy.top_k_votes])
        return CandidateTimeAnchor(int(selected_ts), float(selected_score), self._policy_name(bucket), selected, rejected)

    def _metric_votes(self, candidate: RootCandidate, metric_df: pd.DataFrame) -> list[TimeAnchorVote]:
        if metric_df is None or metric_df.empty:
            return []
        components = self._metric_components(candidate.component)
        rows = metric_df[metric_df["cmdb_id"].astype(str).isin(components)].copy()
        if rows.empty:
            return []
        bucket = _metric_bucket(candidate.reason_bucket)
        rows = rows[rows["kpi_name"].map(lambda name: kpi_in_bucket(str(name), bucket)).astype(bool)]
        if rows.empty:
            return []
        rows["value"] = pd.to_numeric(rows["value"], errors="coerce")
        rows = rows.dropna(subset=["timestamp", "cmdb_id", "kpi_name", "value"])
        events = []
        for row in rows.itertuples(index=False):
            try:
                result = self.baseline.is_anomalous(row.cmdb_id, row.kpi_name, row.value, threshold="p99")
            except Exception:
                continue
            if not result.is_anomalous:
                continue
            strength = abs(float(result.deviation or 0.0))
            if strength <= 0:
                continue
            events.append({
                "timestamp": self._bucket(int(row.timestamp)),
                "component": str(row.cmdb_id),
                "kpi_name": str(row.kpi_name),
                "strength": strength,
            })
        if not events:
            return []
        df = pd.DataFrame(events)
        votes: list[TimeAnchorVote] = []
        for ts, group in df.groupby("timestamp"):
            raw = float(group["strength"].map(log1p).sum())
            n_kpi = int(group["kpi_name"].nunique())
            score = raw + max(0, n_kpi - 1) * self.policy.metric_cross_vote_weight
            votes.append(TimeAnchorVote(
                int(ts), score, "metric", "cross_metric_vote", float(group["strength"].max()),
                {"kpis": sorted(group["kpi_name"].unique()), "components": sorted(group["component"].unique()), "n_kpi": n_kpi},
            ))
        peak = df.sort_values(["strength", "timestamp"], ascending=[False, True]).iloc[0]
        votes.append(TimeAnchorVote(
            int(peak["timestamp"]), float(log1p(float(peak["strength"])) + self.policy.metric_peak_bonus),
            "metric", "strongest_peak", float(peak["strength"]),
            {"kpi_name": str(peak["kpi_name"]), "component": str(peak["component"])},
        ))
        onset = self._sustained_onset(df)
        if onset is not None:
            votes.append(onset)
        return votes

    def _log_votes(self, candidate: RootCandidate, log_df: pd.DataFrame) -> list[TimeAnchorVote]:
        if log_df is None or log_df.empty or "value" not in log_df.columns:
            return []
        rows = log_df[log_df["cmdb_id"].astype(str) == str(candidate.component)].copy()
        if rows.empty or "timestamp" not in rows.columns:
            return []
        votes = []
        for row in rows.itertuples(index=False):
            reason = reason_for_log(getattr(row, "value", ""))
            if not reason:
                continue
            if _reason_bucket_text(reason) != candidate.reason_bucket:
                continue
            text = str(getattr(row, "value", ""))
            ts = self._bucket(int(getattr(row, "timestamp")))
            votes.append(TimeAnchorVote(
                ts, self.policy.log_match_bonus, "log", "reason_keyword_onset", 1.0,
                {"component": candidate.component, "reason": reason, "text": text[:180]},
            ))
        return votes[: self.policy.top_k_votes]

    def _trace_votes(self, candidate: RootCandidate, trace_summary: Mapping[str, Any] | None) -> list[TimeAnchorVote]:
        summary = trace_summary or {}
        if summary.get("trace_status") != "present":
            return []
        bucket = candidate.reason_bucket
        event_names = ["slow_edges"] if bucket == "network_latency" else ["dropped_edges", "slow_edges"]
        votes = []
        for name in event_names:
            for edge in summary.get("events", {}).get(name, []):
                if str(edge.get("src")) != str(candidate.component) and str(edge.get("dst")) != str(candidate.component):
                    continue
                ts = _edge_ts(edge)
                if ts is None:
                    continue
                strength_key = "slow_ratio" if name == "slow_edges" else "count_drop_ratio"
                strength = float(edge.get(strength_key, 0.0) or 0.0)
                role_bonus = self.policy.trace_role_bonus if str(edge.get("src")) == str(candidate.component) else 0.0
                score = self.policy.trace_first_bonus + log1p(abs(strength)) + role_bonus
                votes.append(TimeAnchorVote(
                    self._bucket(ts), score, "trace", f"{name}_first_seen", abs(strength),
                    {"edge": dict(edge), "component_role": "src" if str(edge.get("src")) == str(candidate.component) else "dst"},
                ))
        return votes

    def _aggregate_votes(self, votes: list[TimeAnchorVote], bucket: str) -> dict[int, float]:
        by_ts: dict[int, float] = {}
        earliest = min(vote.timestamp for vote in votes)
        for vote in votes:
            weight = self._source_weight(bucket, vote)
            score = vote.score * weight
            if vote.timestamp == earliest:
                score += self.policy.early_vote_bonus
            by_ts[vote.timestamp] = by_ts.get(vote.timestamp, 0.0) + score
        return by_ts

    def _source_weight(self, bucket: str, vote: TimeAnchorVote) -> float:
        if bucket == "jvm_oom":
            return {"log": 1.6, "metric": 1.1, "trace": 0.45}.get(vote.source, 1.0)
        if bucket in {"network_latency", "network_packet_loss"}:
            return {"trace": 1.55, "metric": 0.9, "log": 1.0}.get(vote.source, 1.0)
        if bucket == "filesystem":
            return {"metric": 1.35, "log": 1.1, "trace": 0.35}.get(vote.source, 1.0)
        return {"metric": 1.25, "log": 1.0, "trace": 0.55}.get(vote.source, 1.0)

    def _sustained_onset(self, df: pd.DataFrame) -> TimeAnchorVote | None:
        by_ts = df.groupby("timestamp")["strength"].sum().sort_index()
        if len(by_ts) < 2:
            return None
        threshold = max(3.0, float(by_ts.quantile(0.65)))
        ordered = list(by_ts.items())
        for idx, (ts, strength) in enumerate(ordered[:-1]):
            next_ts, next_strength = ordered[idx + 1]
            if int(next_ts) - int(ts) > self.policy.bucket_seconds * 5:
                continue
            if strength >= threshold and next_strength >= threshold:
                return TimeAnchorVote(
                    int(ts), float(log1p(strength) + self.policy.metric_sustained_bonus),
                    "metric", "sustained_onset", float(strength),
                    {"threshold": threshold, "next_timestamp": int(next_ts), "next_strength": float(next_strength)},
                )
        return None

    def _metric_components(self, component: str) -> set[str]:
        out = {str(component)}
        node = self.node_graph.get("containers", {}).get(str(component), {}).get("node_proxy")
        if node:
            out.add(str(node))
        hosted = self.node_graph.get("nodes", {}).get(str(component), {}).get("hosted_containers", [])
        out.update(str(item) for item in hosted)
        return out

    def _bucket(self, timestamp: int) -> int:
        size = max(1, int(self.policy.bucket_seconds))
        return int(timestamp // size * size)

    @staticmethod
    def _policy_name(bucket: str) -> str:
        if bucket in {"network_latency", "network_packet_loss"}:
            return "network_trace_first_seen_with_metric_vote"
        if bucket == "jvm_oom":
            return "jvm_log_then_heap_onset"
        return "metric_sustained_cross_vote"


def _edge_ts(edge: Mapping[str, Any]) -> int | None:
    ts = edge.get("first_timestamp")
    if ts is None:
        return None
    ts = int(ts)
    if ts > 10_000_000_000:
        ts //= 1000
    return ts


def _metric_bucket(bucket: str) -> str:
    return "network" if bucket in {"network_latency", "network_packet_loss"} else bucket


def _reason_bucket_text(reason: str) -> str:
    from refute_b_v2_d32.schema import reason_bucket

    return reason_bucket(reason)
