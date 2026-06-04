"""Reason competition and calibration."""
from __future__ import annotations

from dataclasses import dataclass, field
from statistics import median


@dataclass(frozen=True)
class ReasonAdjustmentConfig:
    """Inter-reason adjustment coefficients.

    Defaults are expert priors, not trained coefficients. They are deliberately
    explicit so later calibration can replace them with data-estimated values.
    """
    source: str = "expert_prior_v0"
    memory_gc_boost: float = 1.5
    cpu_when_gc_downweight: float = 0.7
    packet_loss_tcp_boost: float = 1.4
    latency_when_tcp_downweight: float = 0.8
    disk_when_memory_downweight: float = 0.6

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "memory_gc_boost": self.memory_gc_boost,
            "cpu_when_gc_downweight": self.cpu_when_gc_downweight,
            "packet_loss_tcp_boost": self.packet_loss_tcp_boost,
            "latency_when_tcp_downweight": self.latency_when_tcp_downweight,
            "disk_when_memory_downweight": self.disk_when_memory_downweight,
        }


@dataclass
class ReasonScoreCalibrator:
    samples: dict[str, list[float]] = field(default_factory=dict)

    def add(self, reason_bucket: str, score: float) -> None:
        self.samples.setdefault(str(reason_bucket), []).append(float(score))

    def normalize(self, reason_bucket: str, score: float) -> float:
        values = self.samples.get(str(reason_bucket), [])
        if len(values) < 2:
            return float(score)
        med = median(values)
        abs_dev = [abs(v - med) for v in values]
        mad = median(abs_dev) or 1.0
        return (float(score) - med) / mad


@dataclass(frozen=True)
class ReasonCandidateScore:
    reason: str
    bucket: str
    raw_score: float
    normalized_score: float
    adjusted_score: float
    adjustments: tuple[str, ...] = ()
    adjustment_source: str = "none"

    def to_dict(self) -> dict:
        return {
            "reason": self.reason,
            "bucket": self.bucket,
            "raw_score": self.raw_score,
            "normalized_score": self.normalized_score,
            "adjusted_score": self.adjusted_score,
            "adjustments": list(self.adjustments),
            "adjustment_source": self.adjustment_source,
        }


def compete_reasons(reason_scores: dict[str, float], evidence_flags: dict[str, bool] | None = None,
                    calibrator: ReasonScoreCalibrator | None = None,
                    adjustment_config: ReasonAdjustmentConfig | None = None) -> list[ReasonCandidateScore]:
    evidence_flags = evidence_flags or {}
    calibrator = calibrator or ReasonScoreCalibrator()
    adjustment_config = adjustment_config or ReasonAdjustmentConfig()
    rows = []
    for reason, raw in reason_scores.items():
        bucket = _bucket(reason)
        norm = calibrator.normalize(bucket, float(raw))
        adjusted = norm
        notes = []
        if bucket in {"memory", "jvm_oom"} and evidence_flags.get("gc_or_heap_pressure"):
            adjusted *= adjustment_config.memory_gc_boost
            notes.append("boost_memory_gc_heap")
        if bucket == "cpu" and evidence_flags.get("gc_or_heap_pressure"):
            adjusted *= adjustment_config.cpu_when_gc_downweight
            notes.append("downweight_cpu_when_gc_heap")
        if bucket == "network_packet_loss" and evidence_flags.get("tcp_retransmit_or_reset"):
            adjusted *= adjustment_config.packet_loss_tcp_boost
            notes.append("boost_packet_loss_tcp")
        if bucket == "network_latency" and evidence_flags.get("tcp_retransmit_or_reset"):
            adjusted *= adjustment_config.latency_when_tcp_downweight
            notes.append("downweight_latency_when_packet_loss")
        if bucket == "disk_io" and evidence_flags.get("memory_pressure"):
            adjusted *= adjustment_config.disk_when_memory_downweight
            notes.append("downweight_disk_when_memory_pressure")
        rows.append(ReasonCandidateScore(reason, bucket, float(raw), norm, adjusted, tuple(notes), adjustment_config.source if notes else "none"))
    return sorted(rows, key=lambda row: (-row.adjusted_score, row.reason))


def _bucket(reason: str) -> str:
    from refute_b_v2.rules import normalize_reason_bucket
    return normalize_reason_bucket(reason)
