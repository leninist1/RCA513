"""
layer3_llm.py -- Phase 5 semantic/report layer.

This file provides a deterministic local implementation of the LLM-facing
interface. A real model can later replace `generate_report` without changing
the pipeline contract.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from refute.src.layer2_refutation import RefutationReport
from refute.src.llm_client import BaseLLMClient, LLMRequestError


@dataclass(frozen=True)
class SemanticDecision:
    winner: str | None
    confidence: float
    needs_llm: bool
    report: str
    questions: List[str]
    llm_provider: str = "offline"
    llm_error: str | None = None

    def to_dict(self) -> dict:
        return {
            "winner": self.winner,
            "confidence": self.confidence,
            "needs_llm": self.needs_llm,
            "report": self.report,
            "questions": self.questions,
            "llm_provider": self.llm_provider,
            "llm_error": self.llm_error,
        }


def needs_semantic_arbitration(reports: Iterable[RefutationReport], gap_threshold: float = 0.3) -> bool:
    ranked = list(reports)
    if len(ranked) < 2:
        return False
    first, second = ranked[0], ranked[1]
    score_gap = second.rebuttal_score - first.rebuttal_score
    strength_gap = first.support_strength - second.support_strength
    return score_gap < gap_threshold and abs(strength_gap) < 1.0


def estimate_confidence(reports: Iterable[RefutationReport]) -> float:
    ranked = list(reports)
    if not ranked:
        return 0.0
    best = ranked[0]
    base = 0.35
    base += min(0.45, best.support_strength / 50.0)
    base -= min(0.25, best.rebuttal_score * 0.15)
    if len(ranked) > 1 and best.rebuttal_score < ranked[1].rebuttal_score:
        base += 0.15
    return max(0.0, min(0.95, base))


def _build_llm_prompt(case_id: str, reports: List[RefutationReport], local_report: str) -> tuple[str, str]:
    system = (
        "你是 SRE 根因分析助手。你只能基于给定结构化证据做判断；"
        "不要编造不存在的指标。输出中文，包含 winner、confidence、证据、需要补的数据。"
    )
    compact = []
    for report in reports[:5]:
        compact.append(report.to_dict())
    user = (
        f"case_id: {case_id}\n"
        f"本地算法报告: {local_report}\n"
        "候选与反驳证据 JSON:\n"
        f"{compact}\n"
        "请给出简洁调查结论。"
    )
    return system, user


def generate_report(case_id: str, reports: List[RefutationReport],
                    llm_client: BaseLLMClient | None = None,
                    force_llm: bool = False) -> SemanticDecision:
    if not reports:
        return SemanticDecision(
            winner=None,
            confidence=0.0,
            needs_llm=True,
            report=f"{case_id}: no candidates produced; data blind spot.",
            questions=["Check whether metric/log/trace files are present for the fault window."],
        )
    best = reports[0]
    confidence = estimate_confidence(reports)
    needs_llm = needs_semantic_arbitration(reports)
    support = "; ".join(best.supporting[:3]) if best.supporting else "no direct support"
    refuting = "; ".join(best.refuting[:2]) if best.refuting else "no strong refutation"
    report = (
        f"{case_id}: most suspicious candidate is {best.candidate.service} "
        f"({best.candidate.reason}); rebuttal_score={best.rebuttal_score:.2f}, "
        f"support_strength={best.support_strength:.2f}. Support: {support}. "
        f"Refutation: {refuting}."
    )
    questions = []
    if needs_llm:
        questions.append("Top candidates are close; inspect trace direction and service-local logs.")
    if confidence < 0.5:
        questions.append("Confidence is low; collect denser metric samples around the injection time.")
    local_decision = SemanticDecision(
        winner=best.candidate.service,
        confidence=confidence,
        needs_llm=needs_llm,
        report=report,
        questions=questions,
    )
    if llm_client is None or (not force_llm and not needs_llm):
        return local_decision
    system, user = _build_llm_prompt(case_id, reports, report)
    try:
        response = llm_client.complete(system, user)
        return SemanticDecision(
            winner=best.candidate.service,
            confidence=confidence,
            needs_llm=needs_llm,
            report=response.text or report,
            questions=questions,
            llm_provider=response.provider,
        )
    except LLMRequestError as exc:
        return SemanticDecision(
            winner=best.candidate.service,
            confidence=confidence,
            needs_llm=needs_llm,
            report=report,
            questions=questions,
            llm_provider=getattr(llm_client, "provider", "unknown"),
            llm_error=str(exc),
        )
