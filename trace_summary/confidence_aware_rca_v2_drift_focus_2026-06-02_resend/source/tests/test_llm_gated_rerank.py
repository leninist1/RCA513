import json

from refute_b_v2_d32.llm_gated_rerank import GatedRerankConfig, apply_gated_rerank, apply_pairwise_gated_rerank


def _completed():
    return {
        0: {
            "row_id": 0,
            "prediction": {
                "1": {
                    "root cause occurrence datetime": "2021-03-04 10:00:00",
                    "root cause component": "A",
                    "root cause reason": "high CPU usage",
                }
            },
            "debug": {
                "d32_result": {
                    "debug": {
                        "case_id": "query_000",
                        "all_decisions": [
                            {
                                "candidate": {"component": "A", "reason": "CPU fault"},
                                "rebuttal_score": -1.0,
                                "support_strength": 1.0,
                                "refute_strength": 0.0,
                            },
                            {
                                "candidate": {"component": "B", "reason": "network delay"},
                                "rebuttal_score": -0.5,
                                "support_strength": 0.8,
                                "refute_strength": 0.1,
                            },
                            {
                                "candidate": {"component": "C", "reason": "network loss"},
                                "rebuttal_score": -0.4,
                                "support_strength": 0.6,
                                "refute_strength": 0.0,
                            },
                        ],
                    }
                }
            },
        }
    }


def _judgment(rank, verdict, support, refute, parse_ok=True):
    return {
        "case_id": "query_000",
        "candidate_rank": rank,
        "candidate": {"component": chr(ord("A") + rank - 1), "reason": "x"},
        "parse_ok": parse_ok,
        "llm_judgment": {
            "verdict": verdict,
            "support_score": support,
            "refute_score": refute,
        },
    }


def _pairwise(rank, preferred, alt_support, margin, top1_refute=0.0, top1_support=0.0, alt_refute=0.0, parse_ok=True):
    return {
        "case_id": "query_000",
        "alternative_rank": rank,
        "top1_candidate": {"component": "A", "reason": "CPU fault"},
        "alternative_candidate": {"component": chr(ord("A") + rank - 1), "reason": "x"},
        "parse_ok": parse_ok,
        "llm_pairwise_judgment": {
            "preferred_candidate": preferred,
            "top1_support_score": top1_support,
            "top1_refute_score": top1_refute,
            "alternative_support_score": alt_support,
            "alternative_refute_score": alt_refute,
            "relative_margin": margin,
        },
    }


def test_gated_rerank_changes_only_when_top1_refuted_and_alt_supported(tmp_path):
    reranked, summary = apply_gated_rerank(
        completed=_completed(),
        judgment_rows=[
            _judgment(1, "refute", 0.1, 0.9),
            _judgment(2, "support", 0.82, 0.05),
        ],
        trace_path=tmp_path / "trace.jsonl",
        summary_path=tmp_path / "summary.json",
        config=GatedRerankConfig(),
        reason_name_map={"network delay": "network latency"},
    )
    top1 = reranked[0]["prediction"]["1"]
    assert top1["root cause occurrence datetime"] == "2021-03-04 10:00:00"
    assert top1["root cause component"] == "B"
    assert top1["root cause reason"] == "network latency"
    trace = json.loads((tmp_path / "trace.jsonl").read_text(encoding="utf-8").strip())
    assert trace["changed"] is True
    assert trace["new_top1"]["old_rank"] == 2
    assert summary["changed_cases"] == 1


def test_gated_rerank_tie_breaks_by_refute_then_d32_rank(tmp_path):
    reranked, _ = apply_gated_rerank(
        completed=_completed(),
        judgment_rows=[
            _judgment(1, "refute", 0.1, 0.9),
            _judgment(2, "support", 0.82, 0.20),
            _judgment(3, "support", 0.82, 0.05),
        ],
        trace_path=tmp_path / "trace.jsonl",
        summary_path=tmp_path / "summary.json",
        config=GatedRerankConfig(),
        reason_name_map={"network loss": "network packet loss"},
    )
    assert reranked[0]["prediction"]["1"]["root cause component"] == "C"
    assert reranked[0]["prediction"]["1"]["root cause reason"] == "network packet loss"


def test_gated_rerank_keeps_when_top1_parse_failed(tmp_path):
    reranked, summary = apply_gated_rerank(
        completed=_completed(),
        judgment_rows=[
            _judgment(1, "uncertain", 0.0, 0.0, parse_ok=False),
            _judgment(2, "support", 0.95, 0.0),
        ],
        trace_path=tmp_path / "trace.jsonl",
        summary_path=tmp_path / "summary.json",
        config=GatedRerankConfig(),
    )
    assert reranked[0]["prediction"]["1"]["root cause component"] == "A"
    trace = json.loads((tmp_path / "trace.jsonl").read_text(encoding="utf-8").strip())
    assert trace["changed"] is False
    assert trace["reason_for_keep"] == "llm_parse_failed"
    assert summary["llm_parse_failures"] == 1


def test_pairwise_gated_rerank_changes_when_alternative_preferred_and_top1_refuted(tmp_path):
    reranked, summary = apply_pairwise_gated_rerank(
        completed=_completed(),
        pairwise_rows=[
            _pairwise(2, "alternative", alt_support=0.84, margin=0.30, top1_refute=0.85, top1_support=0.25),
        ],
        trace_path=tmp_path / "trace.jsonl",
        summary_path=tmp_path / "summary.json",
        config=GatedRerankConfig(),
        reason_name_map={"network delay": "network latency"},
    )
    top1 = reranked[0]["prediction"]["1"]
    assert top1["root cause occurrence datetime"] == "2021-03-04 10:00:00"
    assert top1["root cause component"] == "B"
    assert top1["root cause reason"] == "network latency"
    trace = json.loads((tmp_path / "trace.jsonl").read_text(encoding="utf-8").strip())
    assert trace["changed"] is True
    assert trace["rerank_judge"] == "pairwise"
    assert trace["reason_for_change"] == "pairwise_alternative_preferred_top1_refuted"
    assert summary["changed_cases"] == 1
    assert summary["rerank_judge"] == "pairwise"


def test_pairwise_gated_rerank_keeps_when_only_top1_support_low_by_default(tmp_path):
    reranked, _ = apply_pairwise_gated_rerank(
        completed=_completed(),
        pairwise_rows=[
            _pairwise(2, "alternative", alt_support=0.82, margin=0.35, top1_refute=0.40, top1_support=0.30),
        ],
        trace_path=tmp_path / "trace.jsonl",
        summary_path=tmp_path / "summary.json",
        config=GatedRerankConfig(),
        reason_name_map={"network delay": "network latency"},
    )
    assert reranked[0]["prediction"]["1"]["root cause component"] == "A"
    trace = json.loads((tmp_path / "trace.jsonl").read_text(encoding="utf-8").strip())
    assert trace["reason_for_keep"] == "pairwise_margin_or_top1_support_gate_failed"


def test_pairwise_gated_rerank_can_change_when_top1_support_low_if_opted_in(tmp_path):
    reranked, _ = apply_pairwise_gated_rerank(
        completed=_completed(),
        pairwise_rows=[
            _pairwise(2, "alternative", alt_support=0.82, margin=0.35, top1_refute=0.40, top1_support=0.30),
        ],
        trace_path=tmp_path / "trace.jsonl",
        summary_path=tmp_path / "summary.json",
        config=GatedRerankConfig(allow_pairwise_low_top1_support=True),
        reason_name_map={"network delay": "network latency"},
    )
    assert reranked[0]["prediction"]["1"]["root cause component"] == "B"
    trace = json.loads((tmp_path / "trace.jsonl").read_text(encoding="utf-8").strip())
    assert trace["reason_for_change"] == "pairwise_alternative_preferred_with_low_top1_support"


def test_pairwise_gated_rerank_keeps_when_margin_gate_fails(tmp_path):
    reranked, summary = apply_pairwise_gated_rerank(
        completed=_completed(),
        pairwise_rows=[
            _pairwise(2, "alternative", alt_support=0.90, margin=0.10, top1_refute=0.95, top1_support=0.10),
        ],
        trace_path=tmp_path / "trace.jsonl",
        summary_path=tmp_path / "summary.json",
        config=GatedRerankConfig(),
    )
    assert reranked[0]["prediction"]["1"]["root cause component"] == "A"
    trace = json.loads((tmp_path / "trace.jsonl").read_text(encoding="utf-8").strip())
    assert trace["changed"] is False
    assert trace["reason_for_keep"] == "pairwise_margin_or_top1_support_gate_failed"
    assert summary["changed_cases"] == 0


def test_pairwise_gated_rerank_tie_breaks_by_margin_support_refute_then_rank(tmp_path):
    reranked, _ = apply_pairwise_gated_rerank(
        completed=_completed(),
        pairwise_rows=[
            _pairwise(2, "alternative", alt_support=0.90, margin=0.40, top1_refute=0.70, top1_support=0.20),
            _pairwise(3, "alternative", alt_support=0.91, margin=0.40, top1_refute=0.70, top1_support=0.20),
        ],
        trace_path=tmp_path / "trace.jsonl",
        summary_path=tmp_path / "summary.json",
        config=GatedRerankConfig(allow_pairwise_low_top1_support=True),
        reason_name_map={"network loss": "network packet loss"},
    )
    assert reranked[0]["prediction"]["1"]["root cause component"] == "C"
