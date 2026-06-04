"""
pipeline.py -- end-to-end refute v2 API.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional

from refute.src.baseline_distributions import BaselineStore
from refute.src.data_loader import BankDataPaths, filter_window, load_log_day, load_metric_day
from refute.src.evidence_query import EvidenceQuery
from refute.src.layer1_knowledge import candidate_services_for_reason
from refute.src.iterative_refutation import IterativeRefutationEngine
from refute.src.layer2_refutation import Candidate, RefutationEngine
from refute.src.layer3_llm import generate_report
from refute.src.llm_client import BaseLLMClient


class RefutePipeline:
    def __init__(
        self,
        baseline_store: BaselineStore,
        node_graph: dict,
        fault_clusters: Optional[dict] = None,
        refutation_rules: Optional[dict] = None,
        llm_client: BaseLLMClient | None = None,
    ):
        self.baseline_store = baseline_store
        self.node_graph = node_graph
        self.fault_clusters = fault_clusters or {"clusters": {}}
        self.refutation_rules = refutation_rules or {"rules": []}
        self.llm_client = llm_client
        self.known_services = sorted(node_graph.get("containers", {}).keys())

    @classmethod
    def from_files(
        cls,
        baseline_path: str | Path,
        node_graph_path: str | Path,
        fault_clusters_path: str | Path | None = None,
        refutation_rules_path: str | Path | None = None,
        llm_client: BaseLLMClient | None = None,
    ) -> "RefutePipeline":
        baseline = BaselineStore.load_json(baseline_path)
        with Path(node_graph_path).open("r", encoding="utf-8") as f:
            node_graph = json.load(f)
        fault_clusters = {"clusters": {}}
        if fault_clusters_path and Path(fault_clusters_path).exists():
            with Path(fault_clusters_path).open("r", encoding="utf-8") as f:
                fault_clusters = json.load(f)
        refutation_rules = {"rules": []}
        if refutation_rules_path and Path(refutation_rules_path).exists():
            with Path(refutation_rules_path).open("r", encoding="utf-8") as f:
                refutation_rules = json.load(f)
        return cls(baseline, node_graph, fault_clusters, refutation_rules, llm_client=llm_client)

    def candidate_services(self, reason: str, fallback_services: Optional[Iterable[str]] = None) -> list[str]:
        fallback = list(fallback_services or self.known_services)
        return candidate_services_for_reason(self.fault_clusters, reason, fallback)

    def analyze_window(
        self,
        metric_df,
        log_df,
        reason: str,
        case_id: str,
        trace_df=None,
        modal_status: Optional[dict] = None,
        candidate_services: Optional[Iterable[str]] = None,
        top_k: int = 5,
        force_llm: bool = False,
        use_iterative: bool = False,
    ) -> dict:
        services = list(candidate_services or self.candidate_services(reason))
        modal_status = modal_status or infer_modal_status(metric_df, log_df, trace_df)
        query = EvidenceQuery(metric_df, self.baseline_store, node_graph=self.node_graph,
                              log_df=log_df, trace_df=trace_df, modal_status=modal_status)
        candidate_objs = [Candidate(service, reason) for service in services]
        engine = RefutationEngine(query)
        reports = engine.rank_candidates(candidate_objs)
        iterative = None
        if use_iterative:
            iterative = IterativeRefutationEngine(query).run(candidate_objs)
        decision = generate_report(case_id, reports, llm_client=self.llm_client, force_llm=force_llm)
        result = {
            "case_id": case_id,
            "reason": reason,
            "high_suspicion": [r.to_dict() for r in reports[:top_k]],
            "low_suspicion": [r.to_dict() for r in reports[top_k:]],
            "semantic_decision": decision.to_dict(),
            "modal_status": modal_status,
            "decision_source": decision_source(modal_status, reports),
            "data_blind_spots": self._blind_spots(metric_df, services, modal_status),
        }
        if iterative is not None:
            result["iterative_diagnosis"] = iterative
            result["evidence_matrix"] = iterative.get("evidence_matrix", [])
        return result

    @staticmethod
    def _blind_spots(metric_df, services: Iterable[str], modal_status: Optional[dict] = None) -> list[dict]:
        blind = []
        for name, status in (modal_status or {}).items():
            if status != "present":
                blind.append({"area": name, "reason": status})
        present = set(metric_df["cmdb_id"].astype(str).unique()) if not metric_df.empty else set()
        blind.extend([
            {"service": svc, "reason": "no metric rows in fault window"}
            for svc in services if svc not in present
        ])
        return blind

    def analyze_bank_record(
        self,
        data_root: str | Path,
        date_key: str,
        timestamp: int,
        reason: str,
        case_id: str,
        fault_window: int = 600,
        force_llm: bool = False,
    ) -> dict:
        paths = BankDataPaths.from_root(data_root)
        metric_day = load_metric_day(paths, date_key)
        log_day = load_log_day(paths, date_key)
        return self.analyze_window(
            filter_window(metric_day, timestamp, fault_window),
            filter_window(log_day, timestamp, fault_window),
            reason=reason,
            case_id=case_id,
            force_llm=force_llm,
        )


def infer_modal_status(metric_df, log_df, trace_df) -> dict:
    def status(df, loaded=True):
        if not loaded:
            return "unloaded"
        if df is None:
            return "missing"
        if getattr(df, "empty", True):
            return "empty_window"
        return "present"
    return {
        "metric": status(metric_df),
        "log": status(log_df),
        "trace": status(trace_df, loaded=trace_df is not None),
    }


def decision_source(modal_status: dict, reports) -> str:
    if all(v in {"missing", "empty_window", "unloaded"} for v in modal_status.values()):
        return "prior_only"
    if any(r.supporting for r in reports[:3]):
        return "evidence_driven"
    if any(v != "present" for v in modal_status.values()):
        return "partial_modal_prior"
    return "weak_evidence"
