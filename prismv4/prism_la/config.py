"""PRISM-LA configuration dataclass and constants."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional
import os
import sys


@dataclass
class PRISMLAConfig:
    max_investigation_steps: int = 6
    max_event_hypotheses: int = 8
    max_events_in_state: int = 12
    evidence_mode: str = "structured"
    require_dual_modality: bool = True
    require_evidence_citations: bool = True
    min_root_margin: float = 0.08
    min_source_isolation: float = 0.38
    stop_gap_threshold: float = 0.12
    stop_entropy_threshold: float = 0.42
    stop_coverage_threshold: int = 1
    top_anchor_count: int = 5
    max_anatomaly_window_seconds: int = 3600
    pairwise_compare_top_k: int = 4
    max_plan_actions_per_step: int = 5
    llm_model: str = os.environ.get("PRISM_LLM_MODEL", "gpt-4o")
    llm_temperature: float = 0.15
    llm_max_tokens: int = 4096
    llm_api_key: Optional[str] = field(default_factory=lambda: os.environ.get("OPENAI_API_KEY"))
    llm_base_url: Optional[str] = field(default_factory=lambda: os.environ.get("OPENAI_BASE_URL"))
    forced_revision_count: int = 2
    allow_parent_symptom_demotion: bool = True
    allow_counterfactual_proxy: bool = True
    log_llm_messages: bool = True
    debug_dir: Optional[Path] = None


DEFAULT_CONFIG = PRISMLAConfig()


BENCHMARK_SYSTEMS: Dict[str, Dict[str, Any]] = {
    "Bank": {
        "has_logs": True,
        "has_traces": True,
        "entity_types": ["pod"],
    },
    "Telecom": {
        "has_logs": False,
        "has_traces": True,
        "entity_types": ["service", "node", "container", "middleware"],
    },
    "Market": {
        "has_logs": True,
        "has_traces": True,
        "entity_types": ["service", "container", "node", "mesh", "runtime"],
    },
}


FORBIDDEN_LLM_INPUTS = frozenset({
    "ground_truth",
    "gt_component",
    "gt_reason",
    "gt_time",
    "scoring_points",
    "record_csv",
    "benchmark_label",
    "evaluation_score",
    "historical_result",
    "learned_weight",
    "calibrated_threshold",
    "benchmark_tuned_examples",
})
