"""Shared constants, dataclasses, and system paths."""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

# ======== Paths ========
OPENRCA_ROOT = "/home/dell2/RCA513/yyx/OpenRCA"

SYSTEM_PATHS = {
    "Bank": {
        "root": os.path.join(OPENRCA_ROOT, "Bank", "Bank"),
        "sub_systems": [""],
        "has_logs": True,
        "has_traces": True,
    },
    "Telecom": {
        "root": os.path.join(OPENRCA_ROOT, "Telecom", "Telecom"),
        "sub_systems": [""],
        "has_logs": False,
        "has_traces": True,
    },
    "Market": {
        "root": os.path.join(OPENRCA_ROOT, "Market", "Market"),
        "sub_systems": ["cloudbed-1", "cloudbed-2"],
        "has_logs": True,
        "has_traces": True,
    },
}

# ======== RCA Parameters ========
BASELINE_WINDOW_SECONDS = 300
FAULT_WINDOW_SECONDS = 300
TOP_K = 7
ANOMALY_Z_THRESHOLD = 3.0
TIME_TOLERANCE_SECONDS = 60
EDGE_IMPORTANCE_MIN_CALLS = 1

# ======== Controller Parameters ========
MAX_ITERATIONS = 4
BUDGET_TOTAL_API_CALLS = 10
BUDGET_TOTAL_TOKENS = 100_000
CONFIDENCE_STOP_THRESHOLD = 0.85
ANXIETY_STOP_WITH_ANSWER = 0.9
ANXIETY_STOP_EXHAUSTED = 0.95

# ======== Metric Weights (per metric dimension) ========
DEFAULT_METRIC_WEIGHTS = {
    "rr": 1.0,      # request rate
    "sr": 1.0,      # success rate
    "cnt": 1.0,     # count
    "mrt": 1.0,     # mean response time
    "count": 1.0,
    "avg_time": 1.0,
    "num": 1.0,
    "succee_num": 1.0,
    "succee_rate": 1.0,
    "value": 1.0,   # generic key-value metric
    "duration": 1.0,
}

# ======== Log Keyword Tiers (from Phase 1) ========
LOG_KEYWORD_TIERS = {
    "OOM Killed": 10, "Container Killed": 10, "OutOfMemoryError": 10,
    "oom": 10, "killed": 10,
    "Connection Refused": 5, "connection refused": 5,
    "500 Internal Server Error": 5, "500": 5,
    "Timeout": 2, "timeout": 2, "timed out": 2,
    "Exception": 1, "exception": 1, "ERROR": 1, "error": 1,
}


# ======== Core Dataclasses ========

@dataclass
class ScoringPoint:
    field_type: str     # "time" | "component" | "reason"
    rank: int           # 0 for single-fault, 0/1 for multi-fault
    expected_value: str


@dataclass
class GroundTruth:
    component: Optional[str] = None
    reason: Optional[str] = None
    timestamp: Optional[float] = None
    datetime_str: Optional[str] = None


@dataclass
class QueryCase:
    task_index: str                     # "task_1" through "task_7"
    system: str                         # "Bank", "Telecom", "Market"
    sub_system: str                     # "" or "cloudbed-1", "cloudbed-2"
    instruction: str                    # Natural language query
    time_window: Tuple[str, str]        # (start_dt, end_dt) as "YYYY-MM-DD HH:MM:SS"
    scoring_points: List[ScoringPoint] = field(default_factory=list)
    inject_time: Optional[float] = None         # Unix timestamp of root cause
    telemetry_date: Optional[str] = None        # "YYYY_MM_DD" folder name
    ground_truth: Optional[GroundTruth] = None


@dataclass
class UnifiedTelemetry:
    """Standardized telemetry across all OpenRCA systems."""
    metrics: Any = None          # pd.DataFrame: timestamp, entity, metric_name, value
    logs: Any = None             # pd.DataFrame or None: timestamp, entity, message
    traces: Any = None           # pd.DataFrame or None: timestamp, entity, trace_id, span_id, parent_id, duration, status_code
    entities: List[str] = field(default_factory=list)
    entity_types: Dict[str, str] = field(default_factory=dict)  # entity -> "pod"/"service"/"node"/"container"
    system: str = ""


@dataclass
class ResourceBudget:
    total_api_calls: int = BUDGET_TOTAL_API_CALLS
    total_tokens: int = BUDGET_TOTAL_TOKENS
    api_calls_used: int = 0
    tokens_used: int = 0

    @property
    def calls_remaining(self) -> int:
        return max(0, self.total_api_calls - self.api_calls_used)

    @property
    def budget_ratio(self) -> float:
        return self.api_calls_used / max(1, self.total_api_calls)


@dataclass
class ResourceCost:
    api_calls: int = 0
    tokens: int = 0
    compute_seconds: float = 0.0


@dataclass
class Candidate:
    entity: str
    anomaly_score: float = 0.0
    recovery_score: Optional[float] = None
    verification_depth: str = "none"       # "none"|"shallow"|"medium"|"deep"
    evidence: List[Dict] = field(default_factory=list)


@dataclass
class StrategyResult:
    strategy_name: str
    candidates_verified: List[Candidate] = field(default_factory=list)
    evidence_added: Dict[str, List[Dict]] = field(default_factory=dict)
    cost: ResourceCost = field(default_factory=ResourceCost)


@dataclass
class ControllerAction:
    strategy: str           # "A_broad_shallow"|"B_deep_dive"|"C_causal_trace"|"D_log_compare"
    depth: str              # "shallow"|"medium"|"deep"
    stop: bool = False
    stop_reason: str = ""
    focus_entities: List[str] = field(default_factory=list)


@dataclass
class EvalResult:
    correct: bool
    partial: bool
    field_scores: Dict[str, bool] = field(default_factory=dict)
    task_type: str = ""
    system: str = ""
    query_index: str = ""
    prediction: Dict = field(default_factory=dict)
    ground_truth: Optional[GroundTruth] = None


def _required_fields_for_task(task_type: int) -> List[str]:
    """Map task type to required output fields."""
    mapping = {
        1: ["time"],
        2: ["reason"],
        3: ["component"],
        4: ["time", "reason"],
        5: ["time", "component"],
        6: ["component", "reason"],
        7: ["time", "component", "reason"],
    }
    return mapping.get(task_type, ["component", "reason"])
