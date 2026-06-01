"""Shared constants, dataclasses, and system paths."""

import os
import numpy as np
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
TOP_K = 12
ANOMALY_Z_THRESHOLD = 3.0
TIME_TOLERANCE_SECONDS = 60
EDGE_IMPORTANCE_MIN_CALLS = 1

# ======== Controller Parameters ========
MAX_ITERATIONS = 6
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

# ======== Fault Family → Accompanying Metric Patterns ========
FAULT_ACCOMPANYING_METRICS = {
    "cpu": ["throttl", "cpu_user", "cpuutil", "singlecpu", "cpuload", "cpuwio",
            "jvm_cpuload", "cfs_throttled"],
    "memory": ["failcnt", "pgfault", "memfree", "memused", "memperc",
               "cache_mem", "nocachememperc", "container_memory"],
    "disk_io": ["await", "queue", "util", "dskread", "dskwrite", "dskreadwrite",
                "dskbps", "dskpercentbusy", "rkb_s", "w_await", "avg_q_sz",
                "fs_reads", "fs_writes"],
    "disk_space": ["pct_usage", "free", "used", "dskpercentbusy", "fscapacity",
                   "fsusedspace", "fsavailablespace"],
    "network": ["tcp", "fin_wait", "retransmit", "packet", "sent_queue",
                "received_queue", "icmp_ping", "netkbtotalpersec",
                "network_transmit", "network_receive"],
    "jvm": ["allocation failure", "full gc", "heap", "gc", "jvm_cpuload"],
    "db": ["session_pct", "proc_user", "login_per_sec", "call_per_sec",
           "tnsping", "jdbc", "elapsedtime", "success"],
    "process": ["container_threads", "process", "restart"],
}

# ======== Semantic Anchor Mapping ========
SEMANTIC_ANCHOR_MAP = {
    "allocation failure": ("jvm", 10),
    "full gc": ("jvm", 10),
    "OutOfMemoryError": ("jvm", 10),
    "oom killed": ("jvm", 9),
    "container killed": ("jvm", 9),
    "connection refused": ("db", 8),
    "tnsping_result_time": ("db", 8),
    "Session_pct": ("db", 7),
    "Proc_User_Used_Pct": ("db", 7),
    "Login_Per_Sec": ("db", 7),
    "Call_Per_Sec": ("db", 7),
    "JDBC elapsedTime": ("db", 6),
    "timeout": ("network", 6),
    "unavailable": ("network", 6),
    "packet loss": ("network", 7),
    "corruption": ("network", 7),
    "retransmission": ("network", 6),
    "TCP-FIN-WAIT": ("network", 7),
    "container_threads": ("process", 7),
    "process termination": ("process", 8),
    "ERROR": ("weak", 1),
    "Exception": ("weak", 1),
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

    def remaining_ratio(self) -> float:
        """Fraction of budget remaining [0, 1]."""
        return max(0.0, 1.0 - self.budget_ratio)

    def exhausted(self) -> bool:
        return self.calls_remaining <= 0


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
    strategy: str           # "S1_exclusivity"|"S2_local_strength"|"S3_causal_trace"|"S4_layer_disambig"|"S5_semantic_anchor"
    depth: str              # "shallow"|"medium"|"deep"
    stop: bool = False
    stop_reason: str = ""
    focus_entities: List[str] = field(default_factory=list)


@dataclass
class EvalResult:
    correct: bool
    partial: bool
    field_scores: Dict[str, bool] = field(default_factory=dict)
    official_score: float = 0.0
    official_passing: List[str] = field(default_factory=list)
    official_failing: List[str] = field(default_factory=list)
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


# ======== New Dataclasses: Verification & Emotion Vector System ========

@dataclass
class Verdict:
    """A single verification verdict for one entity on one dimension."""
    label: str               # "support" | "refute" | "inconclusive"
    confidence: float        # [0, 1]
    evidence: List[Dict] = field(default_factory=list)
    counterfactual_note: str = ""

    UNVERIFIED_SENTINEL = "unverified"

    @staticmethod
    def unverified() -> "Verdict":
        return Verdict(label=Verdict.UNVERIFIED_SENTINEL, confidence=0.0)


@dataclass
class VerificationResult:
    """Output from a single strategy execution."""
    dimension: str           # "S1_exclusivity" | "S2_local_strength" | ...
    entity_verdicts: Dict[str, Verdict] = field(default_factory=dict)
    cost: ResourceCost = field(default_factory=ResourceCost)

    def support_count(self, entity: str) -> int:
        v = self.entity_verdicts.get(entity)
        return 1 if v and v.label == "support" else 0

    def refute_count(self, entity: str) -> int:
        v = self.entity_verdicts.get(entity)
        return 1 if v and v.label == "refute" else 0


@dataclass
class EmotionVector:
    """6-D emotion vector capturing the RCA diagnostic emotional state.

    Dimensions:
      conviction  [0,1] — certainty about the leading candidate
      curiosity   [0,1] — desire to explore unverified dimensions
      perplexity  [0,1] — confusion from contradictory evidence
      vigilance   [0,1] — alertness to rank-flip risk
      satiety     [0,1] — sense that enough evidence has been gathered
      anxiety     [0,1] — pressure from resource depletion
    """
    conviction: float = 0.0
    curiosity: float = 0.7
    perplexity: float = 0.0
    vigilance: float = 0.5
    satiety: float = 0.0
    anxiety: float = 0.2

    def to_array(self) -> "np.ndarray":
        return np.array([
            self.conviction, self.curiosity, self.perplexity,
            self.vigilance, self.satiety, self.anxiety,
        ])

    @staticmethod
    def from_array(arr: "np.ndarray") -> "EmotionVector":
        return EmotionVector(
            conviction=float(arr[0]), curiosity=float(arr[1]),
            perplexity=float(arr[2]), vigilance=float(arr[3]),
            satiety=float(arr[4]), anxiety=float(arr[5]),
        )

    def __add__(self, other: "EmotionVector") -> "EmotionVector":
        return EmotionVector(
            conviction=self.conviction + other.conviction,
            curiosity=self.curiosity + other.curiosity,
            perplexity=self.perplexity + other.perplexity,
            vigilance=self.vigilance + other.vigilance,
            satiety=self.satiety + other.satiety,
            anxiety=self.anxiety + other.anxiety,
        )

    def __sub__(self, other: "EmotionVector") -> "EmotionVector":
        return EmotionVector(
            conviction=self.conviction - other.conviction,
            curiosity=self.curiosity - other.curiosity,
            perplexity=self.perplexity - other.perplexity,
            vigilance=self.vigilance - other.vigilance,
            satiety=self.satiety - other.satiety,
            anxiety=self.anxiety - other.anxiety,
        )

    def __mul__(self, scalar: float) -> "EmotionVector":
        return EmotionVector(
            conviction=self.conviction * scalar,
            curiosity=self.curiosity * scalar,
            perplexity=self.perplexity * scalar,
            vigilance=self.vigilance * scalar,
            satiety=self.satiety * scalar,
            anxiety=self.anxiety * scalar,
        )

    __rmul__ = __mul__

    def clamp(self) -> "EmotionVector":
        """Clamp all dimensions to [0, 1]."""
        return EmotionVector(
            conviction=max(0.0, min(1.0, self.conviction)),
            curiosity=max(0.0, min(1.0, self.curiosity)),
            perplexity=max(0.0, min(1.0, self.perplexity)),
            vigilance=max(0.0, min(1.0, self.vigilance)),
            satiety=max(0.0, min(1.0, self.satiety)),
            anxiety=max(0.0, min(1.0, self.anxiety)),
        )

    def cosine_similarity(self, other: "EmotionVector") -> float:
        """Cosine similarity between two emotion vectors."""
        a = self.to_array()
        b = other.to_array()
        dot = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-9 or norm_b < 1e-9:
            return 0.0
        return float(max(0.0, min(1.0, dot / (norm_a * norm_b))))


@dataclass
class TrajectoryRecord:
    """One step in a meta-controller trajectory."""
    emotion_before: EmotionVector
    strategy: str
    emotion_after: EmotionVector
    system: str
    reason_family: str
    outcome: str = ""          # "correct" | "partial" | "wrong"
    query_id: str = ""

    @property
    def delta(self) -> EmotionVector:
        return self.emotion_after - self.emotion_before


# ======== Cold-Start Route Table ========
# Used in Phase 1 to collect trajectories before vector methods take over.
# Routes are (system, reason_pattern) → [strategy_sequence]

COLD_START_ROUTES = {
    # Active exploration routes: verify → expand → scan new entities → verify
    # Bank — all pod-level
    ("Bank", "cpu"):          ["VERIFY_S1", "GRAPH_EXPAND", "ENGINE_SCAN", "VERIFY_S2"],
    ("Bank", "memory"):       ["VERIFY_S1", "GRAPH_EXPAND", "ENGINE_SCAN", "VERIFY_S2"],
    ("Bank", "disk_io"):      ["VERIFY_S1", "GRAPH_EXPAND", "ENGINE_SCAN", "VERIFY_S2"],
    ("Bank", "disk_space"):   ["VERIFY_S1", "GRAPH_EXPAND", "ENGINE_SCAN", "VERIFY_S2"],
    ("Bank", "jvm_cpu"):      ["VERIFY_S1", "GRAPH_EXPAND", "ENGINE_SCAN", "VERIFY_S2"],
    ("Bank", "packet_loss"):  ["VERIFY_S1", "VERIFY_S3", "GRAPH_EXPAND", "ENGINE_SCAN", "VERIFY_S1"],
    ("Bank", "latency"):      ["VERIFY_S1", "VERIFY_S3", "GRAPH_EXPAND", "ENGINE_SCAN", "VERIFY_S1"],
    ("Bank", "oom"):          ["VERIFY_S5", "GRAPH_EXPAND", "ENGINE_SCAN", "VERIFY_S2"],

    # Telecom — mixed layers
    ("Telecom", "cpu"):       ["VERIFY_S2", "GRAPH_EXPAND", "ENGINE_SCAN"],
    ("Telecom", "network"):   ["VERIFY_S1", "VERIFY_S3", "GRAPH_EXPAND", "ENGINE_SCAN"],
    ("Telecom", "db"):        ["VERIFY_S5", "VERIFY_S3", "GRAPH_EXPAND", "ENGINE_SCAN"],

    # Market — three-layer, S4 essential
    ("Market", "*"):          ["VERIFY_S1", "VERIFY_S4", "GRAPH_EXPAND", "ENGINE_SCAN", "VERIFY_S4"],
}

# ======== Strategy-to-Dimension Mapping ========
STRATEGY_DIMENSIONS = {
    "S1_exclusivity":       "exclusivity",
    "S2_local_strength":    "local_strength",
    "S3_causal_trace":      "causal_source",
    "S4_layer_disambig":    "layer",
    "S5_semantic_anchor":   "semantic",
}

# All 5 strategy names
ALL_STRATEGIES = list(STRATEGY_DIMENSIONS.keys())

# ======== Emotion Vector Initial Weights for Stop Projection ========
# Semantic prior for w in sigmoid(w·e + b):
#   conviction↑→stop(+), curiosity↑→continue(-), perplexity↑→continue(-),
#   vigilance↑→continue(-), satiety↑→stop(+), anxiety↑→stop(+)
INITIAL_STOP_PROJECTION_W = [0.30, -0.10, -0.25, -0.20, 0.20, 0.25]
INITIAL_STOP_PROJECTION_B = 0.0
INITIAL_STOP_THRESHOLD = 0.80


# ======== Hypothesis & HypothesisList — Dynamic Belief State ========

@dataclass
class Hypothesis:
    """One root-cause hypothesis in the belief state."""
    entity: str
    layer: str = "unknown"               # "node" | "service" | "pod" | "unknown"
    confidence: float = 0.0              # [0, 1] composite belief score
    status: str = "active"               # "active" | "reserved" | "eliminated"
    engine_score: Optional[float] = None # recovery_score from engine
    anomaly_score: float = 0.0
    verdicts: Dict[str, Any] = field(default_factory=dict)  # {dim: Verdict}
    evidence: List[Dict] = field(default_factory=list)
    reason_family: str = "unknown"
    added_by: str = ""                   # tool name that added this hypothesis
    iteration_added: int = 0
    last_updated: int = 0
    confidence_rationale: str = ""       # traceable reason for confidence value

    @property
    def n_supports(self) -> int:
        return sum(1 for v in self.verdicts.values()
                   if hasattr(v, 'label') and v.label == "support")

    @property
    def n_refutes(self) -> int:
        return sum(1 for v in self.verdicts.values()
                   if hasattr(v, 'label') and v.label == "refute")

    @property
    def has_conflict(self) -> bool:
        return self.n_supports > 0 and self.n_refutes > 0


class HypothesisList:
    """Dynamic belief state managed by the meta-controller.

    Maintains three pools:
      active    — candidates under active consideration, sorted by confidence
      reserved  — low-confidence but not eliminated; can be recalled
      eliminated — definitively ruled out

    Supports the 5 core operations: add, reorder, reserve, recall, eliminate.
    Provides distribution statistics that directly feed the emotion vector.
    """

    def __init__(self):
        self.active: List[Hypothesis] = []
        self.reserved: List[Hypothesis] = []
        self.eliminated: List[Hypothesis] = []
        self.history: List[Dict] = []
        self._entity_index: Dict[str, Hypothesis] = {}  # entity → Hypothesis lookup

    # ---- Core Operations ----

    def add(self, h: Hypothesis, target: str = "active", iteration: int = 0):
        """Add a hypothesis to the specified pool."""
        h.iteration_added = iteration
        h.last_updated = iteration
        if target == "active":
            self.active.append(h)
        elif target == "reserved":
            self.reserved.append(h)
        else:
            self.eliminated.append(h)
        self._entity_index[h.entity] = h
        self.history.append({
            "op": "add", "entity": h.entity, "target": target,
            "confidence": h.confidence, "iteration": iteration,
            "added_by": h.added_by,
        })

    def get(self, entity: str) -> Optional[Hypothesis]:
        return self._entity_index.get(entity)

    def has(self, entity: str) -> bool:
        return entity in self._entity_index

    def update_confidence(self, entity: str, new_confidence: float,
                          reason: str = "", iteration: int = 0):
        """Update confidence and potentially re-evaluate status."""
        h = self._entity_index.get(entity)
        if h is None:
            return
        h.confidence = new_confidence
        h.last_updated = iteration
        h.confidence_rationale = reason
        # Auto-reorder active list
        self.active.sort(key=lambda x: x.confidence, reverse=True)
        self.history.append({
            "op": "update_confidence", "entity": entity,
            "confidence": new_confidence, "reason": reason, "iteration": iteration,
        })

    def add_verdict(self, entity: str, dimension: str, verdict, iteration: int = 0):
        """Record a verification verdict for an entity and recompute confidence."""
        h = self._entity_index.get(entity)
        if h is None:
            return
        h.verdicts[dimension] = verdict
        h.last_updated = iteration
        # Recompute confidence from verdicts + engine_score
        self._recompute_confidence(h, iteration)

    def reorder(self, iteration: int = 0):
        """Recompute all active confidences and re-sort."""
        for h in self.active:
            self._recompute_confidence(h, iteration)
        for h in self.reserved:
            self._recompute_confidence(h, iteration)
        self.active.sort(key=lambda x: x.confidence, reverse=True)

    def reserve(self, entity: str, reason: str = "", iteration: int = 0):
        """Move entity from active → reserved."""
        h = self._entity_index.get(entity)
        if h is None or h.status != "active":
            return
        h.status = "reserved"
        h.confidence_rationale = reason
        h.last_updated = iteration
        self.active = [x for x in self.active if x.entity != entity]
        self.reserved.append(h)
        self.history.append({
            "op": "reserve", "entity": entity, "reason": reason, "iteration": iteration,
        })

    def recall(self, entity: str, reason: str = "", iteration: int = 0):
        """Move entity from reserved → active."""
        h = self._entity_index.get(entity)
        if h is None or h.status != "reserved":
            return
        h.status = "active"
        h.confidence_rationale = reason
        h.last_updated = iteration
        self.reserved = [x for x in self.reserved if x.entity != entity]
        self.active.append(h)
        self.active.sort(key=lambda x: x.confidence, reverse=True)
        self.history.append({
            "op": "recall", "entity": entity, "reason": reason, "iteration": iteration,
        })

    def eliminate(self, entity: str, reason: str = "", iteration: int = 0):
        """Move entity from active/reserved → eliminated."""
        h = self._entity_index.get(entity)
        if h is None:
            return
        h.status = "eliminated"
        h.confidence_rationale = reason
        h.last_updated = iteration
        self.active = [x for x in self.active if x.entity != entity]
        self.reserved = [x for x in self.reserved if x.entity != entity]
        self.eliminated.append(h)
        self.history.append({
            "op": "eliminate", "entity": entity, "reason": reason, "iteration": iteration,
        })

    def top(self, n: int = 3) -> List[Hypothesis]:
        return self.active[:n]

    @property
    def top_entity(self) -> Optional[str]:
        return self.active[0].entity if self.active else None

    def top_entities(self, n: int = 5) -> List[str]:
        return [h.entity for h in self.active[:n]]

    # ---- Confidence Computation ----

    def _recompute_confidence(self, h: Hypothesis, iteration: int = 0):
        """Recompute composite confidence from engine_score + verdicts.

        Formula:
          base = engine_score (if computed) else anomaly_score_normalized
          modifier = +0.10 × n_supports - 0.15 × n_refutes - 0.05 × n_inconclusive
          coherence_bonus = 1.15 if (has support AND no refute) else 1.0
          status_penalty = 0.7 if reserved else 1.0
        """
        base = h.engine_score if h.engine_score is not None else min(1.0, h.anomaly_score / 20.0)
        n_inc = sum(1 for v in h.verdicts.values()
                    if hasattr(v, 'label') and v.label == "inconclusive")
        modifier = 0.10 * h.n_supports - 0.15 * h.n_refutes - 0.05 * n_inc
        coherence = 1.15 if (h.n_supports > 0 and h.n_refutes == 0) else 1.0
        status_penalty = 0.7 if h.status == "reserved" else 1.0

        h.confidence = max(0.0, min(1.0, (base + modifier) * coherence * status_penalty))
        h.last_updated = iteration

    # ---- Distribution Statistics (feed the emotion vector) ----

    def entropy(self) -> float:
        """Normalized entropy of the active confidence distribution."""
        if not self.active:
            return 1.0
        confs = np.array([h.confidence for h in self.active])
        total = confs.sum()
        if total < 1e-9:
            return 1.0
        probs = confs / total
        ent = -np.sum(probs * np.log(probs + 1e-9))
        max_ent = np.log(max(2, len(probs)))
        return float(ent / max_ent if max_ent > 0 else 0.0)

    def concentration(self) -> float:
        """1 - entropy. High = belief concentrated on few candidates."""
        return 1.0 - self.entropy()

    def gap(self) -> float:
        """Confidence gap between #1 and #2."""
        if len(self.active) < 2:
            return 1.0
        return max(0.0, self.active[0].confidence - self.active[1].confidence)

    def conflict_ratio(self) -> float:
        """Fraction of active entities with contradictory verdicts."""
        if not self.active:
            return 0.0
        return sum(1 for h in self.active if h.has_conflict) / len(self.active)

    def verified_ratio(self) -> float:
        """Fraction of active entities that have at least one verdict."""
        if not self.active:
            return 0.0
        return sum(1 for h in self.active if h.verdicts) / len(self.active)

    def fragility(self) -> float:
        """Estimated probability that #1 gets overtaken by #2 with one more verification.

        Based on: gap size + how many unverified dims #2 has vs #1.
        """
        if len(self.active) < 2:
            return 0.0
        h1, h2 = self.active[0], self.active[1]
        gap = h1.confidence - h2.confidence
        # #2's upside potential: unverified dims could add up to 0.10 each
        h2_upside = sum(1 for v in h2.verdicts.values()
                        if hasattr(v, 'label') and v.label == "unverified") * 0.10
        h2_upside = min(0.5, h2_upside)
        # #1's downside risk: has refutes?
        h1_downside = 0.15 if h1.n_refutes > 0 else 0.0
        effective_gap = gap - h2_upside - h1_downside
        if effective_gap > 0.3:
            return 0.0
        return float(max(0.0, min(1.0, 1.0 - effective_gap / 0.3)))

    def leader_has_refute(self) -> bool:
        if not self.active:
            return False
        return self.active[0].n_refutes > 0

    @property
    def n_active(self) -> int:
        return len(self.active)

    @property
    def n_reserved(self) -> int:
        return len(self.reserved)

    # ---- For serialization ----

    def to_dict(self) -> Dict:
        return {
            "active": [
                {"entity": h.entity, "confidence": round(h.confidence, 4),
                 "status": h.status, "n_support": h.n_supports,
                 "n_refute": h.n_refutes, "reason_family": h.reason_family}
                for h in self.active
            ],
            "reserved": [
                {"entity": h.entity, "confidence": round(h.confidence, 4)}
                for h in self.reserved
            ],
            "eliminated": [{"entity": h.entity} for h in self.eliminated],
        }

    def __repr__(self) -> str:
        return (f"HypothesisList(active={len(self.active)}, "
                f"reserved={len(self.reserved)}, eliminated={len(self.eliminated)})")
