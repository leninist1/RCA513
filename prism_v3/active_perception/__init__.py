"""Active perception module (Direction D).

Extends the action space beyond pre-computed signal queries to include:
  - METRIC_RESCAN: Hypothesis-driven metric re-analysis with adaptive thresholds
  - LOG_HYPOTHESIS_SEARCH: Hypothesis-guided log pattern search
  - TRACE_SUBGRAPH: Focused k-hop subgraph extraction
  - HYPOTHESIS_CROSSVAL: Comparative analysis of two competing hypotheses
  - ADAPTIVE_WINDOW: Adaptive temporal window search

These actions allow the system to actively re-process raw data rather than
only query pre-calculated anomaly scores.
"""

from .metric_rescan import MetricRescanAction
from .log_search import LogHypothesisSearch
from .trace_subgraph import TraceSubgraphExtractor
from .hypothesis_crossval import HypothesisCrossval

__all__ = [
    "MetricRescanAction",
    "LogHypothesisSearch",
    "TraceSubgraphExtractor",
    "HypothesisCrossval",
]
