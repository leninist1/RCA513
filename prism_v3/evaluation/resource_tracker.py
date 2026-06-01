"""Resource usage tracking across queries (thread-safe)."""

import threading
import numpy as np
from typing import Dict
from dataclasses import dataclass, field

from ..config import ResourceCost


@dataclass
class PerQueryCost:
    api_calls: int = 0
    tokens: int = 0
    iterations: int = 0
    strategy_counts: Dict[str, int] = field(default_factory=dict)


class ResourceTracker:
    def __init__(self):
        self._lock = threading.Lock()
        self.per_query: Dict[str, PerQueryCost] = {}
        self.total = ResourceCost()

    def start_query(self, query_id: str):
        with self._lock:
            self.per_query[query_id] = PerQueryCost()

    def record(
        self, query_id: str, api_calls: int = 0, tokens: int = 0,
        iterations: int = 0, strategy_name: str = "",
    ):
        with self._lock:
            if query_id not in self.per_query:
                self.per_query[query_id] = PerQueryCost()

            pq = self.per_query[query_id]
            pq.api_calls += api_calls
            pq.tokens += tokens
            if iterations > 0:
                pq.iterations = iterations
            if strategy_name:
                pq.strategy_counts[strategy_name] = pq.strategy_counts.get(strategy_name, 0) + 1

            self.total.api_calls += api_calls
            self.total.tokens += tokens

    def summary(self) -> Dict:
        if not self.per_query:
            return {"total_api_calls": 0, "total_tokens": 0}

        api_calls_list = [c.api_calls for c in self.per_query.values()]
        tokens_list = [c.tokens for c in self.per_query.values()]
        iterations_list = [c.iterations for c in self.per_query.values()]

        # Aggregate strategy counts
        all_strategies = {}
        for pq in self.per_query.values():
            for sname, count in pq.strategy_counts.items():
                all_strategies[sname] = all_strategies.get(sname, 0) + count

        return {
            "total_api_calls": self.total.api_calls,
            "total_tokens": self.total.tokens,
            "avg_api_calls_per_query": round(np.mean(api_calls_list), 2),
            "avg_tokens_per_query": round(np.mean(tokens_list), 1),
            "max_tokens_per_query": max(tokens_list),
            "queries_with_zero_calls": sum(1 for c in api_calls_list if c == 0),
            "queries_with_llm": sum(1 for c in api_calls_list if c > 0),
            "avg_iterations": round(np.mean(iterations_list), 1),
            "strategy_distribution": all_strategies,
        }
