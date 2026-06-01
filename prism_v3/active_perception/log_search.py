"""Hypothesis-guided log search (Active Perception D2).

Unlike LOG_QUERY which just returns pre-computed fatal_boost,
this searches raw log messages with hypothesis-specific regex patterns.

Example hypotheses and their search patterns:
  "JVM OOM" → OutOfMemory, heap, GC, allocation failure
  "CPU fault" → throttl, cpu, load, utilization
  "network latency" → timeout, slow, delay, refused

Cost: ~0.8s
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import math
import re
import numpy as np


class LogHypothesisSearch:
    """Hypothesis-driven log message search."""

    _PATTERNS = {
        "JVM Out of Memory": [
            r"(?i)outofmemory|oom\s+killed|heap\sspace|gc\s+overhead|allocation\sfailure|java\.lang\.OutOfMemory",
        ],
        "CPU fault": [
            r"(?i)cpu\s+throttl|cfs\s+throttled|cpu\s+utilization|high\s+cpu|cpu\s+load",
        ],
        "high memory usage": [
            r"(?i)memory\s+usage|high\s+memory|rss|mem\s+limit|container\s+memory|swap",
        ],
        "network latency": [
            r"(?i)timeout|slow\s+response|high\s+latency|connection\s+timed?\s+out",
        ],
        "network fault": [
            r"(?i)connection\s+refused|connection\s+reset|unreachable|network\s+error",
        ],
        "network packet loss": [
            r"(?i)packet\s+loss|retransmit|dropped\s+packet",
        ],
        "disk I/O consumption": [
            r"(?i)disk\s+io|high\s+io|io\s+bottleneck|iowait|disk\s+slow",
        ],
        "db fault": [
            r"(?i)database\s+error|db\s+connection|jdbc\s+exception|sql\s+exception|connection\s+pool\s+exhausted",
        ],
        "process termination": [
            r"(?i)sigkill|killed|process\s+exit|container\s+killed|crash",
        ],
    }

    def __init__(self):
        self._compiled_patterns: Dict[str, List[re.Pattern]] = {}
        for reason, pats in self._PATTERNS.items():
            self._compiled_patterns[reason] = [re.compile(p) for p in pats]

    def search(
        self,
        entity: str,
        log_messages: List[str],
        hypothesis_reasons: List[str],
    ) -> Dict:
        """Search log messages for hypothesis-specific patterns.

        Args:
            entity: Entity name
            log_messages: Raw log message strings for the entity
            hypothesis_reasons: List of hypothesis reason labels
                                (e.g., ["JVM Out of Memory", "CPU fault"])

        Returns:
            {
                "match_count": int,
                "match_rate": float,
                "matched_messages": List[str],
                "evidence_score": float (0-1),
                "primary_hypothesis": str,
            }
        """
        results = {}
        best_reason = ""
        best_score = 0.0

        for reason in hypothesis_reasons:
            patterns = self._compiled_patterns.get(reason, [])
            if not patterns:
                continue
            matched = []
            for msg in log_messages:
                if any(p.search(msg) for p in patterns):
                    matched.append(msg[:200])
            score = len(matched) / max(1, len(log_messages))
            results[reason] = {
                "count": len(matched),
                "match_rate": round(score, 4),
                "matched": matched[:5],
            }
            if score > best_score:
                best_score = score
                best_reason = reason

        total_matches = sum(r["count"] for r in results.values())
        all_matched = []
        seen = set()
        for reason, info in results.items():
            for msg in info["matched"]:
                if msg not in seen:
                    all_matched.append(msg)
                    seen.add(msg)

        return {
            "match_count": total_matches,
            "match_rate": round(total_matches / max(1, len(log_messages)), 4),
            "matched_messages": all_matched[:10],
            "evidence_score": min(1.0, total_matches / max(1, len(log_messages)) * 3.0),
            "primary_hypothesis": best_reason if best_score > 0 else "",
            "per_hypothesis": results,
        }

    def quick_search(
        self,
        entity: str,
        log_messages: List[str],
    ) -> Dict[str, int]:
        """Quick scan of all hypothesis categories, return hit counts."""
        counts = {}
        for reason, patterns in self._compiled_patterns.items():
            hits = 0
            for msg in log_messages:
                if any(p.search(msg) for p in patterns):
                    hits += 1
            if hits > 0:
                counts[reason] = hits
        return counts
