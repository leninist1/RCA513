"""Evaluation metrics aggregation."""

from collections import defaultdict
from typing import Dict, List

from ..config import EvalResult


class EvalAggregator:
    def __init__(self):
        self.results: List[EvalResult] = []

    def add(self, result: EvalResult):
        self.results.append(result)

    def summary(self) -> Dict:
        total = len(self.results)
        if total == 0:
            return {"total": 0, "correct": 0, "partial": 0}

        correct = sum(1 for r in self.results if r.correct)
        partial = sum(1 for r in self.results if r.partial) - correct  # partial but NOT correct

        # Per-system
        by_system = defaultdict(list)
        for r in self.results:
            by_system[r.system].append(r)

        # Per-task
        by_task = defaultdict(list)
        for r in self.results:
            by_task[r.task_type].append(r)

        # Per-field
        field_correct = defaultdict(lambda: {"correct": 0, "total": 0})
        for r in self.results:
            for field, ok in r.field_scores.items():
                field_correct[field]["total"] += 1
                if ok:
                    field_correct[field]["correct"] += 1

        return {
            "total": total,
            "correct": correct,
            "correct_pct": round(correct / total * 100, 2),
            "partial": partial,
            "partial_pct": round(partial / total * 100, 2),
            "correct_or_partial_pct": round((correct + partial) / total * 100, 2),
            "neither": total - correct - partial,
            "by_system": {
                sys: {
                    "total": len(v),
                    "correct": sum(1 for r in v if r.correct),
                    "correct_pct": round(sum(1 for r in v if r.correct) / len(v) * 100, 2),
                    "partial": sum(1 for r in v if r.partial and not r.correct),
                    "partial_pct": round(sum(1 for r in v if r.partial and not r.correct) / len(v) * 100, 2),
                }
                for sys, v in sorted(by_system.items())
            },
            "by_task": {
                task: {
                    "total": len(v),
                    "correct": sum(1 for r in v if r.correct),
                    "correct_pct": round(sum(1 for r in v if r.correct) / len(v) * 100, 2),
                }
                for task, v in sorted(by_task.items())
            },
            "by_field": {
                field: {
                    "correct": stats["correct"],
                    "total": stats["total"],
                    "rate": round(stats["correct"] / stats["total"] * 100, 2) if stats["total"] > 0 else 0,
                }
                for field, stats in sorted(field_correct.items())
            },
        }
