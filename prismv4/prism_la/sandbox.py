"""Evidence sandbox with strict leakage guard for PRISM-LA.

Ensures the LLM agent can only access public case evidence and tool results.
GT labels, scoring points, record.csv, and historical results are blocked.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple
import re

from .config import FORBIDDEN_LLM_INPUTS


class SandboxViolation(RuntimeError):
    """Raised when forbidden data leaks into the sandbox."""


@dataclass
class EvidenceSandbox:
    query_text: str = ""
    metrics_summary: Dict[str, Any] = field(default_factory=dict)
    logs_summary: Dict[str, Any] = field(default_factory=dict)
    traces_summary: Dict[str, Any] = field(default_factory=dict)
    entities: List[str] = field(default_factory=list)
    entity_types: Dict[str, str] = field(default_factory=dict)
    topology_summary: Dict[str, Any] = field(default_factory=dict)
    time_window: Tuple[str, str] = ("", "")
    instruction: str = ""
    tool_results: List[Dict[str, Any]] = field(default_factory=list)
    forbidden_keys_checked: bool = False

    def add_tool_result(self, result: Dict[str, Any]) -> None:
        sanitized = self._sanitize(result)
        self.tool_results.append(sanitized)

    def check_no_leakage(self, data: Dict[str, Any]) -> None:
        self._recursive_check(data, path="root")
        self.forbidden_keys_checked = True

    def _sanitize(self, data: Dict[str, Any]) -> Dict[str, Any]:
        def walk(obj: Any, path: str = "") -> Any:
            if isinstance(obj, dict):
                cleaned = {}
                for key, value in obj.items():
                    if self._key_is_forbidden(key):
                        raise SandboxViolation(
                            f"forbidden key '{key}' in sandbox at {path}"
                        )
                    cleaned[key] = walk(value, f"{path}.{key}")
                return cleaned
            if isinstance(obj, list):
                return [walk(item, f"{path}[{i}]") for i, item in enumerate(obj)]
            return obj
        return walk(dict(data), "data")

    def _recursive_check(self, obj: Any, path: str = "") -> None:
        if isinstance(obj, dict):
            for key, value in obj.items():
                if self._key_is_forbidden(key):
                    raise SandboxViolation(
                        f"forbidden key '{key}' found in sandbox at {path}"
                    )
                self._recursive_check(value, f"{path}.{key}")
        elif isinstance(obj, (list, tuple)):
            for idx, item in enumerate(obj):
                self._recursive_check(item, f"{path}[{idx}]")
        elif isinstance(obj, str):
            for forbidden in FORBIDDEN_LLM_INPUTS:
                if forbidden.lower() in obj.lower() and len(forbidden) >= 4:
                    raise SandboxViolation(
                        f"forbidden string '{forbidden}' found in sandbox at {path}"
                    )

    @staticmethod
    def _key_is_forbidden(key: str) -> bool:
        key_lower = key.lower().replace("_", "").replace("-", "").replace(" ", "")
        for forbidden in FORBIDDEN_LLM_INPUTS:
            normalized = forbidden.lower().replace("_", "").replace("-", "").replace(" ", "")
            if key_lower == normalized:
                return True
        return False

    def to_llm_input(self) -> Dict[str, Any]:
        return {
            "query_text": self.query_text,
            "instruction": self.instruction,
            "time_window": list(self.time_window),
            "system_entities": {
                "count": len(self.entities),
                "entity_list": self.entities[:50],
                "entity_types": self.entity_types,
            },
            "modality_summary": {
                "metrics": self.metrics_summary,
                "logs": self.logs_summary,
                "traces": self.traces_summary,
            },
            "topology_summary": self.topology_summary,
            "tool_results": [r for r in self.tool_results[-40:]],
        }

    def build_public_view(
        self,
        query: Any,
        entities: Sequence[str],
        entity_types: Dict[str, str],
        topology_edges: Sequence[Tuple[str, str, float]],
        telemetry_meta: Dict[str, Any],
    ) -> EvidenceSandbox:
        self.check_no_leakage(telemetry_meta)
        self.query_text = str(getattr(query, "instruction", "") or "")
        self.instruction = self.query_text
        time_window = getattr(query, "time_window", ("", ""))
        self.time_window = (str(time_window[0]), str(time_window[1])) if len(time_window) == 2 else ("", "")
        self.entities = [str(e) for e in entities]
        self.entity_types = {str(k): str(v) for k, v in entity_types.items()}
        edge_list = [
            {"source": str(src), "target": str(dst), "weight": round(float(w), 4)}
            for src, dst, w in topology_edges
        ]
        self.topology_summary = {
            "edge_count": len(edge_list),
            "top_edges_by_weight": sorted(edge_list, key=lambda e: e["weight"], reverse=True)[:30],
        }
        self.metrics_summary = {
            "available": bool(telemetry_meta.get("has_metrics", True)),
            "entity_count": telemetry_meta.get("metric_entity_count", len(entities)),
            "column_count": telemetry_meta.get("metric_column_count", 0),
        }
        self.logs_summary = {
            "available": bool(telemetry_meta.get("has_logs", False)),
            "entity_count": telemetry_meta.get("log_entity_count", 0),
            "error_keyword_count": telemetry_meta.get("error_keyword_count", 0),
        }
        self.traces_summary = {
            "available": bool(telemetry_meta.get("has_traces", False)),
            "span_count": telemetry_meta.get("trace_span_count", 0),
            "entity_count": telemetry_meta.get("trace_entity_count", 0),
        }
        return self


def assert_no_answer_leakage(query: Any) -> None:
    violations = []
    if getattr(query, "ground_truth", None) is not None:
        violations.append("ground_truth")
    if list(getattr(query, "scoring_points", ()) or ()):
        violations.append("scoring_points")
    if list(getattr(query, "record_ids", ()) or ()):
        violations.append("record_ids")
    if violations:
        raise SandboxViolation(
            "inference query contains evaluation-only fields: " + ", ".join(violations)
        )
