"""Replay artifacts for PRISM v3 inference debugging."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping
import json


@dataclass
class ReplayEvent:
    step: int
    kind: str
    payload: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReplayLog:
    query_id: str
    system: str
    created_at: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    events: List[ReplayEvent] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add(self, kind: str, payload: Mapping[str, Any], *, step: int | None = None) -> None:
        self.events.append(
            ReplayEvent(
                step=len(self.events) if step is None else int(step),
                kind=str(kind),
                payload=dict(payload),
            )
        )

    def to_mapping(self) -> Dict[str, Any]:
        return {
            "query_id": self.query_id,
            "system": self.system,
            "created_at": self.created_at,
            "metadata": dict(self.metadata),
            "events": [
                {"step": event.step, "kind": event.kind, "payload": dict(event.payload)}
                for event in self.events
            ],
        }

    def write(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.to_mapping(), ensure_ascii=True, indent=2), encoding="utf-8")
        return output


def replay_from_result(result: Mapping[str, Any]) -> ReplayLog:
    query_id = str(result.get("query_id", ""))
    system = str(result.get("system", ""))
    log = ReplayLog(query_id=query_id, system=system, metadata={"source": "prism_result"})
    for key in ("runtime_debug", "debug", "prediction", "top_candidates", "belief_top20"):
        if key in result:
            log.add(key, {"value": result.get(key)})
    return log
