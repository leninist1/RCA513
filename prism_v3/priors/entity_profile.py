"""Entity Profile Store: accumulate historical fault statistics per entity.

Maintains per-entity counts of fault sub-categories over successful RCA cases.
Used by HierarchicalPrior to boost entities that historically match the current
suspected fault type.
"""

from __future__ import annotations

from typing import Dict, List, Optional
import json
import os


DEFAULT_PROFILE_PATH = "/home/dell2/RCA513/yyx/prism_v2/priors/entity_profiles.json"


class EntityProfileStore:
    """Persistent store of per-entity fault sub-category occurrence counts."""

    def __init__(self, profile_path: Optional[str] = None):
        self.profile_path = profile_path or DEFAULT_PROFILE_PATH
        self._profiles: Dict[str, Dict[str, int]] = {}
        self._load()

    def _load(self):
        if os.path.exists(self.profile_path):
            try:
                with open(self.profile_path, "r") as f:
                    self._profiles = json.load(f)
            except (json.JSONDecodeError, IOError):
                self._profiles = {}

    def save(self):
        os.makedirs(os.path.dirname(self.profile_path), exist_ok=True)
        with open(self.profile_path, "w") as f:
            json.dump(self._profiles, f, indent=2)

    def get_profile(self, entity: str) -> Dict[str, int]:
        return self._profiles.get(entity, {})

    def get_all_profiles(self) -> Dict[str, Dict[str, int]]:
        return dict(self._profiles)

    def record_case(
        self,
        root_entity: str,
        fault_subcategory: str,
        confidence: float = 1.0,
    ):
        weight = max(0.5, confidence)
        if root_entity not in self._profiles:
            self._profiles[root_entity] = {}
        self._profiles[root_entity][fault_subcategory] = (
            self._profiles[root_entity].get(fault_subcategory, 0) + weight
        )

    def record_batch(self, cases: List[Dict]):
        for case in cases:
            self.record_case(
                root_entity=case.get("root_entity", ""),
                fault_subcategory=case.get("fault_subcategory", "unknown"),
                confidence=case.get("confidence", 1.0),
            )

    def top_entities_for_subcategory(
        self,
        subcategory: str,
        top_k: int = 5,
    ) -> List[Dict[str, any]]:
        scored = []
        for entity, counts in self._profiles.items():
            count = counts.get(subcategory, 0)
            if count > 0:
                total = sum(counts.values())
                ratio = count / max(1, total)
                scored.append(
                    {"entity": entity, "count": count, "ratio": round(ratio, 3)}
                )
        scored.sort(key=lambda x: x["count"], reverse=True)
        return scored[:top_k]

    def total_records(self) -> int:
        return sum(sum(counts.values()) for counts in self._profiles.values())

    def entity_count(self) -> int:
        return len(self._profiles)

    def merge_profiles(self, other: "EntityProfileStore"):
        for entity, counts in other.get_all_profiles().items():
            if entity not in self._profiles:
                self._profiles[entity] = {}
            for subcat, count in counts.items():
                self._profiles[entity][subcat] = (
                    self._profiles[entity].get(subcat, 0) + count
                )
