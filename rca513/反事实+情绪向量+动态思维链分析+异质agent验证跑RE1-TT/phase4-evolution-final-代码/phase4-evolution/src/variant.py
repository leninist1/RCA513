from dataclasses import dataclass, field
from typing import Dict, Optional

@dataclass
class Variant:
    variant_id: str
    parent_version: str
    target_operator: str          # "1","1.5","2"...
    modification: Dict            # 变体具体改动描述
    proposer_agent: str
    proposer_internal_state: Dict
    metrics: Dict = field(default_factory=dict)

    @classmethod
    def from_proposal(cls, proposal: dict, agent_id: str, state: dict) -> 'Variant':
        return cls(
            variant_id=proposal["variant_id"],
            parent_version=proposal.get("parent_version", "V7.0"),
            target_operator=proposal["target_operator"],
            modification=proposal["modification"],
            proposer_agent=agent_id,
            proposer_internal_state=state.copy()
        )