"""Hierarchical prior structure (Direction E).

Three-level Bayesian prior:
  Level 0: System Prototype Prior (Bank-like, Telecom-like, Market-like)
  Level 1: Fault Category Prior (Resource, Network, Application, Database)
  Level 2: Fault Sub-Category Prior (CPU, Memory, JVM OOM, etc.)
  Level 3: Entity Profile Prior (historical failure patterns per entity)
"""

from .system_prototype import SystemPrototype, SYSTEM_PROTOTYPES
from .hierarchical_prior import HierarchicalPrior, FaultCategory, FaultSubCategory
from .entity_profile import EntityProfileStore
from .graph_prior import LayerTransitionsConditional, LAYER_TRANSITIONS_CONDITIONAL

__all__ = [
    "SystemPrototype",
    "SYSTEM_PROTOTYPES",
    "HierarchicalPrior",
    "FaultCategory",
    "FaultSubCategory",
    "EntityProfileStore",
    "LayerTransitionsConditional",
    "LAYER_TRANSITIONS_CONDITIONAL",
]
