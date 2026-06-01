"""PRISM v2.0 — Unified Active Inference RCA Framework.

This package implements five major architectural improvements over v1:
  Direction A: Unified Active Inference (active_inference/)
  Direction B: Learned Representations (learned/)
  Direction C: MCTS Action Planning (mcts/)
  Direction D: Active Perception (active_perception/)
  Direction E: Hierarchical Priors (priors/)

All modules are optional — the legacy v1 pipeline remains fully functional
when no v2 flags are enabled.
"""

from .prism import PRISMPipeline, PRISMConfig, PRISMState, PRISMAction

__version__ = "2.0.0"
