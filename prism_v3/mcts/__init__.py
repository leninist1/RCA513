"""Monte Carlo Tree Search for action planning (Direction C).

Replaces the greedy one-step EIG/c max selection with multi-step look-ahead
planning using MCTS in belief space.

Core algorithm:
  For n_simulations:
    1. Select: traverse tree via UCT = Q + c*sqrt(ln N_parent / N_child)
    2. Expand: add child nodes for all legal actions at the selected leaf
    3. Simulate: fast rollout using default policy (greedy EIG/c)
    4. Backpropagate: update Q-values and visit counts up the tree

Returns the action with the highest visit count at the root.
"""

from .belief_mcts import BeliefMCTS, MCTSNode, MCTSConfig
from .hierarchical_mcts import HierarchicalMCTS

__all__ = [
    "BeliefMCTS",
    "MCTSNode",
    "MCTSConfig",
    "HierarchicalMCTS",
]
