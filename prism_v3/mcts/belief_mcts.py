"""Monte Carlo Tree Search for planning in PRISM belief space.

Key integration: uses PRISMState._clone_state() and the existing action
enumeration/execution infrastructure.  MCTS searches over the discrete
action space with the cloned state for simulation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import math
import random
import numpy as np


EPS = 1e-9


@dataclass
class MCTSConfig:
    n_simulations: int = 50
    c_uct: float = 1.4
    rollout_depth: int = 3
    gamma: float = 0.95
    max_branching: int = 12
    deterministic_rollout: bool = False


@dataclass
class MCTSNode:
    state_id: int = 0
    parent: Optional["MCTSNode"] = None
    action_name: str = ""
    action_detail: Dict = field(default_factory=dict)
    _action_ref: Any = None
    visits: int = 0
    total_reward: float = 0.0
    q_value: float = 0.0
    children: List["MCTSNode"] = field(default_factory=list)
    is_expanded: bool = False


class BeliefMCTS:
    """MCTS over the PRISM belief space for action planning."""

    def __init__(
        self,
        config: Optional[MCTSConfig] = None,
        random_seed: Optional[int] = None,
    ):
        self.config = config or MCTSConfig()
        if random_seed is not None:
            random.seed(random_seed)

    def search(
        self,
        pipeline: Any,
        state: Any,
        telemetry: Any,
        baseline_df: Any,
        fault_df: Any,
        trace_edges: Any,
        entity_types: Any,
    ) -> Tuple[str, Dict]:
        self._pipeline = pipeline
        self._telemetry = telemetry
        self._baseline_df = baseline_df
        self._fault_df = fault_df
        self._trace_edges = trace_edges
        self._entity_types = entity_types

        self._state_cache: Dict[int, Any] = {0: state}
        root = MCTSNode(state_id=0)

        for _ in range(self.config.n_simulations):
            node = self._select(root)
            if self._is_terminal(node):
                self._backpropagate(node, self._terminal_reward(node))
                continue
            if not node.is_expanded and node.visits > 0:
                node = self._expand(node)
            if node.is_expanded and node.children:
                reward = self._simulate(node)
            else:
                reward = 0.0
            self._backpropagate(node, reward)

        if not root.children:
            return "STOP", {}
        best = max(root.children, key=lambda c: c.visits)
        return best.action_name, best.action_detail

    def _select(self, node: MCTSNode) -> MCTSNode:
        while node.is_expanded and node.children:
            best_child = None
            best_score = -float("inf")
            for child in node.children:
                if child.visits == 0:
                    best_child = child
                    break
                uct = child.q_value + self.config.c_uct * math.sqrt(
                    math.log(max(node.visits, 1)) / max(child.visits, 1)
                )
                if uct > best_score:
                    best_score = uct
                    best_child = child
            if best_child is None:
                break
            node = best_child
        return node

    def _expand(self, node: MCTSNode) -> MCTSNode:
        parent_state = self._state_cache.get(node.state_id)
        if parent_state is None:
            node.is_expanded = True
            return node

        actions = self._pipeline._enumerate_actions(
            parent_state,
            self._telemetry,
            self._baseline_df,
            self._fault_df,
            self._trace_edges,
        )
        if not actions:
            node.is_expanded = True
            return node

        for action in actions[: self.config.max_branching]:
            child_state = self._pipeline._clone_state(parent_state)
            self._pipeline._execute_action(
                child_state,
                action,
                self._telemetry,
                self._baseline_df,
                self._fault_df,
                self._trace_edges,
                self._entity_types,
            )
            child = MCTSNode(
                parent=node,
                action_name=action.name if hasattr(action, "name") else str(action),
                action_detail=action.detail if hasattr(action, "detail") else {},
                _action_ref=action,
            )
            child.state_id = len(self._state_cache)
            self._state_cache[child.state_id] = child_state
            node.children.append(child)

        node.is_expanded = True
        return random.choice(node.children) if node.children else node

    def _simulate(self, node: MCTSNode) -> float:
        current_state = self._state_cache.get(node.state_id)
        if current_state is None:
            return 0.0

        state = self._pipeline._clone_state(current_state)
        total_reward = 0.0

        for d in range(self.config.rollout_depth):
            actions = self._pipeline._enumerate_actions(
                state,
                self._telemetry,
                self._baseline_df,
                self._fault_df,
                self._trace_edges,
            )
            if not actions:
                break
            if self.config.deterministic_rollout:
                action = actions[0]
            else:
                weights = np.array(
                    [
                        max(
                            0.01, a.utility if hasattr(a, "utility") else a.eig_per_cost
                        )
                        for a in actions
                    ]
                )
                weights = weights / (weights.sum() + EPS)
                idx = np.random.choice(len(actions), p=weights)
                action = actions[idx]
            self._pipeline._execute_action(
                state,
                action,
                self._telemetry,
                self._baseline_df,
                self._fault_df,
                self._trace_edges,
                self._entity_types,
            )
            total_reward += float(np.max(state.p)) * (self.config.gamma**d)
        return total_reward

    def _backpropagate(self, node: MCTSNode, reward: float):
        while node is not None:
            node.visits += 1
            node.total_reward += reward
            node.q_value = node.total_reward / max(node.visits, 1)
            node = node.parent

    def _is_terminal(self, node: MCTSNode) -> bool:
        return node.action_name == "STOP" or node.visits > 100

    def _terminal_reward(self, node: MCTSNode) -> float:
        return 1.0 if node.action_name == "STOP" else 0.0
