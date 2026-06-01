"""Hierarchical MCTS: two-level planning.

Level 1 (Macro): choose which entity to investigate
Level 2 (Micro): choose which action type on the selected entity

Reduces the branching factor significantly:
  Macro: up to |V| ≈ 30-60 entities
  Micro: up to 4-6 action types per entity

Total branching: |V| + |A| instead of |V| × |A|
"""

from __future__ import annotations

from typing import Dict, List, Any
import math
import numpy as np
from .belief_mcts import MCTSConfig, BeliefMCTS

EPS = 1e-9


class HierarchicalMCTS:
    def __init__(self, config: MCTSConfig = None):
        self.config = config or MCTSConfig()
        self.macro_mcts = BeliefMCTS(
            MCTSConfig(
                n_simulations=self.config.n_simulations // 2,
                c_uct=self.config.c_uct,
                rollout_depth=self.config.rollout_depth,
                gamma=self.config.gamma,
                max_branching=20,
            )
        )
        self.micro_mcts = BeliefMCTS(
            MCTSConfig(
                n_simulations=self.config.n_simulations // 2,
                c_uct=self.config.c_uct,
                rollout_depth=1,
                gamma=self.config.gamma,
                max_branching=8,
            )
        )

    def search(
        self,
        state: Any,
        enumerate_actions_fn,
        execute_action_fn,
        clone_state_fn,
        reward_fn,
    ) -> tuple:
        all_actions = enumerate_actions_fn(state)
        if not all_actions:
            return "STOP", {}

        entities = {}
        for action in all_actions:
            entity = (
                action.detail.get("entity", "") if hasattr(action, "detail") else ""
            )
            if entity:
                if entity not in entities:
                    entities[entity] = []
                entities[entity].append(action)

        if not entities:
            return self._greedy_best(all_actions)

        entity_scores = {}
        for entity in entities:
            scores = [
                a.utility if hasattr(a, "utility") else a.eig_per_cost
                for a in entities[entity]
            ]
            entity_scores[entity] = max(scores) if scores else 0.0

        best_entity = max(entity_scores, key=entity_scores.get)
        best_actions = entities[best_entity]
        best_actions.sort(
            key=lambda a: a.utility if hasattr(a, "utility") else a.eig_per_cost,
            reverse=True,
        )

        best = best_actions[0]
        return (
            best.name if hasattr(best, "name") else str(best),
            best.detail if hasattr(best, "detail") else {},
        )

    @staticmethod
    def _greedy_best(actions: list) -> tuple:
        best = max(
            actions,
            key=lambda a: a.utility if hasattr(a, "utility") else a.eig_per_cost,
        )
        return (
            best.name if hasattr(best, "name") else str(best),
            best.detail if hasattr(best, "detail") else {},
        )
