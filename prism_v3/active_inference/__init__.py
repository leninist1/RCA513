"""Unified Active Inference framework (Direction A).

Replaces the 6-layer decoupled pipeline with a single variational objective:
  EFE(d) = Epistemic Value(d) + Pragmatic Value(d)

All components (perception, belief update, graph learning, action selection)
derive from minimizing Expected Free Energy.

Modules:
  - generative_model: Joint distribution p(obs, r, W)
  - variational: Mean-field variational inference (E-step, M-step)
  - free_energy: Expected Free Energy computation and action scoring
  - meta_controller: EFE-based action selection and stopping
"""

from .generative_model import GenerativeModel, PropagationModel
from .variational import VariationalInference, MeanFieldState
from .free_energy import ExpectedFreeEnergyComputer
from .meta_controller import ActiveInferenceController

__all__ = [
    "GenerativeModel",
    "PropagationModel",
    "VariationalInference",
    "MeanFieldState",
    "ExpectedFreeEnergyComputer",
    "ActiveInferenceController",
]
