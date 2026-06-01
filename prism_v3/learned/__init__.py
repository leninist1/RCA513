"""Learned representations (Direction B).

Replaces hand-crafted features with learned embeddings, a learned
observation likelihood network, and a learned fault classifier.

Modules:
  - embeddings: Entity embeddings trained via contrastive learning
  - likelihood: Learned observation likelihood (μ, σ per entity/case)
  - classifier: Learned 9-class fault type classifier
  - anomaly_detector: Per-entity adaptive anomaly detection
"""

from .embeddings import EntityEmbedding, EntityEmbedder
from .likelihood import LikelihoodNetwork
from .classifier import FaultClassifier
from .anomaly_detector import PerEntityAnomalyDetector

__all__ = [
    "EntityEmbedding",
    "EntityEmbedder",
    "LikelihoodNetwork",
    "FaultClassifier",
    "PerEntityAnomalyDetector",
]
