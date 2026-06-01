"""Learned fault type classifier.

Replaces the keyword-matching _infer_reason() at prism.py:3800-3861
with a trained classifier that takes aggregated anomaly evidence and
predicts the most likely fault sub-category.

Architecture: Simple 3-layer MLP
  Input:  aggregated metric evidence per fault family (9 x 3 stats)
  Hidden: 32 → 16
  Output: 9-class softmax (fault sub-categories)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np

from ..priors.hierarchical_prior import FaultSubCategory, SUBCAT_TO_KEYWORDS


SUBCAT_ORDER: List[FaultSubCategory] = [
    FaultSubCategory.CPU,
    FaultSubCategory.MEMORY,
    FaultSubCategory.DISK_IO,
    FaultSubCategory.DISK_SPACE,
    FaultSubCategory.NETWORK_LATENCY,
    FaultSubCategory.NETWORK_PACKET_LOSS,
    FaultSubCategory.NETWORK_FAULT,
    FaultSubCategory.JVM_OOM,
    FaultSubCategory.PROCESS_KILL,
    FaultSubCategory.HIGH_MEMORY,
    FaultSubCategory.DB_FAULT,
    FaultSubCategory.UNKNOWN,
]


class FaultClassifier:
    """Learned fault type classifier.

    Supports two pretrained formats:
      - MLP weights (legacy): {"params": {...}}
      - Dict-based (new):   {"reason_map": {...}, "subcat_weights": {...},
                              "keyword_counts": {...}}
    Fallback: keyword matching from hierarchical priors when untrained.
    """

    def __init__(self, load_pretrained: Optional[str] = None):
        self._is_trained = False
        self._W: Optional[np.ndarray] = None
        self._reason_map: Optional[Dict[str, str]] = None
        self._keyword_counts: Dict[str, Dict[str, int]] = {}
        self._subcat_weights: Optional[Dict[str, float]] = None
        if load_pretrained:
            self._load(load_pretrained)

    def _extract_evidence_vector(
        self,
        metric_detail: Dict[str, Dict[str, float]],
        log_detail: Dict[str, Dict[str, any]],
        entities: List[str],
    ) -> np.ndarray:
        n_subcats = len(SUBCAT_ORDER)
        evidence_vec = np.zeros(n_subcats * 3, dtype=float)

        for j, sc in enumerate(SUBCAT_ORDER):
            keywords = SUBCAT_TO_KEYWORDS.get(sc, [])
            if not keywords:
                continue

            metric_hits = 0
            metric_max_z = 0.0
            all_names = []
            for entity in entities:
                all_names.extend(metric_detail.get(entity, {}).keys())
            all_text = " ".join(all_names).lower()
            for kw in keywords:
                if kw in all_text:
                    for entity in entities:
                        for metric_name, z_score in metric_detail.get(
                            entity, {}
                        ).items():
                            if kw in metric_name.lower():
                                metric_hits += 1
                                metric_max_z = max(metric_max_z, z_score)
            evidence_vec[j * 3] = metric_max_z
            evidence_vec[j * 3 + 1] = metric_hits / max(1, len(keywords))
            evidence_vec[j * 3 + 2] = min(1.0, metric_hits / max(1, len(entities)))

        return evidence_vec

    def predict(
        self,
        metric_detail: Dict[str, Dict[str, float]],
        log_detail: Dict[str, Dict[str, any]],
        entities: List[str],
    ) -> Dict[FaultSubCategory, float]:
        evidence = self._extract_evidence_vector(metric_detail, log_detail, entities)

        if self._is_trained and self._W is not None:
            return self._trained_predict(evidence)
        return self._keyword_predict(evidence)

    def _trained_predict(self, evidence: np.ndarray) -> Dict[FaultSubCategory, float]:
        if self._W is not None:
            W1, W2, W3 = self._W["W1"], self._W["W2"], self._W["W3"]
            b1, b2, b3 = self._W["b1"], self._W["b2"], self._W["b3"]
            h1 = np.tanh(W1 @ evidence + b1)
            h2 = np.tanh(W2 @ h1 + b2)
            logits = W3 @ h2 + b3
            logits = logits - np.max(logits)
            probs = np.exp(logits) / (np.sum(np.exp(logits)) + 1e-9)
            return {
                SUBCAT_ORDER[i]: float(probs[i])
                for i in range(len(SUBCAT_ORDER))
                if float(probs[i]) > 0.01
            }

        if self._subcat_weights is not None and self._subcat_weights:
            results: Dict[FaultSubCategory, float] = {}
            n_subcats = len(self._subcat_weights)
            for j, sc in enumerate(SUBCAT_ORDER):
                sc_key = sc.value
                evidence_score = (
                    float(evidence[j * 3]) / 10.0
                    + float(evidence[j * 3 + 1])
                    + float(evidence[j * 3 + 2])
                ) / 3.0
                prior = self._subcat_weights.get(sc_key, 0.0)
                if prior <= 0.0:
                    continue
                combined = evidence_score * prior * max(1.0, n_subcats)
                if combined > 0.0:
                    results[sc] = combined
            if not results:
                return {FaultSubCategory.UNKNOWN: 1.0}
            total = sum(results.values())
            return {k: v / total for k, v in results.items()}

        return self._keyword_predict(evidence)

    def _keyword_predict(self, evidence: np.ndarray) -> Dict[FaultSubCategory, float]:
        results = {}
        n_subcats = len(SUBCAT_ORDER)
        for j, sc in enumerate(SUBCAT_ORDER):
            if n_subcats > 0:
                score = (
                    float(evidence[j * 3]) / 10.0
                    + float(evidence[j * 3 + 1])
                    + float(evidence[j * 3 + 2])
                ) / 3.0
            else:
                score = 0.0
            if score > 0.0:
                results[sc] = score
        if not results:
            results[FaultSubCategory.UNKNOWN] = 1.0
            return results
        total = sum(results.values())
        return {k: v / total for k, v in results.items()}

    def top_prediction(
        self,
        metric_detail: Dict[str, Dict[str, float]],
        log_detail: Dict[str, Dict[str, any]],
        entities: List[str],
    ) -> Tuple[FaultSubCategory, float]:
        probs = self.predict(metric_detail, log_detail, entities)
        if not probs:
            return FaultSubCategory.UNKNOWN, 0.0
        best = max(probs, key=probs.get)
        return best, probs[best]

    def _load(self, path: str):
        import pickle as _pickle

        try:
            with open(path, "rb") as f:
                data = _pickle.load(f)
            if isinstance(data, dict) and "reason_map" in data:
                self._reason_map = data["reason_map"]
                self._keyword_counts = data.get("keyword_counts", {})
                self._subcat_weights = data.get("subcat_weights", {})
                self._is_trained = bool(self._reason_map)
            else:
                self._W = data.get("params", None) if isinstance(data, dict) else None
                self._is_trained = self._W is not None
        except (FileNotFoundError, Exception):
            self._is_trained = False

    def save(self, path: str):
        import pickle as _pickle

        if self._reason_map is not None:
            data = {
                "reason_map": self._reason_map,
                "subcat_weights": self._subcat_weights,
                "keyword_counts": self._keyword_counts,
            }
        else:
            data = {"params": self._W}
        with open(path, "wb") as f:
            _pickle.dump(data, f)
