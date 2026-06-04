"""Evidence signature nearest-neighbor and lightweight clustering."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping


def flatten_signature(signature: Mapping) -> dict[str, float]:
    """Convert a case signature dict to a sparse weighted feature map."""
    features: dict[str, float] = {}
    for service in signature.get("services", []):
        svc = service.get("service", "")
        for modality in ("metric", "log", "trace", "topology"):
            for name, item in service.get(modality, {}).items():
                state = item.get("state", "none")
                intensity = float(item.get("intensity", 0.0) or 0.0)
                key = f"{modality}:{name}:{state}"
                features[key] = features.get(key, 0.0) + max(1.0, intensity)
                features[f"svc:{svc}:{modality}:{name}:{state}"] = max(1.0, intensity)
    for blind in signature.get("blind_spots", []):
        features[f"blind:{blind}"] = 1.0
    for kind in signature.get("dominant_evidence_types", []):
        features[f"dominant:{kind}"] = 1.0
    return features


def cosine_sparse(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    if not left or not right:
        return 0.0
    common = set(left) & set(right)
    dot = sum(float(left[k]) * float(right[k]) for k in common)
    nl = sum(float(v) * float(v) for v in left.values()) ** 0.5
    nr = sum(float(v) * float(v) for v in right.values()) ** 0.5
    return dot / (nl * nr) if nl and nr else 0.0


@dataclass(frozen=True)
class SimilarCase:
    case_id: str
    similarity: float
    metadata: Mapping


@dataclass
class SignatureIndex:
    rows: list[tuple[str, dict[str, float], Mapping]]

    @classmethod
    def from_signatures(cls, signatures: Iterable[Mapping], id_key: str = "case_id") -> "SignatureIndex":
        rows = []
        for sig in signatures:
            case_id = str(sig.get(id_key, ""))
            rows.append((case_id, flatten_signature(sig), sig))
        return cls(rows)

    def nearest(self, signature: Mapping, k: int = 5, min_similarity: float = 0.0) -> list[SimilarCase]:
        query = flatten_signature(signature)
        out = [
            SimilarCase(case_id, cosine_sparse(query, features), meta)
            for case_id, features, meta in self.rows
        ]
        out = [row for row in out if row.similarity >= min_similarity]
        return sorted(out, key=lambda row: (-row.similarity, row.case_id))[:k]


def threshold_clusters(signatures: Iterable[Mapping], threshold: float = 0.75) -> list[list[str]]:
    """Greedy similarity clusters for small datasets.

    This is intentionally simple and deterministic; HDBSCAN can replace it when
    enough data exists, while the rest of the pipeline keeps the same interface.
    """
    index = SignatureIndex.from_signatures(signatures)
    unassigned = {case_id for case_id, _, _ in index.rows}
    clusters = []
    feature_by_id = {case_id: features for case_id, features, _ in index.rows}
    while unassigned:
        seed = sorted(unassigned)[0]
        group = [seed]
        unassigned.remove(seed)
        for other in sorted(list(unassigned)):
            if cosine_sparse(feature_by_id[seed], feature_by_id[other]) >= threshold:
                group.append(other)
                unassigned.remove(other)
        clusters.append(group)
    return clusters
