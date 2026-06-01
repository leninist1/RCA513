"""Training script for learned modules (Direction B: Learned Representations).

Generates pretrained weight files from OpenRCA ground truth records.
Supports Leave-One-System-Out (LOSO) for clean evaluation without data leakage.

Usage:
    python -m prism_v2.learned.train                          # train on ALL systems (leaky)
    python -m prism_v2.learned.train --exclude-system Bank     # LOSO: train on Telecom+Market
    python -m prism_v2.learned.train --exclude-system Telecom  # LOSO: train on Bank+Market
    python -m prism_v2.learned.train --exclude-system Market   # LOSO: train on Bank+Telecom
"""

from __future__ import annotations

import csv
import json
import os
import pickle
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Set, Tuple


_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)


def _resolve_pretrained_dir(exclude_system: Optional[str] = None) -> str:
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pretrained")
    if exclude_system:
        return os.path.join(base, f"loso_no_{exclude_system}")
    return base


_TRAINING_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training")

_RECORD_PATHS = [
    ("Bank", os.path.join(_PROJECT_ROOT, "OpenRCA", "Bank", "Bank", "record.csv")),
    (
        "Telecom",
        os.path.join(_PROJECT_ROOT, "OpenRCA", "Telecom", "Telecom", "record.csv"),
    ),
    (
        "Market",
        os.path.join(
            _PROJECT_ROOT, "OpenRCA", "Market", "Market", "cloudbed-1", "record.csv"
        ),
    ),
    (
        "Market",
        os.path.join(
            _PROJECT_ROOT, "OpenRCA", "Market", "Market", "cloudbed-2", "record.csv"
        ),
    ),
]

_REASON_TO_SUBCAT = {
    "high cpu usage": "cpu",
    "cpu fault": "cpu",
    "container cpu load": "cpu",
    "node cpu load": "cpu",
    "high jvm cpu load": "cpu",
    "node cpu spike": "cpu",
    "high memory usage": "memory",
    "container memory load": "memory",
    "node memory consumption": "memory",
    "jvm out of memory (oom) heap": "jvm_oom",
    "high disk i/o read usage": "disk_io",
    "container read i/o load": "disk_io",
    "node disk read io": "disk_io",
    "container write i/o load": "disk_io",
    "node disk write io": "disk_io",
    "node disk read i/o consumption": "disk_io",
    "node disk write i/o consumption": "disk_io",
    "high disk space usage": "disk_space",
    "disk space consumption": "disk_space",
    "node disk space consumption": "disk_space",
    "network latency": "network_latency",
    "network delay": "network_latency",
    "container network latency": "network_latency",
    "network packet loss": "network_packet_loss",
    "network loss": "network_packet_loss",
    "container packet loss": "network_packet_loss",
    "packet loss": "network_packet_loss",
    "packet retransmission": "network_fault",
    "packet corruption": "network_fault",
    "container network packet retransmission": "network_fault",
    "container network packet corruption": "network_fault",
    "db connection limit": "db_fault",
    "db close": "db_fault",
    "container process termination": "process_kill",
}

_SUBCAT_KEYWORDS = {
    "cpu": ["cpu", "load", "throttle", "utilization", "busy"],
    "memory": ["memory", "mem", "rss", "resident", "swap"],
    "disk_io": ["disk", "iops", "iowait", "filesystem", "storage", "io"],
    "disk_space": ["disk", "pct_usage", "free", "used", "fscapacity"],
    "network_latency": ["latency", "timeout", "slow", "delay", "rtt"],
    "network_packet_loss": ["packet", "loss", "drop", "dropped", "retransmit"],
    "network_fault": ["network", "connection", "refused", "unreachable", "reset"],
    "jvm_oom": ["oom", "outofmemory", "heap", "memoryerror", "killed", "evicted"],
    "process_kill": ["sigkill", "killed", "terminated", "crash", "exit"],
    "high_memory": ["memory", "mem", "rss", "resident", "swap"],
    "db_fault": ["db", "jdbc", "mysql", "postgres", "redis", "sql", "database"],
    "unknown": [],
}

_LIKELIHOOD_SUBCAT_ORDER = [
    "cpu",
    "memory",
    "disk_io",
    "disk_space",
    "network_latency",
    "network_packet_loss",
    "network_fault",
    "jvm_oom",
    "process_kill",
]

_BASE_PARAMS = {
    "cpu": (0.85, 0.08, 0.18, 0.18),
    "memory": (0.72, 0.12, 0.15, 0.22),
    "disk_io": (0.72, 0.12, 0.15, 0.22),
    "disk_space": (0.72, 0.12, 0.15, 0.22),
    "network_latency": (0.78, 0.10, 0.20, 0.20),
    "network_packet_loss": (0.78, 0.10, 0.20, 0.20),
    "network_fault": (0.78, 0.10, 0.20, 0.20),
    "jvm_oom": (0.72, 0.14, 0.12, 0.25),
    "process_kill": (0.78, 0.10, 0.20, 0.20),
    "db_fault": (0.72, 0.12, 0.15, 0.22),
}

_SYSTEM_SIGMA_MULTIPLIER = {
    "Bank": 1.0,
    "Telecom": 1.4,
    "Market": 1.0,
}

_HEADER_SKIP = {"reason"}


def _map_reason(raw_reason: str) -> str:
    """Map a raw reason string to a FaultSubCategory value string."""
    r = raw_reason.strip().lower()
    if not r or r in _HEADER_SKIP:
        return "unknown"
    if r in _REASON_TO_SUBCAT:
        return _REASON_TO_SUBCAT[r]
    for key in sorted(_REASON_TO_SUBCAT, key=len, reverse=True):
        if key in r:
            return _REASON_TO_SUBCAT[key]
    return "unknown"


def load_records(exclude_system: Optional[str] = None) -> List[Tuple[str, str, str]]:
    """Load all records from OpenRCA CSV files.

    Args:
        exclude_system: If set, exclude records from this system for LOSO training.

    Returns list of (system_name, entity_name, raw_reason).
    """
    records: List[Tuple[str, str, str]] = []
    for system, path in _RECORD_PATHS:
        if exclude_system and system == exclude_system:
            print(f"  [LOSO] Excluding {system} records")
            continue
        if not os.path.exists(path):
            print(f"  [WARN] Missing record file: {path}")
            continue
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                component = row.get("component", "").strip()
                reason = row.get("reason", "").strip()
                if not component or not reason:
                    continue
                if reason.lower() in _HEADER_SKIP:
                    continue
                records.append((system, component, reason))
    return records


def train_classifier(records: List[Tuple[str, str, str]]) -> dict:
    """Build learned fault classifier from ground-truth reason labels.

    Stores reason_map (raw reason -> subcategory) and keyword-frequency
    weights derived from training data.
    """
    reason_map: Dict[str, str] = {}
    subcat_counter: Counter = Counter()
    keyword_counter: Dict[str, Counter] = defaultdict(Counter)

    for _system, _entity, reason in records:
        r = reason.strip().lower()
        subcat = _map_reason(r)
        reason_map[r] = subcat
        subcat_counter[subcat] += 1

        for kw in _SUBCAT_KEYWORDS.get(subcat, []):
            if kw in r:
                keyword_counter[subcat][kw] += 1

    total = sum(subcat_counter.values())
    subcat_weights = {sc: count / max(1, total) for sc, count in subcat_counter.items()}

    classifier_data = {
        "reason_map": reason_map,
        "keyword_counts": {
            sc: dict(kw_counts) for sc, kw_counts in keyword_counter.items()
        },
        "subcat_weights": subcat_weights,
        "subcat_totals": dict(subcat_counter),
        "total_records": total,
    }

    print(
        f"  Classifier: {len(reason_map)} reason mappings, "
        f"{len(subcat_counter)} subcategories, {total} records"
    )
    for sc, count in subcat_counter.most_common():
        pct = subcat_weights[sc] * 100
        print(f"    {sc}: {count} ({pct:.1f}%)")
    return classifier_data


def train_entity_profiles(records: List[Tuple[str, str, str]]) -> dict:
    """Build entity profiles: per-entity fault subcategory occurrence counts."""
    profiles: Dict[str, Counter] = defaultdict(Counter)

    for _system, entity, reason in records:
        subcat = _map_reason(reason)
        profiles[entity][subcat] += 1

    result = {}
    for entity, counter in profiles.items():
        result[entity] = dict(counter.most_common())

    print(f"  Entity profiles: {len(result)} entities")
    return result


def train_likelihood_network(records: List[Tuple[str, str, str]]) -> dict:
    """Build per-system per-fault-type likelihood parameters.

    Uses heuristic base params with system-specific sigma adjustments
    (Telecom gets higher sigma due to no logs → more uncertainty).
    """
    system_subcats: Dict[str, Set[str]] = defaultdict(set)
    for system, _entity, reason in records:
        subcat = _map_reason(reason)
        system_subcats[system].add(subcat)

    systems_params: Dict[str, dict] = {}
    for system in sorted(system_subcats):
        sigma_mult = _SYSTEM_SIGMA_MULTIPLIER.get(system, 1.0)
        system_params: Dict[str, dict] = {}

        for subcat in _LIKELIHOOD_SUBCAT_ORDER:
            mu_root, sigma_root, mu_nonroot, sigma_nonroot = _BASE_PARAMS.get(
                subcat, (0.80, 0.10, 0.20, 0.20)
            )
            system_params[subcat] = {
                "mu_root": mu_root,
                "sigma_root": round(sigma_root * sigma_mult, 4),
                "mu_nonroot": mu_nonroot,
                "sigma_nonroot": round(sigma_nonroot * sigma_mult, 4),
            }

        systems_params[system] = system_params

    likelihood_data = {
        "systems": systems_params,
        "default": {
            "mu_root": 0.80,
            "sigma_root": 0.10,
            "mu_nonroot": 0.20,
            "sigma_nonroot": 0.20,
        },
        "system_index": {"Bank": 0, "Telecom": 1, "Market": 2},
        "subcat_index": {sc: i for i, sc in enumerate(_LIKELIHOOD_SUBCAT_ORDER)},
    }

    print(
        f"  Likelihood network: {len(systems_params)} systems, "
        f"{len(_LIKELIHOOD_SUBCAT_ORDER)} fault types"
    )
    return likelihood_data


def train_embeddings(
    records: List[Tuple[str, str, str]], entity_profiles: dict
) -> dict:
    """Generate deterministic entity embeddings from entity name hashes.

    Uses entity name hash as RNG seed for reproducible 64-dim normalized
    random vectors. Also stores per-entity FaultSubCategory affinity scores
    derived from entity profile counts.
    """
    import numpy as np

    all_entities: Set[str] = set()
    for _system, entity, _reason in records:
        all_entities.add(entity)

    vectors: Dict[str, np.ndarray] = {}
    affinities: Dict[str, Dict[str, float]] = {}

    for entity in sorted(all_entities):
        seed = hash(entity) & 0xFFFFFFFF
        rng = np.random.RandomState(seed)
        vec = rng.randn(64).astype(np.float64)
        norm = float(np.sqrt(np.sum(vec * vec)))
        if norm > 1e-9:
            vec = vec / norm
        vectors[entity] = vec

        profile = entity_profiles.get(entity, {})
        total = sum(profile.values())
        if total > 0:
            affinities[entity] = {sc: cnt / total for sc, cnt in profile.items()}
        else:
            affinities[entity] = {}

    embedding_data = {
        "vectors": vectors,
        "affinities": affinities,
        "dim": 64,
    }

    print(f"  Embeddings: {len(vectors)} entities, dim=64")
    return embedding_data


def main(exclude_system: Optional[str] = None):
    """Main training entry point.

    Loads OpenRCA records, trains all learned modules, and saves
    pretrained weight files to learned/pretrained/ (or pretrained/loso_no_{system}/).

    Args:
        exclude_system: If set, train on all systems EXCEPT this one (LOSO).
    """
    pretrained_dir = _resolve_pretrained_dir(exclude_system)
    tag = f" [LOSO exclude={exclude_system}]" if exclude_system else " [ALL SYSTEMS]"

    print("=" * 60)
    print(f"PRISM v2 Learned Module Training{tag}")
    print("=" * 60)
    print()

    os.makedirs(_TRAINING_DIR, exist_ok=True)
    os.makedirs(pretrained_dir, exist_ok=True)

    training_init = os.path.join(_TRAINING_DIR, "__init__.py")
    if not os.path.exists(training_init):
        with open(training_init, "w") as f:
            f.write('"""Training utilities for learned modules."""\n')

    print("[1/4] Loading records...")
    records = load_records(exclude_system=exclude_system)
    if not records:
        print("  [ERROR] No records loaded. Check OpenRCA data paths.")
        sys.exit(1)
    print(
        f"  Loaded {len(records)} records"
        f"{' (excluding ' + exclude_system + ')' if exclude_system else ''}"
    )

    train_systems = sorted(set(s for s, _, _ in records))
    print(f"  Training systems: {train_systems}")
    print()

    print("[2/4] Training classifier...")
    classifier_data = train_classifier(records)
    classifier_path = os.path.join(pretrained_dir, "classifier.pkl")
    with open(classifier_path, "wb") as f:
        pickle.dump(classifier_data, f)
    print(f"  Saved: {classifier_path}")
    print()

    print("[3/4] Training entity profiles...")
    entity_profiles = train_entity_profiles(records)
    profiles_path = os.path.join(pretrained_dir, "entity_profiles.json")
    with open(profiles_path, "w", encoding="utf-8") as f:
        json.dump(entity_profiles, f, indent=2)
    print(f"  Saved: {profiles_path}")
    print()

    print("[4a/4] Training likelihood network...")
    likelihood_data = train_likelihood_network(records)
    likelihood_path = os.path.join(pretrained_dir, "likelihood.pkl")
    with open(likelihood_path, "wb") as f:
        pickle.dump(likelihood_data, f)
    print(f"  Saved: {likelihood_path}")
    print()

    print("[4b/4] Training entity embeddings...")
    embedding_data = train_embeddings(records, entity_profiles)
    embeddings_path = os.path.join(pretrained_dir, "embeddings.pkl")
    with open(embeddings_path, "wb") as f:
        pickle.dump(embedding_data, f)
    print(f"  Saved: {embeddings_path}")
    print()

    print("=" * 60)
    print("Training complete.")
    print(f"Output directory: {pretrained_dir}")
    print()
    print("Files generated:")
    for fname in [
        "classifier.pkl",
        "likelihood.pkl",
        "entity_profiles.json",
        "embeddings.pkl",
    ]:
        fpath = os.path.join(pretrained_dir, fname)
        if os.path.exists(fpath):
            size_kb = os.path.getsize(fpath) / 1024
            print(f"  {fname} ({size_kb:.1f} KB)")
    print("=" * 60)
    return pretrained_dir


if __name__ == "__main__":
    exclude = None
    if "--exclude-system" in sys.argv:
        idx = sys.argv.index("--exclude-system")
        if idx + 1 < len(sys.argv):
            exclude = sys.argv[idx + 1]
    main(exclude_system=exclude)
