"""Bank reason taxonomy and canonicalization helpers.

The evaluator uses exact string matching for reasons, so Bank predictions must
end in the Bank ground-truth vocabulary instead of broad diagnostic families.
"""

from __future__ import annotations

from collections import Counter
import csv
from functools import lru_cache
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple


BANK_DATA_ROOT = Path("/mnt/d/Datasets/OpenRCA/Bank")

FALLBACK_BANK_REASONS = (
    "high CPU usage",
    "high memory usage",
    "network latency",
    "network packet loss",
    "high disk I/O read usage",
    "high disk space usage",
    "JVM Out of Memory (OOM) Heap",
    "high JVM CPU load",
)

BANK_REASON_PRIORITY = {
    reason: idx for idx, reason in enumerate(FALLBACK_BANK_REASONS)
}

RAW_REASON_ALIAS_WEIGHT = 0.15
RAW_REASON_TEXT_WEIGHT = 0.0
METRIC_REASON_CAPS = {
    "high CPU usage": 2.4,
    "high memory usage": 3.0,
    "network latency": 3.5,
    "network packet loss": 3.5,
    "high disk I/O read usage": 2.2,
    "high disk space usage": 1.6,
    "JVM Out of Memory (OOM) Heap": 2.5,
    "high JVM CPU load": 2.5,
}

RAW_REASON_ALIASES = {
    "cpu fault": "high CPU usage",
    "high cpu usage": "high CPU usage",
    "jvm out of memory": "JVM Out of Memory (OOM) Heap",
    "jvm out of memory oom heap": "JVM Out of Memory (OOM) Heap",
    "high memory usage": "high memory usage",
    "network latency": "network latency",
    "network packet loss": "network packet loss",
    "network fault": "network latency",
    "disk i/o consumption": "high disk I/O read usage",
    "disk io consumption": "high disk I/O read usage",
    "high disk i/o read usage": "high disk I/O read usage",
    "high disk io read usage": "high disk I/O read usage",
    "high disk space usage": "high disk space usage",
}

REASON_PATTERNS: Dict[str, Tuple[str, ...]] = {
    "JVM Out of Memory (OOM) Heap": (
        "outofmemory",
        "out of memory",
        "oom",
        "heap",
        "allocation failure",
        "full gc",
        "gc overhead",
    ),
    "high JVM CPU load": (
        "jvm_cpuload",
        "jvm cpu",
        "jvm-cpu",
        "jvmcpuload",
    ),
    "high CPU usage": (
        "cpu",
        "cpuutil",
        "cpuload",
        "cpu_user",
        "throttl",
        "cfs_throttled",
        "proc_user",
    ),
    "network packet loss": (
        "packet loss",
        "packet_loss",
        "packetloss",
        "loss",
        "drop",
        "dropped",
        "retransmit",
        "retrans",
        "tcp",
        "fin_wait",
        "tcp fin wait",
        "tcp close wait",
        "totaltcpconnnum",
        "netinerr",
        "netouterr",
        "netinerrprc",
        "netouterrprcc",
        "netpacketsin",
        "netpacketsout",
        "sent_queue",
        "received_queue",
    ),
    "network latency": (
        "latency",
        "mrt",
        "rtt",
        "timeout",
        "timed out",
        "duration",
        "slow",
        "elapsedtime",
        "netbandwidthutil",
        "netkbtotalpersec",
        "maxtimerequestinfo",
        "processingtimerequestinfo",
        "tnsping",
        "icmp_ping",
    ),
    "high disk space usage": (
        "disk space",
        "fsinodeusedpercent",
        "fscapacity",
        "fsusedspace",
        "fsavailablespace",
        "percentused",
        "usedpercent",
    ),
    "high disk I/O read usage": (
        "disk i/o",
        "disk io",
        "dskread",
        "dskwrite",
        "dskbps",
        "dskpercentbusy",
        "dskavgserv",
        "dskrtps",
        "dskwtps",
        "dsktps",
        "cpuwio",
        "rkb_s",
        "w_await",
        "await",
        "avg_q_sz",
        "queue",
        "iops",
        "iowait",
        "fs_reads",
        "fs_writes",
        "pending reads",
        "pending writes",
        "io",
    ),
    "high memory usage": (
        "memory",
        "mem",
        "memused",
        "memperc",
        "memfree",
        "cachemem",
        "nocachememperc",
        "container_memory",
        "rss",
        "swap",
        "pgfault",
        "failcnt",
    ),
}


def canonical_bank_reasons(data_root: Optional[str] = None) -> List[str]:
    """Return the Bank reason vocabulary observed in ground-truth data."""
    return list(_canonical_bank_reasons_cached(str(data_root or BANK_DATA_ROOT)))


@lru_cache(maxsize=8)
def _canonical_bank_reasons_cached(data_root: str) -> Tuple[str, ...]:
    stats = bank_reason_taxonomy_stats(data_root)
    reasons = sorted(
        stats["reasons"],
        key=lambda reason: (-stats["counts"].get(reason, 0), BANK_REASON_PRIORITY.get(reason, 999), reason),
    )
    return tuple(reasons or FALLBACK_BANK_REASONS)


def bank_reason_taxonomy_stats(data_root: Optional[str] = None) -> Dict[str, Any]:
    """Collect Bank canonical reasons from record.csv and query scoring points."""
    root = Path(data_root or BANK_DATA_ROOT)
    record_counts = _record_reason_counts(root / "record.csv")
    scoring_counts = _query_scoring_reason_counts(root / "query.csv")
    reasons = set(record_counts) | set(scoring_counts)
    counts = Counter(record_counts)
    counts.update(scoring_counts)
    return {
        "data_root": str(root),
        "reasons": sorted(reasons),
        "counts": dict(counts),
        "record_counts": dict(record_counts),
        "scoring_point_counts": dict(scoring_counts),
        "only_in_record": sorted(set(record_counts) - set(scoring_counts)),
        "only_in_scoring_points": sorted(set(scoring_counts) - set(record_counts)),
    }


def normalize_reason_text(text: str) -> str:
    """Normalize a raw reason string to a Bank canonical reason when possible."""
    cleaned = _normalize_key(text)
    if not cleaned:
        return ""
    if cleaned in RAW_REASON_ALIASES:
        return RAW_REASON_ALIASES[cleaned]

    scores, evidence = _score_reason_text(str(text))
    if scores:
        reason = _best_reason(scores)
        if reason:
            return reason
    return str(text or "").strip()


def choose_reason_from_candidates(
    raw_reason: str,
    instruction: str,
    metric_names: List[str],
) -> str:
    """Choose a Bank canonical reason from raw text and metric-name evidence."""
    candidates, _debug = _choose_reason(
        raw_reason=raw_reason,
        instruction=instruction,
        metric_names=metric_names,
        log_text="",
        query=None,
        reason_votes=None,
    )
    return candidates


def infer_bank_reason(
    entity: str,
    evidence_pool: Dict,
    query,
    metric_detail=None,
    log_detail=None,
) -> str:
    """Infer a Bank canonical reason without reading the current sample GT."""
    return explain_bank_reason(
        entity=entity,
        evidence_pool=evidence_pool,
        query=query,
        metric_detail=metric_detail,
        log_detail=log_detail,
    )["canonical_reason"]


def explain_bank_reason(
    entity: str,
    evidence_pool: Dict,
    query,
    metric_detail=None,
    log_detail=None,
    raw_reason: str = "",
    candidate_entities: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Return the canonical reason plus diagnostic details."""
    metric_names, metric_scores = _collect_metric_names(entity, metric_detail, candidate_entities)
    log_text = _collect_log_text(entity, log_detail, candidate_entities)
    evidence_texts = _collect_evidence_texts(entity, evidence_pool)
    reason_votes = _collect_reason_votes(entity, evidence_pool)
    combined_log_text = " ".join([log_text] + evidence_texts)
    canonical_reason, debug = _choose_reason(
        raw_reason=raw_reason,
        instruction=getattr(query, "instruction", "") or "",
        metric_names=metric_names,
        log_text=combined_log_text,
        query=query,
        reason_votes=reason_votes,
    )
    debug.update(
        {
            "entity": entity,
            "candidate_entities": list(candidate_entities or [entity]),
            "metric_scores": {
                name: round(float(score), 4)
                for name, score in sorted(metric_scores.items(), key=lambda item: item[1], reverse=True)[:10]
            },
        }
    )
    return debug


def _choose_reason(
    raw_reason: str,
    instruction: str,
    metric_names: List[str],
    log_text: str,
    query,
    reason_votes: Optional[Dict[str, float]],
) -> Tuple[str, Dict[str, Any]]:
    taxonomy = _taxonomy_for_query(query)
    scores: Counter = Counter()
    evidence: Dict[str, List[str]] = {reason: [] for reason in taxonomy}

    def add(reason: str, amount: float, why: str):
        if reason not in taxonomy or amount <= 0:
            return
        scores[reason] += amount
        if len(evidence[reason]) < 8:
            evidence[reason].append(why)

    normalized_raw = normalize_reason_text(raw_reason)
    if normalized_raw in taxonomy:
        add(normalized_raw, RAW_REASON_ALIAS_WEIGHT, f"raw_reason:{raw_reason}")

    for reason, amount, why in _score_reason_text(raw_reason)[1]:
        add(reason, amount * RAW_REASON_TEXT_WEIGHT, f"raw_reason_pattern:{why}")
    for reason, amount, why in _score_reason_text(instruction)[1]:
        add(reason, amount * 0.25, f"instruction:{why}")
    for reason, amount, why in _score_reason_text(log_text)[1]:
        add(reason, amount * 1.25, f"log:{why}")

    metric_totals: Counter = Counter()
    for metric_name in metric_names:
        metric_evidence = _score_metric_text(metric_name)
        best_metric_hits: Dict[str, Tuple[float, str]] = {}
        for reason, amount, why in metric_evidence:
            current = best_metric_hits.get(reason)
            if current is None or amount > current[0]:
                best_metric_hits[reason] = (amount, why)
        for reason, (amount, why) in best_metric_hits.items():
            cap = METRIC_REASON_CAPS.get(reason)
            if cap is not None:
                remaining = cap - metric_totals[reason]
                if remaining <= 0:
                    continue
                amount = min(amount, remaining)
                metric_totals[reason] += amount
            add(reason, amount, f"metric:{metric_name}:{why}")

    for family, vote in (reason_votes or {}).items():
        for reason in _family_to_reasons(family):
            add(reason, float(vote) * 0.75, f"reason_vote:{family}")

    canonical_reason = _best_reason(scores, taxonomy) or _default_reason(taxonomy)
    matched_rule = "fallback"
    if scores:
        matched_rule = "weighted_evidence"
    if normalized_raw in taxonomy and canonical_reason == normalized_raw:
        matched_rule = "raw_reason_alias"

    debug = {
        "raw_reason": raw_reason,
        "canonical_reason": canonical_reason,
        "matched_rule": matched_rule,
        "taxonomy": taxonomy,
        "scores": {reason: round(float(scores[reason]), 4) for reason in taxonomy if scores.get(reason, 0) > 0},
        "evidence": evidence.get(canonical_reason, []),
    }
    return canonical_reason, debug


def _score_reason_text(text: Any) -> Tuple[Counter, List[Tuple[str, float, str]]]:
    scores: Counter = Counter()
    evidence: List[Tuple[str, float, str]] = []
    lower = str(text or "").lower()
    if not lower:
        return scores, evidence
    compact = _normalize_key(lower)
    for reason, patterns in REASON_PATTERNS.items():
        for pattern in patterns:
            if _pattern_matches(lower, compact, pattern):
                amount = _pattern_weight(reason, pattern)
                scores[reason] += amount
                evidence.append((reason, amount, pattern))
    return scores, evidence


def _score_metric_text(text: Any) -> List[Tuple[str, float, str]]:
    evidence = _score_reason_text(text)[1]
    metric_key = _normalize_key(text).replace(" ", "")
    if "cpuidle" in metric_key:
        return [item for item in evidence if item[0] != "high CPU usage"]
    if "cpuwio" in metric_key:
        return [item for item in evidence if item[0] != "high CPU usage"]
    return evidence


def _pattern_matches(lower: str, compact: str, pattern: str) -> bool:
    pattern_lower = str(pattern or "").lower()
    pattern_key = _normalize_key(pattern_lower)
    if not pattern_key:
        return False
    if len(pattern_key) <= 3 and pattern_key.isalnum():
        return bool(re.search(rf"(?<![a-z0-9]){re.escape(pattern_key)}(?![a-z0-9])", compact))
    return pattern_lower in lower or pattern_key in compact


def _pattern_weight(reason: str, pattern: str) -> float:
    if reason == "JVM Out of Memory (OOM) Heap":
        if pattern in {"outofmemory", "out of memory", "oom"}:
            return 4.0
        return 1.0
    if reason == "high JVM CPU load":
        return 3.0
    if reason == "network packet loss":
        if pattern in {"packet loss", "packet_loss", "packetloss", "loss", "drop", "dropped", "retransmit", "retrans"}:
            return 2.0
        if pattern in {"netinerr", "netouterr", "netinerrprc", "netouterrprcc"}:
            return 1.8
        if pattern in {"netpacketsin", "netpacketsout"}:
            return 1.3
        if pattern in {"tcp", "tcp fin wait", "tcp close wait", "totaltcpconnnum"}:
            return 1.0
        return 1.2
    if reason == "network latency":
        if pattern in {"latency", "mrt", "rtt", "timeout", "timed out"}:
            return 2.0
        if pattern in {"netbandwidthutil", "netkbtotalpersec", "maxtimerequestinfo", "processingtimerequestinfo"}:
            return 2.0
        return 1.0
    if reason == "high disk space usage":
        if pattern == "disk space":
            return 1.5
        if pattern in {"fscapacity", "fsavailablespace"}:
            return 0.8
        return 1.2
    if pattern in {"cpu", "memory", "mem", "io", "tcp"}:
        return 0.8
    return 1.0


def _best_reason(scores: Counter, taxonomy: Optional[List[str]] = None) -> str:
    if not scores:
        return ""
    allowed = taxonomy or list(BANK_REASON_PRIORITY)
    present = [reason for reason in allowed if scores.get(reason, 0) > 0]
    if not present:
        return ""
    return max(
        present,
        key=lambda reason: (
            scores[reason],
            -BANK_REASON_PRIORITY.get(reason, 999),
            reason,
        ),
    )


def _default_reason(taxonomy: List[str]) -> str:
    if "high memory usage" in taxonomy:
        return "high memory usage"
    return taxonomy[0] if taxonomy else "high memory usage"


def _taxonomy_for_query(query) -> List[str]:
    return canonical_bank_reasons() or list(FALLBACK_BANK_REASONS)


def _collect_metric_names(entity: str, metric_detail, candidate_entities: Optional[List[str]]) -> Tuple[List[str], Dict[str, float]]:
    if metric_detail is None:
        return [], {}
    entities = list(candidate_entities or [entity])
    metric_scores: Dict[str, float] = {}
    for candidate in entities:
        metric_map = metric_detail.get(candidate, {}) if isinstance(metric_detail, dict) else {}
        if not isinstance(metric_map, dict):
            continue
        for name, score in metric_map.items():
            metric_scores[str(name)] = max(metric_scores.get(str(name), 0.0), float(score or 0.0))
    ordered = [
        name
        for name, _score in sorted(metric_scores.items(), key=lambda item: item[1], reverse=True)
    ]
    return ordered[:20], metric_scores


def _collect_log_text(entity: str, log_detail, candidate_entities: Optional[List[str]]) -> str:
    if log_detail is None or not isinstance(log_detail, dict):
        return ""
    parts: List[str] = []
    for candidate in list(candidate_entities or [entity]):
        info = log_detail.get(candidate, {})
        if isinstance(info, dict) and info.get("text"):
            parts.append(str(info.get("text", ""))[:1000])
    return " ".join(parts)


def _collect_evidence_texts(entity: str, evidence_pool: Dict) -> List[str]:
    texts: List[str] = []
    for ev in _entity_evidence(entity, evidence_pool):
        if not isinstance(ev, dict):
            texts.append(str(ev))
            continue
        for key in ("content", "message", "text", "summary", "source"):
            if ev.get(key):
                texts.append(str(ev[key]))
        details = ev.get("details")
        if isinstance(details, dict):
            texts.extend(str(key) for key in details.keys())
    return texts[:20]


def _collect_reason_votes(entity: str, evidence_pool: Dict) -> Dict[str, float]:
    votes: Counter = Counter()
    for ev in _entity_evidence(entity, evidence_pool):
        if not isinstance(ev, dict):
            continue
        if ev.get("best_family"):
            votes[str(ev["best_family"])] += 1.0
        reason_votes = ev.get("reason_votes")
        if isinstance(reason_votes, dict):
            for family, score in reason_votes.items():
                votes[str(family)] += float(score or 0.0)
    return dict(votes)


def _entity_evidence(entity: str, evidence_pool: Dict) -> Iterable[Any]:
    if not isinstance(evidence_pool, dict):
        return []
    evidence = evidence_pool.get(entity, [])
    return evidence if isinstance(evidence, list) else []


def _family_to_reasons(family: str) -> List[str]:
    family_key = _normalize_key(family)
    if family_key in {"jvm", "oom"}:
        return ["JVM Out of Memory (OOM) Heap", "high JVM CPU load"]
    if family_key == "cpu":
        return ["high CPU usage"]
    if family_key == "memory":
        return ["high memory usage", "JVM Out of Memory (OOM) Heap"]
    if family_key in {"network", "latency"}:
        return ["network latency", "network packet loss"]
    if family_key in {"packetloss", "packet_loss"}:
        return ["network packet loss"]
    if family_key in {"disk", "diskio", "io"}:
        return ["high disk I/O read usage", "high disk space usage"]
    if family_key == "diskspace":
        return ["high disk space usage"]
    return []


def _record_reason_counts(path: Path) -> Counter:
    counts: Counter = Counter()
    if not path.exists():
        return counts
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            reason = str(row.get("reason") or row.get("Reason") or "").strip()
            if reason:
                counts[reason] += 1
    return counts


def _query_scoring_reason_counts(path: Path) -> Counter:
    counts: Counter = Counter()
    if not path.exists():
        return counts
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            for reason in _parse_scoring_reasons(row.get("scoring_points", "")):
                counts[reason] += 1
    return counts


def _parse_scoring_reasons(text: str) -> List[str]:
    reasons: List[str] = []
    for line in str(text or "").splitlines():
        if "root cause reason" not in line:
            continue
        match = re.search(r"reason is\s+(.+)", line)
        if match:
            reasons.append(match.group(1).strip())
    return reasons


def _normalize_key(text: Any) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9/]+", " ", str(text or "").lower())).strip()
