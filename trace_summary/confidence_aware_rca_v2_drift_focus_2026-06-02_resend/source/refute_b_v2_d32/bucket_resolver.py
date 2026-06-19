"""Row-aware bucket resolution bridging precomputed bucket columns to the
OpenRCA D32 bucketing call sites.

The OpenRCA D32 pipeline decides KPI->bucket membership in several places via
hardcoded token matching (``evidence.kpi_in_bucket``, ``signature.reason_for_kpi``,
``joint_candidates._kpi_buckets``).  For portable datasets that supply a
dataset-native precomputed ``buckets`` column on ``metric_df`` (see
``portable_bucket_assignment`` + ``portable_adapters``), the algorithm should
read that column instead of inferring membership from KPI name tokens.

This module provides row-aware helpers that:
  * read the precomputed ``buckets`` column when present, and
  * fall back to the original OpenRCA token logic when the column is absent
    (so the OpenRCA-native dataset path stays byte-identical).

A bucket->reason inverse is also supplied so that callers which need a reason
*string* (e.g. candidate construction) can produce one whose
``reason_bucket(reason)`` round-trips back to the precomputed bucket, keeping
candidate identity consistent with the precomputed assignment.
"""
from __future__ import annotations

from typing import Any

from refute_b_v2_d32.portable_schema import BUCKETS_COLUMN


# Bucket -> reason string inverse for the dataset-native path.  Each reason
# string is chosen so that ``reason_bucket(reason)`` round-trips to the bucket
# (verified against schema.REASON_BUCKETS).  These are the same reason strings
# the OpenRCA token path would produce, so downstream scoring/summariser
# vocabulary is unchanged.
BUCKET_TO_REASON: dict[str, str] = {
    "cpu": "high CPU usage",
    "memory": "high memory usage",
    "jvm_oom": "JVM Out of Memory (OOM) Heap",
    "disk_io": "high disk I/O read usage",
    "filesystem": "high disk space usage",
    "network_latency": "network latency",
    "network_packet_loss": "network packet loss",
    "db_connection": "db connection limit",
    "process_termination": "container process termination",
}


def _token_kpi_in_bucket(kpi_name: str, bucket: str) -> bool:
    from refute_b_v2_d32.evidence import kpi_in_bucket
    return kpi_in_bucket(kpi_name, bucket)


def _token_kpi_buckets(component: str, kpi_name: str) -> set[str]:
    from refute_b_v2_d32.joint_candidates import _kpi_buckets
    return _kpi_buckets(component, kpi_name)


def _token_reason_for_kpi(kpi_name: str) -> str | None:
    from refute_b_v2_d32.signature import reason_for_kpi
    return reason_for_kpi(kpi_name)


def row_buckets(row: Any, kpi_name: str, component: str = "") -> set[str]:
    """Return the set of reason buckets a metric row evidences.

    Reads the precomputed ``buckets`` cell when present; otherwise falls back
    to ``joint_candidates._kpi_buckets`` (OpenRCA token logic).
    """
    precomputed = _row_buckets_cell(row)
    if precomputed is not None:
        return set(precomputed)
    return _token_kpi_buckets(component, kpi_name)


def row_in_bucket(row: Any, bucket: str, kpi_name: str) -> bool:
    """Return True if a metric row evidences the given reason bucket.

    Reads the precomputed ``buckets`` cell when present; otherwise falls back
    to ``evidence.kpi_in_bucket`` (OpenRCA token logic).
    """
    precomputed = _row_buckets_cell(row)
    if precomputed is not None:
        return bucket in precomputed
    return _token_kpi_in_bucket(kpi_name, bucket)


def row_reason(row: Any, kpi_name: str) -> str | None:
    """Return a reason string for a metric row, or None if it evidences no bucket.

    When a precomputed ``buckets`` cell is present and non-empty, returns a
    reason string whose ``reason_bucket(reason)`` round-trips to one of the
    precomputed buckets (chosen deterministically by BUCKET_TO_REASON order).
    When the cell is present but empty, returns None (the KPI evidences no
    bucket under the dataset-native assignment).  When no cell is present,
    falls back to ``signature.reason_for_kpi`` (OpenRCA token logic).
    """
    precomputed = _row_buckets_cell(row)
    if precomputed is not None:
        if not precomputed:
            return None
        for bucket, reason in BUCKET_TO_REASON.items():
            if bucket in precomputed:
                return reason
        # Unknown bucket in the cell: slugify-style fallback so reason_bucket
        # still round-trips (matches schema.reason_bucket's fallback branch).
        return next(iter(sorted(precomputed)))
    return _token_reason_for_kpi(kpi_name)


def _row_buckets_cell(row: Any) -> set[str] | None:
    """Extract the precomputed buckets cell from a row, or None if absent.

    ``row`` may be a pandas Row namedtuple (from itertuples), a pandas Series,
    or a plain mapping.  Returns None when the frame has no ``buckets`` column
    so callers can fall back to token logic.  Returns an empty set when the
    column exists but the cell is empty (excluded / non-fault KPI).
    """
    if hasattr(row, "_fields"):
        # namedtuple from itertuples: BUCKETS_COLUMN is a valid field name.
        if BUCKETS_COLUMN in row._fields:
            return _coerce_buckets(getattr(row, BUCKETS_COLUMN))
        return None
    if hasattr(row, "get") and not isinstance(row, str):
        # Mapping or Series-like.
        if BUCKETS_COLUMN in row:
            return _coerce_buckets(row[BUCKETS_COLUMN])
        return None
    return None


def _coerce_buckets(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        stripped = value.strip()
        return {stripped} if stripped else set()
    try:
        return {str(item) for item in value}
    except TypeError:
        return set()
