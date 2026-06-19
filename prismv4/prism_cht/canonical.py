"""Canonicalization and deep-freeze utilities for PRISM-CHT.

Shared helpers for deterministic JSON canonicalization, recursive
immutability enforcement, and the single tool-call signature function
used by EvidenceGraph, ActionGate, and DiscriminativeAction.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from types import MappingProxyType
from typing import Any, Dict, Mapping, Set

_MUTABLE_CONTAINER_TYPES: tuple = (dict, list, set)


def _has_mutable_internals(obj: Any) -> bool:
    """Return True if *obj* is or contains a mutable dict, list, or set.

    Recursively descends into tuples, MappingProxyType, frozenset,
    and nested frozen dataclasses.  Stops and returns True at the
    first mutable container found.
    """
    if isinstance(obj, _MUTABLE_CONTAINER_TYPES):
        return True
    if isinstance(obj, tuple):
        return any(_has_mutable_internals(item) for item in obj)
    if isinstance(obj, MappingProxyType):
        return any(_has_mutable_internals(val) for val in obj.values())
    if isinstance(obj, frozenset):
        return any(_has_mutable_internals(item) for item in obj)
    if hasattr(obj, "__dataclass_fields__") and getattr(obj, "__dataclass_params__", None):
        if getattr(obj.__dataclass_params__, "frozen", False):
            return any(
                _has_mutable_internals(getattr(obj, f.name))
                for f in dataclasses.fields(obj)
            )
    return False


def deep_freeze(value: Any) -> Any:
    """Recursively convert mutable containers to immutable equivalents.

    - Mapping -> MappingProxyType (recursively frozen)
    - list / tuple -> tuple (recursively frozen)
    - set / frozenset -> deterministic tuple (sorted, recursively frozen)
    - Frozen dataclass instances pass through only if all their fields
      are recursively immutable; otherwise TypeError is raised.
    - Scalars (str, int, float, bool, None, bytes) pass through unchanged.

    The result is independent of object addresses; structurally equal
    inputs produce structurally equal frozen outputs.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value
    if isinstance(value, (int, float, bool, type(None))):
        return value
    if isinstance(value, MappingProxyType):
        return value
    if hasattr(value, "__dataclass_fields__") and getattr(value, "__dataclass_params__", None):
        if getattr(value.__dataclass_params__, "frozen", False):
            if _has_mutable_internals(value):
                raise TypeError(
                    f"Frozen dataclass {type(value).__name__!r} contains mutable "
                    f"internals (dict, list, or set); cannot be trusted as "
                    f"deeply safe.  Freeze the nested containers before passing "
                    f"to deep_freeze."
                )
            return value
    if isinstance(value, Mapping):
        return MappingProxyType({
            key: deep_freeze(val)
            for key, val in value.items()
        })
    if isinstance(value, (list, tuple)):
        return tuple(deep_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return tuple(sorted(deep_freeze(item) for item in value))
    raise TypeError(
        f"deep_freeze does not support type {type(value).__name__}"
    )


def canonicalize_json_value(value: Any) -> Any:
    """Convert a Python value into a JSON-serializable canonical form.

    - Mapping -> sorted-key dict (recursively canonicalized)
    - Sequence (list, tuple) -> list (recursively canonicalized)
    - set / frozenset -> list sorted by canonical representation
    - bool, None, int, float, str -> pass through

    Raises TypeError for types that cannot be reliably canonicalized
    (e.g. custom objects, bytes).
    """
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    if isinstance(value, Mapping):
        return dict(
            sorted(
                (str(key), canonicalize_json_value(val))
                for key, val in value.items()
            )
        )
    if isinstance(value, (list, tuple)):
        return [canonicalize_json_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted(
            (canonicalize_json_value(item) for item in value),
            key=lambda x: json.dumps(
                x, sort_keys=True, separators=(",", ":"), ensure_ascii=True
            ),
        )
    raise TypeError(
        f"Cannot canonicalize type {type(value).__name__}; "
        f"expected JSON-compatible type"
    )


def to_dispatch_args(value: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively convert a frozen/mixed args Mapping into a plain mutable dict.

    *MappingProxyType* is unwrapped to plain ``dict``, *tuple* becomes
    ``list``, and every nested container is fully converted so that
    the return value is safe to pass to tool ``execute()`` functions.
    The original ``action.args`` is never modified by writing to the
    result.
    """
    if not isinstance(value, Mapping):
        raise TypeError(
            f"to_dispatch_args requires a Mapping, got {type(value).__name__}"
        )
    result = canonicalize_json_value(value)
    if not isinstance(result, dict):
        raise TypeError(
            f"canonicalize_json_value returned {type(result).__name__} "
            f"instead of dict"
        )
    return result


def build_tool_call_signature(
    *,
    tool_name: str,
    args: Mapping[str, Any],
) -> str:
    """Return a deterministic SHA-256 signature for a tool call.

    The signature depends only on *tool_name* and the canonicalized
    *args* dict.  It is independent of dict key ordering, natural
    language questions, timestamps, random data, or object addresses.

    Raises ValueError if *tool_name* is empty or *args* is not a Mapping.
    """
    if not tool_name or not tool_name.strip():
        raise ValueError("tool_name must be non-empty")
    if not isinstance(args, Mapping):
        raise ValueError(
            f"args must be a Mapping, got {type(args).__name__}"
        )
    canonical = canonicalize_json_value({
        "tool_name": tool_name,
        "args": dict(args),
    })
    canonical_json = json.dumps(
        canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
