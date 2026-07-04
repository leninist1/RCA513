#!/usr/bin/env python3
"""Shared utilities for CAPE-RCA paper artifact generation."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = REPO_ROOT / "paper_artifacts"
RAW_DIR = ARTIFACT_ROOT / "raw"
TABLE_DIR = ARTIFACT_ROOT / "tables"
FIGURE_DIR = ARTIFACT_ROOT / "figures"
LOG_DIR = ARTIFACT_ROOT / "logs"
CASE_STUDY_DIR = ARTIFACT_ROOT / "case_studies"

RCAEVAL_SYSTEMS = [
    "RE1-OB",
    "RE1-SS",
    "RE1-TT",
    "RE2-OB",
    "RE2-SS",
    "RE2-TT",
    "RE3-OB",
    "RE3-SS",
    "RE3-TT",
]

RCAEVAL_SOURCE_FILES = {
    "RE1-OB": RAW_DIR / "RE1-OB_cape_rca_full.json",
    "RE1-SS": RAW_DIR / "RE1-SS_cape_rca_full.json",
    "RE1-TT": RAW_DIR / "RE1-TT_cape_rca_full.json",
    "RE2-OB": REPO_ROOT / "results/prism_cht/v3_full/RE2-OB_v3.json",
    "RE2-SS": REPO_ROOT / "results/prism_cht/v3_full/RE2-SS_v3.json",
    "RE2-TT": REPO_ROOT / "results/prism_cht/v3_full/RE2-TT_v3.json",
    "RE3-OB": REPO_ROOT / "results/prism_cht/v3_full/RE3-OB_v3.json",
    "RE3-SS": REPO_ROOT / "results/prism_cht/v3_full/RE3-SS_v3.json",
    "RE3-TT": REPO_ROOT / "results/prism_cht/v3_full/RE3-TT_v3.json",
}

EADRO_SOURCE_FILES = {
    "Eadro-TT": REPO_ROOT / "results/prism_cht/Eadro-TT_ambig_full.json",
    "Eadro-SN": REPO_ROOT / "results/prism_cht/Eadro-SN_ambig_full.json",
}


def ensure_dirs() -> None:
    for path in [RAW_DIR, TABLE_DIR, FIGURE_DIR, LOG_DIR, CASE_STUDY_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def relpath(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def compact_json(value: Any) -> str:
    if value in (None, ""):
        return ""
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def split_rcaeval_system(dataset_id: str) -> tuple[str, str]:
    parts = dataset_id.split("-", 1)
    suite = parts[0] if parts else dataset_id
    system_code = parts[1] if len(parts) > 1 else ""
    system_name = {
        "OB": "Online Boutique",
        "SS": "Sock Shop",
        "TT": "Train Ticket",
    }.get(system_code, system_code)
    return suite, system_name


def available_modalities(dataset_id: str) -> str:
    if dataset_id.startswith("RE1"):
        return "metrics only"
    if dataset_id.startswith(("RE2", "RE3")):
        return "metrics/logs/traces"
    return ""


def normalize_ranking(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    ranking: list[str] = []
    for item in value:
        if isinstance(item, str):
            name = item.strip()
        elif isinstance(item, dict):
            name = str(
                item.get("component")
                or item.get("root_component")
                or item.get("service")
                or item.get("name")
                or ""
            ).strip()
        else:
            name = str(item).strip()
        if name and name not in ranking:
            ranking.append(name)
    return ranking


def recall_pool_components(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    components: list[str] = []
    for item in value:
        if isinstance(item, str):
            name = item.strip()
        elif isinstance(item, dict):
            name = str(
                item.get("component")
                or item.get("root_component")
                or item.get("candidate_component")
                or item.get("service")
                or item.get("name")
                or ""
            ).strip()
        else:
            name = str(item).strip()
        if name and name not in components:
            components.append(name)
    return components


def rank_of_ground_truth(ground_truth: str, ranking: list[str]) -> int | None:
    if not ground_truth or not ranking:
        return None
    try:
        return ranking.index(ground_truth) + 1
    except ValueError:
        return None


def hit_at_k(case: dict[str, Any], k: int) -> bool | None:
    ground_truth = str(case.get("ground_truth") or "")
    ranking = case.get("predicted_ranking_list") or []
    if isinstance(ranking, str):
        try:
            ranking = json.loads(ranking)
        except json.JSONDecodeError:
            ranking = []
    ranking = normalize_ranking(ranking)
    if ranking:
        return ground_truth in ranking[:k]
    if k == 1:
        return bool(case.get("hit@1"))
    if case.get("hit@1"):
        return True
    return None


def metric_hits(cases: list[dict[str, Any]], k: int) -> tuple[int, int, float | None, int]:
    if not cases:
        return 0, 0, None, 0
    hits = 0
    missing = 0
    for case in cases:
        value = hit_at_k(case, k)
        if value is None:
            missing += 1
            value = False
        if value:
            hits += 1
    return hits, len(cases), hits / len(cases), missing


def avg_at_5(cases: list[dict[str, Any]]) -> tuple[float | None, int]:
    if not cases:
        return None, 0
    missing = 0
    values = []
    for k in range(1, 6):
        _, _, ac, miss = metric_hits(cases, k)
        missing += miss
        values.append(ac or 0.0)
    return sum(values) / 5.0, missing


def ndcg_at_k(case: dict[str, Any], k: int) -> float | None:
    rank = case.get("rank_of_ground_truth")
    try:
        rank_int = int(rank)
    except (TypeError, ValueError):
        if case.get("hit@1") and k >= 1:
            rank_int = 1
        else:
            return None
    if rank_int <= k:
        return 1.0 / math.log2(rank_int + 1)
    return 0.0


def status_category(status: str, error: str = "") -> str:
    text = f"{status or ''} {error or ''}".lower()
    if "timeout" in text:
        return "timeout"
    if error or not (status or "").strip() or "error" in text or "failed" in text:
        return "failed"
    if "shortcut" in text:
        return "shortcut"
    return "egcda"


def fault_type_from_case_id(case_id: str) -> str:
    parts = case_id.split("/")
    if len(parts) < 2:
        return ""
    fault_name = parts[1]
    if "_" in fault_name:
        return fault_name.rsplit("_", 1)[-1]
    return fault_name


def fmt_float(value: float | None, digits: int = 6) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def pct(value: float | None) -> str:
    if value is None:
        return "TODO"
    return f"{100.0 * value:.1f}%"


def compact_text(value: Any, limit: int = 700) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    text = " ".join(text.split())
    if len(text) > limit:
        return text[: limit - 3] + "..."
    return text
