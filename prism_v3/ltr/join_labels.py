"""Training-only label join for no-GT feature tables.

This module is intentionally separate from strict inference. It may read
``record.csv`` because it is only used after label-free feature generation.
"""

from __future__ import annotations

import argparse
from typing import Any, Dict, List

import pandas as pd

from ..data.loader import OpenRCALoader
from ..noise_native.evidence_frame import canonical_entity_name


def join_labels(frame: pd.DataFrame, *, system: str) -> pd.DataFrame:
    out = frame.copy()
    out["label"] = 0
    out["training_component"] = ""
    loader = OpenRCALoader(system)
    query_lookup: Dict[tuple[str, int], Any] = {}
    for sub in loader.sub_systems:
        for idx, query in enumerate(loader.load_queries(sub)):
            query.query_index = idx
            query_lookup[(sub, idx)] = query
    label_cache: Dict[tuple[str, int], List[str]] = {}
    for idx, row in out.iterrows():
        sub = str(row.get("sub_system", ""))
        query_index = int(row.get("query_index", -1))
        key = (sub, query_index)
        if key not in label_cache:
            query = query_lookup.get(key)
            if query is None:
                label_cache[key] = []
            else:
                gts, _inject = loader.match_query_to_records(query)
                label_cache[key] = [str(gt.component or "") for gt in gts if gt.component]
        components = label_cache.get(key, [])
        out.at[idx, "training_component"] = ",".join(components)
        obj = canonical_entity_name(row.get("object_id", ""))
        out.at[idx, "label"] = int(any(canonical_entity_name(component) == obj for component in components))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Join training labels after label-free feature generation")
    parser.add_argument("--features", required=True)
    parser.add_argument("--system", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    frame = pd.read_csv(args.features)
    out = join_labels(frame, system=args.system)
    out.to_csv(args.output, index=False)
    print(args.output)


if __name__ == "__main__":
    main()
