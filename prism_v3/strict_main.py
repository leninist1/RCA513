"""Strict no-leakage entrypoint for PRISM v3.

Use this module for benchmark inference and score reporting:

    python -m prism_v3.strict_main --systems Bank --prism-noise-lab

It validates every externally supplied prior/model/profile before delegating to
the existing runner.  The legacy ``prism_v3.main`` remains available only for
reproducibility comparisons.
"""
from __future__ import annotations

import os
import sys
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from . import main as legacy_main
from .leakage_guard import UntrustedArtifactError, validate_external_artifact


SCORE_OPTION = "--prism-noise-lab-scores"
ARTIFACT_OPTION_TYPES: Dict[str, Tuple[str, ...]] = {
    SCORE_OPTION: (
        "no_leak_ltr_scores",
        "no_leak_noiselab_scores",
        "no_leak_runtime_scores",
    ),
    "--v2-learned-embedding-path": ("no_leak_learned_embedding",),
    "--v2-learned-likelihood-path": ("no_leak_learned_likelihood",),
    "--v2-learned-classifier-path": ("no_leak_learned_classifier",),
    "--v2-entity-profile-path": ("no_leak_entity_profile",),
}
FORBIDDEN_BARE_FLAGS = {
    # These convenience flags enable learned modules without proving that all
    # dependent artifacts were fitted on a certified split.
    "--v2-all",
    "--v2-learned",
}


def _option_value(argv: Sequence[str], option: str) -> Optional[str]:
    for idx, item in enumerate(argv):
        if item == option:
            if idx + 1 >= len(argv):
                raise SystemExit(f"missing value for {option}")
            return argv[idx + 1]
        if item.startswith(option + "="):
            return item.split("=", 1)[1]
    return None


def validate_strict_argv(argv: Sequence[str]) -> Dict[str, object]:
    """Validate strict-mode CLI inputs before any inference code is imported.

    Per-query out-of-fold validation for score CSVs is repeated later inside
    ``NoiseLabEvidenceAdapter`` once the query id is known.
    """
    argv = list(argv)
    forbidden = sorted(flag for flag in FORBIDDEN_BARE_FLAGS if flag in argv)
    if forbidden:
        raise UntrustedArtifactError(
            "strict mode rejects convenience learned flags without explicit "
            "certified artifacts: " + ", ".join(forbidden)
        )

    validated: Dict[str, object] = {}
    for option, allowed_types in ARTIFACT_OPTION_TYPES.items():
        path = _option_value(argv, option)
        if not path:
            continue
        manifest = validate_external_artifact(
            path,
            allowed_types=allowed_types,
            current_query_id="",
        )
        validated[option] = manifest.to_debug()

    enabled_flags = set(argv)
    if "--v2-learned-embeddings" in enabled_flags and "--v2-learned-embedding-path" not in validated:
        raise UntrustedArtifactError("--v2-learned-embeddings requires a certified embedding artifact")
    if "--v2-learned-likelihood" in enabled_flags and "--v2-learned-likelihood-path" not in validated:
        raise UntrustedArtifactError("--v2-learned-likelihood requires a certified likelihood artifact")
    if "--v2-learned-classifier" in enabled_flags and "--v2-learned-classifier-path" not in validated:
        raise UntrustedArtifactError("--v2-learned-classifier requires a certified classifier artifact")
    if "--v2-entity-profiles" in enabled_flags and "--v2-entity-profile-path" not in validated:
        raise UntrustedArtifactError("--v2-entity-profiles requires a certified entity-profile artifact")
    return validated


def main() -> None:
    validated = validate_strict_argv(sys.argv[1:])
    os.environ["PRISM_STRICT_NO_LEAKAGE"] = "1"
    if validated:
        print("Validated strict no-leakage artifacts:")
        for option, manifest in sorted(validated.items()):
            print(f"  {option}: {manifest}")
    legacy_main.main()


if __name__ == "__main__":
    main()
