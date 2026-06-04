"""Write default executable Scheme B rules as JSON."""
from __future__ import annotations

from pathlib import Path

from refute_b_v2.default_rules import default_rule_set


def main() -> int:
    out = Path("knowledge/refutation_rules_v2.json")
    default_rule_set().save_json(out)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
