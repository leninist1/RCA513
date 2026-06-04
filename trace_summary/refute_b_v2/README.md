# refute_b_v2

Independent implementation workspace for the Scheme B core refactor.

This folder is intentionally separate from the earlier `refute/` package. The
first version focuses on the new backbone:

- abstract evidence signatures
- executable refutation rules
- memory/network v0 rules
- bounded trace summaries
- rule audit hooks

The old `refute/` package remains untouched until this package has focused
tests and a stable integration path.
