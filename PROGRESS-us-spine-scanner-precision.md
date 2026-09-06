# US spine scanner precision — lane progress

Resolve the known opacity false positives in the US source-spine static scanner
without weakening source-spine protection. Base: fresh child of
`cd971ff9b7140e9dfa54c82f4b59325b69eb6f8a`, worktree
`_worktrees/us-spine-scanner-precision-cd97`, branch
`us-spine-scanner-precision-20260905`.

Editable surface: `packages/microcosm-build/tests/test_us_spine_blindness.py`
plus changelog/documentation. No production edits, no owner exemptions, no
file/line/function allowlists, no added production annotations.

This lane keeps its progress in its own file. The repository's tracked
`PROGRESS.md` belongs to the F1 portable-worker-identity lane and is left
byte-identical to `cd971ff9`.

## State

- [x] Fresh child worktree at `cd971ff9`
- [x] Read the supported static-indirection contract and adversarial tests
- [x] Narrow inference implemented (three selector shapes + closed getattr set)
- [x] Alias-laundering closure added (container status survives a proven
      row/entry selection) — strictly stronger than the base scanner
- [x] Paired fixtures: 5 new tests, benign allowed / spine + unresolved refused
- [x] Full suite engine-free and network-guarded: 503 tests, 1 failure
- [x] CI group verification `verification=ok`, ruff check + format clean
- [x] Whole-tree base-vs-new finding delta captured (110 opacity findings
      removed, 0 named-column findings removed, 0 added)
- [x] Report written

## Outcome

`us_runtime` scan findings 21 -> 6; offending modules 7 -> 5.
`test_runtime_population_operators_are_source_spine_blind` still fails on one
residual shape: a dict lookup on an unannotated parameter keyed by an
unresolved loop variable. Not forced green. See
`us-spine-scanner-precision-REPORT.md` for the residual list and the minimal
alternative.
