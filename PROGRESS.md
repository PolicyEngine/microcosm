# Progress

## State

PR #583 round-10 fix lane is in progress on
`multispine-pool-build-578` from `9e9acf5`. The review HOLD identifies four
mechanical closure gaps in the spine-blindness structural guard: nested-star
payload poisoning, non-name loop targets, static-dict `.values()` iteration,
and partial structures propagated through bindings.

## Done

- Confirmed the requested branch, clean worktree, and exact starting commit.
- Read `CLAUDE.md`, the GitNexus debugging skill, and the authenticated
  round-10 review log.
- Recorded the required implementation, regression, documentation, precision,
  suite, lint, graph-cleanliness, and no-push constraints.
- Confirmed the GitNexus CLI is installed but the repository has no index; a
  non-augmenting index attempt was blocked by the managed global-registry
  write, and its generated untracked local index was removed.
- Reproduced all six authenticated forms as silent at the starting guard:
  stale-outer nested star, attribute target, subscript target, literal
  `.values()`, constructor `.values()`, and a bound mixed row.
- Added those exact inputs as self-tests, with loop-diagnostic assertions for
  unpropagatable geometry and exact-column assertions for successful
  per-column propagation. The focused red run fails all three test groups as
  expected before the implementation.
- Reworked static-row binding to report recognized and fully propagated
  status separately. Nested star payload names are all poisoned, partial and
  refused targets now enter the loop fallback regardless of flattened-string
  resolution, and fragment state is captured before target bindings mutate
  the scope.
- Added static `.items()`/`.values()` iteration resolution for literal,
  bound, and supported `dict(...)` mappings, with value-only syntactic
  fragment probing so guarded-looking mapping keys do not taint benign
  `.values()` loops.
- Preserved partial list/tuple structures at assignment bind time, allowing
  bound mixed rows to propagate their static columns exactly.
- The 121-case guard and focused 22-case benign/runtime/graph battery pass.
  Direct scans report zero findings for both `acs_transfer.py` and
  `congressional_district_vintage.py`; file-scoped ruff and formatting pass.
- Completed the docstring truth pass: the module contract now names the
  supported scalar, structural-row, and static dict-view forms; helper
  docstrings distinguish exact propagation, deliberate opacity, and partial
  geometry; the round-9 claim now states the actual iteration-site fallback.
- Initial final validation passed (guard 121, focused graph 22, full
  `populace-build` 3,343 passed/85 skipped, repository ruff), but the
  independent net-diff audit found two precision defects before handoff:
  partial Name/non-name targets retained stale outer constants, and opaque
  constructor keys could collapse through the singleton sentinel.
- Added red controls for both audit findings plus an every-name nested-star
  opacity check. The two precision controls fail on the current implementation
  as expected; the nested-star check already passes.
- Corrected partial-row handling so direct-name positions still receive their
  exact column choices while unsupported subtargets are poisoned and keep the
  overall binding partial. This removes stale outer constants without
  weakening the fragment-bearing loop fallback, including star-only payloads.
- Narrowed `dict(iterable)` resolution to fully resolved entries; any opaque
  key/value now refuses construction instead of collapsing distinct runtime
  keys through the shared sentinel, leaving the syntactic value-only fragment
  fallback to catch guarded content.
- The guard (121), focused graph battery (22), explicit `acs_transfer.py` and
  `congressional_district_vintage.py` scans, and file-scoped ruff pass again.
- The last composition audit then identified three related cases not covered
  by the exact reviewer inputs: scalar static `.values()` unnecessarily
  entered the loop fallback, known strings disappeared from mixed columns,
  and partial dict/constructor views were not retained through bindings.
- Added red precision/composition controls for literal, bound, and constructor
  scalar views; inline/bound mixed columns; and bound partial dict views.
- Added an abstract ordered dict-entry representation: fully known keys retain
  normal dict overwrite semantics, while opaque keys remain distinct possible
  rows instead of collapsing through the sentinel. Partial literal and
  constructor mappings now survive assignment bindings and feed their actual
  keys, items, or values to iteration.
- Static iteration now retains known top-level string choices for scalar views
  and known string members within mixed row columns. Empty views preserve
  empty-loop flow, all-opaque columns remain opaque, and the documented
  conservative duplicate-key over-catch applies only to unresolved keys.
- The expanded controls, 121-case guard, 22-case graph battery, and direct
  production-file scans pass with both named files still at zero findings.

## Next

- Re-run the full validation matrix after those corrections.
- Restore `PROGRESS.md` exactly to `origin/main`, commit all coherent steps
  locally without pushing, and write `/private/tmp/583_fix5_handoff.md`.
