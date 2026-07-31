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

## Next

- Propagate direct-name columns while poisoning unsupported partial targets,
  and reject opaque dict-constructor entries before materializing a mapping.
- Re-run the full validation matrix after those corrections.
- Restore `PROGRESS.md` exactly to `origin/main`, commit all coherent steps
  locally without pushing, and write `/private/tmp/583_fix5_handoff.md`.
