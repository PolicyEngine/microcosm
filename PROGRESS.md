# Progress: US reviewed-exclusion machinery

## State

The empty schema-2 register, strict loader, and battery/runtime integration are
implemented on `battery-excl-machinery`, which started from `origin/main` at
`164027e2`. The assigned scope remains machinery only: no band, threshold,
seed, fold, comparator, or exclusion approval has changed.

## Done

- Confirmed the worktree started clean on the named branch.
- Read `CLAUDE.md`, the required adjudication report, and the machine-readable
  adjudication rows for the two reviewed-exclusion candidates.
- Started line-by-line inspection of every cited US mechanism and UK template
  module, plus read-only integration and verification audits.
- Attempted the required `uv sync --all-packages --extra us`. The default uv
  cache is not writable in the managed sandbox; a retry with a private cache
  reached resolution but cannot download `hf-xet` because sandbox DNS is
  disabled. Local-cache recovery is in progress.
- Attempted the GitNexus exploration workflow; no GitNexus MCP resources or
  tools are configured in this session, so discovery is using direct source
  and test call sites.
- Read every module and line range cited by the assigned adjudication rows,
  including the cited UK snapshot and the refactored equivalents on current
  `main`.
- Captured the existing 1% terminal receipt for both assigned checks: each is
  `dead_comparison`, with ASEC/ACS incidence 0/0 and nonzero rows 0/0.
- Confirmed a cold 1% build would violate the lane's 15 GiB RSS ceiling; the
  existing run records primary-QRF peaks above 80 GiB. The after measurement
  will therefore re-run fresh terminal gates from its saved simulated frame.
- Recovered a runnable locked US environment by cloning the already-synced
  adjudication-worktree venv and rebinding its editable paths locally. The
  original exact sync command was still attempted first and remains
  network-blocked as recorded above.
- Baseline battery/manifest tests pass: 8 targeted tests.
- Added the declared, packaged schema-2 US by-origin exclusion resource with
  an intentionally empty `exclusions` object.
- Added a US-specific immutable receipt/policy loader: exact
  `leg#criterion` keys, exact five-field receipts, canonical date validation,
  duplicate/non-finite JSON rejection, canonical policy digest, and
  path-plus-SHA override authentication with partial-arming refusal.
- New schema/resource tests pass: 27 loader tests and 5 country-package tests;
  Ruff check is clean and both new Python files are Ruff-formatted.
- Integrated exact criterion IDs into the existing comparator paths without
  changing their bands, thresholds, status records, or human failure text.
- Added in-force application and stale/dormant/expired/premature anti-rot
  classification. Only an exact in-force failing criterion is suppressed;
  structural invalidity and unevaluated criteria remain unsuppressed.
- Added an independent terminal-manifest validator that reconstructs criterion
  failures from comparison evidence, verifies the canonical policy digest and
  all five receipt fields, rejects inconsistent lifecycle buckets, and permits
  a passing `dead_comparison` only when its exact `#dead` receipt was applied.
- Added runtime tests covering the two assigned dead checks, all eligible
  comparator criteria, lifecycle boundaries, stale/dormant behavior, exact
  isolation, and post-evaluation receipt tampering.

## Next

- Wire authenticated path-plus-SHA overrides through the pool CLI, refusing
  partial arming and the legacy two-spine route before build work.
- Run the affected/full suites, lint, packaging checks, and the required 1%
  after measurement from the saved terminal checkpoint.

## Historical note

The previous contents described the completed mortgage donor outlier-screen
lane. That merged-session journal was historical on this branch as of
2026-08-19; repository history retains it at parent commit `164027e2`.
