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
- Added paired CLI options for an override path and raw-file SHA-256. The
  driver authenticates the complete policy before starting stacked work,
  rejects either half alone, and rejects any override on the legacy route.
- Threaded the typed policy through cold and resumed stacked execution into
  the fresh terminal battery. It is deliberately excluded from input pins and
  simulation checkpoint identity, while the resulting report remains in the
  sealed terminal receipt and final manifest.
- Replayed the fresh terminal battery from the authenticated saved 1% simulated
  checkpoint (sample seed 578) at 1,544,634,368 bytes peak RSS. The manifest
  validator accepted the new committed empty-policy report.
- Recorded the honest per-check result: both assigned checks remain
  `dead_comparison` at ASEC/ACS incidence 0/0 and rows 0/0, with no exclusion
  applied. This is unchanged by design because approving either exclusion is
  explicitly outside this lane and the shipped register is empty.

## Next

- Run the full suite, lint, and packaging checks; finalize code-cited notes and
  the required output report.

## Historical note

The previous contents described the completed mortgage donor outlier-screen
lane. That merged-session journal was historical on this branch as of
2026-08-19; repository history retains it at parent commit `164027e2`.
