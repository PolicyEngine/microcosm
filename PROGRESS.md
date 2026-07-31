# Progress

## State

PR #583 round-1 HOLD remediation is in progress on
`multispine-pool-build-578`. The four blocking seams are the raw-stage producer
boundary, crash-safe pool publication, pandas string-read guard coverage, and
the deprecated-but-supported local ACS release producer.

## Done

- Confirmed the worktree starts clean at review head `a20e847`.
- Read `CLAUDE.md` and the full adversarial review at
  `_buildo-runtime/reviews/sol_583.log`.
- Selected the GitNexus debugging and impact-analysis workflows for lineage
  tracing and blast-radius checks.
- Attempted to build the GitNexus index; the managed lane denied its global
  registry write, so direct source/call-site/test tracing is the active
  fallback. Removed the generated untracked local cache.
- Extended the spine-blindness visitor over pandas string APIs and committed
  the exact reviewer `.get()` and `.query()` mutations plus `.eval()`,
  `.filter(items=...)`, and `.loc[...]` regressions in `884c255`. The focused
  guard suite passes (11 tests) and Ruff is clean.
- Established the required final constraint: this live checkpoint will be
  restored byte-for-byte to `origin/main` before handoff so no root journal
  ships in the PR diff.

## Next

- Finish tracing the producer/pool and legacy-release artifact contracts.
- Add each reviewer repro as a failing regression before its implementation
  fix.
- Fix and commit the remaining three blockers in coherent steps.
- Run the full `populace-build` suite and Ruff, write the external handoff, and
  restore `PROGRESS.md` to `origin/main`.
