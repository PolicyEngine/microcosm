# Progress: 25% replacement-candidate host runbook

## State

The `candidate-25pct-runbook` lane is auditing the current two-stage US build
chain and preparing an exact, off-chain, dry-run-validated host script. This
lane will not run either builder, publish, promote, push, tune, touch the
pending logbook chain, or exceed 15 GiB RSS.

The required US-extra environment is complete and relinked to this worktree.
The default cache was not writable and the first task-cache retry could not
resolve PyPI through disabled DNS. Recovery cloned a known-good venv whose
`uv.lock` SHA-256 exactly matches this branch, provisioned cached Hatch editable
build requirements, and completed `uv sync --offline --inexact
--no-build-isolation --all-packages --extra us`. The final Microcosm and
PolicyEngine import check passed.

## Done

- Read `CLAUDE.md` and recorded the PR-CI/certification boundary, root-journal
  convention, no-side-effect publication rule, and workspace commands.
- Read the GitNexus exploration skill selected for the requested release-flow
  audit. This session exposes no GitNexus repository resources or query tools,
  so its direct-source fallback applies.
- Confirmed the clean branch `candidate-25pct-runbook` starts at
  `d69131a3`, identical to `origin/main` (post-#748).
- Completed the prescribed US-extra sync offline without changing the
  lockfile; all five workspace packages now point at this worktree.
- Established this committed current-state journal before implementation.

## Next

- Trace both builders, release documentation, preflight gates, scorer CLI, and
  all stage-2 argument/validation paths with exact source citations.
- Inventory and hash every required host input, stopping rather than inventing
  any missing required pin.
- Write the serial off-chain host script, add dry-run validation, capture the
  committed dry-run receipt, update `_LANE-NOTES.md`, and write the final report
  to the requested output location.
