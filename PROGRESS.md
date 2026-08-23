# Progress: 25% replacement-candidate host runbook

## State

The `candidate-25pct-runbook` lane is auditing the current two-stage US build
chain and preparing an exact, off-chain, dry-run-validated host script. This
lane will not run either builder, publish, promote, push, tune, touch the
pending logbook chain, or exceed 15 GiB RSS.

The required `uv sync --all-packages --extra us` was attempted first. The
default cache is not writable in the managed sandbox; a retry with a writable
task cache reached the locked `policyengine-core==3.26.11` wheel but could not
resolve PyPI because network access is disabled. An offline host-cache recovery
is the next environment step; source and host-input inspection can proceed
read-only in parallel.

## Done

- Read `CLAUDE.md` and recorded the PR-CI/certification boundary, root-journal
  convention, no-side-effect publication rule, and workspace commands.
- Read the GitNexus exploration skill selected for the requested release-flow
  audit. Its graph-resource availability is still to be checked.
- Confirmed the clean branch `candidate-25pct-runbook` starts at
  `d69131a3`, identical to `origin/main` (post-#748).
- Attempted the prescribed US-extra sync without changing the lockfile.
- Established this committed current-state journal before implementation.

## Next

- Recover or verify the locked environment offline, and query the GitNexus
  index if it is available.
- Trace both builders, release documentation, preflight gates, scorer CLI, and
  all stage-2 argument/validation paths with exact source citations.
- Inventory and hash every required host input, stopping rather than inventing
  any missing required pin.
- Write the serial off-chain host script, add dry-run validation, capture the
  committed dry-run receipt, update `_LANE-NOTES.md`, and write the final report
  to the requested output location.
