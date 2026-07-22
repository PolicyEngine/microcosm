# Progress

## State

Populace #462 critical-row loss alignment is in progress on
`loss-contract-alignment`, starting from clean `origin/main` at `c3e378a`.
The work will align the builder's critical-target register with the publish
contract and add a configurable post-normalization loss boost for those rows.
No target values or existing gate semantics will change beyond register
coverage.

## Done

- Confirmed the worktree is clean on `loss-contract-alignment`.
- Confirmed `HEAD` exactly matches `origin/main` at `c3e378a`.
- Replaced the prior merged-task progress record with this task's baseline.

## Next

- Trace the publish-contract requirements, builder register, loss-weight
  construction, CLI options, diagnostics, and build record.
- Add the anti-drift register-coverage test.
- Implement and test the critical-row loss multiplier.
- Run the requested targeted tests and Ruff, then write `FINAL_REPORT.md`.
