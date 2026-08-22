# Progress: replacement scorecard

## State

The `replacement-scorecard` lane is active at `2aa96795`, based on the
post-#741 `origin/main`. The task is to build one head-to-head scoring path for
the live US incumbent and the not-yet-built 25% bundle-mode candidate, score
the incumbent now, and leave the exact candidate command for the owner.

The locked US-extra environment is synchronized. No pool build, scoring run,
push, gate change, threshold change, or band change has occurred.

## Done

- Read `CLAUDE.md` and the GitNexus exploration skill. GitNexus repository
  resources are not exposed in this session, so the code-flow audit will use
  direct source and call-site inspection.
- Attempted the required `uv sync --all-packages --extra us`. The managed
  sandbox denied writes to the default uv cache and has no network/DNS. A
  sibling Microcosm worktree at the identical commit and with the identical
  `uv.lock` supplied a copy-on-write environment; a narrow writable cache of
  the locked build requirements then allowed
  `uv sync --offline --all-packages --extra us` to rebuild and relink all five
  editable workspace packages to this worktree.
- Confirmed the starting worktree was clean and the branch tracks
  `origin/main` without any local changes.
- Historicized the inherited one-target-surface root journal before recording
  current state.
- Established a green journal-step baseline: the scorer-signature suite
  (`uv run python -m pytest -q
  packages/microcosm-build/tests/test_us_state_files_scorer.py`) passes, and
  `uv run ruff check .` passes. The complete workspace suite is continuing in
  a separate read-only runner.

## Next

- Finish the complete workspace baseline (`uv run python -m pytest`).
- Trace the fiscal scorers, compiled registry, artifact loaders, terminal
  battery, and packaged PolicyEngine-US dataset identity.
- Implement and test the common head-to-head scorer.
- Check the build queue, score the incumbent, record the owner command and
  comparison doctrine, run final verification, and update `FINAL_REPORT.md`.
