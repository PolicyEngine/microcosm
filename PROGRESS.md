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
- Audited the canonical scoring seam. The existing fiscal scorer already
  normalizes a legacy PolicyEngine flat H5 and a Microcosm entity H5 before a
  shared population-repair, materialization, weighting, and scoring path
  (`tools/score_us_fiscal_targets.py:240-365,432-489`). The new scorer will
  compile the registry once and fail if materialization or the constraint
  matrix omits any row (`tools/build_us_fiscal_refresh_release.py:4323-4347`;
  `packages/microcosm-calibrate/src/microcosm/calibrate/matrix.py:286-355`).
- Pinned the loss yardstick to the production concept-budget weighting: square
  root of target scale within amount/count basis, concept-group budget caps,
  equal total budget across present bases, mean normalization, and the existing
  100% capped target-scaled loss
  (`tools/build_us_fiscal_refresh_release.py:5781-5814,6214-6290`;
  `packages/microcosm-calibrate/src/microcosm/calibrate/solve.py:473-518`).
- Verified the eCPS incumbent bytes and identity. PolicyEngine.py 4.15.0's
  bundled US manifest resolves
  `policyengine/policyengine-us-data@9531fe1d.../enhanced_cps_2024.h5` with
  SHA-256 `0a6b961a...`; Microcosm's frozen parity pin uses immutable revision
  `21280dca...` for the same bytes
  (`policyengine/data/release_manifests/us.json@4.15.0:12-27,37-42`;
  `packages/microcosm-build/src/microcosm/build/us/ecps_parity_reference.json:7-14`).
- Classified the terminal battery. Its 131 marginal comparisons plus one joint
  comparison require ASEC/ACS support-channel and clone-role provenance
  (`packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:11366-11378,11523-11533,11577-11580`).
  A finished pool's input-only H5 cannot replay them; the authenticated
  companion manifest carries the sealed receipt
  (`tools/build_us_multispine_pool.py:3533-3550,3766-3786`;
  `packages/microcosm-build/src/microcosm/build/us_runtime/h5_io.py:348-430,672-811`).
- Compiled the superseding v9.4 Ledger feed (`b3c08356...`) without scoring:
  the one surface has 32,842 specs and registry version `c4ac617743f2`.

## Next

- Finish the complete workspace baseline (`uv run python -m pytest`).
- Implement and test the common head-to-head scorer.
- Check the build queue, score the incumbent, record the owner command and
  comparison doctrine, run final verification, and update `FINAL_REPORT.md`.
