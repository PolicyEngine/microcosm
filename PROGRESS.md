# Progress: microcosm #722

## State

Dual-review remediation is in progress on `passive-pass-through-722` as of
2026-08-21. The real merge of already-fetched `origin/main` at `2c7a7218` is
committed as `d9f19c66`, with the supervisor-approved resolutions reproduced
exactly. The post-merge resource closure and first-class typed passive contract
are regenerated and their fail-closed coverage ledger is verified. Work
remains offline: no fetch, push, PR mutation, publication, or promotion is in
scope.

## Done

- Read the binding adversarial verdict before inspecting or changing the
  branch.
- Re-read `CLAUDE.md` and confirmed the PR-CI/certification and root-journal
  contracts for this remediation pass.
- Attempted the GitNexus debugging workflow; no GitNexus query/context tools
  are configured in this session, so direct source and call-path analysis is
  the documented fallback.
- Confirmed the branch, detached pre-approved merge worktree, merge base, and
  current `origin/main` identities without network access.
- Resolved the only three merge conflicts with the approved typed country
  resource rows, both independent multispine imports, and one measured runtime
  graph pin. The three resolved file blobs match the pre-approved detached
  index exactly.
- Ran the post-merge source-blindness graph test itself with repository
  conftests disabled because the pre-sync environment does not yet contain
  main's new `jsonschema` dependency. The single registered tool reaches the
  pinned 67-module graph (base 64, branch +2, main +1).
- Traced the country-package closure failure to
  `tools/generate_us_bundle_from_constants.py::LEGACY_RESOURCE_PATHS`; the
  loader's `_assert_file_closure` compares that emitted manifest with package
  files.
- Committed the real merge of current `origin/main` as `d9f19c66`; its second
  parent is `2c7a7218` and it is not a rebase or squash.
- Declared the three passive/repeal JSON resources on
  `LEGACY_RESOURCE_PATHS`, completed the full generator run, and ran the exact
  offline `uv sync --all-packages --all-extras` equivalent against the locked
  environment.
- Traced the remaining constants-adapter failure to the passive contract being
  present only as generated pool code. Added it to the typed pipeline schema,
  compiler identity contract, and stacked-authority projection, then regenerated
  the bundle at spec SHA-256
  `11b5f830141d007b53131f6c863d15a6be10787fc6d88aeb147317d2c1f593e0`.
- Independently reviewed the changed field-usage claims: 62 new pointers total
  (29 authored, 33 resolved), all accounted for by the three typed manifest
  rows and the typed passive pipeline/pool-code contract.
- Regenerated the F0 evidence report with exact 41,442/41,442 field coverage
  and 40/40 inventory checks. The targeted adapter, schema/identity,
  field-ledger, inventory, and coverage suite passes 45 tests; the US generator
  check and targeted Ruff checks are also clean.
- The implementation and verification bullets below describe the historical
  pre-review checkpoint at `a4d93f78`; the verdict supersedes its former
  completion claim.

- Read `CLAUDE.md` and established the PR-CI/certification and journal
  contracts.
- Confirmed the primary checkout has unrelated untracked paths and left it
  untouched.
- Created the requested isolated worktree at
  `.claude/worktrees/passive-722` from current `origin/main`.
- Attempted the GitNexus exploration workflow; no GitNexus MCP resources or
  tools are configured, so source and Git-history inspection are the fallback.
- Confirmed the old QBI v3 stack is opt-in, diverges before the repository
  rename, and does not expose its latent entity form on the output frame.
- Selected a sibling passive assignment stage before current QBI
  reconciliation. This preserves the archived 15-leaf QBI contract and all
  prior QBI random streams while still accepting a latent form for routing
  when one becomes available.
- Reproduced the six SCF Schedule-E-band cells from the local 2022 public
  extract. Presence cells all clear effective n=30; the three middle-band
  conditional-share cells fall back to the pooled holder sample.
- Confirmed the restricted PUF replay artifact is present and pinned its
  size, digest, row counts, weighted positive pass-through aggregate, and
  provisional Form 8960 midpoint calibration target.
- Added the reproducible provisional SCF evidence resource. Holding
  prevalence rises from 3.35% in the nonpositive Schedule-E band to 38.98%
  above $1 million; conditional-share cells in the three thin middle bands
  use the documented all-holder fallback.
- Added the version-1 sibling assignment and assumptions build. The persisted
  log-odds shift is `-1.157105426398319`; its expected aggregate is the
  $54.628492 billion midpoint and the seed-0 replay produces $55.021132
  billion (0.72% high, inside the 5% diagnostic tolerance).
- Verified the restricted replay (12 focused tests), deterministic assumptions
  regeneration, strict resource hashes, isolated PCG64 families, latent-form
  routing, and byte preservation of all 15 existing QBI leaves.
- Integrated the sibling between Schedule-D completion and QBI reconciliation
  in multispine and direct support builds, while preserving already-assigned
  rebuilt support. The 997-row remaining-stage audit and support/pool
  checkpoint identities now bind the evidence, assumptions, shift, and RNG.
- Added the new output to ownership, L0 export, and release-coverage contracts.
  The exact pending-engine exception is limited to the input arriving in
  PolicyEngine-US PR #9306 and becomes a no-op once that engine version lands.
- Added separate, diagnostics-only Form 8960 line 12 and line 4c aggregate
  rows. They cannot affect the release gate and retain row-local error
  containment under the currently locked pre-#9306 engine.
- Completed the exact final-tree workspace suite: 6,394 passed and 73 skipped
  in 42m45s. The restricted artifact replay separately passed all 12 tests.
- Completed repository-wide `ruff check .`, changed-file Ruff format checks,
  `git diff --check`, the spec-only/entrypoint guards, and a manual incumbent
  package-name sweep over the branch diff. The manual sweep is required because
  the ordinary tree guard deliberately excludes `.claude/` worktrees.
- Added and committed the towncrier fragment and wrote the completion report
  to `FINAL_REPORT.md`.

## Next

- Replace the vacuous preservation proxy with a nonzero-realization regression
  that executes the current-main QBI pipeline and byte-compares every incumbent
  QBI column before and after the passive stage.
- Fail the release-input gate when a required manifest variable is absent from
  the locked engine registry; cover both absent-variable and all-zero-required
  failure modes and document the 1.764.6/#9306 engine-inert boundary honestly.
- Make restricted certification independently re-solve and compare the
  calibration shift.
- Run targeted and full workspace tests with the prescribed restricted-data
  environment, run Ruff, commit regenerated artifacts, and write the final
  report to `FINAL_REPORT.md`.
