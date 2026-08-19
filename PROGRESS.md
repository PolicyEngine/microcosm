# Progress: microcosm #722

## State

Implementation is in progress on `passive-pass-through-722`, based on
`origin/main` at `9c20c1d2`. The verified #530 QBI branch chain was inspected
and is not used as a base because its tip still has the retired pre-rename
namespace and package layout. Only the required QBI v3 concepts will be
ported into the current `microcosm.*` tree.

## Done

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

## Next

- Finish integration of the passive sibling into every base/pool path and its
  checkpoint, ownership, and release-coverage contracts.
- Add and verify the diagnostics-only Form 8960 validation rows.
- Run the full workspace tests, lint, contract sweeps, and write the final
  report.
