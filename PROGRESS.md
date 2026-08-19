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

## Next

- Add the provisional SCF evidence resource and persisted calibration
  assumptions.
- Implement and integrate the isolated passive assignment stage while proving
  prior QBI outputs are byte-identical.
- Add the diagnostics-only Form 8960 validation rows and restricted replay.
- Run restricted replay if the declared PUF artifact is present, then run the
  full workspace tests, lint, contract sweeps, and write the final report.
