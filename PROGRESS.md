# Progress: microcosm #722

## State

Implementation is in progress on `passive-pass-through-722`, based on
`origin/main` at `9c20c1d2`. The verified #530 QBI branch chain was inspected
and is not used as a base because its tip still has the retired `populace.*`
namespace and `packages/populace-*` layout. Only the required QBI v3 concepts
will be ported into the current `microcosm.*` tree.

## Done

- Read `CLAUDE.md` and established the PR-CI/certification and journal
  contracts.
- Confirmed the primary checkout has unrelated untracked paths and left it
  untouched.
- Created the requested isolated worktree at
  `.claude/worktrees/passive-722` from current `origin/main`.
- Attempted the GitNexus exploration workflow; no GitNexus MCP resources or
  tools are configured, so source and Git-history inspection are the fallback.

## Next

- Map the minimal QBI v3 resource, assumptions-build, assignment, RNG, and
  validation surfaces onto the renamed tree.
- Derive the SCF conditional cells from the pre-verified local inputs and add
  the provisional evidence resource.
- Implement and test simulation version 4 while preserving v1-v3 byte output.
- Run restricted replay if the declared PUF artifact is present, then run the
  full workspace tests, lint, contract sweeps, and write the final report.
