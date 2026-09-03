# PR #847 gate round 4 progress

## State

In progress on the standalone `fix-847-r5` lane at the requested starting
head. The frozen interface and all named source surfaces have been read; the
ten final-round findings are now under focused reproduction. No implementation
change has been made yet.

## Done

- Read the repository operating instructions and the GitNexus debugging
  workflow. GitNexus query tools are not exposed in this lane, so source and
  focused-test traces will provide the debugging evidence.
- Confirmed `HEAD` matches `HEAD_SHA.txt`; the standalone clone's local branch
  is named `fix-847-r5`.
- Preserved the historical repository-root journals. This lane journal lives
  under `docs/graph-847-r5/` to comply with the explicit root-journal boundary.
- Read `docs/graph-acceptance.md`, including amendments 11--17 and the
  interface freeze; `docs/graph-interface.lock`; all seven named graph modules;
  the acceptance burndown tool; both named fit modules; and the H-parity test.
- Confirmed amendment 17's `NumericScope` interface already exists in frozen
  `kernel.py`, while executor context construction still supplies only the old
  tolerance projection.
- Reproduced finding 3 with five red parametrized cases: annotated
  `pytestmark`, an assigned skip alias, and all three `unittest.skip*`
  decorators returned no suppression problem. The scanner now rejects every
  decorator on a collected test that it cannot prove is an allowed direct
  pytest mark, and recognizes annotated/augmented module `pytestmark`
  assignments. The complete burndown-tool unit file and focused Ruff pass.
- Reproduced finding 5 through the graph store: `POPULACE_FIT_N_JOBS=1` and
  `=2` produced the same node/artifact identities but pickle bytes differed at
  byte 798. `_Forest` now serializes a shallow model copy with canonical
  `n_jobs=1`, leaves the live fitted model untouched, and restores the current
  runtime setting on trusted unpickle. The cache-collision regression and all
  fit-kernel tests pass.

## Next

- Reproduce each finding against the starting head before implementing it.
- Add one regression per finding, make small named commits, run the complete
  required verification block, and write the lane report to the requested
  output file.
