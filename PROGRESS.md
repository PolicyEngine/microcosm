# B7 implementation progress

## State

Amendment 14 and its strict-red B7 acceptance test are merged onto the
implemented amendments 11--13. Implementation and validation are in progress
on branch `impl-b7` in the dedicated `/private/tmp/microcosm-impl-b7` worktree.

## Done

- Created the requested isolated worktree from `graph-amendments-impl`.
- Merged `graph-amendments-11-13`, including amendment 14, the frozen
  `KernelResult.strata` field, and the strict-red B7 fixtures and test.
- Resolved the suite-pin conflicts in favor of the amendment branch's
  pre-implementation four-red checkpoint; the pins will move to zero only
  after the sanctioned flip.

## Next

- Trace EXPAND patching, mass-ledger, cached-replay, and lineage presentation.
- Implement and test entrant-person strata without editing frozen interfaces or
  acceptance fixtures.
- Run the sanctioned B7 marker flip, update the two zero-red suite pins, and run
  every requested verification command.
