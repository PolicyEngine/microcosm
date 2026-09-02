# B7 implementation progress

## State

The entrant-person strata implementation, cache attestation, mass receipt, and
lineage presentation coverage are complete. B7 passes with `--runxfail`; the
strict marker and pre-implementation suite pins remain until the sanctioned
flip.

## Done

- Created the requested isolated worktree from `graph-amendments-impl`.
- Merged `graph-amendments-11-13`, including amendment 14, the frozen
  `KernelResult.strata` field, and the strict-red B7 fixtures and test.
- Resolved the suite-pin conflicts in favor of the amendment branch's
  pre-implementation four-red checkpoint; the pins will move to zero only
  after the sanctioned flip.
- Validated `KernelResult.strata` against exactly the null-lineage person ids,
  including id/label dtypes, uniqueness, completeness, and non-null labels.
- Materialized copied-person and entrant-person strata separately, while
  allowing new labels and preserving the copied-person path.
- Kept entrant membership overlays intact during copied-person remapping so an
  entrant can join an incumbent or entrant group.
- Added the executor-authored `entrant_strata` receipt as ordered `[id, label]`
  pairs, verified it against the stored frame on cached replay, and exposed it
  through the existing receipt-based view and explorer.
- Added executor-authored mass receipt data for entrant-person EXPAND nodes;
  the existing household-weight broadcast now counts entrant persons.
- Added population validation/replay tests and a cold/warm explanation test.
- Focused result: `58 passed in 2.38s`; B6/C5/D6 result:
  `3 passed in 0.93s`.

## Next

- Run the complete graph shard with B7 still strict-red and inspect any
  regressions.
- Run the sanctioned B7 marker flip, update the two zero-red suite pins, and run
  every requested verification command.
