# B7 implementation progress

## State

The entrant-person strata implementation, cache attestation, mass receipt, and
lineage presentation coverage are complete. The sanctioned tool flipped
exactly B7 and both suite pins report zero red. A final review found and fixed
receipt ID-width restoration; the required verification matrix is being rerun
against that fix.

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
- Ran `tools/graph_acceptance_flip.py`; it reported only
  `test_b7_entrant_persons_carry_their_stratum` and `flipped 1 property`.
- Updated the burndown total to zero and the explorer summary to 45 green,
  zero red. The focused post-flip tests report `4 passed in 1.03s`, and the
  burndown verifier reports `verification=ok` with `red        0  none`.
- Fixed warm replay for non-`int64` person IDs by rehydrating JSON receipt IDs
  through the base ID dtype, matching lineage restoration. The new `int32`
  regression passes as part of `30 passed in 0.73s` for population tests.
- Made cached entrant-label attestation type-strict so JSON-distinct labels such
  as `1`, `1.0`, and `true` cannot substitute for one another, with bool/float
  tamper regressions. Also limited entrant-triggered mass authorship to declared
  entrant EXPAND nodes so unrelated receipts cannot activate it. The affected
  focused suites report `61 passed in 2.02s`.
- Strengthened that attestation to canonical scalar equality, which also
  distinguishes `-0.0` from `0.0`; the signed-zero replay joins the tamper
  matrix and the affected suites now report `62 passed in 2.02s`.

## Next

- Rerun the complete graph shard twice after the ID-width replay fix (standard
  repository addopts, then the explicit count configuration).
- Run the cross-shard kernel/parity selection, lint, formatting, CI grouping,
  and final burndown verification.
- Record exact summaries and the final branch SHA in `out.md`.
