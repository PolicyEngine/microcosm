# Round 8 progress: cross-origin structural absence

## State

The smoke-r5 mechanism is fixed, the requested proof matrix is green, and the
independent review found no actionable issue. `TYPEHUGQ` now uses `acs_source`
on both the input and the primary structural output; ASEC rows remain absent
with no synthesized value or tolerated-absence receipt. The complete declared
raw-input audit has no cross-origin whole-pool reads. This branch is ready for
smoke r6.

## Done

- Read `CLAUDE.md` and the applicable debugging workflow.
- Confirmed the requested branch and exact starting commit.
- Compared the checkout with the locally cached `origin/main`. The branch is
  75 commits ahead and 15 behind that cache; the requested checkout is being
  preserved because this round explicitly targets PR #660 at `27b07c73` and
  forbids network access.
- Confirmed the GitNexus graph tools are unavailable in this session; the
  equivalent call-site and execution-flow trace will be performed directly
  from source, checkpoint receipts, and tests.
- Read the smoke-r5 launcher, error receipt, chained logbook spool row, assembly
  manifest, and checkpoint H5 without modifying them.
- Proved the exact checkpoint partition: 17,004 households; all 1,688 ASEC
  households have absent `TYPEHUGQ`; all 15,316 ACS households have populated
  codes 1/2/3 (13,421 / 904 / 991). The missing mask exactly equals the ASEC
  household-origin mask.
- Traced schema alignment as the source of the legitimate ASEC nulls and the
  blanket `whole_pool` inventory conversion as the source of the false gate.
- Confirmed every semantic `TYPEHUGQ` read is restricted to ACS households.
  The ACS earnings-universe lineage does not consume it; that producer uses
  ACS person channel, age, WAGP, and SEMP.
- Chose origin scoping over an absence receipt. The primary structural output
  must carry the same ACS scope so post-callback completeness and all 19
  downstream transfer dependencies remain consistent.
- Added the per-requirement scope override, serialized it into late-registry
  identity schema v14, and scoped all 39 physical `TYPEHUGQ` contract
  occurrences to ACS rows.
- Retired the latent whole-pool raw fallbacks `RELSHIPP`, `TEN`, and
  `H_TENURE`; transfer execution identity schema v2 now binds only the
  canonical dual-origin head and tenure predictors.
- Enforced an enumerated raw-origin audit over 101 physical occurrences:
  43 ACS-scoped and 58 ASEC-scoped, with no whole-pool occurrence.
- Added regressions proving exactly 1,688 ASEC `TYPEHUGQ` nulls pass without a
  receipt or fill while one missing ACS value refuses before its callback.
- Updated the operator-ordering schema pins and the changelog.
- Passed the focused suite: 31 tests, zero failures/skips/errors.
- Preserved #583 at exactly 495 passed, with zero skips/failures/errors.
- Partitioned all 225 workspace test files into eight foreground chunks with
  zero duplicates, omissions, or extras. Exact aggregate: 5,909 passed, 68
  skipped, zero failures/errors (5,977 collected). Per-chunk pass/skip counts:
  711/2, 562/21, 779/4, 1,029/2, 773/2, 765/1, 495/0, and 795/36.
- Resolved one collection-only environment gap without network or repository
  mutation by exposing the locked cached `pyarrow==25.0.0` archive to chunks
  that import it; the complete affected chunk was rerun and passed.
- Passed repository-wide `ruff check .`, scoped `ruff format --check` for all
  four changed Python files, and `git diff --check 27b07c73..HEAD`. The
  repository-wide formatter baseline still names 29 unrelated pre-existing
  files; none is part of this round.
- Independent read-only review found no actionable correctness bug,
  regression, or missing required test. Residual maintenance note: future
  ASEC raw inputs must be added to the intentionally explicit provenance list.

## Next

- Run the externally launched 1% smoke r6 (build execution is explicitly out
  of scope for this round). Predict 15,316 ACS `TYPEHUGQ` rows in scope with
  zero unfilled/invalid rows, 1,688 untouched ASEC structural nulls out of
  scope, no `TYPEHUGQ` absence receipt, and successful progress beyond the
  former primary-readiness refusal through the full smoke train.
