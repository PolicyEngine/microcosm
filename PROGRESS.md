# Battery package 3 progress

## State

The host 1% verification build rejected the implementation committed at
`33bf52fe`. Terminal stacked gap-fill validation reports an invalid ACS QRF
pattern record binding for
`person/puf_tax_itemization/taxable_interest_income`, which is outside the
assigned calibration rows. The continuation is therefore in diagnosis; the
suite-only green result is not sufficient and no after artifact is accepted.
The uncontaminated 1% before artifact remains recorded at commit `5f5e5e91`.
No frozen battery band, threshold, comparator, seed, or fold has changed.

The required `uv sync --all-packages --extra us` was attempted first. The
default cache is sandbox-read-only; a retry with a writable cache reached PyPI
but DNS is unavailable. Verification therefore uses the already-synced
`microcosm-707` environment, whose `uv.lock` SHA-1 matches this worktree, with
`PYTHONPATH` pinned to this worktree's five package source directories.

## Done

- Read `CLAUDE.md`, the adjudication and all assigned machine-readable rows,
  citation registry entries C04-C08, and every cited source range.
- Built and measured the canonical off-chain 1% before artifact with sample
  seed 578, clone fraction 1, clone seed 578, no predecessor digest, and no
  pending-chain file mutation.
- Added a declared nine-target positive-leg policy. It preserves unemployment
  and disability carriers, matches the other assigned positive carrier
  margins, maps mutable positive amounts to exact reference support at the
  frozen five quantiles, and byte-preserves negative, negative-zero,
  zero-weight, and nonmutable values.
- Bound production application to exact ASEC clone-0 reference rows, ACS
  clone-0 recipients, and transferred-null mutable cells. Adult-care additions
  are limited to qualifying people and one candidate per empty tax unit;
  positive weeks additions require positive unemployment compensation.
- Reused the adult-care qualifying predicate for both calibration and the
  final reconciliation, which must be a verified no-op after calibration.
- Recorded and validated exact donor-support QRF regimes for every ordinary
  and banked ACS availability-pattern fit without changing fit seeds, folds,
  estimator counts, or draw behavior.
- Added schema-v2 calibration receipts with explicit terminal-versus-
  generation verification boundaries. Terminal validation independently
  replays live masks, row identities, output bytes, weights, carrier metrics,
  conditional quantiles, and coupled constraints; adversarial fully rehashed
  scope and diagnostic forgeries are rejected.
- Changed the two pinned SIPP chunk readers to streaming type inference. This
  leaves downstream explicit numeric coercion and locked output facts intact
  while reducing the 3.73 GB donor-test peaks from above the safety ceiling to
  0.49 GiB and 0.53 GiB.
- Verified all 225 `microcosm-build` test files green, split into fresh pytest
  processes where needed. Also verified `microcosm-fit` (93 passed),
  `microcosm-calibrate` (201 passed), `microcosm-frame` (294 passed, 36
  skipped), and `microcosm-data` (275 passed, one skipped).
- Ran repository-wide `ruff check .`, touched-file `ruff format --check`, and
  `git diff --check` successfully. Repository-wide `ruff format --check .`
  still identifies 49 pre-existing, mostly unrelated files; none was
  reformatted as part of this lane.
- Kept final verification processes below 10 GiB with a 10 GiB/20 ms guard.
  During diagnosis, the earlier 13.5 GiB/250 ms guard observed one rapid SIPP
  parser spike at 15.424 GiB before terminating it; the reader fix and all
  successful reruns remained far below the cap. This exception is retained in
  the journal rather than concealed.
- Recorded the owner-provided host failure at the exact
  `_validate_acs_imputed_pattern_evidence` raise site. The failing target is
  outside this lane's assigned rows, so shared gap-fill behavior must be
  restored before another artifact build.

## Next

1. Trace construction and terminal replay of the ACS QRF pattern record
   binding and identify how post-transfer calibration affects an unassigned
   target.
2. Add a regression test that reproduces the host binding rejection and proves
   unassigned gap-fill targets remain byte- and receipt-identical.
3. Narrow the implementation to the assigned source-operator, adult-care, and
   model-required targets; run the focused and full PR suite.
4. Rebuild off-chain at exactly 1% with sample/clone seed 578 under the
   tightened memory guard when host data access is available, then record the
   16 after measurements and source-preservation invariants.
