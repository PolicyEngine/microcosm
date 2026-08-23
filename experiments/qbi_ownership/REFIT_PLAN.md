# Coupled QBI amount-surface refit plan

The ownership replay narrows the defect but does not justify a guessed model
change. All four incidence failures and UBIA QED are transfer-first; BDC,
REIT/PTP, and W2 also carry producer-side magnitude defects. All four targets
are `zero_inflated_positive` in all four realized availability patterns. The
safe path is therefore a coupled producer-and-transfer refit evaluated through
whole-pool reconciliation, not an isolated terminal calibration.

No gate, band, ceiling, fold, seed, or exclusion is changed by this plan.

## Safe work completed in this lane

1. **Persist complete origin and regime evidence.** Both transfer fit paths
   recompute regimes from frozen donor support, compare them with the QRF, and
   retain the complete pattern map
   (`packages/microcosm-build/src/microcosm/build/us_runtime/acs_transfer.py:1237-1242,1631-1651,1705-1885,1908-2206`).
   Target origins distinguish `preexisting`, deterministic derivation, and
   `qrf_transfer`, with an exact model-target binding
   (`packages/microcosm-build/src/microcosm/build/us_runtime/acs_transfer.py:561-631,634-723`).

2. **Close the publication receipt route.** The stacked validator now binds
   the exact donor selection, clone, recipient complement, producer roles,
   row accounting, group/aggregate/execution copies, predictor catalog, and
   sibling model-target regimes
   (`packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:5072-5198,5201-5754,8766-8994`).
   The ready-H5 loader authenticates both the early gap-fill and late-DAG
   receipts (`packages/microcosm-build/src/microcosm/build/us_runtime/h5_io.py:594-640`).

3. **Keep unrelated calibration off the QBI amount surface.** The current
   nine-target post-transfer calibration policy is asserted disjoint from all
   four QBI amounts
   (`packages/microcosm-build/tests/test_us_post_transfer_calibration.py:71-87`).
   This prevents a selective calibration from masquerading as the coupled
   QBI refit.

4. **Make ownership reproducible.** The regression requires the same nonempty
   origin across the group, aggregate, and signed execution receipts for all
   four amounts, and proves deletion of `origin.channel` from any copy fails
   validation
   (`packages/microcosm-build/tests/test_us_stacked_spine.py:6581-6720`).
   Fully rehashed donor-route, producer-role/count, predictor-catalog, and
   sibling-catalog forgeries are also rejected
   (`packages/microcosm-build/tests/test_us_stacked_spine.py:6723-6849`).

These are receipt, validation, and regression changes. They do not alter the
amount model and therefore do not claim to improve a battery margin.

## Host-owned 1% diagnostic/refit demonstration

The frozen host baseline is
`/Users/maxghenis/PolicyEngine/_buildo-runtime/out/battery-verify/baseline1pct/pool.gates.json`,
SHA-256
`1d6059868680f872fe04d452a536bcc3c215bafabb4c50d7740a469fe6a8b56a`.
It has all eight QBI amount checks red and both old clone-0 SSTB comparisons
dead. This lane was explicitly forbidden from starting an after-build, so it
does not present a before/after claim.

The host 1% run must add diagnostic evidence without changing seeds or
thresholds:

1. For every QBI target, channel, and realized availability pattern, record
   eligible recipients, donor rows, predicted carrier probability, realized
   carrier outcome, conditional amount quantiles, and chained prior state.
   Availability patterns are deterministic partitions of the recipient
   feature surface (`packages/microcosm-build/src/microcosm/build/us_runtime/acs_transfer.py:2335-2408`).

2. Record the exact reconciliation exposure bases beside BDC and REIT/PTP
   draws. The current kernel caps BDC by non-qualified dividends and REIT/PTP
   by non-qualified dividends plus positive partnership/S-corporation income
   (`packages/microcosm-build/src/microcosm/build/us_runtime/qbi_inputs.py:1342-1359`).

3. Compare donor, clone-1 producer, raw clone-0 draw, and reconciled terminal
   surfaces by target × channel × pattern. That decomposition must identify
   whether each incidence change comes from pattern composition, the
   zero/positive gate, chained conditioning, or exposure reconciliation.

4. Refit the smallest owner-supported coupled surface. A valid candidate must
   preserve frozen donor rows, observed producer cells, target order, and seed
   protocol. Because three magnitude defects already exist on clone 1, the
   candidate must evaluate the producer and transfer together; a recipient-
   only amount map cannot establish closure.

5. Run whole-pool reconciliation and require all nine invariants to return to
   zero. The production kernel and invariant definitions are at
   `packages/microcosm-build/src/microcosm/build/us_runtime/qbi_inputs.py:1228-1374,1377-1487`.

6. Rerun the same eight terminal amount checks plus the two clone-1 SSTB
   checks. The comparator must retain the registered role and unchanged
   incidence/QED rules
   (`packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:13891-13893,14006-14391,14441-14555,14598-14643`).

Acceptance at 1% is structural: complete authenticated receipts, zero residual
nulls, zero nine-invariant violations, no regression outside the coupled QBI
surface, and directionally credible held-out diagnostics. The sample is too
small to certify rare-carrier margins, so a green 1% result is not a release
decision.

## What requires the 25% host run

Only after the 1% candidate and its diagnostic receipt are fixed should the
host run the adjudication-scale 25% build. That run must:

- authenticate all stage, bank, late-DAG, terminal-gate, and publication
  receipts under the new authority;
- rerun all eight QBI amount checks and the exact-two SSTB checks on the full
  terminal by-origin surface;
- rerun the nine coupled invariants after reconciliation;
- compare non-QBI battery legs and full fiscal/structural gates against the
  frozen incumbent; and
- remain `simulation_ready=false` unless every ordinary publication condition
  independently passes.

No 25% certification, amount-model refit, pool publication, or logbook-chain
operation was performed in this lane.
