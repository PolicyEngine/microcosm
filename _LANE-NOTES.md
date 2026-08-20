# Battery package 3 lane notes

## Scope and frozen boundary

This lane owns 16 adjudicated FIX-CANDIDATE checks: 13 in
`source_operator_two_part_calibration`, two in
`adult_care_post_reconciliation`, and one in
`model_required_targeted_calibration`. The assigned baselines and named
remedies are the rows in the read-only reference
`../microcosm-arm-split/experiments/battery_burndown/adjudication.json`.

No comparator, band, threshold, seed, fold, or sample contract may change. All
builds in this lane are off-chain at `--sample-fraction 0.01` and
`--sample-seed 578`; they omit `--logbook-prev-row-digest` and do not touch
`logbook-pending-chain.txt`.

## Source-cited mechanism record

- The battery computes positive and negative carrier incidence separately,
  compares ACS/ASEC weighted incidence to the frozen band, then computes five
  weighted conditional carrier quantiles when both sides have enough rows
  (`packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:11860-11930`).
- Its quantile-envelope diagnostic is the maximum symmetric normalized
  separation across those five conditional quantiles
  (`packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:11990-12021`).
- ACS transfer partitions each family by recipient optional-predictor
  availability and fits separately seeded QRFs on complete donor rows
  (`packages/microcosm-build/src/microcosm/build/us_runtime/acs_transfer.py:896-916`).
- The banked transfer path writes each availability-pattern QRF raw draw into
  recipient positions before target decoding
  (`packages/microcosm-build/src/microcosm/build/us_runtime/acs_transfer.py:1581-1648`).
- Adult-care expenses are reconciled after transfer, only for transfer-filled
  cells; the reconciliation is explicitly documented as qualifying-carrier,
  at-most-one-per-tax-unit structure
  (`packages/microcosm-build/src/microcosm/build/us_runtime/acs_transfer.py:1100-1117,1205-1227`).

These code paths support artifact-side carrier and conditional-amount
calibration. They do not justify changing the comparator.

## Environment receipt

- `uv sync --all-packages --extra us`: failed before resolution because the
  sandbox denied writes to `/Users/maxghenis/.cache/uv`.
- Writable-cache retry: failed downloading `pandas==3.0.3` because sandbox DNS
  is unavailable.
- Exact-lock fallback: `uv.lock` SHA-1
  `6b213e740b114d008c0191fa492832a957a0a948` matches
  `../microcosm-707/uv.lock`; that environment imports NumPy 2.4.6, pandas
  3.0.3, and pytest 8.4.2 while `PYTHONPATH` points at this worktree.
- Initial deterministic contract suite: 4 passed (battery sign separation,
  matching-leg pass behavior, adult-care statute reconciliation, and declared
  transfer-surface coverage). Repository-wide and three-module attempts were
  interrupted after clean partial progress because fitted-model cases make
  them unsuitable for a journal-only checkpoint; complete verification is
  reserved for the final tree.

## Measurement ledger

Before and after values will be recorded here after their respective 1% builds.
