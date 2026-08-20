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

- The fitted regime first draws a weighted sign gate and then a forest amount
  conditional on the selected sign (`packages/microcosm-fit/src/microcosm/fit/qrf.py:950-1003,1333-1429`). This is the two-part mechanism targeted here.
- The transfer fills only recipient nulls and reconstructs the Frame without a
  by-origin marginal constraint (`packages/microcosm-build/src/microcosm/build/us_runtime/acs_transfer.py:1012-1098`). The stacked owner then verifies donor byte identity and residual-null accounting (`packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:8097-8189`).
- Positive weeks-unemployed draws are zeroed where unemployment compensation
  is not positive, and the source gate rejects PUF-role positives without that
  predictor (`packages/microcosm-build/src/microcosm/build/us_runtime/weeks_unemployed.py:966-979,1201-1276,1333-1337`). Carrier additions must therefore stop at that compatible capacity.
- Adult-care reconciliation changes only transfer-filled expense cells,
  requires a qualifying person, and allows at most one surviving carrier per
  tax unit (`packages/microcosm-build/src/microcosm/build/us_runtime/acs_transfer.py:654-713,1205-1227`). Calibration must select qualifying mutable carriers before the final reconciliation.
- Prior-year self-employment reconciliation carries signed ASEC source values,
  so only the positive ACS leg is mutable and the negative leg must remain
  byte-identical (`packages/microcosm-build/src/microcosm/build/us_runtime/prior_year_income.py:829-887,902-985`).
- The late producer registry assigns the child-support, disability, weeks,
  workers'-compensation, energy, and adult-care surfaces to ASEC-scoped source
  producers (`packages/microcosm-build/src/microcosm/build/us_runtime/operator_boundary.py:277-294,331-333`; `packages/microcosm-build/src/microcosm/build/us_runtime/us_late_producer_registry.py:1377-1454`). The late calibration may therefore change only exact ACS clone-0 transfer recipients.

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

The before build used `--sample-fraction 0.01 --sample-seed 578`, clone fraction
1 and clone seed 578, with no chain predecessor. Peak observed per-process RSS
was 10.733 GiB. Artifact receipts:

- `pool.h5`: SHA-256 `258891504201275f8006a1584b7d3e891890d15381724bc9f2b30b1f443d967f`
- `pool.gates.json`: SHA-256 `94ee914bb7490e7f513184e691cf15847d1e585693ef184977196485f86f1fee`
- `pool.manifest.json`: SHA-256 `953aaff72fac8cc17211959d0133f9bce8c22dd87b2957a0f6a89335fcc9c122`

The exact frozen-battery values are below. `unsupported` means the battery did
not emit QED because one side had fewer than five carriers; the parenthesized
number is the same five-grid weighted diagnostic computed manually without
changing that support rule.

| Assigned check | Before at 1% |
| --- | ---: |
| adult care positive incidence ratio | 0.561425035 |
| adult care positive QED | unsupported (manual 1.738865343) |
| unemployment compensation positive QED | 0.352941176 |
| child-support expense positive incidence ratio | 0.171280844 |
| child-support expense positive QED | 0.953846154 |
| child support received positive incidence ratio | 0.242882414 |
| child support received positive QED | 1.000000000 |
| disability benefits positive QED | 1.373534621 |
| prior-year self-employment positive incidence ratio | 1.376752468 |
| prior-year self-employment positive QED | 0.834720589 |
| weeks unemployed positive incidence ratio | 0.025384419 |
| weeks unemployed positive QED | 0.736842105 |
| workers' compensation positive incidence ratio | 0.047614655 |
| workers' compensation positive QED | unsupported (manual 1.918367347) |
| SPM-unit energy subsidy positive incidence ratio | 0.240477284 |
| SPM-unit energy subsidy positive QED | 0.666666667 |

Sparse 1% sampling also puts the otherwise frozen unemployment-compensation
and disability positive carrier ratios at 0.131859569 and 0.098471480. Their
adjudicated remedies freeze carrier membership, so this lane will improve only
their assigned amount-QED checks. The unassigned negative prior-year
self-employment ratio is 1.435953092 and is likewise deliberately untouched.

After values will be added after the calibrated 1% rebuild.
