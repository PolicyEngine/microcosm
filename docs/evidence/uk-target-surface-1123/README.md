# Evidence for the UK target surface change (microcosm#1123, PR #1198)

Two receipts of the compiled UK target surface, written by
`tools/uk_target_surface_receipt.py`, and what they show. The receipts carry
every bound row (6.5 MB and 7.4 MB), so they stay outside the tree under
`data/ukds/acceptance/1123-pr-a/receipts/`. The two extracts here are written
from them by `scripts/extract_surface_receipts.py`.

## The two receipts

- **Before:** `before-aca40a692.json`. The merge base `aca40a692` with only the
  commit that adds the receipt tool (it moves no value), on the feed that
  commit pins, Chronicle `825406f`.
- **After:** `after-5ad4816b5.json`. Branch commit `5ad4816b5`, on the feed it
  pins, Chronicle `942a1fa`, with `--communal`.

Both use review date 2026-10-10 and the 2021 output-area ladder of the
2026-10-04 local build.

**The after receipt was run over one local change.** An unsigned hold is not
in force, so the compile refuses the committed register
(`uk/uprating_holds.json` has `approved_by: null` on all 16 entries). For the
run only, every entry was marked approved by "UNSIGNED: local receipt run
only, never committed". The receipt records the file under
`code_modified_paths`. Signature fields move no value.

## `surface-diff.json`

What the surface moves between the two receipts.

- **National, bound rows:** 1,167 before and 1,203 after. 36 added, none
  removed, 552 moved.
  - Added: the 12 nation household control cells and the 24 Income Tax
    liabilities Table 2.2 marginal-rate cells.
  - Moved by this change's own reconciliation: 360 SPI regional cells (the
    band bridges) and 24 CGT regional cells.
  - Moved by the re-pin and the household controls: 149 population rows, the
    10 household-composition rows, 3 land rows, 3 SLC repayment rows, and one
    row each for DfI, ISC and public sector employment.
- **Local cells:** 20,885 before and 21,804 after. 919 added (higher rate
  taxpayers by area), none removed, 18,374 moved. The file gives each target
  at each grain: cells, totals before and after, and the spread of the
  relative moves.

The national count is of bound rows, after the measure exclusions. The
reference register is larger (1,231 active references before, 1,267 after).

## `reconciliation.json`

How the after surface was reconciled.

- **`national_reconciliation.band_bridges`:** the 24 bridges from the UK
  Table 2.5 bands to the SPI regional cells, each with its factor and its
  declared `max_factor_shift`. The widest is income tax at £12,570 to £15,000,
  1.0573 against a bound of 0.063.
- **`national_reconciliation.sum_bridges`:** the UK household composition
  scaled onto the nations' sum (29,003,000 to 29,082,020).
- **`fanout_sum_bridges`:** each HMRC area bridge at each local grain, read
  before any reconciliation. `source_factor` is what the 2% gate tests;
  `applied_factor` is what the cells take; `index_factor` is the part of it
  that comes from the two sides' different uprating indices.
- **`cross_grain_groups`:** every group's factor by leg.
- **`subset_relations`:** each declared part against its whole on the compiled
  values. All 20 tests hold.
- **`partitions`, `uprating_holds`, `communal_validation`:** counts and
  spreads.

## Reproducing

```
uv run --no-sync python tools/uk_target_surface_receipt.py \
    --ledger-facts <chronicle consumer artifact at the pinned commit> \
    --ladder <uk_oa_ladder_2021.npz> --review-date 2026-10-10 \
    [--communal] --out <receipt.json>
python docs/evidence/uk-target-surface-1123/scripts/extract_surface_receipts.py \
    <before.json> <after.json> docs/evidence/uk-target-surface-1123
```

Neither input is in the repository: the consumer artifact comes from
Chronicle's `build-bundle --suite uk` and `build-consumer-artifact` at the
commit `uk/chronicle_feed.json` pins, and the ladder from a UK local build.
