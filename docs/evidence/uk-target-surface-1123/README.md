# Evidence for the UK target surface change (microcosm#1123, PR #1198)

Two receipts of the compiled UK target surface, written by
`tools/uk_target_surface_receipt.py`, and what they show. The receipts carry
every bound row (6.5 MB and 7.4 MB), so they stay outside the tree under
`data/ukds/acceptance/1123-pr-a/receipts/`. The two extracts here are written
from them by `scripts/extract_surface_receipts.py`.

## The two receipts

- **Before:** `before-aca40a692.json`. The merge base `aca40a692` with only the
  commit that adds the receipt tool (it moves no value), on the feed that
  commit pins, Chronicle `825406f`. Review date 2026-10-10.
- **After:** `after-bc32a49c1.json`. Branch commit `bc32a49c1`, on the feed it
  pins, Chronicle `942a1fa`, with `--communal`. Review date 2026-10-12.

Both use the 2021 output-area ladder of the 2026-10-04 local build.

**Each receipt records what its working tree added to the commit it names**
(`code_modifications`): every file under `packages` and `tools` that differs
from `code_head`, with the SHA-256 of its content, and the diff of the tracked
ones. The extracts keep the file list and the diff's digest.

- **After** ran on a clean tree: no file differs from `bc32a49c1`. The hold
  register is signed, so the compile needs no overlay.
- **Before** names six files. `aca40a692` does not carry the receipt tool, so
  the tool's commit was laid over it, uncommitted, with the provenance record
  added to the tool. The receipt holds that 33 kB diff.

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
  taxpayers by area), none removed, 18,811 moved. The file gives each target
  at each grain: cells, totals before and after, and the spread of the
  relative moves.
  - 437 of the moved cells move only because a deferred authority no longer
    leaves its share on its neighbours: the council-tax band A to G cells of
    London and the South West (406) and Scotland's band H cells (31). The
    widest move is 0.5%.

The national count is of bound rows, after the measure exclusions. The
reference register is larger (1,231 active references before, 1,267 after).

## `reconciliation.json`

How the after surface was reconciled.

- **`national_reconciliation.band_bridges`:** the 24 bridges from the UK
  Table 2.5 bands to the SPI regional cells. Each carries its factor, its
  declared `max_factor_shift`, and `explained_factor`: Table 3.11's United
  Kingdom row over the regional cells' sum, which the factor must sit within
  half a point of. 21 factors equal it to five decimals; the widest gap is
  0.19 points. The widest factor is income tax at £12,570 to £15,000, 1.0573
  against a bound of 0.063.
- **`national_reconciliation.sum_bridges`:** the UK household composition
  scaled onto the nations' sum (29,003,000 to 29,082,020).
- **`fanout_sum_bridges`:** each HMRC area bridge at each local grain, read
  before any reconciliation. Both parts of the factor the cells take are
  gated.
  - `source_factor` is the bands over the cells before either side's
    uprating index: 0.999 to 1.006 at constituency grain and 1.003 to 1.008
    at local-authority grain, against 2% (`max_factor_shift`).
  - `index_factor` is the gap between the two sides' indices: 0.997 to 1.014,
    each inside its bridge's `max_index_shift`.
  - `applied_factor` is their product, 0.996 to 1.019.
  - `off_surface_cells` counts the frame's rows for cells that are not bound;
    the totals include them.
- **`off_surface_cells`:** the signed cells that are off the surface and
  still count in their leg. 44 frame rows: 38 deferred cells with a published
  value, and 6 implied for cells the publisher's table does not carry. Torbay
  and the Isles of Scilly are implied at 2.3% of the South West's employees;
  their share of its households is 2.6%. `areas_outside_the_ladder` is empty.
- **`cross_grain_groups`:** every group's factor by leg. The local-authority
  legs under the constituency totals (`constituency_over_la`) are 1.018 to
  1.021 on employment counts and 1.008 to 1.011 on employment amounts. The
  North East's 1.029 on self-employment amounts is the two tables' own gap
  at source: that leg is whole at both tiers.
- **`subset_relations`:** each declared part against its whole on the compiled
  values. All 20 tests hold and none is skipped. The nine age bands and the
  two 90-and-over rows are tested on their sum, which equals the UK total.
- **`partitions`, `uprating_holds`, `communal_validation`:** counts and
  spreads.

## Reproducing

```
uv run --no-sync python tools/uk_target_surface_receipt.py \
    --ledger-facts <chronicle consumer artifact at the pinned commit> \
    --ladder <uk_oa_ladder_2021.npz> --review-date 2026-10-12 \
    [--communal] --out <receipt.json>
python docs/evidence/uk-target-surface-1123/scripts/extract_surface_receipts.py \
    <before.json> <after.json> docs/evidence/uk-target-surface-1123
```

Neither input is in the repository: the consumer artifact comes from
Chronicle's `build-bundle --suite uk` and `build-consumer-artifact` at the
commit `uk/chronicle_feed.json` pins, and the ladder from a UK local build.
