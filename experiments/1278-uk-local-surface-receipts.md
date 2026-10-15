# microcosm#1278: UK local surface, receipts for the first PR

This receipt gives no counts or weighted figures from the licensed spines. Census and publisher figures are public statistics and appear as published.

## C1. Council-tax bands stay inside their nation

**Method.** Two spines were built from the licensed inputs with the same recipe and seed, local only:
- main at `121fd4dd6`;
- the branch at `78de5f7db` (the band restriction and its raw-score redraw).

They hold the same set of base households (households that are not CGT copies), named by lineage (source household, support channel, clone index, income-band donor). The comparison script is `docs/evidence/uk-local-surface-1278/scripts/compare_twin_bands.py`; its outputs stay with the licensed spines, outside the repository.

**Result.**
- **On main**, SPI-channel households in England and Scotland held band I, and SPI-channel households in Northern Ireland held bands other than the placeholder D.
- **On the branch**, no household holds a band its nation does not use.
- **Every band that changed** was outside its nation on main, and no valid band moved. No FRS-channel household's band changed: the shell draws housing only for SPI-channel households and refuses to modify an FRS row.
- **Downstream:** columns that later stages draw for each household from its housing or wealth (the council-tax amounts, the WAS wealth, the LCFS spending, the ETB VAT rate and the NTS travel) move on few households. Columns that a later stage rakes or scales over a whole cell (household weights, energy use, road fuel and the totals that contain them) move a little on more households, because a few moved households change the cell's factor. The number of CGT copies also differs, because the copy stages condition on housing and wealth.

Weighted households by band and nation are not reported for either spine: the difference between the two spines' totals would isolate the households the fix moved.

**The first attempt refused on licensed data.** The first version renormalised `predict_proba` over the nation's bands. For some English households the classifier scores band I so far above A-H that its softmax gives A-H probability exactly zero, and the stage refused. The redraw now takes a softmax of the raw class scores over the nation's bands: the same conditional distribution, without the underflow (`78de5f7db`).

## C2. Signed small-cell rules

Both rules in `uk/local_small_cell_rule.json` are signed: approved by juaristi22 on 2026-10-14, expiring 2027-04-01. `tools/generate_uk_local_target_references.py` regenerated the local references on the pinned feed (`942a1fa`). The cells are publisher target cells, not survey records.

| Status of a local cell | main | This branch |
| --- | --- | --- |
| Active (bound) | 21,804 | 21,596 |
| Signed deferral | 342 | 550 |
| No fact for the area | 1,239 | 1,239 |
| Signed deferral, other compile error | 1 | 1 |

The 208 new deferrals are exactly the cells the two rules name:
- `council_tax_band_household_floor`, published value below 50 dwellings: 16 local-authority cells. Band G 3 (MHCLG 1, Scottish Government 2), band H 7 (Scottish Government 5, Welsh Government 2), band I 6 (Welsh Government).
- `hmrc_rounded_thousand_floor`, published value below 5,500: 192 higher rate taxpayer cells, 155 constituencies and 37 local authorities.

No deferral was removed, and no other status moved. Each deferred cell still counts in its region's or nation's control through reconciliation.

The local compile-parity receipt (`uk/ledger_compile_parity_local_incumbent_2025_signed_differences.json`, regenerated with `tools/build_uk_ledger_compile_parity_signed_differences.py --surface local`) now compares 21,596 compiled cells with the incumbent's 23,545:
- 207 of the 208 deferred cells had no incumbent row, so the ledger-only differences fall from 2,525 to 2,318;
- the other one has an incumbent row, so it moves from calibration drift (19,276 to 19,275) to fixture-only (4,266 to 4,267).

## C7. Sizing the age-universe change from census data

`tools/uk_target_surface_receipt.py --communal-only` on the pinned feed (`942a1fa`). These are public census figures:
- ONS RM120 for England and Wales;
- NRS UV101a for Scotland;
- NISRA CT0105 for Northern Ireland, approximate.

Under the planned change, a band's private-household target sits below the incumbent's total-resident row by the band's communal share. The release's incumbent evaluation gates that row at 25%.

| Band | Constituencies at 20%+ | at 25%+ | Local authorities at 20%+ | at 25%+ | Median constituency share |
| --- | --- | --- | --- | --- | --- |
| 10-20 | 32 | 19 | 9 | 3 | 0.8% |
| 20-30 | 6 | 2 | 3 | 2 | 0.9% |
| 80 and over | 0 | 0 | 0 | 0 | 7.3% |
| Every other band | 0 | 0 | 0 | 0 | under 1.2% |

- Northern Ireland's districts reach neither threshold in any band.
- The region-tier shares, composed from local authorities, run from 5.4% to 8.5% for 80 and over and from 2.2% to 6.3% for 10-20 outside Northern Ireland; Northern Ireland's approximate shares are 2.7% and 0.7%.

**What this means.** Under the change, about 21 constituency cells (19 in the 10-20 band, 2 in 20-30) and 5 local-authority cells would sit at least 25% below the incumbent's rows by construction. That blocks release unless the evaluation compares on one universe too: a ruling for the November change.

## C8. CGT stage residual at design weights

`docs/evidence/uk-local-surface-1278/scripts/cgt_stage_residual.py` measures the CGT stage's residual on a built spine at its design weights (microcosm#725). It compares the liable gainers (gains above the annual exempt amount) in the stage's own groups with the HMRC margins the stage rakes onto, folded by the stage's own functions:
- Table 6 by age, in seven groups with 75 and over pooled;
- Table 5 by region, in four groups (taxpayers only).

A group below 30 distinct survey source households is pooled with its neighbour, so no group can be recovered by subtracting the others from a total. The script has been run on the spine built from main (`121fd4dd6`); its results are not in this receipt. The calibrated view, by region and by seat, waits for the pre-Budget build (M4).
