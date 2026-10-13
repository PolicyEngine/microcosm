# SOI Historic Table 2 national controls

Fourteen IRS SOI quantities reach the US target surface from two sources: the
Historic Table 2 (HT2) state file and a Pub 1304 Table 1.x national table.
This page records why the two count the same thing, which one supplies the
national level, and how `fiscal_targets.py` combines them (microcosm#1243).
The congressional-district file carries 11 of the same columns; [its
rows](#congressional-district-rows-1250) take the same controls (#1250).
The taxable-interest (#489, #518) and capital-gains (#1035, #1036) families
follow the same doctrine through their own passes.

## The defect

On the pinned feed (`51aa40f`, compiled at 2024 as the release does), the
`national_state` surface bound two national all-AGI targets for each of these
quantities: the TY2022 HT2 US row and a TY2023 Table 1.x row. Both measure
the same model variable, filing status, return universe and filters, so no
weight vector can meet both. They were 0.3% to 12.2% apart. For
`ira_distributions` sum and `taxable_pension_income` count, the 51 HT2 state
rows also summed past the Table 1.4 target (1.019x and 1.009x). The return
count had a fourth national row: HT2's US workbook (`22in54us.xlsx`,
"Historic Table 2 state data, United States total"), which the feed packages
as `irs_soi.ty2023.state_2022.us`: TY2022 data stamped ty2023.

## The columns and what they count

Each HT2 column is defined at the same Form 1040 line as its Table 1.4
column. The HT2 documentation guides (`20incmdocguide.doc` to
`23incmdocguide.doc`) give the line for every year, and the Table 1.4
headers (`20in14ar.xls` to `23in14ar.xls`) name the same item. The HT2 US
row also matches Table 1.4's "All returns, total" cell in the same tax year,
for every year:

| HT2 column | Form 1040 line | Control | HT2 / Table 1.4, TY2020-TY2023 | Model variable |
|---|---|---|---:|---|
| N1 | returns | Table 1.1 `return_count` | 0.990-0.998 | `count` |
| A00100 | 11 | Table 1.1 `adjusted_gross_income` | 0.997-0.999 | `adjusted_gross_income` |
| N00200 / A00200 | 1z (1 before TY2022) | Table 1.4 `wages_salaries_*` | 0.995-0.999 / 0.997-0.999 | `employment_income` |
| N01400 / A01400 | 4b | Table 1.4 `taxable_ira_distributions_*` | 0.999-1.009 / 0.992-1.001 | `ira_distributions` |
| N01700 / A01700 | 5b | Table 1.4 `taxable_pension_income_*` | 0.996-1.005 / 0.986-0.996 | `taxable_pension_income` |
| N02300 / A02300 | Schedule 1, 7 | Table 1.4 `unemployment_compensation_*` | 0.983-1.002 / 0.982-0.994 | `unemployment_compensation` |
| N02500 / A02500 | 6b | Table 1.4 `taxable_social_security_*` | 0.991-0.996 / 0.993-1.002 | `taxable_social_security` |
| A04470 | 12 (12a in TY2021) | Table 1.2 `total_itemized_deductions_amount` | 0.974-0.998 | `itemized_taxable_income_deductions` |
| A04800 | 15 | Table 1.2 `taxable_income_amount` | 0.997-0.999 | `taxable_income` |

- The return count and AGI come from Table 1.1, whose "all returns" cells
  Table 1.4 repeats.
- The feed carries the Table 1.4 itemized-deduction and taxable-income
  totals (cols 135-136 and 141-142) only in Table 1.2. Table 1.2 prints the
  same TY2023 cells: $690.845B and $11,944.447B.
- The same-year differences are small, and HT2 is usually the lower. The
  widest is itemized deductions in TY2022 (HT2 is 2.6% below). None
  approaches the 2.36x of the capital-gains columns, which count different
  populations ([us-soi-capital-gains-concepts.md](us-soi-capital-gains-concepts.md)).
- Their cause is not established. HT2 tabulates the full population of
  returns the IRS processed in the following calendar year, using prior-year
  returns as a proxy for late filers. The HT2 guide's endnote 1 says that
  practice is consistent with SOI's national tabulations, and Pub 1304's
  sample is drawn from the same processing year (Pub 1304, "Domain of
  Study"). Sampling error and SOI's record editing are candidates, untested
  here. Either way, the share step below removes the difference.
- Both rows of each pair map to the same model variable and measure mode.
  `test_each_registered_column_and_its_control_bind_the_same_model_quantity`
  checks this against the compile's measure maps.

The identities are reproduced by
`~/PolicyEngine/_reviews/soi-national-controls-20261012/identities.py`, which
locates each Table 1.4 column by its header text. The Table 1.4 workbooks
for TY2020-TY2023 and the HT2 files for TY2022 and TY2023 are byte-identical
(sha256) to the feed's sources; the TY2020 and TY2021 HT2 files come from
irs.gov.

## Where the gap came from

The aged HT2 US row differs from the aged control by two factors:

- **Vintage.** HT2 amounts age from TY2022 to TY2023 by realized AGI growth
  (wages by realized wage growth); counts do not age. Each column's own
  change differed.
- **Same-year difference.** HT2's TY2022 total differs slightly from Table
  1.4's (previous section).

| Quantity | Table 1.4 TY2023 / TY2022 | HT2 aging link | Vintage factor | HT2 / Table 1.4, TY2022 | National max / min |
|---|---:|---:|---:|---:|---:|
| Taxable Social Security, amount | 1.1495 | 1.0305 | 1.1155 | 0.9943 | 1.1219 |
| Taxable Social Security, returns | 1.0425 | 1 | 1.0425 | 0.9922 | 1.0507 |
| Itemized deductions, amount | 1.0342 | 1.0305 | 1.0036 | 0.9738 | 1.0306 |
| Taxable IRA distributions, amount | 1.0009 | 1.0305 | 0.9713 | 0.9929 | 1.0223 |
| Taxable IRA distributions, returns | 1.0253 | 1 | 1.0253 | 1.0091 | 1.0161 |
| Taxable pensions, returns | 0.9840 | 1 | 0.9840 | 0.9977 | 1.0139 |
| Unemployment compensation, returns | 0.9934 | 1 | 0.9934 | 0.9829 | 1.0107 |
| Taxable income, amount | 1.0197 | 1.0305 | 0.9895 | 0.9970 | 1.0076 |
| Wages, returns | 1.0016 | 1 | 1.0016 | 0.9947 | 1.0069 |
| Taxable pensions, amount | 1.0224 | 1.0305 | 0.9922 | 0.9857 | 1.0066 |
| Return count | 0.9954 | 1 | 0.9954 | 0.9896 | 1.0060 |
| AGI | 1.0305 | 1.0305 | 1 | 0.9965 | 1.0035 |
| Wages, amount | 1.0478 | 1.0478 | 1 | 0.9969 | 1.0031 |
| Unemployment compensation, amount | 1.0229 | 1.0305 | 0.9926 | 0.9898 | 1.0028 |

The last column is the vintage factor divided by the TY2022 ratio, or its
inverse. Taxable Social Security grew 15.0% from TY2022 to TY2023, far past
AGI's 3.05%.

## How the targets combine

`_rebase_stale_soi_ht2_distributions` runs after the capital-gains rebase and
before the state AGI bands.

- **Controls.** `_SOI_HT2_NATIONAL_CONTROLS` registers one Pub 1304 table and
  column per HT2 column. Only a national all-AGI fact of that table and
  column qualifies, and the latest one not after the build period wins. HT2,
  HT2's US workbook, the congressional-district file, Table 4.3 (excluding
  dependents) and Table 2.1 (itemizers) never qualify. Two different records
  at the winning period raise `AmbiguousSoiNationalControlError`, so feed
  order never decides. The capital-gains control uses the same chooser.
- **Shares.** Each all-AGI HT2 state row becomes its share of the HT2 US
  total, scaled to the control:
  `value x control / HT2 US`. The value lands at the control's period and
  target aging adds the remaining years, so the control and its states age by
  the same factor. Each row records `uprating_factor`, the control's record
  id, and `soi_source_concept` / `soi_control_concept` (the Form 1040 line,
  equal on both sides).
- **One national level.** Once a quantity's states are shares of the
  control, the HT2-lineage national all-AGI rows of that quantity retire:
  the HT2 US row and HT2's US workbook row. Matching is by model quantity
  (variable, mode, filing status, universe and filters), so the workbook's
  EITC row, which shares the `return_count` measure id, stays. Table 1.2's
  copies of Table 1.1's count and AGI equal the control and stay.
- **Per-row eligibility.** A row is rebased or retired only when the
  control is at or after its own data year. The workbook's data year comes
  from its family name (`state_2022`), not its ty2023 stamp. Where HT2
  vintages mix (a state falling back to an older year), the older rows
  become shares and a newer HT2 US row keeps its level.
- **No eligible control.** When the control is older than the HT2 row,
  after the build period, or missing, or when the HT2 US total is missing,
  the HT2 rows keep their levels and the HT2 US row stays.
- **One denominator.** Two different HT2 US records for one measure, filing
  status, universe and year raise `AmbiguousSoiNationalControlError`, so
  feed order cannot pick the rebase denominator.
- **State AGI bands.** The #940 bands are shares of the state total that
  binds. A rebased state total sits at its control's period, and
  `_soi_state_agi_band_controls` now reads that landing period
  (`uprating_to_period`), so the AGI bands age with the total from TY2023.
  Without that, AGI bands would age from TY2022 and overshoot their total by
  3.05%.

## Effect on the pinned feed

Compiled at 2024 with target aging and the packaged congressional-district
crosswalk, then narrowed to `national_state`. Before is `main` plus #1036.

| Quantity | 51 states, before | 51 states, after | Move | States / smaller national, before | after |
|---|---:|---:|---:|---:|---:|
| `taxable_social_security` sum | $509.58B | $571.71B | +12.19% | 0.998 | 0.998 |
| `taxable_social_security` count | 24,397,150 | 25,634,869 | +5.07% | 0.997 | 0.997 |
| `itemized_taxable_income_deductions` sum | $726.41B | $748.63B | +3.06% | 0.997 | 0.997 |
| `ira_distributions` sum | $485.38B | $474.78B | -2.18% | **1.019** | 0.997 |
| `ira_distributions` count | 16,385,750 | 16,649,213 | +1.61% | 0.997 | 0.997 |
| `taxable_pension_income` count | 29,818,930 | 29,410,814 | -1.37% | **1.009** | 0.996 |
| `unemployment_compensation` count | 4,637,960 | 4,687,496 | +1.07% | 0.998 | 0.998 |
| `taxable_income` sum | $12,978.42B | $12,880.42B | -0.76% | 0.999 | 0.992 |
| `employment_income` count | 127,029,870 | 127,910,705 | +0.69% | 0.995 | 0.995 |
| `taxable_pension_income` sum | $1,001.75B | $1,008.37B | +0.66% | 0.995 | 0.995 |
| `count` | 158,710,000 | 159,655,171 | +0.60% | 0.994 | 0.994 |
| `adjusted_gross_income` sum | $16,441.78B | $16,499.02B | +0.35% | 0.993 | 0.993 |
| `employment_income` sum | $10,654.77B | $10,687.80B | +0.31% | 0.992 | 0.992 |
| `unemployment_compensation` sum | $33.43B | $33.52B | +0.28% | 0.997 | 0.997 |

- Every state row of a family moves by the same factor. After the change,
  the 51 states sum to the control times their share of the HT2 US row in
  `22in55cmcsv.csv`, exactly (error at most 4e-16).
- 15 national rows retire: the 14 HT2 US rows and the workbook's return
  count. No row is added. Compiled targets go from 33,250 to 33,235, and
  `national_state` from 6,102 (`882f4764a248`) to 6,087 (`0e899cd7f893`);
  the full registry from `f09571684885` to `0bbbad9b22d1`.
- 1,221 values move, each family by one factor:
  - the 714 HT2 state rows on `national_state` (14 x 51);
  - 99 HT2 rows on `full` only, which stand in for at-large congressional
    districts (9 states x 11 measures);
  - the 408 #940 state AGI bands, by their state total's factor (AGI
    x1.00348, returns x1.00596). Each stays its recorded share of the total
    (error at most 6e-16).
- On `national_state`, the 63 compared SOI families each sum to at most the
  smaller national target of their quantity, and every compared quantity has
  one national value. Before, 14 had two values and two state families summed
  past the smaller one.

The diff and checks are reproduced by `scripts/diff_compiles.py` in the
evidence folder.

## Congressional-district rows (#1250)

The congressional-district file (`22incd.csv`, packaged as
`irs_soi.ty2023.congressional_district_2022.all_returns`: TY2022 data
stamped ty2023) carries 11 of the 14 columns on the `full` surface. It has
no wage or taxable-income rows. Each column has a US row, 51 state rows
(`<st>_total`), and 428 district rows that the packaged crosswalk maps onto
the 436 districts of the 119th plan; an at-large state enters as its state
total. Before #1250 these rows were TY2022 levels, and the US row was a
second national target beside each control.

### Same columns, a narrower count

`22incddocguide.docx` defines each column at the same line as the HT2
guides, under the same IRS name. Its US row counts fewer returns than Table
1.4 in the same tax year, because the file covers only returns processed in
calendar 2023, drops returns whose ZIP code is missing or does not match the
state, and drops APO/FPO, foreign and territory addresses (guide sections C
and D). It also suppresses items with fewer than 20 returns in a district
and items dominated by one return (section E). It is a narrower count of the
same concept, like HT2 is of Table 1.4, but about 2% narrower again:

| Column | Line | District US / Table 1.4, TY2022 | District US / HT2 US, TY2022 | Control / district US, pinned feed |
|---|---|---:|---:|---:|
| N1 | returns | 0.9690 | 0.9793 | 1.0272 |
| A00100 | 11 | 0.9724 | 0.9758 | 1.0597 |
| N01400 / A01400 | 4b | 0.9966 / 0.9767 | 0.9877 / 0.9837 | 1.0288 / 1.0247 |
| N01700 / A01700 | 5b | 0.9827 / 0.9702 | 0.9850 / 0.9843 | 1.0014 / 1.0538 |
| N02300 / A02300 | Schedule 1, 7 | 0.9671 / 0.9699 | 0.9839 / 0.9798 | 1.0272 / 1.0546 |
| N02500 / A02500 | 6b | 0.9797 / 0.9831 | 0.9874 / 0.9888 | 1.0641 / 1.1692 |
| A04470 | 12 | 0.9404 | 0.9657 | 1.0997 |

The last column is what the aged control is over the aged district US row
on the pinned feed: the coverage gap times the TY2022 to TY2023 change of the
column. The district rows age only 2023 to 2024, because of the stamp.
Within the file, the 51 states sum exactly to the US row and each state's
districts sum exactly to the state, for all 11 columns. The register
records each column's IRS name (`source_column`), so the guide check is a
test (`test_the_register_names_each_columns_reviewed_irs_id`).
`~/PolicyEngine/_reviews/cd-soi-national-controls-20261012/scripts/cd_identities.py`
reproduces the table from the sha256-matched `22incd.csv`, `22in14ar.xls`
and `22in55cmcsv.csv`.

### How the rows combine

`_rebase_stale_soi_district_distributions` runs after the HT2 pass. It uses
the same register, the same control chooser and the same refusal
(`AmbiguousSoiNationalControlError`). Its rule is the one #1229 (taxable
interest) and #1040 (capital gains) use for other district families:

- **Shares of the file's own US row.** Each district-file state and district
  row becomes `value x control / district US`. The US row is the same
  record set's national row of the same measure, filing status, universe and
  period. One factor per measure scales the states and the districts alike,
  so the district-to-state reconciliation that runs next keeps factor 1.
  Every share names its denominator in `soi_share_total_source_record_id`
  and lands at the control's period, as the HT2 shares do.
- **One national level.** The district US row retires.
- **No eligible control.** With no control, one after the build period, or
  one older than the row's period, the rows keep their levels and the US row
  stays.
- **No denominator.** With an eligible control but no nonzero district US
  row, a row cannot be a share, and as a level it would be the stale second
  value this pass removes, so it is dropped. That is the district rule of
  #1229 and #1040. The HT2 pass keeps HT2 rows as levels in the same case.
  Neither case occurs on the pinned feed.
- **One denominator.** Two different district US values for one key refuse
  the compile; equal twins name the smaller record id, so feed order never
  picks the denominator.
- **Reviewed columns only.** A district fact of a registered measure whose
  control is eligible must read the register's `source_column`. Otherwise
  the compile refuses (`UnreviewedSoiNationalControlColumnError`), as it
  does for a missing column. The check runs before the crosswalk, which
  merges source districts into one fact with a single contributor's layout.
  The district package's SALT and PTC measures read the wrong columns
  (microcosm#1038), so a re-pin could do the same here.
- **The stamp cancels.** Row and US row carry the same stamp, so stamping the
  file truthfully at 2022 gives the same values
  (`test_the_share_ignores_the_package_stamp`). Under #1030's data-year
  correction each row is a share of a restamped total of its own package,
  which #1030 admits.

### Effect on the pinned feed

Compiled as the release does; before is #1247's head (`490b67181`).

| Quantity | District file, 51 states, before | After | Move |
|---|---:|---:|---:|
| `taxable_social_security` sum | $490.09B | $573.04B | +16.92% |
| `taxable_social_security` count | 24,167,050 | 25,716,763 | +6.41% |
| `itemized_taxable_income_deductions` sum | $682.97B | $751.10B | +9.97% |
| `adjusted_gross_income` sum | $15,682.85B | $16,619.16B | +5.97% |
| `unemployment_compensation` sum | $31.89B | $33.64B | +5.46% |
| `taxable_pension_income` sum | $961.67B | $1,013.42B | +5.38% |
| `ira_distributions` count | 16,227,430 | 16,694,154 | +2.88% |
| `count` | 156,342,760 | 160,602,107 | +2.72% |
| `unemployment_compensation` count | 4,573,040 | 4,697,502 | +2.72% |
| `ira_distributions` sum | $464.87B | $476.36B | +2.47% |
| `taxable_pension_income` count | 29,501,160 | 29,541,284 | +0.14% |

- **Retired: 11 rows**, the district US rows. **Added: none.** Compiled
  targets go from 33,235 to 33,224; the full registry from `0bbbad9b22d1` to
  `ce77f86efb24`. `national_state` holds no district-file row and is
  unchanged (6,087, `0e899cd7f893`).
- **5,357 values move** (11 x (51 + 436)), each family by its one factor
  (the last column above). No other spec changes, in value or metadata.
- After aging, the file's 51 states, and its 436 districts, each sum to the
  control (relative error at most 1e-12), and every district reconciles to
  its state with factor 1.
- On `full`, every quantity a control of this register measures now has one
  national value. Quantities with more than one national value go from 38 to
  27, the 22 and 5 under "What this does not change". (Two more EITC groups
  look like duplicates only to `_soi_national_quantity`: their Table 2.5 rows
  take the child count from the record set, which the key does not read.)
- **At-large districts are bound twice**, as before: the district file's
  row, and HT2's state row standing in for the district (AK, DE, DC, ND, SD,
  VT, WY and Montana's two districts; 99 rows). Both are now shares of the
  same control, so the two differ only by the two files' shares of that
  state: HT2 / district file is 0.966 to 1.072 (median 0.988), against
  0.986 to 1.17 (median 1.037) before, except Wyoming's and South Dakota's
  itemized deductions (1.64 and 1.35), where the two files disagree about the
  state. Counting both, the 445 district rows of a measure sum to 1.015 to
  1.022 times the control; each family alone sums to at most the control.
- **ACS-local `state_cd`.** All 26,425 values are unchanged (relative
  difference at most 4.2e-15), as are its counts, drops and band verdicts.
  It takes district rows only as within-state shares of HT2, which a
  national factor does not move. Its level bridges read a rebased sibling
  at the file's own level ([us-acs-local-soi-target-surface.md](us-acs-local-soi-target-surface.md)).
  Only the receipt's per-measure rebase factors for these columns move, to
  0.975 to 1.063 (medians 0.995 to 1.009): what is left is where the two
  files disagree about a state's share.

`scripts/diff_cd.py`, `scripts/at_large_pairs.py` and
`scripts/compare_statecd.py` in the evidence folder reproduce these figures.

## What this does not change

- **Share assumption.** TY2022 HT2 state shares stand in for TY2023 shares.
  HT2 TY2023 is published (`23in55cmcsv.csv`, already the source of the
  #940 bands). If Chronicle packages its state totals under the same record
  sets, latest-vintage selection, which strips the period from record-set
  ids, keeps them instead, and this pass rebases them onto the TY2023
  controls by the same-year difference alone
  (`test_a_control_at_the_hts_own_period_still_rebases`).
- **Congressional-district rows of other columns.** The district file's US
  row is still a second national value on `full` for 22 quantities whose
  other national row is the TY2022 HT2 US row (dividends, Schedule C,
  partnership, rental and tax-exempt income, EITC, ACTC, income tax, medical
  deductions), and for capital gains, SALT and PTC returns, which #1040 and
  #1224 handle.
- **Columns without an all-returns control** keep their HT2 levels: N04470
  (itemizing returns) and N04800 (returns with taxable income) appear
  nationally only in Table 4.3, which excludes dependent filers.
- **A newer HT2 than its control.** If HT2 ever lands a tax year before
  Table 1.x, the HT2 rows keep their levels and the HT2 US row stays beside
  the older control. The pinned-feed invariant then fails on the re-pin.
- **Wages concept.** Both wage rows bind gross `employment_income`;
  microcosm#1191 covers the engine taxing `irs_employment_income`.
