# SOI Historic Table 2 national controls

Fourteen IRS SOI quantities reach the US target surface from two sources: the
Historic Table 2 (HT2) state file and a Pub 1304 Table 1.x national table.
This page records why the two count the same thing, which one supplies the
national level, and how `fiscal_targets.py` combines them (microcosm#1243).
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
- HT2 tabulates returns the IRS processed in the following calendar year,
  with prior-year returns standing in for late filers, so it usually runs a
  little below the sample-based tables. The widest gap is itemized deductions in
  TY2022 (HT2 is 2.6% below). None approaches the 2.36x of the capital-gains
  columns, which count different populations
  ([us-soi-capital-gains-concepts.md](us-soi-capital-gains-concepts.md)).
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
- **Processing year.** HT2's TY2022 total is a little below Table 1.4's
  (previous section).

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
  control, the HT2-lineage national all-AGI rows of that quantity retire: the
  HT2 US row and HT2's US workbook row. Matching is by model quantity
  (variable, mode, filing status, universe and filters), so the workbook's
  EITC row, which shares the `return_count` measure id, stays. Table 1.2's
  copies of Table 1.1's count and AGI equal the control and stay.
- **No eligible control.** When the control is older than the HT2 row, after
  the build period, or missing, or when the HT2 US total is missing, the HT2
  rows keep their levels and the HT2 US row stays.
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

## What this does not change

- **Share assumption.** TY2022 HT2 state shares stand in for TY2023 shares.
  HT2 TY2023 is published (`23in55cmcsv.csv`, already the source of the
  #940 bands). If Chronicle packages its state totals under the same record
  sets, latest-vintage selection, which strips the period from record-set
  ids, keeps them instead, and this pass rebases them onto the TY2023
  controls by the processing-year factor alone
  (`test_a_control_at_the_hts_own_period_still_rebases`).
- **Congressional-district rows** of these columns stay TY2022 levels on the
  `full` surface. They are not on `national_state`; #1040, #1229 and #1231
  reconcile other district families.
- **Columns without an all-returns control** keep their HT2 levels: N04470
  (itemizing returns) and N04800 (returns with taxable income) appear
  nationally only in Table 4.3, which excludes dependent filers.
- **A newer HT2 than its control.** If HT2 ever lands a tax year before
  Table 1.x, the HT2 rows keep their levels and the HT2 US row stays beside
  the older control. The pinned-feed invariant then fails on the re-pin.
- **Wages concept.** Both wage rows bind gross `employment_income`;
  microcosm#1191 covers the engine taxing `irs_employment_income`.
