# UK spine comparison ledger — microcosm#686 · PR #747

Every variable the E workstream manipulates, measured against the incumbent,
with the mechanism and evidence behind each difference — and what is still
unruled. This is the committed rendition of the living adjudication packet;
the interactive version is a Claude artifact (private; ask María for access).
Updated as adjudications land.

**Candidate**: `spine-c.h5` · 52,846 hh / 61,211 bu / 113,649 p · 147 input layers
**Reference**: `enhanced_frs_2024_25.h5` @ 1.56.16 (`a9e52499`)

## What the numbers are

Every figure in the share tables is a **nonzero share**: the fraction of the
column's owning-entity rows carrying a non-zero value, 0–1. `0.6072` for
savings means 60.7% of households report some savings — it is not an amount.
The parity screen measures *incidence*, never level. That limit is real: a
column can match perfectly on share while being badly wrong on level — the
Scottish water fix produces an identical share of 0.878377 under both the
retired and corrected mappings while the amount moved from ~£185 to ~£395 per
Scottish household. Figures that are levels carry a £ sign and sit in the
Levels section, as conditional means (mean over nonzero carriers) with the
carrier count alongside.

## Which incumbent

Three artifacts get called "the incumbent"; they answer different questions.

| | artifact | role |
|---|---|---|
| A | `enhanced_frs_2024_25.h5` @ 1.56.16 | the published post-calibration artifact; every share delta here is against it |
| B | `incumbent_{base,wealth,consumption}_ebf733c.h5` | partial pipelines rebuilt locally at 1.56.14 for E3/E5/E6 method head-to-heads; one release behind, carries the benunit-sort defect |
| C | WAS R8 · LCFS 2023-24 · ETB · SPI 2022-23 | the source surveys — ground truth neither build controls |

The 1.56.14→1.56.16 re-pin moved no reference share by more than 0.0023
(E7: exactly zero); the deltas below are robust to it.

## Gate status

| leg | state |
|---|---|
| L0 build + determinism | complete — 3 rungs clean, twins payload-identical, record identity exact at 52,846 |
| L1 identity receipts | e4 e5 e6 e7 e8 green (e7 written at #686; ladder complete) |
| L2 whole-spine parity | measured — 26 beyond ±0.02, verdict `defect` until the queue is ruled |
| L3 baselines | measured — 177 input-mass totals, 47 QRF tail grids |

## E6 · consumption — needs ruling

Measured fresh against the donors. On the eleven LCFS columns **ours is closer
on seven**, the incumbent on three, one tie — weaker than "the incumbent
collapses zero-inflated targets" (which stays true and dramatic on education:
incumbent 0.0003 vs donor 0.0476) but honest. Petrol/diesel are partly by
design (our `has_fuel` gate zeroes non-fuel households); on petrol the *level*
favours us while the share favours the incumbent — see Levels.

| column | donor truth (share) | incumbent | ours | closer |
|---|---|---|---|---|
| dfe_education_spending | see ETB note | 0.0003 | 0.2258 | undecidable |
| bus_subsidy_spending | see ETB note | 0.3167 | 0.5554 | undecidable |
| rail_subsidy_spending | see ETB note | 0.1277 | 0.1422 | undecidable |
| restaurants_and_hotels_consumption | 0.7651 | 0.6324 | 0.7903 | ours |
| petrol_spending | 0.3911 | 0.4446 | 0.3002 | incumbent |
| education_consumption | 0.0476 | 0.1256 | 0.0170 | ours |
| household_furnishings_consumption | 0.9104 | 0.8136 | 0.9216 | ours |
| miscellaneous_consumption | 0.9728 | 0.9003 | 0.9918 | ours |
| communication_consumption | 0.8696 | 0.7974 | 0.8814 | ours |
| alcohol_and_tobacco_consumption | 0.5383 | 0.5603 | 0.5150 | tie |
| domestic_energy_consumption | 0.9879 | 0.9549 | 0.9952 | ours |
| diesel_spending | 0.2040 | 0.1910 | 0.1580 | incumbent |
| transport_consumption | 0.8702 | 0.8668 | 0.8934 | incumbent |
| health_consumption | 0.5426 | 0.5128 | 0.5386 | ours |

**ETB truth is not yet decidable**: the donor share moves with the weight
basis (education 0.0943 unweighted / 0.1327 household / 0.2150 individual)
and none reconcile with the E6 receipt's 0.2546. Until the stage's weight
convention is pinned, quoting one as truth would be false precision.

## E5 · wealth — signed at E5 · carry-forward?

Adjudicated 2026-08-19 ("E5 is not required to reproduce the incumbent's
inflated totals; donor-benchmark evidence"). Fresh measurement corroborates.

| column | donor truth · WAS R8 (share) | incumbent | ours | closer |
|---|---|---|---|---|
| savings | 0.6072 | 0.6620 | 0.6107 | ours (0.0035 off) |
| property_wealth | 0.6433 | 0.7081 | 0.6607 | ours |
| other_residential_property_value | 0.0363 | 0.0763 | 0.0367 | ours (0.0004 off) |
| main_residence_value | 0.6236 | 0.6747 | 0.6356 | ours |
| corporate_wealth | not measured (fold of several donor columns) | 0.8222 | 0.7792 | — |
| student_loan_balance | not measured (separate source) | 0.0197 | 0.0493 | — |

`owned_land` is **not** in this list; its reviewed exclusion expiring
2026-09-20 is a separate question.

## E7 · SPI channel — evidence gap closed, favours the spine

The three columns have three different sources of truth. Measured against
each, **ours is closer on all three**; the direction that looked "uniformly
one-way and unexplained" at #717 is uniformly toward the source. Ruling here
closes #717's open item.

| column | truth (share) | source of truth | incumbent | ours | closer |
|---|---|---|---|---|---|
| savings_interest_income | 0.3960 | SPI donor `INCBBS`, FACT-weighted | 0.4250 | 0.3946 | ours (0.0014 off) |
| tax_free_savings_income | 0.1540 | raw FRS at the `frs_spine` stage | 0.1897 | 0.1352 | ours |
| employer_pension_contributions | 0.2587 | the 3× derive at `frs_hmrc_spine_leaves` | 0.3149 | 0.2682 | ours |

Three of the six columns #717 flagged have since fallen inside the band:
dividend_income (−0.0021), gift_aid (−0.0097),
pension_contributions_via_salary_sacrifice (−0.0035).

## E8 and entity counts — signed at #684

| surface | reference | ours | delta | mechanism |
|---|---|---|---|---|
| employee_pension_contributions | 0.2735 | 0.2246 | −0.0488 | salsac conversion depth; incumbent's conversion step was inert |
| person rows | 113,617 | 113,649 | +32 | donor-selection RNG over id-sorted candidates |
| benunit rows | 61,223 | 61,211 | −12 | same draw |
| household rows | 52,846 | 52,846 | exact | identity closes — proves selection, not miscount |

## Column coverage

| column | status | origin | disposition |
|---|---|---|---|
| free_school_meals | in band | raw `fsmval` · child.tab | ported; ref 0.034220 → ours 0.034131 (−0.000089) |
| free_school_fruit_veg | in band | raw `fsfvval` · child.tab | ported; −0.000001 |
| healthy_start_vouchers | in band | raw `heartval` · child + adult | ported; +0.000008 |
| free_school_breakfasts | outside contract | raw `fsbval` | ported; not engine-known, never enters the 145-column surface |
| num_bedrooms | extra | `frs_spine` | net-new; predictor-quality question open on #145 |
| other_investment_income | extra | `hmrc_spi_income_spine` | declared by the incumbent's own restoration — ahead, not diverging |

These four were mislabelled "E9 derived-benefit class" in the #723 receipt;
#685 (E9) is UC deduction attributes, bus fares and WAS debt. This was a port
omission from a merged increment, now closed — parity reports **0 missing**.

## Levels — annual £ per carrier

Conditional means over nonzero rows; carrier counts withheld here for brevity
but recorded in the interactive ledger and reproducible from the licensed
evidence dir; no cell is dominated by a single record; all cells ≥10 carriers.
Donor figures survey-weighted and annualised where the stage annualises.

| column | donor truth | incumbent | ours | closer on level |
|---|---|---|---|---|
| **E5 · WAS** | | | | |
| savings | 27,860 | 85,294 | 52,061 | ours (1.9× donor vs 3.1×) |
| property_wealth | 387,430 | 311,208 | 341,415 | ours |
| other_residential_property_value | 259,375 | 493,846 | 291,791 | ours (1.1× vs 1.9×) |
| main_residence_value | 340,534 | 286,378 | 280,812 | incumbent |
| **E6 · LCFS** | | | | |
| education_consumption | 5,543 | 10,783 | 6,134 | ours |
| restaurants_and_hotels_consumption | 2,995 | 5,304 | 3,473 | ours |
| miscellaneous_consumption | 2,441 | 4,228 | 2,777 | ours |
| petrol_spending | 1,614 | 2,429 | 1,713 | ours — opposite of the share |
| diesel_spending | 1,911 | 2,389 | 2,065 | ours |
| transport_consumption | 5,278 | 7,588 | 6,656 | ours |
| health_consumption | 726 | 1,249 | 1,110 | ours |
| alcohol_and_tobacco_consumption | 1,075 | 1,091 | 1,373 | incumbent |
| household_furnishings_consumption | 2,217 | 3,501 | 3,589 | incumbent |
| communication_consumption | 773 | 836 | 863 | incumbent |
| **E7 and raw mappings** | | | | |
| savings_interest_income (taxable part, SPI channel) | 339 | 41 | 187 | ours (0.55× donor vs 0.12×) |
| tax_free_savings_income | — | 602 | 747 | no donor |
| employer_pension_contributions | — | 6,357 | 6,347 | ≈ equal |
| water_and_sewerage_charges | — | 483 | 475 | ≈ equal — but the incumbent's is England+Wales-only (it zeroes Scotland); coincidence of averaging, not agreement |
| free_school_meals | — | 629 | 629 | exact |

The SPI comparison must be taxable-to-taxable on the SPI channel: donor
`INCBBS` is interest *as reported for tax* (no ISA, nothing under the personal
savings allowance) while the spine's column is gross (taxable draw + FRS-side
tax-free part, identity-guarded). The residual 0.55× gap is the right
direction — SPI is a taxpayer population, the recipient channel is broader.

## Diagnostics · both spines

| check | result |
|---|---|
| twin determinism | pass — two independent full builds payload-identical; bytes differ only from HDF5 write-time stamps |
| record-count identity | exact — (16,288 + 10,000) × 2 + 270 = 52,846 |
| rung ladder | pass — f001 / f010 / full, each landing a Logbook row |
| identity receipts e4–e8 | pass, on both the post-E8 and pre-E8 spines |
| weighted-integrity baselines | measured — 177 input-mass totals, 47 QRF tail grids, 12 household QRF tail |
| Scottish water fix | verified — share 0.878377 at every rung; £390.80 / £375.08 / £405.34 per Scottish household vs ~£185 under the retired mapping |

## Open

- **ETB weight basis** — pins the truth for three columns.
- **E6 is not one class** — a single entry over all fifteen would sign columns where we are further from the donor than the incumbent is.
- **Entity-count entry scope** — surface-wide vs scoped to the two entities.
- **Stale donor ratios** in the E5/E6 prose receipts were measured at 1.56.14; the tables above are fresh at the current pin.
- **Upstream defect filed** as policyengine-uk-data#467 (no prescriptive fix); our side is correct and signed.

## Disclosure and citation

All figures are aggregates under disclosure control (CD171 §5.2.1): no unit
records, minimum cell count 10, thin columns suppressed, dominance checked.
Data collections used, cited and acknowledged per UKDS EUL clauses 11–12:

- DWP, *Family Resources Survey 2024-25*, UKDS SN 9563, DOI 10.5255/UKDA-SN-9563-1
- DWP/ONS, *Living Costs and Food Survey 2023-24*, UKDS SN 9468, DOI 10.5255/UKDA-SN-9468-3
- ONS, *Wealth and Assets Survey, Round 8*, UKDS SN 7215, DOI 10.5255/UKDA-SN-7215-20
- HMRC, *Survey of Personal Incomes 2022-23 Public Use Tape*, UKDS SN 9422
- ONS, *Effects of Taxes and Benefits on Household Income 1977–2024*, UKDS SN 8856, DOI 10.5255/UKDA-SN-8856-4

Crown copyright material is reproduced with the permission of the Controller
of HMSO and the King's Printer for Scotland. The original data creators,
depositors and funders bear no responsibility for this analysis.
