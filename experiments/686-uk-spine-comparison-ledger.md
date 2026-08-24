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
| L2 whole-spine parity | **signed_parity** — 26 beyond ±0.02, all signed; 0 unsigned; `--strict` clean |
| L3 baselines | measured — 177 input-mass totals, 47 QRF tail grids |

## What signing did, and what the band means

`spine_swap_signed_differences.json` now holds **13 entries covering 26
beyond-band share columns, 2 entity counts and 2 net-new columns**. The
instrument reads it and returns `signed_parity` with nothing unsigned.

| entry | class | covers |
|---|---|---|
| `lcfs-consumption-regime-gated-incidence` | mechanism_change | 10 LCFS columns where ours is closer to the donor |
| `lcfs-fuel-consumption-incidence-gate` | mechanism_change | petrol, diesel — incumbent closer on share, ours on level |
| `lcfs-transport-aggregate-incidence` | mechanism_change | transport — incumbent marginally closer on share |
| `etb-services-regime-gated-incidence` | **defect_fix** | dfe_education, bus_subsidy — incumbent degenerate |
| `was-wealth-qrf-incidence` | qrf_implementation | 5 benchmarked wealth columns |
| `was-student-loan-balance-fold` | qrf_implementation | student_loan_balance — no benchmark |
| `spi-channel-qrf-incidence` | qrf_implementation | the 3 SPI-rewritten columns |
| `salary-sacrifice-conversion-depth` | mechanism_change | employee_pension_contributions |
| `donor-selection-rng-entity-counts` | rng_stream | person +32, benunit −12 (household stays exact) |
| `num-bedrooms-net-new-column` | net_new_column | spine adds it |
| `other-investment-income-net-new-column` | net_new_column | spine adds it |
| `scottish-water-incumbent-nan-zeroing` | defect_fix | water share |
| `scottish-water-sewerage-successor-level` | mechanism_change | water + council_tax **levels** (dormant until calibration) |

**Three entries where the incumbent is closer** — petrol, diesel, transport —
are scoped apart on purpose. A single LCFS class entry would have signed them
under a verdict they do not share; that was your objection and it is what
drove the split. Each says in its own text that the evidence runs the other
way on incidence, so re-opening one does not require re-opening the class.

**The band.** The instrument now holds the share surface to the ±0.02 #723
band rather than the reference's 6-decimal grain. Without that, 90 further
columns — third-decimal drift from re-running every stochastic stage — would
each have needed a permanent adjudication, which is precisely the blanket
amnesty the register exists to stop. Nothing is hidden: all 90 are still in
the receipt under `within_band` (max 0.019), `--share-band 0` restores the
exact check, and **structural differences ignore the band entirely** — a
column appearing or vanishing, and every entity count, signs exactly at any
band.

## E6 · consumption — signed

Re-measured against each donor through the **stage's own committed cleaning
function**, on the **survey-weighted** basis. That convention matters: it
reproduces the E6 acceptance receipt's education figure (0.2546 unweighted on
the ETB services frame) exactly, which is what confirms it is the house
convention rather than one of several defensible choices.

On the thirteen LCFS columns **ours is closer on ten shares and twelve
levels**. Petrol and diesel are the `has_fuel` gate under-placing incidence —
signed as that, with the direction stated. Level is population mean per
household, as a ratio to the donor's.

| column | donor (share) | incumbent | ours | closer | donor £/hh | inc × | our × | closer |
|---|---|---|---|---|---|---|---|---|
| restaurants_and_hotels_consumption | 0.7651 | 0.6305 | 0.7903 | ours | 2,291 | 1.75 | 1.25 | ours |
| education_consumption | 0.0476 | 0.1258 | 0.0170 | ours | 264 | 5.00 | 0.41 | ours |
| household_furnishings_consumption | 0.9104 | 0.8126 | 0.9216 | ours | 2,019 | 1.68 | 1.63 | ours |
| electricity_consumption | 0.9228 | 0.8614 | 0.9644 | ours | 845 | 1.06 | 0.99 | ours |
| gas_consumption | 0.9815 | 0.9523 | 0.9952 | ours | 651 | 1.07 | 1.00 | ours |
| miscellaneous_consumption | 0.9728 | 0.8975 | 0.9918 | ours | 2,375 | 1.81 | 1.21 | ours |
| communication_consumption | 0.8696 | 0.7951 | 0.8814 | ours | 672 | 1.16 | 1.20 | incumbent |
| alcohol_and_tobacco_consumption | 0.5383 | 0.5630 | 0.5150 | ours | 579 | 1.25 | 1.24 | ours |
| domestic_energy_consumption | 0.9835 | 0.9541 | 0.9953 | ours | 1,496 | 1.06 | 1.00 | ours |
| health_consumption | 0.5426 | 0.5128 | 0.5386 | ours | 394 | 1.90 | 1.59 | ours |
| transport_consumption | 0.8702 | 0.8623 | 0.8934 | **incumbent** | 4,593 | 1.63 | 1.37 | ours |
| petrol_spending | 0.3911 | 0.4442 | 0.3002 | **incumbent** | 631 | 2.06 | 0.85 | ours |
| diesel_spending | 0.2040 | 0.1903 | 0.1580 | **incumbent** | 390 | 2.29 | 0.81 | ours |

### ETB — the weight-basis question is closed

It was never undecidable; it was measured on the wrong frame. The stage's own
convention is explicit in code: donor SN 8856, **year 2023 only**, complete
cases on the **thirteen-column services subset** (4,199 rows), weighted by
**`hhold_adj_weight`**. The incumbent cleans an eighteen-column subset, so a
"donor share" taken off its frame has a different denominator — that
mismatch, not a weighting ambiguity, produced the three irreconcilable
candidates recorded earlier.

| column | donor (share) | incumbent | ours | closer | donor £/hh | incumbent | ours | closer |
|---|---|---|---|---|---|---|---|---|
| dfe_education_spending | 0.2794 | 0.000265 | 0.2258 | ours | 3,461 | 2 | 3,111 | ours |
| bus_subsidy_spending | 0.5255 | 0.3167 | 0.5554 | ours | 87 | 114 | 89 | ours |
| rail_subsidy_spending | 0.1652 | 0.1277 | 0.1422 | ours | 225 | 691 | 211 | ours |

The incumbent's education column is **degenerate**: 14 nonzero households in
52,846, £2 per household against a donor £3,461. That is decidable on any
weight basis, so these are signed as a **defect fix on the incumbent side**,
not a method preference. Rail is inside the band and needs no signature; it is
shown because it moves the same way.

The incumbent additionally divides each household total by household size and
stores the per-head figure in a household-entity column
(`policyengine_uk_data/datasets/imputations/services/etb.py`). That is a
second, independent defect — but it does **not** reconcile the levels (rail is
3.07× the donor even after it), so it is recorded as an observation rather
than as the explanation. **Worth an upstream issue alongside #467; not filed,
pending your call.**

## E5 · wealth — signed, carried forward

Adjudicated 2026-08-19 ("E5 is not required to reproduce the incumbent's
inflated totals; donor-benchmark evidence"). Re-measured through
`clean_was_household_table` over the pinned WAS R8 tab (15,128 rows, weighted
by `R8xshhwgt`), the ruling holds and the level evidence behind it is the more
dramatic surface.

| column | donor (share) | incumbent | ours | closer | donor £/hh | inc × | our × | closer |
|---|---|---|---|---|---|---|---|---|
| savings | 0.6072 | 0.6621 | 0.6107 | ours (0.0035 off) | 16,918 | 5.70 | 1.97 | ours |
| property_wealth | 0.6433 | 0.7082 | 0.6607 | ours | 249,247 | 1.03 | 0.98 | ours |
| other_residential_property_value | 0.0363 | 0.0750 | 0.0367 | ours (0.0004 off) | 9,422 | 6.18 | 1.26 | ours |
| main_residence_value | 0.6236 | 0.6761 | 0.6356 | ours | 212,344 | 1.02 | 0.88 | incumbent |
| corporate_wealth | 0.7629 | 0.8225 | 0.7792 | ours | 160,010 | 1.85 | 1.45 | ours |
| student_loan_balance | no like-for-like benchmark | 0.0197 | 0.0493 | — | — | — | — | — |

Ours is closer on **five of five** benchmarked shares and four of five levels.
`corporate_wealth`, previously "not measured", now has a benchmark — it is the
committed fold of five WAS aggregates, so it can be reconstructed on the donor
frame exactly as the stage builds it.

**A caveat that matters, and a correction.** The *unweighted* donor shares tell
the opposite story on incidence — the incumbent looks closer on all five. WAS
oversamples wealth-holders by design, so the unweighted frame is not a
population; the weighted basis is, and it is the basis used throughout this
document. The earlier "ours closer on 4/4" reading was on the weighted basis
and stands; a mid-review unweighted re-measurement briefly appeared to
overturn it and did not.

`student_loan_balance` is signed **on its own** rather than inside the wealth
class: it is a fold of two WAS loan aggregates at a different entity grain,
with no like-for-like donor share available. It is the one E5 column whose
direction is unevidenced, and the register says so.

`owned_land` is **not** in this list; its reviewed exclusion expiring
2026-09-20 is a separate question. For the record it is in band (−0.0026) and
ours is closer to the donor on both share (0.0071 donor · 0.0143 incumbent ·
0.0119 ours) and level (9.28× against 2.44×).

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

> **Two different level statistics — read the label.** This section is the
> **conditional mean**: £ per *carrier*, over nonzero rows only. The E5 and E6
> tables above are the **population mean per household**, over every household
> including zeros, as a ratio to the donor's. They answer different questions
> and can disagree on which side is closer — a build can put too little on each
> carrier while putting carriers on too many households, and come out right in
> aggregate. **The signing used the population mean**, because that is what a
> calibration target binds and what survives a difference in incidence. Where a
> verdict here differs from one above, the one above is operative.


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

## Universal Credit — pre-calibration health: healthy

UC is what the armed calibration binds (`dwp.uc.households`, CY-2025 avg
≈6.76m), so its health check is about the raw material the solve receives.
Modeled `universal_credit` materialized through the engine on both artifacts:

| measure | spine (pre-calibration) | incumbent (post-calibration) | read |
|---|---|---|---|
| modeled UC benunits, unweighted | 5,323 (8.70%) | 4,869 (7.95%) | ours richer, ratio 1.09 |
| modeled UC per recipient, annual £ | 10,285 | 10,130 | equal within 1.5% |
| reported UC share at `frs_spine` | 0.063261 | — | **exact** vs raw tab 0.063261 |
| reported UC per recipient, annual £ | 11,436 | 10,780 | admin per-unit ≈ £11.3k (#731) |
| modeled caseload, weighted | 3.15m | 6.30m | see below |
| modeled UC annual total, weighted | £30.6bn | £74.7bn | see below |

**Why the weighted rows halve, and why that is the boundary, not a bug.** The
spine carries design grossing weights (29.2m households); the incumbent's
weights are *calibrated*, and since 1.56.15 that calibration includes a UC
caseload target — its solve moved roughly 2× mass onto UC-positive benunits.
Comparing our design-weighted caseload to their calibrated one compares
before-medicine to after-medicine. Like-for-like (unweighted) the spine hands
calibration **more** raw material than the incumbent had, at an equal
per-recipient level; reaching 6.76m means lifting ~2.1× mass onto 8.7% of
benunits — the same order the incumbent's own solve performed — policed at the
armed run by the ESS and weight-ratio gates.

**Watch-item:** `would_claim_uc` is frozen at 0.55 by the U8 adjudication for
incumbent parity, with uk-data#452 (the take-up vintage behind #731's June UC
diagnosis) as the recorded follow-up. The freeze caps the eligible pool; it is
the pre-registered lever if the armed run's UC fit is strained. The
final-surface share gap on `universal_credit_reported` (+0.0069) is the E7
SPI-rewrite class — both builds rewrite synthetic rows, with different QRF
implementations.

## Open

- **The queue is signed.** All 26 beyond-band divergences carry register
  entries; whole-spine parity is `signed_parity`, `--strict` clean. The two
  items that blocked it are closed: the ETB weight basis (above) and the
  entity-count scope (signed to `person` and `benunit` only, so a future
  household-count divergence stays a defect).
- **The incumbent's ETB per-head division** — a second upstream defect of the
  #467 family, observed but not filed. Your call.
- **`communication_consumption`** is the one LCFS column where the incumbent is
  closer on level (1.16× against our 1.20×) while we are closer on share. It is
  signed inside the donor-faithful class; if that bothers you it should be
  scoped out, and that is a one-line register change.
- **`student_loan_balance`** carries no donor benchmark — signed on the
  standing E5 adjudication, direction unevidenced.
- **Upstream defect filed** as policyengine-uk-data#467 (no prescriptive fix);
  our side is correct and signed.

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
