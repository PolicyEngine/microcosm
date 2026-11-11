# microcosm#1095 receipts: the unblocked uk-data ports

Plan: `repos/microcosm-1095-ports-plan.md` (approved 2026-10-02). Branch `uk-data-ports-1095`. It was
stacked on `uk-spine-followups-1063` (#1089). After #1089 merged (`48fa06166`, 2026-10-04) it targets
main and is rebased onto it. The rebase replayed every commit; the only conflicts were the generated
coverage manifest and H2 fixture, which were regenerated after each replayed commit, and each commit's
other changes are line-for-line those of its pre-rebase original. The tracker is #1095 (uk-data items
up to uk-data#519).

Commits, in order (hashes on main):

- 1d1c10da3 C4a: the DWP and OBR table 4.9 benefit rows bind the households their publisher counts.
- 3c1b33f34 C4b: `obr.pip` binds the engine's `pip + dla`.
- 04ea9943b C5: Scottish Adult and Child Disability Payment codes on the FRS spine (uk-data#500).
- 90621f74a C1: FRS-reported dividends kept (uk-data#498).
- f0c30271d C2: SPI draws only for FRS claimants and partners (uk-data#504).
- 07e5b0f9c C3a: council tax before council tax reduction (uk-data#496/#499).
- 746940855 C3b: the Scottish water and sewerage charge paid (uk-data#499).
- 790f36ffd C6: the `spi_benefit_coherence` stage and its gate (uk-data#514).
- c58fc84d3 C7: SPI benefit-unit capital conditioned on investment income (uk-data#495).
- adfcc079e C8a: the WAS predictors measured alike on donor and recipient (uk-data#486/#495).
- 49d26eceb C8b: buy-to-let as other residential property (uk-data#501).
- 39af74a21 C9a: the ETB adult and child counts by FRS family role (uk-data#486).
- 8ede090d8 C9b: the constituency UC child bands on the national UC composition (uk-data#486).
- 7efa9d00b C10: the South East £12,570 to £15,000 income-tax deferral retired, because the ports
  make it stale.
- 1a4f934ce C11: the first version of this note.
- eaf7e80cf R1: income-related ESA zeroed on SPI rows (review round 1, item 3).
- d89a3ddfe R2: the Scottish water charge at the reduction scheme's combined maximum (item 4).
- 894a33c19 R2 test: the FRS mapping test's Scottish water value.
- a0c678c21 R3: the three input-mass exclusions the ports make stale, retired.
- R4: this update of the note.

C4c, the Housing Benefit pension-age spending row, was dropped. The DWP forecast-table facts carry
one `measure_id` per year (`expenditure_2024`, `expenditure_2025`), and `calendar_year_window`
needs both fiscal years from one series identity. The row needs a Chronicle series-identity change
or a ruling to bind FY2025-26 alone.

## Rulings (2026-10-02)

- Buy-to-let (`DVBltValR8_sum`) goes into `other_residential_property_value`.
- Scottish council tax follows two rules. The netting is uk-data#499's: the gross `CWATAMT1` +
  `CSEWAMT1`, in full for non-recipients and at 65% for reduction recipients. The charge paid is the
  gross charges less the status discount, at 65% for reduction recipients. Review round 1 refines the
  charge to the scheme's own formula (below).
- `obr.pip` binds PIP + DLA over England and Wales.
- The full SPI benefit-coherence stage, including the UC take-up redraw.

## Review round 1 (2026-10-04)

- **Income-related ESA (R1).** policyengine-uk pays `esa_income_reported` as the award, screened
  only by a capital test, so an SPI row keeping its stage-2 draw was paid its twin's award whatever
  its new incomes.
  - On the A8 spine at design weights, SPI rows held 0.15m of the 0.53m income-related ESA
    reporters (29%) and £1.45bn of reports; 56% of their benefit units had incomes above £12,570 and
    43% above £20,000.
  - `spi_benefit_coherence` now zeroes it with income-based JSA and Income Support, as uk-data#514
    does.
  - Contributory ESA keeps its draw: it is open to new claims and not income-tested, its incapacity
    test follows the person, its SPI draw sits at half the FRS rate per working-age adult, and 9.9%
    of its SPI reporters by weight earn above the permitted-work limit (41% on the uk-data build
    where #514 zeroes it).
- **Scottish water (R2).** The Scottish Government's charging principles for 2021-27 (Annex A) set
  a reduction recipient's water reduction at R = 35 × (A/B) − D points of the gross charge, unless
  negative, on top of the status discount D; A is the council tax reduction and B the council tax
  before it.
  - The combined reduction is max(D, 35% × A/B). C3b had stacked the 35% on the discount, so a
    single person on full reduction paid 48.75% of the gross charge instead of 65%; 207 of the 245
    Scottish recipients on the raw tab carry the 25% discount.
  - Every recipient takes the full-reduction case, A/B = 1. 86% of Scotland's 458,120 recipients
    receive full reduction (Council Tax Reduction in Scotland 2024-25, March 2025; average award
    £16.55 a week), while none of the 234 Scottish recipients with a recorded `CTREBAMT` on the FRS
    2024-25 tab pays nothing after it (19% of English recipients do), and their amounts have a
    median of £6 a week. The recorded amount cannot carry the share.
  - On the raw tab, recipients pay 0.649 of the gross charges instead of 0.511. The netting of
    `CTANNUAL` keeps 0.65.
- **Input-mass register (R3).** See Verification.
- **VAT.** No change here; see Fit by arm.


## Builds

Every arm is a licensed local build in the detached measurement tree `repos/populace-1095-shard`
(`data/ukds/acceptance/1095-uk-data-ports/scripts/arm.sh <commit> <name> calibrate`, queued by
`run_arms.sh`): the spine with the build_930 recipe (the five NTS tabs, `--was-person-tab`,
`--no-staging`), then the national calibration with local staging only, against the Chronicle
artifact 825406f. Nothing is released from an arm, and the engine is policyengine-uk at the repo's
pin.

The arms are cumulative. A0 to A7 ran on the pre-rebase chain on `747313ae7`. A0 is the control:
#1089 at `747313ae7`, calibrate-only on #1089's own stack-2 spine. A1 recalibrates that spine with
C4, since C4 only changes targets. A2 to A7 each build their own spine. A8 built the head rebased on
#1089's `45014f48c` (`9c7394086`, the ports without C10 and C11, which touch no spine code).

A9 builds `d89a3ddfe`: every port and both review fixes, on main. The commits after it (the test pin,
the input-mass register and this note) touch no spine or calibration code, so A9 measures the code
at the PR head. The control for A9 is main's own final #1089 build at `64c460b66`
(`runs/uk-national-1063-64c460b66`), whose spine and calibration code main carries unchanged.

Figures from the spines are at design weights. Counts resting on fewer than 10 records are
suppressed.

## Fit by arm

The national problem has 1,168 targets throughout. The gate failures listed come from the
calibrated-seam battery; arms A3 to A7 wrote their evidence bundle but no H5.

- **A0, control.** Loss 0.28290 to 0.00720; 98.3% of targets within 10%; ESS 4,937. All 7 gates
  pass.
- **A1, + C4.** Loss 0.00719; 98.3% within 10%; all 7 gates pass.
  - `obr.pip` design value £19.4bn becomes £23.6bn once DLA is bound with PIP; it calibrates on
    target either way.
  - The rescoped benefit rows keep a mean absolute error of 1.0%.
- **A2, + C5.** Loss 0.00716; 98.4% within 10%; all 7 gates pass.
- **A3, + C1 and C2.** Loss 0.00707; 98.5% within 10%. `uk_target_fit` fails twice:
  - `obr.vat` at +25.2%, just over the 25% bound (+22.8% at A0);
  - #1089's reviewed exclusion of `hmrc.spi_region.income_tax_by_region_12570_15000@E12000008@2025`
    is stale: the South East £12,570 to £15,000 income-tax cell calibrates at +8.0% (+31.3% at A0),
    and the gate requires stale exclusions to be removed.
  - All 16 rescoped benefit rows and all 4 ESA rows are within 10%.
- **A4, + C3a and C3b.** Loss 0.00703; 98.6% within 10%. Only the stale exclusion fails.
  - `obr.council_tax` is at −10.3% (−11.6% at A3): the design value rises from £44.1bn to £44.9bn.
- **A5, + C6 and C7.** Loss 0.00697; 98.5% within 10%.
  - `uk_target_fit` fails on `obr.vat` at +25.0% and on the stale exclusion.
  - `uk_weight_ratio` fails at 1,161.8 against the reviewed maximum of 1,151.3.
- **A6, + C8a and C8b.** Loss 0.00693; 98.5% within 10%. Only the stale exclusion fails.
  - `obr.council_tax` is at −9.9%, inside 10% for the first time; all 98 council-tax rows are
    within 10%.
- **A7, + C9a and C9b.** Loss 0.00702; 98.5% within 10%; ESS 4,795. Only the stale exclusion
  fails.
  - `obr.vat` is at +24.4%, the South East cell at +12.1%, and `obr.council_tax` at −9.9%.
- **A8, the rebased PR head.** Loss 0.28848 to 0.00693; 98.8% within 10%, the best of the arms;
  ESS 4,844.
  - The only gate failure is the stale South East deferral: the cell fits at +11.7%. C10 retires
    it, so the battery passes on the code this PR proposes.
  - `obr.vat` is at +24.998% against the 25% bound, and `obr.council_tax` at −9.9%.

- **A9, the review head on main.** Loss 0.29242 to 0.00715; 99.0% within 10%, the best of the
  arms, with 12 targets outside 10%; ESS 4,828. All 7 gates pass.
  - Against main's own build (loss 0.00713, 98.4% within 10%, 19 targets outside 10%, ESS 4,973):
    the South East £12,570 to £15,000 cell is at +11.5% (+30.3% on main, deferred there), and
    `obr.council_tax` at −9.9% (−11.1%).
  - `obr.vat` is at +24.4% (+23.1% on main): 0.6 points inside the bound.
  - Zeroing income-related ESA on SPI rows lowers the design `obr.esa` from £5.73bn (A8) to £4.80bn
    and the design `dwp.esa_income_claimants` from 404k to 313k. Calibration restores both: +0.4%
    and −0.1%. `dwp.esa_claimants` lands at +11.9% (+9.3% at A8, +11.0% on main), inside the gate
    bound.

`obr.vat` sits 1.6 to 2.4 points nearer its bound from A3 on than at A0. Design VAT falls from
£227bn (A0) to £224bn (A3) and £222bn (A7), while calibrated VAT rises from £218bn to £223bn and
£221bn. The closeness therefore comes from how calibration trades VAT against the income rows once
FRS dividends stand as reported.

At A8 VAT passed by 0.002 points (+24.998%); at A9 it passes by 0.6 points (+24.4%), against
+23.1% on main without the ports. The overshoot itself predates the ports: design VAT is 25% to 28%
over the OBR line in every arm and on main. Its level comes from the engine, which divides household
VAT by `microdata_vat_coverage`, 0.383 since 2010, where this dataset's consumption implies about
0.46 (PolicyEngine/policyengine-uk#1996, filed from #1063, where María ruled VAT an engine fix).

## Spine receipts by commit

- **C5 (A2).** Scottish PIP daily-living reporters per 1,000 residents rise from 13.2 to 43.9.
  Wales is at 61.3 and the North West at 54.7.
- **C1 (A3).** Dividends on the FRS channel fall from £40.5bn to £5.1bn, the respondents' own
  reports. The SPI channel goes from £55.5bn to £57.3bn.
  - Benefit units whose interest and dividends imply more than £16,000 at 4% while they hold
    £16,000 or less fall on the FRS channel from 1.62m to 0.21m.
  - The redrawn dividends had made that incoherence.
- **C2 (A3).** SPI-channel dependants aged 16 to 19 with earnings fall from 94.2% to none. Their
  earnings were £12.0bn; too few records keep any to report a figure.
  - Income band donor carriers who are dependants fall from 40 to 0.
  - The income stage keeps 692 dependant rows (0.84m people) on their twin's values.
- **C3a (A4).** Among households reporting a council tax reduction, the share with a £0 bill falls
  from 10.6% to 0, and the share billed below their own reduction from 13.8% to 0.
  - Council tax at design weights moves from £37.0bn to £38.3bn in England and from £2.08bn to
    £2.18bn in Wales.
  - Scotland falls from £3.21bn to £3.06bn, because the gross water and sewerage charges now come
    off the bill in full for non-recipients.
- **C3b (A4).** The mean Scottish water and sewerage charge rises from £404 to £465. England and
  Wales are at roughly £490.
- **C6 (A5).** At the stage, 435 SPI rows reported contributory or income-based JSA, Income
  Support, Working or Child Tax Credit, SDA or the Sure Start Maternity Grant; none do afterwards.
  - At design weights that was £2.50bn of reports, £0.81bn of it Child Tax Credit.
  - AFCS on SPI rows returns from £0.79bn to its twins' £0.21bn, bereavement support from £0.38bn
    to £0.30bn, and IIDB moves from £0.13bn to £0.15bn.
  - `receives_benefits_in_own_right` changes on 2,316 SPI rows (2.98m people), and no row
    disagrees with its own reports afterwards.
  - The UC take-up redraw sets 2,612 SPI units to claim and 2,422 not to, at the contract rate of
    0.55. Every SPI reporter unit claims, and no unit outside the UC age population claims without
    reporting.
  - `uc_capital_coherence`'s OR refresh becomes a no-op (0 units).
- **C7 (A5).** SPI units whose investment income implies more than £16,000 while they hold £16,000
  or less fall from 1.86m to 0.18m (above £6,000: from 1.94m to 0.14m).
  - 13,836 of 14,245 SPI units drew from their exact cell; 359 merged children bands, 50 merged
    adjacent income bands, and none fell back to reporter status alone.
  - Four SPI reporter units hold more than £16,000 after the redraw.
- **C8b (A6).** Households with rental income that hold other residential or non-residential
  property rise from 11.6% to 88.3%.
- **C9a and C9b.** In 1.8% of households the engine's age-18 adult count differs from the FRS
  family roles. Those households now feed the WAS and ETB predictors and the constituency UC bands
  the role counts. The national UC family-type measurement is byte-identical by construction.
- **R1 (A9).** No SPI row reports income-related ESA; on A8 the SPI rows carried 258 such records,
  £1.45bn at design weights. The FRS rows are unchanged (860 records, £3.58bn), and contributory ESA
  keeps its SPI draws (187 records, £0.58bn).
- **R2 (A9).** The mean Scottish water and sewerage charge at design weights rises from £465 (A4 to
  A8) to £474. On the raw tab the mean annual charge of a reduction recipient rises from £238 to
  £300 at FRS grossing weights.

## Verification

On main the PR gets CI. Locally, with the pinned feed set:

- **Touched surface on the rebased head (`d89a3ddfe`):** 95 test files, 2,047 tests. They are every
  test the commits change, every test importing a touched runtime module, test-support helper or
  tool, H2 parity, and the tests where #1089's later commits meet these: the national graph, target
  references, UC family, battery bindings, identity stability, terminal gates, release certification
  and weighted integrity.
  - 2,038 passed, 6 skipped and 3 failed.
  - `test_household_and_benunit_mapping_values_are_ported` pinned the Scottish water value at the
    stacked rule. It is fixed in 894a33c19, and it was CI's only engine-free failure.
  - `test_engine_is_reported_unavailable_without_the_uk_extra` needs a venv without the UK extra:
    an environment failure.
  - `test_committed_surfaces_regenerate_from_pinned_feed[national]` fails on main as it does here.
    The committed `target_references.json` (bf7549…) no longer regenerates (e8f829…). Neither the
    file nor its generator has changed since #1089's `2fe7d8bbb`, and CI skips the test because it
    has no licensed feed.
  - The other three failures listed on the old base pass on main, after #1089's later commits:
    national compile parity, the national graph's `uk_target_fit`, and the UC payment registry.
- **Gate digests:** `repin_digests --check` reports 0 of 12 moved after the rebase. R1 moves 5, for
  the gate's new zeroed column, and re-pins them in the same commit.
- **Input-mass register (R3).**
  - I ran the release-cut `input_mass_parity` gate with the certifier's own functions on A9's spine
    at its calibrated weights, against the v1.56.16 reference. The script is
    `data/ukds/acceptance/1095-uk-data-ports/scripts/input_mass_check.py`, and it reproduces the
    certifier's verdict on main's `64c460b66` build exactly.
  - The gate fails only on three stale exclusions, all inside the 452% fence:
    - income-based JSA at £97.7m (+316%);
    - Working Tax Credit at £149.6m (+103%);
    - the access fund at £258.6m (+62%).
  - With R3's register the gate passes, with `dfe_education_spending` its only exclusion.
  - The other columns the ports touch stay inside the fence: income-related ESA +5.0%, Income
    Support −71.6%, AFCS −78.9%, and `other_residential_property_value` −31.1% (−73.5% on main).
- `tools/build_uk_release_input_coverage_manifest.py --check` and `tools/ci_test_plan.py verify`:
  ok.

## Signature drafts

These are not committed:

- the new signed difference `council-tax-gross-of-reduction`;
- the amended `scottish-water-sewerage-successor-level`;
- the reworded `uc_unit_vs_household_grain` adjudication and its census fence.

The drafts are in the PR description.
