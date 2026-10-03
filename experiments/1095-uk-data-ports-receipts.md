# microcosm#1095 receipts: the unblocked uk-data ports

Plan: `repos/microcosm-1095-ports-plan.md` (approved 2026-10-02). Branch `uk-data-ports-1095`,
stacked on `uk-spine-followups-1063` (#1089), rebased onto its head `45014f48c` without conflicts.
The tracker is #1095 (uk-data items up to uk-data#519).

Commits, in order (hashes after the rebase):

- f5fffc55f C4a: the DWP and OBR table 4.9 benefit rows bind the households their publisher counts.
- 7acd6f7aa C4b: `obr.pip` binds the engine's `pip + dla`.
- 7c44e7f49 C5: Scottish Adult and Child Disability Payment codes on the FRS spine (uk-data#500).
- d90450ae6 C1: FRS-reported dividends kept (uk-data#498).
- 6444dc8f6 C2: SPI draws only for FRS claimants and partners (uk-data#504).
- 22bfc7e81 C3a: council tax before council tax reduction (uk-data#496/#499).
- 856de4abe C3b: the Scottish water and sewerage charge paid (uk-data#499).
- 0fb1e1179 C6: the `spi_benefit_coherence` stage and its gate (uk-data#514).
- d55f317da C7: SPI benefit-unit capital conditioned on investment income (uk-data#495).
- 606f13793 C8a: the WAS predictors measured alike on donor and recipient (uk-data#486/#495).
- f07c779af C8b: buy-to-let as other residential property (uk-data#501).
- 9b9243078 C9a: the ETB adult and child counts by FRS family role (uk-data#486).
- 9c7394086 C9b: the constituency UC child bands on the national UC composition (uk-data#486).
- a4fb0e303 C10: the South East £12,570 to £15,000 income-tax deferral retired, because the ports
  make it stale.
- C11: this note.

C4c, the Housing Benefit pension-age spending row, was dropped. The DWP forecast-table facts carry
one `measure_id` per year (`expenditure_2024`, `expenditure_2025`), and `calendar_year_window`
needs both fiscal years from one series identity. The row needs a Chronicle series-identity change
or a ruling to bind FY2025-26 alone.

## Rulings (2026-10-02)

- Buy-to-let (`DVBltValR8_sum`) goes into `other_residential_property_value`.
- Scottish council tax follows two rules. The netting is uk-data#499's: the gross `CWATAMT1` +
  `CSEWAMT1`, in full for non-recipients and at 65% for reduction recipients. The charge paid is the
  gross charges less the status discount, at 65% for reduction recipients.
- `obr.pip` binds PIP + DLA over England and Wales.
- The full SPI benefit-coherence stage, including the UC take-up redraw.

## Builds

Every arm is a licensed local build in the detached measurement tree `repos/populace-1095-shard`
(`data/ukds/acceptance/1095-uk-data-ports/scripts/arm.sh <commit> <name> calibrate`, queued by
`run_arms.sh`): the spine with the build_930 recipe (the five NTS tabs, `--was-person-tab`,
`--no-staging`), then the national calibration with local staging only, against the Chronicle
artifact 825406f. Nothing is released from an arm, and the engine is policyengine-uk at the repo's
pin.

The arms are cumulative and ran on the pre-rebase chain on `747313ae7`. A0 is the control: #1089 at
`747313ae7`, calibrate-only on #1089's own stack-2 spine. A1 recalibrates that spine with C4, since
C4 only changes targets. A2 to A7 each build their own spine. A8 builds the rebased PR head
`9c7394086`, which adds #1089's three new commits.

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

`obr.vat` sits 1.6 to 2.4 points nearer its bound from A3 on than at A0. Design VAT falls from
£227bn (A0) to £224bn (A3) and £222bn (A7), while calibrated VAT rises from £218bn to £223bn and
£221bn. The closeness therefore comes from how calibration trades VAT against the income rows once
FRS dividends stand as reported.

At the rebased head VAT passes by 0.002 points (+24.998%), so any later spine change can tip it.
The overshoot itself predates the ports: design VAT is 25% to 28% over the OBR line in every arm,
the control included.

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

## Verification

Stacked microcosm PRs get no CI, so this local evidence is the record. The pinned feed was set
throughout.

- **Touched surface:** 74 test files and 1,452 tests: every test the commits change; every test
  importing a touched runtime module or its test-support helper; H2 parity; and the integration-uk
  lane. The incumbent-name guard ran too.
  - 1,440 passed, 7 skipped and 5 failed.
  - One failure, the E6 receipt fixture without `is_uc_claimant`, is fixed in C9a.
  - The other four fail identically on #1089's head `45014f48c`:
    - the national target-references regeneration;
    - the national compile parity, whose regenerated resources are byte-identical on both trees;
    - the national graph test's `uk_target_fit`;
    - the UC payment registry count of 81 against 83.
- `tools/ci_test_plan.py verify`, the coverage-manifest `--check`, and the WAS and LISA
  support-bounds `--check` on the licensed tabs all pass.
- C10 is checked by the terminal-gate, country-spec, battery-binding and data-contract tests: 303
  passed in the binding and contract lanes. The gate digests are unchanged (`repin_digests --check`:
  0 of 12 moved).

## Signature drafts

These are not committed:

- the new signed difference `council-tax-gross-of-reduction`;
- the amended `scottish-water-sewerage-successor-level`;
- the reworded `uc_unit_vs_household_grain` adjudication and its census fence.

The drafts are in the PR description.
