# US county calibration pilot (2026-10-24)

Tracking issue: microcosm#922, the LWI County Files. That deliverable is US
microdata calibrated to every county-equivalent in its 2023 geography: 3,144,
with Connecticut as its nine planning regions.

This pilot asks whether the published ACS-local release can carry county
distributions. It runs one recalibration with Microcosm's own calibrator on the
real release, adding county target rows from public sources, then measures fit,
weight movement, and per-county effective sample size. It is an experiment, not
a release. The production rung (county rows inside
`tools/build_us_acs_local_release.py`, 2026/2027 periods, ACS county tables as
Chronicle source packages) is separate work.

## Inputs

| Input | Identity |
| --- | --- |
| Release microdata | `policyengine/populace-us` @ `fb8ba90e7f576ad0bb3f29b5aff2d1ec7007e912`, `populace_us_2024_acs_local.h5`, sha256 `769756c3…a7ec` (build `buildo-acs-local-767312d60`, 2026-09-23) |
| ACS 2020–2024 5-year county tables (B01003, B11001, B19001, B19013, B19025, B19051, B19061) | Census API `data/2024/acs/acs5`, `group(<table>)`, `for=county:*`, fetched 2026-10-24; sha256 in `_build_artifacts/lwi-county-20261024/census_api_acs5_2024_county/SHA256SUMS` |
| PEP Vintage 2024 county population | Chronicle `db/data/census/pep_county_2024/co-est2024-alldata.csv` @ `200bf6d5`, sha256 `abcc8720…22b9` |
| IRS SOI county TY2022 | Chronicle `db/data/irs_soi/county_2022/22incyallnoagi.csv` @ `200bf6d5`, sha256 `2c04ff23…3343` |
| CT county-to-town crosswalk | Census `ct_cou_to_cousub_crosswalk.txt`, sha256 `13ad0025…cff8` |
| CT 2020 block assignment, town layer | Census BAF `BlockAssign_ST09_CT_MCD.txt`, sha256 `fe56bf38…b653` |
| 2020 tract to 2020 PUMA | Census `rel2020/2020_Census_Tract_to_2020_PUMA.txt`, sha256 `5262f460…0531` |
| 2020 CT tract population | Census API `data/2020/dec/pl` `P1_001N`, sha256 `2b31267f…aa2` |

## Geography

The release assigns each household a 2020 county inside its PUMA, drawn from
population-weighted PUMA overlaps. Only Connecticut differs between the 2020
and 2023 county sets. Each CT household gets its 2022 planning region from its
2020 tract where it carries one; tracts nest in towns, and towns nest in
regions. Otherwise the region comes from its PUMA and 2020 county.

Every CT (PUMA, 2020 county) cell lies inside one planning region except one:
Tolland County's part of PUMA 20301. That cell holds 12,257 people in 2020;
785 of them (the town of Union) are in the Northeastern region and 11,472 in
the Capitol region. Households in that cell are drawn by those populations with
a fixed seed, the way the PUMA ladder draws counties. All three public county
sources already publish CT as planning regions.

## Target design

Every county row is a within-state share target:

```
target(c) = M_s(w0) * public(c) / sum over counties c' in state s of public(c')
```

`M_s(w0)` is the state total of the measure under the release weights. State
totals stay where the release calibration put them, and the county rows only
redistribute each total across its counties.

| Family | Role | Measure (household weight) | Public source |
| --- | --- | --- | --- |
| `county_households` | train | 1 | ACS B11001_001E |
| `county_population` | train | persons in household | PEP V2024, July 2024 |
| `county_earnings` | train | wages + self-employment (stored inputs) | ACS B19061_001E |
| `cd_households`, `cd_population` | train (hold) | 1, persons | the district's own total under `w0` |
| `county_earning_households` | holdout | earnings ≠ 0 | ACS B19051_002E |
| `county_soi_wages` | holdout | wages (stored input) | SOI A00200 (no Kalawao row) |
| `county_income` | holdout | earnings + stored income inputs (approximate) | ACS B19025_001E |

Solver settings match the ACS-local release: Adam, learning rate 0.02,
conserved mass, `max_weight_ratio` 5 relative to the release weights,
`target_loss_cap` 1. The release weights are the warm start.

The pilot drops the release's other target families (SOI, national and state
admin totals) from this solve. The weight-ratio quantiles in the summary bound
how far the county rows moved records. The production rung trains the county
rows together with every release family, so the release fit is kept.

## Reliability grades (provisional)

Each county is graded by its Kish effective sample size under the calibrated
weights: A ≥ 300, B ≥ 100, C ≥ 30, D below 30. The thresholds are a starting
point for the grading scheme in microcosm#922, not a decision.

## Running

```bash
modal run experiments/us-county-pilot-20261024/pilot.py --step extract
modal run experiments/us-county-pilot-20261024/pilot.py --step calibrate --label run1
```

Tests (sources, CT mapping, share-target invariants):

```bash
cd experiments/us-county-pilot-20261024
uv run --no-project --with numpy --with pandas --with scipy --with pytest \
  --with hypothesis pytest --noconftest -c /dev/null --rootdir . test_county_targets.py
```

## Results (2026-10-24)

Every run starts from the release weights and trains 10,290 rows for 800 epochs
(Adam, learning rate 0.02, conserved mass), taking 9 to 13 minutes on Modal.
"Before" is the release itself. Errors are absolute relative errors across
counties (3,144 rows per county family, 436 per district family), measured
against each row's target: the county's published share of its state, scaled
to the release's state total (see "Target design"). The `out/<run>/report.md`
tables come from `report.py`.

| | Release | Run 1 | Run 2 | Run 3 | Run 4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Per-record weight cap (× release weight) | | 5 | 20 | 5 | 20 |
| Target loss cap | | 1 | 1 | 10 | 10 |
| **Trained, share of counties within 10%** | | | | | |
| Households (B11001) | 13.7% | 93.0% | 97.9% | 94.3% | 99.2% |
| Population (PEP V2024) | 13.5% | 90.5% | 97.3% | 91.3% | 98.3% |
| Aggregate earnings (B19061) | 9.5% | 89.0% | 95.4% | 92.2% | 98.7% |
| District households, held | 100% | 96.6% | 97.7% | 95.9% | 96.8% |
| **Held out, median county error** | | | | | |
| Households with earnings (B19051) | 58.8% | 10.7% | 10.4% | 10.6% | 10.3% |
| Wages (SOI A00200) | 58.0% | 7.5% | 7.2% | 7.2% | 7.0% |
| Aggregate household income (B19025, approximate concept) | 56.2% | 12.0% | 11.5% | 10.7% | 10.4% |
| **Held out, share of counties within 10%** | | | | | |
| Households with earnings | 12.4% | 47.1% | 48.4% | 48.1% | 49.0% |
| Wages | 9.0% | 61.1% | 62.5% | 62.8% | 64.1% |
| Aggregate household income | 9.7% | 42.7% | 44.5% | 47.1% | 48.7% |
| **Weights** | | | | | |
| Total Kish ESS | 13,631 | 13,915 | 12,997 | 27,407 | 23,611 |
| Donor (asec_puf) weight share | 68.0% | 63.4% | 65.1% | 47.3% | 50.6% |
| Median county ESS | 8.0 | 11.6 | 11.2 | 13.2 | 12.6 |
| Counties graded A / B / C / D | 2 / 41 / 457 / 2,644 | 0 / 26 / 464 / 2,654 | 0 / 17 / 437 / 2,690 | 5 / 66 / 589 / 2,484 | 2 / 51 / 559 / 2,532 |
| Population share in A–C counties | 48.3% | 39.6% | 36.1% | 61.5% | 58.2% |
| Counties of 250,000+ graded D (of 292) | 169 | 208 | 218 | 98 | 118 |

Run 4 combines both relaxations. It fits the trained families best and keeps
most of run 3's donor-to-ACS shift, but lets a few records rise further: the
99.9th percentile of the weight ratio (calibrated over release weight, across
records) is 19.7, against run 3's cap of 5. The median record's ratio is 0.43
in run 1, 0.28 in run 2, 2.3 in run 3 and 1.9 in run 4; under conserved mass,
a median above 1 means weight moved from the few donor records to the many ACS
records.

### Against published levels

Because the targets are shares of the release's own state totals, the table
above measures distribution within states, not agreement with published county
levels. The release's state totals differ from the published state sums by a
median of 6.5% for households, 1.9% for population and 6.3% for earnings
(nationally the release has 5.2%, 2.6% and 4.1% less). Measured against the
raw published county levels instead:

| Share of counties within 10% of the published level | Release | Run 1 | Run 2 | Run 3 | Run 4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Households (B11001) | 12.9% | 81.8% | 86.7% | 84.1% | 89.4% |
| Population (PEP V2024) | 13.4% | 89.0% | 97.0% | 89.9% | 98.2% |
| Aggregate earnings (B19061) | 8.4% | 82.2% | 89.1% | 86.2% | 93.1% |
| Households with earnings (B19051), held out | 12.2% | 44.1% | 45.0% | 47.6% | 48.2% |
| Aggregate household income (B19025), held out | 10.3% | 45.5% | 47.6% | 49.7% | 51.2% |

SOI wages are TY2022 dollars against 2024 weights, so their level error mixes
in two years of wage growth (the release's national total is 10.6% above
SOI's): median county error falls from 56.2% to 12.4–12.8%. Closing the
state-total gap is the production rung's job, where the county rows train
together with the release's state families.

### Findings

1. **The release's counties are placeholders, not estimates.** Each household's
   county is a population-weighted draw inside its PUMA, and the weights were
   never calibrated below the district. The median county is about 55% off on
   households, population and earnings. Calibration brings 89–99% of counties
   within 10% of their within-state share targets on every trained family
   (82–98% against published levels), and the district holds stay within
   about 0.2% at the median.
2. **Fit generalizes to measures the solve never saw.** Median county error on
   households with earnings, SOI wages and aggregate income falls from 56–59%
   to 7–12% in every run. That is the out-of-sample test that the county
   distribution is real, not memorized.
3. **The release's weight concentration bounds county reliability.** The
   57,240 donor households (3.6% of records) carry 68% of the release's weight
   (mean 1,456, against 25.6 for the 1.53M ACS records). In Los Angeles County
   636 donor records hold 69% of the county's weight and have a Kish ESS of
   177, while 39,230 ACS records hold 31%. Most counties' precision is set by a
   handful of heavy donor records, and that reaches large counties too: 169 of
   the 292 counties with 250,000 or more people have a Kish ESS below 30 under
   the release weights.
4. **The release's loss cap hides the hardest counties.** Capped MAPE gives a
   row no gradient once its error passes the cap. 297 counties (9.4%) start
   more than 100% off on households. Under the release's cap of 1, 52 counties
   (0.15% of the population) are still more than 100% off on households after
   800 epochs, and 111 on earnings. A cap of 10 (run 3) leaves one county,
   Culberson TX (10 records), in each family. It also shifts weight from donor to ACS records
   (donor share 68% → 47%), which doubles total ESS and lifts the population
   share in ESS ≥ 30 counties from 48% (release) to 62%.
5. **The 5× per-record cap binds.** Raising it to 20× (run 2) lifts the
   trained families to 95–98% within 10%, but ESS falls slightly. With the
   looser loss cap as well (run 4), 98–99% are within 10% and total ESS is 1.7
   times the release's, short of run 3's 2.0.
6. **Four counties have no records:** Arthur NE (31009), Kenedy TX (48261),
   King TX (48269) and Loving TX (48301), about 1,050 people. Two more have
   records but none with earnings: Kalawao HI (15005) and Borden TX (48033).
   SOI publishes no Kalawao row.

### Limits of this pilot

- The solve holds only district households and population. It omits the
  release's national, state and district SOI and administrative families, so
  the donor-to-ACS shift in run 3 may cost fit on tax targets (top incomes,
  capital gains) that donor records carry. Only the production solve, which
  trains county rows with every release family, can measure that.
- The county rows are 2020–2024 ACS and V2024 PEP shares applied to 2024 state
  totals. The deliverable's 2026 and 2027 files need aged state totals.
- The income concept for B19025 and the median comparison is approximate
  (earnings plus the stored income inputs). The partnership, TANF and public
  assistance inputs are absent from the release.
- The reliability grades are provisional ESS bands.
- County rows rescale published shares to the release's state totals, which
  sit a median 6.5% from the published state sums for households and
  earnings. The fit tables measure within-state distribution; the
  "Against published levels" table measures what a user comparing to ACS
  would see.

### For the production rung

- Train county rows with every release family. Report the donor weight share
  and the tax-target fit beside county fit and county ESS.
- Use a loss cap well above 1 for county rows (or warm-start them in stages);
  the release's cap leaves about 9% of counties without gradient at the start.
- Guarantee support for every county: the four empty counties need records,
  for example donor households cloned from their PUMA at split weight.
- Treat donor concentration as a design choice. Run 3 shows county
  calibration can halve it, but the tax-target cost must be measured first.
