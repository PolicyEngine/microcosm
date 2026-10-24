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

`pilot.py` pins every input below by sha256 (`HF_SHA256`, `INPUT_SHA256`), reads
the Chronicle files at revision `200bf6d5` and refuses any other bytes. Each
run's `summary.json` records the sha256 of every input it used and the release
identity.

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

`M_s(w0)` is the state total of the measure under the release weights, so a
family's targets sum to the release's state totals. The targets only
redistribute each state total across its counties. They don't hold it: the
rows are soft constraints and only national household mass is conserved, so
achieved state totals can move. Each run reports that drift (below). A held-out
family whose source omits a county (SOI has no Kalawao row) takes both the
share and the state total over the counties it publishes.

| Family | Role | Measure (household weight) | Public source |
| --- | --- | --- | --- |
| `county_households` | train | 1 | ACS B11001_001E |
| `county_population` | train | persons in household | PEP V2024, July 2024 |
| `county_earnings` | train | earnings (below) | ACS B19061_001E |
| `cd_households`, `cd_population` | train (hold) | 1, persons | the district's own total under `w0` |
| `county_earning_households` | holdout | earnings ≠ 0 | ACS B19051_002E |
| `county_soi_wages` | holdout | wages (stored input) | SOI A00200 (no Kalawao row) |
| `county_income` | holdout | earnings + other stored income (approximate) | ACS B19025_001E |

**Earnings** is ACS B19061's concept, wages plus all self-employment, summed
over a household's people from stored inputs (`county_targets.household_measures`):

- `employment_income_before_lsr`, `self_employment_income_before_lsr` and
  `sstb_self_employment_income_before_lsr` on every record. The US build's QBI
  reconciliation (`us_runtime/qbi_inputs.py`) moves a person's whole Schedule
  C amount to the SSTB leaf when the business is an SSTB, so both leaves are
  needed.
- `farm_operations_income` on donor (`asec_puf`) records only. It comes from
  ASEC FRSE_VAL or PUF E02100 (`us_runtime/farm_business_income.py`). On ACS
  records the base self-employment leaf holds SEMP, which already includes
  farm self-employment.

Weighted national totals under the release weights (`out/extract_run1_report.json`):
wages $10,654B, base self-employment $468B, SSTB self-employment $456B, donor
farm $15B; earnings $11,593B. The household income approximation adds the 20
other income inputs the release stores, plus donor-only `partnership_income`
and `s_corp_income` (ACS SEMP already counts partnership income). TANF and
public assistance income are absent from the release.

Solver settings match the ACS-local release: Adam, learning rate 0.02,
conserved mass, `max_weight_ratio` 5 relative to the release weights,
`target_loss_cap` 1. The release weights are the warm start. Runs 2–4 relax
the two caps.

The pilot drops the release's other target families (SOI, national and state
admin totals) from this solve. The weight-ratio quantiles in the summary bound
how far the county rows moved records.

## Reliability grades (provisional)

Each county is graded by its Kish effective sample size under the calibrated
weights, after merging the rows of each source household (a donor household
cloned into two rows counts once): A ≥ 300, B ≥ 100, C ≥ 30, D below 30. The
thresholds are a starting point for the grading scheme in microcosm#922, not a
decision. Row-level ESS is kept as a diagnostic (`ess_rows_after`).

## Running

```bash
MODAL_PROFILE=policyengine modal run experiments/us-county-pilot-20261024/pilot.py --step extract
MODAL_PROFILE=policyengine modal run experiments/us-county-pilot-20261024/pilot.py \
  --step calibrate --label run3-losscap10 --max-weight-ratio 5 --target-loss-cap 10
modal volume get lwi-county-pilot run3-losscap10 experiments/us-county-pilot-20261024/out/
python experiments/us-county-pilot-20261024/report.py \
  experiments/us-county-pilot-20261024/out/run3-losscap10 <households.parquet>
```

Tests run in CI with the engine-free job:
`packages/microcosm-build/tests/engine_free/us/test_us_county_pilot.py`, on
small cuts of the pinned sources in
`packages/microcosm-build/tests/fixtures/us_county_pilot/`. They cover the
source readers, the CT mapping, share-target invariants, the donor-only
earnings concept, the design's state totals, clone invariance of ESS, and the
drift, baseline and published-level metrics.

## Results (2026-10-24)

Every run starts from the release weights and trains 10,290 rows for 800 epochs,
taking 14 to 18 minutes on Modal with the four runs in parallel. "Before" is
the release itself. Errors are absolute relative errors across counties (3,144
rows per county family, 436 per district family), measured against each row's
target: the county's published share of its state, scaled to the release's
state total (see "Target design"). `out/<run>/report.md` holds each run's full
tables (`report.py`). These runs supersede an earlier set that left SSTB
self-employment and donor farm income out of earnings.

| | Release | Run 1 | Run 2 | Run 3 | Run 4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Per-record weight cap (× release weight) | | 5 | 20 | 5 | 20 |
| Target loss cap | | 1 | 1 | 10 | 10 |
| **Trained, share of counties within 10%** | | | | | |
| Households (B11001) | 13.7% | 92.8% | 97.9% | 94.2% | 99.1% |
| Population (PEP V2024) | 13.5% | 90.4% | 97.4% | 91.2% | 98.4% |
| Aggregate earnings (B19061) | 9.5% | 89.7% | 95.4% | 93.0% | 98.9% |
| District households, held | 100% | 96.6% | 97.7% | 95.9% | 96.8% |
| **Achieved state totals vs release, median (max) drift** | | | | | |
| Households | | 0.2% (2.5%) | 0.2% (2.4%) | 0.2% (2.1%) | 0.1% (2.3%) |
| Population | | 0.1% (1.5%) | 0.1% (1.7%) | 0.2% (1.9%) | 0.1% (0.6%) |
| Earnings | | 1.2% (7.9%) | 1.3% (8.0%) | 0.1% (1.3%) | 0.0% (0.3%) |
| **Weights** | | | | | |
| Total Kish ESS, by source household | 10,995 | 11,472 | 10,826 | 23,038 | 19,617 |
| Total Kish ESS, by row | 13,631 | 13,753 | 12,944 | 27,570 | 23,568 |
| Donor (asec_puf) weight share | 68.0% | 63.6% | 65.2% | 47.2% | 50.7% |
| Median county ESS, among the 3,140 counties with records | 8.0 | 11.6 | 11.1 | 13.1 | 12.5 |
| Counties graded A / B / C / D | 2 / 41 / 456 / 2,645 | 0 / 22 / 460 / 2,662 | 0 / 15 / 429 / 2,700 | 3 / 72 / 590 / 2,479 | 2 / 51 / 560 / 2,531 |
| Population share in A–C counties | 48.3% | 39.1% | 35.5% | 61.6% | 57.9% |
| Counties of 250,000+ graded D (of 292) | 169 | 209 | 221 | 98 | 121 |

The largest state drift in runs 1 and 2 is Wyoming's earnings (+7.9% and
+8.0%). The looser loss cap all but removes earnings drift.

The median record's weight ratio (calibrated over release weight) is 0.41 in
run 1, 0.27 in run 2, 2.3 in run 3 and 1.9 in run 4. Under conserved mass, a
median above 1 means weight moved from the few donor records to the many ACS
records. Run 4 lets a few records rise further: its 99.9th percentile ratio is
19.7, against run 3's cap of 5.

### Held-out families against a rescaling baseline

The three held-out families never enter the solve, but they describe the same
counties as the trained families and are closely related measures. To show
what reweighting adds, each is compared with a baseline that changes no
weights. The baseline multiplies each county's release value by its trained
driver's target-to-release ratio: households for households with earnings,
earnings for SOI wages and household income.

| Median county error | Release | Baseline | Run 1 | Run 2 | Run 3 | Run 4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Households with earnings (B19051) | 58.0% | 12.6% | 10.4% | 10.1% | 10.1% | 9.9% |
| Wages (SOI A00200) | 58.0% | 8.4% | 8.3% | 8.1% | 8.0% | 8.1% |
| Household income (B19025, approximate concept) | 56.9% | 17.6% | 13.8% | 13.5% | 12.4% | 12.0% |
| **Share of counties within 10%** | | | | | | |
| Households with earnings | 12.2% | 42.1% | 48.0% | 49.6% | 49.6% | 50.3% |
| Wages | 9.0% | 57.6% | 56.6% | 58.2% | 58.2% | 59.7% |
| Household income | 9.3% | 28.6% | 36.0% | 36.7% | 40.9% | 41.4% |

Most of the fall from about 57% comes from getting each county's size right,
which the baseline does too. Reweighting adds a little on households with
earnings, nothing beyond the baseline on SOI wages, and the most on household
income, where it narrows the median error by 4–6 points.

### Against published levels

Because the targets are shares of the release's own state totals, the tables
above measure distribution within states, not agreement with published county
levels. The release's state totals differ from the published state sums by a
median of 6.5% for households, 1.9% for population and 3.9% for earnings.
Nationally the release has 5.2% fewer households, 2.6% less population and
0.1% less earnings. Measured against the raw published county levels instead:

| Share of counties within 10% of the published level | Release | Run 1 | Run 2 | Run 3 | Run 4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Households (B11001) | 12.9% | 81.8% | 87.3% | 84.0% | 89.4% |
| Population (PEP V2024) | 13.4% | 89.1% | 97.1% | 89.9% | 98.2% |
| Aggregate earnings (B19061) | 9.0% | 88.9% | 95.2% | 92.3% | 98.8% |
| Households with earnings (B19051), held out | 12.1% | 43.2% | 43.1% | 47.9% | 47.8% |
| Household income (B19025), held out | 10.4% | 45.5% | 47.0% | 46.6% | 48.0% |

SOI wages are TY2022 dollars against 2024 weights, so their level error mixes
in two years of wage growth: the release's national total is 10.6% above
SOI's. Median county error falls from 56% to 13.1–13.8%. The approximate
income concept runs 12.7% above ACS B19025 nationally; the median county's
approximate weighted median household income is 32% from ACS B19013 before
calibration and 22–23% after.

### Findings

1. **The release's counties are placeholders, not estimates.** Each household's
   county is a population-weighted draw inside its PUMA, and the weights were
   never calibrated below the district. The median county is 54–57% off on
   households, population and earnings. Calibration brings 89.7–99.1% of
   counties within 10% of their within-state share targets on every trained
   family (81.8–98.8% against published levels). District holds stay within
   about 0.2% at the median, and achieved state totals within 0.0–1.3% at the
   median.
2. **Held-out families improve, mostly through county size.** Median county
   error on households with earnings, SOI wages and household income falls
   from 57–58% to 8–14%. A rescaling baseline that changes no weights gets to
   8–18%, so this is evidence that the trained county totals are right, and
   only partly evidence of better within-county composition: reweighting
   beats the baseline on household income and households with earnings, and
   matches it on wages.
3. **The release's weight concentration bounds county reliability.** The
   57,240 donor rows (47,277 source households, 3.6% of records) carry 68% of
   the release's weight: mean weight 1,456, against 25.6 for the 1.53M ACS
   records. In Los Angeles County, 636 donor rows hold 69% of the county's
   weight with a source-household Kish ESS of 170, while 39,230 ACS records
   hold 31%. Most counties' precision is set by a handful of heavy donor
   records, and that reaches large counties too: 169 of the 292 counties with
   250,000 or more people have an ESS below 30 under the release weights.
4. **The release's loss cap hides the hardest counties.** Capped MAPE gives a
   row no gradient once its error passes the cap. 297 counties (9.4%) start
   more than 100% off on households and 393 on earnings. Under the release's
   cap of 1 (run 1), 49 counties are still more than 100% off on households
   after 800 epochs (0.15% of the population) and 117 on earnings (1.8%). A
   cap of 10 (run 3) leaves one county, Culberson TX (10 records), in each
   family. It also shifts weight from donor to ACS records: the donor weight
   share falls from 68.0% to 47.2%, source-household ESS rises 2.1 times, and
   the population share in counties with ESS of 30 or more rises from 48% to
   62%.
5. **The 5× per-record cap binds.** Raising it to 20× (run 2) lifts the
   trained families to 95–98% within 10%, but ESS falls slightly. With the
   looser loss cap as well (run 4), 98–99% are within 10% and source-household
   ESS is 1.8 times the release's, short of run 3's 2.1.
6. **Four counties have no records:** Arthur NE (31009), Kenedy TX (48261),
   King TX (48269) and Loving TX (48301), about 1,050 people. Two more have
   records but none with earnings: Kalawao HI (15005) and Borden TX (48033).
   SOI publishes no Kalawao row.

### Limits of this pilot

- The solve holds only district households and population. It omits the
  release's national, state and district SOI and administrative families, so
  the donor-to-ACS shift in runs 3 and 4 may cost fit on tax targets (top
  incomes, capital gains) that donor records carry. Only a solve that trains
  county rows with every release family can measure that, and training them
  together doesn't by itself guarantee either fit: it needs measured
  acceptance criteria.
- The held-out families are related to the trained ones and describe the same
  counties, so they test consistency, not independent generalization.
- The county rows are 2020–2024 ACS and V2024 PEP shares applied to 2024 state
  totals. The deliverable's 2026 and 2027 files need aged state totals.
- The income concept for B19025 and the median comparison is approximate. It
  lacks TANF and public assistance income, and donor records carry no SSI.
- The reliability grades are provisional ESS bands.
- County rows rescale published shares to the release's state totals. The fit
  tables measure within-state distribution; the "Against published levels"
  table measures what a user comparing to ACS would see.

### For the production rung

- Train county rows with every release family. Report the donor weight share,
  the tax-target fit and achieved state totals beside county fit and county
  ESS, and set acceptance thresholds for each.
- Use a loss cap well above 1 for county rows, or warm-start them in stages.
  The release's cap leaves about 9% of counties without gradient at the start.
- Guarantee support for every county. The four empty counties need records,
  for example donor households cloned from their PUMA at split weight.
- Treat donor concentration as a design choice. Run 3 cuts the donor weight
  share from 68% to 47% and doubles source-household ESS, but the tax-target
  cost must be measured first.
