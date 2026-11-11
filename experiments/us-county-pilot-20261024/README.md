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
run's `summary.json` records the sha256 of its calibration inputs, and under
`extract` the release identity, the pinned sha256 of the four Connecticut files
and the sha256 of the extract's `households.parquet` and `lean.h5`.

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
2020 tract where it carries one, through the towns that tract's blocks sit in
(2020 block assignment file) and the regions those towns belong to. 18 of the
883 tracts span more than one town, each within one region; the extract
refuses a tract that spans two regions. Otherwise the region comes from its PUMA and 2020
county.

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
rows are soft constraints and the solver conserves only total record weight
(households and group-quarters records together), so achieved state totals,
and the national household total, can move. Each run reports that drift (below). A held-out
family whose source omits a county (SOI has no Kalawao row) takes both the
share and the state total over the counties it publishes.

| Family | Role | Measure (household weight) | Public source |
| --- | --- | --- | --- |
| `county_households` | train | 1 per household (not group quarters) | ACS B11001_001E |
| `county_population` | train | persons, group quarters included | PEP V2024, July 2024 |
| `county_earnings` | train | earnings (below), households only | ACS B19061_001E |
| `cd_households`, `cd_population` | train (hold) | households, persons | the district's own total under `w0` |
| `county_earning_households` | holdout | households with earnings ≠ 0 | ACS B19051_002E |
| `county_soi_wages` | holdout | wages (stored input), everyone | SOI A00200 (no Kalawao row) |
| `county_income` | holdout | earnings + other stored income (approximate), households only | ACS B19025_001E |

ACS household tables exclude group quarters and PEP counts everyone, so the
release's ACS group-quarters records (`TYPEHUGQ` 2 and 3: 183,206 records,
1.03 million weighted, 0.8% of the release's 122.5 million weighted records)
count toward population and SOI wages only. The median-income comparison also
uses households only.

**Earnings** is ACS B19061's concept, wages plus all self-employment, summed
over a household's people from stored inputs (`county_targets.household_measures`):

- On ACS records: `employment_income_before_lsr` and
  `self_employment_income_before_lsr`. The base leaf is `SEMP * ADJINC / 1e6`
  (`operator_boundary.py`), which already includes farm and partnership
  self-employment. The extract refuses to run unless that identity holds to
  the dollar on every ACS person, and it does on all 3,422,888.
- On donor (`asec_puf`) records, also `sstb_self_employment_income_before_lsr`
  and `farm_operations_income`. On the PUF tax-detail channel the QBI
  reconciliation (`us_runtime/qbi_inputs.py`) moves a person's whole Schedule C
  amount to the SSTB leaf when the business is an SSTB, so both leaves are
  needed; farm self-employment is ASEC FRSE_VAL or PUF E02100
  (`us_runtime/farm_business_income.py`). On ASEC-channel donors the base leaf
  equals `SEMP_VAL` on every record and the SSTB leaf is zero.

The ACS records' SSTB and farm leaves are not added: they are donor draws
stacked on top of SEMP. The income audit (`pilot.py --step audit`,
`out/audit_semp_report.json`) shows the base leaf equal to SEMP on every ACS
person while 123,226 of them also carry an SSTB value, $202.6B under the
release weights. An earlier version of this pilot added that leaf and so
counted those dollars twice; the release itself carries the same stacking into
PolicyEngine-US, whose `total_self_employment_income` adds both leaves
(microcosm#1380).

Weighted national totals over all records, group quarters included, under the
release weights (`out/extract_run1_report.json`): wages $10,654B, ACS base
self-employment $158B, donor base and SSTB self-employment $563B, donor farm
$15B; earnings $11,391B, of which $11,382B is in the household universe the
earnings rows use. Two gaps remain.
Donor earnings omit `partnership_self_employment_net_earnings` ($53.0B, all on
the PUF channel, 0.7% of donor earnings), which ACS SEMP counts; adding it
would count it twice in the income approximation, where donor
`partnership_income` already holds most of it. And `s_corp_income` is zero on
every record (the release keeps combined income in `partnership_income`). The
household income approximation adds the 20 other income inputs the release
stores, plus donor-only `partnership_income` (ACS SEMP already counts
partnership income). TANF and public assistance income are absent from the
release.

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
source readers, the CT mapping, share-target invariants, the earnings concept
(ACS base self-employment must equal SEMP; donor-only leaves never reach ACS
records), group quarters leaving the household universe, the design's state
totals, clone invariance of ESS, and the drift, baseline and published-level
metrics.

## Results (2026-10-24)

Every run starts from the release weights and trains 10,290 rows for 800 epochs,
taking 17 to 24 minutes on Modal with the four runs in parallel. "Before" is
the release itself. Errors are absolute relative errors across counties (3,144
rows per county family, 436 per district family), measured against each row's
target: the county's published share of its state, scaled to the release's
state total (see "Target design"). `out/<run>/report.md` holds each run's full
tables (`report.py`).

These runs supersede two earlier sets, kept in `out/v1/` and `out/v2/`. The
first left SSTB self-employment and donor farm income out of earnings. The
second added the ACS records' SSTB leaf, counting about $203B twice (see
"Target design"), and counted group-quarters records as households. The second
set showed the looser loss cap moving weight from donor to ACS records (donor
weight share 47%, source-household ESS 2.1 times the release's). With the
corrected measures that shift mostly disappears (finding 5). This pilot doesn't
separate which of the two corrections removed it.

| | Release | Run 1 | Run 2 | Run 3 | Run 4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Per-record weight cap (× release weight) | | 5 | 20 | 5 | 20 |
| Target loss cap | | 1 | 1 | 10 | 10 |
| **Trained, share of counties within 10%** | | | | | |
| Households (B11001) | 13.6% | 92.3% | 97.9% | 93.7% | 99.1% |
| Population (PEP V2024) | 13.5% | 91.0% | 97.6% | 92.0% | 98.6% |
| Aggregate earnings (B19061) | 9.2% | 88.4% | 95.1% | 91.7% | 98.6% |
| District households, held | 100% | 96.1% | 97.7% | 95.6% | 96.6% |
| **Achieved state totals vs release, median (max) drift** | | | | | |
| Households | | 0.2% (3.0%) | 0.1% (2.5%) | 0.3% (2.6%) | 0.1% (2.4%) |
| Population | | 0.1% (1.7%) | 0.1% (1.7%) | 0.1% (1.8%) | 0.1% (0.5%) |
| Earnings | | 1.3% (7.8%) | 1.4% (8.3%) | 0.1% (1.5%) | <0.1% (0.3%) |
| **Weights** | | | | | |
| Total Kish ESS, by source household | 10,995 | 13,253 | 12,582 | 14,558 | 13,289 |
| Total Kish ESS, by row | 13,631 | 15,937 | 15,083 | 17,480 | 15,941 |
| Donor (asec_puf) weight share | 68.0% | 59.8% | 61.1% | 57.3% | 59.7% |
| Median county ESS, among the 3,140 counties with records | 8.0 | 11.8 | 11.6 | 12.5 | 12.0 |
| Median ESS of the 292 counties of 250,000+ | 25.1 | 25.4 | 24.1 | 27.0 | 25.3 |
| Counties graded A / B / C / D | 2 / 41 / 456 / 2,645 | 1 / 28 / 495 / 2,620 | 0 / 24 / 472 / 2,648 | 1 / 31 / 530 / 2,582 | 0 / 27 / 496 / 2,621 |
| Population share in A–C counties | 48.3% | 46.3% | 43.2% | 50.2% | 46.2% |
| Counties of 250,000+ graded D (of 292) | 169 | 180 | 192 | 163 | 180 |

The largest state drift in runs 1 and 2 is Wyoming's earnings (+7.8% and
+8.3%). The looser loss cap all but removes earnings drift.

The median record's weight ratio (calibrated over release weight) is 0.74 in
run 1, 0.59 in run 2, 0.94 in run 3 and 0.67 in run 4. Weight moves from
donor to ACS records in total, but onto a minority of them: in run 1 the ACS
records' share of weight rises from 32.0% to 40.2% while 69% of them lose
weight. Runs 2 and 4 let a few records rise further: their 99.9th percentile
ratios are 18.3 and 19.3, against the cap of 5 in runs 1 and 3.

### Held-out families against a rescaling baseline

The three held-out families never enter the solve, but they describe the same
counties as the trained families and are closely related measures. To show
what reweighting adds, each is compared with a baseline that changes no
weights. The baseline multiplies each county's release value by its trained
driver's target-to-release ratio: households for households with earnings,
earnings for SOI wages and household income. Every column is scored on the
counties where the baseline is defined: 3,140 for households with earnings
and 3,138 for the other two (no records, or no earnings, leaves the ratio
undefined; SOI has no Kalawao row).

| Median county error | Release | Baseline | Run 1 | Run 2 | Run 3 | Run 4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Households with earnings (B19051) | 58.9% | 12.8% | 10.5% | 10.2% | 10.3% | 10.0% |
| Wages (SOI A00200) | 57.9% | 8.1% | 8.2% | 8.1% | 8.0% | 7.8% |
| Household income (B19025, approximate concept) | 58.0% | 17.5% | 13.9% | 13.1% | 13.0% | 12.6% |
| **Share of counties within 10%** | | | | | | |
| Households with earnings | 12.2% | 41.4% | 48.0% | 49.4% | 49.1% | 50.0% |
| Wages | 9.1% | 58.1% | 57.2% | 57.8% | 58.6% | 59.2% |
| Household income | 9.4% | 29.3% | 36.5% | 38.4% | 39.5% | 40.5% |

Most of the fall from about 58% comes from getting each county's size right,
which the baseline does too. Reweighting adds 2–3 points on households with
earnings, nothing beyond the baseline on SOI wages, and the most on household
income, where it narrows the median error by 4–5 points.

### Against published levels

Because the targets are shares of the release's own state totals, the tables
above measure distribution within states, not agreement with published county
levels. The release's state totals differ from the published state sums by a
median of 7.2% for households, 1.9% for population and 5.5% for earnings.
Nationally the release has 6.0% fewer households than ACS B11001 (121.5
million against 129.2 million), 2.6% less population than PEP and 1.9% less
earnings than B19061. Part of that may be vintage. The release's incomes are
2024, but it calibrates state and district population to 2020 Census
populations from its PUMA ladder (`us_puma_ladder_2020` in
`tools/build_us_acs_local_release.py`), and its 331.4 million people sit
below PEP's 340.1 million for July 2024. This pilot doesn't establish how much
of the household gap is vintage.
Measured against the raw published county levels instead:

| Share of counties within 10% of the published level | Release | Run 1 | Run 2 | Run 3 | Run 4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Households (B11001) | 12.6% | 70.5% | 76.5% | 70.5% | 76.6% |
| Population (PEP V2024) | 13.4% | 89.9% | 97.4% | 90.8% | 98.5% |
| Aggregate earnings (B19061) | 8.5% | 85.5% | 93.0% | 88.8% | 96.4% |
| Households with earnings (B19051), held out | 12.2% | 45.5% | 46.5% | 46.9% | 47.8% |
| Household income (B19025), held out | 10.2% | 44.7% | 46.9% | 46.8% | 48.3% |

SOI wages are TY2022 dollars against 2024 weights, so their level error mixes
in two years of wage growth: the release's national total is 10.6% above
SOI's. Median county error falls from 56% to 14.3–14.7%. The approximate
income concept runs 11.2% above ACS B19025 nationally; the median county's
approximate weighted median household income is 32% from ACS B19013 before
calibration and 22–23% after.

### Findings

1. **The release's counties are placeholders, not estimates.** Each household's
   county is a population-weighted draw inside its PUMA, and the weights were
   never calibrated below the district. The median county is 54–59% off on
   households, population and earnings. Calibration brings 88.4–99.1% of
   counties within 10% of their within-state share targets on every trained
   family. Against published levels it's 70.5–98.5%, lowest for households,
   where the release itself is 6.0% short nationally. The district rows are
   soft targets: they stay within about 0.2% at the median, but 10–19 of the
   436 district household totals move more than 10%. Achieved state totals
   stay within 0.1–1.4% at the median.
2. **Held-out families improve, mostly through county size.** Median county
   error on households with earnings, SOI wages and household income falls
   from 58–59% to 8–14%. A rescaling baseline that changes no weights gets to
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
4. **Calibrating to county totals doesn't consistently make large counties
   more reliable, and most stay graded D.** Total source-household ESS rises
   14–32% and the median county's ESS from 8.0 to 11.6–12.5. Among the 292
   counties of 250,000+, ESS rises in 90 to 177 of them, depending on the run,
   and their median moves from 25.1 to 24.1–27.0. Under the release's settings
   (run 1), 10 leave grade D and 21 enter it, so 180 are graded D against 169
   before, and the population share in A–C counties falls from 48.3% to
   46.3%. Run 3 does best: ESS rises in 177, 15 leave grade D and 9 enter it
   (163 graded D), and 50.2% of the population is in A–C counties. These are
   the provisional ESS grades.
5. **Rows past the loss cap get no gradient, and the release's cap leaves
   many there.** Under capped MAPE a row whose error passes the cap adds a
   constant to the loss (`microcosm.calibrate.diagnostics.past_cap_census`).
   At the start, 988 trained county rows are more than 100% off, past the
   release's cap of 1: 296 households, 299 population and 393 earnings rows,
   in 534 counties (17.0%). In run 1, 770 of them come back inside the cap
   through records they share with live rows. The other 218, in 149 counties,
   stay past it and 72 end worse than they started. Those counties hold 0.15%
   of the population on households and 1.8% on earnings. Under a cap of 10,
   31 rows in 24 counties start past it and 28 recover. The other three are
   Culberson TX's. One donor record holds 19,038 of the county's 19,117
   weighted households, against a target of 810. With no gradient from its
   county rows it ends at 43,223 in run 3, 53 times the target. No run pushes
   a row from inside the cap to past it. Unlike in the superseded runs, the
   looser cap barely changes weight concentration: donor weight share 57.3%
   against run 1's 59.8%, source-household ESS 14,558 against 13,253.
6. **The 5× per-record cap binds on fit, not on reliability.** Raising it to
   20× (run 2) lifts the trained families to 95–98% within 10%, but ESS
   falls (12,582 against run 1's 13,253) and 192 large counties are graded D.
   With the looser loss cap as well (run 4), 98.6–99.1% are within 10% at
   run 1's ESS.
7. **The release counts ACS self-employment twice.** ACS records carry a
   donor SSTB self-employment draw on top of their SEMP: 123,226 ACS persons,
   $202.6B under the release weights, more than the $158.5B ACS base leaf.
   PolicyEngine-US 2.2.1 adds both in `total_self_employment_income` and in
   gross income (microcosm#1380). This pilot counts SEMP only.
8. **Four counties have no records:** Arthur NE (31009), Kenedy TX (48261),
   King TX (48269) and Loving TX (48301), about 1,050 people. Two more have
   records but none with earnings: Kalawao HI (15005) and Borden TX (48033).
   SOI publishes no Kalawao row.

### Limits of this pilot

- Besides the county rows, the solve trains only district households and
  population, as soft targets at their release values. It omits the
  release's national, state and district SOI and administrative families, so
  the donor-to-ACS shift in these runs may cost fit on tax targets (top
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
  Donor earnings omit partnership self-employment ($53.0B), which ACS SEMP
  counts.
- The reliability grades are provisional ESS bands.
- County rows rescale published shares to the release's state totals. The fit
  tables measure within-state distribution; the "Against published levels"
  table measures what a user comparing to ACS would see.

### For the production rung

- Fix microcosm#1380 first, and gate it: on ACS persons, base plus SSTB
  self-employment equals `SEMP * ADJINC / 1e6`. County earnings rows need
  that.
- Keep ACS group quarters out of household-universe rows (households,
  earnings, income) and in population rows, as this pilot does.
- Train county rows with every release family. Report the donor weight share,
  the tax-target fit and achieved state totals beside county fit and county
  ESS, and set acceptance thresholds for each.
- Use a loss cap well above 1 for county rows, or warm-start them in stages.
  The release's cap leaves 988 trained county rows in 534 counties (17%)
  without gradient at the start. Report the past-cap census, and treat rows
  that end worse than they started, like Culberson's, as failures to fix
  rather than accept.
- Guarantee support for every county. The four empty counties need records,
  and a county whose weight sits on one donor record, like Culberson, needs
  more.
- Treat county reliability as a property of the release's weights, not of the
  county solve. Reweighting to county totals leaves most large counties below
  an ESS of 30, so the County Files' grades will mostly reflect how
  concentrated the release's own weights are.
