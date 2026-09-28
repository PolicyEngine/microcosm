# US support mix: CPS years vs ACS rows at a fixed row budget

Status: bake-off run 2026-09-28 on main `bda72cb02`, policyengine-us 2.2.1,
pinned Chronicle feed `b8543739` (32,842 targets). Tool:
[`tools/bakeoff_us_support_mix.py`](../tools/bakeoff_us_support_mix.py);
primitives and invariants:
[`us_runtime/support_mix.py`](../packages/microcosm-build/src/microcosm/build/us_runtime/support_mix.py).

## Question

Max, 2026-09-27: the ACS matters mainly as variation in the support, meaning
more distinct households for calibration to choose among. Pooling more CPS
years is the other route to support, at full CPS richness. Cloning CPS
households to new locations adds rows that differ only in geography. At a
household budget we can afford to build and serve, which mix gives the best
held-out accuracy, for the national compact file and for the CD and county
local file?

<!-- RESULTS -->

## Design

### Arms

A **source household** is one survey household record: one ASEC
household-year or one ACS household. Budgets count source households, as in
the design note, so the 2022–2024 ASEC pool is 168,852. A CPS source
household keeps its PUF tax-detail clone rows from the Route A pipeline, so
it spans two or three physical rows. The receipts report physical rows
separately, and physical rows drive engine and serving cost.

Each arm keeps every household of its CPS income years and fills the rest of
the budget with:

- **ACS** households (`acs100`): the first *n* in a salted hash order of
  `SERIALNO`, so a smaller arm's ACS households are always a subset of a
  larger arm's;
- **CPS location clones** (`acs000`): copies of the arm's own CPS households,
  each physical row drawing a new (district, county) within its state in
  proportion to ACS household weight (group-quarters rows excluded), the way
  Route A draws rows independently. A clone keeps its source row's engine
  values, so anything that varies by county within a state (ACA benchmark
  premiums by rating area) is carried from the origin county. Re-running the
  engine per clone would remove this approximation;
- a 50/50 mix (`acs050`, at 600k only).

| Budget | CPS years | Fills |
|---|---|---|
| natural size | 2024; 2023–24; 2022–24 | none |
| 300k, 600k, 1.2M | 2024; 2023–24; 2022–24 | ACS; clones |
| 300k, 600k, 1.2M | none | ACS only |
| 600k | 2024; 2022–24 | 50/50 |
| 300k | 2024; 2022–24 | ACS, replicates 1 and 2 |

That is 30 arms. Each is calibrated twice, once per product:

- **national compact**: the release's `national_state` surface (5,683
  usable national and state targets before the split), with loss weights
  computed on that surface as the release does;
- **local**: the full surface, which adds 27,148 targets the release
  classifies as district targets (24,340 district-level, plus 2,808 state
  and national totals from the SOI district file), plus district household
  population (ACS B25008). The ACS local release trains district population
  from its PUMA ladder instead; household population is used here so that
  group-quarters rows, which only the ACS has, neither count toward nor are
  pushed by the district total.

Replicate arms take the next disjoint blocks of the one fixed ACS order, so
they are independent draws; seed-0 arms are nested across budgets.
Group-quarters households (12% of ACS records) stay in the ACS side, as in
the ACS local release; the CPS has none.

**Five CPS years were not run.** Income years 2020 and 2021 pass the
response-quality check below, but main pins ASEC role and person sources
only for 2022–2024 and refuses other years. The pipeline stages that would
build them peak at 72–80 GB, above this task's 60 GB ceiling, while a Route A
release waits for memory on the same machine. The tool takes more years as
more rows, so a 5-year arm is a rerun once the pins and a base build land (see
Follow-ups).

### Superpool: one engine pass, many arms

Rerunning the pipeline per arm would cost hours and 70–94 GB each. Instead
the bake-off computes every household's target contributions once and
reweights subsets:

- **CPS rows**: the Route A 310842b release export (352,932 physical rows:
  income years 2022–2024, PUF clones, all enrichment stages), with design
  weights from its base. The run failed only its post-export smoke gate.
- **ACS rows**: the 2026-09-23 ACS local staging (1,531,614 ACS 2024 1-year
  households with CPS-only inputs transferred from the 57,240-household
  receipt-qualified donor spine). The staging's own 57,240 donor rows are not
  used.

Both carry the same PolicyEngine input schema (141 and 140 person inputs).
Imputations are fixed across arms: an ACS row's CPS-only variables come from
the same donor spine whatever CPS years the arm pools. The bake-off
therefore measures the support effect with donor quality held constant; in a
real rebuild, a 1-year arm would also have a thinner donor pool.

**Concepts.** The release materializer builds each state or district
target's household column as a geography-free value times an indicator of
the household's own state or district
(`_base_simulation_household_columns`). `compile` groups the 32,842 targets
into 419 such concepts, and `materialize` runs the release tool's own
`_materialize_target_frame` on them in 50,000-household shards (peak about
5 GB). Every arm's target matrix is then concept value × geography mask,
assembled sparse. `diffcheck` verifies the identity on real rows: production
per-target columns and concept × mask agree exactly on sampled state and
district targets of every family (see Evidence). The 11 JCT tax-expenditure
targets need reform simulations and are excluded from every arm.

### Holdout

The US holdout ("Port the UK holdout to US release targets") has not merged.
The bake-off mirrors its pending `target_split` spec byte for byte: SHA-256
of `salt + "\x1f" + key`, sealed if below 0.05 on the sealed salt, otherwise
held out if below 0.10 on the holdout salt. District children take their
state parent's key, so a held-out state total is held out with all its
districts; national and state targets are drawn independently, so a
held-out national total can still be pinned by trained state targets.
Vintage tokens are dropped from keys. The pending spec's pins, which force
some groups to train, were not yet written and are not mirrored, so the
bake-off split differs from the eventual release split for pinned groups.
Sealed targets are neither trained nor scored.

On the pinned feed: 27,711 train, 3,470 holdout (36 national, 787 state,
2,647 district), 1,661 sealed.

**ACS-native truth.** The registry has no housing targets and no county
targets. ACS-native dimensions are scored against the ACS 2024 1-year
table-based summary file (B25003 tenure, B25060 and B25065 aggregate rent,
B25090 aggregate real-estate taxes, B25008 and B01003 population, B01001 age
and sex) at national, state, district (119th Congress) and county (the 850
counties the 1-year file covers). They are never calibrated, except
district total population in the local product. These tables come from the
same ACS year as the ACS rows; the PUMS is a subsample of the published
sample and district and county are drawn from PUMA, so ACS rows do not
reproduce them by construction, but they share its sampling error. Treat
ACS-native scores as consistency with the published ACS, not as truth.

### Calibration

Every arm uses the production optimizer, `microcosm.calibrate.solve._optimize`,
with the fiscal release's optimizer settings: Adam on log-weights, learning rate 0.02,
capped relative error (cap 1.0), mass conserved, max weight ratio 5, the
release's concept-budget loss weights computed once per surface (the
national_state surface for the national product, the full registry for the
local one) before the split, so train and holdout keep fixed weights; district
household-population rows weigh 1 each; no L0 pruning (so the row budget
stays fixed). A sparse operator with a precomputed transpose replaces the
torch sparse-CSR backward pass, which was the bottleneck; the objective and
projections are unchanged.

Starting weights: each CPS income year gets an equal share of the CPS mass;
a household with *k* location copies splits its weight over 1 + *k* rows;
CPS and ACS sides share mass in proportion to source households; then the
release's population mass repair rescales to its 334.2M-person benchmark.

**Invariants** (property-tested in `test_us_support_mix.py`): an arm's CPS,
ACS and clone households sum to its budget; ACS selections are nested and
independent of input order; clone copies differ by at most one; starting
weights are positive, sum to the requested mass, give each CPS year an equal
share and split CPS and ACS in proportion to households; distinct-household
ESS never exceeds row ESS or the distinct-unit count; the holdout role is a
pure function of the group key and never moves with a vintage token.
**Differential**: `diffcheck` compares production per-target materialization
with concept × mask.

### Scoring

- **Registry holdout**: capped relative error `min(|est − target| /
  max(|target|, 1), 1)`, mean and loss-weighted mean, by dimension (CPS-native:
  income components, tax items, program receipt; ACS-native: PEP
  demographics) and level. A target no row in the arm can move scores as a
  miss (error 1), so every arm is averaged over the same held-out set; a
  common-support table is reported too.
- **ACS-native**: households, tenure, aggregate rent (the model's
  `pre_subsidy_rent`, scored against both contract rent B25060 and gross rent
  B25065) and real-estate taxes, household population (B25008) and total
  population, age and sex (B01003, B01001; these include group quarters), by
  level including county. Household measures exclude group-quarters rows.
- **ESS** over distinct households: weights summed within each household
  unit (an ASEC housing unit, `H_IDNUM`, counted once across rotation years,
  PUF clone rows and location clones; an ACS `SERIALNO`), then Kish; also the
  median across districts.
- **Cost**: calibration wall time and peak RSS per arm; engine seconds per
  household from the materialization receipts; physical rows as the serving
  cost driver.
- **Report only**: SPM poverty rate, all persons and children. Never
  calibrated or gated.

## CPS year response quality

`census_cps_YYYY.h5` is keyed by income year (ASEC collected in YYYY+1).
Checked from the Census microdata and technical documentation:

| File (income year) | ASEC | Households | Combined response rate | Whole-supplement imputed | Earnings allocated |
|---|---|---|---|---|---|
| 2019 | 2020 | 60,460 | 61.1% | 21.6% | 29.4% |
| 2020 | 2021 | 62,850 | 65.0% | 20.3% | 30.1% |
| 2021 | 2022 | 59,148 | 61.4% | 20.5% | 29.5% |
| 2022 | 2023 | 56,839 | 59.6% | 20.0% | 28.8% |
| 2023 | 2024 | 56,251 | 59.3% | 19.2% | 27.8% |
| 2024 | 2025 | 55,762 | 60.1% | 17.8% | 28.1% |

Combined response is the product of basic-CPS and ASEC household response,
from each year's Source and Accuracy statement; imputation and allocation
are weighted shares computed from the Census microdata (`FL_665 != 1`;
earnings allocated among respondent earners).

Income years 2020 and 2021 **pass** on response and processing quality:
their combined response rates are above the pinned years', their imputation,
allocation and weight-dispersion metrics sit within about 1.5 points of the
pinned years', and they show no composition break. Three caveats:

- Census asks users to take care comparing data years 2019, 2020 and 2021
  with other years, and documents that nonrespondents in the 2020–2026
  surveys differ more from respondents than before (2026 CPS ASEC technical
  documentation). That caution covers the pinned 2022–2024 years' collections
  too.
- Their income years carry pandemic program receipt: 10,915 and 4,065
  unemployment-compensation recipients, against about 1,200–1,700 in other
  years. That matters for program-receipt targets.
- About 35% of each file's households repeat in the neighbouring year.

Income year 2019 (ASEC 2020, collected as COVID hit) is the file that fails:
it shows composition jumps in characteristics that are not weighting
controls (weighted BA+ share +1.5 points, homeownership +1.9 points, a 11.5%
smaller household sample).

Rotation overlap: a housing unit appears in at most two adjacent files, so
2022–2024 has 130,294 distinct units in 168,852 households (77.2%), and
2020–2024 would have 211,834 in 290,850 (72.8%).
