# US support mix: CPS years vs ACS rows at a fixed row budget

Status: bake-off run 2026-09-28 to 2026-09-30 on main `bda72cb02`,
policyengine-us 2.2.1, pinned Chronicle feed `b8543739` (32,842 targets); 30
arms × 2 products, all scored. Receipts:
[`experiments/us-support-mix-bakeoff-20260930/`](../experiments/us-support-mix-bakeoff-20260930/). Tool:
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

## Verdict

- **National compact file: pool the newest three CPS years and fill to
  300,000 source households with ACS.** With 2022–2024 that is 168,852 CPS
  households plus 131,148 ACS households, 484,080 physical rows. Against the
  three-year CPS file alone (352,932 rows), held-out state income error falls
  from 0.250 to 0.218 and state program-receipt error from 0.208 to 0.184;
  district and county measures improve throughout; state tax items cost 0.008
  (0.103 to 0.111). Larger ACS fills keep improving local measures but give
  back national program receipt (0.184 at 300k, 0.225 at 600k, 0.239 at
  1.2M), so the compact file stops at 300k.
- **CD and county local file: pool the newest three CPS years and fill to
  1.2M source households with ACS households, not CPS location clones.** That
  is 1,384,080 physical rows, 13% fewer than today's 1,588,854-row ACS local
  file. At 1.2M, the ACS fill matches clones on held-out district income and
  tax items (0.248 and 0.189 against 0.248 and 0.185) with 55% of the clones'
  rows, and cuts ACS-native error by 40% in districts (0.123 against 0.206)
  and 28% in counties (0.166 against 0.229). Against an ACS-only file of the
  same size (today's local file is 57,240 CPS rows plus every ACS household),
  the three CPS years cut state program-receipt error from 0.318 to 0.256 and
  state tax items from 0.127 to 0.117, and ACS-native error is no worse
  (0.069 / 0.123 / 0.166 against 0.074 / 0.128 / 0.172).
- **More CPS years help whenever the fill is ACS.** At 300k in the national
  product, one, two and three years give state income error 0.249, 0.236 and
  0.218, program receipt 0.260, 0.217 and 0.184, and tax items 0.133, 0.124
  and 0.111. Income years 2020 and 2021 pass the response-quality check, so a
  five-year pool is the next arm to run (see Follow-ups); it was not run here.
- **Clones add geography and nothing else.** In the national product they
  leave every state measure where the CPS-only file had it. In both products
  they improve district and county measures slowly (national-product district
  income 0.378, 0.362, 0.337 and 0.324 from no clones to 1.2M), at about
  twice the physical rows of an ACS fill of the same budget.

The pattern matches the design note's reasoning. ACS rows carry real
variation in what the ACS observes, so income components (the ACS records
wages, self-employment, Social Security, retirement and investment income),
housing and local demographics improve as ACS rows enter. Program receipt on
ACS rows is imputed from CPS donors, so it degrades as the ACS share rises:
state program-receipt error is 0.208 with no ACS, 0.184 with 131k ACS
households beside three CPS years, and 0.303 with ACS alone at 300k.

## Results

All errors are mean capped relative error on held-out targets, `min(|est −
target| / max(|target|, 1), 1)`; lower is better. "ACS-native" averages
households, tenure, contract rent, real-estate taxes, household population and
two age bands against the ACS 2024 1-year tables. Rows are physical household
rows (a CPS household carries its PUF tax-detail clone rows); ESS is Kish over
distinct households (an ASEC housing unit counted once across rotation years,
PUF rows and clones). Seed 0 throughout.

### National product (trained on the 5,683 national and state targets)

| Arm | Rows | ESS | Income, state | Tax items, state | Program receipt, state | Income, CD | Tax items, CD | ACS-native state / CD / county |
|---|---|---|---|---|---|---|---|---|
| CPS 2022–24 only | 352,932 | 44,943 | 0.250 | 0.103 | 0.208 | 0.378 | 0.420 | 0.112 / 0.219 / 0.298 |
| CPS 2022–24 + clones to 300k | 627,089 | 44,984 | 0.250 | 0.103 | 0.208 | 0.362 | 0.393 | 0.112 / 0.208 / 0.272 |
| CPS 2024 + ACS to 300k | 361,228 | 63,297 | 0.249 | 0.133 | 0.260 | 0.353 | 0.367 | 0.075 / 0.160 / 0.253 |
| CPS 2023–24 + ACS to 300k | 422,543 | 65,908 | 0.236 | 0.124 | 0.217 | 0.341 | 0.361 | 0.084 / 0.165 / 0.250 |
| **CPS 2022–24 + ACS to 300k** | **484,080** | **69,065** | **0.218** | **0.111** | **0.184** | **0.343** | **0.356** | **0.094 / 0.174 / 0.253** |
| ACS only, 300k | 300,000 | 60,492 | 0.261 | 0.151 | 0.303 | 0.351 | 0.358 | 0.083 / 0.166 / 0.273 |
| CPS 2022–24 + 50/50 to 600k | 1,019,084 | 77,830 | 0.213 | 0.112 | 0.187 | 0.319 | 0.332 | 0.096 / 0.165 / 0.223 |
| CPS 2022–24 + ACS to 600k | 784,080 | 113,041 | 0.221 | 0.116 | 0.225 | 0.326 | 0.326 | 0.084 / 0.151 / 0.221 |
| CPS 2022–24 + ACS to 1.2M | 1,384,080 | 222,920 | 0.217 | 0.118 | 0.239 | 0.300 | 0.301 | 0.072 / 0.124 / 0.182 |
| Spread across 3 ACS draws (SD) | | | 0.007 | 0.002 | 0.014 | 0.007 | 0.005 | 0.000 / 0.001 / 0.000 |

### Local product (adds the 27,148 district-classified targets and district household population)

| Arm | Rows | ESS | Income, state | Tax items, state | Program receipt, state | Income, CD | Tax items, CD | ACS-native state / CD / county |
|---|---|---|---|---|---|---|---|---|
| CPS 2022–24 only | 352,932 | 52,074 | 0.217 | 0.101 | 0.220 | 0.292 | 0.240 | 0.111 / 0.219 / 0.284 |
| CPS 2022–24 + clones to 600k | 1,254,039 | 54,305 | 0.211 | 0.097 | 0.221 | 0.257 | 0.198 | 0.111 / 0.209 / 0.241 |
| CPS 2022–24 + clones to 1.2M | 2,508,197 | 54,292 | 0.214 | 0.096 | 0.218 | 0.248 | 0.185 | 0.112 / 0.206 / 0.229 |
| CPS 2022–24 + 50/50 to 600k | 1,019,084 | 91,100 | 0.226 | 0.096 | 0.243 | 0.261 | 0.195 | 0.095 / 0.173 / 0.216 |
| CPS 2022–24 + ACS to 600k | 784,080 | 150,595 | 0.228 | 0.109 | 0.267 | 0.262 | 0.199 | 0.079 / 0.142 / 0.198 |
| **CPS 2022–24 + ACS to 1.2M** | **1,384,080** | **276,474** | **0.206** | **0.117** | **0.256** | **0.248** | **0.189** | **0.069 / 0.123 / 0.166** |
| CPS 2024 + ACS to 1.2M | 1,261,228 | 281,840 | 0.204 | 0.123 | 0.278 | 0.245 | 0.191 | 0.065 / 0.121 / 0.167 |
| ACS only, 1.2M | 1,200,000 | 272,607 | 0.206 | 0.127 | 0.318 | 0.244 | 0.193 | 0.074 / 0.128 / 0.172 |
| Spread across 3 ACS draws (SD) | | | 0.007 | 0.002 | 0.006 | 0.002 | 0.002 | 0.000 / 0.001 / 0.001 |

The spread row comes from three disjoint ACS draws of the 300k arms with
three CPS years. Differences under about 0.01 (0.03 for program receipt,
which has 53 held-out targets) are within draw-to-draw noise. National-level
held-out targets (36) are noisier still and are left out of the tables; they
are in the receipts.

**Where the local file gives ground.** Program receipt and state tax items
are the two dimensions an ACS fill worsens. If program-receipt accuracy
matters more than local housing and demographics, the 50/50 fill at 600k is
the middle choice: program receipt 0.243 and tax items 0.096, at district
ACS-native 0.173.

**Report only: SPM poverty** (never calibrated or gated). In the national
product the SPM rate is 12.8% for the three-year CPS file, 12.2% for the
recommended 300k mix and 12.3% for ACS only; child SPM poverty is 14.7%, 13.6%
and 12.3%. ACS rows' SPM resources rest on imputed CPS-only inputs, so these
differences are a property of the imputation as much as of the support.

## Cost

| | Measured |
|---|---|
| Engine pass (all 419 concepts), wall time per household on the shared host | CPS rows about 25 ms, ACS rows about 50 ms; 5,000-household engine batches, peak 16–23 GB per 50,000-household shard |
| Calibration, national recommendation (484k rows, 1,500 epochs) | 22 min, 6.2 GB peak |
| Calibration, local recommendation (1.38M rows, 1,500 epochs) | 36 min, 15.9 GB peak |
| Calibration, local clones to 1.2M (2.51M rows) | 55–210 min, 18–28 GB peak |
| Serving | proportional to physical rows: 484k national (+37% over CPS only), 1.38M local (−13% against today's local file) |

Wall times were measured with the host at load 10 to 400 and are comparable
only within this run. Training loss still fell a median 5.5% between epochs
1,000 and 1,500; every arm had the same budget.


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
the ACS local release; the CPS has none. The 5,246 group-quarters records
(0.34% of ACS records, all group quarters) whose SPM unit has no member the
pinned engine classifies as an adult are excluded: policyengine-us 2.2.1
refuses them, and Census puts ACS group quarters outside the SPM universe
(`us_runtime/spm_universe_source.py`).

**Five CPS years were not run.** Income years 2020 and 2021 pass the
response-quality check below, but main pinned ASEC role and person sources
only for 2022–2024 when this ran, and the base and release stages that would
enrich two more years peaked at 72–80 GB in recorded runs, above this task's
60 GB ceiling, while a Route A release held the machine. The tool takes more
years as more rows, so a five-year arm is a rerun once a Route A export with
those years exists (see Follow-ups). The same holds for the newest three
years (2023–2025): the 2025 ASEC was pinned on 2026-09-29
(PolicyEngine/microcosm#1044), after these CPS rows were built.

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
  used. The staging carries `is_spm_independent_minor_role` as all-null on
  ACS rows; the column is dropped so the engine derives the role from
  household structure (its formula), as the production ACS local build did.

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
`_materialize_target_frame` on them in 50,000-household shards (peak 16–23
GB). Every arm's target matrix is then concept value × geography mask,
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

## Caveats

- **CPS rows are income years 2022–2024**, the Route A export available when
  this ran. The recommendation is stated for "the newest three years"; the
  2023–2025 export is the rerun.
- **Imputations are held fixed.** ACS rows' CPS-only inputs come from one
  57,240-household donor spine in every arm. A production one-year build
  would also impute from a thinner donor pool, which would widen the
  one-year arms' program-receipt gap.
- **Clones keep their origin row's engine values**, including anything that
  varies by county within a state.
- **The holdout mirrors the unmerged US split without its pins**, and holds out
  district children with their state parent but national and state targets
  independently. Scores are comparable across arms, not with the eventual
  release split.
- **ACS-native truth is the published ACS 2024 1-year**, the same survey year
  as the ACS rows. Read it as consistency with the published ACS.
- **Older CPS years are not uprated**, as in Route A: calibration absorbs the
  2022–2024 nominal income gap.
- **Group-quarters minors with no SPM adult are out of the ACS support** (5,246
  records), pending the engine's SPM-universe input (policyengine-us#9462).
- **JCT tax-expenditure targets are excluded** (they need reform simulations).

## Follow-ups

- Rerun the CPS-involving arms on the newest three years (2023–2025) once a
  Route A export on main at or after `5187fce25` exists: shard it, run
  `materialize --source cps`, delete the CPS receipts and rerun the grid. The
  ACS concepts are reused.
- Add five-year arms (2020–2024 or 2021–2025) once those years have Route A
  enrichment.
- Swap the split mirror for `microcosm.build.us_runtime.target_split` when it
  merges, and rescore from the saved weights (`arm` re-scores a receipt whose
  weights exist without re-solving).

## Evidence

- Differential: production per-target materialization against concept ×
  mask, 4,000 households from each source, 349 state and district targets
  across nine family-level groups: maximum scaled difference 0.0 for both CPS
  and ACS rows (`diffcheck.json`).
- Property and differential tests: `test_us_support_mix.py` (14 tests).
- Independent review (Opus 5.5, Subfleet): three high-severity and six
  medium-severity findings, all fixed before the grid ran.
- Receipts (in `experiments/us-support-mix-bakeoff-20260930/`):
  `dimensions_by_arm.csv`, `headline.md`, `arms.csv`,
  `holdout_by_dimension_level.csv`, `holdout_by_dimension_level_common_support.csv`,
  `replicate_spread.csv`, `compile.json`, `diffcheck.json`, `acs_truth.json`.

Reproduce: `compile`, `shard` (both sources), `truth`, `materialize` (both
sources; ACS by rank range), `diffcheck`, `run-grid --products national` and
`--products local`, `report`. Each step writes a receipt with wall time and
peak RSS.
