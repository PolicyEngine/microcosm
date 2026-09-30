# US support mix: CPS years vs ACS rows at a fixed household budget

Status: bake-off run 2026-09-28 to 2026-09-30 on main `bda72cb02`,
policyengine-us 2.2.1, pinned Chronicle feed `b8543739` (32,842 targets); 30
arms × 2 products, all scored. Receipts:
[`experiments/us-support-mix-bakeoff-20260930/`](../experiments/us-support-mix-bakeoff-20260930/).
Tool: [`tools/bakeoff_us_support_mix.py`](../tools/bakeoff_us_support_mix.py);
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

The headline metric is the release's own objective on held-out targets: the
loss-weighted capped relative error over every held-out CPS-native target
(income components, tax items, program receipt) at a level, with the
release's fixed concept-budget weights. Per-dimension columns show the
trade-offs.

- **National compact file: the newest three CPS years plus ACS to 300,000
  source households.** With 2022–2024 that is 168,852 CPS households plus
  131,148 ACS households, 484,080 physical rows (+37% over the three-year CPS
  file's 352,932). Against the CPS-only file, averaged over three ACS draws:
  - held-out state CPS-native error falls from 0.119 to 0.113 (draw SD
    0.0003) and district error from 0.353 to 0.291;
  - state income components improve (0.250 to 0.224), state tax items get
    worse (0.103 to 0.113), and state program receipt does not measurably
    change (0.208 against 0.198, draw SD 0.014);
  - every ACS-native level improves (district 0.236 to 0.188, county 0.307
    to 0.259).

  Stopping at 300k is a cost call. Larger ACS fills keep lowering district
  error (0.291, 0.266, 0.244 at 300k, 600k, 1.2M) but leave state error flat
  (0.113, 0.115, 0.113), keep raising state tax items (0.113, 0.116, 0.118)
  and worsen state program receipt (0.198, 0.225, 0.239). The best state-level
  arm is the 600k 50/50 fill (0.111; income 0.213, tax items 0.112, program
  receipt 0.187) at 1,019,084 rows, 2.1 times the recommendation. If state tax
  items and serving cost outweigh everything else, the three-year CPS-only
  file remains the choice.

- **CD and county local file: the newest three CPS years plus ACS households,
  not CPS location clones, to 1.2M source households.** That is 1,384,080
  physical rows, 13% fewer than today's 1,588,854-row ACS local file. At
  1.2M:
  - held-out district CPS-native error ties the clones (0.169 against 0.170)
    at 55% of their rows, and state error is lower (0.112 against 0.114);
  - ACS-native error is 44% lower in districts (0.141 against 0.253) and 31%
    lower in counties (0.174 against 0.253);
  - the clones, and the CPS-only file, do better on state program receipt
    (0.218 and 0.220 against 0.256) and state tax items (0.096 and 0.101
    against 0.117). The 50/50 fill at 600k splits the difference (program
    receipt 0.243, tax items 0.096, district ACS-native 0.209).

  Against an ACS-only file of the same size (today's local file is 57,240 CPS
  rows plus every ACS household), the three CPS years lower state CPS-native
  error from 0.120 to 0.112, state program receipt from 0.318 to 0.256 and
  state tax items from 0.127 to 0.117, and ACS-native error is no worse.
  One CPS year instead of three ties on districts and ACS-native measures at
  1.2M and saves 123k rows (9%), but loses on state error (0.116 against
  0.112) and, at 300k where there are three draws of each, clearly so (0.119
  against 0.112). Three years also let the local file share the national
  file's CPS support.

- **More CPS years help the national file.** At 300k, one year against three
  (three draws each): state income 0.249 against 0.224, tax items 0.130
  against 0.113, program receipt 0.241 against 0.198. In the local file the
  effect is mixed: each CPS year displaces ACS households within the budget,
  so tax items and program receipt improve while state income and ACS-native
  measures get slightly worse. Income years 2020 and 2021 pass the
  response-quality check, so a five-year pool is the next arm to run; it was
  not run here (see Follow-ups).

- **Clones add geography and little else.** In the national file they leave
  every state measure where the CPS-only file had it. In both files they
  improve district measures slowly (national-file district CPS-native error
  0.353, then 0.332 at 300k) and household ACS-native measures barely
  (local-file county 0.299, then 0.263 and 0.253 at 600k and 1.2M), at 1.3 to
  1.8 times the physical rows of an ACS fill of the same budget.

This matches the design note's reasoning. ACS rows carry real variation in
what the ACS observes, so income components (the ACS records wages,
self-employment, Social Security, retirement and interest, dividend and rental
income), housing and local geography improve as ACS rows enter. Program
receipt on ACS rows is imputed from CPS donors: beside three CPS years it
holds level at 131k ACS households, then worsens as the fill grows, and ACS
alone at 300k scores 0.303.

## Results

All errors are capped relative error on held-out targets, `min(|est − target|
/ max(|target|, 1), 1)`; lower is better. "CPS-native" columns are
loss-weighted over all held-out CPS-native targets at the level; the
per-dimension columns are plain means. "ACS-native" averages households,
owner and renter households, contract rent, real-estate taxes and household
population against the ACS 2024 1-year tables (age bands, which count
group-quarters people only the ACS rows have, are in
`acs_native_by_level_measure.csv`). Rows are physical household rows (a CPS
household carries its PUF tax-detail clone rows); ESS is Kish over distinct
households (an ASEC housing unit counted once across rotation years, PUF rows
and clones). Rows marked "3 draws" average three disjoint ACS draws; the rest
are one draw. Tables are generated from the receipts by `memo_tables.py`.

### National product (4,828 train-role national and state targets)

| Arm | Rows | ESS | CPS-native, state | CPS-native, CD | Income, state | Tax items, state | Program receipt, state | Age (PEP), state | ACS-native state / CD / county |
|---|---|---|---|---|---|---|---|---|---|
| CPS 2022–24 only | 352,932 | 44,943 | 0.119 | 0.353 | 0.250 | 0.103 | 0.208 | 0.169 | 0.133 / 0.236 / 0.307 |
| CPS 2022–24 + clones to 300k | 627,089 | 44,984 | 0.119 | 0.332 | 0.250 | 0.103 | 0.208 | 0.169 | 0.133 / 0.226 / 0.283 |
| CPS 2024 + ACS to 300k (3 draws) | 361,228 | 64,090 | 0.117 | 0.300 | 0.249 | 0.130 | 0.241 | 0.159 | 0.086 / 0.166 / 0.255 |
| CPS 2023–24 + ACS to 300k | 422,543 | 65,908 | 0.119 | 0.288 | 0.236 | 0.124 | 0.217 | 0.171 | 0.097 / 0.175 / 0.251 |
| **CPS 2022–24 + ACS to 300k (3 draws)** | 484,080 | 69,353 | 0.113 | 0.291 | 0.224 | 0.113 | 0.198 | 0.163 | 0.110 / 0.188 / 0.259 |
| ACS only, 300k | 300,000 | 60,492 | 0.132 | 0.289 | 0.261 | 0.151 | 0.303 | 0.176 | 0.096 / 0.172 / 0.269 |
| CPS 2022–24 + 50/50 to 600k | 1,019,084 | 77,830 | 0.111 | 0.272 | 0.213 | 0.112 | 0.187 | 0.163 | 0.113 / 0.182 / 0.232 |
| CPS 2022–24 + ACS to 600k | 784,080 | 113,041 | 0.115 | 0.266 | 0.221 | 0.116 | 0.225 | 0.159 | 0.097 / 0.163 / 0.227 |
| CPS 2022–24 + ACS to 1.2M | 1,384,080 | 222,920 | 0.113 | 0.244 | 0.217 | 0.118 | 0.239 | 0.166 | 0.081 / 0.135 / 0.189 |
| SD across 3 draws (300k, 2022–24) | | | 0.000 | 0.002 | 0.007 | 0.002 | 0.014 | 0.005 | 0.001 / 0.001 / 0.002 |

### Local product (adds the 27,148 district-classified targets and district household population)

| Arm | Rows | ESS | CPS-native, state | CPS-native, CD | Income, state | Tax items, state | Program receipt, state | Age (PEP), state | ACS-native state / CD / county |
|---|---|---|---|---|---|---|---|---|---|
| CPS 2022–24 only | 352,932 | 52,074 | 0.116 | 0.209 | 0.217 | 0.101 | 0.220 | 0.154 | 0.134 / 0.263 / 0.299 |
| CPS 2022–24 + clones to 600k | 1,254,039 | 54,305 | 0.114 | 0.179 | 0.211 | 0.097 | 0.221 | 0.175 | 0.134 / 0.256 / 0.263 |
| CPS 2022–24 + clones to 1.2M | 2,508,197 | 54,292 | 0.114 | 0.170 | 0.214 | 0.096 | 0.218 | 0.183 | 0.135 / 0.253 / 0.253 |
| CPS 2022–24 + 50/50 to 600k | 1,019,084 | 91,100 | 0.111 | 0.173 | 0.226 | 0.096 | 0.243 | 0.176 | 0.113 / 0.209 / 0.232 |
| CPS 2022–24 + ACS to 600k | 784,080 | 150,595 | 0.115 | 0.179 | 0.228 | 0.109 | 0.267 | 0.168 | 0.092 / 0.166 / 0.207 |
| **CPS 2022–24 + ACS to 1.2M** | 1,384,080 | 276,474 | 0.112 | 0.169 | 0.206 | 0.117 | 0.256 | 0.184 | 0.078 / 0.141 / 0.174 |
| CPS 2024 + ACS to 1.2M | 1,261,228 | 281,840 | 0.116 | 0.169 | 0.204 | 0.123 | 0.278 | 0.188 | 0.072 / 0.136 / 0.174 |
| ACS only, 1.2M | 1,200,000 | 272,607 | 0.120 | 0.170 | 0.206 | 0.127 | 0.318 | 0.193 | 0.084 / 0.146 / 0.180 |
| SD across 3 draws (300k, 2022–24) | | | 0.001 | 0.002 | 0.007 | 0.002 | 0.006 | 0.002 | 0.000 / 0.002 / 0.000 |

Differences under about twice the draw SD are noise. The one-year arms are
noisier than the three-year arms (program receipt SD 0.041 national and
0.024 local at 300k; `replicate_spread.csv`). National-level held-out
targets (36) are too few to rank arms and are left out of the tables; they
are in the receipts.

**Calibration worsens held-out state age cells, in every arm.** The mean
capped error on held-out Census PEP state age bands is 0.066 at the starting
weights and 0.168 after calibration, across all 60 calibrations (0.131 to
0.193). The design weights already carry Census age controls, and the solve
trades them away for its training targets. Arms differ by less than the trade
(national 0.159 to 0.176 in the table), so the ranking stands; the size of the
trade is a finding about the calibration objective, filed as a follow-up.

**Report only: SPM poverty** (never calibrated or gated). In the national
product the SPM rate is 12.8% for the three-year CPS file, 12.2% for the
recommended 300k mix (seed 0) and 12.3% for ACS only; child SPM poverty is
14.7%, 13.6% and 12.3%. These rates include the ACS group-quarters persons
still in the support, whom Census places outside the SPM universe; the tool
now excludes them, and these receipts predate that fix. ACS rows' SPM
resources rest on imputed CPS-only inputs, so the differences reflect the
imputation as much as the support.

## Cost

| | Measured |
|---|---|
| Engine pass (all 419 concepts), wall time per household on the shared host | CPS parts 7.7 to 57 ms (median 26); ACS parts 7.5 to 226 ms (median 39); 5,000-household engine batches |
| Engine pass peak RSS | 50,000-household CPS shards 5.8 to 15.6 GB; up to 22.7 GB per ACS process (a running maximum over its parts) |
| Calibration, national recommendation (484k rows, 1,500 epochs) | 22 min, 6.2 GB peak |
| Calibration, local recommendation (1.38M rows, 1,500 epochs) | 36 min, 15.9 GB peak |
| Calibration, local clones to 1.2M (2.51M rows) | 55 to 210 min, 18 to 28 GB peak |
| Serving | proportional to physical rows: 484k national (+37% over CPS only), 1.38M local (−13% against today's local file) |

Wall times were measured with the host at load 10 to 400 and compare only
within this run. Training loss still fell a median 5.5% between epochs 1,000
and 1,500 (`arms.csv` has `loss_1000` and `loss_final`); every arm had the
same 1,500-epoch budget. For 22 arms first scored before scoring covered
every level, the receipts carry the solve peak restored from the grid log and
no solve wall time.

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
  premiums by rating area) is carried from the origin county;
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
  usable national and state targets, 4,828 of them train-role after the
  split), with loss weights computed on that surface as the release does;
- **local**: the full surface, which adds 27,148 targets the release
  classifies as district targets (24,340 district-level, plus 2,808 state
  and national totals from the SOI district file), plus district household
  population (ACS B25008). The ACS local release trains district population
  from its PUMA ladder instead; household population is used here so that
  group-quarters rows, which only the ACS has, neither count toward nor are
  pushed by the district total.

Replicate arms take the next disjoint blocks of the one fixed ACS order, so
they are independent draws; seed-0 arms are nested across budgets.
Group-quarters households (11.96% of ACS records) stay in the ACS side, as in
the ACS local release; the CPS has none. The 5,246 ACS records (0.34%, all
group quarters) whose SPM unit has no member the pinned engine counts as an
adult are excluded: policyengine-us 2.2.1 refuses them, and Census puts ACS
group quarters outside the SPM universe (`us_runtime/spm_universe_source.py`).
`shard` and `spm-flags` compute the flag with the engine's rule
(`spm_flags.json`).

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
  ACS rows; the column is dropped so the engine derives the role from its
  formula (`is_household_head | is_household_spouse`), as the production ACS
  local build did.

Both carry the same PolicyEngine input schema (141 and 140 person inputs).
Imputations are fixed across arms: an ACS row's CPS-only variables come from
the same donor spine whatever CPS years the arm pools. The bake-off
therefore measures the support effect with donor quality held constant; in a
real rebuild, a one-year arm would also have a thinner donor pool.

**Concepts.** The release materializer builds each state or district
target's household column as a geography-free value times an indicator of
the household's own state or district
(`_base_simulation_household_columns`). `compile` groups the 32,842 targets
into 419 such concepts, and `materialize` runs the release tool's own
`_materialize_target_frame` on them in household shards. Every arm's target
matrix is then concept value × geography mask, assembled sparse. `diffcheck`
verifies the identity on real rows (see Evidence). The 11 JCT
tax-expenditure targets need reform simulations and are excluded from every
arm.

### Holdout

The US holdout ("Port the UK holdout to US release targets") has not merged.
The bake-off mirrors its pending spec: SHA-256 of `salt + "\x1f" + key`,
sealed if below 0.05 on the sealed salt, otherwise held out if below 0.10 on
the holdout salt. The hash is tested against draws from the port branch's own
implementation (`PORT_BRANCH_FIXTURES`); the group-key and role rules follow
the port session's written spec, since its module had not been written.
District children take their state parent's key, so a held-out state total is
held out with all its districts; national and state targets are drawn
independently, so a held-out national total can still be pinned by trained
state targets. Vintage tokens are dropped from keys. The spec's pins, which
force some groups to train, are not mirrored, so this split differs from the
eventual release split for pinned groups. Sealed targets are neither trained
nor scored.

On the pinned feed: 27,711 train, 3,470 holdout (36 national, 787 state,
2,647 district), 1,661 sealed.

**ACS-native truth.** The registry has no housing targets and no county
targets. ACS-native dimensions are scored against the ACS 2024 1-year
table-based summary file (B25003 tenure, B25060 and B25065 aggregate rent,
B25090 aggregate real-estate taxes, B25008 and B01003 population, B01001 age
and sex) at national, state, district (119th Congress) and county (the 850
counties the 1-year file covers). None are calibrated, except district
household population (B25008) in the local product. These tables come from
the same ACS year as the ACS rows; the PUMS is a subsample of the published
sample and district and county are drawn from PUMA, so ACS rows do not
reproduce them by construction, but they share its sampling error. Treat
ACS-native scores as consistency with the published ACS.

### Calibration

Every arm uses the production optimizer, `microcosm.calibrate.solve._optimize`,
with the fiscal release's optimizer settings: Adam on log-weights, learning
rate 0.02, capped relative error (cap 1.0), mass conserved, max weight ratio 5,
the release's concept-budget loss weights computed once per surface (the
national_state surface for the national product, the full registry for the
local one) before the split, so train and holdout keep fixed weights; district
household-population rows weigh 1 each; no L0 pruning, so the row budget
stays fixed. A sparse operator with a precomputed transpose replaces the torch
sparse-CSR backward pass, which was the bottleneck; the objective and
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
**Differential**: the sparse target matrix equals a naive dense construction,
including cells that carry several targets; `diffcheck` compares production
per-target materialization with concept × mask on real rows.

### Scoring

- **Registry holdout**: capped relative error by dimension (CPS-native:
  income components, tax items, program receipt; ACS-native: PEP
  demographics) and level, as a mean and loss-weighted. A target no row in the
  arm can move scores as a miss (error 1), so every arm is averaged over the
  same held-out set; a common-support table is reported too. Two held-out
  targets are unsupported in every arm, and 22 in the one-year CPS-only arm.
- **ACS-native**: households, tenure, aggregate rent (the model's
  `pre_subsidy_rent`, scored against both contract rent B25060 and gross rent
  B25065) and real-estate taxes, household population (B25008), and total
  population, age and sex (B01003, B01001), by level including county.
  Household measures exclude group-quarters rows.
- **ESS** over distinct households: weights summed within each household unit,
  then Kish; also the median across districts.
- **Cost**: calibration wall time and peak RSS per arm; engine seconds per
  household from the materialization receipts; physical rows as the serving
  cost driver.
- **Report only**: SPM poverty rate, all persons and children.

## CPS year response quality

`census_cps_YYYY.h5` is keyed by income year (ASEC collected in YYYY+1).
`asec_quality.py` computes the table from the processed files and the Census
public-use archives; the response rates are Census's own.

| Income year (ASEC) | Households | Combined response rate | Earners, whole record imputed | Earners, items allocated | Homeownership | UC recipients |
|---|---|---|---|---|---|---|
| 2018 (2019) | 68,345 | 67.6% | 24.2% | 21.3% | 64.5% | 1,723 |
| 2019 (2020) | 60,460 | 61.1% | 23.8% | 21.3% | 66.4% | 1,563 |
| 2020 (2021) | 62,850 | 65.0% | 22.4% | 22.6% | 65.8% | 10,915 |
| 2021 (2022) | 59,148 | 61.4% | 22.4% | 21.8% | 65.5% | 4,065 |
| 2022 (2023) | 56,839 | 59.6% | 21.9% | 21.7% | 65.9% | 1,225 |
| 2023 (2024) | 56,251 | 59.3% | 20.7% | 21.0% | 65.6% | 1,360 |
| 2024 (2025) | 55,762 | 60.1% | 19.3% | 21.5% | 65.1% | 1,400 |

Combined response is basic-CPS times ASEC household response, from each
year's CPS ASEC Source and Accuracy statement. Imputation shares are
`A_FNLWGT`-weighted among wage earners (`I_ERNVAL` 9 is a whole imputed
record; 1–8 an allocated item). Household weights have a coefficient of
variation of 0.598 to 0.610 and a Kish share of 0.729 to 0.737 in every year.
The public-use archives give whole-supplement imputation (`FL_665`) of 20.5%,
20.0%, 19.2%, 17.8% and 16.7% for income years 2021 to 2025.

Income years 2020 and 2021 **pass**: their response rates are above the
pinned years', their imputation and allocation shares sit between the
pre-pandemic years' and the pinned years', and homeownership and weight
dispersion show no break. Three caveats:

- Census asks users to take care comparing data years 2019, 2020 and 2021
  with other years, and documents that nonrespondents in the 2020–2026
  surveys differ more from respondents than before (2026 CPS ASEC technical
  documentation). That caution covers the pinned 2022–2024 years too.
- Their income years carry pandemic program receipt: 10,915 and 4,065
  unemployment-compensation recipients, against 1,225 to 1,723 in other
  years. That matters for program-receipt targets.
- About a third of each file's households repeat in the neighbouring year
  (32.5% to 37.8%).

Income year 2019 (ASEC 2020, collected as COVID hit) is the file that fails:
homeownership jumps 1.9 points (64.5% to 66.4%) and falls back the next year,
and the sample has 11.5% fewer households than the year before (60,460
against 68,345).

Rotation overlap: a housing unit appears in at most two adjacent files, so
2022–2024 has 130,294 distinct units in 168,852 households (77.2%), and
2020–2024 would have 211,834 in 290,850 (72.8%).

## Caveats

- **CPS rows are income years 2022–2024**, the Route A export available when
  this ran. The recommendation is stated for "the newest three years"; the
  2023–2025 export is the rerun.
- **Imputations are held fixed.** ACS rows' CPS-only inputs come from one
  57,240-household donor spine in every arm. A production one-year build
  would also impute from a thinner donor pool.
- **Clones keep their origin row's engine values**, including anything that
  varies by county within a state.
- **The holdout mirrors the unmerged US split without its pins.** Scores are
  comparable across arms, not with the eventual release split.
- **ACS-native truth is the published ACS 2024 1-year**, the same survey year
  as the ACS rows. Read it as consistency with the published ACS.
- **Older CPS years are not uprated**, as in Route A; calibration absorbs the
  2022–2024 nominal income gap.
- **5,246 group-quarters records with no SPM adult are out of the ACS
  support**, pending the engine's SPM-universe input (policyengine-us#9462).
- **JCT tax-expenditure targets are excluded**; they need reform simulations.

## Follow-ups

- Rerun on the newest three CPS years (2023–2025) once a Route A export on main
  at or after `5187fce25` exists: `shard --source cps` on it, `materialize
  --source cps`, then `run-grid`. Receipts carry digests of their inputs, so
  `run-grid` re-solves every arm whose inputs changed and keeps the ACS-only
  arms.
- Add five-year arms (2020–2024 or 2021–2025) once those years have Route A
  enrichment.
- When `microcosm.build.us_runtime.target_split` merges, swap it in and
  **re-solve**: a new split changes the training set, so saved weights cannot
  simply be re-scored.
- Find out why calibration worsens held-out state age cells 2.5-fold.

## Evidence

- Differential: production per-target materialization against concept ×
  mask, 4,000 households from each source, 349 state and district targets
  across nine family-level groups: maximum scaled difference 0.0 for both
  (`diffcheck.json`). The ACS half was written by the tool on 2026-09-29; the
  CPS half is the tool's own log line from 2026-09-28, copied into the file
  after the CPS shard files were removed to free disk.
- Tests: `test_us_support_mix.py`, 22 tests (properties, the dense
  differential and the port-branch hash fixtures).
- Independent review (Opus 5.5, Subfleet), two rounds; every finding is fixed
  or stated here.
- Receipts in `experiments/us-support-mix-bakeoff-20260930/`: the report
  tables, `compile.json`, `diffcheck.json`, `spm_flags.json`,
  `asec_quality.json` (with its script), `acs_truth.json`, and the
  per-part materialization receipts under `materialize/`.

Reproduce: `compile`, `shard` (both sources), `truth`, `materialize` (both
sources; ACS by rank range), `diffcheck`, `run-grid --products national` and
`--products local`, `report`, then `memo_tables.py`. Each step writes a receipt
with wall time and peak RSS.
