# US support mix: CPS years vs ACS rows at a fixed household budget

Status: the bake-off ran from 2026-09-28 to 2026-09-30 on main `bda72cb02`,
policyengine-us 2.2.1 and the pinned Chronicle feed `b8543739` (32,842
targets), and scored all 60 calibrations (30 arms × 2 products). Receipts:
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
  file's 352,932). Against the CPS-only file, as means over three ACS draws:
  - held-out state CPS-native error falls from 0.119 to 0.113 (draw SD
    0.0003) and district error from 0.353 to 0.291;
  - state income components improve (0.250 to 0.224), state tax items get
    worse (0.103 to 0.113), and state program receipt does not measurably
    change (0.208 against 0.198, draw SD 0.014);
  - every ACS-native level improves (district 0.236 to 0.188, county 0.307
    to 0.259).

  Stopping at 300k is a cost call. Larger ACS fills keep lowering district
  error (0.291, 0.266, 0.244 at 300k, 600k, 1.2M) but leave state error flat
  at 300k and 1.2M (0.113) and slightly worse at 600k (0.115), edge state tax
  items up (0.113, 0.116, 0.118) and do
  not help state program receipt (0.198, 0.225, 0.239; the 600k and 1.2M arms
  are single draws, and only the 1.2M gap exceeds draw noise). The best
  state-level arm is the 600k 50/50 fill (0.111; income 0.213, tax items
  0.112, program receipt 0.187) at 1,019,084 rows, 2.1 times the
  recommendation. Against one CPS year plus ACS to 300k, three years buy state
  income (0.249 to 0.224) and tax items (0.130 to 0.113) with ACS-native
  accuracy (state 0.086 to 0.110, district 0.166 to 0.188). The
  recommendation puts the CPS-native gains first; if ACS-native accuracy
  matters as much for the national file, one year plus ACS is the
  alternative. If state tax items and serving cost outweigh everything else,
  the three-year CPS-only file remains the choice.

- **CD and county local file: the newest three CPS years plus ACS households
  to 1.2M source households, with no CPS location clones.** That is 1,384,080
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
  One CPS year instead of three saves 123k rows (9%) at 1.2M. There the data
  do not separate them on CPS-native measures: district error is level
  (0.169 each), and the state gaps (0.116 against 0.112; tax items 0.123
  against 0.117; program receipt 0.278 against 0.256, single draws) sit
  within the one-year arms' draw noise, or for state error at its edge. One
  year is better on ACS-native state and district measures (0.072 and 0.136
  against 0.078 and 0.141). Three years win at 300k, where there are three
  draws of each: clearly on state error (0.119 against 0.112) and tax items
  (0.137 against 0.103), and just beyond draw noise on program receipt (0.288
  against 0.237). The three-year choice rests on that, on the national
  evidence below, and on the local file sharing the national file's CPS
  support.

- **More CPS years improve CPS-native measures and cost ACS-native ones.**
  Within a budget, each CPS year displaces ACS households. In the national
  file at 300k, one year against three (three draws each): state income 0.249
  against 0.224 and tax items 0.130 against 0.113, both well beyond draw
  noise; program receipt 0.241 against 0.198 falls within the one-year arms'
  draw noise (SD 0.041), and so does the loss-weighted state error (0.117
  against 0.113, SD 0.002). ACS-native error rises (state 0.086 to 0.110,
  district 0.166 to 0.188; county level at 0.255 and 0.259). The local file at
  300k shows the same trade more strongly: tax items (0.137 to 0.103) and
  program receipt (0.288 to 0.237) improve, state income is level, and
  ACS-native error rises by a third or more (state 0.080 to 0.110, district
  0.153 to 0.204). At 1.2M the displacement is small (see above). Income
  years 2020 and 2021 pass the response-quality check, so a five-year pool is
  the next arm to run; this bake-off did not run it (see Follow-ups).

- **Clones add geography and little else.** In the national file they leave
  every state measure where the CPS-only file had it. In both files they
  improve district measures slowly (national-file district CPS-native error
  0.353, then 0.332 at 300k) and household ACS-native measures barely
  (local-file county 0.299, then 0.263 and 0.253 at 600k and 1.2M), at 1.3 to
  2.0 times the physical rows of an ACS fill of the same budget (1.3 to 1.8
  for three-year clones).

This matches the design note's reasoning. ACS rows carry real variation in
what the ACS observes, so income components (the ACS records wages,
self-employment, Social Security, retirement and interest, dividend and rental
income), housing and local geography improve as ACS rows enter. ACS rows
draw program receipt from CPS donors: beside three CPS years, state program
receipt error holds level at 131k ACS households, turns clearly worse by 1.2M
(0.239), and reaches 0.303 for ACS alone at 300k.

## Results

Every column reports capped relative error on held-out targets, `min(|est −
target| / max(|target|, 1), 1)`; lower is better. The "CPS-native" columns
take the loss-weighted mean over every held-out CPS-native target at the
level; the per-dimension columns take plain means. "ACS-native" averages
households, owner and renter households, contract rent, real-estate taxes and
household population against the ACS 2024 1-year tables;
`acs_native_by_level_measure.csv` adds age bands, which count group-quarters
people only the ACS rows have. The Rows column counts physical household rows
(a CPS household carries its PUF tax-detail clone rows); ESS is Kish over
distinct households (each ASEC housing unit counts once across rotation
years, PUF rows and clones). The "3 draws" rows average three disjoint ACS
draws; each other row comes from one draw. `memo_tables.py` builds the tables
from `dimensions_by_arm.csv`.

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

Differences under about twice the draw SD are noise. The one-year arms vary
more across draws than the three-year arms (program receipt SD 0.041
national and 0.024 local at 300k; `replicate_spread.csv`). The 36
national-level held-out targets are too few to rank arms, so the tables leave
them out; `holdout_by_dimension_level.csv` has them.

**Calibration worsens held-out state age cells, in every arm.** The mean
capped error on held-out Census PEP state age bands is 0.066 at the starting
weights and 0.168 after calibration, across all 60 calibrations (0.131 to
0.193). The design weights already carry Census age controls, and the solve
trades them away for its training targets. The tables report the age column
but rank no arm on it: it does separate arms (national 0.159 to 0.176, draw
SD 0.005), but every arm pays the trade, and its size says more about the
calibration objective than about the support (see Follow-ups).

**Report only: SPM poverty** (no arm calibrates to it or gates on it). In the
national product the SPM rate is 12.8% for the three-year CPS file, 12.2% for
the recommended 300k mix (seed 0) and 12.3% for ACS only; child SPM poverty
is 14.7%, 13.6% and 12.3%. These rates include the ACS group-quarters persons
still in the support, whom Census places outside the SPM universe; the tool
now leaves them out, but these rates predate that fix. ACS rows' SPM
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

We measured wall times with the host at load 10 to 400, so they compare only
within this run. Training loss still fell a median 5.2% of its epoch-1,000
value between epochs 1,000 and 1,500 (`arms.csv` has `loss_1000` and
`loss_final`); every arm had the same 1,500-epoch budget. The tool first
scored 22 arms before scoring covered every level; for those, `arms.csv`
carries the solve peak from the grid log and no wall time (`runtime_restored`
names the log line). We added each arm receipt's `inputs` digests after its
solve, once the tool recorded them, from the files then in place, and each
receipt's `inputs_note` said so. We likewise added source-h5, concept-spec
and null-fill-summary digests afterwards to the work copies of the
concept-part receipts, from `shard.json`, `concept_specs.pkl` and the
`partcheck` below (`*_note` fields). The copies under `materialize/` predate
those digests; on 2026-09-30 their numerical fields matched the work
copies'.

## Design

### Arms

A **source household** is one survey household record: one ASEC
household-year or one ACS household. Budgets count source households, as in
the design note, so the 2022–2024 ASEC pool is 168,852. A CPS source
household keeps its PUF tax-detail clone rows from the Route A pipeline, so
it spans two or three physical rows. The tables report physical rows
separately, and physical rows drive engine and serving cost.

Each arm keeps every household of its CPS income years and fills the rest of
the budget with:

- **ACS** households (`acs100`): the first *n* in a salted hash order of
  `SERIALNO`, so a smaller arm's ACS households are always a subset of a
  larger arm's;
- **CPS location clones** (`acs000`): copies of the arm's own CPS households,
  each physical row drawing a new (district, county) within its state in
  proportion to ACS household weight (leaving out group-quarters rows), the
  way Route A draws rows independently. A clone keeps its source row's engine
  values, so anything that varies by county within a state (ACA benchmark
  premiums by rating area) comes from the origin county;
- a 50/50 mix (`acs050`, at 600k only).

| Budget | CPS years | Fills |
|---|---|---|
| natural size | 2024; 2023–24; 2022–24 | none |
| 300k, 600k, 1.2M | 2024; 2023–24; 2022–24 | ACS; clones |
| 300k, 600k, 1.2M | none | ACS only |
| 600k | 2024; 2022–24 | 50/50 |
| 300k | 2024; 2022–24 | ACS, replicates 1 and 2 |

That is 30 arms. Each arm gets two calibrations, one per product:

- **national compact**: the release's `national_state` surface (5,683
  usable national and state targets, 4,828 of them train-role after the
  split), with loss weights the tool computes on that surface as the release
  does;
- **local**: the full surface, which adds 27,148 targets the release
  classifies as district targets (24,340 district-level, plus 2,808 state
  and national totals from the SOI district file), plus district household
  population (ACS B25008). The ACS local release trains district population
  from its PUMA ladder instead; this bake-off uses household population so
  that the district total neither counts nor moves group-quarters rows, which
  only the ACS has.

Replicate arms take the next disjoint blocks of the one fixed ACS order, so
they are independent draws; seed-0 arms nest across budgets. Group-quarters
households (11.96% of ACS records) stay in the ACS side, as in the ACS local
release; the CPS has none. The bake-off drops the 5,246 ACS records (0.34%,
all group quarters) whose SPM unit has no member the pinned engine counts as
an adult: policyengine-us 2.2.1 refuses them, and Census puts ACS group
quarters outside the SPM universe (`us_runtime/spm_universe_source.py`).
`shard` and `spm-flags` compute the flag with the engine's rule
(`spm_flags.json`).

**This bake-off ran no five-year arm.** Income years 2020 and 2021 pass the
response-quality check below, but main pinned ASEC role and person sources
only for 2022–2024 when this ran, and the base and release stages that would
enrich two more years peaked at 72–80 GB in recorded runs, beyond this
task's 60 GB ceiling, while a Route A release held the machine. The tool
takes more years as more rows, so a five-year arm is a rerun once a Route A
export with those years exists (see Follow-ups). The same holds for the
newest three years (2023–2025): main pinned the 2025 ASEC on 2026-09-29
(PolicyEngine/microcosm#1044), after the Route A export these CPS rows come
from.

### Superpool: one engine pass, many arms

Rerunning the pipeline per arm would cost hours and 70–94 GB each. Instead
the bake-off computes every household's target contributions once and
reweights subsets:

- **CPS rows**: the Route A 310842b release export (352,932 physical rows:
  income years 2022–2024, PUF clones, all enrichment stages), with design
  weights from its base. The run failed only its post-export smoke gate.
- **ACS rows**: the 2026-09-23 ACS local staging (1,531,614 ACS 2024 1-year
  households, whose CPS-only inputs the staging transferred from the
  57,240-household receipt-qualified donor spine). The bake-off leaves out
  the staging's own 57,240 donor rows. The staging carries
  `is_spm_independent_minor_role` as all-null on ACS rows; the tool drops the
  column so the engine derives the role from its formula
  (`is_household_head | is_household_spouse`), as the production ACS local
  build did.

Both carry the same PolicyEngine input schema (141 and 140 person inputs).
Imputations stay the same across arms: an ACS row's CPS-only variables come
from the same donor spine whatever CPS years the arm pools. The bake-off
therefore measures the support effect while holding donor quality constant;
in a real rebuild, a one-year arm would also have a thinner donor pool.

**Concepts.** The release materializer builds each state or district
target's household column as a geography-free value times an indicator of
the household's own state or district
(`_base_simulation_household_columns`). `compile` groups the 32,842 targets
into 419 such concepts, and `materialize` runs the release tool's own
`_materialize_target_frame` on them in household shards. The tool then
assembles every arm's target matrix, sparse, as concept value × geography
mask. `diffcheck` verifies the identity on real rows (see Evidence). The 11
JCT tax-expenditure targets need reform simulations, so every arm leaves them
out.

### Holdout

The US holdout ("Port the UK holdout to US release targets") had not merged
when this ran. The bake-off mirrors its pending spec: it hashes
`salt + "\x1f" + key` with SHA-256, seals a group whose hash on the sealed
salt falls below 0.05, and otherwise holds the group out if its hash on the
holdout salt falls below 0.10. Tests check the hash against draws from the
port branch's own implementation (`PORT_BRANCH_FIXTURES`); the group-key and
role rules follow the port session's written spec, since that session had
not yet written its module. District children take their state parent's
key, so holding out a state total holds out all its districts; the split
draws national and state targets independently, so trained state targets
can still pin a held-out national total. Keys drop vintage tokens. The
bake-off does not mirror the spec's pins, which force some groups to train,
so this split differs from the eventual release split for pinned groups. No
arm trains on or scores a sealed target.

On the pinned feed: 27,711 train, 3,470 holdout (36 national, 787 state,
2,647 district), 1,661 sealed.

**ACS-native truth.** The registry has no housing targets and no county
targets. The bake-off scores ACS-native dimensions against the ACS 2024
1-year table-based summary file (B25003 tenure, B25060 and B25065 aggregate
rent, B25090 aggregate real-estate taxes, B25008 and B01003 population,
B01001 age and sex) at national, state, district (119th Congress) and county
(the 850 counties the 1-year file covers). No arm calibrates to them, except
to district household population (B25008) in the local product. These
tables come from the same ACS year as the ACS rows; the PUMS subsamples the
published sample, and ACS rows take district and county as draws within
their PUMA, so ACS rows do not reproduce the tables by construction, but
they share the published sample's sampling error. ACS-native scores
therefore measure consistency with the published ACS.

### Calibration

Every arm uses the production optimizer, `microcosm.calibrate.solve._optimize`,
with the fiscal release's optimizer settings: Adam on log-weights, learning
rate 0.02, capped relative error (cap 1.0), conserved mass, max weight ratio
5, and the release's concept-budget loss weights, which the tool computes
once per surface (the national_state surface for the national product, the
full registry for the local one) before the split, so train and holdout keep
fixed weights; district household-population rows weigh 1 each; no L0
pruning, so the row budget stays fixed. A sparse operator with a precomputed
transpose replaces the torch sparse-CSR backward pass, which was the
bottleneck; the objective and projections stay the same.

Starting weights: each CPS income year gets an equal share of the CPS mass;
a household with *k* location copies splits its weight over 1 + *k* rows;
CPS and ACS sides share mass in proportion to source households; then the
release's population mass repair rescales to its 334.2M-person benchmark.

**Invariants** (`test_us_support_mix.py` tests each as a property): an arm's
CPS, ACS and clone households sum to its budget; ACS selections nest and do
not depend on input order; clone copies differ by at most one; starting
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
  demographics) and level, as a plain and a loss-weighted mean. A target no
  row in the arm can move scores as a miss (error 1), so every arm's average
  covers the same held-out set; a common-support table covers only targets
  every arm supports. Two held-out targets lack support in every arm, and 22
  in the one-year CPS-only arm.
- **ACS-native**: households, tenure, aggregate rent (the model's
  `pre_subsidy_rent`, against both contract rent B25060 and gross rent
  B25065) and real-estate taxes, household population (B25008), and total
  population, age and sex (B01003, B01001), by level including county.
  Household measures exclude group-quarters rows.
- **ESS** over distinct households: Kish on each household unit's total
  weight; also the median across districts.
- **Cost**: calibration wall time and peak RSS per arm; engine seconds per
  household from the materialization receipts; physical rows as the serving
  cost driver.
- **Report only**: SPM poverty rate, all persons and children.

## CPS year response quality

In `census_cps_YYYY.h5`, YYYY is the income year; Census collects that
year's ASEC in YYYY+1. `asec_quality.py` computes the table from the
processed files and the Census public-use archives; the response rates are
Census's own.

| Income year (ASEC) | Households | Combined response rate | Earners, whole record imputed | Earners, items allocated | Homeownership | UC recipients |
|---|---|---|---|---|---|---|
| 2018 (2019) | 68,345 | 67.6% | 24.2% | 21.3% | 64.5% | 1,723 |
| 2019 (2020) | 60,460 | 61.1% | 23.8% | 21.3% | 66.4% | 1,563 |
| 2020 (2021) | 62,850 | 65.0% | 22.4% | 22.6% | 65.8% | 10,915 |
| 2021 (2022) | 59,148 | 61.4% | 22.4% | 21.8% | 65.4% | 4,065 |
| 2022 (2023) | 56,839 | 59.6% | 21.9% | 21.7% | 65.9% | 1,225 |
| 2023 (2024) | 56,251 | 59.3% | 20.7% | 21.0% | 65.6% | 1,360 |
| 2024 (2025) | 55,762 | 60.1% | 19.3% | 21.5% | 65.1% | 1,400 |

Combined response is basic-CPS times ASEC household response, from each
year's CPS ASEC Source and Accuracy statement. Imputation shares weight wage
earners by `A_FNLWGT` (`I_ERNVAL` 9 marks a whole imputed record; 1–8 an
allocated item). Household weights have a coefficient of
variation of 0.598 to 0.610 and a Kish share of 0.729 to 0.737 in every year.
The public-use archives give whole-supplement imputation (`FL_665`) of 20.5%,
19.9%, 19.2%, 17.8% and 16.7% for income years 2021 to 2025.

Income years 2020 and 2021 **pass**: their response rates exceed the pinned
years'; their whole-record imputation shares sit between the pre-pandemic
years' and the pinned years'; their item-allocation shares, the highest of
the seven years (22.6% and 21.8%), stay within 1.6 points of every other
year; and homeownership and weight dispersion show no break.
Three caveats:

- Census asks users to take care comparing data years 2019, 2020 and 2021
  with other years, and documents that nonrespondents in the 2020–2026
  surveys differ more from respondents than before (2026 CPS ASEC technical
  documentation). That caution covers the pinned 2022–2024 years too.
- Their income years carry pandemic program receipt: 10,915 and 4,065
  unemployment-compensation recipients, against 1,225 to 1,723 in other
  years. That matters for program-receipt targets.
- About a third of each file's households repeat in the neighbouring year
  (32.5% to 37.8%).

Income year 2019 (ASEC 2020, which Census collected as COVID hit) fails:
homeownership jumps 1.9 points (64.5% to 66.4%) and falls back the next year,
and the sample has 11.5% fewer households than the year before (60,460
against 68,345).

Rotation overlap: a housing unit appears in at most two adjacent files, so
2022–2024 has 130,294 distinct units in 168,852 households (77.2%), and
2020–2024 would have 211,834 in 290,850 (72.8%).

## Caveats

- **The CPS rows cover income years 2022–2024**, from the Route A export
  available when this ran. The recommendation names "the newest three
  years"; a rerun would use the 2023–2025 export.
- **Imputations stay the same across arms.** ACS rows' CPS-only inputs come
  from one 57,240-household donor spine in every arm. A production one-year build
  would also impute from a thinner donor pool.
- **Clones keep their origin row's engine values**, including anything that
  varies by county within a state.
- **The holdout mirrors the unmerged US split without its pins.** Scores
  compare across arms, but the eventual release split would give different
  ones.
- **ACS-native truth is the published ACS 2024 1-year**, the same survey year
  as the ACS rows, so it measures consistency with the published ACS.
- **Nothing uprates older CPS years**, as in Route A; calibration absorbs the
  2022–2024 nominal income gap.
- **The ACS support leaves out 5,246 group-quarters records with no SPM
  adult**, pending the engine's SPM-universe input (policyengine-us#9462).
- **Every arm leaves out the JCT tax-expenditure targets**, which need reform
  simulations.

## Follow-ups

- Rerun on the newest three CPS years (2023–2025) from a Route A export of
  main at or after `5187fce25`. The run's work directory no longer exists, so
  a rerun takes every step under Reproduce below, with `--newest-year 2025` on
  `run-grid` and `report`. Within a rerun, `materialize` refuses a concept
  part whose receipt is missing or names another source h5, other concept
  specs or another null-fill summary, unless run with `--replace-stale`, and
  an arm's receipt counts as current only while every input the arm reads
  stays the same: the targets, truth and epoch count, and, for each source it
  draws on, the row index, the concept parts, the concept specs and the
  measure list.
- Add five-year arms (2020–2024 or 2021–2025) once those years have Route A
  enrichment.
- When `microcosm.build.us_runtime.target_split` merges, swap it in and
  **re-solve** every arm: a new split changes the training set.
- Find out why calibration worsens held-out state age cells 2.5-fold.

## Evidence

- Differential: production per-target materialization against concept ×
  mask, 4,000 households from each source, 349 state and district targets
  across nine family-level groups: maximum scaled difference 0.0 for both
  (`diffcheck.json`). The tool wrote the ACS half on 2026-09-29; the CPS half
  is the tool's own log line from 2026-09-28, which we copied into the file
  after removing the CPS shard files to free disk.
- Stored parts against the final concept specs: `partcheck` re-materialized
  200 households from each of the 8 CPS parts and 8 of the 62 ACS parts
  (both rank ranges) with the final specs and the branch's code, and compared
  every column with the stored values: maximum scaled difference 5.8e-8,
  float32 rounding (`partcheck_cps.json`, `partcheck_acs.json`). Three CPS
  parts predate the last `compile`, which added loss weights only; this check
  confirms their columns. For the check we rebuilt the CPS shards from the
  same h5, and their row index equals the one the arms used.
- On 2026-09-30, rerunning `report` on the 60 arm receipts reproduced every
  committed table byte for byte. Those receipts (10–15 MB each, with every
  held-out target's estimate) and the saved weights stayed in the run's work
  directory, which no longer exists, so the committed tables now carry the
  record.
- Tests: `test_us_support_mix.py`, 22 tests (properties, the dense
  differential and the port-branch hash fixtures), and
  `test_bakeoff_us_support_mix_tool.py`, 24 tests of the driver's SPM flag,
  staleness guards, `partcheck` rules, grid and report.
- After the run, the tool gained four guards a final review asked for:
  receipt digests also cover the concept specs and the measure list,
  `partcheck` fails a column with any nonfinite value and refuses an unknown
  or empty part selection, and `--allow-stale` alone now writes a report.
  The recorded checks had finite differences and matched every part they
  named, so no committed number moves.
- Receipts in `experiments/us-support-mix-bakeoff-20260930/`: the report
  tables, `compile.json`, `diffcheck.json`, `partcheck_*.json`,
  `spm_flags.json`, `asec_quality.json` (with its script), `acs_truth.json`,
  and the per-part materialization receipts under `materialize/`.

Reproduce: `compile`, `shard` (both sources), `truth`, `materialize` (both
sources; ACS by rank range), `diffcheck`, `partcheck` (both sources),
`run-grid --products national` and `--products local`, `report --truth`, then
`memo_tables.py`. Each step writes a receipt with wall time and peak RSS.
