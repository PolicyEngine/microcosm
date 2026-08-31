# K=1 rowwise pilot receipts (#761, #495 increment 5)

Round A: spine-i, 2026-08-29, licensed machine. Round B (spine-j, after the
#804 licensed phase re-cuts the acceptance receipt) re-runs the same two
commands and appends its section below; per María's 2026-08-28 ruling the K
adjudication cites Round B, with Round A as the control.

## R0 — input identity (Round A)

| item | value |
|---|---|
| input | `spine-i.h5`, sha256 `0cce03992207b4e16d96483642e2edd943fd25112a6425208d748b31c7742416` (= the committed `uk/spine_candidate_acceptance.json` candidate pin), 153,184,127 bytes |
| pin arm | `--input-sha256` verified fail-closed; manifest `inputs.dataset.pin_verified: true` |
| weight kind | `importance` in, `importance` out; input mass log 6 records |
| ladder | `uk_oa_ladder_2021.npz`, sha256 `9c6d56b90d2e975d750106b175020a54c5ec6acf42ef8909d304a9d7fc3868a7`, `matches_local_area_crosswalk_pin: true` |
| code | `652d2891` (`uk-rowwise-pilot-761` rung-8 tip, on the merged #795+#801 main) |
| receipts | dry-run plan sha `a44db1464e3a50ed…`, build manifest sha `b1a1294afeecff6e…`, evidence dir `data/ukds/acceptance/761-rowwise-pilot/spine-i/` (licensed-side) |

## R1 — the multi-K dry-run plan

One invocation (`--n-clones 1 --candidate-clone-counts 1,2,4,10 --dry-run`,
seed 42). Every candidate K runs the identical fenced clone → weight-divide →
ladder-assign → gate pipeline at the build seed; `expected` is the analytic
expectation from the sampler's own stage weights. Bytes are the declared
input×K lower bound; the real K=1 build measured ×1.044 over it (added
geography columns), giving the calibrated ceiling column.

### Constituency grain (650 areas, all assigned at every K)

| K | households | H5 est. (calibrated) | rows min / med (realized) | rows min / med (expected) | ESS min / med | distinct sources min / med |
|--:|--:|--:|--:|--:|--:|--:|
| 1 | 52,846 | 153 MB (≈160 MB) | 28 / 72 | 27.3 / 71.7 | 15.3 / 62.2 | 28 / 71 |
| 2 | 105,692 | 306 MB (≈319 MB) | 42 / 144 | 54.6 / 143.5 | 28.8 / 120.7 | 42 / 137 |
| 4 | 211,384 | 613 MB (≈640 MB) | 101 / 287 | 109.3 / 286.9 | 86.4 / 240.4 | 97 / 256 |
| 10 | 528,460 | 1,532 MB (≈1,600 MB) | 301 / 714.5 | 273.2 / 717.3 | 245.0 / 592.3 | 276 / 552.5 |

### Local-authority grain (361 areas, all assigned at every K)

| K | rows min / med (realized) | rows min / med (expected) | ESS min / med | distinct sources min / med |
|--:|--:|--:|--:|--:|
| 1 | 1 / 106 | 1.7 / 106.6 | 1.0 / 89.4 | 1 / 102 |
| 2 | 3 / 215 | 3.3 / 213.2 | 2.4 / 182.5 | 3 / 201 |
| 4 | 7 / 420 | 6.7 / 426.4 | 6.8 / 355.4 | 7 / 364 |
| 10 | 17 / 1,064 | 16.6 / 1,066.1 | 12.1 / 892.2 | 17 / 721 |

The LA minimum is the Isles of Scilly (`E06000053`; ~0.05% of national
households), then the City of London (`E09000001`): population-proportional
assignment gives them almost nothing at any affordable K — a support floor,
if one is wanted there, is target-side or assignment-side work, not a clone
multiplier. The constituency floor is Na h-Eileanan an Iar (`S14000027`),
exactly the area the epic predicted (~240 rows at the old pool's K=1 scale ⇒
~24 at the spine's, measured 28).

The analytic expectation tracks the realized draw at every K on both grains
(largest relative gap at the thin tail, e.g. K=2 constituency min 54.6
expected vs 42 realized — a single seed's draw at a 42-row floor). Full
per-K stats: `experiments/761-uk-rowwise-pilot-support.json`.

Basis note (post review round): the stats block now reads
`nonzero_households` (`rows_basis` recorded), so a zero-weight row can never
count as support. Round A measured `assigned == nonzero` for all 1,011 areas
— the spine's SPI channel carries positive importance mass, unlike the old
pool's zero-weight convention — so every figure above stands unchanged;
Round B emits the labelled shape natively.

## R2 — the real K=1 build

`--n-clones 1 --seed 42`, both pins armed, 2026-08-29T00:26Z:

| check | result |
|---|---|
| `uk_geography_ladder` gate | **passed** — constituency/ward/ITL nonempty weighted shares 1.0, London weighted share 0.1261 ∈ (0.08, 0.20) |
| assignment | 650/650 constituencies, 361/361 local authorities, `missing_geography_rows: 0` |
| mass conservation | `abs_delta: 0.0` (input = output = 29,247,433.0), rtol 1e-9, **exact** |
| weight-kind chain | `importance` → `importance`; mass log 6 → 7 records (the single clone record) |
| lineage | `explicit_lineage_columns` basis: 16,288 distinct `source_household_id` (frs 16,288 / spi 10,000 — the SPI channel's sources nest inside the FRS set); flags: spi_synthetic 20,085, cgt_clone 26,420, band_donor 270; `pool: null`, no modulus |
| duplicate source×constituency pairs | 1,454 (K=1: SPI/CGT copies of one source landing in its own constituency — the honest support discount the lineage columns exist to expose) |
| cost | 17.3 s wall, **1.51 GB peak RSS**, output H5 159,984,959 bytes (sha `8bb2b5fa5f28e03e…`) |
| Logbook | pipeline `uk-local-rowwise`, rung `f100`, build id `…-20260829T002607Z-42440bbd`, row digest `a4d2c0b2b356493c…` — the **genesis row of the ratified `uk/local` chain** (`logbook/uk/local.jsonl`, chain validates) |

Memory ceiling at K, linear in the clone factor from the measured K=1 point:
≈1.5 GB (K=1) → ≈3 GB (K=2) → ≈6 GB (K=4) → ≈15 GB (K=10) peak RSS, with
H5 sizes per R1. All of K ∈ {1,2,4} fit ordinary hardware; K=10 remains
feasible on the licensed machine.

## R3 — grain (the #688 evidence)

Pre-clone: 52,846 households = (16,288 raw FRS + 10,000 SPI-synthetic) × 2
+ 270 CGT band donors, carrying 16,288 distinct source households. Post-clone
at K=1: row counts unchanged (52,846 / 61,211 / 113,649), ids untouched
(`_remap_ids` is the identity at clone 0), weights divided by 1, one mass
record appended. K is a declared build parameter: `parameters.n_clones` in
the manifest, `candidates.clone_counts` in the plan — the declaration #688's
open half asked for. The old `household_id mod 10^8` lineage rule does not
apply to this spine and is structurally refused; distinct-source accounting
reads `source_household_id` directly.

## R4 — rung ruling

The Logbook rung stays `f100` at every K: rungs are sample-fraction tokens
(`DESIGN.md` scale ladder: f001 = 1% smoke, f010 = 10% development, f100 =
full; `logbook.py` `_validate_rung`: "#624 fraction token"), and the pilot
consumes 100% of the spine regardless of the clone multiplier. K lives in
`parameters.n_clones`, not the rung.

## R5 — K recommendation (adjudication pending Round B)

Measured support says the epic's "first published candidate at K=2–4" range
survives the spine swap, at the top of the range:

- **K=1 is a pilot, not a candidate**: constituency min-ESS 15.3 means the
  thinnest areas calibrate on ~15 effective households.
- **K=4 is the recommendation for #762's first calibrated candidate**:
  constituency floor 101 rows / ESS 86 / 97 distinct sources, median 287
  rows — between the US sparse and dense arms per area — at 640 MB and
  ~6 GB solve headroom. LA-grain thinness (Scilly, City of London) is
  unfixable by K and should be handled target-side.
- **K=10 buys diminishing returns** (median 714 ≈ US-dense 775, but 1.6 GB
  H5 and ~15 GB RSS) and adds zero information; hold it unless #762's solve
  diagnostics show support-limited misses at K=4.

**Adjudication (María, 2026-08-30): K=4 approved** for #762's first
calibrated rowwise candidate, on the measurement above — constituency floor
101 rows / ESS 86.4 / 97 distinct sources, median 287 rows, at ≈640 MB and
≈6 GB solve headroom — confirmed unchanged on spine-k (Round B: support
inputs measured-preserved, tables reproduce exactly). K=10 stays held unless
#762's solve diagnostics show support-limited misses at K=4; the LA-grain
floor (Scilly, City of London) is target-side work at any K. The ruled K
becomes #762's declared `--n-clones`, recorded in that build's manifest.

## Round B — spine-k (2026-08-30)

The spine line moved past spine-i in the recert lane before any acceptance
re-cut: spine-j (`spine-j-remeasure`) and then **spine-k**
(`data/ukds/acceptance/spine-k-stack/spine-k.h5`, built 2026-08-30, release
id `uk-frs-spine-20260830T120239Z`, **14/14 spine gates passed**, importance
weights, mass total 29,247,433.0, the same 52,846/61,211/113,649 **row-count
roster** — the artifact bytes differ, and the Logbook `identity_digest` is
re-derived for this run, never carried over). Round B ran against spine-k,
sha-pinned at the artifact:
`b4403ea4b2d345de06e9f297fdeb9f144519407e98d653b09cf77b16c1e436cb`. The
committed `spine_candidate_acceptance.json` still names spine-i — the
acceptance re-mint is the recert lane's step; this pilot pins bytes, not the
receipt, and records that status honestly.

Same two commands as Round A (multi-K dry-run at seed 42; real K=1 build,
Logbook row chained onto the genesis digest `a4d2c0b2…`):

- **Support is bit-identical to Round A at every K on both grains** — every
  min/median rows, ESS, and distinct-sources figure in R1's tables
  reproduces exactly. Read precisely, this establishes two things and
  entails a third. Measured: (a) spine-k **preserves the support inputs** —
  household roster, region mix, and the importance-weight vector — which was
  not knowable a priori (the recert changes could have moved any of them);
  (b) the seeded two-stage draw is **reproducible** and the assignment
  surface is **value-column-independent**. Entailed, once (a) holds: the
  support tables reproduce bit-for-bit — the metric is a function of those
  inputs only, so identity here is arithmetic, not an independent check.
  The K adjudication is defined over the support surface (#761 items 6–7),
  so it carries to spine-k because its inputs are measured-unchanged.
  Anything value-dependent — whether the solve at the ruled K fits spine-k's
  revised wealth columns — is invisible to this metric by design and belongs
  to #762's fit diagnostics, not this pilot.
- Build receipts: gate passed 650/650 + 361/361, `missing_geography_rows: 0`,
  mass `abs_delta: 0.0`, `importance` carried, mass log 6 → 7, lineage
  identical (16,288 distinct sources; frs 16,288 / spi 10,000; flags
  20,085 / 26,420 / 270). 11.4 s wall, 1.51 GB peak RSS.
- Receipts: dry-run plan sha `5edc378eeac7e1ab…`, build manifest sha
  `ded07a6122a3c36e…`, evidence `…/761-rowwise-pilot/spine-k/`; Logbook row
  `15f6b09944f749cd…` (chain tail; `logbook/uk/local.jsonl` validates at 2
  rows). Stats emitted in the post-review `rows_basis: nonzero_households`
  shape; `assigned == nonzero` for all 1,011 areas here too.
