# US ACS local-area SOI target surface

`tools/build_us_acs_local_release.py --stage materialize` calibrates the ACS
local-area artifact to an administrative surface: USDA SNAP, CMS Medicaid
enrollment and IRS SOI, plus PUMA-ladder population marginals. `--soi-mode`
decides which SOI specs it keeps.

| `--soi-mode` | What it keeps | Status |
|---|---|---|
| `state` | State-level `irs_soi` specs at `ledger_geography_level == "state"` whose `ledger_layout_record_set_spec_id` is not a congressional-district file, of any `target_role` | Default |
| `totals` | State-level `irs_soi` specs whose `target_role` is not `soi_fiscal_distribution` | Explicit opt-in |
| `full` | Every state-level `irs_soi` spec | Explicit opt-in |
| `state_cd` | `state`, plus the congressional-district file's district rows reconciled to one vintage per state concept, and the district file's state rows for concepts Historic Table 2 lacks (see [The `state_cd` surface](#the-state_cd-surface)) | Explicit opt-in |

Every mode now materializes into a sparse target matrix (see
[Target matrix storage](#target-matrix-storage)), so the surface size is no
longer bounded by a dense households x targets matrix.

"State-level" means the spec's metadata carries `state_fips`, which the
congressional-district SOI rows also carry. The rule lives in
`soi_surface_predicate`; `state_admin_specs` applies it after the production
compile (`compile_us_fiscal_target_registry(age_targets=True)` and the RI
Medicaid substitution). `state` does not select a spec that lacks either
`ledger_geography_level` or `ledger_layout_record_set_spec_id`, since it cannot
tell which contract such a spec belongs to. An unknown mode is refused before
the feed is read. The national release does not use this switch:
`tools/build_us_fiscal_refresh_release.py` has no SOI mode and does not call
`state_admin_specs`.

## Why `state` is the default

`state` is the SOI contract every earlier ACS local-area release was calibrated
to. Build O (`populace-us-2024-buildo-acs-local-77e2061-20260724T110908Z`)
selected it as `full` while `state_admin_specs` compiled with
`include_congressional_district_targets=False` (populace `77e2061`,
`tools/build_us_acs_local_release.py:151-154`). Build P's ACS local release
(`populace-us-2024-buildp-acs-local-592ae5d6-20260819T020303Z`) calibrated to
the same 4,459 targets. Commit `b7922b089` (2026-08-21, "compile one target
surface unconditionally") removed that switch, so `full` grew to include the
TY2023 congressional-district SOI file and no mode reproduced the historical
contract until `state`.

Max chose `state` as the default on 2026-09-22, after the measurements below
showed that `totals` carries no state AGI, income-tax or EITC total and that
`full` does not fit one 128 GB machine. It is the only surface that keeps the
state income targets and fits, and it lets a new local build be scored against
the published Build O on the same targets. Before that day the default was
`full`; for part of 2026-09-22, on this branch only, it was `totals`.

`state` depends on the restated-filter support of #969 (merged 2026-09-22):
without it the materializer's Ledger-filter guard refuses 1,014 of the
surface's specs.

## The surfaces on the pinned feed

Measured on 2026-09-22 by calling `state_admin_specs(feed, ["snap",
"medicaid", "soi"], soi_mode=...)` on the pinned
`consumer_facts_us_c5e5bf8.jsonl` (sha256 `b8543739…`, the `facts_sha256` in
`packages/microcosm-build/src/microcosm/build/us/chronicle_feed.json`). The
Chronicle `b571381` consumer artifact gives the same counts.

| Family | `state` | `totals` | `full` |
|---|---:|---:|---:|
| `usda_snap` | 102 | 102 | 102 |
| `cms_medicaid` (enrollment) | 51 | 51 | 51 |
| `irs_soi` | 3,819 | 607 | 30,913 |
| **Admin specs** | **3,972** | **760** | **31,066** |

The 487 population marginals (51 states and 436 congressional districts) are
added on top in every mode, so `state` calibrates to 4,459 targets: Build P's
set, and Build O's 4,461 minus the Vermont under-$1 taxable-interest pair the
compiler now excludes as unsupported (`US_FISCAL_TARGET_SUPPORT_EXCLUSIONS` in
`us_runtime/fiscal_targets.py`).

### What `state` contains

All 3,819 SOI specs sit at state geography and come from the three TY2022
Historic Table 2 state tables:

| Record set spec | Specs | Content |
|---|---:|---|
| `irs_soi.historic_table_2.state_broad_totals.v1` | 2,397 | 47 all-income-range measures x 51 states, including AGI, income tax, and ACA premium tax credit returns and amounts |
| `irs_soi.historic_table_2.state_agi_counts_and_amounts.v1` | 912 | taxable interest by AGI band |
| `irs_soi.historic_table_2.state_eitc.v1` | 510 | EITC returns and amounts by number of qualifying children |

It holds no congressional-district SOI row and no row from the TY2023
congressional-district file. The feed-gated test
`test_pinned_feed_soi_surfaces_match_their_contracts` pins these counts.

### What `totals` contains, and what it does not

All 607 SOI specs that `totals` keeps are ACA premium tax credit measures (547
returns, 60 amounts), 454 of them at congressional-district geography.
`_soi_target_role` in `us_runtime/fiscal_targets.py` gives named roles only to
country-level all-income-range facts, the all-income-range PTC measures and
the W-2 tip items; every other SOI fact gets `soi_fiscal_distribution`,
whether or not it is an AGI band. **`totals` therefore carries no state or
district AGI, income-tax or EITC total.**

### What `full` adds

`full`'s 30,306 `soi_fiscal_distribution` specs split by
`ledger_geography_level` and by whether an AGI bound is finite:

| Geography | Income range | Geographies | SOI variables | Specs |
|---|---|---:|---:|---:|
| Congressional district | All | 436 | 25 | 23,886 |
| State | All | 51 | 28 | 5,508 |
| State | AGI band | 51 | 1 (`taxable_interest_income`) | 912 |

Its state all-income-range rows include the same state concepts twice, once
from the TY2022 Historic Table 2 and once from the TY2023 congressional-district
file (`<st>_total` rows); the two copies disagree by a few percent after both
are aged to 2024. Its district rows reconcile to their state parents in the
same file. Using them needs one vintage per state concept and a sparse target
matrix.

### Matrix size before the sparse checkpoint

Until 2026-09-27 `materialize_chunked` allocated one float32 column per admin
spec for every household (`np.memmap(..., dtype=np.float32,
shape=(n_households, len(names)))`), and `write_lean_checkpoint` copied that
memmap into memory. At 1,588,854 households, 4 bytes x households x specs
gave (arithmetic from the code):

| Surface | Admin specs | Dense float32 admin matrix |
|---|---:|---:|
| `totals` | 760 | 4.5 GiB |
| `state` (default) | 3,972 | 23.5 GiB |
| `full` | 31,066 | 183.9 GiB |

Measured peaks for comparison: Build P's ACS local release, on the `state`
surface, recorded a 93.9 GB materialize peak (`build_manifest.json` →
`materialize.peak_rss_gb`); the 2026-09-23 `state` run peaked at 77.9 GB, of
which the checkpoint write was the jump from 55.4 GB; the 2026-09-22 hours
rebuild on `totals` peaked at 75.6 GB. [Target matrix
storage](#target-matrix-storage) replaces this.

## The `state_cd` surface

`state_cd` binds district-level SOI targets. It is the `state` surface plus
the TY2022 SOI congressional-district file (`22incd.csv`, record-set spec
`irs_soi.congressional_district_2022.all_returns.v1`), reconciled so that
every state concept has one vintage. The rule is
`us_acs_local_cd_surface.state_cd_soi_surface`.

### One vintage per state concept

Both files carry state totals for 48 of the district file's measures:
Historic Table 2 (TY2022) and the district file's own `<st>_total` rows.
After aging they disagree by a few percent. `state_cd` keeps **Historic Table
2 as the single vintage of every state concept it carries**, and uses the
district file only for each district's **share** of its state:

```
district target = Historic Table 2 state total
                  x district-file district value / sum of the district file's districts in that state
```

Why Historic Table 2:

- **It is the level `state` already calibrates to** (Build O, Build P and the
  2026-09-23 run), so a `state_cd` build is comparable to them state by state.
- **The district file's state levels are off in ways the shares are not.** The
  feed labels the file "Congressional District Data 2022" but stamps it tax
  year 2023 (PR #1040, #1030), so it is aged one year less than Historic Table
  2; its taxable-interest rows are not rebased to Table 4.3 (Historic Table
  2's are; the rebase factor here is about 2.47). A within-state share is
  unaffected by a level factor common to the state.
- **Two vintages of one concept are contradictory constraints**; the solve can
  only split the difference.

The rebase factors on the pinned feed, by measure across the 43 states with
district rows, are mostly 1.00 to 1.10.

- **Counts cluster near 1.017** (median `return_count` factor), from 1.00 to
  1.03. Counts are never aged in either file (`aging_factor` 1,
  `not_dollar_amount`), so this is purely a level difference between the two
  publications: the district file's state totals count about 1.7% fewer
  returns than Historic Table 2 (California: 18,242,570 against 18,487,690).
- **Amounts cluster near 1.05.** That is the same level difference times the
  aging gap: the district file is aged 2023→2024 by one CBO growth factor
  (about 1.087 for AGI), while Historic Table 2 is aged 2022→2024 on the
  chained SOI and CBO series (about 1.120). The smallest amount factor,
  1.03, is that aging ratio alone. Correcting the district file's stamp
  (#1030) would remove the aging part only.
- **Two factors reflect known level differences:** taxable interest (2.35 to
  2.70, the Table 4.3 rebase) and capital-gains amounts (0.77 to 1.00).
- **The largest single-state factors** were on qualified and ordinary
  dividends (Hawaii, 2.68 and 2.17), tax-exempt interest (Utah, 1.43) and
  rental income (New York, 0.72), where the two tables disagree about one
  state's level. The factor band below drops those blocks.

`materialize_rss.json` records the full table under
`soi_surface.rebase_factor_by_measure`.

### The factor band

A factor mixes a part common to every state (the coverage and aging gap, or
a measure-wide rebase such as taxable interest's Table 4.3 factor) with a
part specific to one state. The median across states absorbs the common
part. A state whose factor sits more than 1.25x from its concept's median
(`STATE_CD_FACTOR_BAND`, either direction) is one where the two publications
disagree about that state, so neither the district file's shares nor a
bridge built on that sibling is trusted there:

- a rebase block out of band loses its district rows
  (`rebase_out_of_band:<measure>`); its Historic Table 2 state row stays;
- a bridge is distrusted if either verdict on its sibling fails: the
  sibling's two state levels (Historic Table 2 over the district file's state
  row, against the median over every state carrying both), or, where the
  sibling has district rows, the rebase of its own district block (against
  the median over the states with district blocks). The two medians can
  differ, because at-large states count only in the first. A distrusted
  bridge loses the bridged state row and its district rows
  (`level_bridge_out_of_band:<measure>`).

Every failing verdict is recorded in the receipt under
`factor_band.out_of_band` with the verdict, its ratio, the median and the
relative gap. The band only judges finite positive ratios: a sibling whose
two levels differ in sign is a data defect and is refused outright. So is a
district file that disagrees with itself: wherever the file carries a state
row beside its district rows, the two must agree to 1e-6
(`STATE_CD_INTERNAL_RTOL`), since no median absorbs a column or crosswalk
defect inside the file. On the pinned feed all 2,193 blocks agree to float
precision (largest relative gap 4e-16; the receipt records it under
`internal_consistency`, and the feed-gated test pins it).

On the pinned feed the band drops eight:

| State | Measure | Basis | Ratio | Median | Relative |
|---|---|---|---:|---:|---:|
| Hawaii | `qualified_dividends_amount` | rebase | 2.68 | 1.13 | 2.37 |
| Hawaii | `ordinary_dividends_amount` | rebase | 2.17 | 1.07 | 2.03 |
| New York | `rental_royalty_income_amount` | rebase | 0.72 | 1.05 | 0.68 |
| Utah | `tax_exempt_interest_amount` | rebase | 1.43 | 1.09 | 1.32 |
| Wyoming | `charitable_amount`, `interest_paid_deduction_amount` | bridge (itemized) | 1.75 | 1.06 | 1.65 |
| South Dakota | `charitable_amount`, `interest_paid_deduction_amount` | bridge (itemized) | 1.44 | 1.06 | 1.36 |

That removes 34 district rows (Hawaii 2 x 2, New York 26, Utah 4) and four
bridged state rows (Wyoming and South Dakota are at-large, so they had no
district rows). The largest gaps the band keeps are 1.22x (Wisconsin
partnership and S-corporation income, West Virginia capital gains) and 1.20x
(Mississippi capital gains); every other kept block is within 1.17x. So on
this feed any tolerance above the largest kept gap (1.220) and below Utah's
(1.316) drops the same eight. The band
is a reviewed tolerance, not an estimate; changing it moves the contract
counts below, which a feed-gated test pins.

The six measures only the district file has (`charitable_*`,
`interest_paid_deduction_*`, `qualified_business_income_deduction_*`) have no
Historic Table 2 parent. Their parent is the district file's own state row,
**lifted onto the Historic Table 2 basis** by a sibling concept both files
carry in the same state (`STATE_CD_LEVEL_BRIDGES`):

- counts use `return_count`;
- charitable and interest-paid amounts use `itemized_deductions_amount`;
- QBI deduction amounts use `adjusted_gross_income`.

The bridge assumes the measure shares its sibling's Historic Table 2 /
district-file ratio in that state: the coverage and aging gap above, plus
whatever the two publications disagree about for the sibling. It is an
estimate, not an identity, which is why the factor band applies to it too.
Without it these state rows and their 2,562 district rows would sit 2–6%
below every related concept. On the pinned feed the bridge factors are
1.00–1.03 for counts (median 1.016), 1.03–1.75 for the itemized-deduction
amounts (median 1.062; the two above 1.25x the median are dropped) and
1.03–1.08 for QBI amounts (median 1.049). The bridge is stamp-invariant: if #1030 corrects the
stamp, the sibling ratio shrinks with it. Its district rows keep their
within-state shares, as every other district row does.

Every district row records its parent (`state_cd_parent_target_name`), the
parent's basis, the district file's own value and the rebase factor; a
bridged state row records its sibling and bridge factor.

### What stays off the surface

| Rows | Specs on the pinned feed | Why |
|---|---:|---|
| District-file `limited_state_local_taxes_*`, both geographies | 974 | Column A18425/N18425 is state and local income taxes, not the limited SALT deduction (microcosm#1038) |
| District-file `premium_tax_credit_returns`, both geographies | 487 | Column N85530 is the additional Medicare tax, not the premium tax credit (microcosm#1038) |
| District `tax_filer_individual_count` | 436 | No state parent in either vintage, so it could not nest in a bound state target |
| District rows of states with one district on the 117th plan (AK, DE, DC, MT, ND, SD, VT, WY) | 459 | The SOI file has no sub-state rows there. These rows are the state total copied, or for Montana split by population, so they carry no district information |
| Historic Table 2 rows copied to at-large districts | 360 | The same copies from the other vintage |
| District-file state rows for concepts Historic Table 2 carries | 2,295 | Second vintage of a state concept |
| District rows of the four out-of-band rebase blocks | 34 | The factor band (above) |
| Bridged state rows whose sibling ratio is out of band | 4 | The factor band (above) |

PR #1040 (awaiting a ruling) excludes the same SALT and PTC columns in the
compiler and rescales the district file's capital-gains rows by one national
factor. `state_cd` already takes district capital gains as within-state
shares of Historic Table 2, which a national factor does not move, so their
level changes only when Historic Table 2's own capital-gains rows do
(#1036).

North Carolina's district rows are bound. Before #1043 the packaged
117th→119th crosswalk was built from NC's 2016 plan rather than the 2019 plan
the 117th Congress used, and 37.0% of NC's population mapped to a different
119th district than the block plan registry (#1041) puts it in. #1043 rebuilt
the crosswalk from the registry. `STATE_CD_EXCLUDED_CD_STATES` is empty, and a
test pins the crosswalk digest it was reviewed against, so a later crosswalk
change forces the same review.

### Counts on the pinned feed

| Family | `state_cd` |
|---|---:|
| `usda_snap` | 102 |
| `cms_medicaid` (enrollment) | 51 |
| `irs_soi` state, Historic Table 2 | 3,819 |
| `irs_soi` state, district file (district-file-only measures) | 302 |
| `irs_soi` district (427 districts x 51 measures, less 34 banded) | 21,743 |
| **Admin specs** | **26,017** |

Of the 21,743 district rows, 19,181 are rebased to a Historic Table 2 parent
and 2,562 keep a district-file parent. The 2,189 (state, concept) district
blocks each sum to their parent within 1e-9. Adding the 487 population
marginals gives 26,504 targets, before the holdout. The feed-gated test
`test_pinned_feed_state_cd_surface_matches_its_contract` pins these counts
and the reconciliation.

### District plans

The SOI district file is tabulated on the 117th-Congress plan and the
households carry 119th-plan districts (drawn within their PUMA from the
ladder). The compiler maps the 117th rows onto the 119th plan with the
packaged 2020-block population crosswalk, which #1043 builds from the block
plan registry (#1041); `state_cd` uses that mapping as is and records the
crosswalk's sha256. The mapping assumes returns spread with population inside
each 117th/119th intersection. A household column on the 117th plan
(`congressional_district_geoid__117th_congress`, written by the location v1
block draw once the registry is attached) would let these targets bind as
exact block sums with no crosswalk; the district rows are materialized
against one named household column, so that change is a parameter, not a
rewrite.

### Sigma

No fact in the pinned feed carries an uncertainty field, and IRS SOI tables
are administrative. `target_roles.json` records `sigma` for every target (`null`,
`sigma_basis: "not_provided_by_feed"`), so a feed that supplies standard
errors surfaces them. The calibration loss is unchanged: fixed-scale capped
relative error.

## CD holdout and the pro-rata baseline

`state_cd` holds a hash-assigned subset of its district targets out of
calibration and scores it afterwards (`--cd-holdout-fraction`, default 0.1 in
`state_cd`, 0 elsewhere).

**The held unit is a (state, SOI concept family) block** of district targets,
not a single district or target. With a state total and its sibling districts
trained, one held district is pinned by adding up, so it would score
perfectly for free. A concept family also groups measures that add up to each
other: every EITC measure is one family, because the per-child rows sum to
the EITC total. The state totals stay trained; the question a held block
answers is how well the calibrated file splits a known state total across
districts.

**Assignment** is `microcosm.build.holdout.hash_holdout_unit`: SHA-256 of a
salt and the unit key `"<state_fips>|<family>"`, read as a uniform on
[0, 1), held iff below the fraction. It is deterministic, does not depend on
which other units exist or their order, and is nested in the fraction. Held
targets are never built into the calibrator's `TargetSet`
(`calibration_target_set`); a property test spies on the solve to check it.

**The baseline** allocates each held target's state parent by population:
`parent x district population / state population`, both from the PUMA
ladder's 119th-plan district overlap populations, so a held block's baseline
sums to its parent exactly. `calibration_summary.json` → `cd_holdout`
scores the held targets under the design weights, the calibrated weights and
the baseline (mean, median and p90 absolute relative error, share within 10%,
the capped loss the solve minimizes, per family, and the share of targets
where the calibration beats the baseline). It is report-only.

## Effective sample size and weight by origin

`calibration_summary.json` → `weight_origin` records, at the design and at
the calibrated weights:

- Kish ESS over rows: nationally, per spine, per state and per district
  (each with its distribution);
- Kish ESS over **distinct households**: rows summed by (`household_spine`,
  `household_source_id`). ACS source ids are pre-offset and collide with donor
  ids, so the spine is part of the key; a donor household's native row and its
  PUF-detail clone count once;
- the share of total weight on the heaviest 1% of records;
- household-weight share by spine.

The weight cap and `l2_lambda` are unchanged; the concentration they allow is
a known issue (`low_effective_sample_size_lambda_zero`) under review. This
reports it rather than tuning it.

## Development rungs

`--sample-fraction` draws a development rung, one of the stacked pool's
f001, f004, f010 or f025 (`tools/build_us_multispine_pool.py`; DESIGN.md
"Production US stacked spine" names f001, f010 and f100).

- **The draw.** It samples whole households with
  `microcosm.build.frame_sampling.sample_frame_households`, taking
  `floor(fraction x n_h)` in every (spine, district) stratum `h`.
- **The weights.** Each drawn household's weight is scaled by its stratum's
  inverse sampling rate `n_h / k_h`, so a drawn stratum keeps its household
  mass in expectation (exactly, when a stratum's weights are equal). Each
  spine is then scaled to its full household mass.
- **Strata that draw nothing.** A stratum that floors to zero draws loses its
  households, and its weight is spread over the rest of its spine. The
  receipt records how many strata that is and their weight share. At f001
  this is material for the donor spine, whose district strata are small, so
  read district-level evidence from f010 or above.
- **Refusals.** `run_identity.json` records the rung, seed and selected-id
  digest. The calibrate stage re-draws the same households before attaching
  weights and refuses a mismatch, and `--stage package` refuses any rung but
  f100.
- **Memory.** The staging frame is still loaded in full before sampling, so a
  rung lowers the engine pass and the solve, not the load.

## Target matrix storage

The materialize stage writes four files:

- `target_frame_lean.h5`: structure only (household id, geography, spine,
  source id, design weight; person memberships; group ids).
- `target_registry.json`: every target as a `TargetSpec` with its
  calibration hierarchy, held-out targets included. Its spec count and
  digest therefore identify the materialized surface, not the trained set;
  the trained count is `calibration_summary.json`'s `n_targets`, and the
  roles file says which rows trained.
- `target_matrix.npz`: a (targets x households) CSR matrix with float32
  values, row *i* of which is registry spec *i*.
- `target_roles.json`, row-aligned with both: role (train or holdout),
  holdout unit, geography, sigma, and for district rows their state parent
  and pro-rata populations.

No dense households x targets matrix exists at any point. Each engine chunk's
columns go straight into the CSR. District SOI rows are not materialized one
column each: the engine pass materializes one geography-free **carrier**
column per distinct SOI concept, and each district row is its carrier
restricted to the district's households. That equals the direct
materialization, because the SOI slice masks a tax unit by its household's
state and district; the first chunk of every run also materializes one
district row per carrier directly and compares it with the row the assembler
actually stored for it, refusing any difference. That check covers one row
per carrier in one chunk, which could miss the households a concept touches,
so every chunk also checks the stored district rows against a state row
materialized directly in the same pass (`cd_surface.district_row_parents`):

- In `state_cd`, each district row names its state parent, and a block's
  rows partition the parent's state. Laid side by side, they must equal the
  parent's column on every household of the chunk, bit for bit, and every
  block is checked in every chunk. A `state_cd` row without a compiled parent
  is refused, so the check cannot silently skip it.
- In the other modes, a district row is checked against a state row of the
  same materializer semantics in the same state (in `full`, the district
  file's own state total), on the row's own households, since those blocks
  need not cover the state. A row with no such state row is counted as
  unchecked. This check and the carriers share one key
  (`soi_materializer_semantics`), so a field the materializer reads that the
  key omitted would slip past both; the `state_cd` check compares against a
  named parent and does not share that blind spot.

`materialize_rss.json` → `carrier_check` records the blocks and rows checked,
the nonzero households they covered, and the unchecked rows. The calibrate
stage builds
each training target from its registry spec (value, metadata, hierarchy)
with a callable measure that reads its CSR row, so the calibrate kernel's own
`build_constraint_matrix` compiles them one row at a time, unchanged, and the
schema-8 `calibration_diagnostics.json` names each measure by its matrix row
(`target_matrix_row[i]`).

A differential test materializes the fixture both ways: dense float32
columns compiled by the kernel from frame columns, and the carrier-split CSR
through the checkpoint. It requires the identical constraint matrix and the
identical calibrated weights.

## Where the chosen mode is recorded

- `materialize_rss.json` in the checkpoint directory (`soi_mode`), written by
  `--stage materialize`. Later stages never re-select targets; they read this
  file, whatever `--soi-mode` they are given.
- `gate_summary.json`, under `gates.calibration.calibrated_surface.soi_mode`
  (finalize), which `build_manifest.json` also embeds under `gates`.
- `build_manifest.json`, under `materialize.soi_mode` (package).
- The `refresh_recipe.release` command in `build_manifest.json` and
  `release_manifest.json`, which names `--soi-mode <mode>`, so re-running it
  reproduces the recorded surface even if the default changes again.

`--stage package` refuses a checkpoint whose `materialize_rss.json` does not
record one of the four modes, before any release directory exists.

`state_cd` runs also record:

- `run_identity.json`: the sha256 of `target_registry.json`,
  `target_roles.json`, `target_matrix.npz` (with its shape and nnz) and the
  lean H5; the CD holdout (unit, salt, fraction, held units and targets); the
  sampling rung. Calibrate and finalize refuse any of the four files whose
  bytes changed.
- `weights_latest.npz` and `calibration_summary.json` carry the digest of
  that run identity (`run_identity_sha256`) and the solver settings (weight
  cap, loss cap, `l2_lambda`, seed, epoch batch). `--resume` and the
  calibrate stage's "already complete" shortcut refuse weights or a summary
  from another materialization (another staging file, surface, holdout or
  sample) or other settings; finalize and package refuse a summary from
  another materialization.
- `consumer_export.json` (written with the calibrated H5) records the H5's
  sha256, the run-identity digest, the solver settings and a digest of the
  calibrated weights, which `calibration_summary.json` also records;
  `spine_qa.json` records the run-identity digest and the sha256 of the bytes
  it loaded. Before any release directory exists, finalize and package
  refuse an H5 whose bytes are not the export's, an export and summary that
  describe different weights (a recalibration that stopped part way), and QA
  evidence from another materialization or of other bytes. The H5's path is
  not compared; its sha binds the bytes wherever they are reached from.
- A new materialize deletes the previous calibration outputs, consumer
  export, spine QA and gate report. A solve deletes the previous summary,
  diagnostics, consumer export, null-fill manifest and spine QA (keeping the
  resume weights) before it starts; a calibrated H5 or gate report left from
  before fails the sha checks above. The "already complete" shortcut also
  requires the summary to record the saved weights' digest.
- Materialize refuses a district row without a positive ladder district and
  state population, since the pro-rata baseline could not score it.
- `calibration_summary.json`: `cd_holdout`, `weight_origin` (ESS nationally,
  per spine, per state and per district, over distinct households, and the
  top-1% weight share, at the design and the calibrated weights),
  `n_holdout_targets` and the checkpoint matrix's shape and nnz, beside the
  fit summary.
- `materialize_rss.json`: the full SOI surface receipt (`soi_surface`: counts,
  every drop reason, rebase factors, crosswalk digest, sigma), the holdout
  receipt with every held unit, and the carrier check.
- `gate_summary.json`: `cd_holdout` and `weight_origin` (report-only), and
  the calibrated surface's trained and held-out counts.
- The release refresh recipe, which names `--cd-holdout-fraction`.

## Operating notes

- The default command calibrates to `state`. The dense admin matrix is gone,
  so the materialize peak is the staging frame plus one engine chunk; the
  calibrate stage's peak is the artifact write.
- A checkpoint written before the sparse matrix (measures as dense H5
  columns) is refused by `--stage calibrate`; re-run `--stage materialize`.
- To reproduce the 2026-09-22 hours rebuild (#974), pass `--soi-mode totals`.
  To reproduce a build that ran as `full` after `b7922b089`, pass
  `--soi-mode full`.
- A checkpoint materialized under an earlier default records its own mode and
  packages as that mode.
- `--soi-mode` only matters when `soi` is in `--families`. `state_cd` is
  meant to run with `cd` in `--geographies` (the default), so district
  population marginals are bound too; `--cd-holdout-fraction` above 0 is
  refused without `state_cd`.
