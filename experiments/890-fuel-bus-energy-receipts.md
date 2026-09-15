# #890 receipts — bind DfT, HMRC, DESNZ, Ofgem, ORR and NTS facts for bus fares, road fuel, energy and rail

Plan: `repos/uk-890-implementation-plan.md` (María's rulings 2026-09-10/11: OBR cars receipts as
the fuel target; BUS0415 alignment on a calendar-year basis; donors uprated to FY2024-25; bus
incidence modelled take-up style on NTS frequency of use; two PRs, target binding first). Licensed
evidence lives under `data/ukds/acceptance/890-fuel-bus/` (SDC-safe aggregates only); this file
carries the digest-pinned receipts the PR's declarations anchor to.

## PR-T — calibration target binding

### Part A — Chronicle feed re-pin (national surface only)

Feed rebuilt from `PolicyEngine/chronicle` main `c6f9361492056b9fab7b8535a2be77eb2b6c93bb`
(2026-09-11, the merge of #258 on top of #255) with `chronicle build-bundle --suite uk` →
`build-consumer-artifact` (built 2026-09-11 from a worktree at that commit):

- `consumer_facts.jsonl` sha256 `45bda3ae730d4ae3fa059d9e03304e902f7f6e74c5099355ef937625ca03b72b`
- `manifest.json` sha256 `33a3031523e2cea2f0092a547b97065efbe103cdf85f842142e847b921bcdd9d`
- rows / schema: 138,847 / `policyengine_ledger.consumer_artifact.v2`; consumer-fact schema
  digest `72ad3149…` (was `76ac268e…`: chronicle added the `quarter` and `week` period types)
- previous national pin: `ec7169b`, 131,450 rows (`4a50ee95…` / `a95d0ee9…`)

Effect on the committed national surface before any binding change: none. 415 active, 7
deferred, 8 signed out, no resolved value or fact key moved; the compile-parity receipts
regenerate byte-identical. Only the membership feed label moves
(`chronicle-uk-ec7169b5-consumer-facts` → `chronicle-uk-c6f93614-consumer-facts`). As written
in round 1 the local surfaces kept a separate `ec7169b` pin with their own default feed file;
round 2 (María's ruling) unified both surfaces on one pin, `uk/chronicle_feed.json`, and the
feed-gated tests read one artifact (`.codex-work/consumer_facts_uk.jsonl`, or the
`.codex-work/uk-artifact/` directory for the runtime compile), skipping when absent; Part I
records the later move of that single pin to `474a0ae`.

### Part B — vendored Chronicle facts (`tools/vendor_uk_ledger_facts.py`)

Nine per-concern resources under `uk/`, each carrying the feed identity above and
regenerating byte-identical from it under test; selection counts are asserted by the register
`uk/ledger_fact_vendor_selections.json`:

- `dft_bus_value_anchors.json` — 81 rows (BUS05ai 9, BUS05bi 9, BUS0415 39, NTS0705a 24); sha256 `2a92a656…`
- `road_fuel_anchors.json` — 69 rows (HMRC 12, DESNZ annual 6, OBR by vehicle 49, ONS population 2); `8dc10402…`
- `licensed_cars_fuel_type.json` — 36 rows; `83750dff…`
- `need_energy_facts.json` — 228 rows (E&W 144, Scotland 84); `5453d7fd…`
- `ofgem_price_cap_facts.json` — 268 rows (cap levels 256, benchmark consumption 12); `b668e351…`
- `nts_bus_use_frequency.json` — 54 rows (NTS0313 27, NTS0621 27); `2738c4f5…`
- `devolved_bus_finance.json` — 86 rows (Scotland 18, Wales 36, NITHC 8, DfI 24); `6bd24101…`
- `orr_rail_facts.json` — 99 rows; `02c8870b…`
- `ons_household_expenditure_facts.json` — 18 rows; `7c13e88a…`

### Part C — region leg and the London bus rows

- `_geography_pins` pins a region-only contract target at region level; the compiler refuses a
  region pin outside the nine English region codes (`UK_NATIONAL_REGION_ROSTER`) and a
  region-pinned reference that resolves a non-region fact.
- `dft.bus_fare_receipts.london` → fact FY label 2025 (comparable 2024), GBP 1,347,434,943.01;
  `dft.bus_net_support.london` → GBP 1,130,214,000; both region `E12000007`.
- England outside London: signed exclusion `derived_partition_member` (the partition
  reconciles within GBP 1,000 in `test_uk_bus_targets.py`). UK fares row: `no_publisher_uk_total`
  (Wales fare revenue unpublished). UK support row: `covered_by_country_legs`.

### Part D — BUS0415 alignment (fares only)

Factor = mean of the four quarter-end index values inside the calibration calendar year over
the mean of the four inside the receipts' April-to-March fiscal year, per series:

- England `E92000001`: from (2024-06, 2024-09, 2024-12, 2025-03) mean 193.125; to (2025-03,
  2025-06, 2025-09, 2025-12) mean 204.125; factor 1.05695792880259; value GBP 3,417,388,656.44
  → 3,612,036,036.22.
- London `E12000007`: 196.1 flat in both windows; factor 1.0.
- Net support rows are never aligned (#789 scope item 5).
- Round 2 (Vahid's review of PR #904, should-fix 1 and 2): the alignment is a declaration on the
  reference, not a runtime post-pass keyed on a family and concept string. The four
  `dft.bus_fare_receipts.*` contract rows declare `uprating_index: dft.local_bus_fares_index`;
  the generator carries it onto the reference, applies the UK applier while authoring, and the
  membership records the applied value with `index`, `factor` and `value_before_uprating` on
  the hold (England 3,417,388,656.44 → 3,612,036,036.22; London ×1). The runtime applies the
  same applier keyed on the declaration, so membership, compile-parity receipts and the run
  manifest carry one value; an index no UK applier implements refuses the compile. The BUS0415
  series is read from the vendored `dft_bus_value_anchors.json` (refused if its feed identity
  differs from `uk/chronicle_feed.json`), so the production factors reproduce in CI without the
  licensed feed (`test_bus0415_alignment_from_the_vendored_series`), and the vendor register's
  `consumers` field is now a statement of fact (that resource has one present reader; every
  other resource lists its readers under `planned_consumers` until PR-S).
- Production-2023 parity receipt: the same alignment at target period 2023 moves England fares
  3,278,976,692 → 3,258,254,371 (factor 0.99368) and gives the new London row 0.99769 of the
  FY2023-24 fact; both factors are below one because the fare cap held index growth inside
  calendar 2023 under the fiscal-year mean.

### Part E — new national targets (pinned facts, registry scope, unmapped declarations)

- `obr.fuel_duties_cars`: OBR receipts by vehicle category, cars, on `fuel_duty`. Round 1 bound
  the FY2025-26 figure (GBP 15,900,000,000, label 2025); María's ruling of 2026-09-11 is that a
  fiscal-year source binds FY2024-25 (the FRS 2024-25 base year) or CY2025, never a later
  fiscal year, so round 2 pins `period_value: 2024` and binds GBP 14,400,000,000 held to 2025
  (the same hold shape as the HMRC CGT rows). Every row of the vehicle-type release is the
  April 2024 vintage (`april_2024`), so the FY2024-25 figure is that vintage's projection for
  the base year; the OBR publishes no outturn by vehicle type. Bound on the vintage: its
  all-vehicle FY2024-25 total is GBP 24.7bn against the March 2026 EFO outturn of GBP 24.359bn
  (`obr.fuel_duties`, observation), a 1.4% overshoot. Note for the ruling: the rest of the
  `obr` family (income tax, NI, VAT, benefits) still binds the FY2025-26 projection of the
  latest EFO at period 2025 by the family's `allow_source_projection` policy inherited from
  uk-data; that convention is untouched here and is flagged in the PR body for María.
  `obr.fuel_duties` (all road users) signed out; its #757 measure exclusion retired (register
  51 → 50 entries).
- `orr.rail_government_support`: ORR 7270, FY2024 GBP 21,621,100,549 held to 2025, Great
  Britain (`region != NORTHERN_IRELAND`), on `rail_subsidy_spending`.
- `ons.household_energy_expenditure`: ONS COICOP 04.5 CY2025 GBP 43,427,000,000 on
  `domestic_energy_consumption`.
- `scotgov.bus.passenger_revenue` GBP 391,000,000; `scotgov.bus.government_support`
  GBP 499,000,000 (FY2024, region SCOTLAND).
- `welshgov.bus.public_support` GBP 131,489,970 (FY2024, concessionary fares GBP 62,443,460 +
  support to operators GBP 69,046,510 gross, region WALES).
- `dfi_ni.bus.passenger_receipts` GBP 150,082,817.49 (FY2024, Ulsterbus + Metro/Glider,
  region NORTHERN_IRELAND).
- `nithc.public_transport_support` GBP 111,700,000 (FY2024, PSO 61.8m + concessionary 49.9m,
  bus plus rail) on `bus_subsidy_spending + rail_subsidy_spending`, region NORTHERN_IRELAND.

Surface after Part E: 244 contract targets (registry scope 210, profile scope 34), 424 active
references, 7 deferred, 7 signed out; national runtime compile 424 / 0 unsupported.

### Part F — baseline national calibration on the current spine (round 1, FY2025-26 fuel)

Runs on the newest gated national spine (`spine-p`, `data/ukds/acceptance/spine-p-355/spine-p.h5`,
sha `ae83e307…`, FRS 2024-25, stamped 2024), 1,500 epochs, `family_equal`, code `3a135c5f`:

- `pr-t-baseline-spine-p/` (`uk-frs-calibration-attempt-20260911T113749Z-0c7e79e2`): the terminal
  battery blocked on one row, `obr.fuel_duties_cars@2025` at −25.6%; every other new row inside the
  bound. Loss 0.01154, 96.0% within 10%, ESS 10,461 (0.198), max/median 100, 374 targets solved
  (424 compiled − 50 measure exclusions).
- `pr-t-baseline-spine-p-2/` (`…20260911T114304Z-61b7818a`): identical solve with the dated
  reviewed exclusion for that row in force (register entry approved 2026-09-11, expires
  2026-10-11, tracking the PR-S landing); battery passes, staging H5 written (not a release
  candidate).

Pre- and post-calibration fit of the new rows (initial → final relative error):

- bus fares England −46.3% → −0.0%; London −72.4% → +0.1%; support England −28.8% → +0.1%,
  London −72.5% → +0.1%
- `obr.fuel_duties_cars` −46.6% → −25.6% (excluded from the fence, still bound)
- `orr.rail_government_support` −71.8% → +0.2%
- `ons.household_energy_expenditure` +0.2% → +0.0%
- Scotland fares −63.0% → +0.0%; Scotland support −54.5% → +0.1%; Wales support −9.9% → +0.2%;
  NI fares −65.3% → −0.0%; NI public transport support +65.3% → −0.2%

How the solver closed them, measured on the calibrated staging H5 against spine-p design
weights (the receipt PR-S is built to change): the closures are weight stretch on the households
carrying the imputed column, not level. Share of the calibrated mass sitting on households whose
weight rose more than 3× (more than 5× in brackets): London fares 83% (73%), London support 85%
(77%), England fares 60% (48%), Scotland fares 64% (33%), Scotland support 69% (57%), NI fares
78% (3%), rail support GB 83% (71%), motor fuel 51%. Rail is the largest single stretch: the
frame carries GBP 6.1bn of `rail_subsidy_spending` at design weights against the ORR 7270 broad
total of GBP 21.6bn, so the 14% of households with rail use are stretched ×3.6 on average; the
7271 all-sources operational figure for FY2024 is recorded below for the open 7270-versus-7271
ruling. Northern Ireland public transport support runs the other way (frame GBP 185m at design
weights against GBP 112m published), because the ETB donor is a GB survey. Domestic energy
already sits on ONS 04.5 at design weights (GBP 43.5bn against 43.4bn), so that row costs
nothing; PR-S repricing to FY2024-25 cap rates with standing charges will move the frame off it
and is the point at which the NEED-versus-ONS level question is decided.

### Part G — round-2 baseline on the same spine (FY2024-25 fuel, declared alignment)

Same spine-p, epochs and weight rule as Part F; the surface differs only by the round-2 changes
(cars fuel duty at FY2024-25 GBP 14.4bn instead of the FY2025-26 GBP 15.9bn projection; the
BUS0415 alignment declared on the references, same values). Runs under
`data/ukds/acceptance/890-fuel-bus/`:

- `pr-t-round2-spine-p/` (`uk-frs-calibration-attempt-20260911T140557Z-abb86ff5`): the solve
  reaches every new row inside the 25% bound, and the terminal battery then blocked on the
  register itself: "stale reviewed target-fit exclusions are back inside the bound:
  `obr.fuel_duties_cars@2025`". The dated exclusion of Part F was therefore retired in this PR
  (register back to empty, as on main) rather than carried to PR-S.
- `pr-t-round2b-spine-p/` (`uk-frs-calibration-attempt-20260911T141216Z-504ea8bc`): identical solve with the register empty;
  battery passes with no failed gate (staging H5 written; not a release candidate, as in Part F).

Fit (initial → final relative error), loss 0.01123, 95.45% within 10%, ESS 10,528 (Part F:
0.01154, 96.0%, 10,461), 374 targets solved, 0 rows outside 25%:

- bus fares England −46.3% → +0.2%; London −72.4% → +0.2%; support England −28.8% → −0.0%,
  London −72.5% → +0.1%
- `obr.fuel_duties_cars` −41.0% → −16.7% (Part F against 15.9bn: −46.6% → −25.6%); the
  frame's calibrated cars fuel duty is GBP 12.0bn either way (12,002m here, 11,830m in Part F),
  so the change is the target, not the frame; the remaining −16.7% is the diary under-capture
  and has-fuel zeroing PR-S removes (A, U)
- `orr.rail_government_support` −71.8% → +0.0%
- `ons.household_energy_expenditure` +0.2% → +0.0%
- Scotland fares −63.0% → −0.2%; Scotland support −54.5% → −0.0%; Wales support −9.9% →
  +0.4%; NI fares −65.3% → +0.1%; NI public transport support +65.3% → +0.0%

The weight-stretch anatomy of Part F is unchanged in kind (the same rows close by stretch, the
rail row by ×3.6) and is not re-measured here; PR-S is what changes it.

### Part H — rebase onto main `3094bfe8` (2026-09-11, after #903, #894, #855, #902)

Main merged the calibration target hierarchy (schema 8, microcosm#855) between round 2 and this
rebase. Two consequences for this PR:

- Every contract target now needs a `category_id` from the contract's normalized catalog and a
  Microcosm-owned `label`. The eight #890 targets declare them (four new providers: ORR, Welsh
  Government, DfI Northern Ireland, NITHC; six new categories: `orr.rail_industry_finance`,
  `ons.consumer_trends`, `scotgov.local_bus_finances`, `welshgov.local_bus_finances`,
  `dfi_ni.local_bus_finances`, `nithc.public_transport_finances`; the cars fuel duty reuses
  `obr.efo_receipts`). The committed `target_references.json` carries the hierarchy lookups
  the generator emits for the 196 contract targets with an active reference (providers and
  categories copied from the contract, `target_categories` and `target_labels` keyed by
  contract target id), which is contract-derived and does not depend on the facts.
- The generator and the runtime compile cannot run against any Chronicle artifact until
  chronicle#261 lands: main's hierarchy completion requires `dimension_labels`,
  `dimension_value_labels` or `layout.groupby_dimension_label` on every selected fact, and no
  consumer artifact carries them (checked on both `c6f9361` and `ec7169b`: 0 of 138,847 and 0 of
  131,450 rows). On this tree `compile_uk_target_registry` returns 0 compiled / 424 unsupported
  against the pinned artifact, exactly as on main. So the national and local reference
  surfaces, the membership files and the parity receipts in this PR are the round-2
  regenerations plus the contract-derived hierarchy lookups; the byte-identical regeneration
  tests skip in CI (no feed) and are blocked locally by #261 for main as well. María's issue
  names the interim Microcosm fallback as a separate change; this PR does not add one.
- The hierarchy completion labels a fact's geography from `geography.name` or, failing that,
  from Microcosm's authoritative catalog (`microcosm.calibrate.geography_constants.
  UK_GEOGRAPHY_ID_TO_LABEL`). The consumer artifact carries no name on the London region fact
  (`E12000007`: id, level, vintage only) and the catalog held only the UK, GB and country codes,
  so the region leg would be refused at compile once #261 lands; the nine English regions (ONS
  statistical regions, E12 codes) are added to the catalog, with a test that the national region
  roster is covered. `test_bus0415_alignment_from_the_vendored_series` exercises that path
  feed-free (a synthetic fact with a label and no geography name, the London row pinned at
  region).
- Conflict resolutions: `test_uk_measure_simulation` counts to the merged truth (47 entries,
  2026-08-26 tranche 35: #903 retired the three ONS composition entries, #890 the
  `obr.fuel_duties` entry); the UK spec digest re-pinned at the end; `test_uk_terminal_gates`
  and the target-fit register are main's (empty register).

### Part I — re-pin to chronicle `474a0ae` (2026-09-14, chronicle #263 labels; rebase onto main `15ebde80`)

María merged chronicle #263 (closing #261): consumer facts now carry `dimension_labels` and
`dimension_value_labels` and are stamped `chronicle.consumer_fact.v2` (an id Microcosm already
accepted); the artifact keys stay in the ledger epoch. The UK consumer artifact was rebuilt at
`474a0ae` (141,400 rows: the c6f9361 rows plus chronicle #259/#260's nine Universal Credit
packages; 141,400 of 141,400 rows carry `dimension_labels`) and pinned in `uk/chronicle_feed.json`
(facts `bb12d77a…`, manifest `c649b7fe…`, consumer-fact schema `6a42e4a5…`), with a copy under
`data/ukds/acceptance/chronicle-uk-artifact-474a0ae/`.

- National surface: the generator runs again on main's hierarchy code and regenerates the
  references and membership byte-identical to the round-2 surface apart from the feed label
  (424 active, 7 deferred, 7 signed out; 438 candidates, 0 value or period moves), and its
  hierarchy lookups equal the hand-completed ones of Part H for all 196 targets. The nine
  vendored resources regenerate from the new pin (rows unchanged, headers move).
- Local surface: every local-level fact (constituency, local authority) is identical between
  the two artifacts (109,693 of 109,693 rows by key, value, period and concept), so the local
  references, membership and parity receipts hold their values and only the membership's feed
  label moves. The local generator itself cannot run yet: main's hierarchy completion labels a
  fact's geography from `geography.name` or Microcosm's geography catalog, Chronicle emits no
  geography name, and the catalog carries the UK, GB, country and (since Part H) region codes
  only, so every constituency and local-authority reference is refused
  (`hmrc.self_employment_income.amount@E14001063`: "no display label and is not present in
  Microcosm's authoritative geography catalog"). The same refusal applies to the local runtime
  compile on main. Per `docs/calibration-target-hierarchy.md` geography labels are Chronicle's
  (`geography.name`), with the catalog as the fallback for shared codes, so the fix is filed as
  chronicle#266 (emit `geography.name` from the publisher's area label, which the same fact
  already carries in `layout.groupby_value_label`) and tracked on this side as microcosm#920;
  not part of this PR.
- Runtime compile on the pinned feed: `compile_uk_target_registry` compiles 424 / 0 unsupported
  again (Part H had 0 / 424).

### Part J — round-3 baseline on spine-r (474a0ae feed, main `15ebde80` surface)

spine-p cannot host the post-#903 surface (its ten ONS household-composition rows bind on
`ons_household_type`, the `frs_relationships` stage column spine-p predates; the spine-p attempt
`pr-t-round3-spine-p/` stopped at measure resolution), so round 3 runs on #903's twin spine-r
(`data/ukds/acceptance/791-relationships/spine-r/spine-r.h5`, sha `921612e8…`), 1,500 epochs,
`family_equal`, 377 targets solved (424 compiled − 47 measure exclusions):

- `pr-t-round3-spine-r/` (`uk-frs-calibration-attempt-20260914T115335Z-ec8910d9`): loss 0.01146,
  95.76% within 10%, ESS 10,342. Every #890 row inside the 25% bound: bus fares England −46.6% →
  −0.1%, London −72.3% → +0.1%; support England −28.9% → −0.1%, London −72.7% → +0.0%;
  `obr.fuel_duties_cars` −41.1% → −15.6% (frame GBP 12.15bn); rail support −71.9% → −0.0%;
  ONS 04.5 +0.2% → −0.1%; Scotland fares −63.0% → −0.2%, support −54.4% → +0.1%; Wales support
  −10.4% → +0.1%; NI fares −65.3% → −0.2%; NI public transport support +66.5% → −0.6%. The ten
  #903 composition cells sit within ±4.6%.
- The terminal battery blocked on one row outside #890's scope:
  `hmrc/self_employment_income_income_band_20_000_to_30_000@2025` at +25.8%. On #903's own
  spine-r calibration (`791-relationships/calibration-r/`, 367 targets, loss 0.0114) the same
  cell was +24.1% and the 30–40k cell +22.4%, i.e. already at the fence; with the ten #890 rows
  added the 20–30k cell crosses it and the 30–40k cell eases to +16.6%. This is a spine-r SPI
  band-cell posture for María to rule on (dated reviewed exclusion under her name, or a lever),
  not a defect of the #890 binding; no entry is added here.

ORR FY2024 for the rail ruling: table 7270 total government support GBP 21.62bn (bound); table 7271 all sources GBP 11.86bn, of which Department for Transport GBP 9.47bn.

## PR-S — spine changes and remeasurement (branch `uk-890-spine`, from PR-T `e80553b9`)

### Part K — declarations and unit-level receipts (commits C0–C8, 2026-09-14)

Everything a stage now reads from a publisher reaches it through the nine vendored resources
(never the feed), through one fail-closed reader `ledger_fact_vendoring.vendored_rows` (pin
check against `uk/chronicle_feed.json`, fiscal-start and dimension selection); the register
lists each consuming module and the `test_register_consumers_are_modules_that_read_the_resource_today`
check holds. Resource digests after the consumer re-registration (rows unchanged from Part B):
`dft_bus_value_anchors` `dfcb3071…`, `road_fuel_anchors` `fefffb8f…`, `licensed_cars_fuel_type`
`f85d067d…`, `need_energy_facts` `aca144e3…`, `ofgem_price_cap_facts` `acdccb4e…`,
`nts_bus_use_frequency` `8c24bf7c…`, `devolved_bus_finance` `133d3aec…`, `orr_rail_facts`
`a75d1037…`, `ons_household_expenditure_facts` `81b90e70…`; the new hand-authored
`ofgem_region_crosswalk.json` is `2c9e70df…`.

- C0 (Vahid's #904 carry-overs): `test_vendored_fares_index_refuses_a_foreign_feed` (a vendored
  payload whose `source_fact_feed` differs from the pin raises "different Chronicle feed");
  Part A's retired local-feed default corrected.
- C1 (microcosm-fit): `FittedRegimeGatedQRF.predict_positive_from_uniforms` and
  `QRFChainStepResult.fitted`; `fit.qrf` parity pins re-pinned with `direct.csv` byte-identical;
  seed digests and every country's `spec_sha256` moved once.
- C2/C3 (U, rail): `uprate_donor_columns` on `lcfs_consumption` (2023→2024) and `etb_services`
  (FYE2024→FY2024-25). Resolved factors: CPI 1.02500 (engine `consumer_price_index`
  1.51964→1.55763) on the twelve COICOP columns and bus fares; earnings 1.05100 on
  `employment_income`, `hbai_household_net_income`, `household_gross_income`; mixed income
  1.02730; private pensions 1.05000; petrol 0.98455 (DESNZ ULSP 147.75→141.48 p/l ×
  HMRC litres 17.28→18.03 bn ÷ ONS population 68.27→69.28 m); diesel 0.89821 (ULSD ratio ×
  29.10→28.29 bn litres ÷ population). Engine litre-proxy audit recorded beside them: petrol
  0.852, diesel 1.032 (the 2023 pump-price parameters; upstream issue draft in the PR). ETB:
  education 1.02500, rail support 0.93513 (ORR 7270 GBP 23.12bn→21.62bn), bus support 1.14073
  (BUS05bi England GBP 2.65bn→3.02bn). `compute_ratio` denominator `rail_fare_index_2024` =
  1.165, lockstepped; engine 2025/2024 = 1.0446 against ORR regulated-standard 1.0451.
- C4 (A, E2): `num_vehicles` is a numeric QRF predictor on both sides (LCFS `a124`, the
  was_wealth draw on the recipient, both clipped to 0–5); the WAS→LCFS has-fuel bridge,
  `was_bridge_donor` and `lcfs_consumption_anchors.json` retired. Fuel-buyer share from VEH1103
  2024 GB: 1 − (1,266,421 + 107) / 32,914,842 = 0.96152 (plug-in hybrids count as fuel buyers).
- C5 (I, D): NTS 2024 local-bus user share (at least once a year) 50.49 % of people, 50.04 % of
  people aged 60+; band trips per year declared as midpoints (208, 78, 36, 18, 6, 1.5, 0); bands
  joined by declared labels because NTS0313 and NTS0621 word two of them differently.
  Fare cells (users only, FY2024-25): London GBP 1,347,434,943 and England outside London
  GBP 2,069,953,713 spread over design-weighted person income quintiles by NTS0705a trips per
  person (London 18.8/12.3/7.4/11.6/15.9, other local bus 47.6/37.5/24.8/18.5/13.0 trips per
  person, lowest→highest); Scotland GBP 391,000,000; Northern Ireland GBP 150,082,817 (Ulsterbus
  plus Metro and Glider); Wales untouched. Support cells (all households): London
  GBP 1,130,214,000; outside London GBP 1,894,690,321; Scotland GBP 499,000,000; Wales
  GBP 131,489,970 (concessionary fares plus support to operators, gross); Northern Ireland
  GBP 111,700,000 jointly on bus plus rail support. Cells whose regions are absent from a frame
  are skipped and receipted (the H2 fixture has four regions); a populated region with nobody in
  scope refuses.
- C6 (F): FY2024-25 cap rates from the four quarterly cap levels, direct debit, single-rate
  electricity, including VAT 0.05 (the GB including-VAT rows reproduce it to 1e-6 on all 16
  rows): GB electricity 24.05 p/kWh + GBP 220.98, gas 6.02 p/kWh + GBP 115.11; London
  electricity 25.25 p + GBP 150.32; Yorkshire 22.99 p + GBP 247.76. NEED margins in kWh: 22
  income cells, 8 tenure, 16 property, 11 region (Scotland's region margin = its all-dwellings
  mean 3,251 / 11,929 kWh); E&W terraced houses take the mid-terrace row, mobile homes the
  all-dwellings mean. `uk_aggregate_admin` anchors: electricity GBP 1,081.88 (mean of the ten
  E&W band means 3,579 kWh at the GB cap), gas GBP 850.12 (12,202 kWh), tolerance 15 %.
- C7: `fuel_litres_audit` evidence (frame litres at DESNZ 2024 prices against HMRC FY2024-25
  litres × OBR cars share 14.4/24.7 = 0.583), diagnostic only.
- C8: H2 parity fixture regenerated (`test_acceptance_h_parity` green); UK `spec_sha256`
  `82001d5c…`; gate-battery policy `3f4ba191…`, manifest `1286757d…`, fingerprint
  `39edfe80…`; certification part digests re-pinned; E6 support bounds regenerated from the
  uprated, priced donors (bus columns dropped; one LCFS bound moved 60,000→70,000);
  release-input coverage manifest regenerated.

Donor tape probe (licensed LCFS 2023-24, committed declaration, SDC-safe shares): 4,202 donor
households, 28.46 m weighted; vehicles 0/1/2/3/4/5 = 21.9/45.2/25.8/5.1/1.4/0.7 %; 78.1 % with
a vehicle; positive diary fuel 54.0 % overall, 67.6 % of vehicle households, 5.3 % of no-vehicle
households; positive bus fares 15.7 % (mean GBP 610 a year among them) against the NTS person
user share of 50.5 %, the gap the incidence draw and the positive-regime fill close; weighted
mean electricity GBP 1,034 and gas GBP 811 after the kWh rake and FY2024-25 pricing, petrol
GBP 622 (39.1 % positive) and diesel GBP 350 (20.4 % positive) after uprating.

First twin (spine-s at `4c590172`, sha `fe8f8aba…`, 337 s, 52,846 / 113,590 / 61,213 rows as
spine-r) exposed one defect before any calibration: Wales bus fares GBP 602m at design weights
against GBP 71m on spine-r, because the incidence override filled every NTS user with a
positive-regime amount (a diary-positive fortnight annualised, mean GBP 610) and no fare cell
levels Wales (receipts unpublished). Fix `a5c71477`: the override is imposed only in the regions
a fare-rake cell covers; Wales keeps the chain's raw draw, and the receipt names the households
outside scope. The same twin's energy reading stands and is a finding, not a defect of the
declaration: domestic energy GBP 52.9bn at design weights (electricity mean GBP 1,011, gas
GBP 799 over 96.5 % gas-positive households) against ONS 04.5 GBP 42.3bn in CY2024 (+25 %)
and GBP 43.4bn in CY2025 (+22 %), before the engine's CPI uprating, where the C6 ruling text
expected 4–5 % under. The old pricing (NEED kWh at Q2-2026 unit rates, no standing charge)
matched ONS by coincidence (GBP 1,470 a household); FY2024-25 cap pricing adds GBP 336 of
standing charges a household and the LCFS-derived gas signal (the `p537` split assigns gas to
almost every diary household) keeps 96.5 % of households gas-connected where the published
share is nearer 85 %. Parts L–N carry the second twin and the calibration; the gas-connection
signal is a ruling for María (LCFS `b490`, or a published connection share), not changed here.

### Part L — spine-s (the PR-S twin of spine-r)

Built with the #791 twin recipe (`build_791.sh`: all stages, no sampling, checkpoints on) from
`uk-890-spine` at `a5c71477` (engine policyengine-uk 2.97.0), 369 s, 8.2 GB peak, into
`data/ukds/acceptance/890-fuel-bus/spine-s/`: `spine-s.h5` sha256
`87c913c030f0e1a1235bcc82606900610e8059d9dc1779ab7c0f74ab6b3eb53c`, 52,846 households /
113,590 persons / 61,213 benefit units (spine-r's counts), frame content identity `7fb0595c…`,
design-weight total 29.25 m households; sidecars `spine-s.build.json` (stage evidence),
`spine-s.nonzero_shares.json`, `spine-s.hmrc_replay.json`, checkpoints and the logbook row.

Stage evidence (SDC-safe aggregates, the pre-clone 16,288 FRS households):

- `donor_uprating`: the Part K factors, applied.
- `has_fuel_consumption`: fuel-buyer share 0.96152 (VEH1103); recipient households with a
  vehicle 76.6 %, flagged 73.8 %; donor with a vehicle 78.1 %, positive diary fuel 54.0 %
  (67.6 % of vehicle households).
- `bus_use_incidence`: person user share 50.5 % (60+ 50.0 %, under-60 50.7 %, the 60+ share
  of the frame 25.2 %), household user share 73.9 %; 15,112 households in the raked regions,
  1,176 (Wales) outside keeping the raw draw; of the 10,810 users in scope 1,532 were drawn
  positive by the chain and 9,278 took the positive-regime draw, 569 non-users drawn positive
  were zeroed; positive share 13.5 % before, 67.0 % after; regime zero-inflated positive.
- `bus_fare_rake` (users only, quintile edges GBP 27,282 / 39,909 / 53,496 / 73,855 of
  unequivalised household net income by design-weighted persons): every cell fits its
  published total exactly; factors London lowest→highest 1.40 / 0.88 / 0.52 / 0.92 / 0.57,
  England outside London 0.47 / 0.40 / 0.29 / 0.18 / 0.09, Scotland 0.46, Northern Ireland
  0.49, none skipped. The pre-rake level is the positive-regime amount (a diary-positive
  fortnight annualised) on 67 % of households, so the rake's job is mostly to bring that
  level down to the receipts; the quintile shares follow NTS0705a trips per person, which puts
  8 % of outside-London receipts in the highest quintile where the QRF alone put the most.
- `energy_pricing` / `energy_rake`: VAT check 16/16 rows at 1.05; 50 iterations over income
  (20 populated cells), tenure (6), accommodation (9), region (11), no zero-current cells;
  gas-connected share 96.5 %.
- `fuel_litres_audit`: frame litres 19.06 bn against HMRC FY2024-25 × the OBR cars share
  (0.583) 27.01 bn, ratio 0.706 (petrol 1.22, diesel 0.38): the uniform cars share overstates
  cars' diesel (HGVs and vans burn most of it), so the audit is directional only.
- `support_clip`: zero clipped rows on every non-exempt column in both stages.
- etb `bus_support_rake`: factors London 3.59, England outside London 1.02, Scotland 2.22,
  Wales 0.96, Northern Ireland joint (bus GBP 83.7m + rail GBP 110.8m → GBP 111.7m) 0.574.

### Part M — twin diff against spine-r and the recipient tape probe

`compare_uk_h5_payload.py` spine-r (`921612e8…`) → spine-s (`87c913c0…`), classified against
`890-fuel-bus/spine-s-payload-expectation.json`: 24 observed differences, all expected, none
unexpected, none expected-but-unobserved — the 19 lcfs household outputs, the 4 etb household
outputs, and `full_rate_vat_expenditure_rate`, which moves because etb_vat conditions on the
engine's `household_net_income` and `HOUSEHOLD_TAX_VARIABLES` includes `fuel_duty`, itself a
function of the re-imputed petrol and diesel spend (donor, predictors and seed unchanged).
Row counts, weights, indexes and every other column are byte-equal.

Design-weight probe on the whole artifact (post-clone, 29.25 m households), spine-r → spine-s:

- Vehicles: 23.6 / 48.0 / 23.4 / 5.0 % of households with 0 / 1 / 2 / 3+ (the was_wealth draw,
  clipped), against the LCFS donor's 21.9 / 45.2 / 25.8 / 7.1 %.
- Fuel: has_fuel 68.8 % → 73.5 %; any positive fuel 39.8 % → 48.2 %; petrol mean GBP 517 →
  609 (donor 622, ratio 0.98 against 0.82 before), diesel GBP 310 → 305 (donor 350, ratio
  0.87 against 0.79); petrol plus diesel GBP 24.2bn → 26.8bn; the OBR cars receipts target is
  measured through the engine's `fuel_duty` in Part N.
- Bus fares: positive share 13.1 % → 69.8 % (74 % in the raked regions, 10 % in Wales);
  GBP 2.20bn → 3.99bn. By area after the CGT clone stage re-weights: London GBP 373m →
  1,312m (fact 1,347m), England outside London 1,555m → 2,065m (2,070m), Scotland 145m →
  379m (391m), Northern Ireland 52m → 143m (150m), Wales 71m → 93m (raw draw). The 1–3 %
  shortfalls against the facts are the clone stage's re-weighting after the lcfs rake.
- Bus support GBP 2.56bn → 3.86bn (London 1,186m, outside 1,991m, Scotland 505m, Wales
  138m); Northern Ireland bus plus rail GBP 112m; rail support GBP 6.19bn → 5.09bn (the ORR
  0.935 uprating and the NI joint cell).
- Energy: electricity mean GBP 834 → 1,011, gas GBP 653 → 799 (96.5 % gas-positive), domestic
  total GBP 43.5bn → 52.9bn — the finding recorded under Part K: +22 % against ONS 04.5
  CY2024 before the engine's CPI uprating, from standing charges and the gas-connection
  signal; the `uk_aggregate_admin` anchors themselves (GBP 1,082 / 850 over carriers) are met.

### Part O — rowwise dry run (bridge receipt)

`build_uk_rowwise_candidate.py --dry-run --sample-fraction 0.01 --n-clones 4 --seed 42` on
spine-s with the `474a0ae` artifact and the pinned OA ladder (`bed3f13d…`) refuses at the local
compile: "20430 local target references failed to compile" (106 s, exit 1). This is the
chronicle#266 / microcosm#920 block recorded in Part I (Chronicle emits no `geography.name`
for constituencies and local authorities, so the local surface cannot be authored or compiled
on the pinned feed), the same on main; no rowwise receipt is possible until the feed is
re-pinned after chronicle#266 lands. The national bridge is unaffected (Part N).

### Part N — national calibration on spine-s (PR-S round 1, #904 surface, `474a0ae` feed)

`pr-s-round1-spine-s/` (`uk-frs-calibration-attempt-20260914T193653Z-18e5230f`, code `a5c71477`,
`calibrate_uk_national_dataset.py` 1,500 epochs, `family_equal`, 377 targets solved, 47 measure
exclusions, 203 s): loss 0.01122 (round 3 on spine-r: 0.01146), 95.76 % within 10 % (same),
ESS 9,073 (10,342), max weight ratio 10.0 (10.0), top-1 % weight share 16.3 % (15.5 %). Every
#890 row at design weights (initial) and after calibration (final):

- bus fares England −6.5 % → −0.0 % (round 3: −46.6 % → −0.1 %); London −2.6 % → +0.0 %
  (−72.3 % → +0.1 %); support England +5.0 % → +0.0 % (−28.9 %); London +4.9 % → +0.0 %
  (−72.7 %); Scotland fares −3.0 % → −0.1 % (−63.0 %), support +1.1 % → −0.2 % (−54.4 %);
  Wales support +4.9 % → +0.0 % (−10.4 %); NI fares −4.8 % → +0.4 % (−65.3 %); NI public
  transport support +0.7 % → +0.2 % (+66.5 %). The design-weight residuals are the CGT clone
  stage's re-weighting after the lcfs and etb rakes, so the solver no longer stretches riders:
  PR-S's purpose for these rows.
- `obr.fuel_duties_cars` −34.7 % → −11.8 % (round 3: −41.1 % → −15.6 %; frame at design
  weights GBP 9.40bn through the engine's `fuel_duty` after its 0.973 litre-proxy uprating).
- `orr.rail_government_support` −76.8 % → −0.1 % (−71.9 %): the ETB donor carries about a
  quarter of ORR 7270 and the 0.935 uprating plus the NI joint cell lower it further; the
  solver closes it by weight as before. A publisher-facts rake for rail support (ORR 7270 by
  nation is not published; 7271 by source is) would be the PR-S-style fix, a ruling.
- `ons.household_energy_expenditure` +21.9 % → −0.1 % (+0.2 % → −0.1 %): the Part K finding.

Terminal battery: `uk_target_fit` passed — maximum absolute relative error 0.25, no failing
target; the HMRC self-employment 20–30k band cell that blocked round 3 at +25.8 % sits at
+24.6 % here (inside the fence, not fixed; the 30–40k cell +16.6 % → within 10 %); the five
other terminal entries passed, including both stage support-clip gates (zero clipped rows) and
`uk_support` on the regenerated bounds. The battery BLOCKED on `uk_aggregate_admin`:
`need_electricity_mean_spending` achieved GBP 811 against the anchor GBP 1,082 (−25 %, tolerance
15 %) and `need_gas_mean_spending` GBP 703 against 850 (−17 %). At design weights the same
means are GBP 1,011 and 828 (inside tolerance); the calibrated weights move them because the
solver reconciles ONS 04.5 (GBP 43.4bn, −18 % below the design-weight frame) by shifting mass
to low-energy households. The two facts are inconsistent on this frame by about 20 %: NEED
mean kWh priced at the FY2024-25 cap with standing charges, over 96.5 % gas-connected
households, exceeds the national-accounts household energy spend per household. Before PR-S
the anchor was NEED kWh at Q2-2026 unit rates with no standing charge (GBP 882 / 700) and the
frame matched ONS by coincidence. Candidate resolutions, all rulings (re-worded after Vahid's round 1, which showed the gas
rake was spreading a per-connected-household NEED mean over unconnected zeros; commit `31204e19`
rakes gas over gas-positive rows only, so the connected share now scales the frame's total gas
kWh as well as the standing charges — Part P re-measures on that basis): (a) a truthful
gas-connection signal on the donor (LCFS `a151`, 80.9 % of diary households weighted, if the
codebook confirms it as mains-gas central heating; `b490 > 0` marks only 12 %), which lowers
total gas kWh and standing charges to about 85 % of households; (b) an anchor that measures
what NEED measures — mean kWh over connected households, not pounds — so the anchor cannot
disagree with the pricing; (c) accept the ONS total as the level and widen the NEED tolerance. No
staging H5 is written when the battery blocks, so the per-row weight-stretch anatomy (mass on
households stretched >3× and >5×) is not re-measured here; the global ESS and top-1 % share
above are the stretch reading.

### Part P — round 2 after Vahid's review of PR #927 (gas raked over connected households)

Vahid's should-fix 1: NEED gas means are per gas-metered household and the cell-mean IPF was
spreading them over unconnected zeros. Commit `31204e19` rakes gas over the gas-positive rows of
each cell only (electricity over every row), on the donor and the recipient, with the rake
population receipted; the same commit declares the NTS share scope on the incidence operation
(`share_geography: E92000001`, `share_age_coverage: all_ages`, `applied_to: fare_rake_regions`)
and gives `ofgem_region_crosswalk.json` a provenance block, and `12628088` / `47ea5a03` carry
the regenerated H2 fixture, coverage manifest, the re-pinned UK `spec_sha256` (`d2e82feb…`) and
the stage-note wording on the uprating factors the rakes level away.

spine-s2 (`855b513d7d4ed637a2e9ce359ac01c90ec079ca77fc84a858e35e17c8bd56f04`, 504 s, rows as
spine-r, content identity `b899f40a…`) under `890-fuel-bus/spine-s2/`; the twin diff against
spine-r under the same expectation is clean (24 expected, none unexpected). Design-weight energy,
spine-s → spine-s2: gas mean over connected households GBP 829 → 802, over all households
799 → 774, electricity unchanged at GBP 1,012, domestic total GBP 52.94bn → 52.25bn (+20.3 %
against ONS 04.5 CY2025, +23.6 % against CY2024); gas-connected share 96.5 % on both. Bus, fuel
and support columns are byte-identical to spine-s at the lcfs stage (the fix touches energy
only), so Parts L–M's bus and fuel readings stand.

National calibration `pr-s-round2-spine-s2/` (`uk-frs-calibration-attempt-20260915T101054Z-bba6d82b`,
code `12628088`, 1,500 epochs, `family_equal`, 377 targets, 212 s): loss 0.01138 (round 1
0.01122), 95.76 % within 10 %, ESS 9,137 (9,073), top-1 % share 16.2 %. The #890 rows are
round 1's to the rounding (bus rows exact, cars fuel duty −34.7 % → −12.1 %, rail −76.8 % →
+0.0 %, ONS 04.5 +20.3 % → −0.0 %). The terminal battery blocks on two entries, both rulings
for María: `uk_aggregate_admin` (electricity mean GBP 824 against 1,082, gas GBP 688 against
850 after calibration; the Part N options, re-worded there for the connected-household rake)
and `uk_target_fit`, where the inherited HMRC self-employment 20–30k band cell that round 1 left
at +24.6 % sits at +25.1 %, one tenth of a point over the fence — the spine-r posture recorded in
Part J (a dated reviewed exclusion under her name, or a lever), not a #890 effect. No staging
H5 is written while the battery blocks.
