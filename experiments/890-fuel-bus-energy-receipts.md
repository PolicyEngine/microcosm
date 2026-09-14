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
