# #930 receipts — local bus journeys from the NTS microdata, priced at the published yield

Plan: `repos/uk-930-implementation-plan.md` (approved 2026-09-17; María's rulings: Chronicle
facts first, price per fare-paying boarding on the landed BUS01 and concessionary facts with a
declared boardings-per-resident-trip translation, statutory age eligibility at zero fare, Wales
keeps the raw LCFS draw, the ONS 04.5 sign-out stands). This is the structural fix ruled on
2026-09-15 (#890 receipts Part R) for the rake-then-bind bus fares PR #927 merged with. Worktree
`repos/populace-930`, branch `uk-930-nts-bus` on main `7c4a5ac0`. Licensed evidence, once the
NTS tabs land, goes under `data/ukds/acceptance/930-nts-bus/` (SDC-safe aggregates only); this
file carries the digest-pinned receipts the declarations anchor to.

## Part A — Chronicle landing and the feed re-pin (C2 + C3, commit `f7540332`)

`PolicyEngine/chronicle#274` (posted 2026-09-17 on plan approval) asked for the DfT BUS01
passenger journeys and concessionary journeys by area, the BUS05 operating-revenue components,
NTS0303/NTS0601 trip rates and the devolved concessionary journeys. Chronicle PR #275 landed
them at `78466057401af48f9a53da41b87295241e4af1ba`. Feed rebuilt from that commit with
`chronicle build-bundle --suite uk` → `build-consumer-artifact`:

- `consumer_facts.jsonl` sha256 `8dc4336d776533d0c878d008311f81f936568d6471ee82f025dc8c996ad8867a`
- `manifest.json` sha256 `0ff28c71e7d3c6b61ad776ff223aaad32aa01c8bb832e9834b8969a232ad77d2`
- rows / schema: 277,183 / `policyengine_ledger.consumer_artifact.v2`, consumer-fact schema
  `chronicle.consumer_fact.v3` (`bdb51e2a…`)
- previous national pin: `ec200856` (the #927 pin); copies of the artifact at
  `data/ukds/acceptance/chronicle-uk-artifact-7846605/` and `.codex-work/` in the worktree

Effect on the committed national and local surfaces before any binding change: none. The
generators regenerate byte-identical apart from the membership feed label
(`chronicle-uk-ec200856-consumer-facts` → `chronicle-uk-78466057-consumer-facts`); the
`docs/uk-chronicle-feed-repin.md` order was followed (vendoring before the generators; the
three new resources were undeclared in `country_package.json` while first vendored, since the
loader refuses a declared resource that does not exist yet).

Vendored resources (`tools/vendor_uk_ledger_facts.py`, `--check` clean on the pin):

- `dft_bus_journeys.json` — 197 rows: BUS01a journeys by area FY2005–2025 (39, thirteen areas
  by three years), BUS01c concessionary journeys by area and `journey_category`
  (elderly_disabled_concessionary, total_concessionary; 78), the nine BUS05i revenue components
  by area (54), ONS mid-year population for England, its regions, Scotland and Northern Ireland
  (26); sha256 `f67d0b92…`; consumers `uk_runtime.lcfs_consumption`, `uk_runtime.bus_fare_pricing`,
  `uk_runtime.etb_services`, `uk_runtime.bus_support_per_journey`
- `nts_trip_rates.json` — 90 rows: NTS0303 trips per person by main mode and `dft.survey_year`
  (9), NTS0601 by `dft.age_band` and mode (81); sha256 `377f811c…`; consumer
  `uk_runtime.nts_bus_travel`
- `nts_car_availability.json` — 6 rows: NTS0205 household car availability shares, England 2024
  and 2023; sha256 `bdef4153…`; consumer `uk_runtime.nts_bus_travel`
- `devolved_bus_finance.json` gains the six DfI
  `dfi_ni.translink_bus.full_fare_concession_passenger_journeys` rows
- `dft_bus_value_anchors.json`: the NTS0705a selection is pinned to
  `groupby_dimension: dft.household_income_quintile` (the feed now carries a second NTS0705a
  breakdown), so its 24 rows and the #927 targets are unchanged
- `nts_bus_use_frequency.json`: consumers are now `uk_runtime.bus_use_incidence` (the NTS0313
  fact-check helper) and `uk_runtime.nts_bus_travel`; the lcfs stage no longer reads it

DfT labels fiscal years by the closing year, so every FY2024-25 selection here is by
`fiscal_start: 2024-04-01`, never by label.

## Part B — Vahid's third pass on PR #927 (C0, commit `5326ab68`)

All nine items closed, none of which changed a value: the stale "FY2024-25 Ofgem pricing set
their level" sentence (stage notes, packaged mirror, H2 fixture, the #890 changelog fragment);
the `price_domestic_energy` schema branch's conditional connection keys (the first `if`/`then`
in `sources.schema.json`; `validate_and_inject_defaults` honours it); the declared
`connection_fallback_regions: [NORTHERN_IRELAND]` refused at runtime if the crosswalk differs;
per-sweep IPF residuals (`raking_sweep_residuals` → the energy receipt's `sweep_residuals`) with
the tolerance-governance and connection-cap sentences on the `energy_rake` gate note; the
gas-connection walk documented as the greedy weight-fitting walk it is, with the unequal-weights
skip test; the `evidence_absent_blocks` plus `synthetic_smoke` seam test; #890 receipts Parts R
and S re-attributed to the rebased `713b5064`, Part T recording that ONS 04.5.3 and 04.5.4
(about GBP 1.57bn) are vendored and unbound and that María confirmed the 04.5 sign-out; PR
#927's body edited with the rebased hashes.

## Part C — the weight-stretch instrument (C1, commit `52e363bd`)

`calibration_run.py` writes `target_support_matrix.npz`, `target_support_vectors.npz` and
`target_support_manifest.json` beside the diagnostics before `_run_calibration_gate_battery`
(`uk_runtime/target_support.py::write_uk_target_support_sidecars`), so a blocked attempt keeps
its anatomy readable; the staging H5 posture is unchanged. `tools/diagnose_uk_target_support.py`
decomposes any compiled target: target, design and final aggregates, carriers, top-carrier
concentration, the carrier weight-ratio distribution, the share of the target's final mass on
households stretched beyond 3x and 5x their design weight (each beside the frame-wide share),
and the near-cap fraction, with the recomputed final estimates verified against the attempt's
diagnostics. This is the never-measured #890 acceptance line (England bus-fare mass on
households stretched more than 3x) and #930's measurement, to be read on the first calibration
round after the tabs land.

## Part D — the stage, the pricing and the support-side fence (C4–C7, unit-level receipts)

Declarations and runtime, every publisher number reaching the stages through `vendored_rows`:

- `nts_bus_travel` (person grain, between `was_wealth` and `lcfs_consumption`; 32 declared
  stages, 30 on the H2 fixture): `clean_nts_travel_tables` on the Household, Individual and Trip
  tabs (UKDS SN 5340, End User Licence; England residents only since 2013) through a declared
  codebook; one regime-gated QRF on the interview frequency band (`impute_bus_use_band`,
  identity-keyed uniforms, rounded to the ordinal; devolved recipients on a declared proxy
  region); `assign_trips_from_band_means` (W5 × JJXSC trips over W2 persons, × 52.14, per band ×
  residence group, split into `bus_in_london_trips` and `other_local_bus_trips`, summed to
  `household_local_bus_trips`); `assign_bus_pass_eligibility` (England outside London 66+,
  London under 18 or 60+, Scotland under 22 or 60+, Wales 60+, Northern Ireland 65+). If the
  extract carries no frequency column the band is drawn from the vendored NTS0313/NTS0621 shares
  and declared midpoints scaled to the vendored NTS0303 rate, receipted. Gates
  `uk_stage_nts_bus_travel_support` (support clip) and `uk_stage_nts_bus_travel_facts`
  (`bus_travel_facts`, population fact check: user share within 0.05 of NTS0313, trips per
  person within 15 % of NTS0303, period 2024); NTS0205 car availability is a receipt line.
  The artifact pins are zero placeholders refused at runtime until the tabs land.
- `lcfs_consumption`: `assign_bus_use_incidence` and the fare cells of `rake_to_vendored_facts`
  leave the declaration; `price_bus_journeys` (`uk_runtime/bus_fare_pricing.py`) sets
  `bus_fare_spending` as Σ over the household's non-eligible persons of trips × k × y per
  series, at the chain step as the override (the chain conditions on the raw draw; the support
  clip precedes it; every other LCFS column is byte-identical on the fixture). Wales keeps the
  clipped raw draw. Gate `uk_stage_lcfs_consumption_bus_pricing` (`bus_pricing`, population fact
  check, tolerance 1e-9) recomputes every factor from the vendored rows.
- `etb_services`: the `bus_subsidy_spending` rake stays; gate
  `uk_stage_etb_services_support_rake` (`fact_rake`, population fact check) recomputes every
  cell's published net support through the stage's own declaration, requires the receipt's
  cells and each cell's design-weighted total after the rake to match (Northern Ireland jointly
  over bus and rail), and fails closed on a skipped cell; `record_support_per_journey`
  (`uk_runtime/bus_support_per_journey.py`) records the value-side alternative, `applied: false`.

Factors resolved from the vendored rows at FY2024-25 (`fiscal_start: 2024-04-01`; y = receipts
over fare-paying boardings; k = boardings over resident trips, i.e. BUS01 over NTS0705a trips
per resident CY2024 times the ONS mid-2024 population; fare per trip = k × y; C/B = the
publisher's concessionary boarding share):

- London series (every England resident's bus-in-London trips; BUS05ai London receipts
  GBP 1,347.4m; BUS01 1,821.5m boardings; concessionary 497.5m; NTS0705a 13.12 trips per
  resident; England 58.62m): k 2.369, y GBP 1.018, fare per trip GBP 2.41, C/B 0.273.
- England outside London (other-local-bus trips of every England resident; receipts
  GBP 2,070.0m; 1,850.5m boardings; concessionary 519.1m; 28.07 trips per resident): k 1.125,
  y GBP 1.555, fare per trip GBP 1.75, C/B 0.281.
- Scotland (`scotgov.bus.operator_revenue` passenger revenue GBP 391m; 334m journeys;
  concessionary 183.6m; England's other-local-bus rate as the declared proxy; population
  5.55m): k 2.145, y GBP 2.600, fare per trip GBP 5.58, C/B 0.550 (the Scottish concession
  covers under-22s since 2022, so its boarding share is twice England's; the eligibility rule
  matches it).
- Northern Ireland (Translink receipts summed over services GBP 150.1m; 67.8m journeys; DfI
  full-fare-concession journeys 8.96m, the 65+ SmartPass series, so half-fare 60–64 boardings
  are not deducted, declared; proxy rate; population 1.93m): k 1.253, y GBP 2.551, fare per
  trip GBP 3.20, C/B 0.132.
- Wales: unpriced (María's ruling); its imputed trips are receipted, not priced.

Support components resolved for the diagnostic (England only, where DfT publishes them):
London reimbursement GBP 183.3m over 497.5m concessionary boardings = GBP 0.369 per
concessionary boarding, the rest of the GBP 1,130.2m net support = GBP 0.520 per boarding;
England outside London reimbursement GBP 630.6m over 519.1m = GBP 1.215, the rest of
GBP 1,894.7m net support = GBP 0.683 per boarding. The frame-side numbers (eligible boardings,
value-side support over raked support) wait on the tabs.

Derived surfaces: the H2 fixture regenerated with synthetic NTS tabs (`_nts_donors()`; 30
stages; oracle identity on this machine `8a95b3a9c766af1f88cc127d0bc9b5ee38d69528a8b0eeb056adb6e8ec803828`,
unchanged by C6 and C7 because the pricing override leaves the fixture's other columns alone and
the support diagnostic is receipt-only); the release-input coverage manifest (145 required
inputs); the gate battery (53 gates, four new); the gate-battery and certification-part digests
re-pinned in `microcosm-data` after every `gates.json` edit; `tools/ci_test_groups.py --verify`
ok. Export surface gains `household.household_local_bus_trips`.

## Part E — licensed measurement (pending María's NTS SN 5340 download)

Not run. In order once the tabs are at `data/ukds/nts_2002_24/`: pin the three artifacts in
`sources.yaml`; spine-t with the #791 twin recipe (`build_791.sh`, the three tabs added) under
`data/ukds/acceptance/930-nts-bus/spine-t/`; the stage gates and their receipts (band shares vs
NTS0313, trips per person by series, quintile and age vs NTS0705a/NTS0303/NTS0601, vehicle
shares vs NTS0205, the eligible trip share beside C/B, frame-implied boardings vs BUS01, the
design-weight fares by target cell against the #904 targets without a rake, the ONS 07.3.2 upper
bound, the support-per-journey reading); the twin diff against spine-s5 under its expectation
(only the new cells and `bus_fare_spending` outside Wales may move); the E6 support bounds for
the new stage; the national calibration round (after #937 or with the round-5 scratch
register) and the Part C anatomy per bus row.
