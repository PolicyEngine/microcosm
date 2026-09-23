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

Rebase over #939 (2026-09-18, main `ce76b358`): #939 had re-pinned the feed to chronicle `c5e5bf8`
(the HMRC CGT Tables 7–9) with its own regeneration; `78466057` is that commit's descendant on
chronicle main, so the rebase keeps this branch's pin and regenerates every surface on it over
the merged register (commit "Regenerate the derived surfaces on the 7846605 feed after rebasing
over #939"): main's two CGT resources (`hmrc_cgt_conditioning_facts.json` 254 rows,
`hmrc_cgt_asset_type_facts.json` 60) now carry the `78466057` identity beside the three bus
resources; `tools/vendor_uk_ledger_facts.py --check` clean; the national `target_references.json`
is byte-identical to main's (no compiled value moved); the H2 fixture has 31 stages (#725's
`hmrc_cgt_asset_type_spine` and #930's `nts_bus_travel`; oracle identity on this machine
`a57881b2…`), 33 declared; the coverage manifest (145 required) and the gate and part digests
re-pinned; the stage-count pins both sides had bumped to the same number moved once more.

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

## Part D — the stage, the pricing and the support-side fence (C4–C7, unit-level receipts; superseded)

Written on 2026-09-18 before the tabs landed. Parts E to K supersede it where they differ: the
artifact pins are real (Part E), the stage count is 33 declared and 31 on the fixture (Part A), the
ETB support rake retired (Part H, `uk_runtime/bus_support_pricing.py` replaced the diagnostic module
named below), and Northern Ireland's concession age is 60 (Part K). Kept as the record of the
declarations as first written.

Declarations and runtime, every publisher number reaching the stages through `vendored_rows`:

- `nts_bus_travel` (person grain, between `was_wealth` and `lcfs_consumption`; 32 declared
  stages, 30 on the H2 fixture at the time, 33 and 31 since the rebase): `clean_nts_travel_tables` on the Household, Individual and Trip
  tabs (UKDS SN 5340, End User Licence; England residents only since 2013) through a declared
  codebook; one regime-gated QRF on the interview frequency band (`impute_bus_use_band`,
  identity-keyed uniforms, rounded to the ordinal; devolved recipients on a declared proxy
  region); `assign_trips_from_band_means` (W5 × JJXSC trips over W2 persons, × 52.14, per band ×
  residence group, split into `bus_in_london_trips` and `other_local_bus_trips`, summed to
  `household_local_bus_trips`); `assign_bus_pass_eligibility` (England outside London 66+,
  London under 18 or 60+, Scotland under 22 or 60+, Wales 60+, Northern Ireland 65+ as first
  declared, 60+ with under-5s free everywhere since Part K). If the
  extract carries no frequency column the band is drawn from the vendored NTS0313/NTS0621 shares
  and declared midpoints scaled to the vendored NTS0303 rate, receipted. Gates
  `uk_stage_nts_bus_travel_support` (support clip) and `uk_stage_nts_bus_travel_facts`
  (`bus_travel_facts`, population fact check: user share within 0.05 of NTS0313, trips per
  person within 15 % of NTS0303, period 2024); NTS0205 car availability is a receipt line.
  The artifact pins were zero placeholders refused at runtime until the tabs landed (Part E).
- `lcfs_consumption`: `assign_bus_use_incidence` and the fare cells of `rake_to_vendored_facts`
  leave the declaration; `price_bus_journeys` (`uk_runtime/bus_fare_pricing.py`) sets
  `bus_fare_spending` as Σ over the household's non-eligible persons of trips × k × y per
  series, at the chain step as the override (the chain conditions on the raw draw; the support
  clip precedes it; every other LCFS column is byte-identical on the fixture). Wales keeps the
  clipped raw draw. Gate `uk_stage_lcfs_consumption_bus_pricing` (`bus_pricing`, population fact
  check, tolerance 1e-9) recomputes every factor from the vendored rows.
- `etb_services` (as first written; the rake retired in Part H): the `bus_subsidy_spending` rake stays; gate
  `uk_stage_etb_services_support_rake` (`fact_rake`, population fact check) recomputes every
  cell's published net support through the stage's own declaration, requires the receipt's
  cells and each cell's design-weighted total after the rake to match (Northern Ireland jointly
  over bus and rail), and fails closed on a skipped cell; `record_support_per_journey`
  (`uk_runtime/bus_support_per_journey.py`, since renamed `bus_support_pricing.py`) records the
  value-side alternative, `applied: false`.

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
  journeys on the full-fare-concession travel status 8.96m, the count of free-travel boardings
  across every concession, not an age series: the only travel status the feed carries (Vahid's
  round-one correction; the receipts first called it a 65+ series and inferred a half-fare tier
  that does not exist); proxy rate; population 1.93m): k 1.253, y GBP 2.551, fare per
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

## Part E — the tabs land: pins and the verified codebook (C10, commit `2b7368cc`, 2026-09-22)

María placed SN 5340 (19th edition, November 2025; End User Licence) under
`data/ukds/nts_2002_24/tab/`. The branch was rebased onto main `87e0ac07` first (clean: main had
moved only the calibration attempt-id minting and the data contract since `ce76b358`). The three
tabs are pinned by size and digest and verified at runtime through the FRS spine's pinned-tab reader:

- `household_eul_2002-2024.tab` 117,993,975 bytes, sha256 `b70252b6…`
- `individual_eul_2002-2024.tab` 270,677,548 bytes, sha256 `e8b56849…`
- `trip_eul_2002-2024.tab` 1,055,711,332 bytes, sha256 `878b61c9…`

The declared codebook was checked against the deposited lookup tables (variables and response
levels) and the data extract user guide. Three corrections, each declared, receipted and tested:

- The frequency-of-use question is `OrdBus2Freq_B01ID` ("How frequently use local buses", asked
  since 2019; `OrdBusFreq_B01ID` ended in 2018; the declared `LocalBusFreq_B01ID` never existed).
  Its ten codes fold onto the seven NTS0313 bands (1–3 → three or more times a week, 4 → once or
  twice a week, 5, 6, 7, 8 one band each, 9 and 10 → less than once a year or never). Code -9
  (not applicable, proxy responses; 2,122 persons in 2022–24, none of them under-5s) is declared a
  non-user because the published NTS0313 shares reproduce only with those persons in the base:
  the 2024 at-least-yearly share is 0.526 without them, 0.505 with them, published 0.505 (and
  0.556 vs 0.500 for the 60-and-over series, published 0.500). Code -8 (no answer) is dropped and
  receipted (41 persons in the declared years).
- `Age_B01ID` carries 21 bands (under 1 to 85+). The codebook now maps each code to its lower age
  and the declared edges (0, 5, 11, 17, 21, 30, 40, 50, 60, 70) band donor and recipient alike;
  every NTS band lies inside one declared band. The earlier one-to-one assumption (code minus one
  as the ordinal) would have put a 5-to-10-year-old in the fourth declared band while the frame
  put the same child in the second, misaligning the QRF's first predictor.
- The trip weight is `W5`, which already carries the diary household weight W2 (the user guide's
  trips per person per year is 52.14 × Σ(JJXSC × W5) / Σ W2 on the diary sample; W5xHH, the
  earlier declaration, is the within-household factor only, and W5 / (W2 × W5xHH) has median
  0.99 and a 1st–99th percentile range of 0.90–1.24 on the 2022–24 bus trips). Each person's
  counted trips are divided by W2 before the W2-weighted mean (`trip_weight_basis:
  household_and_trip`, a declared and receipted basis). Interview-only households (W2 of zero;
  3,074 of 20,564 in 2022–24) leave the donor, as the guide prescribes for trip measures.

Also verified unchanged: `HHoldGOR_B02ID` 1–9 are the English regions (10 Wales and 11 Scotland
occur only before 2013), `HHIncome2002_B02ID` 1–3 at £25,000 and £50,000 (-8 dropped, 33
households), `Sex_B01ID` 2 = female, `MainMode_B04ID` 7 = bus in London and 8 = other local bus,
`NumCarVan` the raw count (clipped 0–5), `HHoldNumPeople`, `JJXSC` ∈ {0, 1, 7}.

The E6 bounds `uk/nts_bus_travel_support_bounds.json` (band 0–6, each trip column 0–2,000 after
outward rounding, from the pinned tabs through the stage's own cleaning) join the terminal
`uk_support` gate; the H2 fixture's synthetic tabs follow the verified codebook (oracle identity
on this machine `6c5dffbc…`); coverage manifest and digests regenerated.

## Part F — spine-t and its twin (2026-09-22)

Twin recipe `data/ukds/acceptance/930-nts-bus/build_930.sh` (the #791 recipe on the post-#939 tool:
no `--cgt-ods`, `--no-staging`). Control spine-u from main `87e0ac07` in a scratch worktree with its
own venv (uk extra), 299 s, 20 of 20 stage gates, sha256 `16814f8b…`. Candidate spine-t from
`2b7368cc` with the three tabs, 369 s, engine 2.98.0, 24 of 24 stage gates (the four #930 gates
among them), sha256 `bbbf40e4…`.

Stage receipts on spine-t (design weights, the FRS 2024-25 sample of 34,966 persons in 16,288
households before the CGT incidence clone):

- Donor: 17,457 diary households, 39,047 persons over 2022–24 (weighted 41,280), region
  unmapped 0, income missing 33, frequency -8 dropped 41, age unmapped 0. Donor trips per
  person 13.73 (bus in London) and 26.05 (other local bus) against the published 2024 rates
  13.12 and 28.07 (the pooled donor years straddle the post-pandemic recovery).
- Band draw: regime-gated QRF, zero-inflated positive regime, 14 predictors (age band, sex, cars,
  household size, income band, nine region indicators), identity-keyed uniforms; donor band
  shares reproduce the NTS0313 shape (non-user 0.511, three-or-more-a-week 0.127).
- Incidence on the frame: person user share 0.5047 against NTS0313 2024 0.5049; 60-and-over user
  share 0.5010 against NTS0621 0.5004; household user share 0.723. `bus_travel_facts` passes at
  a 0.05 tolerance with a deviation of 0.0002.
- Trip rates on the frame: 12.42 bus-in-London trips per person (−5.3 % on NTS0303's 13.12) and
  31.11 other-local-bus trips (+10.8 % on 28.07), inside the 15 % fence; by age the rest-of-England
  series peaks at 17–20 (70.7 trips) and 70-and-over (37.9), the London series at 21–29 (20.5).
- Band means (diary, per band and residence group): London residents in the top band average
  206 bus-in-London trips a year, the once-or-twice-a-week band 62, non-users 13; rest-of-England
  top band 2.0 London trips; no cell fell back to the all-England mean.
- Eligibility (statutory age rule): 10,874 eligible persons (England outside London 5,739 of
  22,466; London 1,311 of 2,860; Scotland 1,929 of 3,325; Wales 1,007 of 2,370; Northern Ireland
  888 of 3,945). The donor's eligible share of trips is 0.442 for the London series and 0.168 for
  other local bus (age-band lower bounds).
- Vehicle shares (frame vs NTS0205 2024, receipt line): no car 0.236 vs 0.218, one car 0.463 vs
  0.442, two-plus 0.301 vs 0.340. A `was_wealth` finding, as the plan said; not fenced here.
- Support clip: 0 rows clipped on the four columns (donor maxima 1,690 and 1,818 trips).

Pricing on spine-t (lcfs `bus_pricing`, gate `uk_stage_lcfs_consumption_bus_pricing` passing at
1e-9): 32,596 persons priced, 9,867 eligible persons zero-priced, 15,112 households priced, 1,176
Wales households keep the raw draw. Design-weight fares against the bound receipts, the #930
acceptance line:

- England outside London: frame £2,316m against BUS05ai £2,070m, frame / R 1.119; frame-implied
  boardings over BUS01 1.018; frame eligible trip share 0.209 beside the publisher's concessionary
  boarding share 0.281.
- London series: £1,215m against £1,347m, 0.902; boardings ratio 1.101; eligible share 0.405
  beside 0.273.
- Scotland: £558m against £391m, 1.428; boardings ratio 1.534; eligible share 0.581 beside 0.550.
- Northern Ireland: £264m against £150m, 1.757; boardings ratio 1.938; eligible share 0.213
  beside 0.132.

Reading: the two England cells sit inside the 25 % fence without any rake (the algebra's two
factors are 1.02 × 1.10 for England outside London and 1.10 × 0.82 for London: the composition
check is near one, the concession check says the frame's London eligible share is above the
publisher's boarding share). The two devolved cells are outside it, and the boardings ratio says
why: their persons are predicted on a declared proxy region (`region_remap`: Scotland and Northern
Ireland → North East, a high-bus region), so the frame gives them 1.5× and 1.9× the boardings the
publisher counts. That is a translation choice, not a level the stage set; the receipts carry it
for María's ruling (a different proxy region, or the devolved trip rates scaled to the published
boardings, are the candidate fixes). The weighted fares total moves from £2,783m (the #890 raked
level on the same frame) to £4,354m before calibration, of which £2,316m + £1,215m is England.

Support-per-journey diagnostic (ETB, recorded, not applied): England outside London value-side
£1,807m against the raked £1,895m (0.953); London £1,308m against £1,130m (1.158; the frame's
eligible boarding share 0.410 against the publisher's 0.273 drives it). The `fact_rake` gate
passes on all five cells at 1e-6.

Twin diff (`twin_diff_930.sh`, expectation `spine-t-payload-expectation.json`): compare and
classify report nine observed differences, seven expected (the five new person columns at
positions 99–103 of 104, the household sum at position 40 of 75, and `bus_fare_spending` values),
two unexpected, both the `column_order` structural surface the classifier cannot declare
expected; adjudicated as the #791 twin was, by the position-wise check that every shared column
keeps its order (person 99 shared, household 74, benunit 22) and no shared column other than
`bus_fare_spending` moves. `bus_fare_spending`: 3,806 Wales rows byte-equal, 42,540 of 49,040
other rows changed (the remainder are zero both sides). Weights, indices, row counts, root
attributes and every other column byte-equal.

## Part G — national calibration on spine-t and the stretch anatomy (2026-09-22)

`calibrate_930.sh`: the rowwise driver's national release role (`tools/build_uk_rowwise_candidate.py
--release-role national`, the seam that replaced the retired calibration runner under #823; doctrine
solve, no flags: 1,500 epochs, `family_equal`, learning rate 0.02, seed 0), input spine-t sha256
`bbbf40e4…`, the branch's own Chronicle artifact `7846605` (facts `8dc4336d…`, manifest
`0ff28c71…`), staging local-only, release id `uk-930-spine-t-calibration`, code `2b7368cc` clean.
473 s. Outputs under `data/ukds/acceptance/930-nts-bus/calibration-t/`: `microcosm_uk_2024_25.h5`
sha256 `e6919c04…`, `calibration_diagnostics.json` `e24078e2…`, the frozen
`national_target_registry.json`, the terminal gate report and the Part C sidecars
(`target_support_matrix.npz`, `target_support_vectors.npz`, `target_support_manifest.json`).

- Terminal battery: all six gates pass, nothing blocked, `release_candidate: false` (the national
  role never signs shippability). The H5 was written.
- Fit: 638 compiled targets on 52,846 records (all carrying weight); loss 0.3065 → 0.01036;
  96.55 % of rows within 10 %; ESS 9,231; realised max weight ratio 10.0 (the cap); top-1 %
  weight share 0.162. The v20 twin on main `ce76b358` (same register, 638 rows): loss 0.3009 →
  0.01041, 96.39 % within 10 %, ESS 9,305, top-1 % 0.163. The structural change costs nothing in
  fit: loss and the within-10 % share move a hair in #930's favour, ESS 0.8 % against. 22 rows sit
  beyond 10 % after the solve on spine-t and 23 on v20, the same rows but one: v20's CGT gains 25–34
  age band comes inside the fence on spine-t. None is a bus row (income tax −13 %, VAT +18 %, child
  benefit +19 %, council tax −12 %, the two HMRC self-employment bands, four state pension bands, ESA
  claimants, two UC household rows, the SLC adult dependants' grant, fuel duties on cars and the CGT
  age bands make up the rest), and none moved with #930.
- Bus rows, design weights → final (the #930 acceptance reading, now a measurement instead of
  the rake's tautology): England fare receipts (£3,612m, BUS0415-aligned to 2025) −10.9 % → 0.0 %
  (v20, raked at base year then uprated: −6.1 %); London fare receipts (£1,347m) −21.6 % → +0.1 %
  (v20 −2.6 %); Scotland passenger revenue (£391m) +27.8 % → −0.1 % (v20 −1.7 %); Northern
  Ireland passenger receipts (£150m) +69.6 % → 0.0 % (v20 −4.0 %). Net support rows (England
  +4.7 %, London +3.8 %, Scotland +1.3 %, Wales +3.8 %) are unchanged from v20, as the ETB rake
  is kept. The two England fare rows sit inside the 25 % fence without a rake; the two devolved
  rows do not, for the proxy-region reason Part F gives.
- Stretch anatomy (`tools/diagnose_uk_target_support.py`, thresholds 3× and 5×): the England
  fare row's final mass on households stretched beyond 3× their design weight is 47.6 % against
  the frame-wide 46.6 %, beyond 5× 36.2 % against 36.4 %, on 25,682 carriers with a top-1 carrier
  share of 0.6 % and a median weight ratio of 0.46. The row is carried by the frame at large, not
  by a stretched tail: this is the #890 acceptance line, measured for the first time, and it
  passes. London fares: 55.2 % beyond 3× (46.6 %), 41.6 % beyond 5× (36.4 %), 2,875 carriers,
  top-1 1.5 %. Scotland (13.5 % beyond 3×) and Northern Ireland (12.6 %) sit on down-weighted
  carriers (median ratio ~0.5) because their design-weight estimates were above target. The
  frame-wide 46.6 % beyond 3× is a property of this calibration campaign (the 10× cap binds), not
  of #930.
- Score against the incumbent: not run. The in-tree `score_uk_national_candidate.py` refuses
  the incumbent `enhanced_frs_2024_25.h5` (`e433e532…`) at the frame bundle's global-column rule
  (`region`, `country`, `esa_contrib` on both person and household), and the evaluation repo's
  `score_uk_pass2.py` (the v22 recipe) has drifted from this tree's scorer signature
  (`band_edge_registry`). The rule-1 score belongs to the #823 assessment lane, which carries
  that machinery; #930's acceptance is the fit, the bus rows and the anatomy above.

Rulings this measurement puts to María: (1) the devolved proxy: Scotland and Northern Ireland
persons are predicted on North East's bus profile and price to 1.43× and 1.76× their receipts at
design weights (the calibration closes them, at the cost the anatomy shows is modest); the
alternatives are a different declared proxy region, or scaling the devolved series' trips to the
published boardings ratio; (2) whether the ETB support rake now retires in favour of the
support-per-journey value side (0.95 and 1.16 of the raked levels); (3) the vehicle-share gap
(frame two-plus-car households 0.301 against NTS0205's 0.340) as a `was_wealth` follow-up.

## Part H — the support rake retires: bus support priced from journeys (C12, 2026-09-23)

María's ruling (2026-09-23): raking retires with the imputation; Wales and Northern Ireland keep
the raw ETB draw for support; the change goes into #954. Commits `f9b3fcc4` (the pricing step,
the retired `rake_to_vendored_facts` kind, `fact_raking.py` and the `fact_rake` check leaving with
their last consumer, the `uk_stage_etb_services_support_pricing` gate) and `742a2346` (the ETB
support-clip gate no longer exempts the column: spine-t2's first build blocked on that pin, the
column is clipped like education and rail at the same zero allowance), both rebased below.

Declaration: `price_bus_support` on `etb_services` — support areas London and England outside
London (DfT BUS05bi reimbursement and net support, BUS01 boardings and concessionary boardings)
and Scotland (Scottish Transport Statistics: concessionary fares £392m inside all government
support £499m; 334m journeys, 183.6m concessionary); `raw_draw_regions: [WALES,
NORTHERN_IRELAND]` (no published Welsh boardings; the NITHC accounts carry bus and rail jointly).
Per household: reimbursement per concessionary boarding × its eligible persons' boardings plus
the rest of the net support per boarding × every boarding, boardings being the `nts_bus_travel`
trips times the fare pricing's boardings per resident trip. The published totals stay bound as
calibration targets.

spine-t2 (code `742a2346`, same base as spine-t, 364 s): 24 of 24 stage gates. Support clip:
0 rows clipped on `bus_subsidy_spending` (donor max £14,074). Pricing receipt (design weights,
the FRS sample before the CGT clone; 13,373 households priced, 2,915 on the raw draw):

- England outside London: £1.215 per concessionary boarding, £0.683 per boarding; frame boardings
  1,929m, eligible boarding share 0.208 beside the publisher's 0.281; support £1,856m (clipped
  raw draw) → £1,807m priced against the published £1,895m, 0.953.
- London: £0.369 and £0.520; boardings 1,950m, eligible share 0.410 beside 0.273; £307m → £1,308m
  against £1,130m, 1.158.
- Scotland: £2.135 and £0.320; boardings 519m (1.55× the published 334m, the proxy-region reading
  of Part F), eligible share 0.581 beside 0.550; £224m → £810m against £499m, 1.623.
- Raw draw: Wales £133.6m (the published concessionary-plus-operator support is £131.5m),
  Northern Ireland £84.0m (the joint bus-plus-rail cell was £111.7m).
- Weighted support in total: £2,605m before, £4,142m after (the priced areas' published total is
  £3,524m).

Twin diff spine-t → spine-t2 (`twin_diff_930_t2.sh`, expectation `spine-t2-payload-expectation.json`):
three observed differences, three expected, none unexpected. `bus_subsidy_spending` moves on every
row (priced in the three areas; Wales and Northern Ireland return from the raked value to the
clipped raw draw). `rail_subsidy_spending` and `rail_usage` move on exactly the 679 Northern
Ireland rows the retired joint bus-plus-rail cell used to scale. Weights, indices, every other
column and root attribute byte-equal.

The `bus_support_pricing` gate recomputes the three areas' components and per-boarding rates from
the vendored rows at 1e-9 and reports the priced-over-published ratios (0.953, 1.158, 1.623)
without fencing them: the calibration targets act on them, as they act on the fares.

## Part I — the stage and ticket tables: the single-fare share (C13, 2026-09-23)

María's ask (2026-09-23, after the fare-cap question): read the ticket table for the single-fare
share. The NTS extract's Stage table records every boarding (`NumBoardings` counts the vehicles on a
stage), the cost paid at the point of use (`StageFareCost`) and the held ticket used (`IndTicketID`);
the Ticket table records the season tickets and passes a person holds (`SpecialTicket_B01ID`: 1–6
non-concessionary, 7–11 concessionary, from the deposited lookup tables). Pinned as two further
artifacts: `stage_eul_2002-2024.tab` 1,039,281,193 bytes sha256 `4a6015ed…`, `ticket_eul_2002-2024.tab`
8,655,532 bytes sha256 `276a99bd…`.

Boarding payment classes on the 2022–24 diary (England, bus stages, unweighted counts read while
designing the declaration): 30,403 bus boardings; 12,634 paid at the point of use (no ticket record,
positive cost), 10,399 on concessionary passes, 3,450 on non-concessionary season tickets or passes,
2,758 free with no ticket (children, by age), 1,162 with no cost recorded. Single-fare share of
fare-paying boardings 0.79 (London series 0.90, other local bus 0.71). Design-weighted by residence
group and survey year (W2): outside London the point-of-use share of boardings rose from 0.34 (2022)
to 0.43 (2023) and 0.46 (2024) while the season-ticket share fell from 0.25 to 0.16 and 0.15, the
diary's reading of the £2 cap's pass-to-single shift; the fare paid per boarding outside London
averaged £2.10 in 2022 and £1.95 and £2.02 under the cap, median £2.00 in all three years; London
£1.57, £1.85, £1.87 (median £1.65 then £1.75, the TfL single). The NTS boardings per bus trip are 1.13
(London series 1.15, other local bus 1.13): the declared translation for England outside London
(1.12) is the same number from the publisher's side, and the London series' 2.37 is twice it, the
scope difference (non-resident boardings in TfL's count) Part D declared.

Declaration and runtime: the stage classes every bus stage, annualises each person's weighted
boardings by class and series (the trip's weight on the household-and-trip basis, times the
vehicles boarded), takes per series, residence group and band the boarding-weighted share of
point-of-use boardings among fare-paying ones (a cell without fare-paying boardings takes its
group's share, then England's, receipted), and assigns each person the trips-weighted combination
of the two series' cell shares as `local_bus_single_fare_share` (a person with no trips takes the
plain mean). The column joins the export surface (person grain), the stage's support clip and E6
bounds (0 to 1); the receipts carry the class shares, fares and boardings per trip above. No
published table fences the share (DfT does not publish boardings by ticket type), so it is a
receipt line, not a gate. The parameter side, a per-journey cap on the single-fare boardings of
fare-paying persons, is policyengine-uk#1871.

spine-t3 (code `bf9f15e1`, the tree rebased onto main `b78f5a88` plus C13, 371 s): 24 of 24
stage gates; support clip 0 rows clipped on the share (donor range 0 to 1). Design-weighted diary
readings from the stage's receipt (W2 on the diary sample, by residence group and survey year):
London boardings 2024 are 0.42 paid at the point of use, 0.31 on concessionary passes, 0.05 on
season tickets or passes, 0.18 free with no ticket (children under 11 ride free in London), the
single share of fare-paying boardings 0.90 in every year, the fare paid £1.64 mean and £1.75
median (2022: £1.47 and £1.65); the rest of England moves from 0.32 paid at the point of use, 0.21
on season tickets and a single share of 0.60 in 2022 to 0.39, 0.14 and 0.74 in 2023 and 0.39, 0.14
and 0.74 in 2024, the fare paid £1.97 → £1.76 → £1.82 mean, £2.00 median throughout, the diary's
reading of the £2 cap. All-England single share of fare-paying boardings by series: 0.90 (bus in
London), 0.70 (other local bus); per band outside London it runs from 0.86–0.94 in the occasional
bands to 0.61 in the three-or-more-times-a-week band, where the season tickets sit. NTS boardings
per bus trip 1.125 (London series) and 1.121 (other local bus), the same 1.12 the declared
translation gives for England outside London from the publisher's side. No cell fell back.

Twin diff spine-t2 → spine-t3 (`twin_diff_930_t3.sh`, expectation `spine-t3-payload-expectation.json`):
three observed differences, one expected (`person.local_bus_single_fare_share`, surface
column_only_right, at position 104 of 105), two the structural surfaces the classifier cannot
declare: the person `column_order` (the grown column set) and the root attribute
`populace_mass_log_json`, which differs by exactly the six per-stage household-mass conservation
receipts main's #967 added (WAS, regional uprating, NTS, LCFS, ETB VAT, ETB services, each old and
new total 28,972,433 and factor 1.0). No shared column moves on any entity, `bus_subsidy_spending`
included (the raw-draw rows and the priced rows byte-equal to spine-t2), weights and indices
byte-equal.

On the frame (spine-t3, design weights, 113,590 persons after the CGT clone): the trip-weighted
single-fare share of fare-paying riders is 0.746 overall, 0.703 in England outside London, 0.901 in
London (the TfL single dominates), 0.688 in Scotland, 0.694 in Wales and 0.672 in Northern Ireland
(the three devolved nations take England's cell shares through their proxy region); per person it
runs from 0.61 (the frequent band outside London) to 1.00. Every person carries positive trips
because the non-user band's diary mean is small but positive (never-users do record the odd
trip), so the share is defined everywhere and the cap's bite on a person is share × fare-paying
boardings × the capped single-fare gap.

## Part J — national calibration on spine-t3 and the anatomy of the priced support rows (2026-09-23)

`calibrate_930.sh` on spine-t3 (code `bf9f15e1`, the tree rebased onto main `b78f5a88` with C12 and
C13; artifact `7846605`; the #823 doctrine solve), 463 s, outputs under `calibration-t3/`.

- The terminal battery blocks at `uk_target_fit`, and not on a bus row: main's #967 register carries
  María's signed four-week deferral of `hmrc/self_employment_income_income_band_20_000_to_30_000@2025`
  (v20: +25.3 %, just outside the 25 % fence, a calibration-competition regression from the #906/#939
  surface expansion), and on this attempt the row solves to +24.2 %, inside the fence, so the gate
  refuses the deferral as stale by design ("stale reviewed target-fit exclusions are back inside the
  bound; remove them"). No H5 is written. spine-t's calibration (Part G) ran on the pre-#967 register
  and passed; the row solved to +24.6 % there, inside the fence as well. Whether the deferral comes off this branch or main
  is María's ruling; the receipts below are read from the Part C sidecars, which the seam writes
  before the battery.
- Fit: 643 compiled rows (five more than Part G: #967's additions) on 52,846 records; loss 0.3181 →
  0.01048; 96.73 % within 10 %; ESS 9,033; max ratio 10.0. 21 rows sit beyond 10 % after the
  solve against spine-t's 22, the same rows but one (the UC households-with-two-children row comes
  inside the fence); none is a bus row, and the self-employment 20–30k band sits at +24.2 %.
- Bus rows, design weights → final. Fares unchanged from Part G: England −10.9 % → 0.0 %, London
  −21.6 % → 0.0 %, Scotland +27.8 % → 0.0 %, Northern Ireland +69.6 % → 0.0 %. Support, now priced
  from journeys rather than raked: England net support +2.5 % → −0.1 % (Part G's raked column sat at
  +4.7 % because the calibration targets are BUS0415-aligned and the rake was at base year), London
  +12.8 % → +0.1 % (the value side at 1.158 of the published net support, the frame's eligible
  boarding share 0.41 against the publisher's 0.27), Scotland government support +61.8 % → −0.3 %
  (1.623: the proxy-region boardings, as for its fares), Wales +5.5 % → 0.0 % (the raw draw).
- Anatomy (`diagnose_uk_target_support.py` on the blocked attempt, thresholds 3× and 5×). Fares:
  England 47.9 % of the row's final mass beyond 3× against the frame-wide 47.9 % (Part G: 47.6 % against
  46.6 %), London 56.8 % against 47.9 %, Scotland 14.7 %, Northern Ireland 13.2 %. Support: England net
  support 48.6 % against 47.9 % on 37,990 carriers with a top-1 share of 0.4 % (the priced column
  reaches every household with a bus rider, where the raked one reached the ETB draw's carriers);
  London 50.4 %, Scotland 10.7 % (its carriers are down-weighted, median ratio 0.43, because the
  design-weight estimate was 62 % above target), Wales 27.9 %. The frame-wide 47.9 % beyond 3× is the
  campaign's property (the 10× cap binds on the same tail as v20), not #930's.

Reading: retiring the support rake costs nothing in fit (loss 0.01048 on 643 rows against Part G's
0.01036 on 638, the difference being #967's rows), moves the England support row's mass no more than
the frame's, and turns the Scotland support row into the same proxy-region finding as its fares.
The rulings for María stay the two from Part G (the devolved proxy region; the vehicle-share gap) plus
the stale self-employment deferral, which blocks any national line on this tree until it is removed.

Post-build fix-up: the C13 sweep's H2 parity test showed the stage writing the share column before the
eligibility flag while the declared outputs (and the graph's cells) put it after; the write order is
aligned and the fixture regenerated (oracle `d7486a5d…`, now equal to the graph's identity). spine-t3
was built one commit earlier with the two columns in the other order; values, gates, the twin diff and
the calibration are unaffected, and the rebuild is not repeated for a column-order change.
