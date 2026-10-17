# Re-pin the UK Chronicle consumer feed

One reviewed Chronicle artifact identity, `uk/chronicle_feed.json`, governs
every UK consumer of the feed: the national references and membership, the
local census, the local references, the local validation levels and the
vendored per-concern fact resources. The target contract
(`uk/uk_population_targets.json`) is likewise one file for both grains. A
re-pin is therefore one reviewed change; the surfaces regenerate from the same
artifact and cannot drift apart silently (the local census and the validation
register restate the pin and are drift-gated against it).

Rebuild the complete UK bundle and consumer artifact in
`PolicyEngine/chronicle` at the declared commit. Keep the resulting
`consumer_facts.jsonl` and `manifest.json` together; do not commit either
file. The untracked default location is `.codex-work/consumer_facts_uk.jsonl`
+ `.codex-work/consumer_facts_uk_manifest.json` (also mirrored as the artifact
directory `.codex-work/uk-artifact/` for the calibration runner); the hermetic
regeneration tests accept a `CHRONICLE_UK_FACTS` override and skip only when
neither is present.

Verify both SHA-256 digests and the manifest's `facts_sha256`, row count, and
schema version, then update `uk/chronicle_feed.json` and the two tests that
restate it: `tests/engine_free/uk/test_uk_chronicle_feed.py` (the pin and the
consumer-fact schema digest) and `tests/engine_free/uk/test_uk_full_targets.py`
(`test_current_national_and_local_pins_share_one_reviewed_identity`). A schema
digest moves for every country whose feed is built after the Chronicle change,
so the NZ pin (`nz/chronicle_feed.json`) moves at its own next re-pin. Then
regenerate. Vendor
the per-concern fact resources (step 5) before the national generator: the
generator refuses to compile against a vendored resource whose feed identity
differs from the pin, so on a fresh pin step 5 has to run first. The rest go in
this order:

1. the national references and membership with
   `tools/generate_uk_target_references.py` (pass the stable
   `--source-fact-feed` label recorded in the membership);
2. the local census with `uv run --no-sync python tools/census_uk_local_targets.py`
   (it restates the pin from the shared declaration);
3. the local references and membership with
   `tools/generate_uk_local_target_references.py` (same label);
4. the local validation-level register's `source_feed` block
   (`uk/local_validation_levels.json`), which the loader checks against the pin;
5. the vendored fact resources with `tools/vendor_uk_ledger_facts.py`
   (`uk/ledger_fact_vendor_selections.json` names them; each records the feed
   identity it was taken from);
6. the signed compile parity receipts with
   `tools/build_uk_ledger_compile_parity_signed_differences.py --surface all`;
7. the release input coverage manifest with
   `tools/build_uk_release_input_coverage_manifest.py`: it hashes the CGT
   conditioning resource, whose vendored header records the feed identity.

Both compile parity receipts are read at build time: the release-blocking
preflight gates `uk_ledger_compile_parity_*` compare each signed value with the
live compile, so a receipt left on the old pin blocks every full or dense build
and the release-cut certifier. Regenerate them on the head that will be built.

The high-value home stock (`uk/high_value_home_stock.json`) is built from feed
facts too: the Census 2021 TS054 tenure record sets, the OBR non-owner-occupied
share and the ONS HPSSA seat cells. Check that those facts are unchanged on the
new pin, and rebuild the stock with `tools/build_uk_high_value_home_stock.py`
if they are not; its provenance names the feed it was built from.

Verify the complete compiled target diff on both surfaces, including targets
outside the intended policy area, and record the value moves in the changelog
fragment. A national reference whose metadata moves (a fact key or a
geography stamp) moves the national register version even when no value
does, so the frozen scoring register of the candidate evaluation
(uk-candidate-eval) must be re-frozen before the next national solve; record
that in the history entry.

A re-pin also moves the charter-H2 parity fixture: every stage that declares a
vendored resource as an artifact binds that resource's digest into its
`stage_contract_sha256`, and the vendored headers restate the feed identity.
Regenerate it with `tools/graph_uk_spine_fixture.py` and check that only those
contract hashes move.

Two-level (country + region) contract targets fan out over the region tier
(`UK_REGION_TIER` in `microcosm.calibrate.geography_constants`), one reference per area
(microcosm#905); their cells resolve Chronicle's region- and country-stamped
facts, so a re-pin must carry all twelve areas or the national generator
refuses. The cross-grain legs of English constituencies and authorities come
from `region_code_by_area` in `local_area_crosswalk.json`, regenerated from
the sha-pinned ladder with `tools/generate_uk_local_area_crosswalk.py`.

Both release roles of `microcosm-build-uk` refuse a feed whose facts or
manifest digest differs from the committed pin. On the national role
`--allow-unpinned-feed` is an explicit diagnostic override recorded in the run
manifest; it is not a re-pin procedure. The dense role refuses that flag: the
graph's target compilation checks the supplied hashes and the artifact against
the committed pin and has no override.

History: the `ec7169b` re-pin (#887/#900) moved census household targets onto
the same Chronicle compile path as every other bound UK local family; the
`c6f9361` re-pin (#890) added the chronicle #254/#255 and #257/#258 transport
and energy packages and unified the national and local pins into this one
declaration; the `474a0ae` re-pin (#904, chronicle #263) moved the rows to
`chronicle.consumer_fact.v2`, which carries the dimension and value labels the
schema-8 target hierarchy completes from (141,400 rows, including chronicle #260's
Universal Credit packages); the `df35af7` re-pin (#929, chronicle #264 and #267)
moved the rows to `chronicle.consumer_fact.v3`, which adds the publisher's
`geography.name` to every fact (the label the hierarchy needs for constituencies
and local authorities, microcosm#920) and brought the council taxbase packages
for England, Wales and Scotland (266,390 rows); the `ec20085` re-pin (#890 PR-S, chronicle #269/#270 via PR #271) brought the domestic energy facts the energy stage levels and prices against (DESNZ Energy Trends domestic consumption, subnational consumption and meter counts, QEP average prices paid, NEED 2024, ONS 04.5 sub-classes) and the census central-heating tables (275,698 rows).
The `c5e5bf8` re-pin (microcosm#725, chronicle #273, on top of `ec20085`) then brought the HMRC CGT
Tables 7, 8 and 9 (asset type, residential property, carried interest), 276,205 rows, with no
compiled value moving on either surface.
The `7846605` re-pin (microcosm#930, chronicle #274 via PR #275, on top of `c5e5bf8`) brought the DfT BUS01 passenger and concessionary journeys by area, the BUS05i operating-revenue and support components, the NTS0303/NTS0601 trip rates by mode and age and the DfI full-fare concession journeys (277,183 rows), with the NTS0705a selection pinned to its quintile groupby because the new packages reuse the trip-rate concepts.

The `00b4b14` re-pin (PolicyEngine/chronicle#280 lane, chronicle #280/#282 and #274/#275) moved the rows to
`chronicle.consumer_fact.v4`, which names every geography once per identifier from Chronicle's
register and keeps the publisher's own text as an optional `geography.publisher_name`, and brought
the HMRC Income Tax liabilities statistics of July 2026 (Tables 2.1 to 2.6, 2023-24 outturn and the
2024-25 to 2026-27 projections), the SPI 2023-24 Tables 3.3, 3.4, 3.5, 3.8 and 3.11 and the Table
3.7 remainder, the ESA caseload by payment type and phase, the property rental income statistics
2026, the DWP Spring 2026 benefit expenditure and caseload tables and the bus journeys revenue
components (287,024 rows). No compiled value moved on either surface: the national and local
reference files are byte-identical, the vendored resources carry the same rows, and the three
compile-parity receipts are unchanged; the NTS0705a bus-trips vendor selection is pinned to its
income-quintile dimension because the #275 facts share its concepts. The compiled register version moves (`d131ebf617e3` to `4b2dc4698207`) on display metadata alone: eleven region-tier specs carry the v4 spelling in `ledger_geography_name` and `ledger_fact_label` (Yorkshire and The Humber; the East of England label), so the frozen scoring register in uk-candidate-eval needs a re-freeze before the next national solve.
The pin then moved to `5324aa2`, Chronicle main after PR #284 (the validator fix the v4 export
needed: `build-consumer-artifact` at `00b4b14` refused every renamed-geography row). The export
re-run at the fix from the same suite bundle is byte-identical (287,024 rows, the same digests), so
only the commit and the feed label move: no value, row or receipt changes.
The `505e0e7` re-pin (microcosm#1014, PolicyEngine/chronicle#287 via #288, on top of `5324aa2`)
brought HMRC CGT statistics 2026 Table 4 (Business Asset Disposal Relief and Investors' Relief
claimants, qualifying gains and tax by band of qualifying gain, individuals, trusts and all
taxpayers, 2021-22 to 2024-25; 287,150 rows). The addition is purely additive: the national and
local reference files are byte-identical, the vendored resources carry the same rows under the new
feed identity, and the three compile-parity receipts are unchanged.
The `825406f` re-pin (microcosm#1069, PolicyEngine/chronicle#302 via #305, on top of `505e0e7`) brought
the UK pension facts: Stat-Xplore State Pension and Pension Credit caseloads and amounts from
February 2023 to March 2026 (by type, age, gender, region, partner and amount band), Northern
Ireland State Pension and Pension Credit (DfC, May 2026), the DWP Spring 2026 State Pension and
Pension Credit forecast lines, Attendance Allowance, Winter Fuel Payment recipients and statistics
for winters 2023-24 to 2025-26 and the HMRC charge, the DWP workplace pension savings trends, ONS
ASHE pension membership and contribution-rate bands, HMRC pension contribution relief (2023-24) and
salary sacrifice relief (2024-25), and the OBR salary-sacrifice costing (344,402 rows). It also
carries chronicle #246, #278, #292 and #304. No compiled value moved on either surface: the
national and local reference files are byte-identical and the vendored resources carry the same
rows under the new feed identity. The artifact carries every `505e0e7` value but is not a strict
row superset: #305 rebuilt the DWP Spring 2026 package's record sets, so 44 incapacity-benefit
forecast rows come back under new keys with the same forecast line and value, and 192 rows change
only their layout fields; no UK target binds them. The five salary-sacrifice relief targets now
match two facts each (the new 2023-24 Table 6.2 rows beside the bound 2024-25 ones) and still
resolve to the 2024-25 fact. For the same reason the 2023 production compile-parity receipt gains
five ledger-only rows (the relief targets now compile at that comparison period from the 2023-24
facts); the two 2025 incumbent receipts are unchanged.

The `1ee7dfe` re-pin (microcosm#1113, PolicyEngine/chronicle#322 via #323, on top of `825406f`)
brought ONS Consumer Trends household final consumption expenditure, annual for 2020 to 2025 and
quarterly to 2026 Q1: the national and domestic totals, the twelve domestic COICOP divisions and net
tourism (sheet 0CN), their classes (sheets 01CN to 12CN) and the tourism detail (TOURCN). It also
brought DESNZ road transport energy consumption by vehicle type and fuel for 2005 to 2024, for the
UK, Great Britain, the four countries and the English regions (352,549 rows). #323 also restates the
seven Consumer Trends series already in the feed as domestic household spending: the 217 rows keep
their concepts and values under new keys, with the territory role and the price-basis,
seasonal-adjustment, unit, sheet and concept dimensions. Every other `825406f` row is
byte-identical. The national and local reference files are byte-identical; the electricity, gas and
road-fuel targets resolve to the same values through the new keys. The vendored resources carry the
same rows under the new feed identity, the Consumer Trends resource's 42 under their new keys, and
the re-pin leaves the three compile-parity receipts unchanged.

The `942a1fa` re-pin (microcosm#1123, PolicyEngine/chronicle#313 via #314, on top of `1ee7dfe`)
brought the publisher inputs for the 2025 calibration: household estimates by nation and region
(LFS 2025 for England's regions and the GB countries, NRS 2025, Welsh Government mid-2024, LPS
dwelling stock 2021–2026), mid-2025 population estimates (the UK, country and region workbook,
every local authority and the devolved council and district tables) with ONS's revision of
mid-2024, the census residence-type tables (RM120, UV101a, CT0105), monthly Universal Credit by
area from January 2025, PIPR and Scottish rent statistics, SPREE and the Welsh and Scottish
dwelling stock by tenure, and newer editions of SLC repayments, ISC, public sector employment,
land values and the DfI receipts (927,239 rows, 2.6 times the previous feed). Three selectors had to
move with the data: the ONS mid-2025 workbook lays its rows out by geography, so the population
selectors pin ages through `dimension_values` and no longer require the age layout; the
preliminary end-2025 land rows carry `publication_status: preliminary` and are bound explicitly;
and public sector employment is the mean of its four 2025 quarters. Three vendor selections pin
the periods their stages were built on (UK and regional population 2023 and 2024, the DfI series
to FY2024-25), so the only spine inputs that move are publisher revisions: the revised mid-2024
population (UK 69,256,274 against 69,281,437, twelve regional totals) and the DfI edition's
rounding of seventeen earlier values (eleven receipts and six journeys).

The comparison with the 2023 production surface is retired (microcosm#1257, María's ruling of
2026-10-13). Two compile-parity receipts remain, both at the 2025 calibration period: the national
one against the frozen incumbent registry, and the local one. Earlier entries above that name three
receipts describe the state at their time.

The `735820c` re-pin (microcosm#1240, PolicyEngine/chronicle#343 via #349, on top of `942a1fa`)
brought the inputs for the High Value Council Tax Surcharge work: the OBR's April 2026 costing
note (63 rows: properties in scope by surcharge band and the non-owner-occupied share) and the
ONS house price statistics for small areas by 2024 constituency to the year ending March 2026
(22,940 rows), plus three raw-only registrations that carry no facts (the HM Land Registry Price
Paid yearly files, the UK HPI full file and the NSPL) and are read by
`tools/build_uk_high_value_home_stock.py` against the registered sha256 pins (950,242 rows, up
23,003). No existing package moved, so no compiled target changed; one vendor selection
(`obr_hvcts_costing_facts.json`) pins the OBR note for the benchmarks gate.

The `abf0921` re-pin (microcosm#1300, PolicyEngine/chronicle#359 for chronicle#326, #339, #356, #357
and #358, on top of `735820c`) brought Universal Credit households by area split by Stat-Xplore's
Payment Indicator, HMRC stamp tax statistics, DWP's Move to UC claim rates, HBAI with its vintages,
the DWP geography of OBR table 4.9's lines, DESNZ Northern Ireland and LSOA domestic energy,
rural-urban lookups, the BUS04 country rows, Scotland's UV404 tenure by output area and census
residence type by age (1,499,301 rows, up 549,059). The v4 schema digest moves to `d953306d…` for
the new `gas_year` period type. Three selector families moved with the data, and one interim stays:

- the UC area families bind `payment_indicator: Yes` (`dwp.uc.households_by_area` at both grains
  and the four child bands, whose summed operands name it too), because Chronicle retired the
  all-claims cells. The cells now count paid claims, as the GB caseload that controls them does, so
  the bridge's factor moves from 0.921 to about 1.001, and the bridge refuses a factor more than 2%
  from one (`max_factor_shift`), each grain read on its own input total and on the pass that
  carries the off-surface cells;
- twelve OBR table 4.9 lines are stamped Great Britain or England and Wales
  (`TARGET_ID_GEOGRAPHY_PINS`), and the `obr.state_pension` diagnostic reads its Great Britain
  fact; no value moved;
- the DESNZ vendor selections pin Great Britain (Northern Ireland's rows share its concepts), the
  local-authority one reads gas by `gas_year`, and every sub-national lookup names its period type
  (`energy_pricing.SUBNATIONAL_PERIOD_TYPES`, gas year Y with calendar year Y), so the 56 regional
  and 347 local-authority gas rows come back under new keys with the same values;
- the local energy rows keep Great Britain's share of UK households as the interim for its share of
  the Energy Trends level. The Northern Ireland facts are now in the feed, on a basis DESNZ calls
  not directly comparable, and María ruled on 2026-10-16 that the interim stands until a change
  stacked on this re-pin derives Great Britain's part from them.

Values move only for the UC area rows: each cell is its area's paid claims, where it was its
all-claims count scaled by the GB paid share, so the cells keep the GB caseload's total and each
takes its own area's paid share, as each child band takes its own. The membership reports show it:
every national reference resolves the fact it resolved before except the twelve restamped OBR
lines, and every local one except the 3,508 UC area cells and the 346 local-authority gas
consumption cells, which Chronicle restamped as gas years with the same values. The vendored
resources carry the same values, and the charter-H2 fixture moves twelve stage contract hashes. The
local compile-parity receipt is regenerated on this pin, and only the 3,508 UC cells' signed values
move; the full surface was not recompiled for this entry. The twelve restamped OBR lines move the
national register version with no value moved, so the uk-candidate-eval scoring register is
re-frozen before the next national solve. The high-value home stock still names `735820c`: its
TS054 tenure, OBR share and HPSSA facts are the same on `abf0921`, so it was not rebuilt.
