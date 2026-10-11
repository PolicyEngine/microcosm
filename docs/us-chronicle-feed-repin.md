# Re-pin the US Chronicle consumer feed

One reviewed declaration, `us/chronicle_feed.json`, names the Chronicle
commit, the scope file and the feed digest the US fiscal target registry
compiles from. `us/chronicle_feed_scope.json` names, for every (record set,
period) pair the feed keeps, the Chronicle source package that emits it and
the year to build that package with. `tools/build_us_chronicle_feed.py`
rebuilds the feed from those two, and two runs at the same commit produce the
same bytes. The target-parity resources (`us/target_parity_manifest.json`,
`us/target_parity_feed_families.json`) restate the feed digest;
`tools/build_us_target_parity_manifest.py` refuses to regenerate them against
any other feed, and `test_us_chronicle_feed.py` fails if the parity resources,
the generator and the pin disagree, or if the scope file changes without a
new pin.

## The 2026-10-11 re-pin: BEA private group health insurance (microcosm#454)

Decision d1317 made BEA NIPA Table 7.8 line 17 (series B4923C) the scale
anchor of the `meps_esi_premiums` stage (`docs/us-esi-employer-premiums.md`)
and asked for the feed to carry the banked fact. The pin moved from `51aa40f`
to `cf9ece2`, a commit that is `51aa40f` plus two commits (Chronicle branch
`feed/us-51aa40f-esi-b4923c`, tag `microcosm-us-feed-454-v1`), not Chronicle
main:

- The digit-text guard match from Chronicle main's `aa3e4de3`, and nothing
  else from that commit. BEA's Section 7 workbook writes its year headers as
  text (`'2024'`), and at `51aa40f` no guard could match them. Only a guard
  that used to fail can pass, so every package that built before builds the
  same cells.
- The `bea-nipa-private-group-health-insurance` package as
  PolicyEngine/chronicle#331 merged it to main, with its concept evidence
  (BEA's estimates cover active and retired employees). The test file drops
  main's artifact-year restamp test, whose guard (chronicle#292) postdates
  this base, and the package is not added to the US poverty hard-target
  coverage contract, as on main.

The scope adds three pairs, `bea_nipa.cy<year>.private_group_health_insurance`
for 2023, 2024 and 2025, each built at its own year (640 pairs, 66 package
runs). Built at `cf9ece2`:

- the feed keeps all 40,178 `51aa40f` rows byte for byte (joined on
  `aggregate_fact_key`) and adds exactly the three B4923C facts: $923,195M,
  $977,034M and $1,027,929M, each from `Section7All_xls.xlsx` (sha256
  `de1c34e3…`);
- compiled at 2024 with the packaged CD crosswalk and the Medicaid
  substitutions, with target aging on and again with it off, the registry is
  the same ordered list of targets on both feeds, name, value and metadata
  (33,250 targets with aging on). B4923C is not compiled into a target: the stage scales to it,
  and whether to calibrate the column is decision d1264. The target-parity
  manifest lists it as a `deferred` reviewed exclusion (82 feed families, 32
  compiled, 53 reviewed exclusions; it was a source-absent family before);
- two independent rebuilds gave the same `facts_sha256`.

The comparison scripts and their reports are in
`experiments/us-esi-454/feed-repin/`. The release tool now refuses a feed
whose B4923C values differ from the stage's pinned anchor
(`us_esi_premiums_anchor_feed_mismatches`); a diagnostic run under
`--allow-unpinned-feed` or `--allow-esi-premium-gaps` continues and records
the difference in `esi_premiums.json`. A later re-pin that changes other rows
(microcosm#1192 re-dates the W-2 item facts) must carry the three B4923C
pairs and redo the row comparison against its own previous feed.

## The 2026-10-04 re-pin: TY2023 state AGI bands (microcosm#940)

The state x AGI-band targets of #940 need the TY2023 Historic Table 2 state
file, whose $500k-$1M and $1M+ classes are separate facts
(PolicyEngine/chronicle#291). The pin moved from `c5e5bf8` to `51aa40f`, a
commit that is `c5e5bf8` plus only that package (Chronicle branch
`feed/us-c5e5bf8-state-agi-2023`, tag `microcosm-us-feed-940-v1`), not to
Chronicle main:

- At Chronicle main the artifact-year restamp guard (chronicle#292) stops
  building the TY2022 and TY2020 files at `--year 2023`, so ten scoped pairs
  (the congressional-district file, the W-2 tips and 401(k) items, the IRA
  tables, the `state_2022` US rows) produce no row, and chronicle#304 renames
  the Historic Table 2 and district-file capital-gains concepts. Adopting
  those is microcosm#1030's re-pin and needs its own decisions.
- Built at `51aa40f`, the feed keeps all 39,158 `c5e5bf8` cells with equal
  values and no changed field, and adds exactly the 1,020 TY2023 state AGI
  cells (`irs_soi.ty2023.historic_table_2.state_agi.<st>`, 51 new scope
  pairs). Compiled at 2024 with the packaged CD crosswalk, aging and the
  Medicaid substitutions, every target other than the state AGI bands is
  identical on both feeds, name, value and metadata; the bands go from 306
  TY2022 rows to 408 TY2023 rows. The comparison script and its report are in
  `experiments/940-state-agi-bands/feed-repin/`.
- A later re-pin to Chronicle main carries the 51 pairs over: chronicle#291
  adds the same package to main.

## Why the feed moved (2026-09-18)

`_validate_chronicle_hierarchy_labels`
(`packages/microcosm-build/src/microcosm/build/ledger_targets.py`) requires
exactly one Chronicle-owned label for every dimension a target selects on and
refuses to substitute the identifier. The previous pin,
`consumer_facts_buildn_v9_4.jsonl` (`b3c08356…`, cut 23 July 2026), carries
no `dimension_labels`, so target compilation on main refused it. Chronicle
main has written the labels since its #267 (14 September 2026); this pin is
the first labelled US export.

## The pin

| Field | Value |
|---|---|
| Chronicle commit | `cf9ece2e7be11158c21fdc4dd38b44d92db552f6` (`51aa40f` plus the digit-text guard and the B4923C package; tag `microcosm-us-feed-454-v1`) |
| Feed file | `consumer_facts_us_cf9ece2.jsonl`, 40,181 rows, 169,055,585 bytes |
| `facts_sha256` | `054e5684005a7b143dcb84eabac634135ce9b2b33bb57b540602e474a4ba7c57` |
| Consumer fact schema | `chronicle.consumer_fact.v3`, schema file sha256 `bdb51e2a…` (unchanged from the UK pin) |
| Scope | 640 (record set, period) pairs; 66 package runs over build years 2020 to 2029 |
| Consumer artifact | none: refused at this commit, see below |

The previous pins were `51aa40fd28e2ca81937752515e773d123fecfb71`
(`consumer_facts_us_51aa40f.jsonl`, 40,178 rows, `facts_sha256`
`965a29ac…`, 637 pairs, 63 runs) and, before it,
`c5e5bf8aa84960c1a200ee47303b19c953092d0f` (`consumer_facts_us_c5e5bf8.jsonl`,
39,158 rows, `facts_sha256` `b8543739…`, 586 pairs, 62 runs); the rest of this
document describes the 2026-09-18 rebuild unless it says otherwise.

The feed is too large for the repository. Its home on the build machine is
`~/PolicyEngine/_buildh-runtime/inputs/consumer_facts_us_cf9ece2.jsonl`,
beside the previous pins; `tools/build_us_target_parity_manifest.py` reads it
there by default. A holder of the Chronicle commit regenerates it byte for
byte with the commands below.

## Scope rule

The feed keeps exactly the 586 (record set, period) pairs of the previous
pin, so the family surface the release parity gate reviews does not move in
this change: 32 compiled families and 52 reviewed exclusions before and after,
the same 81 feed families. Widening the feed to Chronicle's newer US packages
is a separate, deliberate change with its own compile-or-fence review per
family. Scoping by record set alone is wrong: it pulls extra Medicaid months
into the snapshot families.

`chronicle build-bundle` takes one `--year`, and some packages are
year-specific (IRS SOI Table 1.4 emits one tax year per run) while others
are year-independent (BEA NIPA emits the same rows for every year). The scope
file records one build year per pair: the pair's own period year when the
package emits the pair for that year, otherwise the earliest export year
that emits it. `experiments/us-chronicle-feed-repin/derive_scope.py` is the
one-time derivation, from the previous pin's pairs and the September
exports; the builder verifies every choice again and refuses a run that does
not emit its pair.

Whole-year bundles are not used: at this commit `build-bundle --year 2020`
and `--year 2021` exit 1 because `census-population-projections-2023` fails
its row criteria for those years, and a whole year takes about twenty
minutes where a targeted package run takes one or two seconds.

## Rebuild

```bash
uv run python tools/build_us_chronicle_feed.py --chronicle-root ~/PolicyEngine/chronicle --out /tmp/us-feed --replace --skip-artifact
```

The checkout must be clean at the scope's commit. The tool runs one
`chronicle build-bundle --year Y --source <package> …` per build year (ten
runs, 62 packages), reads each package's own `consumer_facts.jsonl`, keeps a
row only from the run its pair is scoped to, refuses a pair with no row or
two rows with one `aggregate_fact_key` and different bytes, sorts by
`aggregate_fact_key`, and writes `consumer_facts.jsonl` plus `receipt.json`
(commands, row count, digests). Measured 18 September 2026: 7 minutes 35
seconds; two independent runs gave the same `facts_sha256`, and `cmp` found
the files byte-identical. The same digest came out of the September scratch
assembly from whole-year bundles, so the targeted runs reproduce what the
whole bundles emit.

Then copy the feed to its home, regenerate the parity resources and run the
parity tests:

```bash
uv run python tools/build_us_target_parity_manifest.py
```

```bash
uv run pytest packages/microcosm-build/tests/engine_free/shared/test_release_target_parity.py packages/microcosm-build/tests/engine_free/us/test_us_chronicle_feed.py
```

## What moved and what did not

Fact keys were re-derived between Chronicle generations (334 of 37,405
`aggregate_fact_key`s match), so the comparison joins on what identifies a
cell in the source table: record set, period, `layout.source_row_id` and
`layout.source_column_id`
(`experiments/us-chronicle-feed-repin/value_diff.py`, report in
`value_diff.json`).

- Every one of the previous pin's 37,399 cells is in the new feed with the
  same value: 37,399 shared, 37,399 equal, 0 different, 0 only in the
  previous pin.
- The previous pin carried 6 duplicate cells (`cbo.revenue_projection.ty2023`
  income by source, each twice); the new feed carries each once. That is why
  `cbo.revenue_projection` counts 24 rows where it counted 30.
- The new feed adds 1,759 cells the previous pin lacked, all in record sets
  the pin already had: IRS SOI Table 1.4 for tax years 2020, 2021 and 2022
  (578, 540 and 523), Table 1.1 for 2022 (76), Medicaid state enrollment for
  2024-12 and 2025-12 (20 each), W-2 Social Security tips for 2023 (2). Five
  feed-family row counts move accordingly: `irs_soi.table_1_4` 679 to 2,320,
  `irs_soi.table_1_1` 84 to 160, `cms_medicaid.state_enrollment` 475 to 515,
  `irs_soi.form_w2_social_security_tips` 4 to 6, `cbo.revenue_projection` 30
  to 24.
- The 15 `semantic_fact_key` matches whose values differed in an earlier
  comparison (`irs_soi.ty2022.historic_table_2.us`) are duplicate semantic
  keys, not revisions: on the cell join no shared value differs.
- Every row carries `dimension_labels`; the previous pin carried none.

Compiling the new feed through `compile_us_fiscal_target_registry(
target_period=2024, age_targets=True, packaged CD vintage crosswalk)` and
`apply_us_medicaid_enrollment_substitutions` gives 32,867 targets in 32
families in 11.9 seconds, every target with a hierarchy, against the July
register's 32,842. The previous pin cannot compile on main, so the two
registers are not compared target by target; the only input difference is
the added rows above, since every shared cell is equal.

**Erratum (23 September 2026, microcosm#956).** The added rows did move the
target surface. Two reviewed fences reopened, and the route A release
(`8f63bf000`) failed its zero-support and SOI Table 1.4 gates on the rows
that leaked:

- Twenty of the added Medicaid cells per month are direct
  `total_chip_enrollment` rows for exactly the twenty M-CHIP states that #321
  fences. The fence lived only in the combined-minus-Medicaid CHIP
  derivation, so the 2024-12 rows compiled as `chip_enrolled` targets (the
  2025-12 rows fall after the 2024 target period).
- The Table 1.4 cells for tax years 2020-2022 bypassed #564. Its four
  `other_income` exclusions were keyed to the ty2023 ids, so latest-vintage
  selection calibrated the ty2022 rows in their place.

The fix drops every CMS CHIP row for an M-CHIP state after latest-vintage
selection, scopes the #564 entries to every vintage, and makes the compile
refuse any other fallback to another vintage of an excluded cell unless the
id is a reviewed bypass (`US_FISCAL_TARGET_EXCLUSION_VINTAGE_BYPASSES`).
`us_source_coverage.json` now records the concrete ids each rule drops or
allows (`fiscal_target_exclusion_receipt`). The one reviewed bypass was the
ty2020 W-2 Box 7 tips return count, which falls back past the #451 ty2023
exclusion. The route A remediation plan found it calibrated in the certified
parent `populace-us-2024-spm-receipts-20260923` too, and the committed
incumbent scorecard shows it at -52%
(`experiments/replacement_scorecard/incumbent_48b9d479.md`), so it is not a
re-pin effect. Decision d179 (25 September) ruled to enforce #451 at every
vintage: the ty2023 entry joined `US_FISCAL_TARGET_ALL_VINTAGE_SUPPORT_EXCLUSIONS`,
and the bypass register is now empty.

On this feed, compiled as above and then narrowed with
`--target-surface national_state`, the surface diff is 24 targets removed,
none added and none changed: the 2024-12 CHIP rows for AK CA DC HI IL KY MD
ME MI MN NC ND NE NH NM OH OK SC VT WY, and the ty2022
`table_1_4.all.other_income_net_{income,loss}_{amount,returns}` rows. The
compiled register goes from 32,867 to 32,843 targets, and the
`national_state` surface from 5,719 targets (registry `d5f9d854fe11`) to
5,695 (`386fac439e77`). Decision d179 then removed the ty2020 tips return
count, the only change: 32,842 compiled targets and 5,694 in
`national_state` (`d315c75804ef`).

The labelled feed exposed one latent defect in microcosm, fixed in this
change: `apply_us_medicaid_enrollment_substitutions` built Rhode Island's
substituted spec by cloning a neighbouring state's spec and kept that state's
hierarchy, so the substituted `TargetSpec` refused with a hierarchy whose
target id was not its own. `_substituted_hierarchy` now derives the
hierarchy from the template with the substituted state's geography, label
and target id, and the register entry carries a reviewed `state_name`
(`_geography_fallback_label` is UK-only, so the label cannot be derived from
the FIPS code).

## The consumer artifact is refused at this commit

`chronicle build-consumer-artifact` validates every row against
`consumer_fact.v3`, whose `concept_alignment` object has required `authority`
since Chronicle `cafc583`. `build-bundle` at `c5e5bf8` emits 994 rows whose
`concept_alignment` has `relation: source_label` and no `authority`, in three
packages: `cms-medicaid-chip-monthly-enrollment-dataset` (515),
`census-b01001-female-age-2023` (468) and `jct-tax-expenditures-2024` (11).
So the artifact step refuses (`Consumer fact row 228 … failed schema
validation at 'concept_alignment': 'authority' is a required property`), and
this pin is a bare feed: `manifest_sha256` and `artifact_schema_version` are
null, and `load_us_chronicle_feed().is_bare_feed` is true.

What that means for releases: the `--base-h5` arm of
`tools/build_us_fiscal_refresh_release.py` accepts a bare
`consumer_facts.jsonl` (`--ledger-facts` with `--ledger-facts-sha256`); the
`--exact-k` arm requires `--ledger-manifest-sha256` and therefore an
artifact directory, and cannot use this pin until Chronicle either records
an `authority` for those alignments or relaxes the schema for
`source_label` relations. That is a Chronicle decision, tracked as
[PolicyEngine/chronicle#277](https://github.com/PolicyEngine/chronicle/issues/277);
the builder fails closed without `--skip-artifact`, and a later pin at a
commit that fixes it records the manifest digest in the same declaration.
