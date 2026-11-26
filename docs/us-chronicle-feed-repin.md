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

## The county facts pin (microcosm#922)

The ACS-local county rung (`tools/us_acs_local_county_surface.py`) reads
county-grain facts: IRS SOI county returns and AGI for tax year 2022, Census
PEP county population vintage 2024, and the ACS 2020-2024 5-year county tables
B01003, B11001, B19001, B19013, B19025, B19051 and B19061. They have their own
pin, `us/chronicle_county_facts.json`, with scope
`us/chronicle_county_facts_scope.json` (scope name `us_county_facts`), beside
the main pin. No fiscal compile reads them, and the main feed, its scope and
the target-parity resources do not move when they do.

They are separate because of size. The ACS county rows keep both estimate and
margin-of-error rows in one record set, about 5.2 KB per row, and the eight
ACS record sets come to 170,896 rows and 886 MB. The whole county file is
922 MB; the main feed at `41b2fb2` is 171.7 MB. Folding the county facts into
the main feed would grow it to about 1.09 GB (6.4x; 6.2x for the ACS rows
alone). Measured at the compile and parse sites, that would raise a compile's
peak memory from 2.8 GB to 6.7 GB and keep about 3.2 GB more resident in every
fiscal-refresh run, which holds the parsed feed for its whole length, while
only county builds read the rows. A main-feed rebuild with them also
exceeded the build machine's 3.5 GB folder cap. These were measured on the build
machine on 2026-11-13; the run notes are not in this repository.

- **Commit.** Chronicle main `3e9d3a0694e8ca7e31ed5319c3ceb1bde92b12e0`
  (merged 2026-11-12), which carries every county package. It is on main, so
  it needs no branch or tag; the next main re-pin starts at or after it.
  The rows are on the `chronicle.consumer_fact.v5` contract
  (schema digest `acd33a09…`); the main pin stays on v3.
- **Pairs.** 10 (record set, period) pairs from nine packages, built at their
  own years (2022: `packages/irs_soi/county_2022`; 2024:
  `packages/census/pep_county_2024` and the seven
  `packages/census/acs5_b*_county_2024` packages; B01003 has two record sets). The scopes share no pair,
  and `test_us_chronicle_county_facts.py` refuses a change that adds one to
  both.
- **Build.** The same builder with the county scope:

  ```bash
  uv run python tools/build_us_chronicle_feed.py --scope packages/microcosm-build/src/microcosm/build/us/chronicle_county_facts_scope.json --chronicle-root <clean Chronicle checkout at the scope's commit> --chronicle-command "uv run --frozen --python 3.13 chronicle" --out <dir> --skip-artifact
  ```

  `--python 3.13` matters on a fresh checkout: without it, `uv` can pick a
  newer Python for which Chronicle's locked wheels do not exist. A rebuild
  that differs from the pin is refused by the pin's digest, not accepted.

  The build at that commit wrote 180,326 rows (922 MB), SHA256
  `67b0998aa2ee1aeed6637dab6a846f405246706508074b1c1b5bab7bd98c8bea`. Its
  output folder peaked at 2.89 GB under a disk-cap watcher that deleted each
  finished package's intermediates. Nine of its ten record sets matched an
  earlier build at Chronicle `c32c0b0` row for row, keying each row by record
  set, period type, period value, source row and source column and comparing
  value, unit, measure and geography; PEP county was not in that build.
  `c32c0b0` is on no branch of the Chronicle remote and that build's output
  has been deleted, so the comparison cannot be re-run. Rebuilding at
  `3e9d3a0` with the command above and comparing the file's SHA256 with the
  pin's can.
- **Reading it.** `us_runtime/county_facts.load_us_county_facts` first hashes
  the file and refuses it unless its digest and row count are the pin's, so a
  wrong file is named as such before any row is parsed. A second pass hashes
  the bytes again while parsing; it refuses a row that is not a JSON object,
  is on another row contract, is at a geography level other than county or is
  outside the scope's pairs, and a file that changed between the passes. It
  drops the 78 Puerto Rico municipios (state 72) with a count and projects
  each kept row to the fields the rung reads. The file on disk keeps
  Chronicle's bytes.
- **Tool flags.** County runs pass `--county-facts <file>
  --county-facts-sha256 <pin digest>`, and `--feed-sha256`. The county digest
  must be the pin's. `--feed-sha256` is only the operator's digest of the
  `--feed` file; the tool checks the file against it, not against the main
  pin.
  The run identity records the main feed's digest (`feed_sha256`) and the
  county facts receipt, which carries the county pin.
- **Where the file lives.** Not in git. The repository holds only the pin:
  the file's digest and row count (the file is 922,268,386 bytes) and the scope. An operator builds the
  file with the command above at the pinned commit, or copies a build
  machine's copy, and passes its path with `--county-facts`; the loader's
  digest check is what makes either acceptable.
- **What it holds that the rung must handle.** SOI county 2022 has 3,143
  counties and no Kalawao (15005). ACS B19013 (median household income) has no
  row for De Baca County (35011) and is a median, so it is a diagnostic, not a
  linear target. B01003 carries margin-of-error rows for only some counties.
  Connecticut appears as its nine planning regions (09110-09190) in every
  package. Against the main pin's state SOI parents, the SOI county rows split
  the other 49 states and DC into 6,278 county targets whose shares sum to one
  and whose targets sum to their state parents. The raw county sums are not
  the parents: county AGI falls short of the state parent by up to 5%
  (Montana), and returns differ by at most 0.05%. The rung uses the county
  rows only as shares, so it rescales them to the parents. Hawaii is refused,
  naming 15005, until the rung declares a treatment for it.

A later main re-pin can fold these pairs into the main scope only by retiring
the county pin in the same change.

The two pins move together. The main feed is pinned at `41b2fb2` and the
county facts at `3e9d3a0`; the next main re-pin starts at or after
`3e9d3a0`. From then on, a change that re-pins the main feed at a Chronicle
commit also rebuilds the county facts at that commit and re-pins them, so a
US build reads one Chronicle base.

## A re-pin onto a newer row contract

The US feed is on `chronicle.consumer_fact.v3`; Chronicle `main` writes
`chronicle.consumer_fact.v5` since PolicyEngine/chronicle#377. The US pin
loader refuses a pin whose `consumer_fact_schema_versions` names a row contract
that `microcosm.build.chronicle_epoch` does not declare (microcosm#1509). The
pin describes a bare feed, so the full feed-pin check cannot run on it;
instead the fiscal-refresh release tool and `tools/build_us_target_parity_manifest.py`,
after their digest checks, refuse rows on an undeclared row contract and, on
the committed feed, rows whose contracts are not exactly the pin's list
(`require_us_chronicle_feed_row_contracts`). So set the list from the rebuilt
feed's rows at a re-pin: `tools/build_us_chronicle_feed.py` does not write it,
and the parity generator, which the rebuild runs, refuses a stale list. v5 is
declared, and its schema digest is `acd33a09…`.

v5 rows may carry `package_dimension_value_labels`, a package's own wording of
a value where it differs from Chronicle's label. The congressional-district
vintage translation (`us_runtime/congressional_district_vintage.py`:
`_translated_fact_shell` and `_state_total_proxy_source_district_fact`) copies
each source row whole and rewrites only the geography, the layout and the
group-by entry of `dimension_value_labels`. So after a re-pin onto v5, a
derived district row keeps any package wording keyed by the source row's value
ids (the 118th-Congress district, or the state row's value), not the derived
row's. No code reads the field, so no label, value or refusal changes; v4's
`geography.publisher_name` is copied the same way.

## The 2026-11-03 re-pin: Federal Student Aid loan portfolio tables (microcosm#1074)

microcosm#1357 binds Federal Student Aid's loan portfolio tables as
calibration targets in builds that carry the imputed federal student loan
balances (`--scf-education-debt`). Its compile reads them from the pinned
feed, so the feed has to carry them. PolicyEngine/chronicle#347 added the
six `packages/ed_fsa` packages to Chronicle main. The pin moved from
`a9026cc` to `41b2fb2b864a28c0eda03322b77a97ef147dfa3c` (Chronicle branch
`feed/us-a9026cc-ed-fsa-1103`, tag `microcosm-us-feed-1074-v1`), which is
`a9026cc` plus a cherry-pick of chronicle#347's commit `4448a7c96` and
nothing else, not Chronicle main. That commit also adds a Census ACS S1501
state package and changes `chronicle/source_package.py` and
`chronicle/sources/rows.py`; the scope leaves the ACS package out.

The scope adds 66 pairs, one per (record set, period) the six packages emit:
each record set has one quarter-end month, 2025-12, 2026-03 or 2026-06. All
66 are built at 2026, the only year the packages' manifests carry a file
for, so 2026 is the earliest emitting year for the 2025-12 pairs and the own
year for the rest. The scope has 706 pairs and 72 package runs; the six
packages join the 2026 run.

| Package | Pairs | Rows |
|---|---:|---:|
| `packages/ed_fsa/dl_portfolio_by_repayment_plan_fy2026_q3` | 36 | 227 |
| `packages/ed_fsa/portfolio_by_age_fy2026_q2` | 4 | 24 |
| `packages/ed_fsa/portfolio_by_debt_size_fy2026_q2` | 4 | 36 |
| `packages/ed_fsa/portfolio_by_loan_type_fy2026_q3` | 12 | 42 |
| `packages/ed_fsa/portfolio_by_location_fy2026_q2` | 1 | 108 |
| `packages/ed_fsa/portfolio_summary_fy2026_q3` | 9 | 24 |

Built at `41b2fb2b` with `tools/build_us_chronicle_feed.py --skip-artifact`
(40,642 rows, `facts_sha256` `1d45a89c…`), and compared with the `a9026cc`
feed by `experiments/us-feed-agi-age-labels/compare_feeds.py` (report in
`experiments/us-feed-ed-fsa/compare.json`; the added rows, with their
source, in `added_rows.json`):

- the feed keeps all 40,181 `a9026cc` rows: the same keys, and no field of
  any of them changes. Dropping the 66 new pairs' rows leaves #1343's feed
  byte for byte and in order (`4d89e4a4…`);
- it adds 461 rows, every one from source `ed_fsa`, in the 66 new pairs;
- compiled at 2024 as the release does (packaged CD crosswalk, age targets,
  Medicaid substitutions), both feeds give the same 33,250 targets in the
  same order, with no changed field, the same registry version
  (`b8d571e91c22`) and the same `national_state` surface (6,102 targets,
  `3aa2a0830e7d`). No `ed_fsa` fact compiles by default: microcosm#1357's
  `student_loan_targets=True` binds 136 of the 461, and nothing else binds
  any;
- the parity resources change in the feed name and digest, and the feed
  carries 11 new families (`ed_fsa.*`, 93 feed families). The generator
  declares them `off_by_default` reviewed exclusions under one fence for the
  `ed_fsa` namespace, so the counts are 32 compiled and 64 reviewed
  exclusions. The feed's label variants are unchanged (only `age` =
  `age_20` and `age_21`).

**The schema stamp stays v3.** Every row, old and new, carries
`chronicle.consumer_fact.v3`: the exporter at this commit stamps every row
with one constant (`CONSUMER_FACT_SCHEMA_VERSION`,
`chronicle/consumer_contract.py:35`, the last entry of
`CONSUMER_FACT_ROW_CONTRACTS`, `chronicle/epoch.py:171-176`), and it ends at
v3. The session that cut the commit found that chronicle#347's export at its
merge commit `acbedb5` stamps the same rows v4, that v4 entered Chronicle
main in `0ec21e5`, which would restamp every feed row v4, and that apart from
the stamp all 461 `ed_fsa` rows are that export byte for byte (Chronicle
branch `hub-notes/ed-fsa-feed-cut-1103`, `notes/ed-fsa-feed-cut-1103/NOTES.md`).
The 244 rows of microcosm#1357's test fixture match this feed's rows the
same way.
So the pin keeps `consumer_fact_schema_versions: ["chronicle.consumer_fact.v3"]`,
and `consumer_fact_schema_sha256` stays `bdb51e2a…`: Chronicle's
`policyengine_chronicle/schemas/consumer_fact.v3.schema.json`, the file
whose bytes that field hashes (`policyengine_chronicle/schema.py:59-70`), is
identical at `a9026cc` and `41b2fb2b`. The US loader carries a row's stamp
without checking it (`ledger_artifact.py`, `_load_fact_rows`); since
microcosm#1509 the release tool and the target-parity tool check the stamps
against the pin (see "A re-pin onto a newer row contract").

The B4923C anchor test follows the committed re-pin reports to the pin
(`test_us_esi_premiums.py`). A report may now add rows when it names a
listing of the keys it adds, the one source they all come from and the
record-set namespace they all sit in (`ed_fsa.` here); added rows from any
other source or in any other namespace, or rows added without a listing,
still fail it. The test declares the namespace rather than reading it from
the source, because a source's record sets need not sit in a namespace of
its name.

The feed's home on the build machine is
`~/PolicyEngine/_buildh-runtime/inputs/consumer_facts_us_41b2fb2.jsonl`.

## The 2026-10-27 re-pin: one label per dimension value (microcosm#1343)

The `cf9ece2` feed labelled three dimension values two ways, so compiled
targets that select the same value id showed it under two names. Each value
now carries the one wording microcosm#1343 chose:

| Dimension = value | Before (package) | After |
|---|---|---|
| `us:statutes/26/62#adjusted_gross_income` = `under_1` | "No adjusted gross income and deficit" (`soi-filing-season-week47-2024-eitc-total`, the IRS column header); "No adjusted gross income" (seven other IRS SOI packages) | "No adjusted gross income" |
| `us:statutes/26/62#adjusted_gross_income` = `all` | "All returns" (`soi-historic-table-2`, `soi-table-1-1`); "All returns, total" (`soi-table-1-2`, `-1-4`, `-2-1`, `-4-3`) | "All returns, total" |
| `age` = `85_plus` | "Age 85 and over" (`census-pep-2024-national-age-sex`); "Age 85 plus" (`census-pep-2024-state-age-sex`) | "Age 85 plus" |

The pin moved from `cf9ece2` to `a9026cc`, a commit that is `cf9ece2` plus
three label-only commits (Chronicle branch `feed/us-cf9ece2-agi-age-labels`,
tag `microcosm-us-feed-1343-v1`), not Chronicle main. The three commits were
first written on `51aa40f` (branch `feed/us-51aa40f-agi-age-labels`, head
`9b91e8b`, never pinned on main); after microcosm#1189 moved the pin to
`cf9ece2`, they were cherry-picked onto it without conflicts, since they and
`cf9ece2`'s two commits change different files:

- the week-47 change of PolicyEngine/chronicle#377 (`6d74e42`), without the
  `build-bundle` label check that PR adds;
- `soi-historic-table-2` and `soi-table-1-1` relabel their `all` rows. Table
  1.1's row had no `expected_row_header`, so its row-header check compared the
  label with the IRS text in cell A10; the row now names that text
  ("All returns") explicitly, so the check is unchanged. Chronicle's tests
  that pin Table 1.1's `record_set_spec_hash`, the checked-in consumer fact
  sample built from its first rows, and that sample's pinned digest move
  with it; at `a9026cc` they pass unchanged from the `51aa40f` branch;
- `census-pep-2024-national-age-sex` relabels its `85_plus` row.

Every package's checks of the publisher's text (row and column headers,
guard cells, the single-year age sequence) are unchanged.
`soi-historic-table-2-state-broad-2022` keeps "All returns" for `state` =
`all` (chronicle#267): that is the `state` dimension, which no other feed
package labels. The feed still labels `age` = `age_20` and `age_21` two ways
(Census population projections and ACS B01001), but no compiled target
selects those values; they are outside this change.

Built at `a9026cc` from the same 640 scope pairs (only the scope's
`source_commit` changes), and compared with the `cf9ece2` feed by
`experiments/us-feed-agi-age-labels/compare_feeds.py` (report in
`compare.json`):

- the feed keeps the same 40,181 `aggregate_fact_key`s and every value,
  key and constraint, including the three B4923C facts. 816 rows change, in
  six record sets, and only in label fields and `layout.record_set_spec_hash`,
  which covers row labels: the value label and `layout.groupby_value_label`
  on 66 rows (2 `under_1`, 63 `all`, 1 `85_plus`), and the hash on every row
  of the six record sets;
- compiled at 2024 as the release does (packaged CD crosswalk, age targets,
  Medicaid substitutions), both feeds give the same 33,250 targets in the
  same order with the same values. 50 targets change, each only in its
  hierarchy dimension's `value_label` and its
  `metadata["ledger_layout_groupby_value_label"]`: 2 `under_1`, 47 `all`,
  1 `85_plus`. The registry version moves from `dde00e855af8` to
  `b8d571e91c22` and the `national_state` surface (6,102 targets) from
  `3fad14f85468` to `3aa2a0830e7d`, because the version hashes the
  hierarchy labels;
- no dimension value in the compiled targets carries two labels any more;
- the parity resources change in the feed name and digest, and the
  feed-family inventory gains `dimension_value_label_variants`: every
  dimension value the feed labels more than one way, with its labels. On
  this feed that is only `age` = `age_20` and `age_21`. The generator writes
  it from the feed, and `test_us_chronicle_feed.py` checks the committed
  listing in CI, where the label test above skips for want of the feed. A
  later re-pin built on a Chronicle commit without the three label commits
  lists those values again and fails there. The family counts are unchanged:
  32 compiled, 53 reviewed exclusions, 82 feed families;
- the same tool, run at `cf9ece2` with that commit's scope file, reproduced
  its `054e5684…` byte for byte.

These are the same row and target changes the label commits made on the
`51aa40f` feed (`965a29ac…` to `e477e04a…`, 21 October 2026).

Chronicle main (`272c987`, 25 October 2026) still has all three old
labels in these packages. PolicyEngine/chronicle#377 gives the three values
the same wording on main through its label register
(`chronicle/dimension_values/canonical_labels.json`); until it merges, a
re-pin onto main (#1030) must carry these labels or lose them.

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
the difference in `esi_premiums.json` (in the report, for a dry run). A later re-pin that changes other rows
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
| Chronicle commit | `41b2fb2b864a28c0eda03322b77a97ef147dfa3c` (`a9026cc` plus chronicle#347's `4448a7c96`; branch `feed/us-a9026cc-ed-fsa-1103`, tag `microcosm-us-feed-1074-v1`) |
| Feed file | `consumer_facts_us_41b2fb2.jsonl`, 40,642 rows, 171,661,396 bytes |
| `facts_sha256` | `1d45a89c28177a95371b4cc4068d3657c7bf12bd8edad7c82fe25bf453f9ea07` |
| Consumer fact schema | `chronicle.consumer_fact.v3`, schema file sha256 `bdb51e2a…` (the same file at `a9026cc` and `41b2fb2b`) |
| Scope | 706 (record set, period) pairs; 72 package runs over build years 2020 to 2029 |
| Consumer artifact | none: refused at this commit, see below |

The previous pins were `a9026cc8f80307af0c4527fdb3779fab32ffad9f`
(`consumer_facts_us_a9026cc.jsonl`, 40,181 rows, 169,056,411 bytes,
`facts_sha256` `4d89e4a4…`, 640 pairs, 66 runs; tag
`microcosm-us-feed-1343-v1`), `cf9ece2e7be11158c21fdc4dd38b44d92db552f6`
(`consumer_facts_us_cf9ece2.jsonl`, 40,181 rows, 169,055,585 bytes,
`facts_sha256` `054e5684…`, the same 640 pairs and 66 runs; tag
`microcosm-us-feed-454-v1`), `51aa40fd28e2ca81937752515e773d123fecfb71`
(`consumer_facts_us_51aa40f.jsonl`, 40,178 rows, 169,038,757 bytes,
`facts_sha256` `965a29ac…`, 637 pairs, 63 runs; tag
`microcosm-us-feed-940-v1`) and, before it,
`c5e5bf8aa84960c1a200ee47303b19c953092d0f` (`consumer_facts_us_c5e5bf8.jsonl`,
39,158 rows, `facts_sha256` `b8543739…`, 586 pairs, 62 runs); the rest of
this document describes the 2026-09-18 rebuild unless it says otherwise.

The feed is too large for the repository. Its home on the build machine is
`~/PolicyEngine/_buildh-runtime/inputs/consumer_facts_us_41b2fb2.jsonl`,
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
uv run pytest packages/microcosm-build/tests/engine_free/shared/test_release_target_parity.py packages/microcosm-build/tests/engine_free/us/test_us_chronicle_feed.py packages/microcosm-build/tests/engine_free/us/test_us_eitc_table_2_5_irs_reconciliation.py
```

`test_us_eitc_table_2_5_irs_reconciliation.py` runs its EITC checks in CI on a
fixture that copies the feed's 274 Table 2.5 and filing-season EITC rows byte
for byte. Its last test fails when a re-pin changes any of those rows, and that
test's docstring says how to refresh the fixture.

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

**Erratum (27 September 2026, microcosm#1035).** Row order changed too. The
new feed is sorted by the re-derived `aggregate_fact_key`s, and the
capital-gains rebase kept the first of two equal-period controls. The
congressional-district US row (TY2022 data stamped ty2023) now sorted ahead of
Table 1.4 ty2023. It took over the returns control and raised the 51 Historic
Table 2 state return counts from 12,289,836 to 29,599,604. The control is now
chosen by concept and period alone
([us-soi-capital-gains-concepts.md](us-soi-capital-gains-concepts.md)). The
counts are unchanged and `national_state` is `65e4dde11c83` on this feed. On
the 2026-10-04 `51aa40f` re-pin, whose capital-gains cells are identical, the
same 60 return counts move and `national_state` goes from `e1f9906bf0e0` to
`882f4764a248` at 6,102 targets.

**Follow-up (microcosm#1038).** On the `full` surface, the
congressional-district file's capital-gains rows are shares of its US row
scaled to the same control, and the US row retires (#1040). The package's
four mislabeled columns are dropped (#1224): N18425/A18425 (state and local
income taxes), read as the limited SALT deduction, and N85530/A85530
(additional Medicare tax), read as the premium tax credit. On the `51aa40f`
feed that leaves 31,784 compiled targets, all of them on `full`
(`6e1ac91262bc`). `national_state` keeps its 6,102 targets and values and
moves from `882f4764a248` to `84504a7afd57` only because its 102 Historic
Table 2 capital-gains rows now name their share denominator in
`soi_share_total_source_record_id`.

**On the `41b2fb2` feed** (25 November 2026), with microcosm#1035, #1038 and
#1030 merged together, the release compile keeps 31,748 targets (`full`
`4cc4cfb7a147`, from 33,250 at `bf021cdc9749`), and `national_state` keeps
6,102 at `4dfe25b1238a` (from `c698e70d33ce`).

- Besides the 1,466 specs above, 36 Historic Table 2 SALT and
  premium-tax-credit rows leave `full`. The crosswalk copies a state row to an
  at-large district only for a measure that the district file carries. Once
  the mislabeled columns are dropped before the crosswalk, the district file
  carries no SALT or premium-tax-credit rows, so the copies of those measures
  for the nine at-large districts go too. Each copy repeated its state row.
- In `national_state`, 52 values move. The 51 Historic Table 2 state
  net-capital-gains return counts move ×0.415203: they sum to 12,289,836
  against the national Table 1.4 target of 12,392,020, from 29,599,604. The
  W-2 Box 7 tips amount moves from $28,280,884,269 to $34,287,530,779.

The labelled feed exposed one latent defect in microcosm, fixed in this
change: `apply_us_medicaid_enrollment_substitutions` built Rhode Island's
substituted spec by cloning a neighbouring state's spec and kept that state's
hierarchy, so the substituted `TargetSpec` refused with a hierarchy whose
target id was not its own. `_substituted_hierarchy` now derives the
hierarchy from the template with the substituted state's geography, label
and target id, and the register entry carries a reviewed `state_name`
(`_geography_fallback_label` is UK-only, so the label cannot be derived from
the FIPS code).

## Restamped packages

Five Chronicle packages in this feed restamp their data: 26,893 observation
facts carry a later period than their publisher file describes. A Chronicle
package can pin one file with `artifact.artifact_year` and still render
`{year}` into its period, record ids and vintage from the build year. The
scope builds these five at 2023, so each emits its pinned file as `ty2023`.
PolicyEngine/chronicle#117 reported the congressional-district case in July.
microcosm#1030 found the other four, and PolicyEngine/chronicle#292 (merged
2026-09-28) refuses these builds at source. The `51aa40f` pin is `c5e5bf8`
plus one package, so it still carries all five.

| Package | File | Data year | Facts stamped ty2023 |
|---|---|---|---|
| `soi-congressional-district-2022` | `22incd.csv` | TY2022 | 26,880 |
| `soi-w2-statistics-2020` | `20in04w2all.xlsx` | TY2020 | 5 |
| `soi-state-2022` | `22in54us.xlsx` | TY2022 | 4 |
| `soi-ira-roth-contributions-2022` | `22in06ira.xlsx` | TY2022 | 2 |
| `soi-ira-traditional-contributions-2022` | `22in05ira.xlsx` | TY2022 | 2 |

The data years come from the files themselves: the title cells ("Tax Year
2020", "Tax Year 2022"), cell-for-cell equality with the ty2020 W-2 twins,
and the TY2022 county file for the congressional-district data.

**When it started.** The previous pin already carried 26,891 of these facts:
- every congressional-district, state and IRA row;
- the ty2023 W-2 taxpayer count, 401(k) and Roth 401(k) rows.

The 2026-09-18 re-pin added the other two: the ty2023 tips amount and return
count. These are the "W-2 Social Security tips for 2023 (2)" cells listed
under [What moved and what did not](#what-moved-and-what-did-not). Before
them, the only tips amount was the ty2020 row, which would have aged on the
chained SOI wages bridge from 2020. The previous pin cannot compile on main,
so that is read from the code, not observed. The restamped row won
latest-vintage selection and aged on the direct CBO ratio from 2023,
reaching $28.28B at 2024 instead of the $34.29B it gets aged from its data
year.

**What the compile does** (`us_runtime/source_vintage.py`). Two rules detect a
restamp:
- the fact's `source.source_sha256` is a registered file and its period is
  after that file's data year;
- its `source.raw_r2_key` names an earlier year than its period.

Each detected fact must match a reviewed `US_RESTAMPED_SOURCE_PACKAGES` entry,
or the compile refuses. For matching specs, the readings that aging and the
period contract use move to the data year:
- `source_period`;
- `uprating_to_period`, for specs rebased onto a restamped control;
- `uprating_from_period`, for a restamped row rebased as a share of a
  restamped total of the same package, stamp and data year (the total its
  rebase names in `soi_share_total_source_record_id`).

A share's `uprating_to_period`, where its aging starts, stays at the control's
period. The stamp cancels in row / total, and the level and its period are
the control's. Starting the aging at the data year instead would apply the
control's growth from the data year to its own period twice: for the
congressional-district capital-gains shares that microcosm#1040 proposes,
x0.761. Any other rebased restamp refuses the compile: one with no named
total, a truthful total, or a total from another package, stamp or data year.

The stamp is kept in `source_vintage_stamped_period` (and, for a share's
reading period, `uprating_from_period_vintage_stamped_period`). Values and
target names do not change. Two readings still see the stamp, because moving them would
change which fact wins:
- **Latest-vintage selection.** The compile refuses where a restamp would
  outrank a truthful fact dated after its data year.
- **The choice of a rebase control.**

A restamped fact inside an aging growth index is refused outright.

The key-year rule is exact for single-year IRS files. A multi-year release
keyed by its publication year can carry later columns: BEA's 2024-keyed
`SAINC.zip` has a 2025 column. A truthful build of such a column needs a
`US_LATER_PERIOD_OBSERVATION_EXEMPTIONS` entry.

**On this feed** (`51aa40f`), compiled as the release does:

- **`national_state`** keeps 6,102 targets, and its registry moves from
  `e1f9906bf0e0` to `61f5cac06310`. On `c5e5bf8`, before the 408 state
  AGI-band rows, it moved from `d315c75804ef` to `884bc45ef335` at 5,694; every
  figure below is the same on both feeds.
  - One value moves: the W-2 Box 7 tips amount, from $28,280,884,269 to
    $34,287,530,779 (+21.2%).
  - 53 specs change metadata only: 51 Historic Table 2 net-capital-gains
    return counts rebased onto the congressional-district US row, and the two
    `state_2022` counts. Counts do not age.
  - The 51 rebased counts now read `uprating_from_period` and
    `uprating_to_period` 2022: a same-year rescale of TY2022 shares onto the
    TY2022 CD file's total.
- **Full surface.** 13,176 congressional-district-file dollar targets age from
  TY2022 instead of 2023.
  - The 23 measures that age on the CBO AGI series (AGI, income tax, the EITC
    amounts, taxable interest, dividends, pensions, SALT and the rest) move
    +3.05%.
  - Net capital gains move -23.9%, because the SOI Table 1.4 chain records
    their TY2022 to TY2023 fall.
  - Qualified dividends (+8.4%) and Schedule C and partnership income (-0.5%)
    move mostly because their series changes: with no SOI bridge for them, the
    aging model falls back to the AGI chain from 2022.
  - These are changes relative to aging from the data year under the current
    series policy, not realized growth. Taxable interest rose ×2.35 from
    TY2022 to TY2023 in Table 1.4, far above the AGI link (#117).
- **Metadata only.**
  - 13,614 count specs change `source_period`: 13,612 CD-file counts plus the
    two `state_2022` counts above.
  - 9 more rebased counts are CD-classified and full-surface only.
- **No target.** The 401(k), Roth 401(k), IRA, `state_2022` AGI and EITC
  amount rows compile none.

`test_pinned_feed_restamp_register_matches_the_feed` checks the register
against this feed in both directions. It runs where the feed is present and
skips in CI. A re-pin on a Chronicle commit that stamps truthfully leaves
entries that match nothing, and the test fails until they are deleted. At that
re-pin the scope moves these pairs to their data years (`ty2022`, and `ty2020`
for W-2). IRS has also published TY2023 Historic Table 2 and IRA tables, which
would give true `ty2023` facts.

Two things needed a decision before that re-pin:

- **The Historic Table 2 capital-gains returns control.** The CD US row won a
  same-period tie on feed order. Stamped truthfully, it would lose to Table
  1.4 ty2023, which counts returns with a taxable net gain (a different
  population), and the 51 `national_state` counts would fall 58.5%. The 51
  state counts summed to 2.39× the national Table 1.4 target of the same
  compiled variable. Since microcosm#1036 (decision d468) the control is
  chosen by concept: no restamped row is a capital-gains control, and the
  60 control corrections above are gone.
- **`SOI_CONGRESSIONAL_DISTRICT_RECORD_SET_ID`**, which names the `ty2023`
  record set. This one is still open.

One mis-stamp runs the other way and is out of reach of both rules.
`bea-regional-state-personal-income-components-2024` reads the 2024 column
under a `cy2023` stamp (416 facts). No target compiles from it, but
`bea_regional.state_wages_salaries` is a deferred parity family. Activating it
before chronicle#292 would calibrate 2024 wages as cy2023.

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
