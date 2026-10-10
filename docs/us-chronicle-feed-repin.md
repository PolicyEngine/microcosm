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

The pin has moved twice. September 2026 replaced an unlabelled feed with the
first labelled export (the next section onward). October 2026 moved only the
Form W-2 item rows, which the September feed stamped with the wrong tax year
("The W-2 item tables carry Tax Year 2020" below).

## Why the feed moved

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
| Chronicle commit | `f98acf4edcc9488343446fda46db77914de8611f` (tag `microcosm-us-feed-w2-ty2020-v1`): `c5e5bf8` plus one package file |
| Feed file | `consumer_facts_us_f98acf4.jsonl`, 39,155 rows, 164,591,409 bytes |
| `facts_sha256` | `a81cbcc504caa71e4eb23b4d70a11528c8e6db6523e41283f8482569a23df04c` |
| Consumer fact schema | `chronicle.consumer_fact.v3`, schema file sha256 `bdb51e2a…` (unchanged from the UK pin) |
| Scope | 585 (record set, period) pairs; 61 package runs over build years 2020 to 2029 |
| Consumer artifact | none: refused at this commit, see below |

The September pin was Chronicle `c5e5bf8aa84960c1a200ee47303b19c953092d0f`:
`consumer_facts_us_c5e5bf8.jsonl`, 39,158 rows, 164,603,204 bytes,
`facts_sha256` `b8543739…f4839801`, 586 pairs from 62 package runs. The
compile now refuses that feed (`_check_w2_item_fact_tax_years`).

The feed is too large for the repository. Its home on the build machine is
`~/PolicyEngine/_buildh-runtime/inputs/consumer_facts_us_f98acf4.jsonl`,
beside the previous pins; `tools/build_us_target_parity_manifest.py` reads it
there by default. A holder of the Chronicle commit regenerates it byte for
byte with the commands below.

## Scope rule

The September feed kept exactly the 586 (record set, period) pairs of the
previous pin, so the family surface the release parity gate reviews did not
move: 32 compiled families and 52 reviewed exclusions before and after,
the same 81 feed families. The October change swaps the three W-2 item pairs
stamped ty2023 for their TY2020 record sets, one of which the scope already
had, leaving 585 pairs and the same families. Widening the feed to Chronicle's newer US packages
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

The checkout must be clean at the scope's commit
(`git checkout microcosm-us-feed-w2-ty2020-v1`; the commit is not on
Chronicle main). The tool runs one
`chronicle build-bundle --year Y --source <package> …` per build year (ten
runs, 61 packages), reads each package's own `consumer_facts.jsonl`, keeps a
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

## The W-2 item tables carry Tax Year 2020

10 October 2026. The September feed carried five facts with period 2023 and
vintage `tax_year_2023` whose `source.source_file` is `20in04w2all.xlsx` and
whose `source.source_table` is "Table 4.B. Summary of Items for Taxpayers with
Form W-2, by Return and Earner Type, Tax Year 2020":

| `source_record_id` (as stamped) | Value | Workbook cell |
|---|---:|---|
| `irs_soi.ty2023.form_w2_social_security_tips.box_7_social_security_tips.amount` | 26,786,522,000 | D13 |
| `…form_w2_social_security_tips.box_7_social_security_tips.return_count` | 6,038,613 | B13 |
| `…form_w2_social_security_tips.box_7_social_security_tips.taxpayer_count` | 6,105,713 | C13 |
| `irs_soi.ty2023.form_w2_401k_elective_deferrals.box_12_d_401k_elective_deferrals.amount` | 277,859,181,000 | D22 |
| `irs_soi.ty2023.form_w2_designated_roth_401k_contributions.box_12_aa_designated_roth_401k_contributions.amount` | 32,302,509,000 | D41 |

The three tips rows repeated the feed's correctly stamped `irs_soi.ty2020`
tips facts cell for cell.

**The source.** Checked on 10 October 2026:

- IRS's Form W-2 statistics page
  (`irs.gov/statistics/soi-tax-stats-individual-information-return-form-w2-statistics`,
  last reviewed 27 January 2026) links Table 4 for 2020 and 2019 only.
  `21in04w2all.xlsx` to `24in04w2all.xlsx` return 404. There is no Table 4.B
  for Tax Year 2021, 2022 or 2023. The prior-releases page carries tax years
  2008 to 2018 in one workbook (`18inallw2.xls`), so 2020 is the latest year
  IRS has published.
- A fresh download of `20in04w2all.xlsx` has the sha256 Chronicle pins
  (`1178d776…cfc442f`). Cell A1 of sheet "Table 4.B" is the title above, and
  rows 13, 22 and 41 hold the five values (money amounts in thousands).

So every one of the five values is a Tax Year 2020 value.

**Why the feed stamped them 2023.** At Chronicle `c5e5bf8` the
`soi-w2-statistics-2020` package sets `artifact_year: 2020`, which makes every
build read that one workbook, but rendered its period, record ids, vintage and
legal vintage from `--year`. The scope built it at `--year 2023` for three
record sets, as the July feed's pairs said to. chronicle#292 (merged 28
September) made the labels literal 2020 on Chronicle main.

**The pin.** Chronicle `f98acf4` is `c5e5bf8` plus only that package file as
#292 left it (branch `feed/us-c5e5bf8-w2-2020`, tag
`microcosm-us-feed-w2-ty2020-v1`). The scope drops the three ty2023 W-2 pairs
and adds the TY2020 401(k) and Roth record sets; it already had the TY2020
tips record set. Against the September feed
(`experiments/us-w2-table-year-repin/`):

- 39,153 rows are byte-identical, the three `irs_soi.ty2020` tips rows among
  them;
- the five rows above are gone, and no row carries their ids (a deliberate
  supersession: the ids named a tax year IRS has not published);
- two rows are new, the 401(k) and Roth amounts under their `irs_soi.ty2020`
  ids, each with the value and the source cells of its ty2023 twin. The
  three ty2023 tips rows need no replacement: their TY2020 twins were
  already in the feed.

`tools/build_us_chronicle_feed.py` built the feed from the new scope, and an
independent splice (the September feed without the W-2 package's rows, plus
that package's rows at `f98acf4`, sorted by `aggregate_fact_key`) gives the
same bytes.

A pin on Chronicle main is a larger change. At main (`3dce46a6`) seven more of
the September scope's pairs are no longer declared: the congressional-district
file, the two IRA tables and the four `state_2022` US rows, which #292 gave
literal 2022 labels. Moving those changes the year every district dollar
target ages from; microcosm#1030 measures that and lists the decisions it
needs.

**What moved.** Compiled as the release does (target period 2024, district
crosswalk, aging on, Medicaid substitutions), before on main `7639ef8b0` with
the September feed and after on this change:

| | Before | After |
|---|---|---|
| Compiled targets | 32,842 | 32,842 |
| `national_state` targets | 5,694 | 5,694 |
| `national_state` registry | `d315c75804ef` | `61a081ae55bb` |
| W-2 Box 7 tips amount target | $28,280,884,269 | $34,287,530,779 |
| Its name | `irs_soi.ty2023.form_w2_social_security_tips.…amount` | `irs_soi.ty2020.form_w2_social_security_tips.…amount` |
| Its aging factor | 1.05578784246075 (CBO wages 2024 / 2023) | 1.28002921689246 (SOI Table 1.4 wages 2023 / 2020, then CBO wages 2024 / 2023) |

The tips target rises $6,006,646,510 (21.2%), the SOI wage growth from 2020 to
2023 that the ty2023 stamp skipped. It is the only target that changes: the
other 32,841 keep their name, value and every metadata field. Latest-vintage
selection had picked the ty2023 row over its TY2020 twin, so the target aged
one year where `target_aging.py` chains the wages series from 2020. The
September re-pin introduced this: it added the ty2023 tips amount and return
count (the row count of `irs_soi.form_w2_social_security_tips` went 4 to 6,
above; it is now 3). The July feed already carried the ty2023 401(k), Roth and
tips taxpayer-count rows, none of which compiles to a target. The Route A release `populace-us-2024-0fb05b6-4b57d15a287c-20260930T150401Z`
recorded the $28,280,884,269 target
(`tests/fixtures/us_route_a_target_loss_weights.json`).

The 401(k) and Roth amounts stay reviewed exclusions (`input_side`), so they
move no target. Anything that anchors on them later (microcosm#1191 names the
401(k) amount) is anchoring on a Tax Year 2020 level.

**The exclusion register.** The #451 tips return-count entry in
`US_FISCAL_TARGET_SUPPORT_EXCLUSIONS` and
`US_FISCAL_TARGET_ALL_VINTAGE_SUPPORT_EXCLUSIONS` is keyed to the ty2020 id.
Decision d179 is unchanged: the cell is excluded at every vintage. The receipt
lists one tips id where it listed two.

**The check.** `_check_w2_item_fact_tax_years` runs at the start of
`compile_us_fiscal_target_registry` and of
`us_fiscal_target_exclusion_receipt`. For a fact of the three W-2 item
families (`irs_soi.form_w2_item` layout) it requires the source table's title
to name exactly one tax year, and that year to be the year target aging reads
from the fact's period and the year in its `tax_year_<year>` vintage. It
refuses the September feed and names the five rows.
`test_us_w2_item_tax_year.py` holds the tests. Chronicle's
`tests/test_chronicle_source_table_year.py` checks the same thing at the
source for every pinned package whose title names one tax year.

The parity resources were regenerated: 32 compiled families, 52 reviewed
exclusions and 81 feed families, as before. `tools/route_a/route_a.sh` restates
the new `FEED_SHA`; `test_us_chronicle_feed.py` now fails if it falls behind
the pin.

## The consumer artifact is refused at this commit

`chronicle build-consumer-artifact` validates every row against
`consumer_fact.v3`, whose `concept_alignment` object has required `authority`
since Chronicle `cafc583`. `build-bundle` at `c5e5bf8`, and at `f98acf4`, which
differs from it in one W-2 package file, emits 994 rows whose
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
