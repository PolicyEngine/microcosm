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
| Chronicle commit | `c5e5bf8aa84960c1a200ee47303b19c953092d0f` |
| Feed file | `consumer_facts_us_c5e5bf8.jsonl`, 39,158 rows, 164,603,204 bytes |
| `facts_sha256` | `b85437390021777e746f507c5890305496baf5fc7f2c78ba08ddb090f4839801` |
| Consumer fact schema | `chronicle.consumer_fact.v3`, schema file sha256 `bdb51e2a…` (unchanged from the UK pin) |
| Scope | 586 (record set, period) pairs; 62 package runs over build years 2020 to 2029 |
| Consumer artifact | none: refused at this commit, see below |

The feed is too large for the repository. Its home on the build machine is
`~/PolicyEngine/_buildh-runtime/inputs/consumer_facts_us_c5e5bf8.jsonl`,
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
uv run pytest packages/microcosm-build/tests/test_release_target_parity.py packages/microcosm-build/tests/test_us_chronicle_feed.py
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
