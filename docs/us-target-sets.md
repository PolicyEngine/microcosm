# US target sets

Every US release compiles the same target registry
(`us_runtime/fiscal_targets.py::compile_us_fiscal_target_registry`). Each
release then calibrates to a subset of it. Since microcosm#1728, that subset is
declared as data, for each product and mode, in
[`packages/microcosm-build/src/microcosm/build/us/target_sets.json`](../packages/microcosm-build/src/microcosm/build/us/target_sets.json).

- The shared reader and selector are in
  [`microcosm.calibrate.target_sets`](../packages/microcosm-calibrate/src/microcosm/calibrate/target_sets.py).
  They hold no country code.
- The US names (file, products, predicate, rule) are in
  [`microcosm.build.us_runtime.target_sets`](../packages/microcosm-build/src/microcosm/build/us_runtime/target_sets.py).
- The national tool (`tools/build_us_fiscal_refresh_release.py`,
  `_select_target_surface`) and the ACS local tool
  (`tools/build_us_acs_local_release.py`, `state_admin_surface` and
  `soi_surface_predicate`) select by the declarations.

The declarations replaced the selection code without a change to any target
surface (see "What the declarations select" below). The UK declares its target
sets the same way, with the same shared reader and selector
([UK target sets](uk-target-sets.md), microcosm#1759).

## The declaration

The file has one entry for each product:

| Product | Tool | Option | Default mode | Modes |
|---|---|---|---|---|
| `national` | `tools/build_us_fiscal_refresh_release.py` | `--target-surface` | `full` | `full`, `national_state` |
| `local` | `tools/build_us_acs_local_release.py` | `--soi-mode` | `state` | `state`, `totals`, `full`, `state_cd`, `state_county` |

Each mode states three things:

1. **Compiled clauses.** Each clause keeps the compiled targets of some
   families at some levels that pass its tests:
   - `families` and `levels`: lists of names, or `"*"` for any;
   - `metadata`: tests on metadata keys (`present`, `equals`, `not_equals`,
     `not_prefix`);
   - `exclude`: named predicates that drop a target (the national
     `congressional_district` classifier);
   - `rule`: a named rule that takes the clause's targets and returns the
     targets it keeps (the local `state_cd_district_shares` reconciliation).

   The selection is each clause's targets in declaration order, and the
   compile order inside a clause. The local `--families` option turns
   clauses on by id (`snap`, `medicaid`, `soi`).
2. **Built rows.** The rows a tool makes outside the compile: the population
   totals from the PUMA ladder (`--geographies`), and in `state_county` the
   county SOI shares and the ACS household rows from the county facts. The
   selector does not make them. The declaration names their family and
   levels, so that it states the whole target set.
3. **Rationale.** Each clause and built entry says why the product keeps
   those rows, as the UK membership files
   (`uk/target_reference_membership.json`,
   `uk/local_target_reference_membership.json`) give a signed rationale for
   each exclusion.

A target's family is its `TargetSpec.family`. Its level comes from the level
rule that the file declares, in the form the UK file uses:

```json
"level_rule": {
  "keys": ["ledger_geography_level", "geography_level"],
  "values": "*",
  "conflict": "first"
}
```

- The level is the first of the two keys that has a value. A compiled target
  names its level in `ledger_geography_level`. A row that a tool builds from
  the PUMA ladder or the county facts names it in `geography_level`.
- The rule is open (`"values": "*"`): it accepts any level, and a target with
  neither key has the level `unspecified`.

The target-loss rows read the same rule
(`us_runtime.target_loss_weights.target_geography_level`, microcosm#1729), so
a target has one family and one level everywhere. The US levels are
`country`, `state`, `congressional_district` and `county`. On the pinned feed
every compiled target sets `ledger_geography_level` only, so the two keys of a
compiled target cannot disagree; `test_us_target_sets.py` holds that.

## What the declarations select

On the pinned `41b2fb2` feed. The counts and the `TargetRegistry` versions
(an order-sensitive content hash) are the same as on main `0f963ac77`, before
the declarations:

| Product | Mode | Targets | Registry version |
|---|---|---:|---|
| `national` | `full` | 31,748 | `4cc4cfb7a147` |
| `national` | `national_state` | 6,102 | `4dfe25b1238a` |
| `local` | `state` | 4,380 compiled + 487 population = **4,867** | `0ee16ca42be5` (compiled), `69b952ef181f` (with population) |
| `local` | `totals` | 255 compiled | `c6a0f757c0bc` |
| `local` | `full` | 29,977 compiled | `b6c71f2e8505` |
| `local` | `state_cd` | 26,425 compiled | `e0bb302e3add` |
| `local` | `state_county` | 4,380 compiled, then the county rows | `0ee16ca42be5` (compiled) |

The 487 population rows are 51 state and 436 district totals from the
ladder with SHA256 `39a2ab2a…` (`--geographies state,cd`).
`packages/microcosm-build/tests/engine_free/us/test_us_target_sets.py` pins
all of these. The pinned-feed tests run when `MICROCOSM_US_CHRONICLE_FACTS`
names the feed, and the population test also needs
`MICROCOSM_US_PUMA_LADDER_ARTIFACT`.

### The state rows of the two production surfaces

The production national builds use `national_state`. The production local
builds use `state`. Their state rows, by family:

| Family | `national` `national_state` | `local` `state` |
|---|---:|---:|
| `irs_soi` | 4,227 | 4,227 |
| `usda_snap` | 102 | 102 |
| `cms_medicaid` | 132 | 51 (enrollment only) |
| `census_population` | 918 (51 states × 18 age bands) | 51 (ladder totals, built) |
| `cms_aca` | 102 | — |
| `ssa` | 102 | — |
| `state_income_tax` | 44 | — |
| `hhs_acf_tanf` | 29 | — |
| **State rows** | **5,656** | **4,431** |

`national_state` also has 446 country rows. `local` `state` has none, and it
has 436 district population rows. Whether the local build fits the national
rows and the full state set is decision 1 of microcosm#1728.

## How to change a target set

1. Change the mode's clauses or built entries in `target_sets.json`. The tools
   read the file at run time.
2. Run `test_us_target_sets.py` with the pinned feed. A change that moves a
   surface changes its pinned count and version: update them, and say which
   rows moved in the pull request.
3. A change that moves a surface changes what the calibration fits, so a
   build must measure it (`docs/us-release-build-rule.md`).

A family or level that a feed re-pin adds enters a mode only if a clause names
it:

- The national modes use `"*"` for families and levels. `full` keeps every
  compiled target. `national_state` keeps every compiled target that the
  congressional-district classifier does not drop.
- The local clauses name their families and levels. The SNAP and Medicaid
  clauses keep state rows. The SOI clause keeps state rows in `state` and
  `state_county`, and state and district rows in `totals`, `full` and
  `state_cd`.

Before the declarations, the local SNAP, Medicaid, `totals`, `full` and
`state_cd` filters did not test the level: they kept every row that carries
`state_fips`. On the pinned feed every such row is at a declared level, so no
surface changes. On a feed with such rows at another level (for example county
rows in the main feed), the declarations keep them out until a clause names
the level.

### Rows left out by level

A clause that lists levels drops the rows of its families at every other
level. The selection counts those rows by family and level
(`TargetSetSelection.left_out_by_level`), and the local tool writes the counts
in `materialize_rss.json`, under `target_levels_left_out`, beside the SOI
receipt. So a row that a mode does not keep because of its level leaves a
record. On the pinned feed:

| Family | Level | Rows | Left out by |
|---|---|---:|---|
| `usda_snap` | `country` | 2 | every local mode |
| `cms_medicaid` | `country` | 3 | every local mode |
| `irs_soi` | `country` | 441 | every local mode |
| `irs_soi` | `congressional_district` | 22,996 | `state` and `state_county` |

`test_us_target_sets.py` pins these counts for each mode. After a feed re-pin,
a row of one of the three families at a new level (for example a county SNAP
row) is a new line in the record and a failure of that test, which runs with
the pinned feed. The national modes list no level, so they leave no row out
by level.

The SOI clause of `state` and `state_county` reads a row's level as every
clause does: `ledger_geography_level`, else `geography_level`. The filter it
replaced read only `ledger_geography_level`. So a state SOI row that carries
only `geography_level` now enters these two modes. Every compiled row carries
`ledger_geography_level`, so no compiled registry changes.

A declaration file is read strictly: a repeated JSON key, a non-finite number,
`"*"` inside a list of names and a version that is not the integer are
refused.

## Rows the local tool builds, and its hold-outs

`us/local_target_construction.json` declares how the ACS local release builds
the targets that the compile does not give it. Shared, country-free modules
read each section (microcosm#1728):

| Section | Shared module | What the section declares |
|---|---|---|
| `holdouts` | `microcosm.build.target_holdouts` | The district pass (`--soi-mode state_cd`) and the county pass (`state_county`): which records each pass writes (`scope`), which get a held unit (`eligible`), the unit key (`unit_fields`, with the concept-family rule), the salt, the default and largest fraction, the receipt keys, and how the held rows are scored against the pro-rata baseline. |
| `ladder_population` | `microcosm.build.ladder_population_targets` | The PUMA ladder's state and district levels: the target names, the household column of each level, the code fields of each record, the geography ids and the label lookup of each level. |

The tool functions keep their names (`assign_target_roles`,
`calibration_target_set`, `score_cd_holdout`, `population_rows`,
`population_target_specs` and others) and call the shared modules. A record
has at least `name`, `value`, `family` and `geography_level`. The role pass
adds `holdout_unit` and `role`, and the training target set holds the
training rows only.

### Refusals

The targets, roles and receipts are the same as before the move. A refusal is
still a `ValueError`: the shared modules raise `TargetHoldoutError` and
`LadderPopulationError`, which are subclasses. The texts are the replaced
ones, and a declaration gives the words that name a level (`child_label` and
`parent_label` of a pass's scoring, `code_name` of a ladder level). One shared
check now serves both passes, so three cases differ from the replaced code:

- The county pass's fraction refusal also names the value, as the district
  pass's always did (`..., got 0.51.`).
- The district pass checks its salt before it reads a record, as the county
  pass always did, and refuses with `CD holdout salt must be a non-empty
  string.` The replaced district code left the salt to the hash of the first
  eligible unit.
- The county scorer reads `role` on every record, as the district scorer
  always did. The replaced county scorer skipped a record with no role.

The command line checks the fractions and the salts before any of these, and
the district pass gives every record a role, so no run of the tool reaches a
changed case. `fraction=None` and `salt=None` are refused: only an omitted
argument draws with the declared value. `test_us_target_holdouts.py` and
`test_us_ladder_population.py` compare each refusal's text with the replaced
code and pin the three differences.

Two parts of the local surface are still tool code. The district SOI shares
of `state_cd` (`tools/us_acs_local_cd_surface.py`) and the county rung
(`tools/us_acs_local_county_surface.py`) make child rows as shares of parent
rows. One nesting rule across all levels will replace them, together with the
district rows of microcosm#1731.

## The compile registers

`us/fiscal_target_registers.json` holds the reviewed registers of the US
fiscal target compile (`us_runtime/fiscal_targets.py` and
`us_runtime/source_vintage.py`). The shared
`microcosm.build.declared_registers` reads them. Each register keeps the
comments that stood beside it in the code, as its `description` and as the
`note` of an entry.

| Register | What it holds |
|---|---|
| `support_exclusions`, `all_vintage_support_exclusions`, `exclusion_vintage_bypasses` | The facts the compile drops, with the reason for each, and their vintage scope. |
| `source_column_exclusions` | The IRS columns a package reads under a measure whose concept they do not carry (microcosm#1038). |
| `restamped_source_packages`, `later_period_observation_exemptions` | The source packages whose period stamp is wrong (microcosm#1030). |
| `soi_amount_measure_variables`, `soi_return_measure_variables`, `soi_variables`, `soi_base_variable_overrides` | The model variable of each IRS SOI measure and concept. |
| `direct_targets`, `indicator_targets` | The model variable, family and metadata of each direct and indicator Ledger target. |
| `m_chip_state_fips` | The M-CHIP states whose CHIP rows never calibrate (microcosm#321). |

A section has a `key` (one field, or a list of fields for a tuple key), a
`value` (one field, a list of fields for a tuple value, or `null` for a set)
and its `entries` in order. JSON arrays read back as tuples.

To change a register:

1. Edit its entries. Give each new entry its reason or note.
2. Run `test_us_fiscal_target_registers.py`. It pins each register's digest
   and, with the pinned feed, the compile under four flag sets and the
   exclusion receipt. An intended change moves these pins: update them and
   say which targets moved.
3. A change that moves the compile changes what the calibration fits, so a
   build must measure it.
