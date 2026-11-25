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
surface (see "What the declarations select" below).

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

A target's family is its `TargetSpec.family`. Its level is its
`ledger_geography_level`, else its `geography_level`. These are the labels
that the shared target-loss rule reads (microcosm#1729), so a target has one
family and one level everywhere. The US levels are `country`, `state`,
`congressional_district` and `county`.

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
