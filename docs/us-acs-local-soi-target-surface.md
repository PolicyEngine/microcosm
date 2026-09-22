# US ACS local-area SOI target surface

`tools/build_us_acs_local_release.py --stage materialize` calibrates the ACS
local-area artifact to a state-level administrative surface: USDA SNAP,
CMS Medicaid enrollment and IRS SOI. `--soi-mode` decides how much of the
SOI part it keeps.

| `--soi-mode` | What it keeps | Status |
|---|---|---|
| `totals` | State-level `irs_soi` specs whose `target_role` is not `soi_fiscal_distribution` | Default |
| `full` | Every state-level `irs_soi` spec, `soi_fiscal_distribution` included | Explicit opt-in |

"State-level" means the spec's metadata carries `state_fips`, which the
congressional-district SOI rows also carry. The rule lives in
`soi_surface_predicate`; `state_admin_specs` applies it after the production
compile (`compile_us_fiscal_target_registry(age_targets=True)` and the RI
Medicaid substitution). An unknown mode is refused before the feed is read.

Totals became the default on 2026-09-22, by the maintainer ruling on
question 3 of #969's report (`experiments/us-labelled-filter-support/REPORT.md`
on that branch, sections 6-8). Before that, the parser default and the
`state_admin_specs` default were both `full`. The national release does
not use this switch. `tools/build_us_fiscal_refresh_release.py` has no SOI
mode and does not call `state_admin_specs`.

## The two surfaces on the pinned feed

These counts were measured on 2026-09-22 by calling `state_admin_specs(feed,
["snap", "medicaid", "soi"], soi_mode=...)` at `origin/main` `63897ede9`.
The feed was the pinned `consumer_facts_us_c5e5bf8.jsonl`, whose sha256
`b8543739…` matches `facts_sha256` in
`packages/microcosm-build/src/microcosm/build/us/chronicle_feed.json`.
They equal the 760 and 31,066 in #969's committed receipt.

| Family | `totals` | `full` |
|---|---:|---:|
| `usda_snap` | 102 | 102 |
| `cms_medicaid` (enrollment) | 51 | 51 |
| `irs_soi`, `soi_fiscal_distribution` | 0 | 30,306 |
| `irs_soi`, other roles | 607 | 607 |
| **Admin specs** | **760** | **31,066** |

The population marginals (state and congressional district) are added on
top of these in both modes.

### What `totals` contains, and what it does not

All 607 SOI specs that `totals` keeps are ACA premium tax credit (PTC)
measures:

| Measure | Role | Specs |
|---|---|---:|
| Congressional-district PTC returns (`congressional_district_2022`, current districts) | `aca_ptc_returns` | 436 |
| State-total PTC returns (`congressional_district_2022`, `<st>_total`) | `aca_ptc_returns` | 51 |
| State PTC returns (`historic_table_2.state_broad`) | `aca_ptc_returns` | 51 |
| State PTC amount (`historic_table_2.state_broad`) | `aca_spending` | 51 |
| District PTC returns (`historic_table_2.state_broad`, `current_cd`) | `aca_ptc_returns` | 9 |
| District PTC amount (`historic_table_2.state_broad`, `current_cd`) | `aca_spending` | 9 |

The source explains why. `_soi_target_role` in
`packages/microcosm-build/src/microcosm/build/us_runtime/fiscal_targets.py`
sets each SOI spec's `target_role`. It gives named roles
(`federal_income_tax_total`, `eitc_total` and so on) only to country-level,
all-income-range facts. The other exceptions are the all-income-range PTC
measures and the W-2 tip items. Every other SOI fact gets
`soi_fiscal_distribution`, whether or not it is an AGI band.

So the role `full` adds is mostly not AGI bands. On the same feed, the
30,306 `soi_fiscal_distribution` specs split by `ledger_geography_level` and
by whether `agi_lower_bound` or `agi_upper_bound` is finite:

| Geography | Income range | Geographies | SOI variables | Specs |
|---|---|---:|---:|---:|
| Congressional district | All (`income_range` `all`) | 436 | 25 | 23,886 |
| State | All (`income_range` `all`) | 51 | 28 | 5,508 |
| State | AGI band (for example `1_to_10k`, `500k_plus`) | 51 | 1 (`taxable_interest_income`) | 912 |

Both all-income-range groups include `adjusted_gross_income`, `income_tax`
and `eitc` rows (`eitc` alone is 1,020 state and 4,360 district specs).
**`totals` therefore carries no state or district AGI, income-tax or EITC
total.** `totals` names the role filter; it does not describe the rows that
filter keeps. The rule that drops only the 912 AGI-band slices, or only the
district rows, would be a new mode. This change does not add one; see the
sizes below.

### Matrix size

This section is arithmetic from the code, not a memory measurement.
`materialize_chunked` allocates one float32 column per admin spec for every
household (`np.memmap(..., dtype=np.float32, shape=(n_households,
len(names)))`), and `write_lean_checkpoint` copies that memmap into memory
(`np.array(admin_matrix)`). At the 1,588,854 households #969's report used,
4 bytes x households x specs gives:

| Surface | Admin specs | Dense float32 admin matrix |
|---|---:|---:|
| `totals` (default) | 760 | 4.5 GiB |
| `totals` plus state all-income-range rows (no such mode) | 6,268 | 37.1 GiB |
| `full` minus the state AGI-band slices (no such mode) | 30,154 | 178.5 GiB |
| `full` | 31,066 | 183.9 GiB |

The district all-income-range rows drive the size, not the AGI bands. The
on-disk memmap and the in-memory copy are each that size. Choose `full` only
on a machine sized for both, or after the matrix representation changes.

## Where the chosen mode is recorded

- `materialize_rss.json` in the checkpoint directory (`soi_mode`), written by
  `--stage materialize`. Later stages never re-select targets; they read this
  file, whatever `--soi-mode` they are given.
- `gate_summary.json`, under `gates.calibration.calibrated_surface.soi_mode`
  (finalize), which `build_manifest.json` also embeds under `gates`.
- `build_manifest.json`, under `materialize.soi_mode` (package).
- The `refresh_recipe.release` command in `build_manifest.json` and
  `release_manifest.json`, which now names `--soi-mode <mode>`. Re-running
  it reproduces the recorded surface even if the default changes again.

`--stage package` refuses a checkpoint whose `materialize_rss.json` does not
record `totals` or `full`. It refuses before any release directory exists.
Every checkpoint this tool has written records the mode.

## Operating notes

- The default command now calibrates to `totals`. To reproduce an earlier
  build that ran under the old default, pass `--soi-mode full`.
- A checkpoint materialized before this change under the old default records
  `full`, and packages as `full`.
- `--soi-mode` only matters when `soi` is in `--families`. Without it no SOI
  spec is selected, and the recorded mode has no effect on the surface.
