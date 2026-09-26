# US ACS local-area SOI target surface

`tools/build_us_acs_local_release.py --stage materialize` calibrates the ACS
local-area artifact to a state-level administrative surface: USDA SNAP,
CMS Medicaid enrollment and IRS SOI. `--soi-mode` decides which SOI specs it
keeps.

| `--soi-mode` | What it keeps | Status |
|---|---|---|
| `state` | State-level `irs_soi` specs at `ledger_geography_level == "state"` whose `ledger_layout_record_set_spec_id` is not a congressional-district file, of any `target_role` | Default |
| `totals` | State-level `irs_soi` specs whose `target_role` is not `soi_fiscal_distribution` | Explicit opt-in |
| `full` | Every state-level `irs_soi` spec | Explicit opt-in |

"State-level" means the spec's metadata carries `state_fips`, which the
congressional-district SOI rows also carry. The rule lives in
`soi_surface_predicate`; `state_admin_specs` applies it after the production
compile (`compile_us_fiscal_target_registry(age_targets=True)` and the RI
Medicaid substitution). `state` does not select a spec that lacks either
`ledger_geography_level` or `ledger_layout_record_set_spec_id`, since it cannot
tell which contract such a spec belongs to. An unknown mode is refused before
the feed is read. The national release does not use this switch:
`tools/build_us_fiscal_refresh_release.py` has no SOI mode and does not call
`state_admin_specs`.

## Why `state` is the default

`state` is the SOI contract every earlier ACS local-area release was calibrated
to. Build O (`populace-us-2024-buildo-acs-local-77e2061-20260724T110908Z`)
selected it as `full` while `state_admin_specs` compiled with
`include_congressional_district_targets=False` (populace `77e2061`,
`tools/build_us_acs_local_release.py:151-154`). Build P's ACS local release
(`populace-us-2024-buildp-acs-local-592ae5d6-20260819T020303Z`) calibrated to
the same 4,459 targets. Commit `b7922b089` (2026-08-21, "compile one target
surface unconditionally") removed that switch, so `full` grew to include the
TY2023 congressional-district SOI file and no mode reproduced the historical
contract until `state`.

Max chose `state` as the default on 2026-09-22, after the measurements below
showed that `totals` carries no state AGI, income-tax or EITC total and that
`full` does not fit one 128 GB machine. It is the only surface that keeps the
state income targets and fits, and it lets a new local build be scored against
the published Build O on the same targets. Before that day the default was
`full`; for part of 2026-09-22, on this branch only, it was `totals`.

`state` depends on the restated-filter support of #969 (merged 2026-09-22):
without it the materializer's Ledger-filter guard refuses 1,014 of the
surface's specs.

## The surfaces on the pinned feed

Measured on 2026-09-22 by calling `state_admin_specs(feed, ["snap",
"medicaid", "soi"], soi_mode=...)` on the pinned
`consumer_facts_us_c5e5bf8.jsonl` (sha256 `b8543739…`, the `facts_sha256` in
`packages/microcosm-build/src/microcosm/build/us/chronicle_feed.json`). The
Chronicle `b571381` consumer artifact gives the same counts.

| Family | `state` | `totals` | `full` |
|---|---:|---:|---:|
| `usda_snap` | 102 | 102 | 102 |
| `cms_medicaid` (enrollment) | 51 | 51 | 51 |
| `irs_soi` | 3,819 | 607 | 30,913 |
| **Admin specs** | **3,972** | **760** | **31,066** |

The 487 population marginals (51 states and 436 congressional districts) are
added on top in every mode, so `state` calibrates to 4,459 targets: Build P's
set, and Build O's 4,461 minus the Vermont under-$1 taxable-interest pair the
compiler now excludes as unsupported (`US_FISCAL_TARGET_SUPPORT_EXCLUSIONS` in
`us_runtime/fiscal_targets.py`).

### What `state` contains

All 3,819 SOI specs sit at state geography and come from the three TY2022
Historic Table 2 state tables:

| Record set spec | Specs | Content |
|---|---:|---|
| `irs_soi.historic_table_2.state_broad_totals.v1` | 2,397 | 47 all-income-range measures x 51 states, including AGI, income tax, and ACA premium tax credit returns and amounts |
| `irs_soi.historic_table_2.state_agi_counts_and_amounts.v1` | 912 | taxable interest by AGI band |
| `irs_soi.historic_table_2.state_eitc.v1` | 510 | EITC returns and amounts by number of qualifying children |

It holds no congressional-district SOI row and no row from the TY2023
congressional-district file. The feed-gated test
`test_pinned_feed_soi_surfaces_match_their_contracts` pins these counts.

### What `totals` contains, and what it does not

All 607 SOI specs that `totals` keeps are ACA premium tax credit measures (547
returns, 60 amounts), 454 of them at congressional-district geography.
`_soi_target_role` in `us_runtime/fiscal_targets.py` gives named roles only to
country-level all-income-range facts, the all-income-range PTC measures and
the W-2 tip items; every other SOI fact gets `soi_fiscal_distribution`,
whether or not it is an AGI band. **`totals` therefore carries no state or
district AGI, income-tax or EITC total.**

### What `full` adds

`full`'s 30,306 `soi_fiscal_distribution` specs split by
`ledger_geography_level` and by whether an AGI bound is finite:

| Geography | Income range | Geographies | SOI variables | Specs |
|---|---|---:|---:|---:|
| Congressional district | All | 436 | 25 | 23,886 |
| State | All | 51 | 28 | 5,508 |
| State | AGI band | 51 | 1 (`taxable_interest_income`) | 912 |

Its state all-income-range rows include the same state concepts twice, once
from the TY2022 Historic Table 2 and once from the TY2023 congressional-district
file (`<st>_total` rows); the two copies disagree by a few percent after both
are aged to 2024. Its district rows reconcile to their state parents in the
same file. Using them needs one vintage per state concept and a sparse target
matrix.

### Matrix size

This is arithmetic from the code, not a memory measurement.
`materialize_chunked` allocates one float32 column per admin spec for every
household (`np.memmap(..., dtype=np.float32, shape=(n_households,
len(names)))`), and `write_lean_checkpoint` copies that memmap into memory
(`np.array(admin_matrix)`). At 1,588,854 households, 4 bytes x households x
specs gives:

| Surface | Admin specs | Dense float32 admin matrix |
|---|---:|---:|
| `totals` | 760 | 4.5 GiB |
| `state` (default) | 3,972 | 23.5 GiB |
| `full` | 31,066 | 183.9 GiB |

Measured peaks for comparison: Build P's ACS local release, on this surface,
recorded a 93.9 GB materialize peak (`build_manifest.json` →
`materialize.peak_rss_gb`); the 2026-09-22 hours rebuild on `totals` peaked at
75.6 GB in materialize. `full` does not fit a 128 GB machine as a dense matrix.

## Where the chosen mode is recorded

- `materialize_rss.json` in the checkpoint directory (`soi_mode`), written by
  `--stage materialize`. Later stages never re-select targets; they read this
  file, whatever `--soi-mode` they are given.
- `gate_summary.json`, under `gates.calibration.calibrated_surface.soi_mode`
  (finalize), which `build_manifest.json` also embeds under `gates`.
- `build_manifest.json`, under `materialize.soi_mode` (package).
- The `refresh_recipe.release` command in `build_manifest.json` and
  `release_manifest.json`, which names `--soi-mode <mode>`, so re-running it
  reproduces the recorded surface even if the default changes again.

`--stage package` refuses a checkpoint whose `materialize_rss.json` does not
record one of the three modes, before any release directory exists.

## Operating notes

- The default command calibrates to `state`. Raise the supervisor's RSS cap
  above Build P's 93.9 GB peak before a full-scale run, and leave room on disk
  for the 23.5 GiB memmap.
- To reproduce the 2026-09-22 hours rebuild (#974), pass `--soi-mode totals`.
  To reproduce a build that ran as `full` after `b7922b089`, pass
  `--soi-mode full`.
- A checkpoint materialized under an earlier default records its own mode and
  packages as that mode.
- `--soi-mode` only matters when `soi` is in `--families`.
