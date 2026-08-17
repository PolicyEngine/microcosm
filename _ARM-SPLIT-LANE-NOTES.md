# Phase P arm-split lane notes

## Setup and fixed authority (2026-08-17)

- Worktree `/Users/maxghenis/PolicyEngine/_worktrees/microcosm-arm-split` was
  clean on `arm-split-base` at `f71731202b6ff1f0dfa757b6b52aa87ff2a34161`.
- `uv sync --all-packages --extra us` resolved the 118-package lock but the
  sandbox could not reach PyPI. No network operation was attempted afterward.
  The complete lock-identical environment from the read-only predictor-sets
  worktree was copied locally and its five editable paths were rebound to this
  worktree. The lock and root `pyproject.toml` are byte-identical between
  `f7173120` and the rejected branch.
- Comparator: `/Users/maxghenis/PolicyEngine/_buildo-runtime/out/stacked-f025-r1/`,
  stacked identity
  `9be8ecdf82356f38998e8b620ee36d9134f554fe89a8eafd8406f438e2b5aad6`.
- The gate retains round 1's fold seed `20260816`, forest seed `260816`,
  bootstrap seed `8162026`, 2,000 replicates, 256 clusters, five quantiles,
  `[0.8, 1.25]` ratio band, and all forest/missingness settings. Only each
  candidate artifact's authority and target-bank receipt digests may differ.
- Builds are off-chain: no `--logbook-prev-row-digest`, the
  `POPULACE_LOGBOOK_PREV_ROW_DIGEST` environment variable is removed, and the
  global `logbook-pending-chain.txt` is never read or written.

## Round-1 decomposition boundary

The rejected bundle changed more than target order and the two named early
predictor classes. It also made every late transfer execute with the eight-field
PUF tax-detail predictor set. Baseline runtime calls the late transfer without
an explicit predictor set (`packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:8420`
at `f7173120`), while `a6893634` passes the new declaration. The round-1 verdict
confirms the executed difference in every late row under
`.oos_pinball_loss[].{baseline,candidate}.predictor_set`; 14 of its 21 money
regressions were late transfers. None of Max's three arm definitions authorizes
that late-PUF swap, so every arm retains the live baseline late-transfer
surface. The synthesis must not assign those 14 bundled regressions to one of
the three early interventions.

## Eligibility observability audit

Read-only source artifact:
`/Users/maxghenis/PolicyEngine/_worktrees/microcosm-predictor-sets/_buildo-runtime/out/stacked-f025-predictor-sets-r1/populace_us_2024_stacked_pool.checkpoints/stacked/5c1c042b4b9e65e9a4114d2c3c850ebdda0933dd48cf5dfa97212164bc4b7305/assembled.checkpoint.h5`.
The ACS recipient mask is `person_support_channel == "acs"` and contains
856,626 rows.

| Raw field | Observed ACS rows | Null ACS rows | Qualifies |
|---|---:|---:|---:|
| `MIL` | 708,870 | 147,756 | no |
| `DEAR` | 856,626 | 0 | yes |
| `DEYE` | 856,626 | 0 | yes |
| `DDRS` | 819,036 | 37,590 | no |
| `DPHY` | 819,036 | 37,590 | no |
| `DREM` | 819,036 | 37,590 | no |
| `DOUT` | 729,029 | 127,597 | no |
| `HINS3` | 856,626 | 0 | yes |

Therefore the eligibility arm admits exactly difficulty-hearing (`DEAR`),
difficulty-seeing (`DEYE`), and own Medicare coverage (`HINS3`). No veteran
predictor is fully observed. The MIL-only dtype harmonization from `cf4530b2`
is not needed in that minimal arm.

## Arm records

Arm implementation, test, build, and gate receipts are appended on the branch
that owns each arm.

### `arm-order`

- Scope: reorder only the early
  `person/model_required_boolean` participation chain.  The declared sequence
  puts `is_blind` and `is_disabled` first and CHAMPVA third
  (`packages/microcosm-build/src/microcosm/build/us_runtime/multispine_pool.py:637`),
  and the reorder is applied only while constructing the pre-clone gap-fill
  plan (`packages/microcosm-build/src/microcosm/build/us_runtime/multispine_pool.py:797`).
- Isolation: the baseline three required plus seven optional person predictors
  remain unchanged
  (`packages/microcosm-build/src/microcosm/build/us_runtime/acs_transfer.py:101`
  and `:159`); no loader or runtime transfer call changed.  In particular, the
  late transfer still receives no explicit predictor set
  (`packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:8437`).
  A receipt replay against the held baseline found the same 13 early family
  keys with only this target order changed, and the live authority comparison
  changed only the `gap_fill_plan` component.
- Mirror/gate support: `specs/us_imputation_lineage.yaml` follows the reordered
  early family and the live baseline ACS predictor surface for all late
  families.  The #695 conformance tests cover both.  The copied gate was
  initially byte-identical to round 1 at SHA-256
  `25295af61fece0e124a2658eed469ce1b5ecf9abd0c8e6b622807df8d4d69fe8`;
  only candidate receipt bindings will be refreshed after the build.
- Pre-build verification: `python -m pytest` passed 6,239 tests with 68 skips
  in 3,580.97 seconds; `python -m ruff check .` and `git diff --check` passed.
  The targeted transfer, lineage, and participation-order tests also passed.
- Guard incident (excluded from evidence): a 20:21:21Z launch used an invalid
  reclaimable-memory calculation instead of literal `vm_stat` `Pages free`.
  The attempted stop left its detached build child alive; this was discovered
  from the unchanged supervisor start record plus live `lsof` writers.  A
  reversible write barrier on only that arm-local checkpoint tree forced the
  orphan to exit at 23:30:54Z, before any artifact was accepted.  Its complete
  3.6 GiB output directory was removed.  The first clean post-containment guard
  read 4,094,042 free 16-KiB pages (62.47 GiB) at 23:31:29Z, below the fixed
  5,570,560-page (85 GiB) threshold.  Every reported build and gate below is
  from a later clean launch that passed the literal-free-pages guard.
