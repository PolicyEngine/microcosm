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

### `arm-eligibility` surgery (2026-08-17)

- The branch starts from `arm-split-base` at
  `aec9fdc1cf71eb33019fe3e0e425926573544afb`. The early target declarations
  retain their baseline order, including CHAMPVA at position 0.
- The observability audit above admits exactly the three recipient-complete
  fields: ACS `DEAR` -> `difficulty_hearing`, `DEYE` ->
  `difficulty_seeing`, and `HINS3` -> `has_medicare_coverage`. The donor
  analogues are the reviewed ASEC `PEDISEAR`, `PEDISEYE`, and `MCARE` codecs;
  blanks remain null and unsupported codes fail closed. Both donor and
  recipient mappings are bound into the transfer execution identity.
- `puf_tax_itemization` and `model_required_numeric` receive all three
  predictors. `model_required_boolean` receives own Medicare coverage only;
  the difficulty flags are excluded there to avoid direct disability-target
  leakage. No other early family receives an eligibility predictor.
- The explicit early sets retain the realized baseline predictors and exclude
  investment, which is unavailable on the pre-clone donor. The late-transfer
  runtime remains the baseline call with no explicit predictor sets. The
  #695 mirror declares these exact live early sets and the unchanged live late
  shared ACS-transfer surface. Stacked authority version 11 binds the new
  per-family predictor declarations into the `gap_fill_plan` component.
- The copied round-1 gate is byte-identical to the rejected-branch source and
  the order arm: SHA-256
  `25295af61fece0e124a2658eed469ce1b5ecf9abd0c8e6b622807df8d4d69fe8`.
- The copied virtualenv's `pytest` console script retained a stale shebang to
  the read-only predictor-sets worktree. Tests therefore run through
  `UV_CACHE_DIR=$PWD/.uv-cache uv run python -m pytest`, whose imports resolve
  to this worktree; the analogous Ruff command is `uv run python -m ruff`.
- Focused verification passed the complete `test_us_stacked_spine.py` and
  `test_us_multispine_pool_tool.py` modules plus the ACS PUMS, ACS transfer,
  PUF support, multispine-pool, and lineage-spec modules. `python -m ruff
  check .`, `git diff --check`, and the copied gate's `--help` import smoke all
  passed. The otherwise inert `_complete_numeric_component_sum` helper is
  retained solely because the byte-identical gate imports it for measurement;
  eligibility runtime does not select the investment predictor.
