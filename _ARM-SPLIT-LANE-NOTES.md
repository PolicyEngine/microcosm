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

### `arm-investment` surgery (2026-08-17)

- The branch starts from `arm-split-base` at
  `aec9fdc1cf71eb33019fe3e0e425926573544afb`. Early target declarations retain
  their baseline order, including CHAMPVA at position 0. No eligibility loader,
  CPS mapping, MIL dtype, or operator-boundary change is present.
- The assembled-artifact audit found 856,626 ACS persons. Raw `INTP` and mapped
  `acs_interest_dividend_rental_income` have the same null mask: all 127,597
  people under age 15 are null and all 729,029 people age 15 or older are
  finite. The finite raw and mapped values are deliberately not equated because
  the mapped field applies the ACS income adjustment. On the early ASEC donor,
  `INT_VAL`, `DIV_VAL`, and `RNT_VAL` are complete on all 108,073 people and
  exact zero for all 21,441 children under age 15.
- The donor investment analogue uses the six PolicyEngine components whenever
  that schema is complete and otherwise uses the complete numeric sum of
  `INT_VAL + DIV_VAL + RNT_VAL`. This is schema-level precedence, never
  row-level fallback, and both sources plus the precedence are bound into the
  transfer execution identity.
- Investment remains optional in its historical position after retirement and
  before head/tenure. It is enabled on exactly eight person families (housing;
  model-required numeric, boolean, and discrete; CPS-carried, hours-worked,
  prior-year-income, and relationship source operators) and four SPM-unit
  families (benefit participation, required boolean, required numeric, and the
  housing-input source operator). Person PUF tax itemization is excluded because
  its targets overlap the investment predictor's source leaves.
- Both early directions bind the same authority version-1 availability policy.
  It validates native ACS clone-0 source universes, fills only mapped under-15
  structural cells with zero in a modeling view, proves every fitted opted
  pattern carries the person or group investment feature, then restores the
  full native mapped series before the next direction. Raw `INTP` and restored
  mapped cells are byte-hashed. The structural completion raises complete ACS
  SPM investment aggregates from 309,885/382,903 to 382,903/382,903 without
  persisting the zeroes into the primary PUF pass or late transfer.
- Stacked authority version 11 binds the exact family predictor sets and full
  policy identity into the `gap_fill_plan` component. The #695 mirror matches
  those live early sets and policy assignments exactly; late transfer retains
  the baseline call with no explicit predictor sets.
- The copied round-1 gate remains byte-identical to the rejected-branch source
  and the other arms: SHA-256
  `25295af61fece0e124a2658eed469ce1b5ecf9abd0c8e6b622807df8d4d69fe8`.
- Focused verification passed the complete ACS-transfer, stacked-spine,
  ACS-PUMS, PUF-support, multispine-pool, multispine build-tool, and
  lineage-spec modules. It includes regression coverage for preferred/fallback
  precedence, invalid fallback values, exact family opt-ins, source-mediated
  leakage, structural completion, restoration, fitted-pattern coverage, and
  already-complete opted targets. Repository-wide Ruff, `git diff --check`, and
  the copied gate's `--help` import smoke also passed.
- Pre-build full-suite receipt on implementation commit `42a8c06c`: 6,255
  passed and 68 skipped (6,323 collected), with zero failures/errors in
  1,495.453 seconds. The machine-readable pytest receipt was written to
  `/private/tmp/arm-investment-full-suite.xml`; this journal-only follow-up does
  not change the tested runtime.

## Arm-gate continuation (2026-08-19)

- State: begin the frozen Phase-P attribution gates from `arm-investment` at
  `a91fbcdb81bd602d8e9b3465d9b26f6f423c7941`.
- Done: verified that all three arm artifact directories contain the H5,
  manifest, terminal-gate JSON, build log, and build-status receipt; verified
  that `tools/predictor_set_oos_gate.py` is byte-identical to the round-1 tool
  (SHA-256 `25295af61fece0e124a2658eed469ce1b5ecf9abd0c8e6b622807df8d4d69fe8`).
- Fixed boundary: baseline is the read-only `stacked-f025-r1` artifact with
  identity `9be8ecdf82356f38998e8b620ee36d9134f554fe89a8eafd8406f438e2b5aad6`;
  no build, network, push, gate-parameter change, `PROGRESS.md` edit, or
  `logbook-pending-chain.txt` access is authorized.
- Done: refreshed only the candidate integrity bindings in the frozen gate to
  the investment artifact's receipted stacked authority, gap-fill plan, and two
  early target-bank identities. All statistical, battery, fold, seed, forest,
  missingness, bootstrap, ratio-band, and clause values remain byte-equal to
  round 1's verdict configuration.
- Next: reproduce round 1's configuration byte-for-byte for the investment
  candidate, verify the emitted verdict and report, and commit that evidence on
  this branch before moving to the next arm.

### `arm-investment` frozen gate (2026-08-19)

- Done: the serialized gate completed all 32 baseline and 32 candidate family
  fits in 928.63 seconds and returned rc=1 for a substantive `FAIL`. The emitted
  verdict has `measurement_complete=true`, 118 targets, all 74 money targets,
  and all 211 physical battery legs. Its frozen configuration is byte-equal to
  round 1 after excluding only the candidate artifact-role receipt binding.
- Clause A FAIL: the CHAMPVA battery ratio moves from 3.720258676368841 to
  5.02792444674042; the baseline-minus-candidate ratio CI is
  [-1.8322351158693908, -0.9364953705201008].
- Clause B PASS: zero of 74 money targets has a candidate-minus-baseline OOS
  pinball-loss CI strictly above zero. In particular,
  `social_security_retirement` is exactly unchanged at 114.63718321141073 in
  both arms, with delta and CI [0.0, 0.0].
- Clause C FAIL: three legs leave the band:
  `person/source_operator_relationship_inputs/is_separated[clone_0]`
  (0.9946284697730712 -> 1.268146835387516),
  `spm_unit/model_required_boolean/is_tanf_enrolled[clone_0]`
  (0.9565667955075946 -> 1.623317491512484), and
  `spm_unit/model_required_numeric/spm_unit_pre_subsidy_childcare_expenses[clone_0]/positive`
  (0.989957173017115 -> 0.7183714602383162). No leg enters the band.
- Evidence: `experiments/phase_p_arm_split/investment/verdict.json` (SHA-256
  `bceed0b14f738d8f7664834225252fa5d85d1f2ad64c24ae8c32b1fc4c810a83`)
  and adjacent `report.md` (SHA-256
  `65d5d463e311c1cf3e69b378c2d8d9555de6b5dcd8d50c9b4c62a28b144e3b3a`).
- Resource note: BLAS pools were held to one thread while the frozen forest
  retained `n_jobs=4`, and no other gate or build ran concurrently. The host
  sandbox denied `/usr/bin/time -l` its `kern.clockrate` query, so this run does
  not claim a measured peak-RSS number.
- State: investment evidence is complete and ready for synthesis.
- Next: switch to `arm-eligibility`, bind only its candidate integrity
  receipts, run the same frozen gate, and commit that branch's evidence.

### `arm-investment` bounded receipt replacement (2026-08-19)

- State correction: the first investment verdict is statistically complete,
  but the denied `/usr/bin/time -l` query left it without affirmative evidence
  for the standing 20-GiB RSS ceiling. It will be replaced, not synthesized,
  unless an otherwise identical guarded run completes below the ceiling.
- Done: commit `bb2df330` adds the same private loader-owned Frame path proven
  by the eligibility and order gates. The gate now also requests allocator
  pressure relief at its existing collection boundaries. These changes affect
  only temporary checkpoint memory ownership and release, not the frozen
  estimand, models, folds, seeds, bootstrap, thresholds, bands, or clauses.
  Gate SHA-256 is
  `a9cd60f245221498090d23a2b684dd76165fa873a7e4b4e2784d385c0377f479`.
- Verification: Ruff passed the changed gate and loader files; all 44 Frame
  bundle tests passed. The candidate artifact bindings remain exactly those
  accepted in the first investment run.
- Next: commit the bounded-run setup and rerun serially under the 19.5-GiB
  stop guard. Require complete 32-family/118-target/74-money/211-leg evidence,
  kernel `ru_maxrss` below 20 GiB, and exact statistical equality to the first
  investment verdict before replacing it.
