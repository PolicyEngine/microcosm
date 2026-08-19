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
- Pre-build full-suite receipt on implementation commit `51126d40`: 6,250
  passed and 68 skipped (6,318 collected), with zero failures/errors in
  2,019.224 seconds. The machine-readable pytest receipt was written to
  `/private/tmp/arm-eligibility-full-suite.xml`; this journal-only follow-up
  does not change the tested runtime.

## Arm-gate continuation (2026-08-19)

- State: begin the frozen Phase-P attribution gate from `arm-eligibility` at
  `986578f09d55d73113e8d52fb164b48e46e2edbc`.
- Done: read the tip fix and its regression test. Raw `DEAR`, `DEYE`, and
  `HINS3` codes now bypass only the generic dtype gate; strict numeric coercion
  and supported-code validation still fail closed, including for object-backed
  checkpoint values.
- Done: refreshed only the candidate integrity bindings in the frozen gate to
  the eligibility artifact's receipted stacked authority, gap-fill plan, and
  two early target-bank identities. All statistical, battery, fold, seed,
  forest, missingness, bootstrap, ratio-band, and clause values remain
  byte-equal to round 1's verdict configuration.
- Fixed boundary: baseline remains read-only identity
  `9be8ecdf82356f38998e8b620ee36d9134f554fe89a8eafd8406f438e2b5aad6`;
  no build, network, push, `PROGRESS.md` edit, or pending-chain access occurs.
- Next: run the eligibility measurement under the 20-GiB RSS monitor, verify
  the complete verdict/report, and commit that evidence on this branch.
- Excluded monitor launch: the first wrapper asked psutil for system-wide child
  enumeration, which the macOS sandbox refused at `sysctl(KERN_PROC_ALL)`
  before the gate reached a family fit. The wrapper exited; `lsof` found no
  process holding either selected checkpoint or the intended output directory,
  and no eligibility verdict/report existed. No measurement from that launch
  is accepted. The replacement monitor observes the exact spawned gate PID
  (the forest uses in-process threads) and guarantees child cleanup on any
  monitor exception.
- Excluded fail-closed receipt check: the guarded retry ended after 7.06 seconds
  at 1,151,860,736 bytes peak RSS with `measurement_integrity` because the
  round-1 parser required rebuild-only `write_status`/`write_seconds` fields.
  No family fit or substantive clause ran, so its temporary verdict/report are
  not evidence.
- Root cause and fix: all 48 eligibility early-target records are authenticated
  durable resumes (`source=checkpoint`, `load_status=resumed`). The producing
  `AcsTransferTargetBankStore.receipt()` contract and its interruption/resume
  tests intentionally omit write-only fields for this case. The gate now
  accepts exactly that eight-key resumed variant as well as its original
  ten-key rebuilt variant; `_read_target_execution` still verifies file size,
  file SHA-256, identity, content-metadata SHA-256, raw-draw SHA-256, target
  descriptor, and chain state. No estimand or gate parameter changed.
- Next: rerun the guarded eligibility measurement; the temporary fail-closed
  files will be overwritten only by the complete rerun.
- Excluded memory-guard run: the complete 32-family baseline sweep finished,
  then the guard terminated the process before candidate family 1 when the
  baseline-to-candidate load transition jumped to 21,643,149,312 bytes
  (20.157 GiB) peak RSS. No candidate measurement or substantive verdict was
  emitted; the older temporary integrity-failure files remain non-evidence.
- Memory correction: the gate now asks Darwin's allocator to return unused heap
  pages after each donor-clone projection and after deleting each loaded frame,
  at the existing `gc.collect()` boundaries. This changes neither retained OOS
  measurements nor any fold, forest, bootstrap, threshold, band, or clause; it
  only prevents the freed 5.6-GiB checkpoint representation from remaining in
  process RSS while the opposite arm loads. The guarded rerun also constrains
  allocator magazines at process start.
- Next: rerun under the same hard monitor, requiring both a complete verdict
  and kernel `ru_maxrss` below 20 GiB before accepting evidence.
- Excluded conservative-guard run: allocator pressure relief reduced the same
  baseline-to-candidate handoff peak to 21,202,223,104 bytes (19.746 GiB),
  below the true 20-GiB limit, but the deliberately lower 19.5-GiB guard still
  terminated before candidate family 1. No verdict from that run is evidence.
- Root load profile: with allocator controls but before the loader correction,
  loading the eligibility checkpoint alone peaked at 20,950,433,792 bytes
  (19.512 GiB). `load_frame_checkpoint` built local tables and then invoked the
  public `Frame` constructor, which defensively copied every table even though
  the loader exclusively owned them.
- Done: commit `3ca43c9a` adds a private loader-owned Frame construction path;
  ordinary callers retain the public defensive-copy contract and all Frame
  invariants still run. The identical load-only profile now peaks at
  7,531,757,568 bytes (7.014 GiB). Focused public-copy, loader-owned identity,
  Frame revalidation, byte-identical checkpoint round-trip, and bound-metadata
  tests all pass, as do Ruff and formatting checks.
- Next: run the complete eligibility gate with the 19.5-GiB stop guard; accept
  it only with complete 32-family/118-target/74-money evidence and kernel
  `ru_maxrss` below 20 GiB.
- Accepted eligibility gate: the guarded run completed all 32 families, 118
  targets, 74 money targets, and 211 battery legs in 886.359440 seconds. The
  exact-PID monitor sampled a maximum RSS of 9,900,949,504 bytes; kernel
  `ru_maxrss` was 10,046,619,648 bytes, the 20-GiB limit was not crossed, and
  the 19.5-GiB stop guard did not fire. Exit 1 is the substantive red verdict.
- Configuration audit: after removing only the per-candidate artifact-role
  binding, the eligibility and round-1 configuration hashes are both
  `10f29427d83c340973869fb735188967aff9a4eea770b066f604e7a28a0b5e5a`.
  No fold, seed, forest, bootstrap, threshold, ratio band, battery, or clause
  parameter changed.
- Eligibility verdict: clause A FAIL (CHAMPVA ratio 3.72025867636884 to
  3.0739130288991374; 0.2616509977893486 progress), clause B FAIL (five
  significantly worsened money targets), and clause C FAIL (six battery legs
  left band; six entered). Social-security-retirement pinball instead changed
  114.63718321141073 to 113.23606657519205, delta -1.4011166362186742 with
  CI [-3.2000114150660317, 0.2620452172778749]; this arm does not reproduce
  the round-1 retirement blowup.
- Evidence hashes: `verdict.json`
  `bf531f48f4efa6969680c306718f2e46fb88876ccf46f5cfa1fb543429beb79d`;
  `report.md`
  `553bcb6fddf10a5b64b9f8826a9a6e7459d311589892b8f8779aa68fa2d2de05`;
  gate implementation
  `da6e952b4f902e444b95cd51483271e1f1febab61c16d389878ada32c96fe0ce`.
- Next: commit the accepted eligibility evidence, then measure the order arm
  with the same frozen gate configuration and bounded loader.
