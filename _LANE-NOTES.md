# QBI ownership lane notes

## 2026-08-21 intake

- Branch/worktree: `battery-qbi-ownership` at `2c7a7218`, rooted at
  `/Users/maxghenis/PolicyEngine/_worktrees/microcosm-qbi-ownership`.
- Binding build policy: at most a 1% sample; off-chain only; never pass
  `--logbook-prev-row-digest`; never touch `logbook-pending-chain.txt`; remain
  below 15 GiB RSS; check for a live `build_us_multispine_pool` before building.
- Binding model policy: do not tune gates, bands, ceilings, folds, or seeds;
  exclusions are owner-only and no register entry may be added.
- Binding evidence policy: every mechanism claim in the final record must cite
  current `module:line` locations.

## Environment receipt

- `uv sync --all-packages --extra us` failed before resolution because the
  managed sandbox denies `/Users/maxghenis/.cache/uv`.
- Retry with `UV_CACHE_DIR=/private/tmp/microcosm-qbi-uv-cache` resolved the
  workspace but could not download `pydantic==2.13.4` because network/DNS is
  disabled.
- `/Users/maxghenis/PolicyEngine/_worktrees/microcosm-spec-engine/uv.lock` has SHA-256
  `ea7af7806a0beefe7394adefd5516649f3eba4740ae95ccdaa9aaa252249bc3a`,
  exactly matching this worktree's lock, and its `.venv` contains the locked US
  and development dependencies. Test commands use that environment read-only
  with `UV_NO_SYNC=1`, a writable uv cache, and `PYTHONPATH` entries pointing
  only to this worktree's five package `src` directories.

## Interrupted-lane salvage disposition

- Before any new implementation, inspected `git show --stat 03e23e42` and its
  complete diff from branch HEAD, then verified that the restored worktree was
  byte-for-byte identical to the salvage tree (`git diff --quiet 03e23e42 --
  .`). The salvage is a direct child of the lane's journal commit and contains
  25 relevant files: realized-regime persistence, schema/checkpoint
  invalidation, documentation, golden evidence, and regressions.
- Took the complete coherent source, test, and documentation tree. No recovered
  file was discarded. Did not cherry-pick the generated WIP commit wrapper
  because the same content was already present and needed an audit before
  becoming branch history.
- The audit found two fail-open seams and tightened them. Deterministic origins
  now require an exact declared target/derivation binding; early and late
  receipts require non-boolean nonnegative counts satisfying the null-flow
  accounting equations
  (`packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:4707-5070,5201-5754`).
- A first build-package shard was stopped after the audit exposed those seams;
  it is not cited as a gate. A fresh full five-shard gate supersedes it.

## Source-of-truth evidence map

- Remediation order: adjudication line 42 requires recomputing each realized
  QRF regime from frozen donor support for every availability pattern and
  persisting it before any regime-specific remedy.
- Workstream: adjudication line 112 assigns eight BLOCKER checks/four physical
  QBI amount legs to ownership attribution and coupled-surface refit.
- Separate fix: adjudication lines 69-71 and rows 186-187 assign the two SSTB
  booleans to clone 1; they are not exclusions.
- Reproducibility: adjudication line 228 binds the failed simulated checkpoint
  SHA-256 `5b47eb0ded02f4031e235b7a6e07506b5bd38f87827644752d26f4263e492f5a`
  to release/run IDs and states that all nine QBI invariants were recomputed at
  tolerance `1e-8`.

## Open evidence questions

- For W2 and UBIA, distinguish the clone-1 PUF producer margin from the clone-0
  late-transfer margin before reconciliation.
- For BDC and REIT/PTP, also record the post-transfer QBI reconciliation delta,
  because exposure caps can change both roles.
- A margin must not be assigned an origin unless the receipt includes the exact
  check ID, stage metrics, and a nonempty closed-set origin channel.

## Historical journal-only commit gate

Commands used the exact-lock read-only dependency environment described above
and this worktree's package sources:

- `uv run pytest packages/microcosm-frame/tests`: 294 passed, 36 skipped.
- `uv run pytest packages/microcosm-fit/tests`: 93 passed.
- `uv run pytest packages/microcosm-calibrate/tests`: 201 passed.
- `uv run pytest packages/microcosm-data/tests`: 275 passed, 1 skipped.
- `uv run pytest packages/microcosm-build/tests`: 5,959 passed, 39 skipped.
- `uv run ruff check .`: clean.
- `git diff --check`: clean.

## 2026-08-22 resumed lane: salvage disposition and environment

- Three later salvage snapshots existed beyond the first: `21f95b71`,
  `8942ef97`, and newest `321b3185`. The trees at `21f95b71` and `8942ef97`
  are identical. `321b3185` supersedes them; the current `docs/`, `packages/`,
  and `tools/` contents are byte-identical to it. Its four
  `experiments/qbi_ownership/` assets also match but remain untracked for a
  separate evidence audit and commit. The only technical change from
  `8942ef97` to `321b3185` outside those evidence assets is the recovered
  `unmodeled_rows: 0` fixture line in `test_us_multispine_pool_tool.py`.
- The interrupted lane had also continued past the last snapshot: the
  worktree's `stacked_spine.py` carried ~111 further lines beginning the
  two-check terminal-role fix (`_EXPLICIT_ORIGIN_BATTERY_ROLE_DECLARATIONS`
  and role-aware key construction). That in-progress role work was set aside
  verbatim (`/tmp/stacked_spine.role-work.py`) and the file restored to the
  `8942ef97` state, so the realized-regime prerequisite commits alone and the
  role fix lands as its own audited commit with its ownership evidence.
- Environment: this resumption has network; the mandated
  `uv sync --all-packages --extra us` succeeded and all commands use this
  worktree's own `.venv` (the read-only spec-engine environment fallback is
  no longer in use).
- Gate-command correction: `pyproject.toml` already passes `-q` via
  `addopts`; a `pytest -q` invocation therefore ran at `-qq`, which
  suppresses the final summary line, and piping through `tail` masked exit
  codes. The prior lane's quoted per-shard counts were produced with plain
  `uv run pytest` per shard; this lane reran the gate the same way with
  per-shard logs and explicit exit codes before committing.

## 2026-08-23 complete salvage disposition

- All four salvage commits are sibling snapshots rooted at `9835fb4b`.
  `21f95b71` and `8942ef97` are byte-identical trees. Relative to
  `03e23e42`, their substantive additions are the declared deterministic-
  origin binding and fail-closed target-count/accounting validation in
  `stacked_spine.py`, with corresponding regressions. The remaining changes
  are journal updates and formatting.
- Took the recovered regime/origin receipt implementation, schemas,
  checkpoint invalidation, documentation, and tests into prerequisite commit
  `f042bfa8`. Relative to `8942ef97`, that commit differs technically only by
  three fixture corrections: the newest snapshot's pool-tool
  `unmodeled_rows: 0`, the legacy checkpoint materializer repin from 3 to 4,
  and the missing stacked-HDF5 `unmodeled_rows: 0`.
- Took the four `experiments/qbi_ownership/` assets from newest snapshot
  `321b3185` as an analysis and implementation basis, then superseded them in
  `7e0c5081`. The recovered `evidence.json` had been produced with
  `--skip-sha`, contained a nondeterministic timestamp, lacked closed artifact
  bindings and per-check ownership records, and compared only six-digit
  display values. Its extractor also allowed a SHA-skipped run to overwrite
  canonical output.
- Discarded the generated WIP commit wrappers, stale journal/gate placeholders,
  superseded `03e23e42` variants, and the newest snapshot's conflation of
  terminal value provenance with first-failing-stage ownership. The committed
  evidence instead records all eight terminal values as `qrf_transfer` while
  identifying five transfer-first criteria and three producer-first QED
  criteria. It also replaces the unsupported claim that coarse recipient
  marginals identify the transfer mechanism with a requirement for
  target-by-channel-by-pattern diagnostics.
- No unique production regime-persistence or origin-receipt implementation
  from the newest salvage was discarded. Later terminal-role work builds on
  that recovered prerequisite; therefore byte-identity claims are bound to
  commit `f042bfa8`, not to the subsequently modified working tree. The
  recovered and committed canonical evidence JSON SHA-256 values are
  `f0463f449a260b69bc72663702ff6b1b778dfca1dbcc3c3c5430204a122d7114`
  and `38e60c1ec5e39b86df957148c877b3062ca97028f33ea0d1411013c2911c4b55`,
  respectively.

## Realized-regime receipt prerequisite

- The monolithic transfer recomputes each target's QRF regime from the exact
  availability-pattern donor model frame and the fitter's own `zero_atol`, then
  rejects disagreement with the fitted QRF
  (`packages/microcosm-build/src/microcosm/build/us_runtime/acs_transfer.py:1631-1651,1705-1885`).
- The banked path performs the same recomputation before any target draw,
  compares every `fit_draw_next` result to it, and passes it through the bank
  load/write contract
  (`packages/microcosm-build/src/microcosm/build/us_runtime/acs_transfer.py:1908-2206`).
- Pattern provenance is copied into an immutable mapping; every fitted exported
  leaf names its exact model target, and canonical origin receipts distinguish
  `qrf_transfer`, `deterministic_derivation`, and `preexisting`
  (`packages/microcosm-build/src/microcosm/build/us_runtime/acs_transfer.py:423-607,634-723`).
- Target-bank schema 2 persists the regime beside each state transition,
  authenticates it on load, refuses valid-but-wrong regimes before filesystem
  persistence, and exposes the per-pattern map in its public receipt
  (`packages/microcosm-build/src/microcosm/build/us_runtime/acs_transfer_bank.py:25-32,125-425,612-780`).
- Early and late stacked target receipts now carry canonical origin evidence.
  The production late validator rejects absent/unknown origins, mismatched
  aggregate/group copies, incorrectly renamed model targets, and divergent
  regimes for exported siblings sharing a joint model target
  (`packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:4344-5070,5201-5754,11194-11313,11635-11759`).
- Stacked pool-stage checkpoint materializer 8 and retiring legacy
  materializer 4 make
  transferred/simulated checkpoints that predate canonical regime evidence
  stale (`tools/build_us_multispine_pool.py:228-285`).
- Late-producer receipt schema 4 makes the new mandatory target origin/regime
  envelope explicit and rejects otherwise valid schema-3 receipts
  (`packages/microcosm-build/src/microcosm/build/us_runtime/us_late_producer_registry.py:102-129,2053-2070`;
  `packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:8766-9015`).
- Focused regressions cover immutable/no-alias mappings, JSON serialization,
  cold/warm bank parity, missing/unknown/known-wrong on-disk regimes,
  valid-but-wrong pre-write rejection, exact joint-codec ownership, canonical
  target origin presence, missing-origin rejection, sibling-model renaming,
  legacy-v3 staleness, invented deterministic origins, boolean count coercion,
  and forged count equations.

## Recovered-prerequisite diagnostic gate (2026-08-22)

The exact foreground command ran plain
`uv run pytest packages/microcosm-<shard>/tests` for frame, fit, calibrate,
data, and build under `set -euo pipefail`, followed by Ruff and
`git diff --check` only if every shard passed. Results:

- frame: 294 passed, 36 skipped (29m39s)
- fit: 93 passed (4m14s)
- calibrate: 201 passed (55s)
- data: 275 passed, 1 skipped (37s)
- build: 5,956 passed, 39 skipped, 21 failed (3h15m58s)
- Ruff and `git diff --check`: not reached because `set -e` stopped at build

Twenty build failures traced to two incomplete fixture repins in the newest
salvage: three exact-k cases still named frozen legacy checkpoint materializer
3 instead of 4, and seventeen stacked HDF5 loader cases built canonical target
receipts without the mandatory `unmodeled_rows` count. The remaining failure
was the deterministic trade-entry subprocess exceeding its 300-second test
timeout under host load. No production logic, gate, band, ceiling, fold, or
seed was changed in response.

After the two fixture repairs, focused reruns passed:

- `test_us_exact_k_ladder_e2e.py`: 3 passed
- `test_us_multispine_pool_h5_io.py`: 38 passed
- `test_us_trade_entries_cli.py::test_two_builds_are_byte_identical`: 1 passed

A fresh full prerequisite gate follows and is the only gate that will support
the prerequisite commit.

## Prerequisite commit gate (2026-08-22)

All results below cover the prerequisite code/test tree being committed. The
four package shards outside build used the mandated plain per-shard command and
completed green:

- frame: 294 passed, 36 skipped (3m08s)
- fit: 93 passed (3m51s)
- calibrate: 201 passed (29s)
- data: 275 passed, 1 skipped (26s)

An unbounded full build-shard rerun passed 5,973 tests and skipped 39, but four
unchanged subprocess tests timed out under host contention (one 300-second
trade-entry build and three 60-second crash-recovery probes). No test assertion
failed, and no timeout, gate, band, ceiling, fold, or seed was changed. The four
unchanged nodes passed together (4 passed, 87 deselected in 2m37s) after native
and job pools were bounded to one.

The complete build shard was then rerun with
`OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`,
`VECLIB_MAXIMUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`, and
`LOKY_MAX_CPU_COUNT=1`, plus the same uv cache/`UV_NO_SYNC` settings:

- build: 5,977 passed, 39 skipped (1h10m41s, exit 0)
- `uv run ruff check .`: clean
- `git diff --check`: clean

Resource-pool bounding changes execution concurrency only and keeps this lane
within its RSS contract; model and test thresholds remain untouched. No pool
build was started, because the headless order assigns builds to the host queue.

## Ownership evidence package (2026-08-22)

- `experiments/qbi_ownership/extract_qbi_ownership_evidence.py` authenticates
  the failed-attempt publication manifest/gates, the assembled/transferred/
  simulated stage checkpoints, thirteen QBI target-bank files, and the
  post-PUF transfer receipt before emitting a closed validation receipt. The
  production checkpoint loader binds physical identity before stage use
  (`tools/build_us_multispine_pool.py:2139-2141`;
  `packages/microcosm-build/src/microcosm/build/us_runtime/multispine_pool.py:2851-2908`).
- The extractor reproduced all eight adjudicated comparator cells at exact
  stored precision and recorded two distinct facts for each: the terminal
  value origin and the first stage where its criterion becomes red. All eight
  terminal values come through the late QRF transfer; all four incidence
  checks plus UBIA QED first fail there, while BDC, REIT/PTP, and W2 QED first
  fail on clone 1 and worsen after transfer. The transfer fills only the
  declared complement of producer rows, and the stacked executor validates
  the target accounting/origin envelope
  (`packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:10920-10972,11076-11191,11194-11313`).
- Recomputed realized regimes from the exact frozen 108,073-row donor support
  for four availability patterns per target. The replay covers 52 cells (13
  chained targets × four patterns); all 16 cells for the red amounts are
  `zero_inflated_positive`. The failed-attempt bank did not persist the regime,
  which is the adjudicated receipt gap. That regime
  uses a weighted zero/positive gate followed by the positive-value QRF
  (`packages/microcosm-fit/src/microcosm/fit/qrf.py:104-105,950-1003,1333-1429`).
  Future monolithic and banked fits recompute and persist the exact
  pattern-regime map; stacked receipts validate its schema and cross-copy
  consistency without pretending to replay donor support
  (`packages/microcosm-build/src/microcosm/build/us_runtime/acs_transfer.py:561-607,634-723`;
  `packages/microcosm-build/src/microcosm/build/us_runtime/acs_transfer_bank.py:265-425,612-780`;
  `packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:4344-4687,5201-5754`).
- Reran the same nine whole-pool coupled identities before and after QBI
  reconciliation. The terminal simulated checkpoint has zero violations for
  all nine. The transferred checkpoint records all pre-reconciliation deltas,
  including 2,996 BDC and 22,350 REIT/PTP exposure mismatches; the production
  reconciliation applies those exposure caps and computes the nine-identity
  summary
  (`packages/microcosm-build/src/microcosm/build/us_runtime/qbi_inputs.py:1324-1359,1377-1487`).
- The ownership regression exercises all four amount targets across the group,
  aggregate, and signed execution receipt copies. Each copy must carry the
  same nonempty `qrf_transfer` origin and exact target-pattern regimes; deleting
  `origin.channel` from any one copy is rejected by the closed receipt
  validator
  (`packages/microcosm-build/src/microcosm/build/us_runtime/acs_transfer.py:561-607,634-723`;
  `packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:5201-5754,8766-9015`).
- Canonical extraction ran twice with full SHA verification and produced the
  same `evidence.json` SHA-256
  `38e60c1ec5e39b86df957148c877b3062ca97028f33ea0d1411013c2911c4b55`.
  Validation reports: 8 ownership checks, 13 SHA-verified bank target files,
  52 recomputed regime cells, 9/9 terminal invariants at zero, and no adjudication
  mismatch. A canonical `--skip-sha` invocation exited 2 before write and left
  that digest unchanged.
- The refit plan does not infer a gate remedy from coarse marginals. Current
  closed receipts record regimes and pattern catalogs but not row assignments,
  gate scores/outcomes, or target-by-channel-by-pattern gate cross-tabs
  (`packages/microcosm-build/src/microcosm/build/us_runtime/acs_transfer.py:561-607,634-723`).
  Those diagnostics and the exact BDC/REIT exposure predictors require the
  host-owned 1% demonstration build; the 25% host run remains certification.
  No pool build, model tuning, exclusion, logbook-chain write, or amount-model
  change occurred in this lane.

## Ownership evidence commit gate (2026-08-22)

All commands used the synchronized worktree environment, the writable uv
cache, `UV_NO_SYNC=1`, and bounded native thread pools. Results cover the exact
code/evidence tree being committed:

- frame: 294 passed, 36 skipped (2m39s)
- fit: 93 passed (22s)
- calibrate: 201 passed (21s)
- data: 275 passed, 1 skipped (21s)
- build: 5,981 passed, 39 skipped (58m52s)
- `uv run ruff check .`: clean
- `git diff --check`: clean

## Exact-two SSTB terminal-role correction (2026-08-23)

- The adjudicated live signals are now the only non-native comparison roles:
  `sstb_self_employment_income_would_be_qualified` and `business_is_sstb`
  map to the PUF-detail clone-1 role; every other physical battery target
  defaults to clone 0. The immutable declaration and lookup are at
  `packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:2133-2160`.
- The role scope fails closed unless it is exactly those two QBI boolean PUF
  outputs, and metric registry materialization carries each target's resolved
  role
  (`packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:3030-3131`).
  Surface, plan, producer/transfer authority, and runtime comparator keys are
  role-aware
  (`packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:2163-2187,3320-3341,3490-3506,5201-5754,8766-9015,14006-14391`).
- The spec projection permits nonzero registered roles while rejecting one
  physical target declared across multiple roles
  (`packages/microcosm-build/src/microcosm/build/spec_engine/battery_semantics.py:43-71`).
  The live battery contract similarly indexes by physical target, preserves
  its registered role, and rejects ambiguity
  (`packages/microcosm-build/src/microcosm/build/us_runtime/stacked_battery_contract.py:66-118`).
- The role-fix commit assigned clone 1 to exactly the two targets. The later
  semantic union advanced the combined authority to version 12 because the
  role and calibration source branches independently used version 11
  (`packages/microcosm-build/src/microcosm/build/us/spec/battery.yaml:459-503,824-826`;
  `packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:1703-1707`).
- Direct role, fail-closed, live-contract, projection, transfer-authority, and
  comparator regressions are green as a 424-test focused suite. No gate,
  band, ceiling, fold, seed, exclusion, amount model, or build artifact was
  changed.

## Exact-two correction commit gate (2026-08-23)

The four non-build package shards completed green with bounded native threads:

- frame: 294 passed, 36 skipped (3m45s)
- fit: 93 passed (16s)
- calibrate: 201 passed (17s)
- data: 275 passed, 1 skipped (15s)

Two exact full-build-shard foreground attempts were terminated externally at
the execution service's one-hour ceiling (exit 143), at approximately 11% and
14% progress. Neither reported a test failure. Because the host was running
many concurrent Python lanes, the unchanged all-in-one command could not
finish inside that ceiling. No timeout or test/model threshold was changed.

To retain exhaustive coverage, the 255 sorted `test_*.py` files were split
into ten deterministic, disjoint consecutive slices and passed to the same
`uv run pytest` command. Every file appears exactly once:

- files 1–26: 578 passed, 1 skipped (42m08s)
- files 27–52: 426 passed (22m34s)
- files 53–78: 265 passed, 7 skipped (17s)
- files 79–104: 491 passed, 15 skipped (1m20s)
- files 105–130: 303 passed, 11 skipped (56s)
- files 131–156: 691 passed, 1 skipped (2m00s)
- files 157–182: 664 passed, 1 skipped (29m33s)
- files 183–208: 980 passed (17m22s)
- files 209–234: 924 passed, 2 skipped (49m13s)
- files 235–255: 675 passed, 1 skipped (19m58s)

The partitioned build result is therefore 5,997 passed and 39 skipped over all
255 test files. Together with the four package shards above, it covers the
entire workspace test suite required by this lane. `uv run ruff check .` and
`git diff --check` both completed cleanly on the commit tree.

No pool build was started; this is test execution only. The host queue retains
exclusive ownership of all `build_us_multispine_pool.py` runs.
<details>
<summary>Imported origin/main journal at the semantic-union boundary</summary>

The following replacement-scorecard journal predates this lane on the merged
mainline. It is retained as historical context only; the QBI record above and
the integration record appended below are current.

# Replacement-scorecard lane notes

## 2026-08-22 — lane start and environment

- Branch: `replacement-scorecard`, starting at `2aa96795` (post-#741
  `origin/main`).
- Owner ruling: publication evidence is a same-yardstick incumbent-versus-
  candidate comparison. This lane builds that yardstick and scores only the
  incumbent because the 25% bundle-mode candidate does not yet exist.
- Standing constraints recorded: no pushes, no pool builds, no gate/threshold/
  band tuning, green suite per commit, scoring below 20 GiB RSS, and a build
  queue check before scoring.
- Environment: the default-cache sync was denied by the managed sandbox, and a
  task-local empty cache could not download through disabled DNS. The sibling
  `microcosm-one-surface` checkout has the identical `uv.lock` (`895535...`)
  and a complete environment. Its `.venv` was cloned copy-on-write, after
  which a narrow writable cache of locked Hatch build requirements allowed
  `uv sync --offline --all-packages --extra us` to rebuild and relink all five
  Microcosm workspace packages to this worktree.
- The GitNexus exploration workflow was selected for the requested execution-
  path audit, but this session exposes no GitNexus resources or query tools.
  Repository-wide `rg`, symbol inspection, and focused tests are the fallback.
- The inherited one-target-surface notes below were accurate when written and
  are historical after #741; they are not current replacement-lane state.
- Journal-step validation: `uv run python -m pytest -q
  packages/microcosm-build/tests/test_us_state_files_scorer.py` passed (5/5),
  and `uv run ruff check .` passed. A complete workspace baseline is still
  running; the direct `python -m pytest` form avoids the cloned environment's
  stale console-script shebang.

## 2026-08-22 — yardstick audit

- One compiler surface: `compile_us_fiscal_target_registry` has no
  artifact-membership switch
  (`packages/microcosm-build/src/microcosm/build/us_runtime/fiscal_targets.py:933-1017`),
  and the existing fiscal scorer's only divergent branch is loading before the
  common repair/materialize/score sequence
  (`tools/score_us_fiscal_targets.py:432-489`).
- Full-surface rule: the head-to-head must reject either
  `target_compilation.dropped_target_names` or `CalibrationResult.skipped`;
  both lower layers otherwise report and continue past an unmaterialized or
  uncompileable row
  (`tools/build_us_fiscal_refresh_release.py:4323-4347`;
  `packages/microcosm-calibrate/src/microcosm/calibrate/matrix.py:286-355`).
- Aggregate rule: use the production
  `sqrt_value_concept_budget_weighted_mape_50_50_amount_count_target_scale_cap_100pct`
  constants, with no family multipliers. Target-scale square roots are
  normalized within amount/count basis, semantic concept groups receive one
  concept budget, and present bases receive equal total budgets
  (`tools/build_us_fiscal_refresh_release.py:344-348,5781-5814,6214-6290`).
  The aggregate is the weighted mean of capped target-scaled absolute errors
  (`packages/microcosm-calibrate/src/microcosm/calibrate/solve.py:473-518`).
- Incumbent identity: PolicyEngine.py 4.15.0's bundled manifest names retired
  data package 1.115.5 and its historical HF model repo, immutable
  resolver revision `9531fe1d096244fe7eb45d791d52ef61b8a2a0a5`, filename
  `enhanced_cps_2024.h5`, and SHA-256
  `0a6b961ad363a421bde99f2c8e5d8f20370bcba45fd303050537a25bdd805b14`
  (`policyengine/data/release_manifests/us.json@4.15.0:12-27,37-42`).
  Microcosm's frozen parity reference pins revision `21280dca...` for the same
  hash (`packages/microcosm-build/src/microcosm/build/us/ecps_parity_reference.json:7-14`).
  Both cache-resolved files independently hash to `0a6b961a...`; the scorecard
  will retain both identities and call the former the package-resolved one.
  **[Superseded 2026-08-22, later session: 4.15.0 was tagged 2026-06-10 and is
  not the live package. See "incumbent identity corrected" below — the live
  policyengine.py 5.0.3 default US dataset is the buildp sparse artifact, not
  enhanced_cps_2024.]**
- Terminal-battery scope: all 131 marginal comparisons and the joint
  immigration comparison are by-origin-only. Each needs support-channel and
  clone-role columns to build separate ASEC and ACS masks
  (`packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:14006-14020,14136-14202,14301-14325`).
  The incumbent has neither field, so every battery comparison is
  **inapplicable**, not failed or zero. A finished pool publishes an input-only
  H5 while sealing terminal results into its manifest and diagnostics
  (`tools/build_us_multispine_pool.py:3263-3334,3533-3550,3766-3786`); the
  manifest loader validates the H5, diagnostics, digests, run identity, and
  passing terminal alias before returning the frame
  (`packages/microcosm-build/src/microcosm/build/us_runtime/h5_io.py:348-430,672-811`).
  **[Partially superseded 2026-08-22, later session: the by-origin-only
  classification and the pool-manifest receipt path stand, but the "incumbent
  has neither field" premise described the eCPS. The actual live incumbent
  (buildp sparse) carries both provenance columns; what it lacks is any
  ACS-origin row — see "incumbent identity corrected" below.]**
- Yardstick facts: the v9.4 feed at
  `/Users/maxghenis/PolicyEngine/_buildh-runtime/inputs/consumer_facts_buildn_v9_4.jsonl`
  has SHA-256 `b3c0835631a446eb96aa84d86f3ee962d15ca356174c7114db52974f1cacc080`.
  A read-only compile with the packaged current-vintage CD crosswalk, target
  period 2024, no aging, and the standing scoring period waiver produced
  32,842 specs at registry version `c4ac617743f2`. No artifact was built or
  scored during this compile.

## 2026-08-22 — incumbent identity corrected (fresh session, post-salvage)

- Environment: this session's `uv sync --all-packages --extra us` completed
  normally (network available; the earlier codex-sandbox venv cloning is
  historical).
- **The live policyengine.py US dataset is the microcosm buildp sparse
  artifact, not enhanced_cps_2024.** PyPI's latest `policyengine` is 5.0.3
  (tagged 2026-08-21; 4.15.0, the version the earlier audit read, was tagged
  2026-06-10). The 5.x bundle replaced the per-country release manifests:
  `get_release_manifest` reads the bundled
  `src/policyengine/data/bundle/manifest.json`, and
  `resolve_managed_dataset_reference(country, dataset=None)` returns
  `manifest.default_dataset_uri`
  (`policyengine.py@5.0.3 src/policyengine/provenance/manifest.py:301-320,540-561`);
  `default_dataset_uri` returns the certified artifact URI when
  `certified_data_artifact.dataset == default_dataset`
  (`.../provenance/manifest.py:181-187`), and dataset overlays are additive
  only — an overlay that shadows the default raises
  (`.../provenance/manifest.py:270-299`).
- The 5.0.3 bundle's US entry: `default_dataset: populace_us_2024`,
  `data_producer: populace`, build id
  `populace-us-2024-buildp-sparse-rmloss100-cae8640-20260728T011454Z`,
  HF repo `policyengine/populace-us` (repo_type `dataset`), revision equal to
  the build id, filename `populace_us_2024.h5`, SHA-256
  `48b9d479fb4fd1c3537f9383ce4697d130b6f618658409d74f6233c43b994c7e`,
  certified for policyengine-us 1.764.6 — the same engine version this
  workspace's `uv.lock` pins
  (`policyengine.py@5.0.3 src/policyengine/data/bundle/manifest.json`
  `data_releases.us.{default_dataset,certified_data_artifact,data_package,model_package}`).
- Local bytes verified: the HF cache ref for that revision points at commit
  `26dcad66867687f15735dc4926523e3741920836`, whose snapshot
  `populace_us_2024.h5` (462,915,783 bytes) hashes to `48b9d479...` exactly.
- Observed incumbent shape (read from those bytes this session): entity-table
  layout (six US entities + `_time_period` 2024), 57,240 households, all
  household weights positive; provenance columns present
  (`household_support_channel`, `household_support_clone_index`, person
  equivalents); channel counts `asec` 22,200 (clone 0) and `puf_tax_detail`
  35,040 (clone 1); **zero `acs`-channel rows**. The by-origin battery
  compares `asec` vs `acs` channels
  (`packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:14136-14202`;
  channel constants `support_provenance.py:31` and `stacked_spine.py:276`),
  so every comparison is **inapplicable on observed evidence** — the
  incumbent predates the ACS stack and has no ACS origin to compare, which
  is a different (and observed) reason than the eCPS "no provenance columns"
  premise.
- CD provenance: the buildp H5 root attrs are PyTables boilerplate only — it
  predates the CD vintage crosswalk attributes, so
  `_assert_cd_vintage_support_matches` would fail on the attr comparison
  (`tools/build_us_fiscal_refresh_release.py:2519-2567,2570-2597`). The
  head-to-head scorer probes the attrs: strict when present, and otherwise
  records an explicit legacy waiver receipt while still requiring the
  household `congressional_district_geoid` lookup column to exist with
  positive values (observed present on the incumbent).
- Salvage adoption: the 1306-line sol draft was adopted after verifying every
  imported symbol and cited mechanism against the code this session. Rewrites
  beyond the identity block: entity H5s load through the canonical scorer's
  `release._load_frame` seam (`tools/score_us_fiscal_targets.py:436`,
  `tools/build_us_fiscal_refresh_release.py:2454-2471`), the
  dropped-target check now runs before scoring (clear failure instead of a
  loss-weight shape error), the battery inapplicability receipt is computed
  from observed origin-channel counts instead of asserted, and the Markdown
  scorecard renders rollups plus worst rows with the complete per-target
  table living in the JSON twin.

## 2026-08-22 — scorer landed; incumbent probe on real bytes (resumed session)

- Battery entities are `person`/`tax_unit`/`spm_unit` (114/9/8 single-column
  comparisons plus the joint person immigration comparison) — household is
  not a battery entity
  (`packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:3011-3025`).
  The test fixture originally put provenance columns on household+person and
  its two battery tests failed; the fixture now provisions channel/clone
  columns on exactly the battery entities. The scorer needed no change.
- Real-bytes probe of the incumbent (the scorer's own probe path run against
  the cached `populace_us_2024.h5`, sha256 `48b9d479...`): layout
  `entity_tables`; `read_nullable_us_h5_metadata` raises "no artifact
  metadata" (caught → not a naked pool); CD vintage attrs both null with a
  usable `household.congressional_district_geoid` lookup (57,240 rows, 436
  positive unique geoids → the recorded legacy waiver path); frame loads as
  166,321 persons / 57,240 households / 79,729 tax units / 59,900 spm units
  with calibrated household weights. All three battery entities carry
  provenance columns; per-entity channel counts: person asec 66,001 +
  puf_tax_detail 100,320; tax_unit asec 30,974 + 48,755; spm_unit asec
  23,286 + 36,614; **zero `acs` rows on every entity** → the battery payload
  reports `inapplicable` with the observed empty-ACS-side reason. This
  verifies the earlier journal claim and extends it to tax_unit/spm_unit.
- Suite state at commit: head-to-head tests 7/7; state-files scorer,
  refresh-builder, pool-h5-io, fiscal-targets, and release-target-parity
  files all green (428 tests); ruff check + format clean.

## 2026-08-22 — newest salvage reconciled; candidate boundary made scoreable

- Inspected `refs/claude-salvage/replacement-scorecard-20260822-220105-73402`
  (`5577ee4c`) and verified that its scorer blob exactly matched the inherited
  uncommitted file. The salvage's household slicing was retained deliberately,
  but its full-pool measure-array assembly was replaced: a 25% dense pool at
  roughly 918,350 households would require about 56 GiB for one
  8,192-column float64 copy. The scorer now materializes and scores one fixed
  household slice at a time, checks weights plus target/scale/name/column
  contracts on every slice, accumulates the additive matrix/weight products in
  fixed order, frees the slice, and checks RSS before continuing
  (`tools/score_us_release_head_to_head.py:654-852,1399-1477`;
  `tools/build_us_fiscal_refresh_release.py:3730-3750`;
  `packages/microcosm-calibrate/src/microcosm/calibrate/score.py:79-142`).
- The production pool loader remains a readiness boundary. A separate
  scorecard-evidence loader accepts the exact current stacked
  `status=gate_failed` / `simulation_ready=false` pair without weakening any
  manifest/diagnostics/H5 digest, schema, run-ID, materializer, terminal-alias,
  transition-authority, weight-kind, or row-count authentication
  (`packages/microcosm-build/src/microcosm/build/us_runtime/h5_io.py:371-588,719-869`).
  This implements the owner's head-to-head ruling without relabeling a failed
  candidate as simulation-ready.
- Finished entity H5s do not retain the stacked assembly/tail manifests. Their
  battery path therefore evaluates the canonical authority, 132 comparisons,
  369 nominal scalar legs, clone-0 positive-weight scopes, metrics, support
  rules, and tolerances as artifact evidence, while explicitly reporting
  `production_receipt_authenticated=false`. The one structural ACS
  group-quarters scope is derived from retained origin/clone/`TYPEHUGQ`/
  membership/tenure columns and marks assembly authentication false
  (`packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:10222-10310,13967-13985,14006-14391`).
- Validation: the combined focused run covered 19 scorer/pool/battery cases.
  Eighteen passed; the only failure was an assertion matching the wrong error
  wording, after which that corrected test reran green. Ruff check/format,
  Python byte compilation, and `git diff --check` pass. No pool build, push,
  gate edit, threshold edit, or band edit occurred.
- Mainline reconciliation: merged the six newer `origin/main` commits at
  `34d93846` with no conflict; the only US change among them isolates the
  fiscal-refresh memory canary. Post-merge validation passed all 13 scorecard
  tests and all 8 fiscal-memory tests (21/21).
- Required pre-score process check: `ps ax | grep
  build_us_multispine_pool` was attempted and denied by the managed macOS
  sandbox; `pgrep` and `top` are denied as well. The permitted `lsof -d cwd`
  process scan found no build-runtime working directory, and a full permitted
  file-descriptor scan found no open `build_us_multispine_pool`, pool H5,
  checkpoint, or pool-manifest path. The incumbent score may therefore start;
  no pool build was launched by this lane.

## 2026-08-23 — incumbent yardstick established

- Exact command run from this worktree after the empty queue scan:

  ```bash
  .venv/bin/python tools/score_us_release_head_to_head.py \
    --incumbent /Users/maxghenis/.cache/huggingface/hub/datasets--policyengine--populace-us/snapshots/26dcad66867687f15735dc4926523e3741920836/populace_us_2024.h5 \
    --ledger-facts /Users/maxghenis/PolicyEngine/_buildh-runtime/inputs/consumer_facts_buildn_v9_4.jsonl \
    --out-prefix experiments/replacement_scorecard/incumbent_48b9d479 \
    --maximum-microsim-batch-size 5000
  ```

- The scorer completed with exit 0 and peak RSS 18.666 GiB. It used registry
  `c4ac617743f2` (32,842 unique target rows), Ledger facts SHA-256
  `b3c0835631a446eb96aa84d86f3ee962d15ca356174c7114db52974f1cacc080`,
  and CD crosswalk SHA-256
  `c7cb040b1f57ca2ea2adcbfe60cc2b250ca23acbc4b640cd421e766fa54c1aec`.
  The fixed streaming plan used five registry chunks and twelve household
  slices per chunk, with a conservative live target-column payload bound of
  983,040,000 bytes (`tools/score_us_release_head_to_head.py:654-852,1399-1477`).
- Incumbent fiscal evidence: weighted loss
  `0.11462448275649702`; fraction within 10% `0.2669143170330674`; 57,240
  households and 57,240 nonzero shipped weights. The 32,842 per-target
  contributions sum to `0.11462448275649767`, agreeing with the canonical
  aggregate to floating-point summation tolerance. The production weighting
  and aggregate formulas are code-cited in the result
  (`tools/build_us_fiscal_refresh_release.py:344-348,481-516,5781-5814,6214-6290`;
  `packages/microcosm-calibrate/src/microcosm/calibrate/solve.py:471-537,576-600`).
- Incumbent battery evidence: 132/132 comparisons and all 369 nominal scalar
  legs are explicitly `inapplicable`; the artifact has 120,261 positive-weight
  clone-0 ASEC rows across the three battery entities and zero ACS rows. No
  zero, pass, or failure was synthesized for those by-origin-only legs
  (`packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:13967-13985,14006-14391`).
- Result identities: JSON SHA-256
  `b2ad1a07f9668bc5d796cc9de99ef12da781b1ee8163ea65781871a20da441c8`;
  Markdown SHA-256
  `3f9171b8f63fcef61518a4af1c18a8555c4f449ac62e9283e41ac2fe9c779021`.
  Both record the incumbent artifact SHA-256 `48b9d479...`, exact HF repo,
  revision, resolved commit, filename, PolicyEngine.py bundle version/source
  commit, and certified policyengine-us version.

## Owner command when the 25% candidate exists

First confirm that the host builder has finished. On the owner host (where
process inspection is permitted), `ps ax | grep '[b]uild_us_multispine_pool'`
must print nothing before either scoring command. Set the two published
candidate paths, then run the dense pool and sparse-57k views separately on
the same frozen incumbent and Ledger yardstick:

```bash
CANDIDATE_POOL_MANIFEST=/absolute/path/to/25pct/pool.manifest.json
CANDIDATE_SPARSE_H5=/absolute/path/to/25pct/sparse-57k.h5
CANDIDATE_MANIFEST_SHA256="$(shasum -a 256 "$CANDIDATE_POOL_MANIFEST" | awk '{print $1}')"
CANDIDATE_POOL_SHA8="$(jq -r '.pool_h5.sha256[0:8]' "$CANDIDATE_POOL_MANIFEST")"
CANDIDATE_SPARSE_SHA8="$(shasum -a 256 "$CANDIDATE_SPARSE_H5" | awk '{print substr($1,1,8)}')"

.venv/bin/python tools/score_us_release_head_to_head.py \
  --incumbent /Users/maxghenis/.cache/huggingface/hub/datasets--policyengine--populace-us/snapshots/26dcad66867687f15735dc4926523e3741920836/populace_us_2024.h5 \
  --candidate "$CANDIDATE_POOL_MANIFEST" \
  --candidate-manifest-sha256 "$CANDIDATE_MANIFEST_SHA256" \
  --ledger-facts /Users/maxghenis/PolicyEngine/_buildh-runtime/inputs/consumer_facts_buildn_v9_4.jsonl \
  --out-prefix "experiments/replacement_scorecard/head_to_head_dense_48b9d479_${CANDIDATE_POOL_SHA8}" \
  --maximum-microsim-batch-size 5000

.venv/bin/python tools/score_us_release_head_to_head.py \
  --incumbent /Users/maxghenis/.cache/huggingface/hub/datasets--policyengine--populace-us/snapshots/26dcad66867687f15735dc4926523e3741920836/populace_us_2024.h5 \
  --candidate "$CANDIDATE_SPARSE_H5" \
  --ledger-facts /Users/maxghenis/PolicyEngine/_buildh-runtime/inputs/consumer_facts_buildn_v9_4.jsonl \
  --out-prefix "experiments/replacement_scorecard/head_to_head_sparse_48b9d479_${CANDIDATE_SPARSE_SHA8}" \
  --maximum-microsim-batch-size 5000
```

“Better than the incumbent” on this yardstick means the owner compares each
candidate view with the incumbent’s exact `0.11462448275649702` weighted fiscal
loss and the reported target-by-target balance of lower, equal, and higher
absolute relative errors, while also inspecting every battery leg that is
computable for that candidate. There is no scorecard threshold and no automatic
conjunction or verdict. Because the incumbent’s ASEC-vs-ACS battery is
definitionally inapplicable, the candidate’s battery is standalone evidence—it
is not compared with a fabricated incumbent zero, pass, or failure. The owner
decides whether the dense and sparse evidence justifies the flip.

## Final validation and completion

- The prescribed full workspace suite was run through the current worktree
  interpreter because the copied `.venv/bin/pytest` console script retained a
  stale sibling-worktree shebang. The first correctly routed run reached 7,027
  passes and 76 skips; its only failure was the source-hygiene guard finding a
  retired-package literal in this journal. Commit `6847d245` removed that
  documentation-only literal, and the guard passed independently.
- Final full-suite receipt:

  ```text
  UV_CACHE_DIR=/tmp/microcosm-scorecard-uv-cache uv run python -m pytest
  7028 passed, 76 skipped, 1922 warnings in 5996.15s (1:39:56)
  ```

- Final static receipts: repository-wide `ruff check .` passed; all six Python
  files changed on this branch passed `ruff format --check`; scorer
  `py_compile` and `git diff --check` passed. A whole-tree format audit listed
  69 pre-existing mainline files, which this lane deliberately did not rewrite.
- Independent final audit found no deliverable-level gap. The implementation,
  incumbent JSON/Markdown, exact candidate commands, comparison doctrine,
  `PROGRESS.md`, and `FINAL_REPORT.md` are complete. The only next action is
  external: the host builds the candidate, then the owner scores its dense and
  sparse views and decides the flip.
- No pool build, publication, push, gate edit, threshold edit, tolerance edit,
  or band edit occurred in this lane.

---

# Historical: one-target-surface lane notes

## 2026-08-21 — baseline and doctrine

- Branch: `one-target-surface`, starting at `2c7a7218` (`origin/main`).
- User doctrine: all US calibrated artifacts compile one target surface;
  geography is a constraint dimension, while artifact scale changes only L0 /
  record count.
- The current split is explicit in
  `packages/microcosm-build/src/microcosm/build/us_runtime/fiscal_targets.py`:
  `compile_us_fiscal_target_registry` accepts
  `include_congressional_district_targets`, forwards it through dynamic target
  dispatch, and uses it to omit SOI and ACS congressional-district facts.
- The target profile independently gates CD-to-state hierarchy reconciliation
  on the same flag in
  `packages/microcosm-build/src/microcosm/build/us/fiscal_target_references.json`.
- The SOI taxable-interest rebase doctrine remains distinct: CD aggregate rows
  are processing-window subsets and never act as national controls, as recorded
  in `_rebase_stale_soi_taxable_interest_distributions` and pinned by
  `test_stale_soi_taxable_interest_never_uses_congressional_district_controls`.
- [microcosm#449](https://github.com/PolicyEngine/microcosm/issues/449#issuecomment-5002607353)
  explicitly names deletion of the flag as the one-surface outcome;
  [microcosm#569](https://github.com/PolicyEngine/microcosm/issues/569)
  records that the scorer's opt-in path is dead.
- Environment: default-cache sync failed because the sandbox cannot write
  `~/.cache/uv`; clean-cache sync then failed because network/DNS is disabled.
  The sibling `microcosm-spec-engine` checkout has the identical `uv.lock`
  (`895535...`) and a complete Python 3.14 environment, so its `.venv` was
  cloned copy-on-write. An offline editable reinstall still requires missing
  build-isolation metadata; test commands therefore use `UV_NO_SYNC=1` and put
  every current-worktree package `src` directory first on `PYTHONPATH`.
- GitNexus: repository analysis produced `.gitnexus/lbug`, then registration
  failed on the sandboxed `~/.gitnexus/registry.json`; dependency coverage is
  being checked with repository-wide `rg` plus focused tests.
- Build discipline: no calibration build has run; the off-chain / <=1% rule is
  intact and `logbook-pending-chain.txt` has not been touched.
- Validation baseline: `uv run pytest -q` advanced without a failure for
  1,136.97 seconds, then was interrupted while an unrelated PUF-QRF test was
  waiting on a subprocess
  (`test_puf_qrf_chain.py::test_primary_qrf_rejects_every_stale_schema_version`).
  Per-commit checks use the affected US target/compiler tests; the complete
  workspace suite will run against the final tree. `uv run ruff check .`
  passed.
- Affected-suite baseline: the target/compiler/parity/builder/scorer/spec set
  reached 100% with no failure in 349.59 seconds. `/usr/bin/time -l` itself
  exits 1 in this sandbox because `sysctl kern.clockrate` is forbidden (the
  same wrapper also exits 1 around `true`); later memory receipts use Python's
  child-resource accounting instead.

## 2026-08-21 — runtime surface unified

- The public compiler has no CD membership option and routes IRS SOI, ACS-CD,
  and PEP-CD facts unconditionally
  (`packages/microcosm-build/src/microcosm/build/us_runtime/fiscal_targets.py:933-1010,2299-2390,2588-2625`).
- The row-level doctrine is unchanged: CD rows compile into the registry, but
  `_soi_taxable_interest_control_key_from_fact` rejects CD record sets as
  national controls
  (`packages/microcosm-build/src/microcosm/build/us_runtime/fiscal_targets.py:1478-1510,1726-1744`).
- Builder, fiscal scorer, state scorer, and ACS-local compilation all load the
  packaged source-to-current CD crosswalk by default; the production source
  aliases are active unconditionally
  (`tools/build_us_fiscal_refresh_release.py:1486-1497,8319-8350,11226-11230`;
  `tools/score_us_fiscal_targets.py:383-426`;
  `tools/score_us_state_files.py:313-345`;
  `tools/build_us_acs_local_release.py:141-170`).
- The diagnostic JCT deletion switch and its target-profile-gate bypass were
  deleted from all release/scorer entrypoints. The active registry is now the
  compiled registry in the builder and both scorers
  (`tools/build_us_fiscal_refresh_release.py:8435-8464`;
  `tools/score_us_fiscal_targets.py:415-432`;
  `tools/score_us_state_files.py:335-352`).
- The generated calibration contract declares `national`, `state`, and
  `congressional_district` in one `geography_layers` list and requires CD
  inclusion; there is no default-layer fork
  (`tools/us_bundle_generation/contracts.py:1322-1340`;
  `packages/microcosm-build/src/microcosm/build/spec_engine/schema/calibration.schema.json:35-68`).
- The parity generator compiles with the canonical crosswalk unconditionally,
  and the regenerated manifest records 32 compiled / 52 reviewed families
  (`tools/build_us_target_parity_manifest.py:617-655,689-729`;
  `packages/microcosm-build/src/microcosm/build/us/target_parity_manifest.json:3-12,524-527`).
- This deletes the dead scorer opt-in described by
  [microcosm#569](https://github.com/PolicyEngine/microcosm/issues/569) and the
  regime knob superseded by the one-surface decision in
  [microcosm#449](https://github.com/PolicyEngine/microcosm/issues/449#issuecomment-5002607353).
- Validation: the affected 10-file suite completed to 100% with exit 0 after
  the change; Ruff, Python byte compilation, and `git diff --check` pass. No
  build, push, chain operation, or pending-chain edit occurred.

## 2026-08-21 — parity doctrine and row-level invariant

- `irs_soi.congressional_district_2022` is a red-line compiled family, so the
  anti-rot validator refuses any future downgrade to a reviewed exclusion
  (`packages/microcosm-build/src/microcosm/build/us_runtime/release_target_parity.py:88-123,579-586`).
- The shipped manifest entry is `compiled`, has no exclusion classification,
  reason, evidence, or fence, and states that there is no local-versus-national
  surface. Its header counts are pinned to the parsed 32 compiled and 52
  reviewed families
  (`packages/microcosm-build/src/microcosm/build/us/target_parity_manifest.json:3-12,524-527`;
  `packages/microcosm-build/tests/test_release_target_parity.py:290-317`).
- The always-compiled SOI test exercises CD, state, and national rows
  (`packages/microcosm-build/tests/test_us_fiscal_targets.py:177-356`). The
  taxable-interest doctrine test additionally asserts that the CD aggregate is
  present in the registry, then proves it never supplies the rebase control
  metadata when a true Pub 1304 national control exists
  (`packages/microcosm-build/tests/test_us_fiscal_targets.py:2539-2625`).
- Validation: the standard 10-file affected suite completed to 100% with exit
  0; Ruff and `git diff --check` pass.

## 2026-08-21 — artifact-scale leak removed

- The hidden split was the compiler's per-run support-exclusion mapping: a
  caller could delete otherwise compiled source rows for one artifact. That
  parameter and its dynamic-dispatch branch are gone; the compiler now has
  exactly five inputs, none related to artifact size, sparsity, record count,
  support, inclusion, or diagnostic target deletion
  (`packages/microcosm-build/src/microcosm/build/us_runtime/fiscal_targets.py:933-1005,2069-2090,2287-2300`).
- The release tool no longer parses or loads an artifact-specific exclusion
  file, passes no membership override into compilation, and reports only the
  standing surface-wide source exclusions
  (`tools/build_us_fiscal_refresh_release.py:897-923,7770-7815,8363-8376,11192-11198`).
  The obsolete
  `experiments/build_j_recert/sparse_zero_support_exclusions_buildj.json` was
  deleted and its shell/caller plumbing removed.
- The shared `US_FISCAL_TARGET_SUPPORT_EXCLUSIONS` registry remains: it is a
  single source-row doctrine applied identically to all artifacts, not a scale
  input (`packages/microcosm-build/src/microcosm/build/us_runtime/fiscal_targets.py:554-660,2287-2295`).
- The identity regression compiles one CD-bearing fact set twice under nominal
  57,240-record sparse and 337,704-record dense labels; both the full `specs`
  tuple and content-addressed registry `version` must match, and the exact
  compiler signature is pinned
  (`packages/microcosm-build/tests/test_us_fiscal_targets.py:644-696`).
- Release/fiscal-scorer/state-scorer signature sets are pinned, and the release
  parser rejects all three deleted membership options
  (`packages/microcosm-build/tests/test_us_fiscal_refresh_builder.py:52-97`;
  `packages/microcosm-build/tests/test_us_state_files_scorer.py:26-40`).
- Validation: six new identity/signature/parser tests pass; the standard
  10-file affected suite completed to 100% with exit 0; Ruff and
  `git diff --check` pass. No build ran.
</details>

## 2026-08-23 semantic union and ownership hardening

- Began `git merge --no-commit origin/main` when the remote-tracking snapshot
  was `d69131a3534a5311d2b0c8436ba9dd566e67a914`. The ref later advanced, but
  the active merge remains bound to the snapshot resolved at its start; it was
  not reset or silently retargeted.
- The union keeps main's selective nine-target post-transfer calibration
  evidence and this lane's complete all-active-target origin/regime evidence.
  The two source branches had independently used stacked authority v11, so the
  semantic union is v12
  (`packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:1703-1707`;
  `packages/microcosm-build/src/microcosm/build/us/spec/battery.yaml:824-826`).
  The current identities are late-producer receipt schema 4, target-bank
  schema 2, outer stacked materializer 11, stacked pool-stage materializer 8,
  pool manifest schema 8, and retiring legacy materializer/manifest 4.
- The merged US spec SHA-256 is
  `debc4f418208d57fe704feb444e2907e488655154e78ff6e5f96c1a379028a27`;
  stacked authority SHA-256 is
  `da45195e95addd1db37749a247eae8b29daa076b8c87997a41ba13c92035f589`;
  and the full checkpoint identity is
  `e524d778b4ae5adeb14a8a19ee53a573f7019ec1852ceae4219aec4f3d741122`.
- Receipt-only validation now enforces canonical predictor vocabulary and
  placement, selected-evidence/origin agreement, exact aggregate/group/leaf
  ownership schemas, canonical donor route, producer roles/count equations,
  and sibling catalog/model-target/regime agreement. It explicitly cannot
  replay donor values or seeds
  (`packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:4344-4442,4647-4687,5072-5754`).
- Production finalization recomputes exact producer-mask counts against the
  live frame before accepting the aggregate receipt
  (`packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:5603-5617,11635-11759`).
  Ready stacked-H5 loading now authenticates the canonical early gap-fill
  receipt as well as the late DAG
  (`packages/microcosm-build/src/microcosm/build/us_runtime/h5_io.py:594-640`).
- Main's post-transfer policy remains disjoint from all four QBI amounts; the
  explicit regression is at
  `packages/microcosm-build/tests/test_us_post_transfer_calibration.py:71-87`.
- The four-target ownership regression compares group, aggregate, and signed
  execution copies and rejects a missing `origin.channel` from any copy even
  after every enclosing receipt is rehashed
  (`packages/microcosm-build/tests/test_us_stacked_spine.py:6581-6720`).
  Stronger rehashed tests reject donor-route, producer-role/count, predictor,
  and single-target catalog substitutions at
  `packages/microcosm-build/tests/test_us_stacked_spine.py:6723-6849`.
- Focused merged receipt/ownership validation passed 20 tests with 278
  deselected; two live-count cases corrected after independent review then
  passed exactly. The canonical evidence extractor was rerun with full SHA
  verification and emitted byte-identical evidence at
  `38e60c1ec5e39b86df957148c877b3062ca97028f33ea0d1411013c2911c4b55`
  with zero adjudication mismatches.
- The existing host 1% baseline gates file has SHA-256
  `1d6059868680f872fe04d452a536bcc3c215bafabb4c50d7740a469fe6a8b56a`
  and all eight QBI amount checks remain red. The lane started no pool build,
  so there is no after-artifact and no claimed margin improvement. The managed
  sandbox denied `ps`; this does not change the headless rule that the host
  queue owns every pool build.
- Final merge compatibility audit kept the outer stacked checkpoint
  materializer at v11 while the embedded authority advances to v12. Both
  merge parents and the generated spec bind outer v11; the authority receipt
  itself invalidates pre-union checkpoint identities. Two accidentally merged
  materializer-v12 test expectations were corrected, and the focused identity
  matrix passed 11 tests
  (`tools/build_us_multispine_pool.py:316-335`;
  `packages/microcosm-build/src/microcosm/build/us/spec/spine.yaml:1-7`;
  `packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:1703-1707`;
  `packages/microcosm-build/tests/test_us_multispine_pool_tool.py:3393-3645`).
- Exact ownership-schema validation initially shadowed main's more specific
  row-count, calibration-summary, and target-surface failures. The shared
  count validator now retains field/activation/residual diagnostics, and early
  gap-fill validation calls it before the exact leaf schema. The focused eight
  compatibility cases passed; the complete stacked-spine file subsequently
  passed within the exhaustive final partition
  (`packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:4262-4298,4773-4857`).
- Final package gate on the resolved tree: `microcosm-frame` 294 passed/36
  skipped; `microcosm-fit` 93 passed; `microcosm-calibrate` 203 passed;
  `microcosm-data` 275 passed/1 skipped; `microcosm-build` 6,323 passed/39
  skipped across 6,362 unique tests. Deterministic partitions first covered
  the package. A subsequent exact package-wide process ran for 1h22m25s,
  passing 6,321 and skipping 39 before exactly two
  `test_us_trade_entries_cli.py` subprocesses hit their unchanged 300-second
  limits under accumulated load. The unchanged complete file then passed
  13/13 in 7m58s; `test_us_trade_imdb_bulk.py` likewise exited 0 in its fresh
  full-file run.
- Repository-wide `uv run ruff check .`, targeted Ruff formatting for the two
  final compatibility files, cached and unstaged `git diff --check`, and
  `tools/spec_engine_coverage.py --check` pass. Coverage is 41,471/41,471
  configuration fields and 40/40 inventory checks. No pool build, publication,
  push, exclusion, tuning, logbook-chain operation, or pending-chain edit ran.
