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
  (`packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:3715-3872,3959-3992`).
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
  (`packages/microcosm-build/src/microcosm/build/us_runtime/acs_transfer.py:1512-1532,1627-1674`).
- The banked path performs the same recomputation before any target draw,
  compares every `fit_draw_next` result to it, and passes it through the bank
  load/write contract
  (`packages/microcosm-build/src/microcosm/build/us_runtime/acs_transfer.py:1855-1908,1909-1985`).
- Pattern provenance is copied into an immutable mapping; every fitted exported
  leaf names its exact model target, and canonical origin receipts distinguish
  `qrf_transfer`, `deterministic_derivation`, and `preexisting`
  (`packages/microcosm-build/src/microcosm/build/us_runtime/acs_transfer.py:425-539,582-671`).
- Target-bank schema 2 persists the regime beside each state transition,
  authenticates it on load, refuses valid-but-wrong regimes before filesystem
  persistence, and exposes the per-pattern map in its public receipt
  (`packages/microcosm-build/src/microcosm/build/us_runtime/acs_transfer_bank.py:25-34,135-249,358-425,612-780`).
- Early and late stacked target receipts now carry canonical origin evidence.
  The production late validator rejects absent/unknown origins, mismatched
  aggregate/group copies, incorrectly renamed model targets, and divergent
  regimes for exported siblings sharing a joint model target
  (`packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:3700-4084,8296-8474,8863-9050`).
- Stacked checkpoint materializer 8 and retiring legacy materializer 4 make
  transferred/simulated checkpoints that predate canonical regime evidence
  stale (`tools/build_us_multispine_pool.py:227-284`).
- Late-producer receipt schema 4 makes the new mandatory target origin/regime
  envelope explicit and rejects otherwise valid schema-3 receipts
  (`packages/microcosm-build/src/microcosm/build/us_runtime/us_late_producer_registry.py:102-129,2053-2070`;
  `packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:6884-6920`).
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
  (`packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:8728-8772,8832-8852,8945-9040`).
- Recomputed realized regimes from the exact frozen 108,073-row donor support
  for four availability patterns per target. The 52 target-pattern cells all
  match the fitter; every red amount is `zero_inflated_positive`. That regime
  uses a weighted zero/positive gate followed by the positive-value QRF
  (`packages/microcosm-fit/src/microcosm/fit/qrf.py:104-105,950-1003,1333-1429`).
  Future monolithic, bank, and stacked receipts now persist and validate the
  exact pattern-regime map
  (`packages/microcosm-build/src/microcosm/build/us_runtime/acs_transfer.py:509-555,582-671`;
  `packages/microcosm-build/src/microcosm/build/us_runtime/acs_transfer_bank.py:350-420,612-780`;
  `packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:8945-9040`).
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
  (`packages/microcosm-build/src/microcosm/build/us_runtime/acs_transfer.py:509-555,582-671`;
  `packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:8945-9040`).
- Canonical extraction ran twice with full SHA verification and produced the
  same `evidence.json` SHA-256
  `38e60c1ec5e39b86df957148c877b3062ca97028f33ea0d1411013c2911c4b55`.
  Validation reports: 8 ownership checks, 13 SHA-verified bank target files,
  52 regime cells, 9/9 terminal invariants at zero, and no adjudication
  mismatch. A canonical `--skip-sha` invocation exited 2 before write and left
  that digest unchanged.
- The refit plan does not infer a gate remedy from coarse marginals. Current
  closed receipts record regimes and pattern catalogs but not row assignments,
  gate scores/outcomes, or target-by-channel-by-pattern gate cross-tabs
  (`packages/microcosm-build/src/microcosm/build/us_runtime/acs_transfer.py:509-555,582-671`).
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
  `packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:2112-2141`.
- The role scope fails closed unless it is exactly those two QBI boolean PUF
  outputs, and metric registry materialization carries each target's resolved
  role
  (`packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:2998-3100`).
  Surface, plan, producer/transfer authority, and runtime comparator keys are
  role-aware
  (`packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:2144-2167,3293-3306,3460-3472,8854-8875,9038-9049,9113-9124,11991-12004`).
- The spec projection permits nonzero registered roles while rejecting one
  physical target declared across multiple roles
  (`packages/microcosm-build/src/microcosm/build/spec_engine/battery_semantics.py:43-71`).
  The live battery contract similarly indexes by physical target, preserves
  its registered role, and rejects ambiguity
  (`packages/microcosm-build/src/microcosm/build/us_runtime/stacked_battery_contract.py:66-118`).
- The canonical battery YAML assigns clone 1 to exactly the two targets and
  advances the old-base authority binding from 10 to 11
  (`packages/microcosm-build/src/microcosm/build/us/spec/battery.yaml:459-468,823-826`).
  A later semantic union with current `origin/main` must use fresh version 12,
  because main independently used version 11 for post-transfer calibration.
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
