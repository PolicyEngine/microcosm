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
