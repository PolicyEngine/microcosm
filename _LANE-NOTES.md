# Health-coverage label lane notes

## 2026-08-21 — lane initialization

### Scope and constraints

- Branch: `battery-health-coverage-labels`, based on `origin/main`; no push or
  publication is authorized.
- This lane owns health-label provenance and generating mechanisms. It will not
  tune gates, bands, ceilings, folds, seeds, or add comparator exclusions.
- Any data build must be off-chain, at no more than a 1% sample, below 15 GiB
  RSS, and preceded by a host-process check for `build_us_multispine_pool`.
  The managed sandbox currently rejects `ps` and `pgrep`, so no build has been
  started.

### Environment receipt

1. `uv sync --all-packages --extra us` failed because the sandbox denied uv's
   global cache path.
2. Retrying with `UV_CACHE_DIR=.uv-cache` built the local shards but could not
   resolve the locked `huggingface-hub==1.18.0`; DNS/network access is absent.
3. The `microcosm-f1` worktree has the same `uv.lock` digest and a complete
   environment. Tests use that environment with `uv run --no-sync` and this
   worktree's five `src` roots on `PYTHONPATH`.

### Initial adjudication receipt

- The arm-split source of truth marks three health-label comparisons as
  blockers: CHAMPVA (baseline ACS/ASEC incidence ratio 3.7202586764), Indian
  Health Service (2.3028334070), and other means-tested coverage
  (3.8169365893). Every four-way arm candidate remains outside the comparator's
  inclusive `[0.8, 1.25]` band; eligibility improves CHAMPVA to 3.0739130289
  but does not resolve it
  (`../microcosm-arm-split/experiments/battery_burndown/ADJUDICATION.md:125`,
  `../microcosm-arm-split/experiments/battery_burndown/ADJUDICATION.md:126`,
  `../microcosm-arm-split/experiments/battery_burndown/ADJUDICATION.md:127`).
- The synthesis says no isolated arm is landable and hands this work to the
  health-label/model lane
  (`../microcosm-arm-split/experiments/phase_p_arm_split/SYNTHESIS.md:5`,
  `../microcosm-arm-split/experiments/phase_p_arm_split/SYNTHESIS.md:27`).

### Initial code trace

- ASEC source flags are converted with the strict `code == 1` rule and mapped
  into the nine PolicyEngine leaves in
  `packages/microcosm-build/src/microcosm/build/us_runtime/cps_carried.py:413`
  and
  `packages/microcosm-build/src/microcosm/build/us_runtime/cps_carried.py:503`.
- The ACS loader currently selects no health fields because its required
  person tuple stops at `PWGTP` and its optional tuple is empty
  (`packages/microcosm-build/src/microcosm/build/us_runtime/acs_pums.py:72`,
  `packages/microcosm-build/src/microcosm/build/us_runtime/acs_pums.py:88`,
  `packages/microcosm-build/src/microcosm/build/us_runtime/acs_pums.py:215`).
- Missing ACS leaves enter the declared boolean QRF family
  (`packages/microcosm-build/src/microcosm/build/us_runtime/acs_transfer.py:356`),
  while complete native recipient columns are excluded from active fits
  (`packages/microcosm-build/src/microcosm/build/us_runtime/acs_transfer.py:2383`).

### Next

- Finish the exact source/concept matrix and classify each label.
- Repair only derivation defects, with label-specific regressions.
- Obtain the required host-queue evidence before any 1% before/after run.

### Pre-change quality receipt

- `uv run --no-sync ruff check .`: pass.
- `packages/microcosm-build/tests`: pass; 5,996 tests collected and the full
  shard exited 0.
- `packages/microcosm-calibrate/tests`: 201 passed (2 warnings).
- `packages/microcosm-data/tests`: 275 passed, 1 skipped.
- `packages/microcosm-fit/tests`: 93 passed (1 warning).
- `packages/microcosm-frame/tests`: 294 passed, 36 skipped (1 warning).
- The heavy shard peaked at approximately 10.6 GiB RSS, below the binding
  15 GiB ceiling.
