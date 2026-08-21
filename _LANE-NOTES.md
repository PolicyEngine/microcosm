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

## 2026-08-21 — provenance classification and mechanism repair

### Classification

- Exact ACS source counterparts: ESI (`HINS1`) and IHS (`HINS7`). Their prior
  QRF generation was a derivation defect, not a survey-concept mismatch.
- ASEC Medicaid was also a derivation defect: the narrow PolicyEngine leaf used
  composite `NOW_MCAID` rather than `NOW_CAID`.
- Seven cross-survey detailed labels remain concept mismatches: Marketplace,
  non-Marketplace direct purchase, Medicaid, other means-tested, TRICARE,
  CHAMPVA, and VA. The complete source matrix and owner-only declared-absence,
  aggregate, and common-universe equations are in
  `experiments/health_labels/PROVENANCE.md`.
- No register, absence rule, gate, band, ceiling, fold, or seed changed.

### Generating-mechanism changes

- The ACS person reader requires only `HINS1` and `HINS7` as new fields and
  fails closed if either is absent
  (`packages/microcosm-build/src/microcosm/build/us_runtime/acs_pums.py:72-90`,
  `packages/microcosm-build/src/microcosm/build/us_runtime/acs_pums.py:351-359`).
- Native mapping validates nonblank `{1, 2}` codes and emits `HINS1 == 1` as
  `has_esi` and `HINS7 == 1` as the IHS leaf
  (`packages/microcosm-build/src/microcosm/build/us_runtime/acs_inputs.py:115-128`,
  `packages/microcosm-build/src/microcosm/build/us_runtime/acs_inputs.py:193-219`).
- Exact boundary contracts bind both output names to their source columns and
  transformations
  (`packages/microcosm-build/src/microcosm/build/us_runtime/operator_boundary.py:127-139`).
- The ASEC Medicaid mapping now selects `NOW_CAID`; the other eight ASEC health
  mappings are unchanged
  (`packages/microcosm-build/src/microcosm/build/us_runtime/cps_carried.py:413-433`).
- Complete native targets are omitted from active QRF families while missing
  peers retain their requested target-bank positions
  (`packages/microcosm-build/src/microcosm/build/us_runtime/acs_transfer.py:967-1029`,
  `packages/microcosm-build/src/microcosm/build/us_runtime/acs_transfer.py:2383-2420`).

### Focused regression receipt

- ACS PUMS loader: 19 passed.
- ACS native inputs: 17 passed.
- Operator-boundary selection: 3 passed, 273 deselected.
- ASEC Medicaid focused pair: 2 passed, 57 deselected.
- Native-health transfer preservation and bank-hole cold/warm resumption: 2
  passed (one joblib core-count warning).
- Touched-file Ruff checks and `git diff --check`: pass.

### GitNexus fallback

The requested debugging workflow could parse a local index but could not write
`/Users/maxghenis/.gitnexus/registry.json` under the managed sandbox, so graph
selection was unavailable. The partial untracked 103 MiB index was moved,
recoverably, to
`/private/tmp/microcosm-health-coverage-labels-gitnexus-failed-20260821` and the
impact trace used direct source call sites and focused regressions.

### Next

- Run the complete pre-commit Ruff/package-shard gate and commit this coherent
  mechanism/provenance step.
- Obtain the mandatory host-queue confirmation immediately before the off-chain
  1% verification build.

### Combined pre-commit quality receipt

- `uv run --no-sync ruff check .`: pass.
- `packages/microcosm-build/tests`: 5,973 passed, 39 skipped, 1,887 warnings in
  9,251.38 seconds; peak observed lane RSS approximately 11.6 GiB.
- `packages/microcosm-calibrate/tests`: 201 passed, 2 warnings.
- `packages/microcosm-data/tests`: 275 passed, 1 skipped.
- `packages/microcosm-fit/tests`: 93 passed, 1 warning.
- `packages/microcosm-frame/tests`: 294 passed, 36 skipped, 1 warning.
- The unified command exited 0; `git diff --check` passed.
