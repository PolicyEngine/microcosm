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
  into the nine PolicyEngine leaves
  (`packages/microcosm-build/src/microcosm/build/us_runtime/cps_carried.py:57-69`,
  `packages/microcosm-build/src/microcosm/build/us_runtime/cps_carried.py:431-438`,
  `packages/microcosm-build/src/microcosm/build/us_runtime/cps_carried.py:555-560`).
- The ACS loader currently selects no health fields because its required
  person tuple stops at `PWGTP` and its optional tuple is empty
  (`packages/microcosm-build/src/microcosm/build/us_runtime/acs_pums.py:72`,
  `packages/microcosm-build/src/microcosm/build/us_runtime/acs_pums.py:88`,
  `packages/microcosm-build/src/microcosm/build/us_runtime/acs_pums.py:215`).
- Missing ACS leaves enter the declared boolean QRF family
  (`packages/microcosm-build/src/microcosm/build/us_runtime/acs_transfer.py:356-365`).
  At round-2 entry, commit `7a66c5a9` excluded complete native recipient
  columns from active fits (`7a66c5a9:packages/microcosm-build/src/microcosm/build/us_runtime/acs_transfer.py:2383-2420`).

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
  (`packages/microcosm-build/src/microcosm/build/us_runtime/cps_carried.py:57-69`).
- In round-1 commit `7a66c5a9`, complete native targets were omitted from
  active QRF families while missing peers retained their requested target-bank
  positions (`7a66c5a9:packages/microcosm-build/src/microcosm/build/us_runtime/acs_transfer.py:967-1029`,
  `7a66c5a9:packages/microcosm-build/src/microcosm/build/us_runtime/acs_transfer.py:2383-2420`).

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

## 2026-08-25 — cold 1% round-2 diagnosis

### Host artifact receipt

- Read only `baseline1pct/pool.gates.json`, `health1pct/pool.gates.json`, both
  `simulated.checkpoint.h5` frames, and their health target-bank checkpoints.
  No build, calibration, comparator, band, threshold, seed, or exclusion ran.
- The frames align on every `person_id`, support channel, clone index, and
  person weight. Clone-0 positive-weight denominators are 80,851,529.27715749
  over 4,311 ASEC rows and 79,926,522.10879111 over 34,293 ACS rows.

### Per-leg diagnosis

- IHS, initial frame trace: ASEC has 8 positives / weight 197,606.039624;
  baseline ACS QRF had 12 / 46,033.921631. Native `HINS7 == 1` has exactly 252
  / 482,171.216580, producing 2.468303. No extra ACS category enters the
  mapper. The deeper per-vintage source audit below supersedes the preliminary
  conclusion that this was only a comparison mismatch.
- Marketplace: ASEC remains 157 / 2,908,019.635846. ACS moved from 872 /
  2,360,680.784026 to 795 / 2,125,795.272305; 1,579 same-person rows flipped.
- Non-Marketplace direct purchase: ASEC remains 107 / 2,028,850.769034. ACS
  moved from 491 / 1,023,422.890043 to 528 / 1,054,016.416743; 857 rows flipped.
- TRICARE: ASEC remains 45 / 807,558.923656. ACS moved from 206 /
  502,159.944901 to 153 / 385,243.838170; 277 rows flipped.
- Target metadata proves the causal seam: the first boolean batch changed from
  eight targets `[CHAMPVA, ESI, IHS, Marketplace, Medicaid, direct purchase,
  other means-tested, TRICARE]` to six by deleting the native ESI/IHS chain
  steps. Stable bank indexes did not preserve the QRF completed-target prefix.

### Disposition

- Keep the exact HINS7 and ASEC `NOW_* == 1` mappings. Do not restore the IHS
  QRF underprediction or map `HINS2`/`HINS5` onto narrower leaves.
- Repair family activation so every declared target contributes its latent
  model-draw chain step whenever a peer is missing; target null masks remain
  the authority that prevents overwriting complete native cells.
- `experiments/health_labels/ROUND2_DIAGNOSIS.md` carries the full mass table,
  code trace, and IHS owner-only absence equation. Marketplace, direct purchase,
  and TRICARE retain the previously documented owner-only absence equation.

### Next

- Add the chain-completeness regression, implement the family-selection fix,
  and run focused cold/warm target-bank tests.
- Run the complete permitted Ruff/package-shard gate below 15 GiB RSS.
- End this journal with evidence-grounded next-host-run ratio expectations.

## 2026-08-25 — round-2 generating repairs

### Deeper IHS source diagnosis

- The pooled checkpoint hid a partial input schema. All 1,368 sampled 2022
  rows (weight 25,266,038.707667) and 1,478 sampled 2023 rows
  (26,972,301.358297) have `NOW_IHSFLG` null; the only 8 reporters, weight
  197,606.039624, came from the complete 2024 vintage.
- The generic ASEC source reader filled those nulls with zero before the
  `code == 1` operator. This was a generating defect even though the ACS
  `HINS7` mapping itself was exact.
- Read-only re-extraction receipt
  `/Users/maxghenis/PolicyEngine/_buildo-runtime/inputs/asec-720/receipt_720.json`
  pins corrected 2022/2023 inputs. Joining the same sampled people yields 7
  2022 reporters / weight 72,245.403261, 7 2023 reporters / 23,196.524112,
  and the unchanged 8 2024 reporters / 197,606.039624: pooled ASEC mass
  293,047.966998, incidence 0.003624519779.
- The exact next-host input tuple is
  `2022=/Users/maxghenis/PolicyEngine/_buildo-runtime/inputs/asec-720/census_cps_2022.h5`
  (SHA-256 `797fb7e0ea3e94a52dc6c9c7f5a01d70d28a6c5620f3c112bcbd7c6cf6cffe9f`),
  `2023=/Users/maxghenis/PolicyEngine/_buildo-runtime/inputs/asec-720/census_cps_2023.h5`
  (`5fc2feca08aa125ac2ed127711457038e0d493a5ad1d60aff763fb8e42a9d60a`),
  and the unchanged `2024=<receipt_720.json /2024/source_h5>`
  (`ec36604cb735a660b51b0b2f90be27d803b5878f3464fb30d0eacead59c1260d`).
  The full three-path handoff is recorded in the experiment diagnosis, whose
  run-scaffolding scope may name local source provenance.
- The mapper now has one exact nine-label source contract
  (`packages/microcosm-build/src/microcosm/build/us_runtime/cps_carried.py:57-69`).
  Each source is validated before pooling and must carry every column with only
  nonblank `{1, 2}` codes
  (`packages/microcosm-build/src/microcosm/build/us_runtime/asec_pool.py:240-247`,
  `packages/microcosm-build/src/microcosm/build/us_runtime/cps_carried.py:441-487`).
  Old partial inputs now fail closed; no reporter or ratio is synthesized.
- Corrected-sample IHS is still red at 1.664408372482. Complete receipt-backed
  inputs give a raw ACS/ASEC incidence ratio 1.942759574413, so the documented
  common-universe and owner-only declared-absence equations remain necessary.

### Ordered-chain repair

- `_missing_target_families` now activates the complete bounded family when
  any member is incomplete. Latent ESI/IHS model draws therefore reproduce the
  declared QRF prefix for Marketplace and later labels
  (`packages/microcosm-build/src/microcosm/build/us_runtime/acs_transfer.py:2383-2415`).
- Target-level null masks remain the write authority, so native outputs are
  never overwritten or reported as imputed
  (`packages/microcosm-build/src/microcosm/build/us_runtime/acs_transfer.py:992-1043`).
- The transfer execution contract is advanced to schema 3 for full-prefix
  latent-draw semantics; target-bank materializer version 3 prevents shortened
  version-2 chains from resuming.
- Regressions pin all nine exact ASEC mappings (including overlapping flags),
  partial/absent/invalid-vintage rejection, latent-chain completeness without
  native overwrite, peer-draw invariance, durable target indexes, and
  cold/warm/monolithic equality. The focused source regressions (6) and ACS
  transfer tests (4) pass;
  touched-file Ruff and deterministic generated-spec checks pass.
- Full suite receipt: build 5,979 passed/37 skipped, with the 60 logbook and
  coverage-tool tests isolated to avoid subprocess contamination; calibrate
  201 passed; data 275 passed/1 skipped; fit 93 passed; frame 294 passed/36
  skipped. Ruff and `git diff --check` pass. The generated US bundle is current
  at `0f9ddf0bb16469459a5c87ec1667d74b9a1425d5b3d1753f80e26dba8f000e10`;
  spec-engine coverage is 41,380/41,380 fields and 40/40 inventory checks.

### Expected next host-owned cold 1% ratios

No build was run in this lane. Correcting the 2022/2023 ASEC IHS source changes
the IHS donor outcome before Marketplace, so the existing checkpoints do not
contain the restored-prefix plus corrected-IHS counterfactual. Direct purchase
and TRICARE additionally follow the narrow `NOW_CAID` Medicaid target.
Assigning any of those three a number would be calibration-by-guess. Their
owner-only non-comparability equation remains in
`experiments/health_labels/PROVENANCE.md`.

| Leg | Expected next 1% ACS/ASEC ratio | Expected disposition |
|---|---:|---|
| IHS | `1.664408372482` with the three-path input tuple above | red; owner common-universe/absence adjudication |
| Marketplace | host measurement required (corrected-IHS hybrid unavailable) | concept mismatch; no band tuning |
| non-Marketplace direct purchase | host measurement required (hybrid checkpoint unavailable) | concept mismatch; no band tuning |
| TRICARE | host measurement required (hybrid checkpoint unavailable) | concept mismatch; no band tuning |
