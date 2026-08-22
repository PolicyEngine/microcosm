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
- Incumbent identity: PolicyEngine.py 4.15.0's bundled manifest names data
  package 1.115.5, HF model repo `policyengine/policyengine-us-data`, immutable
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
  (`packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:11366-11378,11523-11533,11577-11580,11679-11739`).
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
  (`packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:11577-11580,11695-11703`;
  channel constants `support_provenance.py:31` and `stacked_spine.py:263`),
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
